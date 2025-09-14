"""
Simple Image Classification App using Pillow and Hugging Face Transformers
"""

import torch
from PIL import Image
import requests
from transformers import ViTImageProcessor, ViTForImageClassification
import numpy as np
from typing import List, Tuple, Optional
import io


class ImageClassifier:
    """
    A simple image classifier using Vision Transformer from Hugging Face
    """
    
    def __init__(self, model_name: str = "google/vit-base-patch16-224"):
        """
        Initialize the image classifier
        
        Args:
            model_name: Hugging Face model name for image classification
        """
        self.model_name = model_name
        self.device = torch.device("cpu")
        
        # Load the processor and model
        print(f"Loading model: {model_name}")
        self.processor = ViTImageProcessor.from_pretrained(model_name)
        self.model = ViTForImageClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded successfully on {self.device}")
    
    def load_image_from_path(self, image_path: str) -> Image.Image:
        """
        Load image from file path using Pillow
        
        Args:
            image_path: Path to the image file
            
        Returns:
            PIL Image object
        """
        try:
            image = Image.open(image_path)
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            return image
        except Exception as e:
            raise ValueError(f"Error loading image from {image_path}: {str(e)}")
    
    def load_image_from_url(self, image_url: str) -> Image.Image:
        """
        Load image from URL using Pillow
        
        Args:
            image_url: URL of the image
            
        Returns:
            PIL Image object
        """
        try:
            response = requests.get(image_url)
            response.raise_for_status()
            image = Image.open(io.BytesIO(response.content))
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            return image
        except Exception as e:
            raise ValueError(f"Error loading image from URL {image_url}: {str(e)}")
    
    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """
        Preprocess image for the model
        
        Args:
            image: PIL Image object
            
        Returns:
            Preprocessed tensor
        """
        inputs = self.processor(images=image, return_tensors="pt")
        return inputs.pixel_values.to(self.device)
    
    def predict(self, image: Image.Image, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Predict the class of an image
        
        Args:
            image: PIL Image object
            top_k: Number of top predictions to return
            
        Returns:
            List of tuples (class_name, confidence_score)
        """
        # Preprocess the image
        inputs = self.preprocess_image(image)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
        
        # Get top predictions
        top_indices = torch.topk(probabilities, top_k).indices[0]
        top_probs = torch.topk(probabilities, top_k).values[0]
        
        # Get class labels
        class_labels = self.model.config.id2label
        
        results = []
        for idx, prob in zip(top_indices, top_probs):
            class_name = class_labels[idx.item()]
            confidence = prob.item()
            results.append((class_name, confidence))
        
        return results
    
    def predict_from_path(self, image_path: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Predict the class of an image from file path
        
        Args:
            image_path: Path to the image file
            top_k: Number of top predictions to return
            
        Returns:
            List of tuples (class_name, confidence_score)
        """
        image = self.load_image_from_path(image_path)
        return self.predict(image, top_k)
    
    def predict_from_url(self, image_url: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Predict the class of an image from URL
        
        Args:
            image_url: URL of the image
            top_k: Number of top predictions to return
            
        Returns:
            List of tuples (class_name, confidence_score)
        """
        image = self.load_image_from_url(image_url)
        return self.predict(image, top_k)


def main():
    """
    Example usage of the ImageClassifier
    """
    # Initialize the classifier
    classifier = ImageClassifier()
    
    # Example with a sample image URL (you can replace with your own image)
    sample_url = r"C:\Users\QCWorkshop22\Downloads\Koala_Bear_With_Baby.jpg"
    
    try:
        print("Predicting image from URL...")
        predictions = classifier.predict_from_path(sample_url, top_k=3)
        
        print("\nTop 3 predictions:")
        for i, (class_name, confidence) in enumerate(predictions, 1):
            print(f"{i}. {class_name}: {confidence:.4f} ({confidence*100:.2f}%)")
            
    except Exception as e:
        print(f"Error: {e}")
        print("You can also test with local images using classifier.predict_from_path('path/to/image.jpg')")


if __name__ == "__main__":
    main()
