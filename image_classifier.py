"""
Simple Image Classification App using Pillow and Hugging Face Transformers
"""

import torch
from PIL import Image
import requests
from transformers import AutoImageProcessor, AutoModelForImageClassification
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
        
        # Prefer CPU for broad compatibility
        self.device = torch.device("cpu")
        
        # Try a sequence of compatible models to improve robustness
        candidate_models = [
            model_name,
            "facebook/deit-base-distilled-patch16-224",
            "microsoft/resnet-50",
        ]
        last_error: Optional[Exception] = None
        
        for candidate in candidate_models:
            try:
                print(f"Loading model: {candidate}")
                # Ensure weights are fully materialized on CPU to avoid meta-tensor issues
                self.processor = AutoImageProcessor.from_pretrained(candidate, local_files_only=False)
                self.model = AutoModelForImageClassification.from_pretrained(
                    candidate,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=False,
                )
                # Do NOT call .to() to avoid moving meta tensors; model is already on CPU
                self.model.eval()
                self.model_name = candidate
                print(f"Model loaded successfully on cpu: {candidate}")
                break
            except Exception as e:
                last_error = e
                print(f"Failed to load model '{candidate}': {e}")
                self.processor = None
                self.model = None
        
        if self.processor is None or self.model is None:
            raise RuntimeError(f"Could not initialize image classifier: {last_error}")
    
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
        # Keep tensors on CPU and avoid boolean evaluation of tensors
        if "pixel_values" in inputs:
            pixel_values = inputs["pixel_values"]
        elif "inputs" in inputs:
            pixel_values = inputs["inputs"]
        else:
            # Some processors may return a dict with different key naming
            # Fallback to the first tensor-like value
            for v in inputs.values():
                if isinstance(v, torch.Tensor):
                    pixel_values = v
                    break
            else:
                raise RuntimeError("Processor did not return pixel_values tensor")
        return pixel_values
    
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
            # Pass as keyword arg to match model signature
            outputs = self.model(pixel_values=inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
        
        # Get top predictions
        k = min(top_k, probabilities.shape[-1])
        topk = torch.topk(probabilities, k)
        top_indices = topk.indices[0]
        top_probs = topk.values[0]
        
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
