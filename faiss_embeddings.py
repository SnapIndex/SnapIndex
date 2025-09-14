import os
import numpy as np
import faiss
import pickle
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
from datetime import datetime

# Import from existing modules
from file_loader import find_pdfs, extract_pdf_text
from vector import embed, TextEmbedding

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FAISSEmbeddingManager:
    """
    A comprehensive FAISS embedding manager that can:
    - Process PDFs from directories
    - Create embeddings using FastEmbed
    - Build and manage FAISS indices
    - Perform semantic search
    - Save/load embedding databases
    """
    
    def __init__(self, 
                 model_name: str = "BAAI/bge-small-en-v1.5",
                 embedding_dim: int = 384,
                 index_type: str = "flat"):
        """
        Initialize the FAISS embedding manager
        
        Args:
            model_name: Name of the embedding model to use
            embedding_dim: Dimension of the embeddings
            index_type: Type of FAISS index ("flat", "ivf", "hnsw")
        """
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.embedding_model = None
        self.index = None
        self.metadata = []
        self.db_path = None
        
        # Initialize embedding model
        self._init_embedding_model()
    
    def _init_embedding_model(self):
        """Initialize the embedding model"""
        try:
            self.embedding_model = TextEmbedding(model_name=self.model_name)
            logger.info(f"Initialized embedding model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            raise
    
    def process_directory(self, 
                         directory_path: str, 
                         chunk_size: int = 1000,
                         chunk_overlap: int = 200) -> List[Dict[str, Any]]:
        """
        Process all PDFs in a directory and extract text chunks
        
        Args:
            directory_path: Path to directory containing PDFs
            chunk_size: Size of text chunks for embedding
            chunk_overlap: Overlap between chunks
            
        Returns:
            List of text chunks with metadata
        """
        logger.info(f"Processing directory: {directory_path}")
        
        if not os.path.exists(directory_path):
            raise ValueError(f"Directory does not exist: {directory_path}")
        
        pdf_files = find_pdfs(directory_path)
        if not pdf_files:
            logger.warning(f"No PDF files found in {directory_path}")
            return []
        
        all_chunks = []
        
        for pdf_path in pdf_files:
            logger.info(f"Processing PDF: {os.path.basename(pdf_path)}")
            
            try:
                # Extract text from PDF
                pdf_data = extract_pdf_text(pdf_path)
                
                # Create chunks from the text
                chunks = self._create_text_chunks(pdf_data, chunk_size, chunk_overlap)
                
                # Add metadata to chunks
                for chunk in chunks:
                    chunk.update({
                        "pdf_path": pdf_path,
                        "pdf_name": os.path.basename(pdf_path),
                        "processed_at": datetime.now().isoformat()
                    })
                
                all_chunks.extend(chunks)
                logger.info(f"Created {len(chunks)} chunks from {os.path.basename(pdf_path)}")
                
            except Exception as e:
                logger.error(f"Error processing {pdf_path}: {e}")
                continue
        
        logger.info(f"Total chunks created: {len(all_chunks)}")
        return all_chunks
    
    def _create_text_chunks(self, 
                           pdf_data: List[Dict], 
                           chunk_size: int, 
                           chunk_overlap: int) -> List[Dict[str, Any]]:
        """
        Create text chunks from PDF data
        
        Args:
            pdf_data: List of page data from PDF
            chunk_size: Maximum size of each chunk
            chunk_overlap: Overlap between chunks
            
        Returns:
            List of text chunks
        """
        chunks = []
        
        for page_data in pdf_data:
            text = page_data["text"]
            if not text.strip():
                continue
            
            # Split text into chunks
            page_chunks = self._split_text_into_chunks(
                text, chunk_size, chunk_overlap
            )
            
            for i, chunk_text in enumerate(page_chunks):
                chunk = {
                    "text": chunk_text,
                    "page_number": page_data["page_number"],
                    "chunk_index": i,
                    "chunk_size": len(chunk_text)
                }
                chunks.append(chunk)
        
        return chunks
    
    def _split_text_into_chunks(self, 
                               text: str, 
                               chunk_size: int, 
                               chunk_overlap: int) -> List[str]:
        """
        Split text into overlapping chunks
        
        Args:
            text: Text to split
            chunk_size: Maximum size of each chunk
            chunk_overlap: Overlap between chunks
            
        Returns:
            List of text chunks
        """
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings within the last 100 characters
                search_start = max(start + chunk_size - 100, start)
                for i in range(end - 1, search_start, -1):
                    if text[i] in '.!?':
                        end = i + 1
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - chunk_overlap
            if start >= len(text):
                break
        
        return chunks
    
    def create_embeddings(self, 
                         text_chunks: List[Dict[str, Any]], 
                         batch_size: int = 64) -> np.ndarray:
        """
        Create embeddings for text chunks
        
        Args:
            text_chunks: List of text chunks
            batch_size: Batch size for embedding generation
            
        Returns:
            Numpy array of embeddings
        """
        if not text_chunks:
            raise ValueError("No text chunks provided")
        
        logger.info(f"Creating embeddings for {len(text_chunks)} chunks")
        
        # Extract text for embedding
        texts = [chunk["text"] for chunk in text_chunks]
        
        # Generate embeddings
        embeddings = embed(texts)
        embeddings_array = np.array(embeddings, dtype='float32')
        
        logger.info(f"Created embeddings with shape: {embeddings_array.shape}")
        return embeddings_array
    
    def build_index(self, 
                   embeddings: np.ndarray, 
                   metadata: List[Dict[str, Any]]) -> faiss.Index:
        """
        Build FAISS index from embeddings
        
        Args:
            embeddings: Numpy array of embeddings
            metadata: List of metadata for each embedding
            
        Returns:
            FAISS index
        """
        logger.info(f"Building FAISS index for {len(embeddings)} embeddings")
        
        vector_size = embeddings.shape[1]
        
        # Create appropriate index type
        if self.index_type == "flat":
            index = faiss.IndexFlatL2(vector_size)
        elif self.index_type == "ivf":
            nlist = min(100, len(embeddings) // 10)  # Number of clusters
            quantizer = faiss.IndexFlatL2(vector_size)
            index = faiss.IndexIVFFlat(quantizer, vector_size, nlist)
        elif self.index_type == "hnsw":
            index = faiss.IndexHNSWFlat(vector_size, 32)  # 32 connections per node
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")
        
        # Train index if needed (for IVF)
        if self.index_type == "ivf":
            index.train(embeddings)
        
        # Add embeddings to index
        index.add(embeddings)
        
        # Store metadata
        self.metadata = metadata
        self.index = index
        
        logger.info(f"Built {self.index_type} index with {index.ntotal} vectors")
        return index
    
    def save_database(self, 
                     save_path: str, 
                     index: faiss.Index, 
                     metadata: List[Dict[str, Any]]):
        """
        Save FAISS index and metadata to disk
        
        Args:
            save_path: Directory to save the database
            index: FAISS index
            metadata: Metadata for each vector
        """
        logger.info(f"Saving database to: {save_path}")
        
        # Create directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)
        
        # Save FAISS index
        index_path = os.path.join(save_path, "index.faiss")
        faiss.write_index(index, index_path)
        
        # Save metadata
        metadata_path = os.path.join(save_path, "metadata.pkl")
        with open(metadata_path, "wb") as f:
            pickle.dump(metadata, f)
        
        # Save configuration
        config = {
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim,
            "index_type": self.index_type,
            "num_vectors": index.ntotal,
            "created_at": datetime.now().isoformat()
        }
        
        config_path = os.path.join(save_path, "config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        
        self.db_path = save_path
        logger.info(f"Database saved successfully to {save_path}")
    
    def load_database(self, db_path: str) -> tuple:
        """
        Load FAISS index and metadata from disk
        
        Args:
            db_path: Path to the database directory
            
        Returns:
            Tuple of (index, metadata, config)
        """
        logger.info(f"Loading database from: {db_path}")
        
        if not os.path.exists(db_path):
            raise ValueError(f"Database path does not exist: {db_path}")
        
        # Load FAISS index
        index_path = os.path.join(db_path, "index.faiss")
        index = faiss.read_index(index_path)
        
        # Load metadata
        metadata_path = os.path.join(db_path, "metadata.pkl")
        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)
        
        # Load configuration
        config_path = os.path.join(db_path, "config.json")
        config = {}
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
        
        self.index = index
        self.metadata = metadata
        self.db_path = db_path
        
        logger.info(f"Loaded database with {index.ntotal} vectors")
        return index, metadata, config
    
    def search(self, 
               query: str, 
               k: int = 5, 
               threshold: float = None) -> List[Dict[str, Any]]:
        """
        Perform semantic search
        
        Args:
            query: Search query
            k: Number of results to return
            threshold: Minimum similarity threshold (optional)
            
        Returns:
            List of search results with metadata
        """
        if self.index is None:
            raise ValueError("No index loaded. Please load a database first.")
        
        logger.info(f"Searching for: '{query}'")
        
        # Generate query embedding
        query_embedding = list(self.embedding_model.embed([query]))[0]
        query_vector = np.array([query_embedding], dtype='float32')
        
        # Search index
        distances, indices = self.index.search(query_vector, k)
        
        # Format results
        results = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx == -1:  # Invalid index
                continue
            
            if threshold is not None and distance > threshold:
                continue
            
            result = {
                "distance": float(distance),
                "similarity": float(1 / (1 + distance)),  # Convert distance to similarity
                "metadata": self.metadata[idx],
                "text": self.metadata[idx]["text"]
            }
            results.append(result)
        
        logger.info(f"Found {len(results)} results")
        return results
    
    def build_and_save_database(self, 
                               directory_path: str, 
                               save_path: str,
                               chunk_size: int = 1000,
                               chunk_overlap: int = 200,
                               batch_size: int = 64) -> str:
        """
        Complete pipeline: process directory, create embeddings, build index, and save
        
        Args:
            directory_path: Path to directory containing PDFs
            save_path: Path to save the database
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            batch_size: Batch size for embeddings
            
        Returns:
            Path to saved database
        """
        logger.info("Starting complete database build pipeline")
        
        # Step 1: Process directory
        text_chunks = self.process_directory(directory_path, chunk_size, chunk_overlap)
        
        if not text_chunks:
            raise ValueError("No text chunks created from directory")
        
        # Step 2: Create embeddings
        embeddings = self.create_embeddings(text_chunks, batch_size)
        
        # Step 3: Build index
        index = self.build_index(embeddings, text_chunks)
        
        # Step 4: Save database
        self.save_database(save_path, index, text_chunks)
        
        logger.info("Database build pipeline completed successfully")
        return save_path


def main():
    """
    Example usage of the FAISSEmbeddingManager
    """
    # Initialize manager
    manager = FAISSEmbeddingManager(
        model_name="BAAI/bge-small-en-v1.5",
        index_type="flat"
    )
    
    # Example: Build database from a directory
    try:
        directory_path = "C:\\Users\\akarsh\\Downloads"  # Change this to your PDF directory
        save_path = "./faiss_database"
        
        # Build and save database
        db_path = manager.build_and_save_database(
            directory_path=directory_path,
            save_path=save_path,
            chunk_size=1000,
            chunk_overlap=200
        )
        
        print(f"Database created at: {db_path}")
        
        # Example: Load database and search
        manager.load_database(db_path)
        
        # Perform a search
        results = manager.search("machine learning", k=3)
        
        print("\nSearch Results:")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Similarity: {result['similarity']:.3f}")
            print(f"   PDF: {result['metadata']['pdf_name']}")
            print(f"   Page: {result['metadata']['page_number']}")
            print(f"   Text: {result['text'][:200]}...")
    
    except Exception as e:
        logger.error(f"Error in main: {e}")


if __name__ == "__main__":
    main()
