"""
Database Module for storing and retrieving face embeddings
"""
import pickle
import os
from pathlib import Path


class EmbeddingDatabase:
    def __init__(self, embeddings_path="data/registered_faces/embeddings.pkl"):
        """
        Initialize embedding database
        
        Args:
            embeddings_path: Path to pickle file storing embeddings
        """
        self.embeddings_path = embeddings_path
        self.embeddings = {}
        self._ensure_directory_exists()
        self.load_embeddings()
    
    def _ensure_directory_exists(self):
        """Ensure the directory for embeddings file exists"""
        os.makedirs(os.path.dirname(self.embeddings_path), exist_ok=True)
    
    def save_embedding(self, name, embedding):
        """
        Save or update embedding for a person
        
        Args:
            name: Person's name
            embedding: 128-dimensional embedding vector
        """
        if name is None or name.strip() == "":
            raise ValueError("Name cannot be empty")
        
        if embedding is None:
            raise ValueError("Embedding cannot be None")
        
        # Load existing embeddings
        self.load_embeddings()
        
        # Add or update embedding
        self.embeddings[name.strip()] = embedding
        
        # Save to file
        try:
            with open(self.embeddings_path, 'wb') as f:
                pickle.dump(self.embeddings, f)
        except Exception as e:
            raise Exception(f"Failed to save embedding: {str(e)}")
    
    def load_embeddings(self):
        """
        Load all embeddings from pickle file
        
        Returns:
            dict: Dictionary mapping names to embeddings
        """
        if os.path.exists(self.embeddings_path):
            try:
                with open(self.embeddings_path, 'rb') as f:
                    self.embeddings = pickle.load(f)
            except Exception as e:
                print(f"Warning: Failed to load embeddings: {str(e)}")
                self.embeddings = {}
        else:
            self.embeddings = {}
        
        return self.embeddings.copy()
    
    def get_all_names(self):
        """
        Get list of all registered names
        
        Returns:
            list: List of registered names
        """
        self.load_embeddings()
        return list(self.embeddings.keys())
    
    def get_embedding(self, name):
        """
        Get embedding for a specific name
        
        Args:
            name: Person's name
            
        Returns:
            embedding: Embedding vector or None if not found
        """
        self.load_embeddings()
        return self.embeddings.get(name.strip())
    
    def delete_embedding(self, name):
        """
        Delete embedding for a person
        
        Args:
            name: Person's name
        """
        self.load_embeddings()
        if name.strip() in self.embeddings:
            del self.embeddings[name.strip()]
            with open(self.embeddings_path, 'wb') as f:
                pickle.dump(self.embeddings, f)
            return True
        return False
    
    def get_count(self):
        """
        Get total number of registered faces
        
        Returns:
            int: Number of registered faces
        """
        self.load_embeddings()
        return len(self.embeddings)

