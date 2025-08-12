import faiss
import numpy as np
import json
import os
import pickle
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import uuid
from app.core.config import settings
from app.core.log_config import logger
from app.models.schemas import DogInfo, SimilarityMatch

class FAISSService:
    def __init__(self):
        self.index = None
        self.dog_database: Dict[str, DogInfo] = {}
        self.embedding_to_dog_id: List[str] = []
        self.load_database()
    
    def load_database(self):
        """Load existing FAISS index and dog database"""
        try:
            # Load FAISS index
            if os.path.exists(settings.database_path):
                self.index = faiss.read_index(settings.database_path)
                logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors")
            else:
                # Create new index
                self.index = faiss.IndexFlatIP(settings.embedding_dimension)
                logger.info("Created new FAISS index")
            
            # Load dog database metadata
            metadata_path = settings.database_path.replace('.faiss', '_metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    data = json.load(f)
                    self.dog_database = {k: DogInfo(**v) for k, v in data['dogs'].items()}
                    self.embedding_to_dog_id = data['embedding_to_dog_id']
                logger.info(f"Loaded {len(self.dog_database)} dogs from database")
            
        except Exception as e:
            logger.error(f"Error loading database: {e}")
            # Initialize empty database
            self.index = faiss.IndexFlatIP(settings.embedding_dimension)
            self.dog_database = {}
            self.embedding_to_dog_id = []
    
    def save_database(self):
        """Save FAISS index and dog database metadata"""
        try:
            # Save FAISS index
            faiss.write_index(self.index, settings.database_path)
            
            # Save metadata
            metadata_path = settings.database_path.replace('.faiss', '_metadata.json')
            metadata = {
                'dogs': {k: v.dict() for k, v in self.dog_database.items()},
                'embedding_to_dog_id': self.embedding_to_dog_id,
                'last_updated': datetime.now().isoformat()
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            logger.info(f"Database saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving database: {e}")
    
    def add_dog(self, dog_info: DogInfo, embeddings: List[np.ndarray]) -> bool:
        """
        Add a new dog with its embeddings to the database
        
        Args:
            dog_info: Dog information
            embeddings: List of embedding vectors
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not embeddings:
                logger.warning("No embeddings provided")
                return False
            
            # Normalize embeddings
            normalized_embeddings = []
            for emb in embeddings:
                norm_emb = self._normalize_embedding(emb)
                normalized_embeddings.append(norm_emb)
            
            # Convert to numpy array
            embeddings_array = np.array(normalized_embeddings, dtype=np.float32)
            
            # Add to FAISS index
            self.index.add(embeddings_array)
            
            # Update metadata
            self.dog_database[dog_info.id] = dog_info
            for _ in range(len(embeddings)):
                self.embedding_to_dog_id.append(dog_info.id)
            
            # Save database
            self.save_database()
            
            logger.info(f"Added dog {dog_info.name} with {len(embeddings)} embeddings")
            return True
            
        except Exception as e:
            logger.error(f"Error adding dog to database: {e}")
            return False
    
    def search_similar(self, query_embedding: np.ndarray, k: int = 5, 
                      threshold: float = 0.6) -> List[SimilarityMatch]:
        """
        Search for similar dogs in the database
        
        Args:
            query_embedding: Query embedding vector
            k: Number of top results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of SimilarityMatch objects
        """
        try:
            if self.index.ntotal == 0:
                return []
            
            # Normalize query embedding
            norm_query = self._normalize_embedding(query_embedding)
            query_array = np.array([norm_query], dtype=np.float32)
            
            # Search in FAISS index
            similarities, indices = self.index.search(query_array, min(k, self.index.ntotal))
            
            matches = []
            for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
                if idx < 0 or idx >= len(self.embedding_to_dog_id):
                    continue
                
                dog_id = self.embedding_to_dog_id[idx]
                if dog_id in self.dog_database:
                    dog_info = self.dog_database[dog_id]
                    
                    # Convert similarity to distance (for consistency)
                    distance = 1.0 - similarity
                    
                    match = SimilarityMatch(
                        dog_id=dog_id,
                        dog_info=dog_info,
                        similarity_score=float(similarity),
                        distance=float(distance)
                    )
                    matches.append(match)
            
            # Filter by threshold and sort by similarity
            matches = [m for m in matches if m.similarity_score >= threshold]
            matches.sort(key=lambda x: x.similarity_score, reverse=True)
            
            return matches
            
        except Exception as e:
            logger.error(f"Error searching database: {e}")
            return []
    
    def get_database_info(self) -> Dict:
        """Get database statistics"""
        try:
            db_size = 0
            if os.path.exists(settings.database_path):
                db_size = os.path.getsize(settings.database_path) / (1024 * 1024)  # MB
            
            return {
                'total_dogs': len(self.dog_database),
                'total_embeddings': self.index.ntotal if self.index else 0,
                'database_size_mb': round(db_size, 2),
                'embedding_dimension': settings.embedding_dimension
            }
            
        except Exception as e:
            logger.error(f"Error getting database info: {e}")
            return {
                'total_dogs': 0,
                'total_embeddings': 0,
                'database_size_mb': 0,
                'embedding_dimension': settings.embedding_dimension
            }
    
    def remove_dog(self, dog_id: str) -> bool:
        """
        Remove a dog and all its embeddings from the database
        
        Args:
            dog_id: ID of the dog to remove
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if dog_id not in self.dog_database:
                logger.warning(f"Dog {dog_id} not found in database")
                return False
            
            # Find all embeddings for this dog
            indices_to_remove = []
            for i, stored_dog_id in enumerate(self.embedding_to_dog_id):
                if stored_dog_id == dog_id:
                    indices_to_remove.append(i)
            
            if not indices_to_remove:
                logger.warning(f"No embeddings found for dog {dog_id}")
                return False
            
            # Remove from FAISS index (this is complex, so we'll rebuild)
            self._rebuild_index_excluding(indices_to_remove)
            
            # Remove from metadata
            del self.dog_database[dog_id]
            
            # Save database
            self.save_database()
            
            logger.info(f"Removed dog {dog_id} with {len(indices_to_remove)} embeddings")
            return True
            
        except Exception as e:
            logger.error(f"Error removing dog: {e}")
            return False
    
    def _normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """Normalize embedding vector"""
        norm = np.linalg.norm(embedding)
        if norm > 0:
            return embedding / norm
        return embedding
    
    def _rebuild_index_excluding(self, indices_to_remove: List[int]):
        """Rebuild FAISS index excluding specified indices"""
        try:
            # Get all vectors except those to be removed
            all_vectors = self.index.reconstruct_n(0, self.index.ntotal)
            keep_indices = [i for i in range(self.index.ntotal) if i not in indices_to_remove]
            
            if keep_indices:
                kept_vectors = all_vectors[keep_indices]
                kept_dog_ids = [self.embedding_to_dog_id[i] for i in keep_indices]
                
                # Create new index
                new_index = faiss.IndexFlatIP(settings.embedding_dimension)
                new_index.add(kept_vectors)
                
                # Update
                self.index = new_index
                self.embedding_to_dog_id = kept_dog_ids
            else:
                # Empty database
                self.index = faiss.IndexFlatIP(settings.embedding_dimension)
                self.embedding_to_dog_id = []
                
        except Exception as e:
            logger.error(f"Error rebuilding index: {e}")

# Global instance
faiss_service = FAISSService() 