import insightface
import numpy as np
from typing import List, Optional, Tuple
import cv2
from app.models.schemas import EmbeddingResult, DetectionResult
import os
from app.services.detection_service import detection_service

class EmbeddingService:
    def __init__(self):
        self.embedding_dimension = 512  # ArcFace uses 512-dimensional embeddings
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load ArcFace model for face embedding generation"""
        try:
            # Initialize insightface app
            self.model = insightface.app.FaceAnalysis(
                name='buffalo_l',  # Use buffalo_l model for better accuracy
                providers=['CPUExecutionProvider']  # Use CPU for compatibility
            )
            self.model.prepare(ctx_id=0, det_size=(640, 640))
            print("ArcFace model loaded successfully")
        except Exception as e:
            print(f"Error loading ArcFace model: {e}")
            # Fallback to smaller model if buffalo_l fails
            try:
                self.model = insightface.app.FaceAnalysis(
                    name='buffalo_s',
                    providers=['CPUExecutionProvider']
                )
                self.model.prepare(ctx_id=0, det_size=(640, 640))
                print("ArcFace model (buffalo_s) loaded successfully")
            except Exception as e2:
                print(f"Error loading fallback ArcFace model: {e2}")
                raise
    
    def generate_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Generate embedding for a face image using ArcFace
        
        Args:
            face_image: Preprocessed face image (RGB, 112x112 for ArcFace)
            
        Returns:
            Embedding vector or None if failed
        """
        try:
            if self.model is None:
                print("ArcFace model not loaded")
                return None
            
            # Ensure image is in RGB format
            if len(face_image.shape) == 3 and face_image.shape[2] == 3:
                # Convert BGR to RGB if needed
                if face_image.dtype != np.uint8:
                    face_image = (face_image * 255).astype(np.uint8)
                
                # ArcFace expects RGB images
                if face_image.shape[2] == 3:
                    face_image_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
                else:
                    face_image_rgb = face_image
                
                # Get face embeddings using ArcFace
                faces = self.model.get(face_image_rgb)
                
                if faces and len(faces) > 0:
                    # Return the embedding of the first detected face
                    embedding = faces[0].embedding
                    return embedding
                else:
                    print("No face detected in the image")
                    return None
            else:
                print("Invalid image format for face encoding")
                return None
                
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return None
    
    def generate_embeddings_from_detections(self, image: np.ndarray, 
                                          detections: List[DetectionResult]) -> List[EmbeddingResult]:
        """
        Generate embeddings for all detected faces in an image
        
        Args:
            image: Original image
            detections: List of detection results
            
        Returns:
            List of EmbeddingResult objects
        """
        embedding_results = []
        
        for detection in detections:
            try:
                # Extract face region from detection
                face_region = detection_service.extract_face_region(image, detection)
                
                if face_region is not None:
                    # Preprocess for ArcFace embedding
                    preprocessed_face = self.preprocess_for_arcface(face_region)
                    
                    # Generate embedding
                    embedding = self.generate_embedding(preprocessed_face)
                    
                    if embedding is not None:
                        embedding_result = EmbeddingResult(
                            embedding=embedding.tolist(),
                            detection_results=[detection]
                        )
                        embedding_results.append(embedding_result)
                    else:
                        print(f"Failed to generate embedding for detection {detection}")
                else:
                    print(f"Failed to extract face region for detection {detection}")
                    
            except Exception as e:
                print(f"Error processing detection for embedding: {e}")
                continue
        
        return embedding_results
    
    def preprocess_for_arcface(self, face_region: np.ndarray) -> np.ndarray:
        """
        Preprocess face region for ArcFace embedding generation
        
        Args:
            face_region: Cropped face region
            
        Returns:
            Preprocessed image for ArcFace (112x112 RGB)
        """
        try:
            # ArcFace expects 112x112 RGB images
            target_size = (112, 112)
            resized = cv2.resize(face_region, target_size)
            
            # Ensure it's in RGB format
            if len(resized.shape) == 3 and resized.shape[2] == 3:
                # Convert BGR to RGB
                resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            
            return resized
            
        except Exception as e:
            print(f"Error preprocessing face region: {e}")
            return face_region
    
    def normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """
        Normalize embedding vector (L2 normalization)
        
        Args:
            embedding: Input embedding vector
            
        Returns:
            Normalized embedding vector
        """
        norm = np.linalg.norm(embedding)
        if norm > 0:
            return embedding / norm
        return embedding
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Similarity score (0-1, higher is more similar)
        """
        try:
            # Normalize embeddings
            norm1 = self.normalize_embedding(embedding1)
            norm2 = self.normalize_embedding(embedding2)
            
            # Compute cosine similarity
            similarity = np.dot(norm1, norm2)
            return float(similarity)
            
        except Exception as e:
            print(f"Error computing similarity: {e}")
            return 0.0
    
    def compute_distance(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute Euclidean distance between two embeddings
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Distance (lower is more similar)
        """
        try:
            # Normalize embeddings
            norm1 = self.normalize_embedding(embedding1)
            norm2 = self.normalize_embedding(embedding2)
            
            # Compute Euclidean distance
            distance = np.linalg.norm(norm1 - norm2)
            return float(distance)
            
        except Exception as e:
            print(f"Error computing distance: {e}")
            return float('inf')

# Global instance
embedding_service = EmbeddingService() 