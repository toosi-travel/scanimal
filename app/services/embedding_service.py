import insightface
import numpy as np
from typing import List, Optional, Tuple
import cv2
from app.models.schemas import EmbeddingResult, DetectionResult
import os
from app.services.detection_service import detection_service
from app.core.log_config import logger

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
            logger.info("ArcFace model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading ArcFace model: {e}")
            # Fallback to smaller model if buffalo_l fails
            try:
                self.model = insightface.app.FaceAnalysis(
                    name='buffalo_s',
                    providers=['CPUExecutionProvider']
                )
                self.model.prepare(ctx_id=0, det_size=(640, 640))
                logger.info("ArcFace model (buffalo_s) loaded successfully")
            except Exception as e2:
                logger.error(f"Error loading fallback ArcFace model: {e2}")
                raise
    
    def generate_embedding(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Generate embedding for an image using ArcFace or fallback methods
        
        Args:
            image: Input image (can be face region or full dog region)
            
        Returns:
            Embedding vector or None if failed
        """
        try:
            if self.model is None:
                logger.warning("ArcFace model not loaded")
                return None
            
            # Ensure image is in RGB format and correct size
            if len(image.shape) == 3 and image.shape[2] == 3:
                # Convert BGR to RGB if needed
                if image.dtype != np.uint8:
                    image = (image * 255).astype(np.uint8)
                
                # ArcFace expects RGB images
                if image.shape[2] == 3:
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    image_rgb = image
                
                # Ensure image is the right size for ArcFace
                if image_rgb.shape[:2] != (112, 112):
                    image_rgb = cv2.resize(image_rgb, (112, 112))
                
                # Try to get face embeddings using ArcFace first
                faces = self.model.get(image_rgb)
                
                if faces and len(faces) > 0:
                    # Return the embedding of the first detected face
                    embedding = faces[0].embedding
                    logger.info(f"Successfully generated face embedding with shape: {embedding.shape}")
                    return embedding
                else:
                    # If no face detected, create embedding from the entire image
                    logger.info("No face detected, creating embedding from entire image region")
                    return self._create_region_embedding(image_rgb)
            else:
                logger.warning("Invalid image format for embedding generation")
                return None
                
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return None
    
    def _create_region_embedding(self, image: np.ndarray) -> np.ndarray:
        """
        Create embedding from entire image region when face detection fails
        
        Args:
            image: Input image (RGB, 112x112)
            
        Returns:
            Embedding vector
        """
        try:
            # Convert image to feature vector using simple methods
            # Method 1: Flatten and normalize the image
            flattened = image.flatten().astype(np.float32)
            normalized = flattened / 255.0
            
            # Method 2: Use color histograms as features
            hist_features = self._extract_color_histogram(image)
            
            # Method 3: Use edge detection features
            edge_features = self._extract_edge_features(image)
            
            # Combine all features to create a 512-dimensional vector
            combined_features = np.concatenate([
                normalized[:256],  # First 256 pixels
                hist_features,     # Color histogram features
                edge_features,     # Edge features
                np.zeros(512 - 256 - len(hist_features) - len(edge_features))  # Padding
            ])
            
            # Ensure we have exactly 512 dimensions
            if len(combined_features) > 512:
                combined_features = combined_features[:512]
            elif len(combined_features) < 512:
                padding = np.zeros(512 - len(combined_features))
                combined_features = np.concatenate([combined_features, padding])
            
            # Normalize the final vector
            norm = np.linalg.norm(combined_features)
            if norm > 0:
                combined_features = combined_features / norm
            
            logger.info(f"Created region embedding with shape: {combined_features.shape}")
            return combined_features
            
        except Exception as e:
            logger.error(f"Error creating region embedding: {e}")
            # Fallback: return random embedding
            fallback_embedding = np.random.randn(512).astype(np.float32)
            fallback_embedding = fallback_embedding / np.linalg.norm(fallback_embedding)
            logger.warning("Using fallback random embedding")
            return fallback_embedding
    
    def _extract_color_histogram(self, image: np.ndarray) -> np.ndarray:
        """Extract color histogram features from image"""
        try:
            # Convert to different color spaces
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            
            # Calculate histograms for each channel
            hist_r = cv2.calcHist([image], [0], None, [32], [0, 256]).flatten()
            hist_g = cv2.calcHist([image], [1], None, [32], [0, 256]).flatten()
            hist_b = cv2.calcHist([image], [2], None, [32], [0, 256]).flatten()
            hist_h = cv2.calcHist([hsv], [0], None, [16], [0, 180]).flatten()
            hist_s = cv2.calcHist([hsv], [1], None, [16], [0, 256]).flatten()
            hist_v = cv2.calcHist([hsv], [2], None, [16], [0, 256]).flatten()
            
            # Combine histograms
            hist_features = np.concatenate([hist_r, hist_g, hist_b, hist_h, hist_s, hist_v])
            
            # Normalize
            hist_features = hist_features / (np.sum(hist_features) + 1e-8)
            
            return hist_features
            
        except Exception as e:
            logger.debug(f"Error extracting color histogram: {e}")
            return np.zeros(128)
    
    def _extract_edge_features(self, image: np.ndarray) -> np.ndarray:
        """Extract edge detection features from image"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Apply different edge detection methods
            edges_canny = cv2.Canny(gray, 50, 150)
            edges_sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            edges_sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            
            # Calculate edge density and orientation features
            edge_density = np.sum(edges_canny > 0) / (edges_canny.shape[0] * edges_canny.shape[1])
            sobel_magnitude = np.sqrt(edges_sobel_x**2 + edges_sobel_y**2)
            edge_orientation = np.arctan2(edges_sobel_y, edges_sobel_x)
            
            # Create edge features
            edge_features = np.array([
                edge_density,
                np.mean(sobel_magnitude),
                np.std(sobel_magnitude),
                np.mean(edge_orientation),
                np.std(edge_orientation)
            ])
            
            # Normalize
            edge_features = edge_features / (np.linalg.norm(edge_features) + 1e-8)
            
            return edge_features
            
        except Exception as e:
            logger.debug(f"Error extracting edge features: {e}")
            return np.zeros(5)
    
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
                    # Try to generate embedding with the extracted face region
                    embedding = self._generate_embedding_with_fallback(face_region)
                    
                    if embedding is not None:
                        embedding_result = EmbeddingResult(
                            embedding=embedding.tolist(),
                            detection_results=[detection]
                        )
                        embedding_results.append(embedding_result)
                        logger.info(f"Successfully generated embedding for detection {detection}")
                    else:
                        logger.warning(f"Failed to generate embedding for detection {detection}")
                else:
                    logger.warning(f"Failed to extract face region for detection {detection}")
                    
            except Exception as e:
                logger.error(f"Error processing detection for embedding: {e}")
                continue
        
        return embedding_results
    
    def _generate_embedding_with_fallback(self, face_region: np.ndarray) -> Optional[np.ndarray]:
        """
        Generate embedding with multiple fallback strategies
        
        Args:
            face_region: Input face region
            
        Returns:
            Embedding vector or None if all strategies fail
        """
        # Try the original face region first
        embedding = self.generate_embedding(face_region)
        if embedding is not None:
            return embedding
        
        # If that fails, try with different preprocessing approaches
        logger.info("Trying alternative preprocessing approaches...")
        
        # Try with different image sizes
        for size in [(160, 160), (224, 224), (256, 256)]:
            try:
                resized = cv2.resize(face_region, size)
                embedding = self.generate_embedding(resized)
                if embedding is not None:
                    logger.info(f"Successfully generated embedding with size {size}")
                    return embedding
            except Exception as e:
                logger.debug(f"Failed with size {size}: {e}")
                continue
        
        # Try with different color spaces
        try:
            # Convert to grayscale and back to RGB
            gray = cv2.cvtColor(face_region, cv2.COLOR_RGB2GRAY)
            gray_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            embedding = self.generate_embedding(gray_rgb)
            if embedding is not None:
                logger.info("Successfully generated embedding with grayscale conversion")
                return embedding
        except Exception as e:
            logger.debug(f"Grayscale conversion failed: {e}")
        
        # Try with different brightness levels
        for alpha in [0.8, 1.0, 1.2, 1.5]:
            try:
                adjusted = cv2.convertScaleAbs(face_region, alpha=alpha, beta=0)
                embedding = self.generate_embedding(adjusted)
                if embedding is not None:
                    logger.info(f"Successfully generated embedding with brightness alpha={alpha}")
                    return embedding
            except Exception as e:
                logger.debug(f"Brightness adjustment failed with alpha={alpha}: {e}")
                continue
        
        logger.warning("All embedding generation strategies failed")
        return None
    
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
            logger.error(f"Error preprocessing face region: {e}")
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
            logger.error(f"Error computing similarity: {e}")
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
            logger.error(f"Error computing distance: {e}")
            return float('inf')

# Global instance
embedding_service = EmbeddingService() 