import insightface
import numpy as np
from typing import List, Optional, Tuple, Dict
import cv2
from app.models.schemas import EmbeddingResult, DetectionResult
import os
from app.services.detection_service import detection_service
from app.core.log_config import logger
from app.core.similarity_config import similarity_config
from sklearn.feature_extraction import image as skimage
from sklearn.decomposition import PCA
import skimage.feature as feature

class EmbeddingService:
    def __init__(self):
        self.embedding_dimension = 512  # ArcFace uses 512-dimensional embeddings
        self.model = None
        self.color_weight = similarity_config.COLOR_WEIGHT
        self.structure_weight = similarity_config.STRUCTURE_WEIGHT
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
                    logger.info("No face detected, creating enhanced embedding from entire image region")
                    return self._create_enhanced_region_embedding(image_rgb)
            else:
                logger.warning("Invalid image format for embedding generation")
                return None
                
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return None
    
    def _create_enhanced_region_embedding(self, image: np.ndarray) -> np.ndarray:
        """
        Create enhanced embedding from entire image region with better feature engineering
        
        Args:
            image: Input image (RGB, 112x112)
            
        Returns:
            Enhanced embedding vector
        """
        try:
            # Enhanced feature extraction
            color_features = self._extract_enhanced_color_features(image)
            texture_features = self._extract_texture_features(image)
            shape_features = self._extract_shape_features(image)
            local_features = self._extract_local_binary_patterns(image)
            
            # Combine features with proper weighting
            combined_features = np.concatenate([
                color_features,      # Color features (higher weight)
                texture_features,    # Texture features
                shape_features,      # Shape features
                local_features       # Local binary patterns
            ])
            
            # Ensure we have exactly 512 dimensions
            if len(combined_features) > 512:
                # Use PCA to reduce to 512 dimensions while preserving important features
                combined_features = self._reduce_dimensions(combined_features, 512)
            elif len(combined_features) < 512:
                # Pad with zeros if needed
                padding = np.zeros(512 - len(combined_features))
                combined_features = np.concatenate([combined_features, padding])
            
            # Normalize the final vector
            norm = np.linalg.norm(combined_features)
            if norm > 0:
                combined_features = combined_features / norm
            
            logger.info(f"Created enhanced region embedding with shape: {combined_features.shape}")
            return combined_features
            
        except Exception as e:
            logger.error(f"Error creating enhanced region embedding: {e}")
            # Fallback: return random embedding
            fallback_embedding = np.random.randn(512).astype(np.float32)
            fallback_embedding = fallback_embedding / np.linalg.norm(fallback_embedding)
            logger.warning("Using fallback random embedding")
            return fallback_embedding
    
    def _extract_enhanced_color_features(self, image: np.ndarray) -> np.ndarray:
        """Extract enhanced color features with multiple color spaces and histograms"""
        try:
            # Convert to different color spaces for better color representation
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            
            # Calculate histograms for each channel with different bin sizes
            hist_r = cv2.calcHist([image], [0], None, [64], [0, 256]).flatten()
            hist_g = cv2.calcHist([image], [1], None, [64], [0, 256]).flatten()
            hist_b = cv2.calcHist([image], [2], None, [64], [0, 256]).flatten()
            
            hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180]).flatten()
            hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256]).flatten()
            hist_v = cv2.calcHist([hsv], [2], None, [32], [0, 256]).flatten()
            
            hist_l = cv2.calcHist([lab], [0], None, [32], [0, 256]).flatten()
            hist_a = cv2.calcHist([lab], [1], None, [32], [0, 256]).flatten()
            hist_b_lab = cv2.calcHist([lab], [2], None, [32], [0, 256]).flatten()
            
            # Calculate color moments (mean, std, skewness)
            color_moments = self._calculate_color_moments(image)
            
            # Calculate dominant colors using k-means clustering
            dominant_colors = self._extract_dominant_colors(image, k=8)
            
            # Combine all color features
            color_features = np.concatenate([
                hist_r, hist_g, hist_b,           # RGB histograms (192 features)
                hist_h, hist_s, hist_v,           # HSV histograms (96 features)
                hist_l, hist_a, hist_b_lab,       # LAB histograms (96 features)
                color_moments,                    # Color moments (9 features)
                dominant_colors                   # Dominant colors (24 features)
            ])
            
            # Normalize
            color_features = color_features / (np.sum(color_features) + 1e-8)
            
            # Apply color weight
            color_features = color_features * self.color_weight
            
            return color_features
            
        except Exception as e:
            logger.debug(f"Error extracting enhanced color features: {e}")
            return np.zeros(417)  # 417 = 192+96+96+9+24
    
    def _extract_texture_features(self, image: np.ndarray) -> np.ndarray:
        """Extract texture features using Gabor filters and Haralick features"""
        try:
            # Convert to grayscale for texture analysis
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Gabor filter features
            gabor_features = self._extract_gabor_features(gray)
            
            # Haralick texture features
            haralick_features = self._extract_haralick_features(gray)
            
            # Local Binary Pattern features
            lbp_features = self._extract_lbp_features(gray)
            
            # Combine texture features
            texture_features = np.concatenate([
                gabor_features,    # Gabor features
                haralick_features, # Haralick features
                lbp_features      # LBP features
            ])
            
            # Normalize
            texture_features = texture_features / (np.linalg.norm(texture_features) + 1e-8)
            
            return texture_features
            
        except Exception as e:
            logger.debug(f"Error extracting texture features: {e}")
            return np.zeros(50)
    
    def _extract_shape_features(self, image: np.ndarray) -> np.ndarray:
        """Extract shape and edge features"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Edge detection with multiple methods
            edges_canny = cv2.Canny(gray, 50, 150)
            edges_sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            edges_sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            edges_laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            
            # Calculate edge density and orientation features
            edge_density = np.sum(edges_canny > 0) / (edges_canny.shape[0] * edges_canny.shape[1])
            sobel_magnitude = np.sqrt(edges_sobel_x**2 + edges_sobel_y**2)
            edge_orientation = np.arctan2(edges_sobel_y, edges_sobel_x)
            
            # Contour features
            contours, _ = cv2.findContours(edges_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour_features = self._extract_contour_features(contours)
            
            # Create shape features
            shape_features = np.array([
                edge_density,
                np.mean(sobel_magnitude),
                np.std(sobel_magnitude),
                np.mean(edge_orientation),
                np.std(edge_orientation),
                np.mean(edges_laplacian),
                np.std(edges_laplacian)
            ])
            
            # Combine with contour features
            shape_features = np.concatenate([shape_features, contour_features])
            
            # Normalize
            shape_features = shape_features / (np.linalg.norm(shape_features) + 1e-8)
            
            return shape_features
            
        except Exception as e:
            logger.debug(f"Error extracting shape features: {e}")
            return np.zeros(10)
    
    def _extract_local_binary_patterns(self, image: np.ndarray) -> np.ndarray:
        """Extract Local Binary Pattern features"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Calculate LBP
            lbp = feature.local_binary_pattern(gray, P=8, R=1, method='uniform')
            
            # Calculate LBP histogram
            lbp_hist, _ = np.histogram(lbp, bins=10, range=(0, 10), density=True)
            
            return lbp_hist
            
        except Exception as e:
            logger.debug(f"Error extracting LBP features: {e}")
            return np.zeros(10)
    
    def _calculate_color_moments(self, image: np.ndarray) -> np.ndarray:
        """Calculate color moments (mean, std, skewness) for each channel"""
        try:
            moments = []
            for channel in range(3):
                channel_data = image[:, :, channel].flatten()
                mean = np.mean(channel_data)
                std = np.std(channel_data)
                skewness = self._calculate_skewness(channel_data)
                moments.extend([mean, std, skewness])
            
            return np.array(moments)
            
        except Exception as e:
            logger.debug(f"Error calculating color moments: {e}")
            return np.zeros(9)
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data"""
        try:
            mean = np.mean(data)
            std = np.std(data)
            if std == 0:
                return 0.0
            skewness = np.mean(((data - mean) / std) ** 3)
            return float(skewness)
        except:
            return 0.0
    
    def _extract_dominant_colors(self, image: np.ndarray, k: int = 8) -> np.ndarray:
        """Extract dominant colors using k-means clustering"""
        try:
            # Reshape image to 2D array of pixels
            pixels = image.reshape(-1, 3)
            
            # Convert to float32
            pixels = np.float32(pixels)
            
            # Define criteria and apply kmeans
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
            _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            
            # Count labels and get dominant colors
            unique_labels, counts = np.unique(labels, return_counts=True)
            
            # Create feature vector with dominant colors and their proportions
            dominant_colors = np.zeros(k * 3)
            for i, (label, count) in enumerate(zip(unique_labels, counts)):
                if i < k:
                    start_idx = label * 3
                    dominant_colors[start_idx:start_idx+3] = centers[label]
            
            # Normalize
            dominant_colors = dominant_colors / 255.0
            
            return dominant_colors
            
        except Exception as e:
            logger.debug(f"Error extracting dominant colors: {e}")
            return np.zeros(24)
    
    def _extract_gabor_features(self, gray_image: np.ndarray) -> np.ndarray:
        """Extract Gabor filter features"""
        try:
            # Define Gabor filter parameters
            angles = [0, 45, 90, 135]
            frequencies = [0.1, 0.3, 0.5]
            
            gabor_features = []
            for angle in angles:
                for freq in frequencies:
                    # Create Gabor kernel
                    kernel = cv2.getGaborKernel((21, 21), 5, np.radians(angle), 2*np.pi*freq, 0.5, 0, ktype=cv2.CV_32F)
                    
                    # Apply filter
                    filtered = cv2.filter2D(gray_image, cv2.CV_8UC3, kernel)
                    
                    # Calculate statistics
                    gabor_features.extend([
                        np.mean(filtered),
                        np.std(filtered),
                        np.var(filtered)
                    ])
            
            return np.array(gabor_features)
            
        except Exception as e:
            logger.debug(f"Error extracting Gabor features: {e}")
            return np.zeros(36)
    
    def _extract_haralick_features(self, gray_image: np.ndarray) -> np.ndarray:
        """Extract Haralick texture features"""
        try:
            # Calculate GLCM (Gray Level Co-occurrence Matrix)
            glcm = feature.graycomatrix(gray_image, [1], [0, 45, 90, 135], levels=256, symmetric=True, normed=True)
            
            # Calculate Haralick features
            contrast = feature.graycoprops(glcm, 'contrast')
            dissimilarity = feature.graycoprops(glcm, 'dissimilarity')
            homogeneity = feature.graycoprops(glcm, 'homogeneity')
            energy = feature.graycoprops(glcm, 'energy')
            correlation = feature.graycoprops(glcm, 'correlation')
            
            # Combine features
            haralick_features = np.concatenate([
                contrast.flatten(),
                dissimilarity.flatten(),
                homogeneity.flatten(),
                energy.flatten(),
                correlation.flatten()
            ])
            
            return haralick_features
            
        except Exception as e:
            logger.debug(f"Error extracting Haralick features: {e}")
            return np.zeros(20)
    
    def _extract_lbp_features(self, gray_image: np.ndarray) -> np.ndarray:
        """Extract Local Binary Pattern features"""
        try:
            # Calculate LBP with different parameters
            lbp_8_1 = feature.local_binary_pattern(gray_image, P=8, R=1, method='uniform')
            lbp_16_2 = feature.local_binary_pattern(gray_image, P=16, R=2, method='uniform')
            
            # Calculate histograms
            hist_8_1, _ = np.histogram(lbp_8_1, bins=10, range=(0, 10), density=True)
            hist_16_2, _ = np.histogram(lbp_16_2, bins=18, range=(0, 18), density=True)
            
            # Combine histograms
            lbp_features = np.concatenate([hist_8_1, hist_16_2])
            
            return lbp_features
            
        except Exception as e:
            logger.debug(f"Error extracting LBP features: {e}")
            return np.zeros(28)
    
    def _extract_contour_features(self, contours: List) -> np.ndarray:
        """Extract features from contours"""
        try:
            if not contours:
                return np.zeros(3)
            
            # Calculate contour features
            areas = [cv2.contourArea(contour) for contour in contours]
            perimeters = [cv2.arcLength(contour, True) for contour in contours]
            
            # Calculate circularity
            circularities = []
            for area, perimeter in zip(areas, perimeters):
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    circularities.append(circularity)
                else:
                    circularities.append(0)
            
            # Return summary statistics
            return np.array([
                np.mean(areas) if areas else 0,
                np.mean(perimeters) if perimeters else 0,
                np.mean(circularities) if circularities else 0
            ])
            
        except Exception as e:
            logger.debug(f"Error extracting contour features: {e}")
            return np.zeros(3)
    
    def _reduce_dimensions(self, features: np.ndarray, target_dim: int) -> np.ndarray:
        """Reduce feature dimensions using PCA while preserving important information"""
        try:
            if len(features) <= target_dim:
                return features
            
            # Reshape for PCA
            features_2d = features.reshape(1, -1)
            
            # Apply PCA
            pca = PCA(n_components=target_dim)
            reduced_features = pca.fit_transform(features_2d)
            
            return reduced_features.flatten()
            
        except Exception as e:
            logger.debug(f"Error reducing dimensions: {e}")
            # Fallback: simple truncation
            return features[:target_dim]
    
    def _extract_color_histogram(self, image: np.ndarray) -> np.ndarray:
        """Extract color histogram features from image (legacy method)"""
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
        """Extract edge detection features from image (legacy method)"""
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
    
    def _create_region_embedding(self, image: np.ndarray) -> np.ndarray:
        """
        Create embedding from entire image region when face detection fails (legacy method)
        
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
    
    def compute_enhanced_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray, 
                                  image1: Optional[np.ndarray] = None, 
                                  image2: Optional[np.ndarray] = None) -> float:
        """
        Compute enhanced similarity using multiple metrics and color pre-filtering
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            image1: First image (optional, for color comparison)
            image2: Second image (optional, for color comparison)
            
        Returns:
            Enhanced similarity score (0-1, higher is more similar)
        """
        try:
            # Color-based pre-filtering if images are provided
            if image1 is not None and image2 is not None:
                color_similarity = self._compute_color_similarity(image1, image2)
                
                # Apply color penalty based on configuration
                if color_similarity < similarity_config.COLOR_SIMILARITY_LOW:
                    if similarity_config.LOG_COLOR_SIMILARITY:
                        logger.info(f"Low color similarity detected: {color_similarity:.3f}")
                    color_penalty = similarity_config.COLOR_PENALTY_LOW
                elif color_similarity < similarity_config.COLOR_SIMILARITY_MEDIUM:
                    color_penalty = similarity_config.COLOR_PENALTY_MEDIUM
                else:
                    color_penalty = similarity_config.COLOR_PENALTY_HIGH
            else:
                color_penalty = similarity_config.COLOR_PENALTY_HIGH
            
            # Compute multiple similarity metrics
            cosine_sim = self.compute_similarity(embedding1, embedding2)
            euclidean_dist = self.compute_distance(embedding1, embedding2)
            
            # Normalize Euclidean distance to 0-1 range (lower distance = higher similarity)
            # Use exponential decay for better sensitivity
            normalized_dist = np.exp(-euclidean_dist * similarity_config.EUCLIDEAN_SCALE_FACTOR)
            
            # Compute Manhattan distance for additional perspective
            manhattan_dist = self._compute_manhattan_distance(embedding1, embedding2)
            normalized_manhattan = np.exp(-manhattan_dist / similarity_config.MANHATTAN_SCALE_FACTOR)
            
            # Weighted combination of multiple metrics using configuration
            enhanced_similarity = (
                similarity_config.COSINE_WEIGHT * cosine_sim +
                similarity_config.EUCLIDEAN_WEIGHT * normalized_dist +
                similarity_config.MANHATTAN_WEIGHT * normalized_manhattan
            )
            
            # Apply color penalty
            final_similarity = enhanced_similarity * color_penalty
            
            if similarity_config.LOG_ENHANCED_SIMILARITY:
                logger.info(f"Enhanced similarity: cosine={cosine_sim:.3f}, "
                           f"euclidean={normalized_dist:.3f}, manhattan={normalized_manhattan:.3f}, "
                           f"color_penalty={color_penalty:.3f}, final={final_similarity:.3f}")
            
            return float(final_similarity)
            
        except Exception as e:
            logger.error(f"Error computing enhanced similarity: {e}")
            # Fallback to cosine similarity
            return self.compute_similarity(embedding1, embedding2)
    
    def _compute_color_similarity(self, image1: np.ndarray, image2: np.ndarray) -> float:
        """
        Compute color similarity between two images using multiple color spaces
        
        Args:
            image1: First image
            image2: Second image
            
        Returns:
            Color similarity score (0-1, higher is more similar)
        """
        try:
            # Ensure both images are the same size for comparison
            target_size = similarity_config.COLOR_COMPARISON_SIZE
            img1_resized = cv2.resize(image1, target_size)
            img2_resized = cv2.resize(image2, target_size)
            
            # Convert to different color spaces
            color_spaces = {
                'rgb': (img1_resized, img2_resized),
                'hsv': (cv2.cvtColor(img1_resized, cv2.COLOR_RGB2HSV), 
                       cv2.cvtColor(img2_resized, cv2.COLOR_RGB2HSV)),
                'lab': (cv2.cvtColor(img1_resized, cv2.COLOR_RGB2LAB), 
                       cv2.cvtColor(img2_resized, cv2.COLOR_RGB2LAB)),
                'yuv': (cv2.cvtColor(img1_resized, cv2.COLOR_RGB2YUV), 
                       cv2.cvtColor(img2_resized, cv2.COLOR_RGB2YUV))
            }
            
            color_similarities = []
            
            for space_name, (img1_space, img2_space) in color_spaces.items():
                # Calculate histogram for each channel
                hist1 = self._calculate_color_histogram_simple(img1_space)
                hist2 = self._calculate_color_histogram_simple(img2_space)
                
                # Compute histogram intersection similarity
                intersection = np.minimum(hist1, hist2)
                union = np.maximum(hist1, hist2)
                
                if np.sum(union) > 0:
                    similarity = np.sum(intersection) / np.sum(union)
                    color_similarities.append(similarity)
                else:
                    color_similarities.append(0.0)
            
            # Return average similarity across all color spaces
            avg_similarity = np.mean(color_similarities)
            
            if similarity_config.LOG_COLOR_SIMILARITY:
                logger.debug(f"Color similarities: {dict(zip(color_spaces.keys(), color_similarities))}, "
                            f"average: {avg_similarity:.3f}")
            
            return float(avg_similarity)
            
        except Exception as e:
            logger.error(f"Error computing color similarity: {e}")
            return 0.5  # Neutral similarity on error
    
    def _calculate_color_histogram_simple(self, image: np.ndarray) -> np.ndarray:
        """Calculate simple color histogram for color similarity comparison"""
        try:
            # Calculate histograms for each channel
            hist_r = cv2.calcHist([image], [0], None, [32], [0, 256]).flatten()
            hist_g = cv2.calcHist([image], [1], None, [32], [0, 256]).flatten()
            hist_b = cv2.calcHist([image], [2], None, [32], [0, 256]).flatten()
            
            # Combine and normalize
            combined_hist = np.concatenate([hist_r, hist_g, hist_b])
            normalized_hist = combined_hist / (np.sum(combined_hist) + 1e-8)
            
            return normalized_hist
            
        except Exception as e:
            logger.debug(f"Error calculating simple color histogram: {e}")
            return np.zeros(96)
    
    def _compute_manhattan_distance(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute Manhattan distance between two embeddings"""
        try:
            return float(np.sum(np.abs(embedding1 - embedding2)))
        except Exception as e:
            logger.debug(f"Error computing Manhattan distance: {e}")
            return float('inf')
    
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

    def compute_combined_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute combined similarity using multiple metrics (legacy method)
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Combined similarity score (0-1, higher is more similar)
        """
        try:
            cosine_sim = self.compute_similarity(embedding1, embedding2)
            euclidean_dist = self.compute_distance(embedding1, embedding2)
            
            # Normalize Euclidean distance to 0-1 range
            normalized_dist = 1.0 / (1.0 + euclidean_dist)
            
            # Weighted combination
            combined_score = 0.7 * cosine_sim + 0.3 * normalized_dist
            return combined_score
            
        except Exception as e:
            logger.error(f"Error computing combined similarity: {e}")
            return 0.0

    def _color_similarity_check(self, img1: np.ndarray, img2: np.ndarray, threshold: float = 0.3) -> bool:
        """Quick color similarity check before expensive embedding comparison"""
        hist1 = self._extract_color_histogram(img1)
        hist2 = self._extract_color_histogram(img2)
        
        color_sim = np.dot(hist1, hist2) / (np.linalg.norm(hist1) * np.linalg.norm(hist2))
        return color_sim >= threshold

# Global instance
embedding_service = EmbeddingService() 