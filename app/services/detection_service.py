import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple, Optional
import os
from app.core.config import settings
from app.core.log_config import logger
from app.models.schemas import DetectionResult

class DogDetectionService:
    def __init__(self):
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load YOLOv8 model for dog detection"""
        try:
            # Check if model file exists, if not download it
            if not os.path.exists(settings.yolo_model_path):
                logger.info(f"Downloading {settings.yolo_model_path}...")
                self.model = YOLO(settings.yolo_model_path)
            else:
                self.model = YOLO(settings.yolo_model_path)
            logger.info("YOLOv8 model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading YOLOv8 model: {e}")
            raise
    
    def detect_dogs(self, image: np.ndarray) -> List[DetectionResult]:
        """
        Detect dogs in the image using YOLOv8 with fallback strategies
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of DetectionResult objects
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        try:
            # Try with normal confidence threshold first
            detections = self._detect_with_threshold(image, settings.confidence_threshold)
            
            # If no detections, try with lower confidence threshold
            if not detections:
                logger.info("No dogs detected with normal threshold, trying lower confidence...")
                detections = self._detect_with_threshold(image, 0.3)  # Lower threshold
            
            # If still no detections, try with even lower threshold
            if not detections:
                logger.info("No dogs detected with lower threshold, trying very low confidence...")
                detections = self._detect_with_threshold(image, 0.1)  # Very low threshold
            
            # If still no detections, try image preprocessing
            if not detections:
                logger.info("No dogs detected with any threshold, trying image preprocessing...")
                detections = self._detect_with_preprocessing(image)
            
            if detections:
                logger.info(f"Successfully detected {len(detections)} dogs")
            else:
                logger.warning("No dogs detected even with all strategies")
            
            return detections
            
        except Exception as e:
            logger.error(f"Error during detection: {e}")
            return []
    
    def _detect_with_threshold(self, image: np.ndarray, confidence_threshold: float) -> List[DetectionResult]:
        """Detect dogs with a specific confidence threshold"""
        try:
            results = self.model(image, conf=confidence_threshold, 
                               iou=settings.nms_threshold, verbose=False)
            
            detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
                        # Get confidence and class
                        confidence = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy())
                        class_name = result.names[class_id]
                        
                        # Filter for dog-related classes (COCO dataset)
                        dog_classes = ['dog']  
                        if class_name.lower() in dog_classes:
                            detection = DetectionResult(
                                bbox=[float(x1), float(y1), float(x2), float(y2)],
                                confidence=confidence,
                                class_id=class_id,
                                class_name=class_name
                            )
                            detections.append(detection)
                            logger.info(f"Detected dog with confidence: {confidence:.3f}")
            
            return detections
            
        except Exception as e:
            logger.error(f"Error in detection with threshold {confidence_threshold}: {e}")
            return []
    
    def _detect_with_preprocessing(self, image: np.ndarray) -> List[DetectionResult]:
        """Try detection with image preprocessing strategies"""
        try:
            # Strategy 1: Try with brightness/contrast adjustment
            logger.info("Trying brightness/contrast adjustment...")
            adjusted_image = self._adjust_brightness_contrast(image)
            detections = self._detect_with_threshold(adjusted_image, 0.1)
            if detections:
                return detections
            
            # Strategy 2: Try with histogram equalization
            logger.info("Trying histogram equalization...")
            equalized_image = self._equalize_histogram(image)
            detections = self._detect_with_threshold(equalized_image, 0.1)
            if detections:
                return detections
            
            # Strategy 3: Try with different color spaces
            logger.info("Trying different color spaces...")
            hsv_image = self._convert_to_hsv(image)
            detections = self._detect_with_threshold(hsv_image, 0.1)
            if detections:
                return detections
            
            # Strategy 4: Try with image resizing
            logger.info("Trying image resizing...")
            resized_image = self._resize_image(image)
            detections = self._detect_with_threshold(resized_image, 0.1)
            if detections:
                # Scale bounding boxes back to original size
                return self._scale_detections(detections, image.shape, resized_image.shape)
            
            return []
            
        except Exception as e:
            logger.error(f"Error in preprocessing detection: {e}")
            return []
    
    def _adjust_brightness_contrast(self, image: np.ndarray) -> np.ndarray:
        """Adjust brightness and contrast of the image"""
        try:
            # Convert to float for processing
            img_float = image.astype(np.float32) / 255.0
            
            # Adjust brightness and contrast
            alpha = 1.2  # Contrast control
            beta = 0.1   # Brightness control
            
            adjusted = cv2.convertScaleAbs(img_float, alpha=alpha, beta=beta)
            return adjusted
        except Exception as e:
            logger.debug(f"Error adjusting brightness/contrast: {e}")
            return image
    
    def _equalize_histogram(self, image: np.ndarray) -> np.ndarray:
        """Apply histogram equalization to improve contrast"""
        try:
            if len(image.shape) == 3:
                # Convert to LAB color space and equalize L channel
                lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
                lab[:,:,0] = cv2.equalizeHist(lab[:,:,0])
                equalized = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
                return equalized
            else:
                # Grayscale image
                return cv2.equalizeHist(image)
        except Exception as e:
            logger.debug(f"Error in histogram equalization: {e}")
            return image
    
    def _convert_to_hsv(self, image: np.ndarray) -> np.ndarray:
        """Convert image to HSV color space"""
        try:
            if len(image.shape) == 3:
                return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            else:
                return image
        except Exception as e:
            logger.debug(f"Error converting to HSV: {e}")
            return image
    
    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        """Resize image to different dimensions"""
        try:
            # Try different sizes
            target_sizes = [(640, 640), (800, 800), (1024, 1024)]
            
            for target_size in target_sizes:
                try:
                    resized = cv2.resize(image, target_size)
                    return resized
                except Exception:
                    continue
            
            return image
        except Exception as e:
            logger.debug(f"Error resizing image: {e}")
            return image
    
    def _scale_detections(self, detections: List[DetectionResult], 
                         original_shape: tuple, resized_shape: tuple) -> List[DetectionResult]:
        """Scale detections from resized image back to original size"""
        try:
            orig_h, orig_w = original_shape[:2]
            resize_h, resize_w = resized_shape[:2]
            
            scale_x = orig_w / resize_w
            scale_y = orig_h / resize_h
            
            scaled_detections = []
            for detection in detections:
                x1, y1, x2, y2 = detection.bbox
                scaled_detection = DetectionResult(
                    bbox=[x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y],
                    confidence=detection.confidence,
                    class_id=detection.class_id,
                    class_name=detection.class_name
                )
                scaled_detections.append(scaled_detection)
            
            return scaled_detections
            
        except Exception as e:
            logger.error(f"Error scaling detections: {e}")
            return detections
    
    def extract_face_region(self, image: np.ndarray, detection: DetectionResult) -> Optional[np.ndarray]:
        """
        Extract the region from a detected dog for embedding generation
        
        Args:
            image: Input image
            detection: Detection result
            
        Returns:
            Cropped dog region for embedding generation
        """
        try:
            x1, y1, x2, y2 = map(int, detection.bbox)
            
            # Ensure coordinates are within image bounds
            height, width = image.shape[:2]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(width, x2)
            y2 = min(height, y2)
            
            if x1 >= x2 or y1 >= y2:
                return None
            
            # Extract the full dog region
            dog_region = image[y1:y2, x1:x2]
            
            # Check if the region is too small
            if dog_region.shape[0] < 20 or dog_region.shape[1] < 20:
                return None
            
            # Add some padding around the detection for better context
            padding_x = int((x2 - x1) * 0.1)  # 10% padding
            padding_y = int((y2 - y1) * 0.1)
            
            # Apply padding with bounds checking
            pad_x1 = max(0, x1 - padding_x)
            pad_y1 = max(0, y1 - padding_y)
            pad_x2 = min(width, x2 + padding_x)
            pad_y2 = min(height, y2 + padding_y)
            
            # Extract padded region
            padded_region = image[pad_y1:pad_y2, pad_x1:pad_x2]
            
            logger.info(f"Extracted dog region with size: {padded_region.shape} (original: {dog_region.shape})")
            return padded_region
            
        except Exception as e:
            logger.error(f"Error extracting dog region: {e}")
            return None
    
    def preprocess_for_embedding(self, face_region: np.ndarray) -> np.ndarray:
        """
        Preprocess region for embedding generation (legacy method)
        
        Args:
            face_region: Cropped region
            
        Returns:
            Preprocessed image
        """
        try:
            # Resize to standard size for embedding
            target_size = (160, 160)
            resized = cv2.resize(face_region, target_size)
            
            # Convert to RGB if needed
            if len(resized.shape) == 3 and resized.shape[2] == 3:
                # BGR to RGB
                resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            
            return resized
            
        except Exception as e:
            logger.error(f"Error preprocessing region: {e}")
            return face_region

# Global instance
detection_service = DogDetectionService() 