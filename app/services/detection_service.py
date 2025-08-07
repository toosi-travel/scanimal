import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple, Optional
import os
from app.core.config import settings
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
                print(f"Downloading {settings.yolo_model_path}...")
                self.model = YOLO(settings.yolo_model_path)
            else:
                self.model = YOLO(settings.yolo_model_path)
            print("YOLOv8 model loaded successfully")
        except Exception as e:
            print(f"Error loading YOLOv8 model: {e}")
            raise
    
    def detect_dogs(self, image: np.ndarray) -> List[DetectionResult]:
        """
        Detect dogs in the image using YOLOv8
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of DetectionResult objects
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        try:
            # Run inference
            results = self.model(image, conf=settings.confidence_threshold, 
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
            
            return detections
            
        except Exception as e:
            print(f"Error during detection: {e}")
            return []
    
    def extract_face_region(self, image: np.ndarray, detection: DetectionResult) -> Optional[np.ndarray]:
        """
        Extract the face region from a detected dog
        
        Args:
            image: Input image
            detection: Detection result
            
        Returns:
            Cropped face region or None if extraction fails
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
            
            # Extract the region
            face_region = image[y1:y2, x1:x2]
            
            # Check if the region is too small
            if face_region.shape[0] < 20 or face_region.shape[1] < 20:
                return None
            
            return face_region
            
        except Exception as e:
            print(f"Error extracting face region: {e}")
            return None
    
    def preprocess_for_embedding(self, face_region: np.ndarray) -> np.ndarray:
        """
        Preprocess face region for embedding generation (legacy method)
        
        Args:
            face_region: Cropped face region
            
        Returns:
            Preprocessed image
        """
        try:
            # Resize to standard size for face recognition
            target_size = (160, 160)
            resized = cv2.resize(face_region, target_size)
            
            # Convert to RGB if needed
            if len(resized.shape) == 3 and resized.shape[2] == 3:
                # BGR to RGB
                resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            
            return resized
            
        except Exception as e:
            print(f"Error preprocessing face region: {e}")
            return face_region

# Global instance
detection_service = DogDetectionService() 