#!/usr/bin/env python3
"""
Debug script for dog detection issues
"""

import sys
import os
import cv2
import numpy as np

# Add the project root to the Python path
sys.path.append(os.path.dirname(__file__))

def debug_dog_detection(image_path: str):
    """Debug dog detection on a specific image"""
    print(f"🔍 Debugging dog detection for: {image_path}")
    print("=" * 60)
    
    try:
        # Load image
        if not os.path.exists(image_path):
            print(f"❌ Image file not found: {image_path}")
            return
        
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Failed to load image: {image_path}")
            return
        
        print(f"✅ Image loaded successfully: {image.shape}")
        
        # Convert BGR to RGB for processing
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Test detection service
        try:
            from app.services.detection_service import detection_service
            print("\n1️⃣ Testing YOLO detection...")
            
            detections = detection_service.detect_dogs(image_rgb)
            
            if detections:
                print(f"   ✅ Found {len(detections)} dogs")
                for i, detection in enumerate(detections):
                    print(f"      Dog {i+1}: confidence={detection.confidence:.3f}")
            else:
                print(f"   ❌ No dogs detected")
                
        except Exception as e:
            print(f"❌ Detection service error: {e}")
        
        print("\n" + "=" * 60)
        print("🔍 Debug complete!")
        
    except Exception as e:
        print(f"❌ Debug failed: {e}")

def main():
    """Main function"""
    if len(sys.argv) != 2:
        print("Usage: python debug_detection.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    debug_dog_detection(image_path)

if __name__ == "__main__":
    main() 