#!/usr/bin/env python3
"""
Debug script for dog detection and embedding generation
"""

import sys
import os
import cv2
import numpy as np

# Add the project root to the Python path
sys.path.append(os.path.dirname(__file__))

def debug_dog_detection(image_path: str):
    """Debug dog detection and embedding generation on a specific image"""
    print(f"🔍 Debugging dog detection and embedding for: {image_path}")
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
            print(f"   Found {len(detections)} dog detections")
            
            for i, detection in enumerate(detections):
                print(f"   Detection {i+1}: {detection.bbox}, confidence: {detection.confidence:.3f}")
                
                # Test region extraction
                dog_region = detection_service.extract_face_region(image_rgb, detection)
                if dog_region is not None:
                    print(f"   ✅ Dog region extracted: {dog_region.shape}")
                    
                    # Save dog region for inspection
                    region_filename = f"debug_dog_region_{i+1}.jpg"
                    cv2.imwrite(region_filename, cv2.cvtColor(dog_region, cv2.COLOR_RGB2BGR))
                    print(f"   💾 Dog region saved as: {region_filename}")
                else:
                    print(f"   ❌ Failed to extract dog region")
                    
        except Exception as e:
            print(f"❌ Detection service error: {e}")
        
        # Test embedding service
        try:
            from app.services.embedding_service import embedding_service
            print("\n2️⃣ Testing embedding generation...")
            
            if detections:
                embeddings = embedding_service.generate_embeddings_from_detections(image_rgb, detections)
                print(f"   Generated {len(embeddings)} embeddings")
                
                for i, emb_result in enumerate(embeddings):
                    print(f"   Embedding {i+1}: shape {len(emb_result.embedding)}")
                    
                    # Test if it's a face embedding or region embedding
                    if len(emb_result.embedding) == 512:
                        # Check if it's a random embedding (fallback)
                        embedding_array = np.array(emb_result.embedding)
                        if np.allclose(embedding_array, embedding_array[0], rtol=1e-5):
                            print(f"     ⚠️  This appears to be a fallback embedding")
                        else:
                            print(f"     ✅ Valid embedding generated")
                    else:
                        print(f"     ❌ Unexpected embedding shape")
            else:
                print("   No detections to process")
                
        except Exception as e:
            print(f"❌ Embedding service error: {e}")
        
        print("\n" + "=" * 60)
        print("🔍 Debug complete! Check the generated dog region images.")
        print("📝 Note: Now using entire dog regions instead of just faces for embeddings.")
        
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main function"""
    if len(sys.argv) != 2:
        print("Usage: python debug_face_detection.py <image_path>")
        print("Example: python debug_face_detection.py uploads/test_dog.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    debug_dog_detection(image_path)

if __name__ == "__main__":
    main() 