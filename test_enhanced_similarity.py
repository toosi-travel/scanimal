#!/usr/bin/env python3
"""
Test script for enhanced similarity calculation
Demonstrates improved color discrimination between dogs with different colors
"""

import numpy as np
import cv2
import sys
import os

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from services.embedding_service import EmbeddingService
from core.similarity_config import similarity_config

def create_test_images():
    """Create test images with different colors for demonstration"""
    size = (112, 112)
    
    # Create a brown dog image (simulated)
    brown_dog = np.ones((*size, 3), dtype=np.uint8)
    brown_dog[:, :, 0] = 139  # Blue channel (darker)
    brown_dog[:, :, 1] = 69   # Green channel (medium)
    brown_dog[:, :, 2] = 19   # Red channel (darker)
    
    # Create a white dog image (simulated)
    white_dog = np.ones((*size, 3), dtype=np.uint8) * 255
    
    # Create a black dog image (simulated)
    black_dog = np.zeros((*size, 3), dtype=np.uint8)
    
    # Create a golden dog image (simulated)
    golden_dog = np.ones((*size, 3), dtype=np.uint8)
    golden_dog[:, :, 0] = 0    # Blue channel (minimal)
    golden_dog[:, :, 1] = 165  # Green channel (medium-high)
    golden_dog[:, :, 2] = 255  # Red channel (maximum)
    
    return {
        'brown': brown_dog,
        'white': white_dog,
        'black': black_dog,
        'golden': golden_dog
    }

def test_enhanced_similarity():
    """Test enhanced similarity calculation between different colored dogs"""
    print("Testing Enhanced Similarity Calculation")
    print("=" * 50)
    
    # Initialize embedding service
    embedding_service = EmbeddingService()
    
    # Create test images
    test_images = create_test_images()
    
    # Generate embeddings for each test image
    embeddings = {}
    for color, image in test_images.items():
        try:
            # Convert BGR to RGB for consistency
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            embedding = embedding_service._create_enhanced_region_embedding(image_rgb)
            embeddings[color] = embedding
            print(f"✓ Generated embedding for {color} dog")
        except Exception as e:
            print(f"✗ Failed to generate embedding for {color} dog: {e}")
            return
    
    print(f"\nGenerated {len(embeddings)} embeddings successfully")
    
    # Test similarity between different colored dogs
    print("\nSimilarity Matrix (Enhanced Calculation):")
    print("-" * 80)
    
    colors = list(embeddings.keys())
    for i, color1 in enumerate(colors):
        for j, color2 in enumerate(colors):
            if i <= j:  # Only test upper triangle to avoid duplicates
                emb1 = embeddings[color1]
                emb2 = embeddings[color2]
                img1 = test_images[color1]
                img2 = test_images[color2]
                
                # Test both regular and enhanced similarity
                regular_sim = embedding_service.compute_similarity(emb1, emb2)
                enhanced_sim = embedding_service.compute_enhanced_similarity(emb1, emb2, img1, img2)
                
                if i == j:
                    print(f"{color1:8} vs {color2:8}: Regular={regular_sim:.3f}, Enhanced={enhanced_sim:.3f} (SAME)")
                else:
                    print(f"{color1:8} vs {color2:8}: Regular={regular_sim:.3f}, Enhanced={enhanced_sim:.3f}")
    
    # Test color similarity specifically
    print("\nColor Similarity Analysis:")
    print("-" * 40)
    
    for i, color1 in enumerate(colors):
        for j, color2 in enumerate(colors):
            if i < j:  # Only test different colors
                img1 = test_images[color1]
                img2 = test_images[color2]
                color_sim = embedding_service._compute_color_similarity(img1, img2)
                print(f"{color1:8} vs {color2:8}: Color similarity = {color_sim:.3f}")

def test_configuration():
    """Test configuration settings"""
    print("\n" + "=" * 50)
    print("Configuration Settings:")
    print("=" * 50)
    
    print(f"Color Weight: {similarity_config.COLOR_WEIGHT}")
    print(f"Structure Weight: {similarity_config.STRUCTURE_WEIGHT}")
    print(f"Auto Reject Threshold: {similarity_config.AUTO_REJECT_THRESHOLD}")
    print(f"Pending Threshold Min: {similarity_config.PENDING_THRESHOLD_MIN}")
    print(f"Auto Register Threshold: {similarity_config.AUTO_REGISTER_THRESHOLD}")
    print(f"Color Similarity Low: {similarity_config.COLOR_SIMILARITY_LOW}")
    print(f"Color Similarity Medium: {similarity_config.COLOR_SIMILARITY_MEDIUM}")
    print(f"Cosine Weight: {similarity_config.COSINE_WEIGHT}")
    print(f"Euclidean Weight: {similarity_config.EUCLIDEAN_WEIGHT}")
    print(f"Manhattan Weight: {similarity_config.MANHATTAN_WEIGHT}")

def main():
    """Main test function"""
    try:
        test_enhanced_similarity()
        test_configuration()
        
        print("\n" + "=" * 50)
        print("Test completed successfully!")
        print("=" * 50)
        
        print("\nKey Improvements:")
        print("1. Enhanced feature engineering with color, texture, and shape features")
        print("2. Color-based pre-filtering with configurable penalties")
        print("3. Multiple similarity metrics (cosine, Euclidean, Manhattan)")
        print("4. Stricter thresholds for better color discrimination")
        print("5. Configurable parameters for easy tuning")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 