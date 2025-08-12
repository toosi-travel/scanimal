#!/usr/bin/env python3
"""
Test script for the new multi-image duplicate detection endpoint
"""

import requests
import json
import os
import sys

def test_multi_image_endpoint():
    """Test the new multi-image duplicate detection endpoint"""
    
    # API endpoint
    url = "http://127.0.0.1:8000/duplicate-detection/check"
    
    # Test with multiple images
    image_files = []
    
    # Look for test images in the uploads directory
    uploads_dir = "uploads"
    if os.path.exists(uploads_dir):
        for file in os.listdir(uploads_dir):
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_files.append(os.path.join(uploads_dir, file))
    
    if not image_files:
        print("❌ No test images found in uploads directory")
        print("Please add some dog images to the uploads/ folder")
        return
    
    print(f"🔍 Found {len(image_files)} test images: {[os.path.basename(f) for f in image_files]}")
    
    # Prepare the request
    files = []
    for i, image_path in enumerate(image_files):
        files.append(('images', (os.path.basename(image_path), open(image_path, 'rb'), 'image/jpeg')))
    
    try:
        print(f"\n🚀 Testing multi-image endpoint with {len(image_files)} images...")
        print(f"URL: {url}")
        print("📤 Sending images only (no additional data)")
        
        # Make the request
        response = requests.post(url, files=files)
        
        print(f"\n📊 Response Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Success! Response:")
            print(json.dumps(result, indent=2))
            
            # Analyze results
            print(f"\n📈 Summary:")
            print(f"   Total images processed: {result['total_images']}")
            print(f"   Successful matches: {result['successful_matches']}")
            print(f"   Failed images: {result['failed_images']}")
            print(f"   Total processing time: {result['total_processing_time']:.3f}s")
            
            # Show individual results
            print(f"\n🔍 Individual Results:")
            for i, img_result in enumerate(result['results']):
                print(f"   Image {i+1}: {os.path.basename(image_files[i])}")
                if img_result['success']:
                    match = img_result['best_match']
                    print(f"     ✅ Match: {match['name']} (Score: {match['similarity_score']:.3f})")
                    print(f"     📍 Dog ID: {match['dog_id']}")
                    if match.get('breed'):
                        print(f"     🐕 Breed: {match['breed']}")
                    if match.get('owner'):
                        print(f"     👤 Owner: {match['owner']}")
                else:
                    print(f"     ❌ Failed: {img_result['error']}")
                print(f"     ⏱️  Processing time: {img_result['processing_time']:.3f}s")
                
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Error testing endpoint: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Close all open files
        for _, file_tuple in files:
            file_tuple[1].close()

def test_single_image_endpoint():
    """Test the legacy single image endpoint for comparison"""
    
    url = "http://127.0.0.1:8000/duplicate-detection/check-single"
    
    # Look for a test image
    uploads_dir = "uploads"
    if os.path.exists(uploads_dir):
        for file in os.listdir(uploads_dir):
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_path = os.path.join(uploads_dir, file)
                break
        else:
            print("❌ No test images found")
            return
    else:
        print("❌ Uploads directory not found")
        return
    
    print(f"\n🔍 Testing single image endpoint with: {os.path.basename(image_path)}")
    
    try:
        with open(image_path, 'rb') as f:
            files = {'image': (os.path.basename(image_path), f, 'image/jpeg')}
            
            response = requests.post(url, files=files)
            
            print(f"Status: {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                print("✅ Single image response:")
                print(json.dumps(result, indent=2))
                
                # Analyze the new response format
                if result['success'] and result.get('best_match'):
                    match = result['best_match']
                    print(f"\n📊 Single Image Result:")
                    print(f"   ✅ Match: {match['name']} (Score: {match['similarity_score']:.3f})")
                    print(f"   📍 Dog ID: {match['dog_id']}")
                    if match.get('breed'):
                        print(f"   🐕 Breed: {match['breed']}")
                    if match.get('owner'):
                        print(f"   👤 Owner: {match['owner']}")
                    print(f"   ⏱️  Processing time: {result['processing_time']:.3f}s")
                else:
                    print(f"   ❌ No match found: {result['message']}")
            else:
                print(f"❌ Error: {response.text}")
                
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    """Main function"""
    print("🐕 Multi-Image Duplicate Detection Test")
    print("=" * 50)
    
    # Test the new multi-image endpoint
    test_multi_image_endpoint()
    
    print("\n" + "=" * 50)
    
    # Test the legacy single image endpoint
    test_single_image_endpoint()
    
    print("\n" + "=" * 50)
    print("🎯 Test complete!")

if __name__ == "__main__":
    main() 