#!/usr/bin/env python3
"""
Bulk Dog Registration Script

This script reads images from a specified folder and registers them using the register API.
It can handle multiple image formats and provides detailed progress reporting.
"""

import os
import sys
import time
import json
import requests
from pathlib import Path
from typing import List, Dict, Any, Optional
import argparse

class BulkDogRegistrar:
    """Bulk dog registration using the API"""
    
    def __init__(self, api_base_url: str = "http://127.0.0.1:8000"):
        self.api_base_url = api_base_url
        self.register_endpoint = f"{api_base_url}/dogs/register"
        self.health_endpoint = f"{api_base_url}/"
        
        # Supported image formats
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        # Statistics
        self.stats = {
            'total_images': 0,
            'successful_registrations': 0,
            'failed_registrations': 0,
            'skipped_images': 0,
            'errors': []
        }
    
    def check_api_health(self) -> bool:
        """Check if the API is running and healthy"""
        try:
            response = requests.get(self.health_endpoint, timeout=10)
            if response.status_code == 200:
                print("✅ API is running and healthy")
                return True
            else:
                print(f"❌ API returned status code: {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"❌ Cannot connect to API: {e}")
            print(f"   Make sure the server is running at: {self.api_base_url}")
            return False
    
    def get_image_files(self, folder_path: str) -> List[Path]:
        """Get all supported image files from the specified folder"""
        folder = Path(folder_path)
        
        if not folder.exists():
            print(f"❌ Folder does not exist: {folder_path}")
            return []
        
        if not folder.is_dir():
            print(f"❌ Path is not a directory: {folder_path}")
            return []
        
        image_files = []
        for file_path in folder.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in self.supported_formats:
                image_files.append(file_path)
        
        # Sort files for consistent processing order
        image_files.sort()
        
        print(f"📁 Found {len(image_files)} image files in {folder_path}")
        return image_files
    
    def register_dog(self, image_path: Path, name: Optional[str] = None, 
                    breed: Optional[str] = None, owner: Optional[str] = None, 
                    description: Optional[str] = None) -> Dict[str, Any]:
        """Register a single dog using the API"""
        try:
            # Prepare the request
            with open(image_path, 'rb') as image_file:
                files = {'image': (image_path.name, image_file, 'image/jpeg')}
                
                data = {}
                if name:
                    data['name'] = name
                if breed:
                    data['breed'] = breed
                if owner:
                    data['owner'] = owner
                if description:
                    data['description'] = description
                
                # Make the API request
                response = requests.post(
                    self.register_endpoint,
                    files=files,
                    data=data,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if result.get('success'):
                        return {
                            'success': True,
                            'dog_id': result.get('dog_id'),
                            'message': result.get('message'),
                            'embedding_count': result.get('embedding_count', 0)
                        }
                    else:
                        return {
                            'success': False,
                            'error': result.get('message', 'Unknown error')
                        }
                else:
                    return {
                        'success': False,
                        'error': f"HTTP {response.status_code}: {response.text}"
                    }
                    
        except Exception as e:
            return {
                'success': False,
                'error': f"Exception: {str(e)}"
            }
    
    def generate_dog_info(self, image_path: Path, naming_strategy: str = "filename") -> Dict[str, str]:
        """Generate dog information based on the naming strategy"""
        filename = image_path.stem  # filename without extension
        
        if naming_strategy == "filename":
            # Use filename as name, try to extract breed/owner info
            name = filename.replace('_', ' ').replace('-', ' ').title()
            breed = None
            owner = None
        elif naming_strategy == "structured":
            # Try to parse structured filename like "Owner_Breed_Name.jpg"
            parts = filename.split('_')
            if len(parts) >= 3:
                owner = parts[0].title()
                breed = parts[1].title()
                name = ' '.join(parts[2:]).title()
            elif len(parts) == 2:
                owner = parts[0].title()
                name = parts[1].title()
                breed = None
            else:
                name = filename.replace('_', ' ').title()
                owner = None
                breed = None
        else:
            # Default to filename
            name = filename.replace('_', ' ').replace('-', ' ').title()
            breed = None
            owner = None
        
        return {
            'name': name,
            'breed': breed,
            'owner': owner,
            'description': f"Bulk imported from {image_path.name}"
        }
    
    def process_folder(self, folder_path: str, naming_strategy: str = "filename",
                      dry_run: bool = False, delay: float = 1.0) -> Dict[str, Any]:
        """Process all images in the folder for bulk registration"""
        print(f"\n🚀 Starting bulk registration from: {folder_path}")
        print(f"📋 Naming strategy: {naming_strategy}")
        print(f"🔍 Dry run: {'Yes' if dry_run else 'No'}")
        print(f"⏱️  Delay between requests: {delay}s")
        print("=" * 60)
        
        # Get image files
        image_files = self.get_image_files(folder_path)
        if not image_files:
            return self.stats
        
        self.stats['total_images'] = len(image_files)
        
        # Process each image
        for i, image_path in enumerate(image_files, 1):
            print(f"\n📸 Processing image {i}/{len(image_files)}: {image_path.name}")
            
            # Generate dog information
            dog_info = self.generate_dog_info(image_path, naming_strategy)
            print(f"   🐕 Name: {dog_info['name']}")
            if dog_info['breed']:
                print(f"   🐕 Breed: {dog_info['breed']}")
            if dog_info['owner']:
                print(f"   👤 Owner: {dog_info['owner']}")
            
            if dry_run:
                print("   🔍 [DRY RUN] Would register this dog")
                self.stats['skipped_images'] += 1
                continue
            
            # Register the dog
            print("   📤 Registering...")
            result = self.register_dog(
                image_path,
                name=dog_info['name'],
                breed=dog_info['breed'],
                owner=dog_info['owner'],
                description=dog_info['description']
            )
            
            if result['success']:
                print(f"   ✅ Successfully registered!")
                print(f"      🆔 Dog ID: {result['dog_id']}")
                print(f"      🔢 Embeddings: {result['embedding_count']}")
                self.stats['successful_registrations'] += 1
            else:
                print(f"   ❌ Registration failed: {result['error']}")
                self.stats['failed_registrations'] += 1
                self.stats['errors'].append({
                    'file': image_path.name,
                    'error': result['error']
                })
            
            # Add delay between requests to avoid overwhelming the API
            if i < len(image_files) and delay > 0:
                print(f"   ⏳ Waiting {delay}s before next request...")
                time.sleep(delay)
        
        return self.stats
    
    def print_summary(self):
        """Print a summary of the bulk registration results"""
        print("\n" + "=" * 60)
        print("📊 BULK REGISTRATION SUMMARY")
        print("=" * 60)
        print(f"📁 Total images found: {self.stats['total_images']}")
        print(f"✅ Successful registrations: {self.stats['successful_registrations']}")
        print(f"❌ Failed registrations: {self.stats['failed_registrations']}")
        print(f"🔍 Skipped images (dry run): {self.stats['skipped_images']}")
        
        if self.stats['errors']:
            print(f"\n❌ Errors encountered:")
            for error in self.stats['errors']:
                print(f"   📄 {error['file']}: {error['error']}")
        
        success_rate = (self.stats['successful_registrations'] / self.stats['total_images'] * 100) if self.stats['total_images'] > 0 else 0
        print(f"\n🎯 Success rate: {success_rate:.1f}%")
        
        if self.stats['successful_registrations'] > 0:
            print("🎉 Bulk registration completed successfully!")
        else:
            print("⚠️  No dogs were registered. Check the errors above.")

def main():
    """Main function with command line argument parsing"""
    parser = argparse.ArgumentParser(
        description="Bulk register dogs from a folder of images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage - register all images in a folder
  python bulk_register.py /path/to/dog/images
  
  # Use structured naming (Owner_Breed_Name.jpg)
  python bulk_register.py /path/to/images --naming structured
  
  # Dry run to see what would be registered
  python bulk_register.py /path/to/images --dry-run
  
  # Custom API URL
  python bulk_register.py /path/to/images --api-url http://localhost:8000
  
  # Add delay between requests
  python bulk_register.py /path/to/images --delay 2.0
        """
    )
    
    parser.add_argument(
        'folder_path',
        help='Path to folder containing dog images'
    )
    
    parser.add_argument(
        '--naming',
        choices=['filename', 'structured'],
        default='filename',
        help='Naming strategy for dogs (default: filename)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be registered without actually doing it'
    )
    
    parser.add_argument(
        '--api-url',
        default='http://127.0.0.1:8000',
        help='Base URL for the API (default: http://127.0.0.1:8000)'
    )
    
    parser.add_argument(
        '--delay',
        type=float,
        default=1.0,
        help='Delay between API requests in seconds (default: 1.0)'
    )
    
    args = parser.parse_args()
    
    # Create registrar and check API health
    registrar = BulkDogRegistrar(args.api_url)
    
    if not registrar.check_api_health():
        print("\n❌ Cannot proceed without a healthy API connection.")
        sys.exit(1)
    
    try:
        # Process the folder
        stats = registrar.process_folder(
            args.folder_path,
            naming_strategy=args.naming,
            dry_run=args.dry_run,
            delay=args.delay
        )
        
        # Print summary
        registrar.print_summary()
        
        # Exit with appropriate code
        if args.dry_run or stats['successful_registrations'] > 0:
            sys.exit(0)
        else:
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n🛑 Bulk registration interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
