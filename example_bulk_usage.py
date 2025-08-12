#!/usr/bin/env python3
"""
Example usage of the bulk registration script

This script demonstrates different ways to use the bulk registration functionality.
"""

import os
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.append(os.path.dirname(__file__))

from bulk_register import BulkDogRegistrar

def example_basic_usage():
    """Example of basic bulk registration"""
    print("🔍 Example 1: Basic Bulk Registration")
    print("=" * 50)
    
    # Create registrar
    registrar = BulkDogRegistrar()
    
    # Check API health
    if not registrar.check_api_health():
        print("❌ API not available, skipping example")
        return
    
    # Example folder path (you would change this to your actual folder)
    example_folder = "example_dog_images"
    
    if not os.path.exists(example_folder):
        print(f"📁 Creating example folder: {example_folder}")
        os.makedirs(example_folder, exist_ok=True)
        print("   (This is just an example - you would use your actual image folder)")
    
    print(f"📁 Example folder: {example_folder}")
    print("🚀 To run: python bulk_register.py example_dog_images")
    print()

def example_structured_naming():
    """Example of structured naming strategy"""
    print("🔍 Example 2: Structured Naming Strategy")
    print("=" * 50)
    
    print("📋 This strategy works with filenames like:")
    print("   - John_GoldenRetriever_Buddy.jpg")
    print("   - Sarah_Labrador_Max.jpg")
    print("   - Mike_Poodle_Fluffy.jpg")
    print()
    print("🚀 To run: python bulk_register.py your_folder --naming structured")
    print()

def example_dry_run():
    """Example of dry run mode"""
    print("🔍 Example 3: Dry Run Mode")
    print("=" * 50)
    
    print("🔍 Dry run shows what would be registered without actually doing it:")
    print("   - Perfect for testing and verification")
    print("   - No dogs are actually registered")
    print("   - Shows all the information that would be used")
    print()
    print("🚀 To run: python bulk_register.py your_folder --dry-run")
    print()

def example_custom_api():
    """Example of custom API URL"""
    print("🔍 Example 4: Custom API URL")
    print("=" * 50)
    
    print("🌐 If your API is running on a different URL:")
    print("   - Local development: http://127.0.0.1:8000")
    print("   - Docker: http://localhost:8000")
    print("   - Remote server: https://your-server.com")
    print()
    print("🚀 To run: python bulk_register.py your_folder --api-url http://localhost:8000")
    print()

def example_with_delay():
    """Example of adding delay between requests"""
    print("🔍 Example 5: Adding Delay Between Requests")
    print("=" * 50)
    
    print("⏱️  Add delay to avoid overwhelming the API:")
    print("   - Default: 1 second between requests")
    print("   - Custom: Set your own delay")
    print("   - Useful for large batches or slower servers")
    print()
    print("🚀 To run: python bulk_register.py your_folder --delay 2.0")
    print()

def create_sample_images():
    """Create some sample image files for demonstration"""
    print("🔍 Creating Sample Images for Demonstration")
    print("=" * 50)
    
    sample_folder = "sample_dog_images"
    
    if not os.path.exists(sample_folder):
        os.makedirs(sample_folder, exist_ok=True)
        print(f"📁 Created sample folder: {sample_folder}")
    
    # Create sample filenames with different naming strategies
    sample_files = [
        "buddy.jpg",
        "max.jpg", 
        "fluffy.jpg",
        "John_GoldenRetriever_Buddy.jpg",
        "Sarah_Labrador_Max.jpg",
        "Mike_Poodle_Fluffy.jpg"
    ]
    
    for filename in sample_files:
        file_path = os.path.join(sample_folder, filename)
        if not os.path.exists(file_path):
            # Create a simple text file as placeholder (in real usage, these would be actual images)
            with open(file_path, 'w') as f:
                f.write(f"Sample image file: {filename}\nThis is a placeholder for demonstration purposes.")
            print(f"   📄 Created: {filename}")
    
    print(f"\n📁 Sample folder ready: {sample_folder}")
    print("🚀 You can now test with:")
    print(f"   python bulk_register.py {sample_folder}")
    print(f"   python bulk_register.py {sample_folder} --naming structured")
    print(f"   python bulk_register.py {sample_folder} --dry-run")
    print()

def main():
    """Main function to run all examples"""
    print("🐕 Bulk Dog Registration Examples")
    print("=" * 60)
    
    # Run examples
    example_basic_usage()
    example_structured_naming()
    example_dry_run()
    example_custom_api()
    example_with_delay()
    
    print("=" * 60)
    print("🎯 Quick Start Guide:")
    print("1. Put your dog images in a folder")
    print("2. Run: python bulk_register.py /path/to/your/images")
    print("3. Check the results and adjust naming strategy if needed")
    print("4. Use --dry-run first to verify everything looks correct")
    print()
    
    # Ask if user wants to create sample images
    try:
        response = input("Would you like to create sample images for testing? (y/n): ").lower().strip()
        if response in ['y', 'yes']:
            create_sample_images()
        else:
            print("📝 No sample images created. You can run the script with your own image folder.")
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    
    print("\n🎉 Examples completed! Check the README for more details.")

if __name__ == "__main__":
    main() 