#!/usr/bin/env python3
"""
Test script to verify the dog face recognition API setup
"""

import sys
import os
import importlib

def test_imports():
    """Test if all required packages can be imported"""
    print("Testing package imports...")
    
    packages = [
        'fastapi',
        'uvicorn',
        'numpy',
        'cv2',
        'ultralytics',
        'faiss',
        'insightface',
        'onnxruntime',
        'PIL',
        'pydantic'
    ]
    
    failed_imports = []
    
    for package in packages:
        try:
            importlib.import_module(package)
            print(f"✓ {package}")
        except ImportError as e:
            print(f"✗ {package}: {e}")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"\nFailed to import: {failed_imports}")
        return False
    
    print("All packages imported successfully!")
    return True

def test_app_imports():
    """Test if the application modules can be imported"""
    print("\nTesting application imports...")
    
    app_modules = [
        'app.core.config',
        'app.models.schemas',
        'app.services.detection_service',
        'app.services.embedding_service',
        'app.services.faiss_service',
        'app.api.routes.dogs',
        'app.main'
    ]
    
    failed_imports = []
    
    for module in app_modules:
        try:
            importlib.import_module(module)
            print(f"✓ {module}")
        except ImportError as e:
            print(f"✗ {module}: {e}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\nFailed to import: {failed_imports}")
        return False
    
    print("All application modules imported successfully!")
    return True

def test_config():
    """Test configuration loading"""
    print("\nTesting configuration...")
    
    try:
        from app.core.config import settings
        print(f"✓ App name: {settings.app_name}")
        print(f"✓ Version: {settings.version}")
        print(f"✓ YOLO model path: {settings.yolo_model_path}")
        print(f"✓ Database path: {settings.database_path}")
        print(f"✓ Embedding dimension: {settings.embedding_dimension} (ArcFace)")
        return True
    except Exception as e:
        print(f"✗ Configuration error: {e}")
        return False

def test_directories():
    """Test if required directories exist or can be created"""
    print("\nTesting directories...")
    
    from app.core.config import settings
    
    directories = [
        settings.upload_dir,
        settings.embeddings_dir
    ]
    
    for directory in directories:
        if os.path.exists(directory):
            print(f"✓ {directory} (exists)")
        else:
            try:
                os.makedirs(directory, exist_ok=True)
                print(f"✓ {directory} (created)")
            except Exception as e:
                print(f"✗ {directory}: {e}")
                return False
    
    return True

def test_arcface_model():
    """Test ArcFace model loading"""
    print("\nTesting ArcFace model...")
    
    try:
        import insightface
        app = insightface.app.FaceAnalysis(
            name='buffalo_s',  # Use smaller model for testing
            providers=['CPUExecutionProvider']
        )
        app.prepare(ctx_id=0, det_size=(640, 640))
        print("✓ ArcFace model loaded successfully")
        return True
    except Exception as e:
        print(f"✗ ArcFace model error: {e}")
        return False

def main():
    """Run all tests"""
    print("Dog Face Recognition API - Setup Test (ArcFace)")
    print("=" * 60)
    
    tests = [
        ("Package Imports", test_imports),
        ("Application Imports", test_app_imports),
        ("Configuration", test_config),
        ("Directories", test_directories),
        ("ArcFace Model", test_arcface_model)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 30)
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 All tests passed! The application is ready to run.")
        print("\nTo start the server, run:")
        print("  python -m app.main")
        print("  # or")
        print("  uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload")
        print("  # or")
        print("  python start.py")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main() 