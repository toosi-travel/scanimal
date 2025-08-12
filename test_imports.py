#!/usr/bin/env python3
"""
Test script to verify imports are working correctly
"""

import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(__file__))

def test_schema_imports():
    """Test importing schemas"""
    try:
        from app.models.schemas import (
            DuplicateCheckResponse, PendingDogsListResponse, ApprovalRequest, 
            ApprovalResponse, SimilarityThresholds, ProcessingLog,
            MultiImageDuplicateCheckResponse, ImageMatchResult, DogMatchInfo, BestMatchResponse
        )
        print("✅ All schemas imported successfully")
        return True
    except Exception as e:
        print(f"❌ Schema import failed: {e}")
        return False

def test_service_imports():
    """Test importing services (without running them)"""
    try:
        # Test if we can import the service class definition
        import app.services.duplicate_detection_service
        print("✅ Duplicate detection service module imported successfully")
        return True
    except Exception as e:
        print(f"❌ Service import failed: {e}")
        return False

def test_route_imports():
    """Test importing routes"""
    try:
        import app.api.routes.duplicate_detection
        print("✅ Duplicate detection routes imported successfully")
        return True
    except Exception as e:
        print(f"❌ Route import failed: {e}")
        return False

def main():
    """Test all imports"""
    print("🔍 Testing imports...")
    print("=" * 40)
    
    results = []
    
    # Test schema imports
    results.append(("Schemas", test_schema_imports()))
    
    # Test service imports
    results.append(("Services", test_service_imports()))
    
    # Test route imports
    results.append(("Routes", test_route_imports()))
    
    # Summary
    print("\n" + "=" * 40)
    print("📊 IMPORT TEST RESULTS")
    print("=" * 40)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All imports are working correctly!")
        return True
    else:
        print("⚠️  Some imports failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 