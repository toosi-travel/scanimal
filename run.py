#!/usr/bin/env python3
"""
Simple run script for the Dog Face Recognition API
"""

import uvicorn
import sys
import os

def main():
    """Start the FastAPI application"""
    
    # Add the current directory to Python path
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    print("🐕 Dog Face Recognition API")
    print("=" * 40)
    print("Starting server...")
    print("API will be available at: http://127.0.0.1:8000")
    print("Interactive docs: http://127.0.0.1:8000/docs")
    print("Press Ctrl+C to stop the server")
    print("=" * 40)
    
    try:
        uvicorn.run(
            "app.main:app",
            host="127.0.0.1",  # Use 127.0.0.1 instead of 0.0.0.0
            port=8000,
            reload=False,  # Disable reload to avoid socket issues
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 