#!/usr/bin/env python3
"""
Startup script for the Dog Face Recognition API
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
    print("API will be available at: http://localhost:8000")
    print("Interactive docs: http://localhost:8000/docs")
    print("Press Ctrl+C to stop the server")
    print("=" * 40)
    
    try:
        uvicorn.run(
            "app.main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 