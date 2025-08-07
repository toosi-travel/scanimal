from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from contextlib import asynccontextmanager

from app.core.config import settings
from app.api.routes import dogs

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print(f"Starting {settings.app_name} v{settings.version}")
    print("Loading models and initializing services...")
    
    # Initialize services (they will load models on import)
    try:
        from app.services.detection_service import detection_service
        from app.services.embedding_service import embedding_service
        from app.services.faiss_service import faiss_service
        print("All services initialized successfully")
    except Exception as e:
        print(f"Error initializing services: {e}")
        raise
    
    yield
    
    # Shutdown
    print("Shutting down application...")

# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description="A FastAPI-based dog face recognition system using YOLOv8 and FAISS",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(dogs.router)

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"}
    )

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": f"Welcome to {settings.app_name}",
        "version": settings.version,
        "docs": "/docs",
        "redoc": "/redoc"
    }

@app.get("/info")
async def info():
    """Get API information"""
    return {
        "app_name": settings.app_name,
        "version": settings.version,
        "debug": settings.debug,
        "endpoints": {
            "register_dog": "/dogs/register",
            "recognize_dog": "/dogs/recognize",
            "database_info": "/dogs/database/info",
            "list_dogs": "/dogs/list",
            "remove_dog": "/dogs/{dog_id}",
            "health_check": "/dogs/health"
        }
    }

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level="info"
    )
