from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from contextlib import asynccontextmanager

from app.core.config import settings
from app.core.log_config import logger
from app.api.routes import dogs, duplicate_detection

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info(f"Starting {settings.app_name} v{settings.version}")
    logger.info("Loading models and initializing services...")
    
    # Initialize services (they will load models on import)
    try:
        from app.services.detection_service import detection_service
        from app.services.embedding_service import embedding_service
        logger.info("All services initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing services: {e}")
        raise
    
    # Initialize database
    try:
        from app.core.database import init_db
        await init_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down application...")
    
    # Close database connections
    try:
        from app.core.database import close_db
        await close_db()
        logger.info("Database connections closed")
    except Exception as e:
        logger.error(f"Error closing database connections: {e}")

# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description="A FastAPI-based dog face recognition system using YOLOv8 and PostgreSQL",
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
app.include_router(dogs.router, prefix="/dogs", tags=["dogs"])
app.include_router(duplicate_detection.router, prefix="/duplicate-detection", tags=["duplicate-detection"])

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Global exception handler caught: {exc}")
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
            "health_check": "/dogs/health",
            "duplicate_detection": {
                "check": "/duplicate-detection/check",
                "pending": "/duplicate-detection/pending",
                "approve": "/duplicate-detection/pending/{id}/approve",
                "reject": "/duplicate-detection/pending/{id}/reject",
                "thresholds": "/duplicate-detection/thresholds",
                "stats": "/duplicate-detection/stats",
                "logs": "/duplicate-detection/logs"
            }
        }
    }

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=8000,
        reload=False,  # Disable reload to avoid socket issues
        log_level="info"
    )
