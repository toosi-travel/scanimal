from fastapi import APIRouter, UploadFile, File, HTTPException, Form, Query
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import uuid
import time
from typing import List, Optional
from datetime import datetime
import io
from PIL import Image

from app.models.schemas import (
    DogInfo, RecognitionResponse, RegistrationResponse, 
    DatabaseInfo, HealthCheck
)
from app.services.detection_service import detection_service
from app.services.embedding_service import embedding_service
from app.services.faiss_service import faiss_service
from app.core.config import settings
from app.core.log_config import logger

router = APIRouter(prefix="/dogs", tags=["dogs"])


def validate_image_file(file: UploadFile) -> bool:
    """Validate uploaded image file"""
    if not file.content_type.startswith('image/'):
        return False
    
    # Check file extension
    file_extension = file.filename.lower().split('.')[-1] if file.filename else ''
    if file_extension not in [ext.replace('.', '') for ext in settings.allowed_extensions]:
        return False
    
    return True

def image_to_numpy(image_file: UploadFile) -> np.ndarray:
    """Convert uploaded image to numpy array"""
    try:
        # Read image data
        image_data = image_file.file.read()
        
        # Convert to numpy array
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Invalid image data")
        
        return image
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

@router.post("/register", response_model=RegistrationResponse)
async def register_dog(
    image: UploadFile = File(...),
    name: str = Form(...),
    breed: Optional[str] = Form(None),
    owner: Optional[str] = Form(None),
    description: Optional[str] = Form(None)
):
    """
    Register a new dog with face recognition
    
    Upload an image of a dog to register it in the database.
    The system will detect the dog's face and generate embeddings for recognition.
    """
    start_time = time.time()
    
    try:
        # Validate image file
        if not validate_image_file(image):
            raise HTTPException(
                status_code=400, 
                detail="Invalid image file. Supported formats: JPG, JPEG, PNG, BMP"
            )
        
        # Convert image to numpy array
        image_array = image_to_numpy(image)
        
        # Detect dogs in the image
        detections = detection_service.detect_dogs(image_array)
        logger.info(f"Detections: {detections}")
        
        if not detections:
            raise HTTPException(
                status_code=400, 
                detail="No dogs detected in the image. Please upload an image with a clear view of a dog."
            )
        
        # Generate embeddings for detected faces
        embedding_results = embedding_service.generate_embeddings_from_detections(
            image_array, detections
        )
        
        if not embedding_results:
            raise HTTPException(
                status_code=400, 
                detail="Could not generate face embeddings. Please ensure the dog's face is clearly visible."
            )
        
        # Create dog info
        dog_id = str(uuid.uuid4())
        dog_info = DogInfo(
            id=dog_id,
            name=name,
            breed=breed,
            owner=owner,
            description=description,
            created_at=datetime.now()
        )
        
        # Extract embeddings
        embeddings = [np.array(result.embedding) for result in embedding_results]
        
        # Add to database
        success = faiss_service.add_dog(dog_info, embeddings)
        
        if not success:
            raise HTTPException(
                status_code=500, 
                detail="Failed to add dog to database"
            )
        
        processing_time = time.time() - start_time
        
        return RegistrationResponse(
            success=True,
            message=f"Dog '{name}' registered successfully with {len(embeddings)} face embeddings",
            dog_id=dog_id,
            embedding_count=len(embeddings)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/recognize", response_model=RecognitionResponse)
async def recognize_dog(
    image: UploadFile = File(...),
    top_k: int = Query(5, ge=1, le=20, description="Number of top matches to return"),
    threshold: float = Query(0.6, ge=0.0, le=1.0, description="Minimum similarity threshold")
):
    """
    Recognize a dog from an uploaded image
    
    Upload an image of a dog to find similar dogs in the database.
    Returns the top matches with similarity scores.
    """
    start_time = time.time()
    
    try:
        # Validate image file
        if not validate_image_file(image):
            raise HTTPException(
                status_code=400, 
                detail="Invalid image file. Supported formats: JPG, JPEG, PNG, BMP"
            )
        
        # Convert image to numpy array
        image_array = image_to_numpy(image)
        
        # Detect dogs in the image
        detections = detection_service.detect_dogs(image_array)
        
        if not detections:
            raise HTTPException(
                status_code=400, 
                detail="No dogs detected in the image. Please upload an image with a clear view of a dog."
            )
        
        # Generate embeddings for detected faces
        embedding_results = embedding_service.generate_embeddings_from_detections(
            image_array, detections
        )
        
        if not embedding_results:
            raise HTTPException(
                status_code=400, 
                detail="Could not generate face embeddings. Please ensure the dog's face is clearly visible."
            )
        
        # Search for similar dogs using the first embedding
        query_embedding = np.array(embedding_results[0].embedding)
        matches = faiss_service.search_similar(query_embedding, k=top_k, threshold=threshold)
        
        processing_time = time.time() - start_time
        
        if not matches:
            return RecognitionResponse(
                success=True,
                message="No similar dogs found in the database",
                matches=[],
                processing_time=processing_time
            )
        
        return RecognitionResponse(
            success=True,
            message=f"Found {len(matches)} similar dogs",
            matches=matches,
            processing_time=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/database/info", response_model=DatabaseInfo)
async def get_database_info():
    """Get information about the dog database"""
    try:
        info = faiss_service.get_database_info()
        return DatabaseInfo(**info)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting database info: {str(e)}")

@router.delete("/{dog_id}")
async def remove_dog(dog_id: str):
    """Remove a dog from the database"""
    try:
        success = faiss_service.remove_dog(dog_id)
        if not success:
            raise HTTPException(status_code=404, detail="Dog not found in database")
        
        return {"success": True, "message": f"Dog {dog_id} removed successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error removing dog: {str(e)}")

@router.get("/list")
async def list_dogs():
    """List all dogs in the database"""
    try:
        dogs = list(faiss_service.dog_database.values())
        return {
            "success": True,
            "dogs": [dog.dict() for dog in dogs],
            "total": len(dogs)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing dogs: {str(e)}")

@router.get("/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint"""
    return HealthCheck(
        status="healthy",
        timestamp=datetime.now(),
        version=settings.version
    ) 