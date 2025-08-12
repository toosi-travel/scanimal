from fastapi import APIRouter, UploadFile, File, HTTPException, Form, Query, Depends
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import uuid
import time
import os
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
from app.core.config import settings
from app.core.log_config import logger
from app.core.database import get_db
from app.repositories.dog_repository import DogRepository
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter()

def validate_image_file(file: UploadFile) -> bool:
    """Validate uploaded image file"""
    if not file.content_type.startswith('image/'):
        return False
    
    # Check file extension
    file_extension = file.filename.lower().split('.')[-1] if file.filename else ''
    if file_extension not in [ext.replace('.', '') for ext in settings.allowed_extensions]:
        return False
    
    return True

def image_to_numpy(image_data: bytes) -> np.ndarray:
    """Convert image bytes to numpy array"""
    try:
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Invalid image data")
        
        return image
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

def save_image_simple(image_data: bytes, filename: str) -> str:
    """Simple image saving to uploads folder"""
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)
    
    filepath = os.path.join(upload_dir, filename)
    with open(filepath, 'wb') as f:
        f.write(image_data)
    
    return filepath

@router.post("/register", response_model=RegistrationResponse)
async def register_dog(
    image: UploadFile = File(...),
    name: Optional[str] = Form(None),
    breed: Optional[str] = Form(None),
    owner: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db)
):
    """
    Register a new dog with face recognition using PostgreSQL
    
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
        
        # Read image data ONCE
        image_data = await image.read()
        
        # Convert image to numpy array
        image_array = image_to_numpy(image_data)
        
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
        
        # Save image to uploads folder
        filename = f"{uuid.uuid4()}.jpg"
        image_path = save_image_simple(image_data, filename)
        
        # Create dog info dictionary
        dog_id = str(uuid.uuid4())
        # Get the first embedding from the results
        embedding = embedding_results[0].embedding
        dog_info = {
            'id': uuid.UUID(dog_id),  # Convert string to UUID object
            'name': name,
            'breed': breed,
            'owner': owner,
            'description': description,
            'image_path': image_path,
            'embedding_vector': embedding,  # Include embedding in the dictionary
            'created_at': datetime.utcnow(),
            'updated_at': datetime.utcnow()
        }
        
        # Add to PostgreSQL database
        dog_repo = DogRepository(db)
        dog = await dog_repo.create_dog(dog_info)
        
        processing_time = time.time() - start_time
        
        return RegistrationResponse(
            success=True,
            message=f"Dog '{name}' registered successfully in PostgreSQL with 1 face embedding",
            dog_id=dog_id,
            embedding_count=1
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error registering dog: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/recognize", response_model=RecognitionResponse)
async def recognize_dog(
    image: UploadFile = File(...),
    top_k: int = Query(5, ge=1, le=20, description="Number of top matches to return"),
    threshold: float = Query(0.6, ge=0.0, le=1.0, description="Minimum similarity threshold"),
    db: AsyncSession = Depends(get_db)
):
    """
    Recognize a dog from an uploaded image using PostgreSQL
    
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
        
        # Read image data ONCE
        image_data = await image.read()
        
        # Convert image to numpy array
        image_array = image_to_numpy(image_data)
        
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
        
        # Search for similar dogs using PostgreSQL
        query_embedding = np.array(embedding_results[0].embedding)
        dog_repo = DogRepository(db)
        
        # Pass the query image for enhanced similarity calculation with color comparison
        matches = await dog_repo.search_similar_dogs(
            query_embedding, 
            threshold=threshold,
            query_image=image_array  # Pass image for color-based similarity
        )
        
        processing_time = time.time() - start_time
        
        if not matches:
            return RecognitionResponse(
                success=True,
                message="No similar dogs found in the database",
                matches=[],
                processing_time=processing_time
            )
        
        # Convert matches to the expected format
        from app.models.schemas import SimilarityMatch
        similarity_matches = []
        
        for match in matches[:top_k]:
            similarity_match = SimilarityMatch(
                dog_id=match['dog_id'],
                dog_info=DogInfo(**match['dog_info']),
                similarity_score=match['similarity_score'],
                distance=match['distance']
            )
            similarity_matches.append(similarity_match)
        
        return RecognitionResponse(
            success=True,
            message=f"Found {len(similarity_matches)} similar dogs",
            matches=similarity_matches,
            processing_time=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error recognizing dog: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/list", response_model=List[DogInfo])
async def list_dogs(db: AsyncSession = Depends(get_db)):
    """List all dogs in the database"""
    try:
        dog_repo = DogRepository(db)
        dogs = await dog_repo.get_all_dogs()
        
        # Convert to DogInfo format
        dog_list = []
        for dog in dogs:
            dog_info = DogInfo(
                id=str(dog.id),
                name=dog.name,
                breed=dog.breed,
                owner=dog.owner,
                description=dog.description,
                created_at=dog.created_at,
                updated_at=dog.updated_at
            )
            dog_list.append(dog_info)
        
        return dog_list
        
    except Exception as e:
        logger.error(f"Error listing dogs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/database/info", response_model=DatabaseInfo)
async def get_database_info(db: AsyncSession = Depends(get_db)):
    """Get database information and statistics"""
    try:
        dog_repo = DogRepository(db)
        dogs = await dog_repo.get_all_dogs()
        
        return DatabaseInfo(
            total_dogs=len(dogs),
            total_embeddings=len(dogs),  # Simple: 1 embedding per dog
            database_size_mb=0.0,  # Not implemented for simplicity
            embedding_dimension=512,  # ArcFace dimension
            last_updated=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Error getting database info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint"""
    return HealthCheck(
        status="healthy",
        timestamp=datetime.now(),
        version="1.0.0"
    ) 