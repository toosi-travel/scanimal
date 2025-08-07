from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime

class DogInfo(BaseModel):
    id: str
    name: str
    breed: Optional[str] = None
    owner: Optional[str] = None
    description: Optional[str] = None
    created_at: datetime

class DetectionResult(BaseModel):
    bbox: List[float]  # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    class_name: str

class EmbeddingResult(BaseModel):
    embedding: List[float]
    detection_results: List[DetectionResult]

class SimilarityMatch(BaseModel):
    dog_id: str
    dog_info: DogInfo
    similarity_score: float
    distance: float

class RecognitionResponse(BaseModel):
    success: bool
    message: str
    matches: List[SimilarityMatch]
    processing_time: float

class RegistrationResponse(BaseModel):
    success: bool
    message: str
    dog_id: str
    embedding_count: int

class DatabaseInfo(BaseModel):
    total_dogs: int
    total_embeddings: int
    database_size_mb: float

class HealthCheck(BaseModel):
    status: str
    timestamp: datetime
    version: str 