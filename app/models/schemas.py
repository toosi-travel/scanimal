from pydantic import BaseModel, Field
from typing import List, Optional, Union
from datetime import datetime
from enum import Enum

class DogInfo(BaseModel):
    id: str
    name: Optional[str] = None
    breed: Optional[str] = None
    owner: Optional[str] = None
    description: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None

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
    matches: List[SimilarityMatch] = []
    processing_time: float

class RegistrationResponse(BaseModel):
    success: bool
    message: str
    dog_id: str
    embedding_count: int

class DatabaseInfo(BaseModel):
    total_dogs: int
    total_embeddings: int
    last_updated: datetime

class HealthCheck(BaseModel):
    status: str
    timestamp: datetime
    version: str

# New schemas for duplicate detection and approval pipeline

class SimilarityThresholds(BaseModel):
    auto_register_threshold: float = Field(0.95, description="Threshold above which dogs are auto-registered")
    auto_reject_threshold: float = Field(0.85, description="Threshold below which dogs are auto-rejected")
    pending_threshold_min: float = Field(0.85, description="Minimum threshold for pending approval")
    pending_threshold_max: float = Field(0.95, description="Maximum threshold for pending approval")

class PendingDogStatus(str, Enum):
    WAITING_APPROVAL = "waiting_approval"
    APPROVED = "approved"
    REJECTED = "rejected"

class PendingDogInfo(BaseModel):
    id: str
    image_path: str
    embedding_vector: List[float]
    similar_dog_id: Optional[str] = None
    similarity_score: float
    status: PendingDogStatus
    name: Optional[str] = None
    breed: Optional[str] = None
    owner: Optional[str] = None
    description: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None

class DuplicateCheckRequest(BaseModel):
    image: bytes
    name: Optional[str] = None
    breed: Optional[str] = None
    owner: Optional[str] = None
    description: Optional[str] = None

class DuplicateCheckResponse(BaseModel):
    success: bool
    message: str
    action: str  # "auto_register", "pending_approval", "auto_reject"
    similarity_score: Optional[float] = None
    similar_dog_id: Optional[str] = None
    similar_dog_name: Optional[str] = None
    dog_id: Optional[str] = None
    processing_time: float

# New schemas for multiple image processing and best match responses
class ImageMatchResult(BaseModel):
    """Result for a single image with best match information"""
    image_index: int
    success: bool
    message: str
    best_match: Optional['DogMatchInfo'] = None
    error: Optional[str] = None
    processing_time: float

class DogMatchInfo(BaseModel):
    """Information about the best matching dog"""
    dog_id: str
    name: str
    breed: Optional[str] = None
    owner: Optional[str] = None
    similarity_score: float
    image_path: Optional[str] = None

class MultiImageDuplicateCheckResponse(BaseModel):
    """Response for multiple image duplicate check"""
    success: bool
    message: str
    total_images: int
    successful_matches: int
    failed_images: int
    results: List[ImageMatchResult]
    total_processing_time: float

class BestMatchResponse(BaseModel):
    """Response for finding the best match for a single image"""
    success: bool
    message: str
    best_match: Optional[DogMatchInfo] = None
    processing_time: float

class TopMatchesResponse(BaseModel):
    """Response for finding the top matches for a single image"""
    success: bool
    message: str
    top_matches: List[DogMatchInfo] = []
    total_matches_found: int
    processing_time: float

class PendingDogResponse(BaseModel):
    success: bool
    message: str
    pending_dog: Optional[PendingDogInfo] = None

class PendingDogsListResponse(BaseModel):
    success: bool
    message: str
    pending_dogs: List[PendingDogInfo]
    total: int

class ApprovalRequest(BaseModel):
    action: str  # "approve" or "reject"
    admin_notes: Optional[str] = None

class ApprovalResponse(BaseModel):
    success: bool
    message: str
    dog_id: Optional[str] = None

class BatchProcessingRequest(BaseModel):
    image_paths: List[str]
    process_immediately: bool = False

class BatchProcessingResponse(BaseModel):
    success: bool
    message: str
    total_processed: int
    auto_registered: int
    pending_approval: int
    auto_rejected: int
    errors: List[str] = []
    processing_time: float

class ProcessingLog(BaseModel):
    id: str
    image_path: str
    action_taken: str
    similarity_score: Optional[float] = None
    matched_dog_id: Optional[str] = None
    decision_reason: str
    created_at: datetime 