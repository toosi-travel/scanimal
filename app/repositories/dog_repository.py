from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete
from sqlalchemy.orm import selectinload
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid
import json
import numpy as np
import os

from app.models.database_models import Dog, PendingDog, ProcessingLog
from app.models.schemas import DogInfo, PendingDogInfo, ProcessingLog as ProcessingLogSchema
from app.services.embedding_service import EmbeddingService
from app.core.log_config import logger

class DogRepository:
    """Repository for dog-related database operations"""
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self.embedding_service = EmbeddingService()
    
    async def create_dog(self, dog_data: Dict[str, Any]) -> Dog:
        """Create a new dog"""
        try:
            dog = Dog(**dog_data)
            self.db.add(dog)
            await self.db.commit()
            await self.db.refresh(dog)
            logger.info(f"Created dog with ID: {dog.id}")
            return dog
        except Exception as e:
            await self.db.rollback()
            raise e
    
    async def get_dog_by_id(self, dog_id: str) -> Optional[Dog]:
        """Get a dog by ID"""
        try:
            result = await self.db.execute(
                select(Dog).where(Dog.id == uuid.UUID(dog_id))
            )
            return result.scalar_one_or_none()
        except Exception as e:
            raise e
    
    async def get_dog_by_image_path(self, image_path: str) -> Optional[Dog]:
        """Get a dog by image path"""
        try:
            result = await self.db.execute(
                select(Dog).where(Dog.image_path == image_path)
            )
            return result.scalar_one_or_none()
        except Exception as e:
            raise e

    async def get_all_dogs(self) -> List[Dog]:
        """Get all dogs"""
        try:
            result = await self.db.execute(select(Dog))
            return result.scalars().all()
        except Exception as e:
            raise e
    
    async def search_similar_dogs(self, query_embedding: np.ndarray, threshold: float = 0.6,
                                 query_image: Optional[np.ndarray] = None) -> List[Dict[str, Any]]:
        """Search for similar dogs using enhanced similarity calculation"""
        try:
            # Get all dogs
            dogs = await self.get_all_dogs()
            matches = []
            
            for dog in dogs:
                if dog.embedding_vector:
                    # Convert stored embedding back to numpy array
                    stored_embedding = np.array(dog.embedding_vector, dtype=np.float32)
                    
                    # Use enhanced similarity calculation if query image is provided
                    if query_image is not None:
                        # Try to get the stored dog's image for color comparison
                        stored_image = self._load_dog_image(dog.image_path)
                        if stored_image is not None:
                            similarity = self.embedding_service.compute_enhanced_similarity(
                                query_embedding, stored_embedding, query_image, stored_image
                            )
                        else:
                            # Fallback to regular similarity if image not available
                            similarity = self.embedding_service.compute_similarity(
                                query_embedding, stored_embedding
                            )
                    else:
                        # Use regular similarity calculation
                        similarity = self.embedding_service.compute_similarity(
                            query_embedding, stored_embedding
                        )
                    
                    if similarity >= threshold:
                        matches.append({
                            'dog_id': str(dog.id),
                            'dog_info': {
                                'id': str(dog.id),
                                'name': dog.name,
                                'breed': dog.breed,
                                'owner': dog.owner,
                                'description': dog.description,
                                'created_at': dog.created_at.isoformat() if dog.created_at else None
                            },
                            'similarity_score': float(similarity),
                            'distance': 1.0 - float(similarity)
                        })
            
            # Sort by similarity score
            matches.sort(key=lambda x: x['similarity_score'], reverse=True)
            return matches
            
        except Exception as e:
            raise e
    
    def _load_dog_image(self, image_path: str) -> Optional[np.ndarray]:
        """Load dog image from path for color comparison"""
        try:
            import cv2
            if image_path and os.path.exists(image_path):
                # Load image and convert to RGB
                image = cv2.imread(image_path)
                if image is not None:
                    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return None
        except Exception as e:
            logger.debug(f"Error loading dog image {image_path}: {e}")
            return None
    
    def _calculate_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings (legacy method)"""
        try:
            # Normalize embeddings
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            # Calculate cosine similarity
            similarity = np.dot(emb1, emb2) / (norm1 * norm2)
            return float(similarity)
            
        except Exception:
            return 0.0
    
    async def delete_dog(self, dog_id: str) -> bool:
        """Delete a dog by ID"""
        try:
            result = await self.db.execute(
                delete(Dog).where(Dog.id == uuid.UUID(dog_id))
            )
            await self.db.commit()
            return result.rowcount > 0
        except Exception as e:
            await self.db.rollback()
            raise e

class PendingDogRepository:
    """Repository for pending dog operations"""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def create_pending_dog(self, pending_dog_data: dict) -> PendingDog:
        """Create a new pending dog entry"""
        try:
            pending_dog = PendingDog(
                id=uuid.uuid4(),
                image_path=pending_dog_data['image_path'],
                embedding_vector=pending_dog_data['embedding_vector'],
                similar_dog_id=uuid.UUID(pending_dog_data['similar_dog_id']) if pending_dog_data.get('similar_dog_id') else None,
                similarity_score=pending_dog_data['similarity_score'],
                status='waiting_approval',
                name=pending_dog_data.get('name'),
                breed=pending_dog_data.get('breed'),
                owner=pending_dog_data.get('owner'),
                description=pending_dog_data.get('description'),
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            
            self.db.add(pending_dog)
            await self.db.commit()
            await self.db.refresh(pending_dog)
            
            return pending_dog
            
        except Exception as e:
            await self.db.rollback()
            raise e
    
    async def get_pending_dogs(self) -> List[PendingDog]:
        """Get all pending dogs"""
        try:
            result = await self.db.execute(
                select(PendingDog).where(PendingDog.status == 'waiting_approval')
            )
            return result.scalars().all()
        except Exception as e:
            raise e
    
    async def get_pending_dog_by_id(self, pending_dog_id: str) -> Optional[PendingDog]:
        """Get pending dog by ID"""
        try:
            result = await self.db.execute(
                select(PendingDog).where(PendingDog.id == uuid.UUID(pending_dog_id))
            )
            return result.scalar_one_or_none()
        except Exception as e:
            raise e
    
    async def update_pending_dog_status(self, pending_dog_id: str, status: str, admin_notes: str = None) -> bool:
        """Update pending dog status"""
        try:
            result = await self.db.execute(
                update(PendingDog)
                .where(PendingDog.id == uuid.UUID(pending_dog_id))
                .values(
                    status=status,
                    admin_notes=admin_notes,
                    updated_at=datetime.utcnow()
                )
            )
            await self.db.commit()
            return result.rowcount > 0
        except Exception as e:
            await self.db.rollback()
            raise e

class ProcessingLogRepository:
    """Repository for processing log operations"""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def create_log(self, log_data: dict) -> ProcessingLog:
        """Create a new processing log entry"""
        try:
            log = ProcessingLog(
                id=uuid.uuid4(),
                dog_id=uuid.UUID(log_data['dog_id']) if log_data.get('dog_id') else None,
                image_path=log_data['image_path'],
                action_taken=log_data['action_taken'],
                similarity_score=log_data.get('similarity_score'),
                matched_dog_id=uuid.UUID(log_data['matched_dog_id']) if log_data.get('matched_dog_id') else None,
                decision_reason=log_data.get('decision_reason'),
                processing_time=log_data.get('processing_time'),
                created_at=datetime.utcnow()
            )
            
            self.db.add(log)
            await self.db.commit()
            await self.db.refresh(log)
            
            return log
            
        except Exception as e:
            await self.db.rollback()
            raise e
    
    async def get_logs(self, limit: int = 100) -> List[ProcessingLog]:
        """Get recent processing logs"""
        try:
            result = await self.db.execute(
                select(ProcessingLog)
                .order_by(ProcessingLog.created_at.desc())
                .limit(limit)
            )
            return result.scalars().all()
        except Exception as e:
            raise e 