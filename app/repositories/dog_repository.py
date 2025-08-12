from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete
from sqlalchemy.orm import selectinload
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid
import json
import numpy as np

from app.models.database_models import Dog, PendingDog, ProcessingLog
from app.models.schemas import DogInfo, PendingDogInfo, ProcessingLog as ProcessingLogSchema

class DogRepository:
    """Repository for dog-related database operations"""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def create_dog(self, dog_info: dict, embedding: np.ndarray) -> Dog:
        """Create a new dog in the database"""
        try:
            # Convert embedding to list for JSON storage
            embedding_list = embedding.tolist() if hasattr(embedding, 'tolist') else embedding
            
            dog = Dog(
                id=uuid.uuid4(),
                name=dog_info.get('name'),
                breed=dog_info.get('breed'),
                owner=dog_info.get('owner'),
                description=dog_info.get('description'),
                image_path=dog_info.get('image_path', ''),
                embedding_vector=embedding_list,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            
            self.db.add(dog)
            await self.db.commit()
            await self.db.refresh(dog)
            
            return dog
            
        except Exception as e:
            await self.db.rollback()
            raise e
    
    async def get_dog_by_id(self, dog_id: str) -> Optional[Dog]:
        """Get dog by ID"""
        try:
            result = await self.db.execute(
                select(Dog).where(Dog.id == uuid.UUID(dog_id))
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
    
    async def search_similar_dogs(self, query_embedding: np.ndarray, threshold: float = 0.6) -> List[Dict[str, Any]]:
        """Search for similar dogs using cosine similarity"""
        try:
            # Get all dogs
            dogs = await self.get_all_dogs()
            matches = []
            
            for dog in dogs:
                if dog.embedding_vector:
                    # Convert stored embedding back to numpy array
                    stored_embedding = np.array(dog.embedding_vector, dtype=np.float32)
                    
                    # Calculate cosine similarity
                    similarity = self._calculate_similarity(query_embedding, stored_embedding)
                    
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
    
    def _calculate_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings"""
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