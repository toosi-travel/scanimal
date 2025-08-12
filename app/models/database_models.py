from sqlalchemy import Column, String, DateTime, Float, Text, Integer
from sqlalchemy.dialects.postgresql import UUID, JSON
from datetime import datetime
import uuid
from app.core.database import Base

class Dog(Base):
    """Main dogs table - SIMPLIFIED"""
    __tablename__ = "dogs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=True)
    breed = Column(String(255), nullable=True)
    owner = Column(String(255), nullable=True)
    description = Column(Text, nullable=True)
    image_path = Column(String(500), nullable=False)
    embedding_vector = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class PendingDog(Base):
    """Pending dogs awaiting approval - SIMPLIFIED"""
    __tablename__ = "pending_dogs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    dog_id = Column(UUID(as_uuid=True), nullable=True)  # No foreign key constraint
    image_path = Column(String(500), nullable=False)
    embedding_vector = Column(JSON, nullable=False)
    similar_dog_id = Column(UUID(as_uuid=True), nullable=True)  # No foreign key constraint
    similarity_score = Column(Float, nullable=False)
    status = Column(String(50), default="waiting_approval")
    name = Column(String(255), nullable=True)
    breed = Column(String(255), nullable=True)
    owner = Column(String(255), nullable=True)
    description = Column(Text, nullable=True)
    admin_notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class ProcessingLog(Base):
    """Audit log - SIMPLIFIED"""
    __tablename__ = "processing_logs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    dog_id = Column(UUID(as_uuid=True), nullable=True)  # No foreign key constraint
    image_path = Column(String(500), nullable=False)
    action_taken = Column(String(255), nullable=False)
    similarity_score = Column(Float, nullable=True)
    matched_dog_id = Column(UUID(as_uuid=True), nullable=True)  # No foreign key constraint
    decision_reason = Column(Text, nullable=True)
    processing_time = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow) 