from pydantic_settings import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    app_name: str = "Dog Face Recognition API"
    version: str = "1.0.0"
    debug: bool = False
    
    # Model settings
    yolo_model_path: str = "yolov8m.pt"
    confidence_threshold: float = 0.3  # Lowered from 0.5 for better detection
    nms_threshold: float = 0.4
    
    # FAISS settings
    embedding_dimension: int = 512  # ArcFace uses 512-dimensional embeddings
    faiss_index_type: str = "Flat"
    
    # Storage settings
    upload_dir: str = "uploads"
    embeddings_dir: str = "embeddings"
    database_path: str = "dog_database.faiss"
    
    # API settings
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    allowed_extensions: list = [".jpg", ".jpeg", ".png", ".bmp"]
    
    # PostgreSQL settings
    postgresql_user: str = "sungjinwoo"
    postgresql_pass: str = "r,g:|zM=4WFE1q]O`iS#c,C}gZe?X4HS"
    postgresql_host: str = "ls-780a6b05538e7e127ae437d0fb6ea0172286c0a9.c36qugwkymv1.ap-south-1.rds.amazonaws.com"
    postgresql_db: str = "scanimal-prod"
    postgresql_port: int = 5432
    
    class Config:
        env_file = ".env"

settings = Settings()

# Create necessary directories
os.makedirs(settings.upload_dir, exist_ok=True)
os.makedirs(settings.embeddings_dir, exist_ok=True) 