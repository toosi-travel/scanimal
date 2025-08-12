import numpy as np
import uuid
import time
import os
from datetime import datetime
from typing import List, Optional, Tuple, Dict, Any

from app.models.schemas import (
    PendingDogInfo, PendingDogStatus, DuplicateCheckResponse,
    SimilarityThresholds, ProcessingLog, DogMatchInfo, BestMatchResponse
)
from app.services.detection_service import detection_service
from app.services.embedding_service import embedding_service, EmbeddingService
from app.core.config import settings
from app.core.log_config import logger
from app.core.similarity_config import Thresholds
from app.repositories.dog_repository import DogRepository, PendingDogRepository, ProcessingLogRepository

class DuplicateDetectionService:
    """
    Duplicate Detection Service fully integrated with PostgreSQL database.
    No local storage - all data persists in the database.
    """
    
    def __init__(self):
        # Enhanced thresholds for better color discrimination
        self.thresholds = Thresholds()
        self.embedding_service = EmbeddingService()
        self.detection_service = detection_service
        logger.info("DuplicateDetectionService initialized with PostgreSQL integration")
    
    def set_thresholds(self, thresholds: SimilarityThresholds):
        """Update similarity thresholds"""
        # Convert SimilarityThresholds to our internal Thresholds format
        self.thresholds.auto_reject_threshold = thresholds.auto_reject_threshold
        self.thresholds.pending_threshold_min = thresholds.pending_threshold_min
        self.thresholds.auto_register_threshold = thresholds.auto_register_threshold
        logger.info(f"Updated thresholds: {thresholds}")
    
    async def check_for_duplicates(self, image: bytes, name: Optional[str] = None, 
                                 breed: Optional[str] = None, owner: Optional[str] = None, 
                                 description: Optional[str] = None, db_session=None) -> DuplicateCheckResponse:
        """
        Check for duplicates and decide action based on similarity scores.
        Fully integrated with PostgreSQL database.
        """
        start_time = time.time()
        
        if not db_session:
            return DuplicateCheckResponse(
                success=False,
                message="Database session required",
                action="error",
                processing_time=time.time() - start_time
            )
        
        try:
            # Initialize repositories
            dog_repo = DogRepository(db_session)
            pending_dog_repo = PendingDogRepository(db_session)
            processing_log_repo = ProcessingLogRepository(db_session)
            
            # Convert image bytes to numpy array
            image_array = self._bytes_to_numpy(image)
            
            # Detect dogs in the image
            try:
                detections = detection_service.detect_dogs(image_array)
                if not detections:
                    return DuplicateCheckResponse(
                        success=False,
                        message="No dogs detected in the image",
                        action="error",
                        processing_time=time.time() - start_time
                    )
            except Exception as e:
                logger.error(f"Error in dog detection: {str(e)}")
                return DuplicateCheckResponse(
                    success=False,
                    message=f"Error detecting dogs: {str(e)}",
                    action="error",
                    processing_time=time.time() - start_time
                )
            
            # Generate embeddings
            try:
                embedding_results = embedding_service.generate_embeddings_from_detections(
                    image_array, detections
                )
                
                if not embedding_results:
                    return DuplicateCheckResponse(
                        success=False,
                        message="Could not generate face embeddings",
                        action="error",
                        processing_time=time.time() - start_time
                    )
            except Exception as e:
                logger.error(f"Error in embedding generation: {str(e)}")
                return DuplicateCheckResponse(
                    success=False,
                    message=f"Error generating embeddings: {str(e)}",
                    action="error",
                    processing_time=time.time() - start_time
                )
            
            # Search for similar dogs using PostgreSQL
            try:
                query_embedding = np.array(embedding_results[0].embedding, dtype=np.float32)
                logger.info(f"Query embedding shape: {query_embedding.shape}, type: {query_embedding.dtype}")
                
                # Search for similar dogs in PostgreSQL
                matches = await dog_repo.search_similar_dogs(
                    query_embedding, 
                    threshold=self.thresholds.auto_reject_threshold
                )
                logger.info(f"Found {len(matches) if matches else 0} similar dogs using PostgreSQL")
                
            except Exception as e:
                logger.error(f"Error in similarity search: {str(e)}")
                return DuplicateCheckResponse(
                    success=False,
                    message=f"Error searching for similar dogs: {str(e)}",
                    action="error",
                    processing_time=time.time() - start_time
                )
            
            processing_time = time.time() - start_time
            
            if not matches:
                # No similar dogs found, auto-register
                return await self._auto_register_dog(
                    image, query_embedding, name, breed, owner, description, 
                    processing_time, dog_repo, processing_log_repo
                )
            
            # Get highest similarity score
            highest_score = max(match['similarity_score'] for match in matches)
            best_match = matches[0]
            best_match_id = best_match['dog_id']
            
            # Log the decision process
            try:
                await self._log_decision(
                    image, highest_score, best_match_id, 
                    f"Highest similarity: {highest_score:.3f}", 
                    processing_time, processing_log_repo
                )
            except Exception as log_error:
                logger.error(f"Error logging decision: {str(log_error)}")
                # Continue with the main operation even if logging fails
            
            # Decision logic based on thresholds
            if highest_score >= self.thresholds.auto_register_threshold:
                # Very high similarity, auto-register as new dog
                return await self._auto_register_dog(
                    image, query_embedding, name, breed, owner, description, 
                    processing_time, dog_repo, processing_log_repo
                )
            elif highest_score >= self.thresholds.pending_threshold_min:
                # Medium similarity, add to pending approval
                return await self._add_to_pending_approval(
                    image, query_embedding, highest_score, best_match_id,
                    name, breed, owner, description, processing_time, 
                    pending_dog_repo, processing_log_repo
                )
            else:
                # Low similarity, auto-reject
                return await self._auto_reject_dog(
                    image, highest_score, best_match_id, processing_time, 
                    processing_log_repo
                )
                
        except Exception as e:
            logger.error(f"Error in duplicate detection: {str(e)}")
            return DuplicateCheckResponse(
                success=False,
                message=f"Error processing image: {str(e)}",
                action="error",
                processing_time=time.time() - start_time
            )
    
    async def find_best_match(self, image: bytes, db_session=None) -> 'BestMatchResponse':
        """
        Find the best matching dog for an image without auto-approving/rejecting.
        Returns the best match with similarity score.
        """
        start_time = time.time()
        
        if not db_session:
            return BestMatchResponse(
                success=False,
                message="Database session required",
                processing_time=time.time() - start_time
            )
        
        try:
            # Initialize repositories
            dog_repo = DogRepository(db_session)
            
            # Convert image bytes to numpy array
            image_array = self._bytes_to_numpy(image)
            
            # Detect dogs in the image
            try:
                detections = detection_service.detect_dogs(image_array)
                if not detections:
                    return BestMatchResponse(
                        success=False,
                        message="No dogs detected in the image",
                        processing_time=time.time() - start_time
                    )
            except Exception as e:
                logger.error(f"Error in dog detection: {str(e)}")
                return BestMatchResponse(
                    success=False,
                    message=f"Error detecting dogs: {str(e)}",
                    processing_time=time.time() - start_time
                )
            
            # Generate embeddings
            try:
                embedding_results = embedding_service.generate_embeddings_from_detections(
                    image_array, detections
                )
                
                if not embedding_results:
                    return BestMatchResponse(
                        success=False,
                        message="Could not generate embeddings",
                        processing_time=time.time() - start_time
                    )
            except Exception as e:
                logger.error(f"Error in embedding generation: {str(e)}")
                return BestMatchResponse(
                    success=False,
                    message=f"Error generating embeddings: {str(e)}",
                    processing_time=time.time() - start_time
                )
            
            # Search for similar dogs using PostgreSQL
            try:
                query_embedding = np.array(embedding_results[0].embedding, dtype=np.float32)
                logger.info(f"Query embedding shape: {query_embedding.shape}, type: {query_embedding.dtype}")
                
                # Search for similar dogs in PostgreSQL with a lower threshold to get more candidates
                search_threshold = 0.5  # Lower threshold to get more potential matches
                matches = await dog_repo.search_similar_dogs(
                    query_embedding, 
                    threshold=search_threshold
                )
                logger.info(f"Found {len(matches) if matches else 0} similar dogs using PostgreSQL")
                
            except Exception as e:
                logger.error(f"Error in similarity search: {str(e)}")
                return BestMatchResponse(
                    success=False,
                    message=f"Error searching for similar dogs: {str(e)}",
                    processing_time=time.time() - start_time
                )
            
            processing_time = time.time() - start_time
            
            if not matches:
                return BestMatchResponse(
                    success=True,
                    message="No similar dogs found in database",
                    processing_time=processing_time
                )
            
            # Find the best match
            best_match = matches[0]  # Matches are already sorted by similarity score
            
            # Create DogMatchInfo object from the match dictionary
            dog_match_info = DogMatchInfo(
                dog_id=best_match['dog_id'],
                name=best_match['dog_info']['name'],
                breed=best_match['dog_info'].get('breed'),
                owner=best_match['dog_info'].get('owner'),
                similarity_score=best_match['similarity_score'],
                image_path=None  # The repository doesn't return image_path
            )
            
            logger.info(f"Best match found: {dog_match_info.name} with score {dog_match_info.similarity_score:.3f}")
            
            return BestMatchResponse(
                success=True,
                message=f"Best match found: {dog_match_info.name}",
                best_match=dog_match_info,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Error in find_best_match: {str(e)}")
            return BestMatchResponse(
                success=False,
                message=f"Error processing image: {str(e)}",
                processing_time=time.time() - start_time
            )
    
    async def _auto_register_dog(self, image: bytes, embedding: np.ndarray, 
                                name: Optional[str], breed: Optional[str], 
                                owner: Optional[str], description: Optional[str],
                                processing_time: float, dog_repo: DogRepository, 
                                processing_log_repo: ProcessingLogRepository) -> DuplicateCheckResponse:
        """Automatically register a dog with high confidence in PostgreSQL"""
        try:
            # Save image to uploads directory
            image_path = self._save_image(image)
            
            # Create dog info dictionary
            dog_info = {
                'id': uuid.uuid4(),  # Use UUID object directly
                'name': name,
                'breed': breed,
                'owner': owner,
                'description': description,
                'image_path': image_path,
                'embedding_vector': embedding.tolist()  # Include embedding in the dictionary
            }
            
            # Add to PostgreSQL database
            dog = await dog_repo.create_dog(dog_info)
            
            logger.info(f"Auto-registered dog {dog.id} in PostgreSQL with similarity above threshold")
            
            return DuplicateCheckResponse(
                success=True,
                message="Dog auto-registered successfully in PostgreSQL (high confidence)",
                action="auto_register",
                dog_id=str(dog.id),
                processing_time=processing_time
            )
                
        except Exception as e:
            logger.error(f"Error in auto-registration: {str(e)}")
            return DuplicateCheckResponse(
                success=False,
                message=f"Auto-registration failed: {str(e)}",
                action="error",
                processing_time=processing_time
            )
    
    async def _add_to_pending_approval(self, image: bytes, embedding: np.ndarray,
                                      similarity_score: float, similar_dog_id: str,
                                      name: Optional[str], breed: Optional[str],
                                      owner: Optional[str], description: Optional[str],
                                      processing_time: float, pending_dog_repo: PendingDogRepository,
                                      processing_log_repo: ProcessingLogRepository) -> DuplicateCheckResponse:
        """Add dog to pending approval queue in PostgreSQL"""
        try:
            # Save image to uploads directory
            image_path = self._save_image(image)
            
            # Ensure embedding is float32 and convert to list
            if embedding.dtype != np.float32:
                embedding = embedding.astype(np.float32)
                logger.info(f"Converted embedding to float32: {embedding.dtype}")
            
            # Create pending dog data
            pending_dog_data = {
                'image_path': image_path,
                'embedding_vector': embedding.tolist(),
                'similar_dog_id': similar_dog_id,
                'similarity_score': similarity_score,
                'name': name,
                'breed': breed,
                'owner': owner,
                'description': description
            }
            
            # Add to PostgreSQL pending dogs table
            db_pending_dog = await pending_dog_repo.create_pending_dog(pending_dog_data)
            
            logger.info(f"Added dog {db_pending_dog.id} to PostgreSQL pending approval (similarity: {similarity_score:.3f})")
            
            return DuplicateCheckResponse(
                success=True,
                message="Dog added to PostgreSQL pending approval queue",
                action="pending_approval",
                similarity_score=similarity_score,
                similar_dog_id=similar_dog_id,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Error adding to pending approval: {str(e)}")
            return DuplicateCheckResponse(
                success=False,
                message=f"Failed to add to pending approval: {str(e)}",
                action="error",
                processing_time=processing_time
            )
    
    async def _auto_reject_dog(self, image: bytes, similarity_score: float,
                              similar_dog_id: str, processing_time: float, 
                              processing_log_repo: ProcessingLogRepository) -> DuplicateCheckResponse:
        """Auto-reject dog with low similarity"""
        try:
            # Save image for audit purposes
            image_path = self._save_image(image)
            
            logger.info(f"Auto-rejected dog with similarity {similarity_score:.3f}")
            
            return DuplicateCheckResponse(
                success=True,
                message="Dog auto-rejected (low similarity)",
                action="auto_reject",
                similarity_score=similarity_score,
                similar_dog_id=similar_dog_id,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Error in auto-rejection: {str(e)}")
            return DuplicateCheckResponse(
                success=False,
                message=f"Auto-rejection failed: {str(e)}",
                action="error",
                processing_time=processing_time
            )
    
    async def get_pending_dogs(self, db_session=None) -> List[PendingDogInfo]:
        """Get all pending dogs for approval from PostgreSQL"""
        if not db_session:
            logger.error("Database session required for get_pending_dogs")
            return []
        
        try:
            pending_dog_repo = PendingDogRepository(db_session)
            db_pending_dogs = await pending_dog_repo.get_pending_dogs()
            
            logger.info(f"Retrieved {len(db_pending_dogs)} pending dogs from PostgreSQL")
            
            # Convert to PendingDogInfo format
            pending_dogs_list = []
            for db_dog in db_pending_dogs:
                pending_dog = PendingDogInfo(
                    id=str(db_dog.id),
                    image_path=db_dog.image_path,
                    embedding_vector=db_dog.embedding_vector,
                    similar_dog_id=str(db_dog.similar_dog_id) if db_dog.similar_dog_id else None,
                    similarity_score=db_dog.similarity_score,
                    status=PendingDogStatus(db_dog.status),
                    name=db_dog.name,
                    breed=db_dog.breed,
                    owner=db_dog.owner,
                    description=db_dog.description,
                    created_at=db_dog.created_at,
                    updated_at=db_dog.updated_at
                )
                pending_dogs_list.append(pending_dog)
            
            return pending_dogs_list
            
        except Exception as e:
            logger.error(f"Error getting pending dogs: {str(e)}")
            return []
    
    async def approve_pending_dog(self, pending_dog_id: str, admin_notes: Optional[str] = None, db_session=None) -> bool:
        """Approve a pending dog and move it to the main database"""
        if not db_session:
            logger.error("Database session required for approve_pending_dog")
            return False
        
        try:
            pending_dog_repo = PendingDogRepository(db_session)
            dog_repo = DogRepository(db_session)
            processing_log_repo = ProcessingLogRepository(db_session)
            
            # Get pending dog
            pending_dog = await pending_dog_repo.get_pending_dog_by_id(pending_dog_id)
            if not pending_dog:
                logger.error(f"Pending dog {pending_dog_id} not found")
                return False
            
            # Create dog info for main database
            dog_info = {
                'id': uuid.uuid4(),  # Use UUID object directly
                'name': pending_dog.name,
                'breed': pending_dog.breed,
                'owner': pending_dog.owner,
                'description': pending_dog.description,
                'image_path': pending_dog.image_path,
                'embedding_vector': pending_dog.embedding_vector  # Add missing embedding_vector field
            }
            
            # Convert embedding back to numpy array
            embedding = np.array(pending_dog.embedding_vector, dtype=np.float32)
            
            # Add to main dogs table
            dog = await dog_repo.create_dog(dog_info)
            
            # Update pending dog status to approved
            await pending_dog_repo.update_pending_dog_status(
                pending_dog_id, 'approved', admin_notes
            )
            
            # Log the approval
            await self._log_decision(
                pending_dog.image_path, pending_dog.similarity_score,
                str(pending_dog.similar_dog_id), 
                f"Approved by admin: {admin_notes or 'No notes'}", 
                0.0, processing_log_repo, str(dog.id)
            )
            
            logger.info(f"Approved pending dog {pending_dog_id} and moved to main database")
            return True
                
        except Exception as e:
            logger.error(f"Error approving pending dog: {str(e)}")
            return False
    
    async def reject_pending_dog(self, pending_dog_id: str, admin_notes: Optional[str] = None, db_session=None) -> bool:
        """Reject a pending dog"""
        if not db_session:
            logger.error("Database session required for reject_pending_dog")
            return False
        
        try:
            pending_dog_repo = PendingDogRepository(db_session)
            processing_log_repo = ProcessingLogRepository(db_session)
            
            # Get pending dog
            pending_dog = await pending_dog_repo.get_pending_dog_by_id(pending_dog_id)
            if not pending_dog:
                logger.error(f"Pending dog {pending_dog_id} not found")
                return False
            
            # Update status to rejected
            await pending_dog_repo.update_pending_dog_status(
                pending_dog_id, 'rejected', admin_notes
            )
            
            # Log the rejection
            await self._log_decision(
                pending_dog.image_path, pending_dog.similarity_score,
                str(pending_dog.similar_dog_id), 
                f"Rejected by admin: {admin_notes or 'No notes'}", 
                0.0, processing_log_repo
            )
            
            logger.info(f"Rejected pending dog {pending_dog_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error rejecting pending dog: {str(e)}")
            return False
    
    async def get_processing_logs(self, db_session=None, limit: int = 100) -> List[ProcessingLog]:
        """Get processing logs from PostgreSQL"""
        if not db_session:
            logger.error("Database session required for get_processing_logs")
            return []
        
        try:
            processing_log_repo = ProcessingLogRepository(db_session)
            db_logs = await processing_log_repo.get_logs(limit)
            
            # Convert to ProcessingLog schema format
            logs_list = []
            for db_log in db_logs:
                log = ProcessingLog(
                    id=str(db_log.id),
                    image_path=db_log.image_path,
                    action_taken=db_log.action_taken,
                    similarity_score=db_log.similarity_score,
                    matched_dog_id=str(db_log.matched_dog_id) if db_log.matched_dog_id else None,
                    decision_reason=db_log.decision_reason,
                    created_at=db_log.created_at
                )
                logs_list.append(log)
            
            return logs_list
            
        except Exception as e:
            logger.error(f"Error getting processing logs: {str(e)}")
            return []
    
    async def get_system_stats(self, db_session=None) -> Dict[str, Any]:
        """Get system statistics from PostgreSQL"""
        if not db_session:
            logger.error("Database session required for get_system_stats")
            return {}
        
        try:
            dog_repo = DogRepository(db_session)
            pending_dog_repo = PendingDogRepository(db_session)
            processing_log_repo = ProcessingLogRepository(db_session)
            
            # Get counts
            all_dogs = await dog_repo.get_all_dogs()
            pending_dogs = await pending_dog_repo.get_pending_dogs()
            recent_logs = await processing_log_repo.get_logs(1000)  # Get more logs for stats
            
            # Calculate statistics
            stats = {
                'total_dogs': len(all_dogs),
                'pending_dogs_count': len(pending_dogs),
                'total_processed': len(recent_logs),
                'auto_registered': len([log for log in recent_logs if 'auto_register' in log.action_taken.lower()]),
                'pending_approval': len([log for log in recent_logs if 'pending' in log.action_taken.lower()]),
                'auto_rejected': len([log for log in recent_logs if 'auto_reject' in log.action_taken.lower()]),
                'last_updated': datetime.utcnow().isoformat()
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting system stats: {str(e)}")
            return {}
    
    def _save_image(self, image_bytes: bytes) -> str:
        """Save image to uploads directory and return path"""
        upload_dir = "uploads"
        os.makedirs(upload_dir, exist_ok=True)
        
        filename = f"{uuid.uuid4()}.jpg"
        filepath = os.path.join(upload_dir, filename)
        
        with open(filepath, 'wb') as f:
            f.write(image_bytes)
            
        return filepath
    
    def _bytes_to_numpy(self, image_bytes: bytes) -> np.ndarray:
        """Convert image bytes to numpy array"""
        import cv2
        
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Invalid image data")
            
        return image
    
    async def _log_decision(self, image_path: str, similarity_score: float, 
                           matched_dog_id: Optional[str], decision_reason: str, 
                           processing_time: float, processing_log_repo: ProcessingLogRepository,
                           dog_id: Optional[str] = None):
        """Log processing decision for audit purposes in PostgreSQL"""
        try:
            # Ensure image_path is a string
            if isinstance(image_path, bytes):
                # If it's bytes, save it and get the path
                image_path = self._save_image(image_path)
            elif not isinstance(image_path, str):
                # If it's neither bytes nor string, create a placeholder
                image_path = f"unknown_image_{uuid.uuid4()}"
            
            # Create log data
            log_data = {
                'dog_id': dog_id,
                'image_path': image_path,
                'action_taken': decision_reason,
                'similarity_score': similarity_score,
                'matched_dog_id': matched_dog_id,
                'decision_reason': decision_reason,
                'processing_time': processing_time
            }
            
            # Create log in PostgreSQL
            log = await processing_log_repo.create_log(log_data)
            logger.info(f"Processing log created in PostgreSQL: {log.id}")
            
        except Exception as e:
            logger.error(f"Error creating processing log: {str(e)}")
            # Don't fail the main operation if logging fails

# Global instance
duplicate_detection_service = DuplicateDetectionService() 