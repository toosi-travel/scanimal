from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional
import time

from app.models.schemas import (
    DuplicateCheckResponse, PendingDogsListResponse, ApprovalRequest, 
    ApprovalResponse, SimilarityThresholds, ProcessingLog,
    MultiImageDuplicateCheckResponse, ImageMatchResult, DogMatchInfo, BestMatchResponse
)
from app.services.duplicate_detection_service import duplicate_detection_service
from app.core.database import get_db
from app.core.log_config import logger

router = APIRouter()

@router.post("/check", response_model=MultiImageDuplicateCheckResponse)
async def check_for_duplicates(
    images: List[UploadFile] = File(...),
    db: AsyncSession = Depends(get_db)
):
    """Check for duplicates in multiple uploaded images and return best matches"""
    start_time = time.time()
    
    try:
        if not images:
            raise HTTPException(status_code=400, detail="At least one image must be provided")
        
        logger.info(f"Processing {len(images)} images for duplicate detection")
        
        results = []
        successful_matches = 0
        failed_images = 0
        
        for i, image in enumerate(images):
            image_start_time = time.time()
            
            try:
                # Read image data
                image_data = await image.read()
                
                # Check for duplicates and get best match
                match_result = await duplicate_detection_service.find_best_match(
                    image_data, db
                )
                
                if match_result.success and match_result.best_match:
                    successful_matches += 1
                    results.append(ImageMatchResult(
                        image_index=i,
                        success=True,
                        message=f"Best match found: {match_result.best_match.name} (Score: {match_result.best_match.similarity_score:.3f})",
                        best_match=match_result.best_match,
                        processing_time=time.time() - image_start_time
                    ))
                else:
                    failed_images += 1
                    results.append(ImageMatchResult(
                        image_index=i,
                        success=False,
                        message="No suitable match found",
                        error=match_result.message,
                        processing_time=time.time() - image_start_time
                    ))
                    
            except Exception as e:
                failed_images += 1
                logger.error(f"Error processing image {i}: {str(e)}")
                results.append(ImageMatchResult(
                    image_index=i,
                    success=False,
                    message="Error processing image",
                    error=str(e),
                    processing_time=time.time() - image_start_time
                ))
        
        total_processing_time = time.time() - start_time
        
        return MultiImageDuplicateCheckResponse(
            success=True,
            message=f"Processed {len(images)} images. {successful_matches} successful matches, {failed_images} failed.",
            total_images=len(images),
            successful_matches=successful_matches,
            failed_images=failed_images,
            results=results,
            total_processing_time=total_processing_time
        )
        
    except Exception as e:
        logger.error(f"Error in duplicate check: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing images: {str(e)}")

@router.post("/check-single", response_model=BestMatchResponse)
async def check_for_duplicates_single(
    image: UploadFile = File(...),
    db: AsyncSession = Depends(get_db)
):
    """Check for duplicates in a single uploaded image and return best match"""
    try:
        # Read image data
        image_data = await image.read()
        
        # Find best match using the new method
        response = await duplicate_detection_service.find_best_match(
            image_data, db
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error in single image duplicate check: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

# @router.get("/pending", response_model=PendingDogsListResponse)
# async def get_pending_dogs(db: AsyncSession = Depends(get_db)):
#     """Get all pending dogs for approval"""
#     try:
#         pending_dogs = await duplicate_detection_service.get_pending_dogs(db)
        
#         return PendingDogsListResponse(
#             success=True,
#             message=f"Found {len(pending_dogs)} pending dogs",
#             pending_dogs=pending_dogs
#         )
        
#     except Exception as e:
#         logger.error(f"Error getting pending dogs: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Failed to load pending dogs: {str(e)}")

# @router.post("/pending/{pending_dog_id}/approve", response_model=ApprovalResponse)
# async def approve_pending_dog(
#     pending_dog_id: str,
#     approval_request: ApprovalRequest,
#     db: AsyncSession = Depends(get_db)
# ):
#     """Approve a pending dog"""
#     try:
#         success = await duplicate_detection_service.approve_pending_dog(
#             pending_dog_id, approval_request.admin_notes, db
#         )
        
#         if success:
#             return ApprovalResponse(
#                 success=True,
#                 message=f"Dog {pending_dog_id} approved successfully"
#             )
#         else:
#             raise HTTPException(status_code=404, detail="Pending dog not found")
            
#     except Exception as e:
#         logger.error(f"Error approving pending dog: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Failed to approve dog: {str(e)}")

# @router.post("/pending/{pending_dog_id}/reject", response_model=ApprovalResponse)
# async def reject_pending_dog(
#     pending_dog_id: str,
#     approval_request: ApprovalRequest,
#     db: AsyncSession = Depends(get_db)
# ):
#     """Reject a pending dog"""
#     try:
#         success = await duplicate_detection_service.reject_pending_dog(
#             pending_dog_id, approval_request.admin_notes, db
#         )
        
#         if success:
#             return ApprovalResponse(
#                 success=True,
#                 message=f"Dog {pending_dog_id} rejected successfully"
#             )
#         else:
#             raise HTTPException(status_code=404, detail="Pending dog not found")
            
#     except Exception as e:
#         logger.error(f"Error rejecting pending dog: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Failed to reject dog: {str(e)}")

# @router.get("/thresholds", response_model=SimilarityThresholds)
# async def get_thresholds():
#     """Get current similarity thresholds"""
#     return duplicate_detection_service.thresholds

# @router.put("/thresholds", response_model=SimilarityThresholds)
# async def update_thresholds(thresholds: SimilarityThresholds):
#     """Update similarity thresholds"""
#     duplicate_detection_service.set_thresholds(thresholds)
#     return duplicate_detection_service.thresholds

# @router.get("/stats")
# async def get_stats(db: AsyncSession = Depends(get_db)):
#     """Get system statistics"""
#     try:
#         stats = await duplicate_detection_service.get_system_stats(db)
#         return {
#             "success": True,
#             "message": "Statistics retrieved successfully",
#             "stats": stats
#         }
#     except Exception as e:
#         logger.error(f"Error getting stats: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")

# @router.get("/logs")
# async def get_processing_logs(
#     limit: int = 100,
#     db: AsyncSession = Depends(get_db)
# ):
#     """Get processing logs"""
#     try:
#         logs = await duplicate_detection_service.get_processing_logs(db, limit)
#         return {
#             "success": True,
#             "message": f"Retrieved {len(logs)} processing logs",
#             "logs": logs
#         }
#     except Exception as e:
#         logger.error(f"Error getting processing logs: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Failed to get logs: {str(e)}")