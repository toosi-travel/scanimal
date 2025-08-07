from fastapi import APIRouter 

from app.models.heartbeat import HeartbeatResult

router = APIRouter(prefix="/health", tags=["Health"])


@router.get("/heartbeat", response_model=HeartbeatResult)
async def get_heartbeat() -> HeartbeatResult:
    heartbeat = HeartbeatResult(is_alive=True)
    return heartbeat