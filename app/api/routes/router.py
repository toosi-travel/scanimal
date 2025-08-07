from fastapi import APIRouter
from app.api.routes import heartbeat, dogs

api_router = APIRouter()

api_router.include_router(heartbeat.router)
api_router.include_router(dogs.router)
