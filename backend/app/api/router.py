"""Top-level API router for the backend service."""

from fastapi import APIRouter

from app.api.routes import health, simplify

api_router = APIRouter()
api_router.include_router(health.router, tags=["health"])
api_router.include_router(simplify.router, prefix="/reports", tags=["reports"])

