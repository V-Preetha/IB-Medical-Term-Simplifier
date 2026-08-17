"""Schemas for service health responses."""

from pydantic import BaseModel, ConfigDict


class HealthResponse(BaseModel):
    """Client-safe health response."""

    model_config = ConfigDict(extra="forbid")

    status: str
    service_name: str
    environment: str
    version: str

