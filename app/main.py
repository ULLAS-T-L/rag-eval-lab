from fastapi import FastAPI

from app.api.health import router as health_router
from app.api.query import router as query_router
from app.core.config import get_settings


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app_name, version=settings.app_version)
    app.include_router(health_router)
    app.include_router(query_router)
    return app


app = create_app()
