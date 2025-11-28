"""API Routes"""
from zeta_mlx_api.routes.chat import router as chat_router
from zeta_mlx_api.routes.models import router as models_router
from zeta_mlx_api.routes.health import router as health_router

__all__ = ["chat_router", "models_router", "health_router"]
