"""API Routes"""
from zeta_mlx.inference.api.routes.chat import router as chat_router
from zeta_mlx.inference.api.routes.models import router as models_router
from zeta_mlx.inference.api.routes.health import router as health_router
from zeta_mlx.inference.api.routes.tokenize import router as tokenize_router

__all__ = ["chat_router", "models_router", "health_router", "tokenize_router"]
