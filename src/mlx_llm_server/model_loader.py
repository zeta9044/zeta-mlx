"""Custom model loader with dynamic model registration."""

import sys
import importlib


def register_qwen3_model():
    """Register Qwen3 model to MLX-LM dynamically."""
    # Import our custom Qwen3 implementation
    from mlx_llm_server.custom_models import qwen3

    # Register it as mlx_lm.models.qwen3 so MLX-LM can find it
    sys.modules['mlx_lm.models.qwen3'] = qwen3

    print("Qwen3 model registered successfully")


def setup_custom_models():
    """Setup all custom model implementations."""
    register_qwen3_model()
