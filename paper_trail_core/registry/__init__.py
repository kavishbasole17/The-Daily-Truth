# paper_trail_core.registry — read-only view of the model registry.
from paper_trail_core.registry.store import get_all_models, get_top_models

__all__ = ["get_all_models", "get_top_models"]
