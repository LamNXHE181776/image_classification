"""Model sub-package: backbone registry and Lightning module.

Quick usage::

    from classify.models import ImageClassificationModule, build_model, register_model
"""

from classify.models.registry import build_model, register_model, list_models
from classify.models.lightning_module import ImageClassificationModule

__all__ = [
    "build_model",
    "register_model",
    "list_models",
    "ImageClassificationModule",
]
