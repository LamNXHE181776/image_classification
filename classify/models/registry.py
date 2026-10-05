"""Model registry and factory for image classification backbones.

Register a custom backbone with the decorator::

    from classify.models.registry import register_model
    import torch.nn as nn

    @register_model("my_custom_cnn")
    def build_my_cnn(num_classes: int) -> nn.Module:
        ...
        return model

Then use it anywhere via::

    from classify.models.registry import build_model
    model = build_model("my_custom_cnn", num_classes=5)

Torchvision models are supported out-of-the-box without explicit registration.
"""

from __future__ import annotations

from typing import Any, Callable

import torch
import torch.nn as nn

# Global registry:  name → builder callable(num_classes, **kwargs) -> nn.Module
_MODEL_REGISTRY: dict[str, Callable[..., nn.Module]] = {}


def register_model(name: str) -> Callable:
    """Decorator that registers a builder function under *name*.

    Args:
        name: Unique model key (e.g. ``"my_custom_resnet"``).

    Returns:
        Decorator that registers the function and returns it unchanged.

    Example::

        @register_model("lite_cnn")
        def build_lite_cnn(num_classes: int) -> nn.Module:
            ...
    """
    def decorator(fn: Callable[..., nn.Module]) -> Callable[..., nn.Module]:
        if name in _MODEL_REGISTRY:
            raise ValueError(
                f"Model '{name}' is already registered. "
                "Use a unique name or remove the existing registration."
            )
        _MODEL_REGISTRY[name] = fn
        return fn

    return decorator


def build_model(name: str, num_classes: int, **kwargs: Any) -> nn.Module:
    """Instantiate a model by name.

    Looks up the custom registry first; falls back to torchvision pretrained
    models with an automatic head replacement.

    Args:
        name: Registered model key or torchvision model name.
        num_classes: Number of output classes for the classification head.
        **kwargs: Extra keyword arguments forwarded to custom builders.

    Returns:
        Constructed :class:`nn.Module` ready for training.

    Raises:
        KeyError: If *name* is not found in either registry.
    """
    if name in _MODEL_REGISTRY:
        return _MODEL_REGISTRY[name](num_classes=num_classes, **kwargs)
    return _build_torchvision_model(name, num_classes)


def _build_torchvision_model(name: str, num_classes: int) -> nn.Module:
    """Build a torchvision pretrained model with a replaced classification head.

    Supports architectures with ``model.fc``, ``model.classifier``, or
    ``model.head`` as the final linear layer.

    Args:
        name: Exact torchvision model name (e.g. ``"efficientnet_v2_s"``).
        num_classes: Number of output classes.

    Returns:
        Modified :class:`nn.Module` with updated output head.

    Raises:
        KeyError: If *name* is not a valid torchvision model.
        ValueError: If the head attribute cannot be inferred automatically.
    """
    import torchvision

    available = {
        k for k, v in torchvision.models.__dict__.items()
        if callable(v) and not k.startswith("_")
    }
    if name not in available:
        raise KeyError(
            f"Model '{name}' not found in torchvision.models or the custom registry.\n"
            f"Available torchvision models: {sorted(available)}"
        )

    model: nn.Module = torchvision.models.__dict__[name](weights="DEFAULT")

    if hasattr(model, "fc"):
        in_features: int = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes, bias=True)
    elif hasattr(model, "classifier"):
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes, bias=True)
    elif hasattr(model, "head"):
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, num_classes, bias=True)
    else:
        raise ValueError(
            f"Cannot infer the classification head for '{name}'. "
            "Register a custom builder via @register_model(name)."
        )

    return model


def list_models() -> list[str]:
    """Return all currently registered custom model names.

    Returns:
        Alphabetically sorted list of keys in the custom registry.
    """
    return sorted(_MODEL_REGISTRY.keys())
