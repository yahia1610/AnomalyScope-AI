"""Anomaly detection model registry and factory.

This module re-exports the :class:`AnomalyModel` abstract base class and
provides a :func:`get_model` factory function that instantiates the correct
model implementation based on the ``config.model.name`` value.

Model implementations are imported lazily inside :func:`get_model` so that
the package can be imported without pulling in heavyweight dependencies
until they are actually needed.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from anomalyscope.models.base import AnomalyModel

__all__ = ["AnomalyModel", "get_model"]

# Registry mapping model names (as they appear in YAML configs) to their
# fully-qualified module path and class name.  New models only need to add
# an entry here and provide the corresponding module.
_MODEL_REGISTRY: dict[str, tuple[str, str]] = {
    "autoencoder": ("anomalyscope.models.autoencoder", "AutoencoderModel"),
    "padim": ("anomalyscope.models.padim", "PaDiMModel"),
    "patchcore": ("anomalyscope.models.patchcore", "PatchCoreModel"),
}


def get_model(config: SimpleNamespace | Any) -> AnomalyModel:
    """Instantiate and return an anomaly detection model from *config*.

    The model name is read from ``config.model.name`` and looked up in the
    internal registry.  The corresponding module is imported lazily and the
    model class is constructed with *config* as its sole argument.

    Args:
        config: A namespace object whose ``model.name`` attribute identifies
            the desired model (e.g. ``"padim"``, ``"patchcore"``,
            ``"autoencoder"``).

    Returns:
        An instance of :class:`AnomalyModel`.

    Raises:
        ValueError: If the requested model name is not in the registry.
        ImportError: If the model's implementation module has not been
            created yet (expected in a later development phase).
    """
    model_name: str = config.model.name.lower()

    if model_name not in _MODEL_REGISTRY:
        available = ", ".join(sorted(_MODEL_REGISTRY.keys()))
        raise ValueError(
            f"Unknown model '{model_name}'. Available models: {available}"
        )

    module_path, class_name = _MODEL_REGISTRY[model_name]

    import importlib

    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as exc:
        raise ImportError(
            f"Model '{model_name}' is registered but its implementation module "
            f"'{module_path}' was not found. It may not have been created yet."
        ) from exc

    model_cls = getattr(module, class_name)
    return model_cls(config)
