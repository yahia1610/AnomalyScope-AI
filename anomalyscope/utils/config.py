"""YAML configuration loader with deep-merge and dot-access support.

Loads ``configs/default.yaml`` first, then deep-merges a model-specific YAML
on top, and finally applies any programmatic overrides via :func:`merge_dict`.
The result is returned as a **frozen** (read-only) namespace object so that
downstream code cannot accidentally mutate shared configuration state.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Keys that every valid configuration must contain after merging.
_REQUIRED_KEYS: list[str] = [
    "dataset_root",
    "category",
    "image_size",
    "batch_size",
    "device",
    "checkpoint_dir",
    "output_dir",
]

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively deep-merge *override* into a **copy** of *base*.

    For every key in *override*:
    - If both *base[key]* and *override[key]* are dicts, merge recursively.
    - Otherwise, *override[key]* replaces *base[key]*.

    Args:
        base: The base dictionary (not modified in place).
        override: The dictionary whose values take priority.

    Returns:
        A new dictionary containing the merged result.
    """
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_dict(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


class FrozenNamespace(SimpleNamespace):
    """A read-only :class:`~types.SimpleNamespace` with dot-access.

    Nested dicts are automatically converted to ``FrozenNamespace`` instances
    so that ``cfg.model.name`` works.  After construction every attribute is
    frozen — attempts to set or delete attributes raise :class:`AttributeError`.
    """

    def __init__(self, **kwargs: Any) -> None:
        converted = {}
        for key, value in kwargs.items():
            converted[key] = self._convert(value)
        super().__init__(**converted)

    # --- freeze after init ---------------------------------------------------

    def __setattr__(self, name: str, value: Any) -> None:  # noqa: D105
        if hasattr(self, "_FrozenNamespace__frozen"):
            raise AttributeError(
                f"Cannot set attribute '{name}' on a frozen config. "
                "Create a new config with the desired overrides instead."
            )
        super().__setattr__(name, value)

    def __delattr__(self, name: str) -> None:  # noqa: D105
        raise AttributeError(
            f"Cannot delete attribute '{name}' on a frozen config."
        )

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

    # Mark instance as frozen once __init__ finishes.
    def __post_init_freeze__(self) -> None:
        """Freeze the namespace after construction."""
        super().__setattr__("_FrozenNamespace__frozen", True)

    # --- helpers -------------------------------------------------------------

    @staticmethod
    def _convert(value: Any) -> Any:
        """Recursively convert dicts to ``FrozenNamespace``."""
        if isinstance(value, dict):
            ns = FrozenNamespace(**value)
            ns.__post_init_freeze__()
            return ns
        if isinstance(value, list):
            return [FrozenNamespace._convert(item) for item in value]
        return value

    def to_dict(self) -> dict[str, Any]:
        """Return a plain-dict copy of this namespace (recursively)."""
        result: dict[str, Any] = {}
        for key, value in vars(self).items():
            if key.startswith("_FrozenNamespace__"):
                continue
            if isinstance(value, FrozenNamespace):
                result[key] = value.to_dict()
            elif isinstance(value, list):
                result[key] = [
                    item.to_dict() if isinstance(item, FrozenNamespace) else item
                    for item in value
                ]
            else:
                result[key] = value
        return result


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def _resolve_paths(cfg: dict[str, Any], project_root: Path) -> dict[str, Any]:
    """Resolve path-like values relative to *project_root*.

    Only the well-known keys ``dataset_root``, ``checkpoint_dir``, and
    ``output_dir`` are resolved.  Absolute paths are left untouched.
    """
    path_keys = ("dataset_root", "checkpoint_dir", "output_dir")
    resolved = copy.deepcopy(cfg)
    for key in path_keys:
        if key in resolved and isinstance(resolved[key], str):
            p = Path(resolved[key])
            if not p.is_absolute():
                resolved[key] = str((project_root / p).resolve())
    return resolved


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate(cfg: dict[str, Any]) -> None:
    """Raise ``ValueError`` if any required key is missing."""
    missing = [k for k in _REQUIRED_KEYS if k not in cfg]
    if missing:
        raise ValueError(
            f"Configuration is missing required keys: {', '.join(missing)}"
        )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def load_config(
    model_config_path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
    project_root: str | Path | None = None,
) -> FrozenNamespace:
    """Load, merge, validate, and freeze the project configuration.

    Loading order (later sources override earlier ones):

    1. ``configs/default.yaml`` (always loaded)
    2. *model_config_path* (e.g. ``configs/padim.yaml``)
    3. *overrides* dict (programmatic / CLI overrides)

    Args:
        model_config_path: Path to a model-specific YAML file.  May be
            ``None`` if only defaults (+ overrides) are desired.
        overrides: A flat or nested dict of values that override everything.
        project_root: The project root directory used to resolve relative
            paths.  Defaults to the parent of the ``configs/`` directory.

    Returns:
        A :class:`FrozenNamespace` with dot-access to every config value.

    Raises:
        FileNotFoundError: If the default config or the model config cannot
            be found.
        ValueError: If required keys are missing after merging.
    """
    # Determine project root.
    if project_root is not None:
        root = Path(project_root).resolve()
    else:
        root = Path(__file__).resolve().parents[2]  # anomalyscope/utils/config.py -> root

    default_path = root / "configs" / "default.yaml"
    if not default_path.exists():
        raise FileNotFoundError(f"Default config not found: {default_path}")

    # 1. Load defaults.
    with open(default_path, "r", encoding="utf-8") as fh:
        cfg: dict[str, Any] = yaml.safe_load(fh) or {}
    logger.info("Loaded default config from %s", default_path)

    # 2. Merge model-specific config.
    if model_config_path is not None:
        model_path = Path(model_config_path)
        if not model_path.is_absolute():
            model_path = root / model_path
        if not model_path.exists():
            raise FileNotFoundError(f"Model config not found: {model_path}")
        with open(model_path, "r", encoding="utf-8") as fh:
            model_cfg: dict[str, Any] = yaml.safe_load(fh) or {}
        cfg = merge_dict(cfg, model_cfg)
        logger.info("Merged model config from %s", model_path)

    # 3. Apply overrides.
    if overrides:
        cfg = merge_dict(cfg, overrides)
        logger.info("Applied %d override(s)", len(overrides))

    # Resolve relative paths.
    cfg = _resolve_paths(cfg, root)

    # Validate.
    _validate(cfg)

    # Build frozen namespace.
    ns = FrozenNamespace(**cfg)
    ns.__post_init_freeze__()
    logger.info("Configuration loaded and frozen.")
    return ns
