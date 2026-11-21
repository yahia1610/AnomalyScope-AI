"""Device selection helper for AnomalyScope AI.

Provides a single function, :func:`get_device`, that returns the appropriate
:class:`torch.device` based on a user preference string and the hardware
available on the current machine.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


def get_device(preference: str = "auto") -> torch.device:
    """Select a compute device based on *preference* and hardware availability.

    Args:
        preference: One of ``"auto"``, ``"cpu"``, or ``"cuda"``.

            * ``"auto"`` — use CUDA if a compatible GPU is detected,
              otherwise fall back to CPU.
            * ``"cpu"`` — always use the CPU.
            * ``"cuda"`` — always use CUDA (raises if unavailable).

    Returns:
        A :class:`torch.device` instance ready to be used with tensors and
        modules.

    Raises:
        ValueError: If *preference* is not one of the accepted values.
        RuntimeError: If ``"cuda"`` is requested but no CUDA device is
            available.
    """
    preference = preference.strip().lower()

    if preference == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(
                "Auto-detected CUDA device: %s", torch.cuda.get_device_name(device)
            )
        else:
            device = torch.device("cpu")
            logger.info("No CUDA device found — using CPU.")
    elif preference == "cpu":
        device = torch.device("cpu")
        logger.info("Using CPU as requested.")
    elif preference == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was explicitly requested but no CUDA device is available. "
                "Install a CUDA-capable version of PyTorch or set device to 'auto'."
            )
        device = torch.device("cuda")
        logger.info("Using CUDA device: %s", torch.cuda.get_device_name(device))
    else:
        raise ValueError(
            f"Unknown device preference '{preference}'. "
            "Expected one of: 'auto', 'cpu', 'cuda'."
        )

    return device
