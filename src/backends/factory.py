"""Backend factory — picks the right backend class from a config string."""
from __future__ import annotations

from typing import Any

from .base import PoseBackend


def build_backend(name: str, options: dict[str, Any]) -> PoseBackend:
    """Construct a pose backend by name.

    Parameters
    ----------
    name : str
        One of ``"mediapipe"``, ``"movenet"``.
    options : dict
        Backend-specific options from the config file.
    """
    name = name.lower()
    if name == "mediapipe":
        from .mediapipe_backend import MediaPipeBackend

        return MediaPipeBackend.from_config(options)
    if name == "movenet":
        from .movenet_backend import MoveNetBackend

        return MoveNetBackend.from_config(options)
    raise ValueError(f"Unknown backend: {name!r}")
