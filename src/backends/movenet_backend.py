"""MoveNet backend — optional alternative to MediaPipe.

Not implemented in Phase 1. Stub is kept so the backend factory can
reference it once someone decides to add it.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .base import PoseBackend


class MoveNetBackend(PoseBackend):
    def __init__(self) -> None:
        raise NotImplementedError(
            "MoveNet backend is not implemented yet. "
            "Use backend=mediapipe in your config."
        )

    def extract_landmarks(self, frame: np.ndarray) -> np.ndarray | None:
        raise NotImplementedError

    def close(self) -> None:
        pass

    @classmethod
    def from_config(cls, options: dict[str, Any]) -> "MoveNetBackend":
        return cls()
