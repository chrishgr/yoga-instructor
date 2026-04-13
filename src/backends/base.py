"""Abstract base for pose-estimation backends.

Every backend exposes the same minimal interface so that the rest of the
pipeline is agnostic to whether we're using MediaPipe, MoveNet, or
something else entirely.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class PoseBackend(ABC):
    """Base class for pose-estimation backends."""

    @abstractmethod
    def extract_landmarks(self, frame: np.ndarray) -> np.ndarray | None:
        """Return landmarks as an (N, 3) array, or ``None`` if no person detected.

        The caller should not assume a specific value for N — different
        backends return different numbers of keypoints. Modules that need a
        specific layout (e.g. ``compute_joint_angles``) assume MediaPipe's 33
        landmarks and should be used with a compatible backend.
        """

    @abstractmethod
    def close(self) -> None:
        """Release any resources held by the backend."""

    @classmethod
    @abstractmethod
    def from_config(cls, options: dict[str, Any]) -> "PoseBackend":
        """Construct a backend from a config-options dict."""
