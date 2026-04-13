"""MediaPipe Pose backend."""
from __future__ import annotations

from typing import Any

import numpy as np

from .base import PoseBackend


class MediaPipeBackend(PoseBackend):
    """Wraps ``mediapipe.solutions.pose.Pose``.

    Returns a (33, 3) array of normalized ``(x, y, z)`` landmarks in image
    coordinates (x, y in [0, 1], z in arbitrary units relative to the hips).
    """

    def __init__(
        self,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        # Import lazily so that unit tests and training scripts that don't
        # need MediaPipe don't have to install it.
        import mediapipe as mp

        self._mp = mp
        self._pose = mp.solutions.pose.Pose(
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def extract_landmarks(self, frame: np.ndarray) -> np.ndarray | None:
        # MediaPipe expects RGB; OpenCV gives BGR.
        import cv2

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._pose.process(rgb)
        if not results.pose_landmarks:
            return None
        lm = results.pose_landmarks.landmark
        return np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)

    def close(self) -> None:
        self._pose.close()

    @classmethod
    def from_config(cls, options: dict[str, Any]) -> "MediaPipeBackend":
        return cls(
            model_complexity=options.get("model_complexity", 1),
            min_detection_confidence=options.get("min_detection_confidence", 0.5),
            min_tracking_confidence=options.get("min_tracking_confidence", 0.5),
        )
