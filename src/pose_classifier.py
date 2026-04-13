"""Pose classifier — extracts landmarks via a backend and maps them to a
pose label.

Phase 1 ships a rule-based classifier over joint angles for a small set
of starter poses. The same interface can later be backed by a kNN or MLP
trained on landmark vectors — the ``type`` field in the config selects
which.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .backends.base import PoseBackend
from .utils import compute_joint_angles


UNKNOWN = "unknown"


@dataclass
class AngleRule:
    """A single rule: the named joint must be within [min, max] degrees."""

    joint: str
    min_deg: float
    max_deg: float

    def matches(self, angles: dict[str, float]) -> bool:
        val = angles.get(self.joint)
        if val is None:
            return False
        return self.min_deg <= val <= self.max_deg


# Starter rule set — deliberately loose so the classifier fires reliably
# in Phase 1. Tighten once real reference data is collected.
RULE_BASED_POSES: dict[str, list[AngleRule]] = {
    "tadasana": [
        AngleRule("left_knee", 160, 185),
        AngleRule("right_knee", 160, 185),
        AngleRule("left_hip", 160, 185),
        AngleRule("right_hip", 160, 185),
    ],
    "vrikshasana": [
        # One leg straight, the other bent sharply — either side.
        AngleRule("left_knee", 160, 185),
        AngleRule("right_knee", 20, 90),
    ],
    "adho_mukha_svanasana": [
        # Downward dog: hips flexed, arms and legs roughly straight.
        AngleRule("left_knee", 150, 185),
        AngleRule("right_knee", 150, 185),
        AngleRule("left_hip", 60, 110),
        AngleRule("right_hip", 60, 110),
    ],
}


class PoseClassifier:
    """High-level interface used by ``main.py``.

    Combines a :class:`PoseBackend` (landmark extraction) with a
    classification strategy (rule-based, kNN, or MLP).
    """

    def __init__(
        self,
        backend: PoseBackend,
        classifier_type: str = "rule_based",
        model_path: str | None = None,
    ) -> None:
        self.backend = backend
        self.classifier_type = classifier_type
        self._model: Any = None
        if classifier_type in {"knn", "mlp"}:
            if model_path is None:
                raise ValueError(
                    f"classifier_type={classifier_type!r} requires model_path"
                )
            self._model = self._load_model(model_path)

    @staticmethod
    def _load_model(path: str) -> Any:
        import pickle

        with open(path, "rb") as f:
            return pickle.load(f)

    # -- Public API --------------------------------------------------

    def extract_landmarks(self, frame: np.ndarray) -> np.ndarray | None:
        return self.backend.extract_landmarks(frame)

    def classify(self, landmarks: np.ndarray) -> str:
        if self.classifier_type == "rule_based":
            return self._classify_rule_based(landmarks)
        if self.classifier_type in {"knn", "mlp"}:
            return self._classify_model(landmarks)
        raise ValueError(f"Unknown classifier_type: {self.classifier_type!r}")

    def close(self) -> None:
        self.backend.close()

    # -- Strategies --------------------------------------------------

    def _classify_rule_based(self, landmarks: np.ndarray) -> str:
        angles = compute_joint_angles(landmarks)
        for pose_name, rules in RULE_BASED_POSES.items():
            if all(rule.matches(angles) for rule in rules):
                return pose_name
        return UNKNOWN

    def _classify_model(self, landmarks: np.ndarray) -> str:
        # Flatten (33, 3) -> (99,) for the model.
        features = landmarks.reshape(1, -1)
        return str(self._model.predict(features)[0])
