"""Deviation estimator — compares live landmarks against a reference
template and returns both a scalar deviation and a per-joint breakdown.

Two metrics are supported:

``angle``
    Mean absolute difference in joint angles (degrees). Invariant under
    translation, scale, and mirror. Easiest to explain to users.

``procrustes``
    Procrustes distance after alignment. More holistic, but harder to map
    back to a specific joint.

Templates live in ``templates/`` as JSON files, one per pose, each
containing either a landmark array or a precomputed angle dict.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .utils import compute_joint_angles


class DeviationEstimator:
    def __init__(
        self,
        templates: dict[str, dict[str, float]],
        metric: str = "angle",
    ) -> None:
        """
        Parameters
        ----------
        templates
            Mapping from pose name to a dict of reference joint angles
            (joint_name -> degrees).
        metric
            Currently only ``"angle"`` is implemented. ``"procrustes"``
            is reserved for future work.
        """
        self.templates = templates
        self.metric = metric
        if metric not in {"angle", "procrustes"}:
            raise ValueError(f"Unknown deviation metric: {metric!r}")

    # -- Factory -----------------------------------------------------

    @classmethod
    def from_dir(cls, templates_dir: str | Path, metric: str = "angle") -> "DeviationEstimator":
        templates_dir = Path(templates_dir)
        templates: dict[str, dict[str, float]] = {}
        if templates_dir.exists():
            for path in templates_dir.glob("*.json"):
                with open(path) as f:
                    data = json.load(f)
                # Templates can store either angles directly or raw
                # landmarks that we convert on load.
                if "angles" in data:
                    templates[path.stem] = data["angles"]
                elif "landmarks" in data:
                    lm = np.array(data["landmarks"], dtype=np.float32)
                    templates[path.stem] = compute_joint_angles(lm)
                else:
                    raise ValueError(
                        f"Template {path.name} must contain 'angles' or 'landmarks'"
                    )
        return cls(templates, metric=metric)

    # -- Queries -----------------------------------------------------

    def has_template(self, pose_name: str) -> bool:
        return pose_name in self.templates

    def compute_deviation(self, pose_name: str, landmarks: np.ndarray) -> float:
        """Return a scalar deviation in degrees (mean absolute joint error).

        Returns ``float('nan')`` if no template is registered for the pose.
        """
        if pose_name not in self.templates:
            return float("nan")
        user_angles = compute_joint_angles(landmarks)
        ref_angles = self.templates[pose_name]
        diffs = [
            abs(user_angles[j] - ref_angles[j])
            for j in ref_angles
            if j in user_angles
        ]
        if not diffs:
            return float("nan")
        return float(np.mean(diffs))

    def joint_deviations(
        self, pose_name: str, landmarks: np.ndarray
    ) -> dict[str, float]:
        """Return per-joint absolute differences in degrees."""
        if pose_name not in self.templates:
            return {}
        user_angles = compute_joint_angles(landmarks)
        ref_angles = self.templates[pose_name]
        return {
            j: abs(user_angles[j] - ref_angles[j])
            for j in ref_angles
            if j in user_angles
        }
