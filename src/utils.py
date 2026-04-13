"""Shared utilities — primarily geometric helpers used by classifier and
deviation estimator."""
from __future__ import annotations

import numpy as np


def angle_between(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Return the angle at vertex ``b`` formed by the segments ``ba`` and ``bc``.

    Parameters
    ----------
    a, b, c : np.ndarray
        2D or 3D coordinates. ``b`` is the vertex; ``a`` and ``c`` are the
        other two points.

    Returns
    -------
    float
        Angle in degrees in the range [0, 180].
    """
    ba = a - b
    bc = c - b
    norm = np.linalg.norm(ba) * np.linalg.norm(bc)
    if norm < 1e-9:
        return 0.0
    cos_angle = np.dot(ba, bc) / norm
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


# MediaPipe Pose landmark indices (33 total). Kept here so every module
# refers to joints by name rather than raw integers.
LANDMARK = {
    "nose": 0,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
}

# Joint triplets used for angle-based features. Each entry is
# (joint_name, (point_a, vertex_b, point_c)).
JOINT_TRIPLETS = {
    "left_elbow": ("left_shoulder", "left_elbow", "left_wrist"),
    "right_elbow": ("right_shoulder", "right_elbow", "right_wrist"),
    "left_shoulder": ("left_elbow", "left_shoulder", "left_hip"),
    "right_shoulder": ("right_elbow", "right_shoulder", "right_hip"),
    "left_hip": ("left_shoulder", "left_hip", "left_knee"),
    "right_hip": ("right_shoulder", "right_hip", "right_knee"),
    "left_knee": ("left_hip", "left_knee", "left_ankle"),
    "right_knee": ("right_hip", "right_knee", "right_ankle"),
}


def compute_joint_angles(landmarks: np.ndarray) -> dict[str, float]:
    """Compute all joint angles defined in ``JOINT_TRIPLETS``.

    Parameters
    ----------
    landmarks : np.ndarray
        Array of shape (33, 3) with MediaPipe Pose landmarks (x, y, z).

    Returns
    -------
    dict[str, float]
        Mapping from joint name to angle in degrees.
    """
    angles: dict[str, float] = {}
    for joint, (a_name, b_name, c_name) in JOINT_TRIPLETS.items():
        a = landmarks[LANDMARK[a_name]]
        b = landmarks[LANDMARK[b_name]]
        c = landmarks[LANDMARK[c_name]]
        angles[joint] = angle_between(a, b, c)
    return angles
