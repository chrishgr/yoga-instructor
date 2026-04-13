"""Tests for geometric utilities."""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import angle_between, compute_joint_angles, LANDMARK


def test_right_angle():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 0.0, 0.0])
    c = np.array([0.0, 1.0, 0.0])
    assert math.isclose(angle_between(a, b, c), 90.0, abs_tol=1e-6)


def test_straight_line():
    a = np.array([-1.0, 0.0, 0.0])
    b = np.array([0.0, 0.0, 0.0])
    c = np.array([1.0, 0.0, 0.0])
    assert math.isclose(angle_between(a, b, c), 180.0, abs_tol=1e-6)


def test_zero_vector_safe():
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([0.0, 0.0, 0.0])
    c = np.array([1.0, 0.0, 0.0])
    # Should not raise; returns 0 by convention.
    assert angle_between(a, b, c) == 0.0


def test_compute_joint_angles_shape():
    lm = np.zeros((33, 3))
    # Put a few landmarks in non-degenerate positions.
    lm[LANDMARK["left_shoulder"]] = [0.0, 0.0, 0.0]
    lm[LANDMARK["left_elbow"]] = [1.0, 0.0, 0.0]
    lm[LANDMARK["left_wrist"]] = [1.0, 1.0, 0.0]
    angles = compute_joint_angles(lm)
    assert "left_elbow" in angles
    assert math.isclose(angles["left_elbow"], 90.0, abs_tol=1e-6)
