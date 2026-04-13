"""Tests for DeviationEstimator."""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.deviation_estimator import DeviationEstimator
from src.utils import LANDMARK


def _make_landmarks_with_right_angle_elbow():
    lm = np.zeros((33, 3), dtype=np.float32)
    lm[LANDMARK["left_shoulder"]] = [0.0, 0.0, 0.0]
    lm[LANDMARK["left_elbow"]] = [1.0, 0.0, 0.0]
    lm[LANDMARK["left_wrist"]] = [1.0, 1.0, 0.0]
    return lm


def test_zero_deviation_when_matching():
    lm = _make_landmarks_with_right_angle_elbow()
    estimator = DeviationEstimator(
        templates={"pose_a": {"left_elbow": 90.0}},
    )
    dev = estimator.compute_deviation("pose_a", lm)
    assert math.isclose(dev, 0.0, abs_tol=1e-4)


def test_nonzero_deviation_when_mismatched():
    lm = _make_landmarks_with_right_angle_elbow()
    estimator = DeviationEstimator(
        templates={"pose_a": {"left_elbow": 45.0}},
    )
    dev = estimator.compute_deviation("pose_a", lm)
    assert math.isclose(dev, 45.0, abs_tol=1e-4)


def test_missing_template_returns_nan():
    lm = _make_landmarks_with_right_angle_elbow()
    estimator = DeviationEstimator(templates={})
    assert math.isnan(estimator.compute_deviation("unknown_pose", lm))


def test_joint_deviations_keys():
    lm = _make_landmarks_with_right_angle_elbow()
    estimator = DeviationEstimator(
        templates={"pose_a": {"left_elbow": 80.0}},
    )
    per_joint = estimator.joint_deviations("pose_a", lm)
    assert "left_elbow" in per_joint
    assert math.isclose(per_joint["left_elbow"], 10.0, abs_tol=1e-4)
