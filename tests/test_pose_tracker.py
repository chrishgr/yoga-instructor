"""Tests for PoseTracker — these are the most important unit tests
because the hysteresis logic is easy to get wrong.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pose_tracker import PoseTracker


def test_single_pose_held_long_enough():
    t = PoseTracker(min_hold_seconds=1.0, smoothing_window=3)
    for i in range(10):
        t.update("tadasana", float(i) * 0.2)
    t.finalize(2.0)
    summary = t.session_summary()
    assert "tadasana" in summary
    assert summary["tadasana"] > 0.5


def test_short_pose_not_logged():
    t = PoseTracker(min_hold_seconds=2.0, smoothing_window=3)
    for i in range(5):
        t.update("tadasana", float(i) * 0.1)
    t.finalize(0.5)
    assert t.session_summary() == {}


def test_single_noisy_frame_does_not_switch():
    t = PoseTracker(min_hold_seconds=0.1, smoothing_window=5)
    # Establish tadasana.
    for i in range(10):
        t.update("tadasana", float(i) * 0.1)
    # One noisy frame — should be absorbed.
    t.update("vrikshasana", 1.05)
    for i in range(11, 20):
        t.update("tadasana", float(i) * 0.1)
    t.finalize(2.0)
    summary = t.session_summary()
    assert "tadasana" in summary
    assert "vrikshasana" not in summary


def test_real_transition_is_detected():
    t = PoseTracker(min_hold_seconds=0.2, smoothing_window=3)
    for i in range(10):
        t.update("tadasana", float(i) * 0.1)
    for i in range(10, 20):
        t.update("vrikshasana", float(i) * 0.1)
    t.finalize(2.0)
    summary = t.session_summary()
    assert "tadasana" in summary
    assert "vrikshasana" in summary
