"""Pose tracker — converts a stream of (possibly noisy) per-frame
classifications into stable pose intervals with durations.

Hysteresis is implemented by a simple majority vote over a sliding
window. Only once the majority label in the window differs from the
currently-active pose do we switch — this absorbs isolated
misclassifications without adding lag to genuine transitions.
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field


@dataclass
class PoseInterval:
    pose: str
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class PoseTracker:
    min_hold_seconds: float = 1.0
    smoothing_window: int = 5

    _window: deque[str] = field(init=False)
    _active_pose: str | None = field(default=None, init=False)
    _active_start: float | None = field(default=None, init=False)
    _history: list[PoseInterval] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self._window = deque(maxlen=self.smoothing_window)

    # -- Main update loop --------------------------------------------

    def update(self, predicted_pose: str, timestamp: float) -> None:
        """Feed one frame's prediction into the tracker."""
        self._window.append(predicted_pose)
        smoothed = self._majority_vote()

        if smoothed is None:
            return

        if self._active_pose is None:
            self._active_pose = smoothed
            self._active_start = timestamp
            return

        if smoothed != self._active_pose:
            # Close the previous interval — but only record it if it met
            # the minimum hold threshold. Short intervals are likely
            # transitions and don't count as "holding" a pose.
            assert self._active_start is not None
            duration = timestamp - self._active_start
            if duration >= self.min_hold_seconds:
                self._history.append(
                    PoseInterval(
                        pose=self._active_pose,
                        start=self._active_start,
                        end=timestamp,
                    )
                )
            self._active_pose = smoothed
            self._active_start = timestamp

    def finalize(self, timestamp: float) -> None:
        """Close any currently-open interval. Call at session end."""
        if self._active_pose is not None and self._active_start is not None:
            duration = timestamp - self._active_start
            if duration >= self.min_hold_seconds:
                self._history.append(
                    PoseInterval(
                        pose=self._active_pose,
                        start=self._active_start,
                        end=timestamp,
                    )
                )
            self._active_pose = None
            self._active_start = None

    # -- Queries -----------------------------------------------------

    def current_pose(self) -> str | None:
        return self._active_pose

    def current_hold_duration(self, now: float) -> float:
        if self._active_start is None:
            return 0.0
        return now - self._active_start

    def session_summary(self) -> dict[str, float]:
        """Total time held, per pose, across the whole session."""
        totals: dict[str, float] = {}
        for interval in self._history:
            totals[interval.pose] = totals.get(interval.pose, 0.0) + interval.duration
        return totals

    def history(self) -> list[PoseInterval]:
        return list(self._history)

    # -- Helpers -----------------------------------------------------

    def _majority_vote(self) -> str | None:
        if not self._window:
            return None
        counts = Counter(self._window)
        label, count = counts.most_common(1)[0]
        # Require a clear majority — more than half of the window.
        if count * 2 > len(self._window):
            return label
        return None
