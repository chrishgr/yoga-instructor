"""Unified video source — wraps webcam, video file, and image directory
behind a single iterator interface.

This is what makes local testing against recorded videos trivial: the
rest of the pipeline doesn't know or care whether frames are coming from
a camera or a file on disk.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np


class VideoSource:
    """Iterable video source. Use as a context manager."""

    def __init__(
        self,
        source_type: str = "webcam",
        device: int = 0,
        path: str | None = None,
        target_fps: int | None = None,
    ) -> None:
        self.source_type = source_type
        self.device = device
        self.path = path
        self.target_fps = target_fps
        self._cap = None
        self._image_paths: list[Path] = []
        self._image_idx = 0

    # -- Context manager ---------------------------------------------

    def __enter__(self) -> "VideoSource":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # -- Lifecycle ---------------------------------------------------

    def open(self) -> None:
        import cv2

        if self.source_type == "webcam":
            self._cap = cv2.VideoCapture(self.device)
            if not self._cap.isOpened():
                raise RuntimeError(f"Could not open webcam device {self.device}")
        elif self.source_type == "file":
            if self.path is None:
                raise ValueError("source_type='file' requires path")
            if not Path(self.path).exists():
                raise FileNotFoundError(self.path)
            self._cap = cv2.VideoCapture(self.path)
            if not self._cap.isOpened():
                raise RuntimeError(f"Could not open video file {self.path}")
        elif self.source_type == "image_dir":
            if self.path is None:
                raise ValueError("source_type='image_dir' requires path")
            d = Path(self.path)
            if not d.is_dir():
                raise NotADirectoryError(self.path)
            exts = {".jpg", ".jpeg", ".png", ".bmp"}
            self._image_paths = sorted(
                p for p in d.iterdir() if p.suffix.lower() in exts
            )
            self._image_idx = 0
        else:
            raise ValueError(f"Unknown source_type: {self.source_type!r}")

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    # -- Iteration ---------------------------------------------------

    def frames(self) -> Iterator[np.ndarray]:
        """Yield frames as BGR numpy arrays."""
        import cv2

        if self.source_type in {"webcam", "file"}:
            assert self._cap is not None
            while True:
                ok, frame = self._cap.read()
                if not ok:
                    break
                yield frame
        elif self.source_type == "image_dir":
            for path in self._image_paths:
                frame = cv2.imread(str(path))
                if frame is not None:
                    yield frame

    # -- Metadata ----------------------------------------------------

    @property
    def fps(self) -> float:
        import cv2

        if self._cap is None:
            return float(self.target_fps or 30)
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        return fps if fps > 0 else float(self.target_fps or 30)
