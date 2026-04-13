"""Build reference templates from images or videos.

Each subdirectory of the input dir is treated as one pose class, and the
script extracts landmarks from every image/frame inside and writes a
template JSON to ``templates/<pose_name>.json`` containing averaged
joint angles.

Example
-------
    python scripts/build_templates.py \\
        --input reference_data/ \\
        --output templates/

Where ``reference_data/`` looks like::

    reference_data/
    ├── tadasana/
    │   ├── 01.jpg
    │   └── 02.jpg
    └── vrikshasana/
        └── pose.mp4
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Make `src` importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.backends.factory import build_backend
from src.utils import compute_joint_angles


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}


def iter_frames(path: Path):
    import cv2

    if path.suffix.lower() in IMAGE_EXTS:
        frame = cv2.imread(str(path))
        if frame is not None:
            yield frame
    elif path.suffix.lower() in VIDEO_EXTS:
        cap = cv2.VideoCapture(str(path))
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
        cap.release()


def build_one_template(pose_dir: Path, backend) -> dict[str, float] | None:
    angle_samples: list[dict[str, float]] = []
    for file in sorted(pose_dir.iterdir()):
        if file.suffix.lower() not in IMAGE_EXTS | VIDEO_EXTS:
            continue
        for frame in iter_frames(file):
            landmarks = backend.extract_landmarks(frame)
            if landmarks is not None:
                angle_samples.append(compute_joint_angles(landmarks))

    if not angle_samples:
        return None

    joints = angle_samples[0].keys()
    return {
        j: float(np.mean([s[j] for s in angle_samples]))
        for j in joints
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Directory with one subfolder per pose")
    p.add_argument("--output", default="templates/", help="Where to write template JSON files")
    p.add_argument("--model-complexity", type=int, default=2)
    args = p.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    backend = build_backend(
        "mediapipe",
        {"model_complexity": args.model_complexity},
    )

    try:
        for pose_dir in sorted(input_dir.iterdir()):
            if not pose_dir.is_dir():
                continue
            print(f"Processing {pose_dir.name}...")
            template = build_one_template(pose_dir, backend)
            if template is None:
                print(f"  WARNING: no landmarks detected in {pose_dir}")
                continue
            out_path = output_dir / f"{pose_dir.name}.json"
            with open(out_path, "w") as f:
                json.dump({"angles": template}, f, indent=2)
            print(f"  wrote {out_path}")
    finally:
        backend.close()


if __name__ == "__main__":
    main()
