"""coachNasarus — main entry point.

Binds the pose classifier, tracker, deviation estimator, and audio
feedback together into a single loop. Reads frames from whatever
``VideoSource`` the config specifies (webcam, file, or image directory)
so the same script works for live yoga and for analyzing recorded
videos.

Usage
-----
    python main.py                                  # default config
    python main.py --config config/local_hq.yaml
    python main.py --video data/sample_videos/x.mp4
    python main.py --no-audio
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import yaml

from src.audio_feedback import AudioFeedback
from src.backends.factory import build_backend
from src.deviation_estimator import DeviationEstimator
from src.pose_classifier import PoseClassifier, UNKNOWN
from src.pose_tracker import PoseTracker
from src.video_source import VideoSource


def load_config(path: str) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="coachNasarus yoga coach")
    p.add_argument(
        "--config",
        default="config/local.yaml",
        help="Path to YAML config file",
    )
    p.add_argument(
        "--video",
        default=None,
        help="Override: read from a video file instead of webcam",
    )
    p.add_argument(
        "--image-dir",
        default=None,
        help="Override: read from a directory of still images",
    )
    p.add_argument(
        "--no-audio",
        action="store_true",
        help="Disable audio feedback",
    )
    p.add_argument(
        "--no-display",
        action="store_true",
        help="Disable the OpenCV preview window (useful for headless runs)",
    )
    p.add_argument(
        "--session-log",
        default=None,
        help="Write session summary JSON to this path on exit",
    )
    return p.parse_args()


def draw_overlay(
    frame,
    pose_name: str | None,
    hold_seconds: float,
    deviation: float,
    joint_deviations: dict[str, float],
) -> None:
    """Draw pose name, hold time, and deviation on the frame."""
    import cv2

    h, w = frame.shape[:2]
    pad = 10
    y = 30

    def put(text: str, color=(255, 255, 255)) -> None:
        nonlocal y
        cv2.putText(
            frame,
            text,
            (pad, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )
        y += 28

    put(f"Pose: {pose_name or '-'}")
    put(f"Hold: {hold_seconds:5.1f}s")
    if deviation == deviation:  # NaN check
        color = (0, 255, 0) if deviation < 10 else (0, 200, 255) if deviation < 25 else (0, 0, 255)
        put(f"Deviation: {deviation:5.1f} deg", color)
    else:
        put("Deviation: n/a")

    # Worst-offending joints, if any.
    if joint_deviations:
        worst = sorted(joint_deviations.items(), key=lambda kv: -kv[1])[:3]
        for joint, dev in worst:
            put(f"  {joint}: {dev:4.1f}", (0, 200, 255))


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    # -- CLI overrides on top of YAML --------------------------------
    vs_cfg = cfg.get("video_source", {})
    if args.video:
        vs_cfg = {"type": "file", "path": args.video}
    elif args.image_dir:
        vs_cfg = {"type": "image_dir", "path": args.image_dir}

    audio_cfg = cfg.get("audio", {})
    if args.no_audio:
        audio_cfg = {**audio_cfg, "enabled": False}

    # -- Build components --------------------------------------------
    backend = build_backend(cfg["backend"], cfg.get("backend_options", {}))
    classifier_cfg = cfg.get("classifier", {})
    classifier = PoseClassifier(
        backend=backend,
        classifier_type=classifier_cfg.get("type", "rule_based"),
        model_path=classifier_cfg.get("model_path"),
    )

    tracker_cfg = cfg.get("tracker", {})
    tracker = PoseTracker(
        min_hold_seconds=tracker_cfg.get("min_hold_seconds", 1.0),
        smoothing_window=tracker_cfg.get("smoothing_window", 5),
    )

    dev_cfg = cfg.get("deviation", {})
    estimator = DeviationEstimator.from_dir(
        dev_cfg.get("templates_dir", "templates/"),
        metric=dev_cfg.get("metric", "angle"),
    )

    audio = AudioFeedback(
        base_frequency=audio_cfg.get("base_frequency", 220.0),
        max_frequency=audio_cfg.get("max_frequency", 880.0),
        enabled=audio_cfg.get("enabled", True),
    )
    audio.start()

    source = VideoSource(
        source_type=vs_cfg.get("type", "webcam"),
        device=vs_cfg.get("device", 0),
        path=vs_cfg.get("path"),
        target_fps=vs_cfg.get("target_fps", 30),
    )

    display_cfg = cfg.get("display", {})
    show_display = not args.no_display and display_cfg.get("show_overlay", True)
    window_title = display_cfg.get("window_title", "coachNasarus")

    # -- Main loop ---------------------------------------------------
    import cv2

    start_time = time.monotonic()
    last_timestamp = start_time

    try:
        with source:
            for frame in source.frames():
                now = time.monotonic()
                last_timestamp = now

                landmarks = classifier.extract_landmarks(frame)
                if landmarks is None:
                    pose_name = UNKNOWN
                    deviation = float("nan")
                    joint_devs: dict[str, float] = {}
                else:
                    pose_name = classifier.classify(landmarks)
                    tracker.update(pose_name, now)
                    if estimator.has_template(pose_name):
                        deviation = estimator.compute_deviation(pose_name, landmarks)
                        joint_devs = estimator.joint_deviations(pose_name, landmarks)
                    else:
                        deviation = float("nan")
                        joint_devs = {}

                audio.update(deviation)

                if show_display:
                    current = tracker.current_pose()
                    hold = tracker.current_hold_duration(now)
                    draw_overlay(frame, current, hold, deviation, joint_devs)
                    cv2.imshow(window_title, frame)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        break
    finally:
        tracker.finalize(last_timestamp)
        audio.stop()
        classifier.close()
        if show_display:
            cv2.destroyAllWindows()

        summary = tracker.session_summary()
        print("\n=== Session summary ===")
        if summary:
            for pose, seconds in sorted(summary.items(), key=lambda kv: -kv[1]):
                print(f"  {pose:30s} {seconds:6.1f} s")
        else:
            print("  (no poses held long enough to log)")

        if args.session_log:
            log_path = Path(args.session_log)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "w") as f:
                json.dump(
                    {
                        "summary": summary,
                        "history": [
                            {
                                "pose": interval.pose,
                                "start": interval.start,
                                "end": interval.end,
                                "duration": interval.duration,
                            }
                            for interval in tracker.history()
                        ],
                    },
                    f,
                    indent=2,
                )
            print(f"\nSession log written to {log_path}")


if __name__ == "__main__":
    main()
