"""Train a pose classifier from a dataset of labeled landmark arrays.

Dataset format
--------------
A directory containing one subdirectory per pose class. Each
subdirectory contains either:

- ``.npy`` files with shape (33, 3) — one landmark array per sample, or
- ``.jpg``/``.png`` images that the script will run through MediaPipe to
  extract landmarks on the fly.

The output is a pickled scikit-learn classifier written to the path in
the config (default ``models/knn_classifier.pkl``).

This is intentionally a CPU-only, sklearn-based script — it runs fine on
Fox compute nodes without needing GPU. Upgrade to a PyTorch MLP later if
accuracy plateaus.

Example
-------
    python scripts/train_classifier.py \\
        --dataset data/landmarks_dataset \\
        --output models/knn_classifier.pkl \\
        --n-neighbors 5
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def load_dataset(dataset_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    X: list[np.ndarray] = []
    y: list[str] = []

    backend = None  # lazy init — only needed if we see images

    for pose_dir in sorted(dataset_dir.iterdir()):
        if not pose_dir.is_dir():
            continue
        label = pose_dir.name
        for file in sorted(pose_dir.iterdir()):
            ext = file.suffix.lower()
            if ext == ".npy":
                arr = np.load(file)
                X.append(arr.reshape(-1))
                y.append(label)
            elif ext in IMAGE_EXTS:
                if backend is None:
                    from src.backends.factory import build_backend

                    backend = build_backend("mediapipe", {"model_complexity": 2})
                import cv2

                frame = cv2.imread(str(file))
                if frame is None:
                    continue
                lm = backend.extract_landmarks(frame)
                if lm is None:
                    continue
                X.append(lm.reshape(-1))
                y.append(label)

    if backend is not None:
        backend.close()

    return np.array(X, dtype=np.float32), np.array(y)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--output", default="models/knn_classifier.pkl")
    p.add_argument("--n-neighbors", type=int, default=5)
    p.add_argument("--test-split", type=float, default=0.2)
    p.add_argument("--random-seed", type=int, default=42)
    args = p.parse_args()

    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier

    print(f"Loading dataset from {args.dataset}...")
    X, y = load_dataset(Path(args.dataset))
    print(f"  {len(X)} samples across {len(set(y))} classes")

    if len(X) == 0:
        raise SystemExit("No samples found — aborting")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_split, random_state=args.random_seed, stratify=y
    )

    clf = KNeighborsClassifier(n_neighbors=args.n_neighbors)
    clf.fit(X_train, y_train)

    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)
    print(f"Train accuracy: {train_acc:.3f}")
    print(f"Test accuracy:  {test_acc:.3f}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(clf, f)
    print(f"Model written to {out_path}")


if __name__ == "__main__":
    main()
