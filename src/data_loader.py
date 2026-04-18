"""Data loading for neuroai-eeg-emotion.

Provides three entry points:
    - generate_synthetic_data(): pipeline-testing data, zero external deps
    - load_seed_data(): SEED Differential Entropy (DE) features from .mat files
    - load_data(): unified interface returning torch tensors with stratified split

SEED dataset layout (expected under data_path):
    ExtractedFeatures/
        1_20131027.mat, 2_20140404.mat, ...   # subject_date.mat per session
        label.mat                              # 15 labels in {-1, 0, +1}

Each session .mat contains per-trial DE feature arrays (keys typically named
``de_LDS{i}`` for i=1..15) with shape (62, N_windows, 5) or (N_windows, 310).
Exact keys and shapes are inspected at runtime — no hardcoded assumptions.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional, Tuple

import h5py
import numpy as np
import scipy.io
import torch
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

N_FEATURES_DEFAULT = 310   # 62 EEG channels x 5 frequency bands
N_CLASSES_DEFAULT = 3
TRIALS_PER_SESSION = 15    # SEED: 15 film clips per session

# SEED labels are stored as {-1, 0, +1}; remap to {0, 1, 2} for CrossEntropyLoss.
SEED_LABEL_MAP = {-1: 0, 0: 1, 1: 2}


def generate_synthetic_data(
    n_samples: int = 3000,
    n_features: int = N_FEATURES_DEFAULT,
    n_classes: int = N_CLASSES_DEFAULT,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic data with the same interface as load_seed_data.

    Random features in [0, 1] with balanced labels. Intended for pipeline
    validation with no external dependencies. Expected downstream accuracy
    is ~1/n_classes (random chance) — this is correct behavior, not a bug.

    Args:
        n_samples: total samples across all classes.
        n_features: feature dimension (default 310 = 62 channels * 5 bands).
        n_classes: number of emotion classes (default 3).
        seed: random seed for reproducibility.

    Returns:
        X of shape (n_samples, n_features), float32 in [0, 1].
        y of shape (n_samples,), int64 in {0, ..., n_classes - 1}.
    """
    rng = np.random.default_rng(seed)
    per_class = n_samples // n_classes
    total = per_class * n_classes

    X = rng.uniform(0.0, 1.0, size=(total, n_features)).astype(np.float32)
    y = np.repeat(np.arange(n_classes, dtype=np.int64), per_class)

    perm = rng.permutation(total)
    return X[perm], y[perm]


def _loadmat_any(path: Path) -> dict:
    """Load a .mat file, falling back to h5py for MATLAB v7.3 (HDF5) format."""
    try:
        raw = scipy.io.loadmat(str(path))
        return {k: v for k, v in raw.items() if not k.startswith("__")}
    except NotImplementedError:
        logger.debug("scipy could not read %s; falling back to h5py", path.name)
        out: dict = {}
        with h5py.File(path, "r") as f:
            for key in f.keys():
                arr = np.array(f[key])
                # h5py returns column-major; transpose to match scipy convention.
                out[key] = arr.T if arr.ndim > 1 else arr
        return out


def _reshape_trial(arr: np.ndarray, n_features: int = N_FEATURES_DEFAULT) -> np.ndarray:
    """Coerce a trial's DE array to shape (N_windows, n_features).

    SEED publishes DE features as (62, N_windows, 5). Some redistributions use
    (N_windows, 62, 5) or already-flattened (N_windows, 310). We handle all.
    """
    if arr.ndim == 2:
        if arr.shape[1] == n_features:
            return arr.astype(np.float32)
        if arr.shape[0] == n_features:
            return arr.T.astype(np.float32)
        raise ValueError(f"Unexpected 2D DE shape {arr.shape}; cannot coerce to (_, {n_features})")

    if arr.ndim == 3:
        shape = arr.shape
        # Identify which axis is 62 (channels) and which is 5 (bands).
        try:
            ch_axis = shape.index(62)
            band_axis = shape.index(5)
        except ValueError as exc:
            raise ValueError(f"Unexpected 3D DE shape {shape}; expected axes of size 62 and 5") from exc
        time_axis = ({0, 1, 2} - {ch_axis, band_axis}).pop()
        # Move time axis to front, then flatten (channels, bands) into features.
        arr = np.moveaxis(arr, time_axis, 0)
        # After moveaxis, remaining axes order is preserved; ensure (N, 62, 5) then flatten.
        if arr.shape[1] != 62:
            arr = np.swapaxes(arr, 1, 2)
        return arr.reshape(arr.shape[0], -1).astype(np.float32)

    raise ValueError(f"Unexpected DE array ndim={arr.ndim}, shape={arr.shape}")


def _extract_trials(mat: dict) -> list[np.ndarray]:
    """Return per-trial DE matrices (N_windows, 310) from a session .mat dict.

    SEED ExtractedFeatures files contain several DE variants per trial
    (``de_LDS{i}`` and ``de_movingAve{i}``) plus non-DE features (asm, dasm,
    dcau, psd, rasm). We select ``de_LDS{i}`` — the Linear Dynamical System
    smoothed DE feature canonical to Zheng & Lu (2015). Fallback: ``de_movingAve``
    if a file lacks LDS keys.
    """
    lds_pattern = re.compile(r"^de_LDS(\d+)$")
    mav_pattern = re.compile(r"^de_movingAve(\d+)$")

    def _collect(pattern: re.Pattern) -> list[Tuple[int, str]]:
        return [(int(m.group(1)), k) for k in mat if (m := pattern.match(k))]

    hits = _collect(lds_pattern) or _collect(mav_pattern)
    if not hits:
        raise KeyError(
            f"No de_LDS* or de_movingAve* trial keys found. "
            f"Session keys: {sorted(k for k in mat if k.startswith('de_'))[:6]}..."
        )

    hits.sort(key=lambda t: t[0])
    return [_reshape_trial(mat[key]) for _, key in hits]


def _load_labels(data_path: Path, label_path: Optional[Path]) -> np.ndarray:
    """Load SEED label.mat and return a (15,) array in {-1, 0, +1}."""
    candidates = []
    if label_path is not None:
        candidates.append(Path(label_path))
    candidates += [data_path / "label.mat", data_path.parent / "label.mat"]

    for cand in candidates:
        if cand.exists():
            mat = _loadmat_any(cand)
            for key in ("label", "labels"):
                if key in mat:
                    labels = np.array(mat[key]).squeeze().astype(np.int64)
                    if labels.size != TRIALS_PER_SESSION:
                        logger.warning("label.mat has %d entries, expected %d", labels.size, TRIALS_PER_SESSION)
                    return labels
            raise KeyError(f"label.mat at {cand} has no 'label' key. Keys: {list(mat.keys())}")

    raise FileNotFoundError(
        f"label.mat not found in {data_path} or its parent. Pass label_path explicitly."
    )


def load_seed_data(
    data_path: str | Path,
    label_path: Optional[str | Path] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load SEED DE features from a directory of session .mat files.

    Args:
        data_path: directory containing session .mat files (e.g. ExtractedFeatures/).
        label_path: optional explicit path to label.mat.

    Returns:
        X of shape (total_windows, 310), float32.
        y of shape (total_windows,), int64 in {0, 1, 2}.
    """
    data_path = Path(data_path)
    if not data_path.is_dir():
        raise NotADirectoryError(f"SEED data_path {data_path} is not a directory")

    session_files = sorted(
        p for p in data_path.glob("*.mat") if p.name.lower() != "label.mat"
    )
    if not session_files:
        raise FileNotFoundError(f"No session .mat files in {data_path}")

    labels_raw = _load_labels(data_path, Path(label_path) if label_path else None)
    labels_mapped = np.array([SEED_LABEL_MAP[int(v)] for v in labels_raw], dtype=np.int64)

    all_X: list[np.ndarray] = []
    all_y: list[np.ndarray] = []

    for i, path in enumerate(session_files):
        mat = _loadmat_any(path)
        if i == 0:
            logger.info("First session keys: %s", sorted(mat.keys()))
        trials = _extract_trials(mat)
        if len(trials) != TRIALS_PER_SESSION:
            logger.warning(
                "%s: found %d trials, expected %d — aligning by position",
                path.name, len(trials), TRIALS_PER_SESSION,
            )
        for trial_idx, trial in enumerate(trials):
            if trial_idx >= len(labels_mapped):
                break
            all_X.append(trial)
            all_y.append(np.full(trial.shape[0], labels_mapped[trial_idx], dtype=np.int64))

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)

    logger.info(
        "Loaded SEED: %d sessions, %d samples, feature_dim=%d, class_counts=%s",
        len(session_files), X.shape[0], X.shape[1], np.bincount(y).tolist(),
    )
    return X, y


def load_data(
    use_synthetic: bool = False,
    data_path: str | Path = "data/",
    test_size: float = 0.2,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load data and return a stratified train/test split as torch tensors.

    Args:
        use_synthetic: if True, skip SEED and use generate_synthetic_data().
        data_path: directory holding SEED ExtractedFeatures (ignored if synthetic).
        test_size: fraction for the test split (default 0.2 → 80/20).
        seed: random seed for reproducibility.

    Returns:
        (X_train, X_test, y_train, y_test) as torch.float32 / torch.long tensors.
    """
    if use_synthetic:
        X, y = generate_synthetic_data(seed=seed)
    else:
        X, y = load_seed_data(data_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed,
    )

    return (
        torch.from_numpy(X_train).float(),
        torch.from_numpy(X_test).float(),
        torch.from_numpy(y_train).long(),
        torch.from_numpy(y_test).long(),
    )


def _smoke_test() -> None:
    """CLI smoke test: load synthetic data and print checkpoint artifacts."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    X_train, X_test, y_train, y_test = load_data(use_synthetic=True)

    print("=" * 60)
    print("Phase 2 smoke test — synthetic data")
    print("=" * 60)
    print(f"X_train: shape={tuple(X_train.shape)}, dtype={X_train.dtype}")
    print(f"X_test:  shape={tuple(X_test.shape)}, dtype={X_test.dtype}")
    print(f"y_train: shape={tuple(y_train.shape)}, dtype={y_train.dtype}")
    print(f"y_test:  shape={tuple(y_test.shape)}, dtype={y_test.dtype}")
    print(f"X value range: [{X_train.min().item():.4f}, {X_train.max().item():.4f}]")
    print(f"Train label distribution: {torch.bincount(y_train).tolist()}")
    print(f"Test  label distribution: {torch.bincount(y_test).tolist()}")
    print(f"Train/test ratio: {X_train.shape[0]} / {X_test.shape[0]} "
          f"= {X_train.shape[0] / (X_train.shape[0] + X_test.shape[0]):.2f} train")

    assert X_train.shape[1] == N_FEATURES_DEFAULT
    assert set(y_train.unique().tolist()) <= {0, 1, 2}
    assert X_train.min() >= 0.0 and X_train.max() <= 1.0
    print("\nAll assertions passed.")


if __name__ == "__main__":
    _smoke_test()
