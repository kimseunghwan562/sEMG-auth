from __future__ import annotations

"""Week 2 sEMG pipeline: filter -> crop -> windows -> TD features -> file split -> scale/save.
Run from semg-auth: python .\\week2_h_pipeline.py
"""
import csv
import json
import random
import re
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, welch

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "H"
OUT = ROOT / "week2_results"
FS, NOTCH_HZ, NOTCH_Q = 1000, 60.0, 30.0
LOW_HZ, HIGH_HZ, FILTER_ORDER = 20.0, 450.0, 4
CROP_START, CROP_SAMPLES = 1500, 2000
WIN, HOP, ZC_SSC_THRESHOLD = 200, 100, 0.01
SEED, TEST_SIZE = 42, 0.2
CHANNELS = [f"bandpass {i}" for i in range(1, 6)]
GESTURES = ["extor", "fist", "open", "pro"]
CLASSES = [f"{angle}_{g}" for angle in (30, 60) for g in GESTURES]
FEATURES = ["RMS", "MAV", "WL", "ZC", "SSC"]


def class_of(path: Path) -> str:
    m = re.fullmatch(r"[a-h]_(30|60)\s+(extor|fist|open|pro)", path.stem, re.I)
    if not m:
        raise ValueError(f"Unexpected filename: {path.name}")
    return f"{m.group(1)}_{m.group(2).lower()}"


def read_record(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        header = next(csv.reader(f))
    x = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64)
    need = CHANNELS + ["time", "mark"]
    if any(c not in header for c in need):
        raise ValueError(f"Required columns {need} not found in {path.name}: {header}")
    return header, x


def preprocess(x: np.ndarray) -> np.ndarray:
    b, a = iirnotch(NOTCH_HZ, NOTCH_Q, fs=FS)
    y = filtfilt(b, a, x, axis=0)
    b, a = butter(FILTER_ORDER, [LOW_HZ, HIGH_HZ], btype="bandpass", fs=FS)
    return filtfilt(b, a, y, axis=0)


def td_features(w: np.ndarray, threshold: float = ZC_SSC_THRESHOLD) -> np.ndarray:
    """For (200, 5), return [RMS(5), MAV(5), WL(5), ZC(5), SSC(5)]."""
    d = np.diff(w, axis=0)
    rms = np.sqrt(np.mean(w ** 2, axis=0))
    mav = np.mean(np.abs(w), axis=0)
    wl = np.sum(np.abs(d), axis=0)
    zc = np.sum((w[:-1] * w[1:] < 0) & (np.abs(d) >= threshold), axis=0)
    ssc = np.sum((d[:-1] * d[1:] < 0) & (np.abs(d[:-1]) >= threshold)
                 & (np.abs(d[1:]) >= threshold), axis=0)
    return np.concatenate([rms, mav, wl, zc, ssc]).astype(np.float32)


def split_recordings(files, labels):
    """Stratified 80/20 split at source CSV level, before windows/features."""
    try:
        from sklearn.model_selection import train_test_split
        return train_test_split(files, test_size=TEST_SIZE, random_state=SEED,
                                stratify=labels)
    except (ImportError, OSError) as exc:
        
        rng = random.Random(SEED)
        tr, te = [], []
        for name in CLASSES:
            group = [p for p, y in zip(files, labels) if y == name]
            rng.shuffle(group)
            n_test = round(len(group) * TEST_SIZE)
            te.extend(group[:n_test]); tr.extend(group[n_test:])
        print(f"sklearn split fallback used ({type(exc).__name__}); stratified by class")
        return tr, te


def scale_train_test(Xtr, Xte):
    try:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler().fit(Xtr)
        return scaler.transform(Xtr).astype(np.float32), scaler.transform(Xte).astype(np.float32), scaler.mean_, scaler.scale_, "sklearn StandardScaler"
    except (ImportError, OSError) as exc:
        mean = Xtr.mean(axis=0)
        scale = Xtr.std(axis=0)
        scale[scale == 0] = 1.0
        print(f"StandardScaler fallback used ({type(exc).__name__}); numpy train statistics")
        return ((Xtr-mean)/scale).astype(np.float32), ((Xte-mean)/scale).astype(np.float32), mean, scale, "NumPy equivalent (fit on train only)"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted(DATA_DIR.glob("*.csv"))
    if len(files) != 64:
        raise ValueError(f"Expected 64 CSVs under {DATA_DIR}; got {len(files)}")
    labels = [class_of(p) for p in files]
    train_files, test_files = split_recordings(files, labels)
    print(f"File split first: {len(train_files)} train / {len(test_files)} test")
    print("Class file split:", {c: [sum(class_of(p)==c for p in train_files), sum(class_of(p)==c for p in test_files)] for c in CLASSES})

    Xtr, ytr, Xte, yte = [], [], [], []
    class_reps = Counter(); file_summary = {}; first_example = None; filter_example = None
    rep_interval_samples = []; mark_counts = []; raw_shapes = []
    train_file_names, test_file_names = {p.name for p in train_files}, {p.name for p in test_files}

    for i, path in enumerate(files, 1):
        header, data = read_record(path)
        raw_shapes.append(list(data.shape))
        sig = data[:, [header.index(c) for c in CHANNELS]]
        marks = np.flatnonzero(data[:, header.index("mark")] > 0)
        if len(marks) < 2: raise ValueError(f"Not enough mark events: {path.name}")
        gaps = np.diff(marks); period = int(round(np.median(gaps)))
        if np.max(np.abs(gaps-period)) > 5: raise ValueError(f"Unexpected mark spacing in {path.name}")
        mark_counts.append(len(marks)); rep_interval_samples.extend(gaps.tolist())
        is_train = path.name in train_file_names
        class_name = class_of(path)
        complete = 0
        for rep, mark in enumerate(marks):
            trial_end = int(mark) + period
            crop_a, crop_b = int(mark)+CROP_START, int(mark)+CROP_START+CROP_SAMPLES
            if trial_end > len(sig) or crop_b > trial_end: continue
            filtered = preprocess(sig[int(mark):trial_end])
            segment = filtered[CROP_START:CROP_START+CROP_SAMPLES]
            if segment.shape != (CROP_SAMPLES, 5): raise ValueError(f"Bad crop {segment.shape}")
            if first_example is None:
                first_example = segment.copy()
                filter_example = (sig[int(mark):trial_end, 0].copy(), filtered[:, 0].copy())
            # floor((2000-200)/100)+1 = 19 windows
            feats = [td_features(segment[s:s+WIN]) for s in range(0, len(segment)-WIN+1, HOP)]
            target_X, target_y = (Xtr, ytr) if is_train else (Xte, yte)
            target_X.extend(feats); target_y.extend([class_name]*len(feats))
            complete += 1
        class_reps[class_name] += complete
        file_summary[path.name] = {"class": class_name, "repetitions": complete, "split": "train" if is_train else "test", "mark_events": int(len(marks)), "period_samples": period}
        print(f"[{i:02d}/64] {path.name}: {complete} trials -> {'train' if is_train else 'test'}", flush=True)

    Xtr, ytr, Xte, yte = np.asarray(Xtr, np.float32), np.asarray(ytr), np.asarray(Xte, np.float32), np.asarray(yte)
    if Xtr.ndim != 2 or Xte.ndim != 2 or Xtr.shape[1] != 25 or Xte.shape[1] != 25:
        raise ValueError(f"Unexpected feature matrix dimensions: {Xtr.shape}, {Xte.shape}")
    Xtr_s, Xte_s, mean, scale, scaler_name = scale_train_test(Xtr, Xte)

    feature_names = np.asarray([f"{feature}_ch{ch+1}" for feature in FEATURES for ch in range(5)])
    class_names = np.asarray(CLASSES)
    npz = OUT / "dataset.npz"
    np.savez_compressed(npz, Xtr=Xtr_s, ytr=ytr, Xte=Xte_s, yte=yte,
                        feature_names=feature_names, class_names=class_names,
                        scaler_mean=mean, scaler_scale=scale,
                        train_files=np.asarray(sorted(train_file_names)),
                        test_files=np.asarray(sorted(test_file_names)),
                        sampling_rate_hz=np.asarray(FS), window_samples=np.asarray(WIN), hop_samples=np.asarray(HOP),
                        crop_start_samples=np.asarray(CROP_START), crop_samples=np.asarray(CROP_SAMPLES))

    # Filter spectrum validation: input -> notch + 20-450 Hz bandpass.
    raw, filtered = filter_example
    f0, p0 = welch(raw, fs=FS, nperseg=min(4096, len(raw)))
    f1, p1 = welch(filtered, fs=FS, nperseg=min(4096, len(filtered)))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.semilogy(f0, p0+1e-30, label="Before filters")
    ax.semilogy(f1, p1+1e-30, label="60 Hz notch + 20–450 Hz bandpass")
    for f, label in [(NOTCH_HZ, "60 Hz notch"), (LOW_HZ, "20 Hz"), (HIGH_HZ, "450 Hz")]:
        ax.axvline(f, color="gray", linestyle="--", linewidth=.8, label=label)
    ax.set_xlim(0, 500); ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("Power spectral density")
    ax.set_title("Filter verification: example recording, channel 1"); ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.tight_layout(); filter_png = OUT / "week2_filter_spectrum.png"; fig.savefig(filter_png, dpi=160); plt.close(fig)

    # Crop example, five channels, matching the week 1 signal-visualization habit.
    fig, axes = plt.subplots(5, 1, figsize=(10, 7), sharex=True)
    t = np.arange(CROP_SAMPLES)/FS
    for ch, ax in enumerate(axes):
        ax.plot(t, first_example[:, ch], lw=.55); ax.set_ylabel(f"Ch {ch+1}"); ax.grid(alpha=.15)
    axes[-1].set_xlabel("Time in extracted 2-second segment (s)")
    fig.suptitle("Filtered 2-second sEMG segment (five channels)"); fig.tight_layout()
    crop_png = OUT / "week2_segment_5channels.png"; fig.savefig(crop_png, dpi=140); plt.close(fig)

    # StandardScaler behavior report: train mean ~0, std ~1 (constant dims remain 0).
    train_mean = Xtr_s.mean(axis=0); train_std = Xtr_s.std(axis=0)
    summary = {
        "data_folder": "data/H", "source_file_count": len(files), "class_names": CLASSES,
        "class_file_counts_total": dict(Counter(labels)),
        "train_file_count": len(train_files), "test_file_count": len(test_files),
        "train_files_per_class": {c: sum(class_of(p)==c for p in train_files) for c in CLASSES},
        "test_files_per_class": {c: sum(class_of(p)==c for p in test_files) for c in CLASSES},
        "source_csv_shapes_unique": sorted({tuple(s) for s in raw_shapes}), "signal_columns": CHANNELS,
        "sampling_rate_hz": FS, "mark_events_min_max": [int(min(mark_counts)), int(max(mark_counts))],
        "median_trial_interval_samples": int(np.median(rep_interval_samples)),
        "median_trial_interval_seconds": float(np.median(rep_interval_samples)/FS),
        "filter": {"notch_hz": NOTCH_HZ, "notch_q": NOTCH_Q, "bandpass_hz": [LOW_HZ,HIGH_HZ], "butterworth_order": FILTER_ORDER, "applied_per_marked_trial": True},
        "analysis_crop": {"start_after_mark_samples": CROP_START, "length_samples": CROP_SAMPLES, "duration_seconds": CROP_SAMPLES/FS},
        "sliding_window": {"window_samples": WIN, "window_ms": WIN/FS*1000, "hop_samples": HOP, "hop_ms": HOP/FS*1000, "overlap_percent": 100*(WIN-HOP)/WIN, "windows_per_segment": (CROP_SAMPLES-WIN)//HOP+1},
        "feature_names": feature_names.tolist(), "features_per_window": int(Xtr.shape[1]), "feature_implementation": "NumPy RMS, MAV, WL, ZC, SSC; feature-major then channel order; threshold=0.01",
        "split": "Source CSV files stratified 80/20 before filtering, cropping, windowing and feature extraction; random_state=42", "scaler": scaler_name,
        "train_file_split_per_class": {c: sum(class_of(p)==c for p in train_files) for c in CLASSES},
        "test_file_split_per_class": {c: sum(class_of(p)==c for p in test_files) for c in CLASSES},
        "train_trials": int(sum(v['repetitions'] for k,v in file_summary.items() if v['split']=='train')),
        "test_trials": int(sum(v['repetitions'] for k,v in file_summary.items() if v['split']=='test')),
        "train_feature_shape": list(Xtr_s.shape), "test_feature_shape": list(Xte_s.shape),
        "train_windows_per_class": {c: int(np.count_nonzero(ytr==c)) for c in CLASSES},
        "test_windows_per_class": {c: int(np.count_nonzero(yte==c)) for c in CLASSES},
        "scaled_train_mean_abs_max": float(np.max(np.abs(train_mean))), "scaled_train_std_min_max": [float(train_std.min()), float(train_std.max())],
        "limitations": ["File-wise split prevents windows from the same source CSV entering both sets; participant-independent generalization is not measured because multiple class files from each participant may occur in both sets.", "The 1.5-second crop offset follows the lecture example and is an analysis choice, not a start time specified by the cited paper."],
        "outputs": {"dataset_npz": "week2_results/dataset.npz", "filter_plot": "week2_results/week2_filter_spectrum.png", "segment_plot": "week2_results/week2_segment_5channels.png"},
        "file_manifest": file_summary,
    }
    (OUT/"week2_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=lambda o: list(o) if isinstance(o, tuple) else o), encoding="utf-8")
    print("\n=== Week 2 results ===")
    print("Files:", len(files), "| class files:", summary["train_files_per_class"], "/", summary["test_files_per_class"])
    print("Trial interval median:", summary["median_trial_interval_samples"], "samples")
    print("Crop:", (CROP_SAMPLES, 5), "| windows:", summary["sliding_window"]["windows_per_segment"], "x", (WIN,5))
    print("Feature vector:", (25,), "| scaler:", scaler_name)
    print("Xtr/ytr:", Xtr_s.shape, ytr.shape, "| Xte/yte:", Xte_s.shape, yte.shape)
    print("Train scaled mean abs max:", f"{summary['scaled_train_mean_abs_max']:.6g}", "| std range:", summary["scaled_train_std_min_max"])
    print("Saved:", npz, filter_png, crop_png, OUT/"week2_summary.json", sep="\n  ")


if __name__ == "__main__":
    main()
