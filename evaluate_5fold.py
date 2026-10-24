"""Grouped 5-fold CV: keep all windows from each source CSV in one fold.

Run: py -3.14 .\\evaluate_5fold.py
The accessible paper abstract reports five-fold CV but does not disclose its
fold unit; these grouped-file results are an approximation, not an exact match.
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
    confusion_matrix, precision_recall_fscore_support)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "week2_results" / "dataset.npz"
OUT = ROOT / "week2_results" / "cv5_grouped"
WINDOWS_PER_FILE = 30 * 19
SEED = 42


def main():
    d = np.load(DATASET, allow_pickle=False)
    classes = d["class_names"].astype(str).tolist()
    mean, scale = d["scaler_mean"].astype(float), d["scaler_scale"].astype(float)
    # Reverse the fixed holdout scaling to recover raw features. Fit a new
    # StandardScaler inside each CV fold to avoid using validation statistics.
    xtr = d["Xtr"].astype(float) * scale + mean
    xte = d["Xte"].astype(float) * scale + mean
    ytr, yte = d["ytr"].astype(str), d["yte"].astype(str)

    def make_groups(key, row_count):
        groups = np.repeat(d[key].astype(str), WINDOWS_PER_FILE)
        if len(groups) != row_count:
            raise ValueError(f"{key}: expected {row_count} rows; got {len(groups)}")
        return groups

    gtr = make_groups("train_files", len(ytr))
    gte = make_groups("test_files", len(yte))
    X = np.vstack((xtr, xte))
    y = np.concatenate((ytr, yte))
    groups = np.concatenate((gtr, gte))
    encoder = LabelEncoder().fit(classes)
    yi = encoder.transform(y)
    ids = np.arange(len(classes))
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    factories = {
        "Random Forest": lambda: RandomForestClassifier(n_estimators=300,
            class_weight="balanced", random_state=SEED, n_jobs=-1),
        "SVM (RBF)": lambda: SVC(C=1.0, kernel="rbf", gamma="scale", cache_size=2048),
        "XGBoost": lambda: XGBClassifier(objective="multi:softmax",
            num_class=len(classes), n_estimators=300, max_depth=6,
            learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
            reg_lambda=1.0, eval_metric="mlogloss", random_state=SEED, n_jobs=-1),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    output = {"evaluation": "StratifiedGroupKFold, 5 folds; group=source CSV; scaler refit within each fold",
        "fold_unit_in_paper": "not specified in accessible article preview; grouped-file CV is an approximation, not a confirmed exact reproduction",
        "random_state": SEED, "source_csv_count": int(len(np.unique(groups))),
        "window_count": int(len(y)), "classes": classes, "models": {}}

    for name, factory in factories.items():
        print(f"\n=== {name}: grouped 5-fold CV ===", flush=True)
        oof = np.full(len(y), -1, dtype=int)
        fold_rows = []
        for fold, (itr, iva) in enumerate(cv.split(X, yi, groups), 1):
            scaler = StandardScaler().fit(X[itr])
            model = factory()
            model.fit(scaler.transform(X[itr]), yi[itr])
            pred = np.asarray(model.predict(scaler.transform(X[iva])), dtype=int)
            oof[iva] = pred
            p, r, f, _ = precision_recall_fscore_support(yi[iva], pred,
                labels=ids, average="macro", zero_division=0)
            fold_rows.append({"fold": fold,
                "train_files": int(len(np.unique(groups[itr]))),
                "validation_files": int(len(np.unique(groups[iva]))),
                "validation_windows": int(len(iva)),
                "accuracy": float(accuracy_score(yi[iva], pred)),
                "precision_macro": float(p), "recall_macro": float(r), "f1_macro": float(f)})
            print(f"Fold {fold}/5: accuracy={fold_rows[-1]['accuracy']:.4f}, "
                  f"macro-F1={f:.4f}; files={fold_rows[-1]['validation_files']}", flush=True)
        if np.any(oof < 0):
            raise RuntimeError("A sample has no out-of-fold prediction")
        cm = confusion_matrix(yi, oof, labels=ids)
        metric_names = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
        stats = {m: {"mean": float(np.mean([z[m] for z in fold_rows])),
                       "std": float(np.std([z[m] for z in fold_rows], ddof=1))}
                 for m in metric_names}
        stats["pooled_oof_accuracy"] = float(accuracy_score(yi, oof))
        stats["pooled_oof_macro"] = dict(zip(["precision", "recall", "f1"],
            map(float, precision_recall_fscore_support(yi, oof, labels=ids,
                average="macro", zero_division=0)[:3])))
        key = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        output["models"][name] = {"fold_metrics": fold_rows, "mean_std": stats,
            "pooled_oof_confusion_matrix": cm.tolist(),
            "per_class": classification_report(yi, oof, labels=ids,
                target_names=classes, output_dict=True, zero_division=0)}
        fig, ax = plt.subplots(figsize=(9, 7))
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set(xticks=ids, yticks=ids, xticklabels=classes, yticklabels=classes,
            xlabel="Predicted class", ylabel="True class",
            title=f"{name} — grouped 5-fold out-of-fold confusion matrix")
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")
        cutoff = cm.max() / 2
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cutoff else "black")
        fig.tight_layout()
        fig.savefig(OUT / f"{key}_confusion_matrix.png", dpi=180)
        plt.close(fig)

    (OUT / "cv5_results.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Grouped five-fold cross-validation", "",
        "All windows from one source CSV stay in one fold. The scaler is fit separately in each fold.",
        "The accessible article preview does not specify the fold unit; this is not claimed as an exact reproduction.", "",
        "| Model | Accuracy (mean ± SD) | Macro Precision (mean ± SD) | Macro Recall (mean ± SD) | Macro F1 (mean ± SD) |",
        "|---|---:|---:|---:|---:|"]
    for name, item in output["models"].items():
        s = item["mean_std"]
        lines.append(f"| {name} | {s['accuracy']['mean']:.2%} ± {s['accuracy']['std']:.2%} "
            f"| {s['precision_macro']['mean']:.2%} ± {s['precision_macro']['std']:.2%} "
            f"| {s['recall_macro']['mean']:.2%} ± {s['recall_macro']['std']:.2%} "
            f"| {s['f1_macro']['mean']:.2%} ± {s['f1_macro']['std']:.2%} |")
    lines += ["", "## Pooled out-of-fold confusion matrices", ""]
    for name in output["models"]:
        key = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        lines += [f"### {name}", "", f"![{name} confusion matrix]({key}_confusion_matrix.png)", ""]
    (OUT / "cv5_results.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSaved results to {OUT}")


if __name__ == "__main__":
    main()
