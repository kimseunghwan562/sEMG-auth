"""Train and compare RF, SVM and XGBoost on week2_results/dataset.npz.

Run from the repository root after week2_h_pipeline.py:
    python .\\train_models.py

The hold-out recordings were split by source CSV in week2_h_pipeline.py before
their windows were extracted. Metrics below therefore describe this fixed
file-wise 80/20 split, not the paper's five-fold cross-validation result.
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
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "week2_results" / "dataset.npz"
OUT = ROOT / "week2_results" / "models"
SEED = 42


def main() -> None:
    if not DATASET.exists():
        raise FileNotFoundError(f"{DATASET} not found. Run python .\\week2_h_pipeline.py first.")
    OUT.mkdir(parents=True, exist_ok=True)
    data = np.load(DATASET, allow_pickle=False)
    X_train, y_train = data["Xtr"], data["ytr"].astype(str)
    X_test, y_test = data["Xte"], data["yte"].astype(str)
    classes = data["class_names"].astype(str).tolist()

    encoder = LabelEncoder().fit(classes)
    y_train_i = encoder.transform(y_train)
    y_test_i = encoder.transform(y_test)
    ids = np.arange(len(classes))

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=SEED,
            n_jobs=-1,
        ),
        "SVM (RBF)": SVC(C=1.0, kernel="rbf", gamma="scale"),
        "XGBoost": XGBClassifier(
            objective="multi:softmax", num_class=len(classes),
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
            eval_metric="mlogloss", random_state=SEED, n_jobs=-1,
        ),
    }

    results = {}
    for name, model in models.items():
        print(f"Training {name}...", flush=True)
        model.fit(X_train, y_train_i)
        pred = model.predict(X_test).astype(int)
        cm = confusion_matrix(y_test_i, pred, labels=ids)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test_i, pred, labels=ids, average="macro", zero_division=0,
        )
        key = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        results[name] = {
            "accuracy": float(accuracy_score(y_test_i, pred)),
            "precision_macro": float(precision),
            "recall_macro": float(recall),
            "f1_macro": float(f1),
            "confusion_matrix": cm.tolist(),
            "per_class": classification_report(
                y_test_i, pred, labels=ids, target_names=classes,
                output_dict=True, zero_division=0,
            ),
        }

        fig, ax = plt.subplots(figsize=(9, 7))
        image = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        ax.set(
            xticks=ids, yticks=ids, xticklabels=classes, yticklabels=classes,
            xlabel="Predicted class", ylabel="True class",
            title=f"{name} — file-wise hold-out confusion matrix",
        )
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")
        threshold = cm.max() / 2 if cm.size else 0
        for row in range(cm.shape[0]):
            for col in range(cm.shape[1]):
                ax.text(col, row, str(cm[row, col]), ha="center", va="center",
                        color="white" if cm[row, col] > threshold else "black")
        fig.tight_layout()
        fig.savefig(OUT / f"{key}_confusion_matrix.png", dpi=180)
        plt.close(fig)
        print(f"{name}: accuracy={results[name]['accuracy']:.4f}, "
              f"macro-F1={f1:.4f}", flush=True)

    summary = {
        "evaluation": "fixed stratified source-CSV 80/20 hold-out; windows from a source CSV stay together",
        "seed": SEED,
        "train_windows": int(len(y_train)),
        "test_windows": int(len(y_test)),
        "classes": classes,
        "models": results,
    }
    (OUT / "model_results.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# Model comparison results", "",
        "Evaluation: fixed stratified source-CSV 80/20 hold-out. "
        "Windows from one CSV remain in the same split.", "",
        f"Train windows: {len(y_train):,} | Test windows: {len(y_test):,}", "",
        "| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, item in results.items():
        lines.append(
            f"| {name} | {item['accuracy']:.4f} | {item['precision_macro']:.4f} "
            f"| {item['recall_macro']:.4f} | {item['f1_macro']:.4f} |"
        )
    lines += ["", "## Confusion matrices", ""]
    for name in results:
        key = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        lines += [f"### {name}", "", f"![{name} confusion matrix]({key}_confusion_matrix.png)", ""]
    (OUT / "model_results.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved metrics and confusion matrices under {OUT}")


if __name__ == "__main__":
    main()
