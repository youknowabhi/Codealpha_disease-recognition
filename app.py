"""
evaluate.py
-----------
Evaluation metrics, plots, and classification reports for trained models.

Usage:
    python src/evaluate.py --dataset all
"""

import argparse
import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)

# ── Config ────────────────────────────────────────────────────────────────────
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
with open(CONFIG_PATH, "r") as f:
    CONFIG = yaml.safe_load(f)

RESULTS_DIR = CONFIG["paths"]["results"]
MODELS_DIR = CONFIG["paths"]["models"]
os.makedirs(RESULTS_DIR, exist_ok=True)

DATASETS = ["heart", "diabetes", "breast_cancer"]
MODEL_NAMES = ["logistic_regression", "svm", "random_forest", "xgboost"]


# ── Core evaluation ───────────────────────────────────────────────────────────

def evaluate_model(model, X_test, y_test, model_name: str, dataset_name: str) -> dict:
    """Compute and return all classification metrics."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    metrics = {
        "model": model_name,
        "dataset": dataset_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob) if y_prob is not None else None,
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
    }

    # Save confusion matrix plot
    _plot_confusion_matrix(y_test, y_pred, model_name, dataset_name)

    return metrics


# ── Plots ─────────────────────────────────────────────────────────────────────

def _plot_confusion_matrix(y_test, y_pred, model_name: str, dataset_name: str):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Negative", "Positive"],
                yticklabels=["Negative", "Positive"])
    plt.title(f"Confusion Matrix\n{dataset_name} | {model_name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f"{dataset_name}_{model_name}_cm.png")
    plt.savefig(path, dpi=150)
    plt.close()


def plot_roc_curves(dataset_name: str, X_test, y_test):
    """Plot ROC curves for all 4 models on one figure."""
    from data_loader import load_dataset
    from feature_engineering import preprocess

    plt.figure(figsize=(8, 6))
    colors = ["steelblue", "darkorange", "forestgreen", "crimson"]

    for color, model_name in zip(colors, MODEL_NAMES):
        model_path = os.path.join(MODELS_DIR, f"{dataset_name}_{model_name}.pkl")
        if not os.path.exists(model_path):
            continue
        model = joblib.load(model_path)
        if not hasattr(model, "predict_proba"):
            continue
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        plt.plot(fpr, tpr, color=color, lw=2,
                 label=f"{model_name} (AUC = {auc:.2f})")

    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curves — {dataset_name.upper()}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f"{dataset_name}_roc_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[✓] ROC curve saved → {path}")


def plot_model_comparison(results: dict, dataset_name: str):
    """Bar chart comparing accuracy & F1 across models."""
    models = list(results.keys())
    accuracy = [results[m]["accuracy"] for m in models]
    f1 = [results[m]["f1_score"] for m in models]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width / 2, accuracy, width, label="Accuracy", color="steelblue")
    bars2 = ax.bar(x + width / 2, f1, width, label="F1-Score", color="darkorange")

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", "\n") for m in models])
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("Score")
    ax.set_title(f"Model Comparison — {dataset_name.upper()}")
    ax.legend()

    for bar in bars1:
        ax.annotate(f"{bar.get_height():.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
    for bar in bars2:
        ax.annotate(f"{bar.get_height():.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f"{dataset_name}_model_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[✓] Comparison chart saved → {path}")


# ── Save / load results ───────────────────────────────────────────────────────

def save_results(metrics: dict, dataset_name: str):
    """Persist metrics dict as JSON."""
    # Remove non-serializable classification_report details for top-level JSON
    clean = {}
    for model_name, m in metrics.items():
        clean[model_name] = {k: v for k, v in m.items() if k != "classification_report"}

    path = os.path.join(RESULTS_DIR, f"{dataset_name}_metrics.json")
    with open(path, "w") as f:
        json.dump(clean, f, indent=2)
    print(f"[✓] Metrics saved → {path}")


def load_results(dataset_name: str) -> dict:
    path = os.path.join(RESULTS_DIR, f"{dataset_name}_metrics.json")
    with open(path) as f:
        return json.load(f)


def print_results_table(dataset_name: str):
    results = load_results(dataset_name)
    df = pd.DataFrame(results).T[["accuracy", "precision", "recall", "f1_score", "roc_auc"]]
    df = df.round(4)
    print(f"\n{'='*60}")
    print(f"  Results — {dataset_name.upper()}")
    print(f"{'='*60}")
    print(df.to_string())


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Disease Prediction — Evaluate")
    parser.add_argument(
        "--dataset",
        choices=["heart", "diabetes", "breast_cancer", "all"],
        default="all",
        help="Dataset to evaluate",
    )
    args = parser.parse_args()
    datasets = DATASETS if args.dataset == "all" else [args.dataset]

    for ds in datasets:
        try:
            print_results_table(ds)
        except FileNotFoundError:
            print(f"[!] No results found for '{ds}'. Run train.py first.")
