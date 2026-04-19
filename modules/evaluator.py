"""
modules/evaluator.py – Evaluation metrics, plots, and ablation study.

For each model computes: accuracy, precision, recall, F1-score, AUC-ROC.
Generates and saves:
    • Confusion matrix (PNG)
    • ROC curve (PNG)
    • Per-CWE-type F1 bar chart (PNG)
    • Ablation comparison table (CSV)

Public API
----------
evaluate_all(models, X_test, y_test, cwe_labels) -> pd.DataFrame
"""

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server environments
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

logger = logging.getLogger(__name__)


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _results_dir() -> Path:
    p = Path(config.RESULTS_DIR)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray],
    model_name: str,
) -> dict:
    """Compute and return a metrics dict for one model."""
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    auc: float | str = "N/A"
    if y_prob is not None and len(np.unique(y_true)) == 2:
        try:
            auc = round(float(roc_auc_score(y_true, y_prob[:, 1])), 4)
        except Exception:
            pass

    return {
        "Model":     model_name,
        "Accuracy":  round(acc,  4),
        "Precision": round(prec, 4),
        "Recall":    round(rec,  4),
        "F1-Score":  round(f1,   4),
        "AUC-ROC":   auc,
    }


def _plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    out_dir: Path,
) -> None:
    """Save a confusion-matrix heatmap as PNG."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, linewidths=0.5)
    ax.set_title(f"Confusion Matrix – {model_name}", fontsize=13, pad=12)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    path = out_dir / f"confusion_matrix_{model_name.replace(' ', '_')}.png"
    fig.tight_layout()
    fig.savefig(str(path), dpi=150)
    plt.close(fig)
    logger.info("Saved confusion matrix → %s", path)


def _plot_roc_curve(
    y_true: np.ndarray,
    y_prob: Optional[np.ndarray],
    model_name: str,
    out_dir: Path,
) -> None:
    """Save an ROC curve as PNG (binary classification only)."""
    if y_prob is None or len(np.unique(y_true)) != 2:
        return
    try:
        fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
        auc_val     = roc_auc_score(y_true, y_prob[:, 1])

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc_val:.4f}")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC Curve – {model_name}", fontsize=13, pad=12)
        ax.legend(loc="lower right")
        path = out_dir / f"roc_curve_{model_name.replace(' ', '_')}.png"
        fig.tight_layout()
        fig.savefig(str(path), dpi=150)
        plt.close(fig)
        logger.info("Saved ROC curve → %s", path)
    except Exception as exc:
        logger.warning("ROC curve failed for %s: %s", model_name, exc)


def _plot_per_cwe_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cwe_labels: np.ndarray,
    model_name: str,
    out_dir: Path,
) -> None:
    """Save a per-CWE-type F1 bar chart as PNG."""
    unique_cwes = sorted(set(cwe_labels))
    f1s = []
    for cwe in unique_cwes:
        mask = cwe_labels == cwe
        if mask.sum() == 0:
            f1s.append(0.0)
        else:
            f1s.append(
                float(f1_score(y_true[mask], y_pred[mask],
                               average="weighted", zero_division=0))
            )

    palette = sns.color_palette("husl", len(unique_cwes))
    fig, ax  = plt.subplots(figsize=(max(8, len(unique_cwes) * 1.5), 5))
    bars = ax.bar(unique_cwes, f1s, color=palette, edgecolor="white")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("CWE Type")
    ax.set_ylabel("Weighted F1-Score")
    ax.set_title(f"Per-CWE F1-Score – {model_name}", fontsize=13, pad=12)

    for bar, val in zip(bars, f1s):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=9,
        )

    path = out_dir / f"per_cwe_f1_{model_name.replace(' ', '_')}.png"
    fig.tight_layout()
    fig.savefig(str(path), dpi=150)
    plt.close(fig)
    logger.info("Saved per-CWE F1 chart → %s", path)


# ─── Public API ───────────────────────────────────────────────────────────────

def evaluate_all(
    models: Dict[str, object],
    X_test: np.ndarray,
    y_test: np.ndarray,
    cwe_labels: np.ndarray,
) -> pd.DataFrame:
    """
    Evaluate every model in *models*, generate all plots, and return a
    comparison DataFrame.

    Parameters
    ----------
    models     : dict of model_name → model (must expose ``.predict()``).
    X_test     : Test feature matrix.
    y_test     : Ground-truth labels.
    cwe_labels : CWE type string for each test sample.

    Returns
    -------
    pd.DataFrame with one row per model and columns:
        Model, Accuracy, Precision, Recall, F1-Score, AUC-ROC
    """
    out_dir = _results_dir()
    rows    = []

    for name, model in models.items():
        logger.info("Evaluating: %s …", name)
        y_pred = model.predict(X_test)

        y_prob: Optional[np.ndarray] = None
        try:
            y_prob = model.predict_proba(X_test)
        except AttributeError:
            logger.debug("%s has no predict_proba – AUC skipped.", name)

        row = _compute_metrics(y_test, y_pred, y_prob, name)
        rows.append(row)

        _plot_confusion_matrix(y_test, y_pred, name, out_dir)
        _plot_roc_curve(y_test, y_prob, name, out_dir)
        _plot_per_cwe_f1(y_test, y_pred, cwe_labels, name, out_dir)

    df = pd.DataFrame(rows)

    # Pretty-print to console
    separator = "=" * 70
    print(f"\n{separator}")
    print("          ABLATION STUDY – MODEL COMPARISON TABLE")
    print(separator)
    print(df.to_string(index=False))
    print(f"{separator}\n")

    csv_path = out_dir / "comparison_table.csv"
    df.to_csv(str(csv_path), index=False)
    logger.info("Comparison table saved → %s", csv_path)

    return df
