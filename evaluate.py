"""
evaluate.py – Phase 7: Ablation study, model comparison, and result plots.

Execution
---------
    python evaluate.py

Prerequisites
-------------
    data/processed/dataset.csv
    data/processed/test_idx.npy
    data/processed/X_fused_attention.npy
    data/processed/X_fused_concat.npy
    data/processed/X_fuzzy.npy
    data/processed/X_codebert_only.npy
    models/  (populated by train.py)

Outputs
-------
    results/comparison_table.csv
    results/confusion_matrix_*.png
    results/roc_curve_*.png
    results/per_cwe_f1_*.png
"""

import sys
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import joblib

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import config
from modules.classifier import (
    load_models,
    CodeBERTClassifierWrapper,
    CodeBERTClassificationHead,
)
from modules.evaluator import (
    _compute_metrics,
    _plot_confusion_matrix,
    _plot_roc_curve,
    _plot_per_cwe_f1,
    _results_dir,
)

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Load all models, evaluate against the held-out test set, save results."""
    logger.info("=" * 60)
    logger.info("VulnDetect – Evaluation Pipeline")
    logger.info("=" * 60)

    proc_dir = Path(config.PROCESSED_DIR)

    # ── Load dataset and test indices ─────────────────────────────────────────
    csv_path = proc_dir / "dataset.csv"
    if not csv_path.exists():
        logger.error("Dataset not found at %s. Run data_prep.py first.", csv_path)
        sys.exit(1)

    df         = pd.read_csv(str(csv_path))
    labels     = df["label"].values.astype(int)
    cwe_labels = df["cwe_type"].values

    idx_test_path = proc_dir / "test_idx.npy"
    if not idx_test_path.exists():
        logger.error("test_idx.npy not found. Run train.py first.")
        sys.exit(1)

    idx_test  = np.load(str(idx_test_path))
    y_test    = labels[idx_test]
    cwe_test  = cwe_labels[idx_test]

    # ── Load feature matrices ─────────────────────────────────────────────────
    def _load_feat(name: str) -> Optional[np.ndarray]:
        p = proc_dir / name
        if p.exists():
            return np.load(str(p))[idx_test]
        logger.warning("Feature file not found: %s", p)
        return None

    X_attn   = _load_feat("X_fused_attention.npy")
    X_concat = _load_feat("X_fused_concat.npy")
    X_fuzzy  = _load_feat("X_fuzzy.npy")
    X_cb     = _load_feat("X_codebert_only.npy")

    # ── Load main models (trained on attention-fused features) ────────────────
    attn_dim    = X_attn.shape[1] if X_attn is not None else 769
    num_classes = int(len(np.unique(y_test)))
    main_models = load_models(input_dim=attn_dim, num_classes=num_classes)

    # ── Collect (model_name, model, X_test_for_that_model) triples ───────────
    eval_triples = []

    for mname, m in main_models.items():
        if X_attn is not None:
            eval_triples.append((mname, m, X_attn))

    # Fuzzy-only baseline
    rf_fuzzy_path = Path(config.MODELS_DIR) / "rf_fuzzy_only.joblib"
    if rf_fuzzy_path.exists() and X_fuzzy is not None:
        rf_fuzzy = joblib.load(str(rf_fuzzy_path))
        eval_triples.append(("fuzzy_only_RF", rf_fuzzy, X_fuzzy))

    # CodeBERT-only baseline
    cb_only_path = Path(config.MODELS_DIR) / "codebert_only.pt"
    if cb_only_path.exists() and X_cb is not None:
        cb_only = CodeBERTClassifierWrapper(input_dim=768, num_classes=num_classes)
        cb_only.model = CodeBERTClassificationHead(768, num_classes)
        cb_only.model.load_state_dict(
            torch.load(str(cb_only_path), map_location=config.DEVICE)
        )
        cb_only.model.to(config.DEVICE)
        cb_only.model.eval()
        eval_triples.append(("codebert_only", cb_only, X_cb))

    # Hybrid concat baseline (RF re-used for speed)
    if X_concat is not None and "random_forest" in main_models:
        eval_triples.append(("hybrid_concat_RF", main_models["random_forest"], X_concat))

    # ── Run evaluation ────────────────────────────────────────────────────────
    out_dir = _results_dir()
    rows    = []

    for name, model, X_ev in eval_triples:
        logger.info("Evaluating: %s …", name)
        y_pred = model.predict(X_ev)

        y_prob: Optional[np.ndarray] = None
        try:
            y_prob = model.predict_proba(X_ev)
        except AttributeError:
            pass

        row = _compute_metrics(y_test, y_pred, y_prob, name)
        rows.append(row)

        _plot_confusion_matrix(y_test, y_pred, name, out_dir)
        _plot_roc_curve(y_test, y_prob, name, out_dir)
        _plot_per_cwe_f1(y_test, y_pred, cwe_test, name, out_dir)

    # ── Print and save comparison table ───────────────────────────────────────
    comparison_df = pd.DataFrame(rows)
    separator     = "=" * 70
    print(f"\n{separator}")
    print("          ABLATION STUDY – MODEL COMPARISON TABLE")
    print(separator)
    print(comparison_df.to_string(index=False))
    print(f"{separator}\n")

    csv_out = out_dir / "comparison_table.csv"
    comparison_df.to_csv(str(csv_out), index=False)
    logger.info("Comparison table saved → %s", csv_out)

    logger.info("=" * 60)
    logger.info("evaluate.py completed successfully.")


if __name__ == "__main__":
    main()
