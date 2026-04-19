"""
train.py – Phase 6: Train all classifiers across all fusion strategies.

Execution
---------
    python train.py

Prerequisites
-------------
    data/processed/dataset.csv   (produced by data_prep.py)
    data/embeddings/             (produced by data_prep.py)

Outputs
-------
    data/processed/X_fused_{concat,weighted,attention}.npy
    data/processed/X_fuzzy.npy
    data/processed/X_codebert_only.npy
    data/processed/test_idx.npy
    data/processed/val_idx.npy
    models/random_forest.joblib
    models/xgboost.json
    models/codebert_head.pt
    models/rf_fuzzy_only.joblib
    models/codebert_only.pt
"""

import os
import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import config
from modules.codebert_module import load_embeddings
from modules.fusion import fuse_features
from modules.classifier import (
    train_all,
    CodeBERTClassifierWrapper,
    CodeBERTClassificationHead,
)

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Run the full training pipeline."""
    logger.info("=" * 60)
    logger.info("VulnDetect – Training Pipeline")
    logger.info("=" * 60)

    os.makedirs(config.MODELS_DIR, exist_ok=True)

    # ── Load processed dataset ────────────────────────────────────────────────
    csv_path = Path(config.PROCESSED_DIR) / "dataset.csv"
    if not csv_path.exists():
        logger.error(
            "Processed dataset not found at %s. Run  python data_prep.py  first.",
            csv_path,
        )
        sys.exit(1)

    df = pd.read_csv(str(csv_path))
    logger.info("Loaded dataset: %s rows", len(df))

    # ── Load embeddings ────────────────────────────────────────────────────────
    embeddings, meta = load_embeddings()

    # Align df rows to match meta order (embedding order)
    df = (
        df.set_index("snippet_id")
        .loc[meta["snippet_id"].values]
        .reset_index()
    )
    labels     = df["label"].values.astype(int)
    cwe_labels = df["cwe_type"].values

    logger.info(
        "Class distribution – vulnerable: %d, benign: %d",
        int((labels == 1).sum()), int((labels == 0).sum()),
    )

    # ── Stratified 80 / 10 / 10 split ────────────────────────────────────────
    idx         = np.arange(len(df))
    idx_tv, idx_test = train_test_split(
        idx,
        test_size=config.TEST_SPLIT,
        random_state=config.RANDOM_SEED,
        stratify=labels,
    )
    # val fraction relative to trainval pool
    val_frac = config.VAL_SPLIT / (1.0 - config.TEST_SPLIT)
    idx_train, idx_val = train_test_split(
        idx_tv,
        test_size=val_frac,
        random_state=config.RANDOM_SEED,
        stratify=labels[idx_tv],
    )

    logger.info(
        "Split → train: %d | val: %d | test: %d",
        len(idx_train), len(idx_val), len(idx_test),
    )

    # Persist indices for evaluate.py
    proc_dir = Path(config.PROCESSED_DIR)
    np.save(str(proc_dir / "test_idx.npy"), idx_test)
    np.save(str(proc_dir / "val_idx.npy"),  idx_val)

    # ── Build fused feature matrices for all three strategies ─────────────────
    for strategy in ("concat", "weighted", "attention"):
        logger.info("Building fused features – strategy='%s' …", strategy)
        X_fused = fuse_features(df, embeddings, strategy=strategy)
        out_path = proc_dir / f"X_fused_{strategy}.npy"
        np.save(str(out_path), X_fused)
        logger.info("Saved %s  shape=%s", out_path.name, X_fused.shape)

    # ── Train all classifiers on attention-fused features ─────────────────────
    X_attn  = np.load(str(proc_dir / "X_fused_attention.npy"))
    X_train = X_attn[idx_train]
    y_train = labels[idx_train]

    logger.info("Training all classifiers on attention-fused features …")
    train_all(X_train, y_train)

    # ── Ablation baselines ────────────────────────────────────────────────────
    # Fuzzy-only: 6-dim vector (5 features + fuzzy risk score)
    feature_cols = [
        "input_validation_score",
        "sensitive_data_exposure",
        "access_control_strength",
        "resource_management_score",
        "control_flow_complexity",
        "fuzzy_risk_score",
    ]
    X_fuzzy = df[feature_cols].values
    np.save(str(proc_dir / "X_fuzzy.npy"), X_fuzzy)

    logger.info("Training fuzzy-only baseline (RandomForest) …")
    rf_fuzzy = RandomForestClassifier(
        n_estimators=config.RF_N_ESTIMATORS,
        random_state=config.RANDOM_SEED,
        n_jobs=-1,
    )
    rf_fuzzy.fit(X_fuzzy[idx_train], y_train)
    joblib.dump(rf_fuzzy, str(Path(config.MODELS_DIR) / "rf_fuzzy_only.joblib"))
    logger.info("Fuzzy-only RF saved.")

    # CodeBERT-only: raw 768-dim embeddings
    X_cb = embeddings
    np.save(str(proc_dir / "X_codebert_only.npy"), X_cb)

    logger.info("Training CodeBERT-only baseline …")
    num_classes = int(len(np.unique(y_train)))
    cb_only = CodeBERTClassifierWrapper(input_dim=768, num_classes=num_classes)
    cb_only.fit(X_cb[idx_train], y_train)
    torch.save(
        cb_only.model.state_dict(),
        str(Path(config.MODELS_DIR) / "codebert_only.pt"),
    )
    logger.info("CodeBERT-only head saved.")

    logger.info("=" * 60)
    logger.info("train.py completed successfully.")


if __name__ == "__main__":
    main()
