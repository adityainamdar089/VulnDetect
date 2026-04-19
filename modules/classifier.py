"""
modules/classifier.py – Classifier implementations for vulnerability detection.

Provides three classifiers with a unified sklearn-compatible interface:

1. ``RandomForestClassifier``       (scikit-learn, 200 estimators)
2. ``XGBClassifier``                (XGBoost, GPU when available)
3. ``CodeBERTClassifierWrapper``    (linear head on fused 769-dim features,
                                     trained with AdamW on GPU)

Public API
----------
train_all(X_train, y_train)  -> dict[str, model]
load_models(input_dim, ...)  -> dict[str, model]
"""

import os
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestClassifier
import joblib

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

logger = logging.getLogger(__name__)

# ─── Optional XGBoost ─────────────────────────────────────────────────────────
try:
    from xgboost import XGBClassifier as _XGBBase
    _XGBOOST_OK = True
except ImportError:
    _XGBOOST_OK = False
    logger.warning("XGBoost not installed – XGBClassifier will be skipped.")


# ─── CodeBERT classification head ─────────────────────────────────────────────

class CodeBERTClassificationHead(nn.Module):
    """
    Lightweight classification head on top of the (fused) 769-dim features.

    Architecture: Linear(input_dim→256) → ReLU → Dropout(0.3) → Linear(256→num_classes)
    """

    def __init__(self, input_dim: int = 769, num_classes: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return self.net(x)


class CodeBERTClassifierWrapper:
    """
    Sklearn-compatible wrapper around :class:`CodeBERTClassificationHead`.

    Trains with AdamW for ``config.NUM_EPOCHS`` epochs; handles GPU OOM by
    retrying the failed batch on CPU.
    """

    def __init__(
        self, input_dim: int = 769, num_classes: int = 2
    ) -> None:
        self.input_dim   = input_dim
        self.num_classes = num_classes
        self.device      = config.DEVICE
        self.model: Optional[CodeBERTClassificationHead] = None

    # ── Training ──────────────────────────────────────────────────────────────
    def fit(
        self, X_train: np.ndarray, y_train: np.ndarray
    ) -> "CodeBERTClassifierWrapper":
        """Train the classification head on (X_train, y_train)."""
        self.model = CodeBERTClassificationHead(
            self.input_dim, self.num_classes
        ).to(self.device)

        optimiser = optim.AdamW(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )
        criterion = nn.CrossEntropyLoss()

        X_t  = torch.tensor(X_train, dtype=torch.float32)
        y_t  = torch.tensor(y_train, dtype=torch.long)
        ds   = TensorDataset(X_t, y_t)
        dl   = DataLoader(ds, batch_size=config.BATCH_SIZE, shuffle=True)

        self.model.train()
        for epoch in range(config.NUM_EPOCHS):
            epoch_loss = 0.0
            for xb, yb in dl:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                optimiser.zero_grad()
                try:
                    logits = self.model(xb)
                except RuntimeError as exc:
                    if "out of memory" in str(exc).lower():
                        logger.warning(
                            "GPU OOM during training – switching to CPU."
                        )
                        torch.cuda.empty_cache()
                        self.device = "cpu"
                        self.model.to(self.device)
                        xb, yb = xb.cpu(), yb.cpu()
                        logits = self.model(xb)
                    else:
                        raise
                loss = criterion(logits, yb)
                loss.backward()
                optimiser.step()
                epoch_loss += loss.item()
            logger.info(
                "Epoch [%d/%d]  loss=%.4f",
                epoch + 1, config.NUM_EPOCHS,
                epoch_loss / max(len(dl), 1),
            )
        self.model.eval()
        return self

    # ── Inference ─────────────────────────────────────────────────────────────
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Return predicted class indices."""
        return np.argmax(self.predict_proba(X_test), axis=1)

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """Return class probabilities (softmax)."""
        assert self.model is not None, "Call .fit() before .predict_proba()."
        self.model.eval()
        X_t = torch.tensor(X_test, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.model(X_t), dim=1).cpu().numpy()
        return probs


# ─── Factory helpers ──────────────────────────────────────────────────────────

def _make_rf() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=config.RF_N_ESTIMATORS,
        random_state=config.RANDOM_SEED,
        n_jobs=-1,
    )


def _make_xgb() -> "_XGBBase":
    if not _XGBOOST_OK:
        raise ImportError("XGBoost is not installed.")
    return _XGBBase(
        n_estimators=200,
        tree_method=config.XGB_TREE_METHOD,  # "hist" for XGBoost 2.0+
        device=config.XGB_DEVICE,            # "cuda" or "cpu"
        eval_metric="logloss",
        random_state=config.RANDOM_SEED,
        verbosity=0,
    )


# ─── Public API ───────────────────────────────────────────────────────────────

def train_all(
    X_train: np.ndarray, y_train: np.ndarray
) -> Dict[str, object]:
    """
    Train all three classifiers and save them to ``config.MODELS_DIR``.

    Parameters
    ----------
    X_train : Feature matrix (N, 769).
    y_train : Label vector  (N,).

    Returns
    -------
    dict mapping model name → trained model object.
    """
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    models: Dict[str, object] = {}

    # ── Random Forest ─────────────────────────────────────────────────────────
    logger.info("Training RandomForestClassifier …")
    rf = _make_rf()
    rf.fit(X_train, y_train)
    rf_path = Path(config.MODELS_DIR) / "random_forest.joblib"
    joblib.dump(rf, str(rf_path))
    models["random_forest"] = rf
    logger.info("RandomForest saved → %s", rf_path)

    # ── XGBoost ───────────────────────────────────────────────────────────────
    if _XGBOOST_OK:
        logger.info("Training XGBClassifier (tree_method=%s) …", config.XGB_TREE_METHOD)
        xgb = _make_xgb()
        xgb.fit(X_train, y_train)
        xgb_path = Path(config.MODELS_DIR) / "xgboost.json"
        xgb.save_model(str(xgb_path))
        models["xgboost"] = xgb
        logger.info("XGBoost saved → %s", xgb_path)
    else:
        logger.warning("XGBoost not available – skipping.")

    # ── CodeBERT classification head ──────────────────────────────────────────
    logger.info("Training CodeBERT classification head …")
    num_classes = int(len(np.unique(y_train)))
    cb = CodeBERTClassifierWrapper(input_dim=X_train.shape[1], num_classes=num_classes)
    cb.fit(X_train, y_train)
    cb_path = Path(config.MODELS_DIR) / "codebert_head.pt"
    torch.save(cb.model.state_dict(), str(cb_path))
    models["codebert_head"] = cb
    logger.info("CodeBERT head saved → %s", cb_path)

    return models


def load_models(
    input_dim: int = 769, num_classes: int = 2
) -> Dict[str, object]:
    """
    Load all saved models from ``config.MODELS_DIR``.

    Parameters
    ----------
    input_dim   : Feature dimensionality used during training.
    num_classes : Number of output classes.

    Returns
    -------
    dict mapping model name → loaded model object.
    """
    models_dir = Path(config.MODELS_DIR)
    models: Dict[str, object] = {}

    # Random Forest
    rf_path = models_dir / "random_forest.joblib"
    if rf_path.exists():
        models["random_forest"] = joblib.load(str(rf_path))
        logger.info("Loaded RandomForest from %s", rf_path)

    # XGBoost
    xgb_path = models_dir / "xgboost.json"
    if xgb_path.exists() and _XGBOOST_OK:
        xgb = _make_xgb()
        xgb.load_model(str(xgb_path))
        models["xgboost"] = xgb
        logger.info("Loaded XGBoost from %s", xgb_path)

    # CodeBERT head
    cb_path = models_dir / "codebert_head.pt"
    if cb_path.exists():
        cb = CodeBERTClassifierWrapper(input_dim=input_dim, num_classes=num_classes)
        cb.model = CodeBERTClassificationHead(input_dim, num_classes)
        cb.model.load_state_dict(
            torch.load(str(cb_path), map_location=config.DEVICE, weights_only=True)
        )
        cb.model.to(config.DEVICE)
        cb.model.eval()
        models["codebert_head"] = cb
        logger.info("Loaded CodeBERT head from %s", cb_path)

    return models
