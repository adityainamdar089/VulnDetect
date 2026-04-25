"""
modules/fusion.py – Feature fusion strategies.

Combines a 768-dim CodeBERT embedding with a 1-dim fuzzy risk score using
three strategies:

1. ``simple_concat``   – trivial concatenation → 769-dim
2. ``weighted_concat`` – alpha-scaled embedding + beta-scaled fuzzy → 769-dim
3. ``attention_fusion``– a 2-layer MLP learns element-wise attention weights

Public API
----------
simple_concat(embedding, fuzzy_score)           -> np.ndarray (769,)
weighted_concat(embedding, fuzzy_score, α, β)   -> np.ndarray (769,)
attention_fusion(embedding, fuzzy_score)         -> np.ndarray (769,)
fuse_features(df, embeddings, strategy)          -> np.ndarray (N, 769)
"""

import logging
from typing import Literal

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

logger = logging.getLogger(__name__)


# ─── Attention MLP ────────────────────────────────────────────────────────────

class AttentionFusionMLP(nn.Module):
    """
    Two-layer MLP that learns element-wise importance weights for a 769-dim
    fused vector.

    Architecture: Linear(769→256) → ReLU → Dropout(0.1) → Linear(256→769) → Sigmoid

    The sigmoid output is used as attention gates: ``output = gates * input``.
    """

    def __init__(self, input_dim: int = 769, hidden_dim: int = 256) -> None:
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return attention-weighted features."""
        gates = self.gate_net(x)
        return x * gates  # element-wise gating


# Singleton instance of the MLP
_mlp: AttentionFusionMLP | None = None


def _get_mlp() -> AttentionFusionMLP:
    """Return the shared (untrained) MLP instance, creating it on first call."""
    global _mlp
    if _mlp is None:
        _mlp = AttentionFusionMLP(input_dim=769, hidden_dim=256)
        _mlp.to(config.DEVICE)
        _mlp.eval()
    return _mlp


# ─── Individual fusion functions ──────────────────────────────────────────────

def simple_concat(embedding: np.ndarray, fuzzy_score: float) -> np.ndarray:
    """
    Concatenate a 768-dim CodeBERT embedding with a scalar fuzzy score.

    Parameters
    ----------
    embedding   : np.ndarray of shape (768,)
    fuzzy_score : float in [0, 1]

    Returns
    -------
    np.ndarray of shape (769,)
    """
    return np.concatenate([embedding, [float(fuzzy_score)]], axis=0)


def weighted_concat(
    embedding: np.ndarray,
    fuzzy_score: float,
    alpha: float = config.FUSION_ALPHA,
    beta: float  = config.FUSION_BETA,
) -> np.ndarray:
    """
    Scale embedding by *alpha* and fuzzy score by *beta* before concatenating.

    Parameters
    ----------
    embedding   : np.ndarray of shape (768,)
    fuzzy_score : float in [0, 1]
    alpha       : Weight for the CodeBERT embedding (default from config).
    beta        : Weight for the fuzzy score (default from config).

    Returns
    -------
    np.ndarray of shape (769,)
    """
    return np.concatenate(
        [embedding * alpha, [float(fuzzy_score) * beta]], axis=0
    )


def attention_fusion(
    embedding: np.ndarray, fuzzy_score: float
) -> np.ndarray:
    """
    Use the 2-layer attention MLP to learn importance weights for fusion.

    Parameters
    ----------
    embedding   : np.ndarray of shape (768,)
    fuzzy_score : float in [0, 1]

    Returns
    -------
    np.ndarray of shape (769,)  – attention-weighted combined vector
    """
    combined = np.concatenate([embedding, [float(fuzzy_score)]], axis=0)  # (769,)
    tensor   = (
        torch.tensor(combined, dtype=torch.float32)
        .unsqueeze(0)  # (1, 769)
        .to(config.DEVICE)
    )
    mlp = _get_mlp()
    with torch.no_grad():
        out: np.ndarray = mlp(tensor).squeeze(0).cpu().numpy()
    return out


# ─── Dataset-level fusion ─────────────────────────────────────────────────────

def fuse_features(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    strategy: Literal["concat", "weighted", "attention"] = "attention",
) -> np.ndarray:
    """
    Apply the chosen fusion strategy to all samples in *df*.

    Parameters
    ----------
    df         : DataFrame that must contain a ``fuzzy_risk_score`` column.
    embeddings : np.ndarray of shape (N, 768) – one row per sample.
    strategy   : ``"concat"``, ``"weighted"``, or ``"attention"``.

    Returns
    -------
    np.ndarray of shape (N, 769) – fused feature matrix.

    Raises
    ------
    ValueError if *strategy* is not recognised.
    AssertionError if DataFrame and embeddings lengths differ.
    """
    n = len(df)
    assert embeddings.shape[0] == n, (
        f"Embeddings row count ({embeddings.shape[0]}) != df rows ({n})."
    )

    _strategy_map = {
        "concat":    lambda emb, s: simple_concat(emb, s),
        "weighted":  lambda emb, s: weighted_concat(emb, s),
        "attention": lambda emb, s: attention_fusion(emb, s),
    }
    fn = _strategy_map.get(strategy)
    if fn is None:
        raise ValueError(
            f"Unknown fusion strategy '{strategy}'. "
            "Choose from: concat, weighted, attention."
        )

    if "fuzzy_risk_score" not in df.columns:
        raise KeyError(
            "DataFrame is missing required column 'fuzzy_risk_score'. "
            "Run run_fuzzy_on_dataset(df) from modules.fuzzy_module before calling fuse_features()."
        )
    fuzzy_scores = df["fuzzy_risk_score"].values
    logger.info("Applying fusion strategy '%s' to %d samples …", strategy, n)

    fused = [fn(embeddings[i], fuzzy_scores[i]) for i in range(n)]
    result = np.vstack(fused)
    logger.info("Fused matrix shape: %s", result.shape)
    return result
