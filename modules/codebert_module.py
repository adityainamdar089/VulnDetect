"""
modules/codebert_module.py – CodeBERT [CLS] embedding generation.

Loads ``microsoft/codebert-base`` via HuggingFace Transformers, moves it to
the configured device (GPU when available), and extracts 768-dimensional
[CLS] token embeddings for every code snippet.

Embeddings are saved as a NumPy array so that expensive inference is skipped
on subsequent runs.

Public API
----------
generate_embeddings(df) -> np.ndarray
load_embeddings()       -> tuple[np.ndarray, pd.DataFrame]
"""

import os
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

logger = logging.getLogger(__name__)

# ─── Embedder class ───────────────────────────────────────────────────────────

class _CodeBERTEmbedder:
    """Internal helper that wraps the HuggingFace model."""

    def __init__(self) -> None:
        self.device = config.DEVICE
        logger.info("Loading CodeBERT (%s) on %s …", config.CODEBERT_MODEL, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(config.CODEBERT_MODEL)
        self.model = AutoModel.from_pretrained(config.CODEBERT_MODEL)
        self.model.to(self.device)
        self.model.eval()
        logger.info("CodeBERT loaded successfully.")

    def embed_batch(self, codes: list[str]) -> np.ndarray:
        """
        Embed a list of code strings and return the [CLS] embeddings.

        Parameters
        ----------
        codes : List of source-code strings (batch).

        Returns
        -------
        np.ndarray of shape (len(codes), 768).
        """
        enc = self.tokenizer(
            codes,
            padding=True,
            truncation=True,
            max_length=config.MAX_TOKEN_LEN,
            return_tensors="pt",
        )
        input_ids      = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        with torch.no_grad():
            try:
                out = self.model(input_ids=input_ids, attention_mask=attention_mask)
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    logger.warning(
                        "GPU OOM – falling back to CPU for this batch."
                    )
                    torch.cuda.empty_cache()
                    self.model.to("cpu")
                    self.device = "cpu"
                    input_ids      = input_ids.cpu()
                    attention_mask = attention_mask.cpu()
                    out = self.model(input_ids=input_ids, attention_mask=attention_mask)
                else:
                    raise

        # Shape: (batch, seq_len, 768) → take [CLS] token at position 0
        cls_emb: np.ndarray = out.last_hidden_state[:, 0, :].cpu().numpy()
        return cls_emb


# ─── Public API ───────────────────────────────────────────────────────────────

def generate_embeddings(df: pd.DataFrame) -> np.ndarray:
    """
    Generate CodeBERT [CLS] embeddings for all rows in *df*.

    Skips generation if the embedding files already exist on disk.

    Parameters
    ----------
    df : DataFrame with at least columns ``code`` and ``snippet_id``.

    Returns
    -------
    np.ndarray of shape (len(df), 768).
    """
    emb_path  = Path(config.EMBEDDING_DIR) / "embeddings.npy"
    meta_path = Path(config.EMBEDDING_DIR) / "embeddings_meta.csv"

    if emb_path.exists() and meta_path.exists():
        logger.info(
            "Embeddings already exist at %s – skipping generation.", emb_path
        )
        return np.load(str(emb_path))

    os.makedirs(config.EMBEDDING_DIR, exist_ok=True)
    embedder = _CodeBERTEmbedder()
    codes      = df["code"].tolist()
    batch_size = config.BATCH_SIZE
    all_embs   = []

    for start in tqdm(range(0, len(codes), batch_size), desc="Generating embeddings"):
        batch = codes[start : start + batch_size]
        all_embs.append(embedder.embed_batch(batch))

    embeddings: np.ndarray = np.vstack(all_embs)

    # Persist
    np.save(str(emb_path), embeddings)
    df[["snippet_id", "label", "cwe_type"]].reset_index(drop=True).to_csv(
        str(meta_path), index=False
    )
    logger.info(
        "Saved embeddings to %s  shape=%s", emb_path, embeddings.shape
    )
    return embeddings


def load_embeddings() -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Load previously saved embeddings and metadata from disk.

    Returns
    -------
    Tuple of:
        - embeddings : np.ndarray of shape (N, 768)
        - meta       : pd.DataFrame with columns ``snippet_id``, ``label``, ``cwe_type``

    Raises
    ------
    FileNotFoundError if embedding files are missing (run data_prep.py first).
    """
    emb_path  = Path(config.EMBEDDING_DIR) / "embeddings.npy"
    meta_path = Path(config.EMBEDDING_DIR) / "embeddings_meta.csv"

    if not emb_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"Embeddings not found at {config.EMBEDDING_DIR}. "
            "Please run  python data_prep.py  first."
        )

    embeddings = np.load(str(emb_path))
    meta       = pd.read_csv(str(meta_path))
    logger.info("Loaded embeddings %s and meta %s", embeddings.shape, meta.shape)
    return embeddings, meta
