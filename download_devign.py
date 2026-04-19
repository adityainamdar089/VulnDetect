"""
download_devign.py – Download the Devign dataset from HuggingFace and
convert it to VulnDetect's expected CSV format.

Devign (Zhou et al., 2019)
--------------------------
  ~27,000 C functions from QEMU & FFmpeg with binary vulnerability labels.
  label=1 → vulnerable, label=0 → safe.

Usage
-----
    python download_devign.py
"""

import sys
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

import config

# ── Output path ───────────────────────────────────────────────────────────────
RAW_CSV = Path(config.PROCESSED_DIR) / "dataset.csv"
os.makedirs(config.PROCESSED_DIR, exist_ok=True)

# ── Max samples (memory-safe for GPU training) ────────────────────────────────
MAX_SAMPLES = 20000  # 20K balanced → 10K vuln + 10K safe


def download_and_convert():
    logger.info("=" * 60)
    logger.info("Downloading Devign dataset from HuggingFace …")
    logger.info("=" * 60)

    try:
        from datasets import load_dataset
    except ImportError:
        logger.error(
            "The 'datasets' library is not installed.\n"
            "Run: D:\\pytorch_env\\Scripts\\pip install datasets"
        )
        sys.exit(1)

    # Try several known locations for the Devign dataset
    DEVIGN_SOURCES = [
        ("json", "https://raw.githubusercontent.com/epicosy/devign/refs/heads/master/data/devign_dataset.json",),
        ("huggingface", "benjaminbeilby/devign"),
        ("huggingface", "VulnerabilityDetection/Devign"),
        ("huggingface", "DetectVul/devign"),
    ]

    df_raw = None
    for source_type, source in DEVIGN_SOURCES:
        try:
            if source_type == "huggingface":
                logger.info("Trying HuggingFace: %s …", source)
                ds = load_dataset(source)
                dfs = [ds[s].to_pandas() for s in ds.keys()]
                df_raw = pd.concat(dfs, ignore_index=True)
                logger.info("  ✓ Loaded %d rows from %s", len(df_raw), source)
            else:
                logger.info("Trying direct JSON download: %s …", source)
                import requests
                resp = requests.get(source, timeout=30)
                resp.raise_for_status()
                import json, io
                data = json.loads(resp.text)
                if isinstance(data, list):
                    df_raw = pd.DataFrame(data)
                else:
                    df_raw = pd.DataFrame(data.get("data", data))
                logger.info("  ✓ Loaded %d rows from JSON", len(df_raw))
            break
        except Exception as e:
            logger.warning("  ✗ Failed (%s): %s", source, e)

    if df_raw is None:
        logger.error("All dataset sources failed. Cannot proceed.")
        sys.exit(1)

    logger.info("Columns available: %s", list(df_raw.columns))


    # ── Normalise column names ────────────────────────────────────────────────
    # DetectVul/devign schema:
    #   'func' or 'func_clean' = C source code
    #   'target' = binary vulnerability label (0/1)
    #   'label'  = list of vulnerable line numbers  ← we do NOT want this

    # Pick code column (prefer func_clean over func)
    code_col = None
    for c in ("func_clean", "func", "code", "function", "src"):
        if c in df_raw.columns:
            code_col = c
            break

    # Pick label column — only 'target' is the binary 0/1 label
    label_col = None
    for c in ("target",):
        if c in df_raw.columns:
            label_col = c
            break

    if code_col is None:
        logger.error("Could not find code column. Available: %s", list(df_raw.columns))
        sys.exit(1)
    if label_col is None:
        logger.error("Could not find label column. Available: %s", list(df_raw.columns))
        sys.exit(1)

    logger.info("Using code_col='%s', label_col='%s'", code_col, label_col)
    df_raw = df_raw[[code_col, label_col]].copy()
    df_raw.columns = ["code", "label"]



    # ── Clean: drop empties, keep only non-trivial snippets ──────────────────
    df_raw = df_raw[["code", "label"]].dropna()
    df_raw = df_raw[df_raw["code"].str.strip().str.len() > 30]
    df_raw["label"] = df_raw["label"].astype(int)

    logger.info(
        "After cleaning – vuln: %d | safe: %d",
        int((df_raw["label"] == 1).sum()),
        int((df_raw["label"] == 0).sum()),
    )

    # ── Balance & cap ────────────────────────────────────────────────────────
    half = MAX_SAMPLES // 2
    vuln = df_raw[df_raw["label"] == 1].sample(
        n=min(half, int((df_raw["label"] == 1).sum())),
        random_state=42,
    )
    safe = df_raw[df_raw["label"] == 0].sample(
        n=min(half, int((df_raw["label"] == 0).sum())),
        random_state=42,
    )
    df_bal = pd.concat([vuln, safe]).sample(frac=1, random_state=42).reset_index(drop=True)

    logger.info(
        "Balanced dataset – %d samples (%d vuln + %d safe)",
        len(df_bal),
        int((df_bal["label"] == 1).sum()),
        int((df_bal["label"] == 0).sum()),
    )

    # ── Add required columns ─────────────────────────────────────────────────
    df_bal.insert(0, "snippet_id", range(len(df_bal)))
    df_bal["cwe_type"] = "CWE(Devign)"   # Devign doesn't provide CWE IDs
    df_bal["language"] = "c"

    # ── Feature extraction ────────────────────────────────────────────────────
    logger.info("Extracting code features (regex-based) …")
    from modules.preprocessor import extract_features
    feat_records = []
    for i, row in df_bal.iterrows():
        if i % 500 == 0:
            logger.info("  Features: %d / %d", i, len(df_bal))
        feat_records.append(extract_features(row["code"], "c"))

    feat_df = pd.DataFrame(feat_records)
    df_out  = pd.concat([df_bal.reset_index(drop=True), feat_df], axis=1)

    # ── Fuzzy risk scoring ────────────────────────────────────────────────────
    logger.info("Running fuzzy risk scoring …")
    from modules.fuzzy_module import run_fuzzy_on_dataset
    df_out = run_fuzzy_on_dataset(df_out)

    # ── Save ─────────────────────────────────────────────────────────────────
    df_out.to_csv(str(RAW_CSV), index=False)
    logger.info("Saved processed dataset → %s  (%d rows)", RAW_CSV, len(df_out))
    logger.info("Label distribution:\n%s", df_out["label"].value_counts().to_string())

    return df_out


if __name__ == "__main__":
    import torch
    logger.info("CUDA available: %s", torch.cuda.is_available())
    if torch.cuda.is_available():
        logger.info("GPU: %s", torch.cuda.get_device_name(0))

    df = download_and_convert()

    # ── Phase 2: Generate CodeBERT embeddings on GPU ─────────────────────────
    logger.info("=" * 60)
    logger.info("Generating CodeBERT embeddings on GPU …")
    logger.info("=" * 60)

    from modules.codebert_module import generate_embeddings
    embeddings = generate_embeddings(df)
    logger.info("Embeddings shape: %s", embeddings.shape)

    logger.info("=" * 60)
    logger.info("download_devign.py complete — ready to run train.py")
    logger.info("=" * 60)
