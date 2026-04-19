"""
data_prep.py – Phase 1 & 2: Dataset loading, AST feature extraction,
fuzzy risk scoring, and CodeBERT embedding generation.

Run
---
    python data_prep.py

When no Juliet files are present under data/raw/, the script automatically
generates a synthetic demonstration dataset so the pipeline can be tested
end-to-end.
"""

import os
import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd

# ─── Project root on sys.path ─────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import config
from modules.preprocessor import parse_juliet_dataset, extract_features
from modules.fuzzy_module import run_fuzzy_on_dataset
from modules.codebert_module import generate_embeddings

# ─── Logging setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─── Synthetic demo data ──────────────────────────────────────────────────────

_VULN_TEMPLATES = {
    "CWE89": (
        'void query(char *input) {{ '
        'char sql[256]; '
        'sprintf(sql, "SELECT * FROM users WHERE id=%s", input); '
        'execute(sql); }}'
    ),
    "CWE798": (
        'void connect_db() {{ '
        'char *password = "admin123"; '
        'db_connect("localhost", "root", password); }}'
    ),
    "CWE121": (
        'void copy_data(char *src) {{ '
        'char buf[64]; '
        'strcpy(buf, src); '
        'process(buf); }}'
    ),
    "CWE122": (
        'void allocate() {{ '
        'char *p = (char*)malloc(100); '
        'memcpy(p, user_data, 200); }}'   # heap overflow
    ),
    "CWE401": (
        'void run() {{ '
        'char *mem = (char*)malloc(1024); '
        'process(mem); '
        '/* free() missing */ }}'
    ),
}

_SAFE_TEMPLATES = {
    "CWE89": (
        'void query(char *input) {{ '
        'if (!input) return; '
        'char *stmt = prepare_statement("SELECT * FROM users WHERE id=?", input); '
        'execute(stmt); free(stmt); }}'
    ),
    "CWE798": (
        'void connect_db() {{ '
        'char *password = getenv("DB_PASS"); '
        'if (!password) {{ log_error("Missing DB_PASS"); exit(1); }} '
        'db_connect("localhost", "root", password); }}'
    ),
    "CWE121": (
        'void copy_data(char *src) {{ '
        'char buf[64]; '
        'strncpy(buf, src, sizeof(buf) - 1); '
        'buf[sizeof(buf) - 1] = \'\\0\'; '
        'process(buf); }}'
    ),
    "CWE122": (
        'void allocate() {{ '
        'char *p = (char*)malloc(100); '
        'if (!p) return; '
        'memcpy(p, user_data, 100); '
        'free(p); }}'
    ),
    "CWE401": (
        'void run() {{ '
        'char *mem = (char*)malloc(1024); '
        'if (mem) {{ process(mem); free(mem); }} }}'
    ),
}


def _generate_demo_data(samples_per_cwe: int = 40) -> pd.DataFrame:
    """
    Build a synthetic dataset for pipeline testing when no Juliet data exists.

    Each CWE type contributes ``samples_per_cwe`` snippets, alternating between
    vulnerable and safe variants.

    Parameters
    ----------
    samples_per_cwe : Number of samples (half vuln, half safe) per CWE type.

    Returns
    -------
    pd.DataFrame with the standard dataset schema.
    """
    records = []
    for cwe in config.TARGET_CWE:
        for i in range(samples_per_cwe):
            label    = i % 2          # 0 = safe, 1 = vulnerable
            template = _VULN_TEMPLATES[cwe] if label == 1 else _SAFE_TEMPLATES[cwe]
            code     = template.format()
            feats    = extract_features(code, "c")
            records.append({
                "snippet_id": len(records),
                "code":       code,
                "label":      label,
                "cwe_type":   cwe,
                "language":   "c",
                **feats,
            })

    df = pd.DataFrame(records)
    logger.info("Generated synthetic demo dataset: %d samples.", len(df))
    return df


# ─── Main pipeline ────────────────────────────────────────────────────────────

def main() -> None:
    """Orchestrate Phase 1 (parsing + feature extraction) and Phase 2 (embeddings)."""
    logger.info("=" * 60)
    logger.info("VulnDetect – Data Preparation Pipeline")
    logger.info("=" * 60)

    # Ensure all output directories exist
    for d in [
        config.PROCESSED_DIR,
        config.EMBEDDING_DIR,
        config.MODELS_DIR,
        config.RESULTS_DIR,
    ]:
        os.makedirs(d, exist_ok=True)

    # ── Phase 1a: Parse Juliet dataset ────────────────────────────────────────
    logger.info("Parsing Juliet dataset from: %s", config.RAW_DIR)
    df = parse_juliet_dataset(config.RAW_DIR)

    if df.empty:
        logger.warning(
            "No source files found in %s.\n"
            "Place Juliet Test Suite files under data/raw/<CWE_TYPE>/*.c|cpp|java\n"
            "Falling back to synthetic demo data …",
            config.RAW_DIR,
        )
        df = _generate_demo_data()

    logger.info("Total snippets parsed: %d", len(df))
    logger.info("Label distribution:\n%s", df["label"].value_counts().to_string())
    logger.info("CWE distribution:\n%s",   df["cwe_type"].value_counts().to_string())

    # ── Phase 1b: Fuzzy risk scoring ──────────────────────────────────────────
    logger.info("Running fuzzy risk scoring …")
    df = run_fuzzy_on_dataset(df)

    # ── Save processed dataset ────────────────────────────────────────────────
    csv_path = Path(config.PROCESSED_DIR) / "dataset.csv"
    df.to_csv(str(csv_path), index=False)
    logger.info("Processed dataset saved → %s", csv_path)

    # ── Phase 2: CodeBERT embeddings ──────────────────────────────────────────
    logger.info("Generating CodeBERT embeddings …")
    embeddings = generate_embeddings(df)
    logger.info("Embeddings shape: %s", embeddings.shape)

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("DATASET STATISTICS")
    logger.info("-" * 60)
    logger.info("Total samples        : %d", len(df))
    logger.info("Vulnerable (label=1) : %d", int((df["label"] == 1).sum()))
    logger.info("Benign     (label=0) : %d", int((df["label"] == 0).sum()))
    logger.info("Avg fuzzy risk score : %.4f", df["fuzzy_risk_score"].mean())
    for cwe, grp in df.groupby("cwe_type"):
        logger.info(
            "  %-8s : %5d snippets | %d vulnerable",
            cwe, len(grp), int((grp["label"] == 1).sum()),
        )
    logger.info("=" * 60)
    logger.info("data_prep.py completed successfully.")


if __name__ == "__main__":
    main()
