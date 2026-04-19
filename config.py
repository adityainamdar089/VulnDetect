"""
config.py – Central configuration for the Hybrid Fuzzy-Transformer Vulnerability Detection System.

All paths, hyperparameters, and settings are defined here.
Never hardcode paths in other modules — always import from config.
"""

import os
import torch

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR       = "D:/VulnDetect"
DATA_DIR       = os.path.join(BASE_DIR, "data")
RAW_DIR        = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR  = os.path.join(DATA_DIR, "processed")
EMBEDDING_DIR  = os.path.join(DATA_DIR, "embeddings")
MODELS_DIR     = os.path.join(BASE_DIR, "models")
RESULTS_DIR    = os.path.join(BASE_DIR, "results")

# ─── Device ───────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─── Model ────────────────────────────────────────────────────────────────────
CODEBERT_MODEL = "microsoft/graphcodebert-base"

# ─── Training Hyperparameters ──────────────────────────────────────────────────
BATCH_SIZE    = 16
MAX_TOKEN_LEN = 512
TEST_SPLIT    = 0.1
VAL_SPLIT     = 0.1
RANDOM_SEED   = 42
NUM_EPOCHS    = 30
LEARNING_RATE = 2e-5
WEIGHT_DECAY  = 0.01

# ─── Target CWE Types ─────────────────────────────────────────────────────────
TARGET_CWE = ["CWE89", "CWE798", "CWE121", "CWE122", "CWE401"]

CWE_DESCRIPTIONS = {
    "CWE89":  "SQL Injection",
    "CWE798": "Hard-coded Credentials",
    "CWE121": "Stack-based Buffer Overflow",
    "CWE122": "Heap-based Buffer Overflow",
    "CWE401": "Memory Leak",
}

# ─── Fusion Weights ───────────────────────────────────────────────────────────
FUSION_ALPHA = 0.7   # weight for CodeBERT embedding in weighted_concat
FUSION_BETA  = 0.3   # weight for fuzzy score in weighted_concat

# ─── Classifier Hyperparameters ───────────────────────────────────────────────
RF_N_ESTIMATORS = 200
XGB_TREE_METHOD  = "hist"   # XGBoost 2.0+: always use 'hist'; GPU selected via XGB_DEVICE
XGB_DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
