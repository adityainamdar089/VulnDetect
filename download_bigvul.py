"""
download_bigvul.py – Download the BigVul dataset from HuggingFace and
convert it to VulnDetect's expected CSV format.
"""

import sys
import logging
import os
from pathlib import Path
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
from datasets import load_dataset
from modules.preprocessor import extract_features
from modules.fuzzy_module import run_fuzzy_on_dataset

OUT_CSV = Path(config.PROCESSED_DIR) / "bigvul_dataset.csv"
MAX_SAMPLES = 50000

def map_cwe(cwe_str):
    if pd.isna(cwe_str):
        return None
    # Ensure format like CWE89
    cwe_str = str(cwe_str).upper()
    cwe_str = cwe_str.replace("CWE-", "CWE")
    if cwe_str in config.TARGET_CWE:
        return cwe_str
    return None

def download_and_convert():
    logger.info("=" * 60)
    logger.info("Downloading BigVul dataset from HuggingFace …")
    logger.info("=" * 60)
    
    SOURCES = [
        "benjaminbeilby/bigvul",
        "mitre/bigvul",
        "VulnerabilityDetection/BigVul"
    ]
    
    df_raw = None
    for src in SOURCES:
        try:
            logger.info("Trying source: %s", src)
            ds = load_dataset(src)
            dfs = [ds[split].to_pandas() for split in ds.keys()]
            df_raw = pd.concat(dfs, ignore_index=True)
            logger.info("Loaded %d rows from %s", len(df_raw), src)
            break
        except Exception as e:
            logger.warning("Source %s failed: %s", src, e)
            
    if df_raw is None:
        logger.error("Failed to load dataset.")
        sys.exit(1)
        
    cwe_col = next((c for c in ["cwe_id", "CWE", "cwe"] if c in df_raw.columns), None)
    if not cwe_col:
        logger.error("No CWE column found.")
        sys.exit(1)
        
    records = []
    for _, row in df_raw.iterrows():
        cwe = map_cwe(row[cwe_col])
        if not cwe: 
            continue
        
        func_before = row.get("func_before")
        func_after = row.get("func_after")
        
        if pd.notna(func_before) and str(func_before).strip():
            records.append({"code": str(func_before), "label": 1, "cwe_type": cwe})
            
        if pd.notna(func_after) and str(func_after).strip():
            records.append({"code": str(func_after), "label": 0, "cwe_type": cwe})
            
    df_clean = pd.DataFrame(records)
    logger.info("Found %d samples matching TARGET_CWE.", len(df_clean))
    if len(df_clean) == 0:
        logger.error("No samples matched TARGET_CWE. Exiting.")
        sys.exit(1)
    
    df_vuln = df_clean[df_clean["label"] == 1]
    df_safe = df_clean[df_clean["label"] == 0]
    
    half = MAX_SAMPLES // 2
    df_vuln_sampled = df_vuln.sample(n=min(half, len(df_vuln)), random_state=42)
    df_safe_sampled = df_safe.sample(n=min(half, len(df_safe)), random_state=42)
    
    df_bal = pd.concat([df_vuln_sampled, df_safe_sampled]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    df_bal.insert(0, "snippet_id", range(len(df_bal)))
    df_bal["language"] = "c"
    
    logger.info("Extracting features...")
    feat_records = []
    for i, row in df_bal.iterrows():
        if i % 1000 == 0:
            logger.info("Feature extraction: %d / %d", i, len(df_bal))
        feat_records.append(extract_features(row["code"], "c"))
        
    feat_df = pd.DataFrame(feat_records)
    df_out = pd.concat([df_bal, feat_df], axis=1)
    
    logger.info("Running fuzzy logic...")
    df_out = run_fuzzy_on_dataset(df_out)
    
    os.makedirs(config.PROCESSED_DIR, exist_ok=True)
    df_out.to_csv(str(OUT_CSV), index=False)
    logger.info("Saved dataset to %s", OUT_CSV)
    
def merge_datasets():
    devign_csv = Path(config.PROCESSED_DIR) / "dataset.csv"
    bigvul_csv = Path(config.PROCESSED_DIR) / "bigvul_dataset.csv"
    
    logger.info("== Merging datasets ==")
    
    df1 = pd.read_csv(str(devign_csv)) if devign_csv.exists() else pd.DataFrame()
    df2 = pd.read_csv(str(bigvul_csv)) if bigvul_csv.exists() else pd.DataFrame()
    
    df_merged = pd.concat([df1, df2], ignore_index=True)
    df_merged = df_merged.drop_duplicates(subset=["code"])
    df_merged = df_merged.sample(frac=1, random_state=42).reset_index(drop=True)
    
    if "snippet_id" in df_merged.columns:
        df_merged = df_merged.drop(columns=["snippet_id"])
    df_merged.insert(0, "snippet_id", range(len(df_merged)))
    
    out_path = Path(config.PROCESSED_DIR) / "merged_dataset.csv"
    df_merged.to_csv(str(out_path), index=False)
    logger.info("Merged dataset saved to %s", out_path)
    if "label" in df_merged.columns:
        logger.info("Final label distribution:\n%s", df_merged["label"].value_counts().to_string())

if __name__ == "__main__":
    download_and_convert()
    merge_datasets()
