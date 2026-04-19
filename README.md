# Hybrid Fuzzy-Transformer Universal Code Vulnerability Detection System

A research-grade vulnerability detection pipeline that fuses **scikit-fuzzy** risk assessment with **Microsoft CodeBERT** semantic embeddings to classify C / C++ / Java code snippets across five CWE categories.

---

## 📁 Project Structure

```
D:/VulnDetect/
├── data/
│   ├── raw/                  # Juliet Test Suite source files → <CWE_TYPE>/*.c|cpp|java
│   ├── processed/            # Cleaned CSVs + NumPy feature matrices
│   └── embeddings/           # Saved CodeBERT .npy embedding files
├── modules/
│   ├── preprocessor.py       # Tree-sitter AST parsing + 5-feature extraction
│   ├── fuzzy_module.py       # scikit-fuzzy Mamdani risk scoring
│   ├── codebert_module.py    # CodeBERT [CLS] embedding generation (GPU)
│   ├── fusion.py             # 3 fusion strategies (concat, weighted, attention MLP)
│   ├── classifier.py         # RandomForest, XGBoost, CodeBERT head
│   └── evaluator.py          # Metrics, plots, ablation table
├── data_prep.py              # Phase 1+2: parse → fuzz → embed
├── train.py                  # Phase 6: train all classifiers
├── evaluate.py               # Phase 7: ablation study + plots
├── app.py                    # Phase 8: Gradio UI
├── config.py                 # All paths, hyperparameters, device settings
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1 — Create & activate the conda environment

```bash
conda create -n vulndetect python=3.10 -y
conda activate vulndetect
```

### 2 — Install PyTorch with CUDA 12.1

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 3 — Install remaining dependencies

```bash
pip install -r requirements.txt
```

### 4 — (Optional) Add Juliet Test Suite data

Download from the [NIST Juliet Test Suite](https://samate.nist.gov/SARD/test-suites/111) and place files under:

```
data/raw/
├── CWE89/   ← SQL Injection
├── CWE798/  ← Hard-coded Credentials
├── CWE121/  ← Stack Buffer Overflow
├── CWE122/  ← Heap Buffer Overflow
└── CWE401/  ← Memory Leak
```

> **Without Juliet data** the pipeline automatically generates a synthetic 200-sample dataset so you can test end-to-end immediately.

---

## 🔄 Run Order

```bash
# Phase 1+2 – data preparation (parse, fuzz-score, embed)
python data_prep.py

# Phase 6 – train all classifiers + ablation baselines
python train.py

# Phase 7 – evaluation, ablation study, plots
python evaluate.py

# Phase 8 – launch Gradio web UI
python app.py
```

The Gradio app will be available at **http://127.0.0.1:7860**.

---

## 🧠 Architecture Overview

```
Code Snippet
     │
     ├──► Tree-sitter / Regex ──► 5 normalised features
     │                                    │
     │                           scikit-fuzzy FIS
     │                                    │
     │                           fuzzy_risk_score  (1-dim)
     │
     └──► CodeBERT ([CLS])  ──► 768-dim embedding
                                         │
                              ┌──────────▼─────────────┐
                              │   Fusion Module         │
                              │  · simple_concat (769)  │
                              │  · weighted_concat(769) │
                              │  · attention MLP (769)  │
                              └──────────┬──────────────┘
                                         │
                              ┌──────────▼──────────────┐
                              │   Classifiers            │
                              │  · RandomForest          │
                              │  · XGBoost               │
                              │  · CodeBERT Head (MLP)   │
                              └─────────────────────────┘
```

---

## 🎯 Target CWE Types

| CWE ID | Name |
|--------|------|
| CWE89  | SQL Injection |
| CWE798 | Hard-coded Credentials |
| CWE121 | Stack-based Buffer Overflow |
| CWE122 | Heap-based Buffer Overflow |
| CWE401 | Memory Leak |

---

## ⚙️ Configuration (`config.py`)

All settings are centralised in `config.py`. Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DEVICE` | auto | `"cuda"` or `"cpu"` |
| `CODEBERT_MODEL` | `microsoft/codebert-base` | HuggingFace model ID |
| `BATCH_SIZE` | 16 | Embedding batch size |
| `MAX_TOKEN_LEN` | 512 | Max tokens per snippet |
| `NUM_EPOCHS` | 10 | CodeBERT head training epochs |
| `FUSION_ALPHA` | 0.7 | Embedding weight in weighted concat |
| `FUSION_BETA` | 0.3 | Fuzzy score weight in weighted concat |
| `RF_N_ESTIMATORS` | 200 | Random Forest tree count |

---

## 📊 Outputs

| File | Description |
|------|-------------|
| `data/processed/dataset.csv` | Processed dataset with features + fuzzy scores |
| `data/embeddings/embeddings.npy` | CodeBERT embeddings (N × 768) |
| `models/*.joblib / *.pt / *.json` | Saved models |
| `results/comparison_table.csv` | Ablation study metrics |
| `results/confusion_matrix_*.png` | Per-model confusion matrices |
| `results/roc_curve_*.png` | ROC curves |
| `results/per_cwe_f1_*.png` | Per-CWE F1 bar charts |

---

## 📝 Notes

- **GPU OOM**: The pipeline gracefully falls back to CPU if a CUDA out-of-memory error occurs.
- **Logging**: All modules use Python `logging` (not `print`) at INFO level.
- **Embeddings cache**: `generate_embeddings()` skips re-computation if `embeddings.npy` already exists on disk.
- **Fuzzy fallback**: If `scikit-fuzzy` is not installed, a weighted arithmetic formula is used instead.
