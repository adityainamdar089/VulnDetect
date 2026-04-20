%%writefile /kaggle/working/VulnDetect/train_codebert_clean.py

import os, sys, logging, numpy as np, pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
import warnings
warnings.filterwarnings("ignore")

# ── Logging ───────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────
PROC_DIR       = "/kaggle/working/VulnDetect/data/processed"
MODEL_SAVE_DIR = "/kaggle/working/VulnDetect/models/codebert_finetuned"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

CODEBERT_MODEL      = "microsoft/graphcodebert-base"
MAX_TOKEN_LEN       = 512
BATCH_SIZE          = 8
GRAD_ACCUM_STEPS    = 8       # effective batch = 64
LEARNING_RATE       = 2e-5
WEIGHT_DECAY        = 0.01
NUM_EPOCHS          = 10
EARLY_STOP_PATIENCE = 2
FOCAL_GAMMA         = 2.0
DEVICE              = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Device: {DEVICE} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

# ── Dataset ───────────────────────────────────────────────────
class VulnDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.codes  = df["code"].astype(str).tolist()
        self.labels = df["label"].astype(int).tolist()
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self): return len(self.codes)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.codes[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label":          torch.tensor(self.labels[idx], dtype=torch.long)
        }

# ── Model ─────────────────────────────────────────────────────
class CodeBERTClassifier(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.bert    = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.fc      = nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = self.dropout(out.last_hidden_state[:, 0, :])
        return self.fc(cls)

# ── Focal Loss ────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=1.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ce    = nn.CrossEntropyLoss(reduction="none")

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        pt      = torch.exp(-ce_loss)
        return (self.alpha * (1 - pt) ** self.gamma * ce_loss).mean()

# ── Train one epoch ───────────────────────────────────────────
def train_epoch(model, loader, optimizer, scheduler, scaler, criterion):
    model.train()
    total_loss = 0
    optimizer.zero_grad()

    for step, batch in enumerate(loader):
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels         = batch["label"].to(DEVICE)

        with torch.amp.autocast("cuda"):
            logits = model(input_ids, attention_mask)
            loss   = criterion(logits, labels) / GRAD_ACCUM_STEPS

        scaler.scale(loss).backward()

        if (step + 1) % GRAD_ACCUM_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * GRAD_ACCUM_STEPS

    return total_loss / len(loader)

# ── Evaluate ──────────────────────────────────────────────────
def evaluate(model, loader):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels         = batch["label"].to(DEVICE)

            with torch.amp.autocast("cuda"):
                logits = model(input_ids, attention_mask)

            probs  = torch.softmax(logits, dim=1)[:, 1]
            preds  = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    f1  = f1_score(all_labels, all_preds)
    acc = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    return f1, acc, auc

# ── Main ──────────────────────────────────────────────────────
def main():
    logger.info("Loading datasets...")
    train_df = pd.read_csv(f"{PROC_DIR}/train.csv")
    val_df   = pd.read_csv(f"{PROC_DIR}/val.csv")
    test_df  = pd.read_csv(f"{PROC_DIR}/test.csv")

    logger.info(f"Train={len(train_df)} | Val={len(val_df)} | Test={len(test_df)}")

    tokenizer = AutoTokenizer.from_pretrained(CODEBERT_MODEL)

    train_loader = DataLoader(VulnDataset(train_df, tokenizer, MAX_TOKEN_LEN),
                              batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(VulnDataset(val_df,   tokenizer, MAX_TOKEN_LEN),
                              batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(VulnDataset(test_df,  tokenizer, MAX_TOKEN_LEN),
                              batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    model     = CodeBERTClassifier(CODEBERT_MODEL).to(DEVICE)
    criterion = FocalLoss(gamma=FOCAL_GAMMA)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    total_steps   = (len(train_loader) // GRAD_ACCUM_STEPS) * NUM_EPOCHS
    warmup_steps  = int(total_steps * 0.10)
    scheduler     = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler        = torch.amp.GradScaler("cuda")

    best_f1       = 0.0
    patience_ctr  = 0

    logger.info("=" * 60)
    logger.info("Starting GraphCodeBERT Fine-Tuning")
    logger.info("=" * 60)

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, scaler, criterion)
        val_f1, val_acc, val_auc = evaluate(model, val_loader)

        logger.info(f"Epoch {epoch}/{NUM_EPOCHS} | Loss={train_loss:.4f} | "
                    f"Val F1={val_f1:.4f} | Val Acc={val_acc:.4f} | Val AUC={val_auc:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_ctr = 0
            model.bert.save_pretrained(MODEL_SAVE_DIR)
            tokenizer.save_pretrained(MODEL_SAVE_DIR)
            torch.save(model.state_dict(), f"{MODEL_SAVE_DIR}/classifier_head.pt")
            logger.info(f"  ✅ Best model saved! F1={best_f1:.4f}")
        else:
            patience_ctr += 1
            logger.info(f"  No improvement. Patience {patience_ctr}/{EARLY_STOP_PATIENCE}")
            if patience_ctr >= EARLY_STOP_PATIENCE:
                logger.info("Early stopping triggered.")
                break

    # Final test eval
    logger.info("\nLoading best model for test evaluation...")
    model.load_state_dict(torch.load(f"{MODEL_SAVE_DIR}/classifier_head.pt"))
    test_f1, test_acc, test_auc = evaluate(model, test_loader)
    logger.info("=" * 60)
    logger.info(f"TEST RESULTS → F1={test_f1:.4f} | Acc={test_acc:.4f} | AUC={test_auc:.4f}")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
