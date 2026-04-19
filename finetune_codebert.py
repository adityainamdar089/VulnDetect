"""
finetune_codebert.py – Full CodeBERT Fine-Tuning with Maximum Accuracy Upgrades
"""

import os
import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    get_cosine_schedule_with_warmup
)
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import config

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss) # prevents nans when probability 0
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

class CodeDataset(Dataset):
    def __init__(self, codes, labels, tokenizer, max_length):
        self.codes = codes
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.codes[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }

def main():
    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory < 6 * 1024 * 1024 * 1024:
        print("WARNING: This script needs at least 8GB VRAM and should be run on Kaggle T4.")
        sys.exit(1)

    logger.info("="*60)
    logger.info("Phase 10: Maximum Accuracy GraphCodeBERT Fine-Tuning")
    logger.info("="*60)

    df_path = Path(config.PROCESSED_DIR) / "dataset.csv"
    if not df_path.exists():
        logger.error(f"Dataset not found at {df_path}")
        sys.exit(1)
        
    df = pd.read_csv(df_path)
    logger.info(f"Loaded dataset with {len(df)} samples.")
    
    val_idx = np.load(Path(config.PROCESSED_DIR) / "val_idx.npy")
    test_idx = np.load(Path(config.PROCESSED_DIR) / "test_idx.npy")
    all_idx = np.arange(len(df))
    train_idx = np.setdiff1d(all_idx, np.concatenate([val_idx, test_idx]))

    logger.info(f"Splits - Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    
    tokenizer = AutoTokenizer.from_pretrained(config.CODEBERT_MODEL)
    
    # MAXIMUM ACCURACY UPGRADES
    max_length = 512              # Double context window
    batch_size = 4                # Lowered physical batch size to fit 512 in 4GB VRAM
    accumulation_steps = 8        # Effectively batch_size = 32

    train_dataset = CodeDataset(df.iloc[train_idx]["code"].tolist(), df.iloc[train_idx]["label"].tolist(), tokenizer, max_length)
    val_dataset = CodeDataset(df.iloc[val_idx]["code"].tolist(), df.iloc[val_idx]["label"].tolist(), tokenizer, max_length)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    logger.info(f"Initializing {config.CODEBERT_MODEL}...")
    model = AutoModelForSequenceClassification.from_pretrained(config.CODEBERT_MODEL, num_labels=2)
    model.to(config.DEVICE)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    focal_loss_fn = FocalLoss(gamma=2.0)
    
    epochs = 3 
    # Calculate effective steps due to accumulation
    effective_steps_per_epoch = len(train_loader) // accumulation_steps
    num_training_steps = epochs * effective_steps_per_epoch
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(num_training_steps * 0.1), 
        num_training_steps=num_training_steps
    )
    
    logger.info(f"Starting Fine-Tuning... Device: {config.DEVICE}")
    logger.info(f"Epochs: {epochs} | Batch Size: {batch_size} (Acc: {accumulation_steps}) | Context: {max_length}")
    
    scaler = torch.amp.GradScaler('cuda')
    
    best_val_f1 = -1.0
    patience_counter = 0
    save_dir = Path(config.MODELS_DIR) / "codebert_finetuned"
    os.makedirs(save_dir, exist_ok=True)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        
        logger.info(f"\n--- Epoch {epoch+1}/{epochs} ---")
        
        for step, batch in enumerate(train_loader):
            b_input_ids = batch["input_ids"].to(config.DEVICE)
            b_attn_mask = batch["attention_mask"].to(config.DEVICE)
            b_labels = batch["label"].to(config.DEVICE)
            
            with torch.amp.autocast('cuda'):
                outputs = model(b_input_ids, attention_mask=b_attn_mask)
                # Apply Focal Loss rather than standard CrossEntropy
                loss = focal_loss_fn(outputs.logits, b_labels)
                # Scale loss to account for accumulation
                loss = loss / accumulation_steps
            
            scaler.scale(loss).backward()
            
            if (step + 1) % accumulation_steps == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                
            total_loss += loss.item() * accumulation_steps  # Revert scaling for logging
            
            if step % (50 * accumulation_steps) == 0 or step == len(train_loader) - 1:
                logger.info(f"  Step [{step}/{len(train_loader)}] | Focal Loss = {loss.item() * accumulation_steps:.4f}")
                
        avg_train_loss = total_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                b_input_ids = batch["input_ids"].to(config.DEVICE)
                b_attn_mask = batch["attention_mask"].to(config.DEVICE)
                b_labels = batch["label"].to(config.DEVICE)
                
                with torch.amp.autocast('cuda'):
                    outputs = model(b_input_ids, attention_mask=b_attn_mask)
                    logits = outputs.logits
                
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_labels.extend(b_labels.cpu().numpy())
                
        acc = accuracy_score(val_labels, val_preds)
        f1 = f1_score(val_labels, val_preds)
        logger.info(f"=== Epoch {epoch+1} Summary ===")
        logger.info(f"Train Loss : {avg_train_loss:.4f}")
        logger.info(f"Val Acc    : {acc:.4f}")
        logger.info(f"Val F1     : {f1:.4f}\n")
        
        if f1 > best_val_f1:
            best_val_f1 = f1
            patience_counter = 0
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            logger.info("New best model saved")
        else:
            patience_counter += 1
            if patience_counter >= 2:
                print("early stopping triggered")
                break

if __name__ == '__main__':
    main()
