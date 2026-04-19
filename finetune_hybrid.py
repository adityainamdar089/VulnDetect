"""
finetune_hybrid.py - Train full hybrid FusionClassifier (GraphCodeBERT + Fuzzy Features)
"""

import os
import sys
import logging
from pathlib import Path
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer, 
    AutoModel, 
    get_cosine_schedule_with_warmup
)
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import config

from modules.cross_attention_fusion import FusionClassifier
from modules.fuzzy_security_features import extract_security_features, SECURITY_FEATURE_COLS

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
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

class HybridDataset(Dataset):
    def __init__(self, codes, languages, labels, tokenizer, max_length):
        self.codes = codes
        self.languages = languages
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        feats = extract_security_features(self.codes[idx], self.languages[idx])
        feat_array = np.array([feats[col] for col in SECURITY_FEATURE_COLS], dtype=np.float32)

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
            "fuzzy_features": torch.tensor(feat_array, dtype=torch.float32),
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }

class FusionDataset(Dataset):
    def __init__(self, embeddings, fuzzy_features, labels):
        self.embeddings = embeddings
        self.fuzzy_features = fuzzy_features
        self.labels = labels
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        return {
            "embeddings": torch.tensor(self.embeddings[idx], dtype=torch.float32),
            "fuzzy_features": torch.tensor(self.fuzzy_features[idx], dtype=torch.float32),
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }

def main():
    SANITY_MODE = True
    
    logger.info("="*60)
    logger.info("Hybrid FusionClassifier Training")
    logger.info("="*60)

    df_path = Path(config.PROCESSED_DIR) / "dataset.csv"
    if not df_path.exists():
        logger.error(f"Dataset not found at {df_path}")
        sys.exit(1)
        
    df = pd.read_csv(df_path)
    
    if SANITY_MODE:
        df = df.sample(n=min(500, len(df)), random_state=42).reset_index(drop=True)
        # Re-create dummy splits
        all_idx = np.arange(len(df))
        np.random.shuffle(all_idx)
        train_idx = all_idx[:int(len(df)*0.8)]
        val_idx = all_idx[int(len(df)*0.8):int(len(df)*0.9)]
        test_idx = all_idx[int(len(df)*0.9):]
        logger.info("SANITY MODE ENABLED – Using 500 samples")
    else:
        val_idx = np.load(Path(config.PROCESSED_DIR) / "val_idx.npy")
        test_idx = np.load(Path(config.PROCESSED_DIR) / "test_idx.npy")
        all_idx = np.arange(len(df))
        train_idx = np.setdiff1d(all_idx, np.concatenate([val_idx, test_idx]))
        logger.info(f"FULL TRAINING MODE – Using complete dataset ({len(df)} samples)")
        
    logger.info(f"Splits - Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    
    tokenizer = AutoTokenizer.from_pretrained("microsoft/graphcodebert-base")
    
    emb_path = Path(config.PROCESSED_DIR) / ("X_hybrid_embeddings_sanity.npy" if SANITY_MODE else "X_hybrid_embeddings.npy")
    feat_path = Path(config.PROCESSED_DIR) / ("X_hybrid_fuzzy_sanity.npy" if SANITY_MODE else "X_hybrid_fuzzy.npy")
    
    if emb_path.exists() and feat_path.exists():
        logger.info(f"Loading precomputed embeddings from {emb_path}")
        all_embeddings = np.load(emb_path)
        all_fuzzy = np.load(feat_path)
    else:
        logger.info("Generating GraphCodeBERT embeddings and fuzzy features...")
        hybrid_dataset = HybridDataset(
            df["code"].tolist(), 
            df["language"].tolist() if "language" in df.columns else ["c"]*len(df), 
            df["label"].tolist(), 
            tokenizer, 
            max_length=256
        )
        prep_loader = DataLoader(hybrid_dataset, batch_size=16, shuffle=False)
        
        extractor_model = AutoModel.from_pretrained("microsoft/graphcodebert-base")
        for param in extractor_model.parameters():
            param.requires_grad = False
        extractor_model.to(config.DEVICE)
        extractor_model.eval()
        
        all_embeddings = []
        all_fuzzy = []
        
        with torch.no_grad():
            for batch in tqdm(prep_loader, desc="Extracting embeddings"):
                b_input_ids = batch["input_ids"].to(config.DEVICE)
                b_attn_mask = batch["attention_mask"].to(config.DEVICE)
                b_fuzzy = batch["fuzzy_features"].numpy()
                
                outputs = extractor_model(b_input_ids, attention_mask=b_attn_mask)
                cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                
                all_embeddings.append(cls_embeddings)
                all_fuzzy.append(b_fuzzy)
                
        all_embeddings = np.concatenate(all_embeddings, axis=0)
        all_fuzzy = np.concatenate(all_fuzzy, axis=0)
        
        np.save(emb_path, all_embeddings)
        np.save(feat_path, all_fuzzy)
        logger.info(f"Saved embeddings to {emb_path}")
        
    labels = df["label"].values
    train_fusion_ds = FusionDataset(all_embeddings[train_idx], all_fuzzy[train_idx], labels[train_idx])
    val_fusion_ds = FusionDataset(all_embeddings[val_idx], all_fuzzy[val_idx], labels[val_idx])
    test_fusion_ds = FusionDataset(all_embeddings[test_idx], all_fuzzy[test_idx], labels[test_idx])
    
    batch_size = 64
    train_loader = DataLoader(train_fusion_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_fusion_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_fusion_ds, batch_size=batch_size, shuffle=False)
    
    logger.info("Initializing FusionClassifier...")
    model = FusionClassifier(fuzzy_dim=6, embed_dim=768, num_classes=2, dropout=0.1)
    model.to(config.DEVICE)
    
    epochs = 2 if SANITY_MODE else 10
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    focal_loss_fn = FocalLoss(gamma=2.0)
    
    num_training_steps = epochs * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(num_training_steps * 0.1), 
        num_training_steps=num_training_steps
    )
    
    logger.info(f"Starting Training... Device: {config.DEVICE}")
    logger.info(f"Epochs: {epochs} | Batch Size: {batch_size}")
    
    best_val_f1 = -1.0
    patience_counter = 0
    save_dir = Path(config.MODELS_DIR) 
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = save_dir / "fusion_classifier_best.pt"
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        logger.info(f"\n--- Epoch {epoch+1}/{epochs} ---")
        
        for step, batch in enumerate(train_loader):
            b_emb = batch["embeddings"].to(config.DEVICE)
            b_fuz = batch["fuzzy_features"].to(config.DEVICE)
            b_lbl = batch["label"].to(config.DEVICE)
            
            optimizer.zero_grad()
            
            logits = model(b_fuz, b_emb)
            loss = focal_loss_fn(logits, b_lbl)
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            
            if step % 50 == 0 or step == len(train_loader) - 1:
                logger.info(f"  Step [{step}/{len(train_loader)}] | Focal Loss = {loss.item():.4f}")
                
        avg_train_loss = total_loss / len(train_loader)
        
        model.eval()
        val_preds, val_labels, val_probs = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                b_emb = batch["embeddings"].to(config.DEVICE)
                b_fuz = batch["fuzzy_features"].to(config.DEVICE)
                b_lbl = batch["label"].to(config.DEVICE)
                
                logits = model(b_fuz, b_emb)
                probs = F.softmax(logits, dim=1)[:, 1]
                preds = torch.argmax(logits, dim=1)
                
                val_preds.extend(preds.cpu().numpy())
                val_probs.extend(probs.cpu().numpy())
                val_labels.extend(b_lbl.cpu().numpy())
                
        acc = accuracy_score(val_labels, val_preds)
        f1 = f1_score(val_labels, val_preds)
        logger.info(f"=== Epoch {epoch+1} Summary ===")
        logger.info(f"Train Loss : {avg_train_loss:.4f}")
        logger.info(f"Val Acc    : {acc:.4f}")
        logger.info(f"Val F1     : {f1:.4f}\n")
        
        if f1 > best_val_f1:
            best_val_f1 = f1
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"New best model saved to {best_model_path}")
        else:
            patience_counter += 1
            if patience_counter >= 3:
                logger.info("Early stopping triggered")
                break
                
    logger.info("--- Testing Best Model ---")
    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path, map_location=config.DEVICE))
    model.eval()
    
    test_preds, test_labels, test_probs = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            b_emb = batch["embeddings"].to(config.DEVICE)
            b_fuz = batch["fuzzy_features"].to(config.DEVICE)
            b_lbl = batch["label"].to(config.DEVICE)
            
            logits = model(b_fuz, b_emb)
            probs = F.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)
            
            test_preds.extend(preds.cpu().numpy())
            test_probs.extend(probs.cpu().numpy())
            test_labels.extend(b_lbl.cpu().numpy())
            
    test_acc = accuracy_score(test_labels, test_preds)
    test_f1 = f1_score(test_labels, test_preds)
    test_auc = roc_auc_score(test_labels, test_probs)
    
    logger.info(f"Test Accuracy : {test_acc:.4f}")
    logger.info(f"Test F1       : {test_f1:.4f}")
    logger.info(f"Test AUC-ROC  : {test_auc:.4f}")
    logger.info("\nClassification Report:\n" + classification_report(test_labels, test_preds))

if __name__ == '__main__':
    main()
