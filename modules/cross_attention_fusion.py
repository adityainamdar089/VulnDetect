import torch
import torch.nn as nn

class CrossAttentionFusion(nn.Module):
    def __init__(self, fuzzy_dim=6, embed_dim=768, num_heads=8, dropout=0.1):
        super(CrossAttentionFusion, self).__init__()
        self.fuzzy_proj = nn.Linear(fuzzy_dim, embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, fuzzy_features, codebert_embeddings):
        # fuzzy_features: (batch, fuzzy_dim)
        # codebert_embeddings: (batch, embed_dim)
        
        query = self.fuzzy_proj(fuzzy_features).unsqueeze(1)
        key_value = codebert_embeddings.unsqueeze(1)
        
        attn_out, _ = self.attention(query, key_value, key_value)
        out = self.layer_norm(attn_out)
        out = self.dropout(out)
        return out.squeeze(1)

class FusionClassifier(nn.Module):
    def __init__(self, fuzzy_dim=6, embed_dim=768, num_classes=2, dropout=0.1):
        super(FusionClassifier, self).__init__()
        self.fusion = CrossAttentionFusion(
            fuzzy_dim=fuzzy_dim, 
            embed_dim=embed_dim, 
            num_heads=8, 
            dropout=dropout
        )
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, fuzzy_features, codebert_embeddings):
        fused = self.fusion(fuzzy_features, codebert_embeddings)
        return self.classifier(fused)

if __name__ == "__main__":
    batch_size = 2
    fuzzy_dummy = torch.rand(batch_size, 6)
    codebert_dummy = torch.rand(batch_size, 768)
    
    model = FusionClassifier(fuzzy_dim=6, embed_dim=768, num_classes=2)
    output = model(fuzzy_dummy, codebert_dummy)
    
    print(f"Output shape: {output.shape}")
    assert output.shape == (2, 2), "Output shape must be (2, 2)"
    print("CrossAttentionFusion test passed!")
