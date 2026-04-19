"""Quick smoke test – CodeBERT embedder."""
import sys
sys.path.insert(0, ".")

print("Importing CodeBERT embedder …")
from modules.codebert_module import _CodeBERTEmbedder

print("Loading model …")
emb = _CodeBERTEmbedder()

snippets = [
    'void login(char *u) { char q[256]; sprintf(q, "%s", u); db_execute(q); }',
    'void safe() { char *p = getenv("DB_PASS"); if (!p) exit(1); }',
]

print("Running embed_batch …")
out = emb.embed_batch(snippets)

print(f"Output shape : {out.shape}")          # expect (2, 768)
print(f"First 5 vals : {out[0][:5].tolist()}")
print("SUCCESS – CodeBERT embedder is working!")
