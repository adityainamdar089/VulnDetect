"""
app.py – Phase 8: Gradio web UI for the Hybrid Fuzzy-Transformer
Vulnerability Detection System.

Run
---
    python app.py

The app loads the best available trained model on startup, then accepts a
code snippet + language and returns:
    • Vulnerable: Yes / No
    • Vulnerability Type: CWE ID + name
    • Severity: Low / Medium / High  (from fuzzy risk score)
    • Confidence: percentage
    • Affected Lines: detected line numbers (best-effort)
"""

import sys
import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import gradio as gr

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import config
from modules.preprocessor import extract_features
from modules.fuzzy_module import compute_fuzzy_risk
from modules.fusion import attention_fusion
from modules.agent_referee import request_referee_review

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─── Model loading ────────────────────────────────────────────────────────────

_finetuned_model = None
_finetuned_tokenizer = None

def _try_load_finetuned_model():
    """Attempt to load the fine-tuned GraphCodeBERT model."""
    global _finetuned_model, _finetuned_tokenizer
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        import torch
        model_path = Path(config.MODELS_DIR) / "codebert_finetuned"
        if model_path.exists():
            logger.info("Loading fine-tuned GraphCodeBERT from %s", model_path)
            _finetuned_tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            _finetuned_model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
            _finetuned_model.to(config.DEVICE)
            _finetuned_model.eval()
            logger.info("Successfully loaded fine-tuned GraphCodeBERT.")
        else:
            logger.warning("codebert_finetuned model not found. Using fuzzy fallback.")
    except Exception as exc:
        logger.error("Failed to load fine-tuned model: %s", exc)

_try_load_finetuned_model()

# ─── Helper utilities ─────────────────────────────────────────────────────────

_CWE_PATTERNS = {
    # SQL Injection – unsanitised format strings / string-concat queries
    "CWE89":  [r"sprintf\s*\([^,]+,\s*[^,]*%s", r"\.execute\s*\(.*\+",
               r"query\s*\+\s*", r"\bexecute\s*\(sql"],
    # Hard-coded credentials – literal string assigned to credential variable
    "CWE798": [r'\bpassword\s*=\s*["\'][^\'"]{3,}["\']',
               r'\bsecret\s*=\s*["\'][^\'"]{3,}["\']',
               r'\bapi_key\s*=\s*["\'][^\'"]{3,}["\']'],
    # Stack-based buffer overflow – unsafe string ops on fixed buffers
    "CWE121": [r"\bstrcpy\s*\(", r"\bgets\s*\(",
               r"\bstrcat\s*\([^,]+,\s*[^,)]+\)"],
    # Heap overflow – memcpy with size > allocation
    "CWE122": [r"memcpy\s*\([^,]+,[^,]+,\s*\d{3,}\)",
               r"\bcalloc\s*\(\s*0\s*,"],
    # Memory leak – malloc/new present but NO matching free/delete in snippet
    "CWE401": [r"\bmalloc\s*\(", r"\bnew\s+\w+"],
}

_SEVERITY_MAP = {
    (0.0, 0.33): "🟢 Low",
    (0.33, 0.66): "🟡 Medium",
    (0.66, 1.01): "🔴 High",
}


def _fuzzy_severity(score: float) -> str:
    for (lo, hi), label in _SEVERITY_MAP.items():
        if lo <= score < hi:
            return label
    return "🔴 High"


def _detect_cwe(code: str) -> str:
    """Heuristically guess the most likely CWE type from code patterns."""
    best_cwe, best_hits = "Unknown", 0
    for cwe, patterns in _CWE_PATTERNS.items():
        hits = sum(1 for p in patterns if re.search(p, code, re.IGNORECASE | re.DOTALL))
        # CWE401: only flag if malloc/new is present but free/delete is absent
        if cwe == "CWE401" and hits > 0:
            if re.search(r"\bfree\s*\(", code, re.IGNORECASE) or \
               re.search(r"\bdelete\b", code, re.IGNORECASE):
                hits = 0  # free/delete found → not a leak
        if hits > best_hits:
            best_hits, best_cwe = hits, cwe
    return best_cwe


def _detect_affected_lines(code: str, cwe: str) -> str:
    """Return a comma-separated list of affected line numbers (best-effort)."""
    lines      = code.splitlines()
    patterns   = _CWE_PATTERNS.get(cwe, [])
    hit_lines  = []
    for lineno, line in enumerate(lines, start=1):
        for p in patterns:
            if re.search(p, line, re.IGNORECASE):
                hit_lines.append(str(lineno))
                break
    return ", ".join(hit_lines) if hit_lines else "N/A"


# ─── Core analysis function ───────────────────────────────────────────────────

def analyze_code(code: str, language: str) -> tuple[str, str, str, str, str, str]:
    """
    Run the full detection pipeline on user-supplied code.

    Returns
    -------
    (vulnerable, vuln_type, severity, confidence, affected_lines, llm_reasoning)  - all str.
    """
    if not code.strip():
        return "—", "—", "—", "—", "—", "—"

    language = language.lower().replace("+", "p")   # "C++" → "cpp"

    # 1. Extract features
    try:
        features = extract_features(code, language)
    except Exception as exc:
        logger.warning("Feature extraction failed: %s", exc)
        features = {k: 0.5 for k in [
            "input_validation_score", "sensitive_data_exposure",
            "access_control_strength", "resource_management_score",
            "control_flow_complexity",
        ]}

    # 2. Fuzzy risk score
    fuzzy_score = compute_fuzzy_risk(features)
    logger.info("Fuzzy score: %.4f | Features: %s", fuzzy_score, features)

    # 3. Regex-based CWE detection (PRIMARY signal – high precision rules)
    cwe = _detect_cwe(code)
    regex_hit = cwe != "Unknown"

    # 4. Neural Network semantic analysis (Secondary signal)
    clf_vulnerable: Optional[bool] = None
    clf_conf: float = 0.0

    if _finetuned_model is not None and _finetuned_tokenizer is not None:
        try:
            enc = _finetuned_tokenizer(
                code,
                padding="max_length",
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            b_input_ids = enc["input_ids"].to(config.DEVICE)
            b_attn_mask = enc["attention_mask"].to(config.DEVICE)
            
            with torch.no_grad():
                with torch.amp.autocast('cuda'):
                    outputs = _finetuned_model(b_input_ids, attention_mask=b_attn_mask)
                    probs = torch.softmax(outputs.logits, dim=1)[0].cpu().numpy()
            
            pred = int(np.argmax(probs))
            clf_conf = float(probs[pred])
            clf_vulnerable = pred == 1
            logger.info("GraphCodeBERT: vulnerability_detected=%s conf=%.4f", clf_vulnerable, clf_conf)
        except Exception as exc:
            logger.warning("GraphCodeBERT inference failed: %s", exc)

    # 5. Decision fusion
    #    Priority: regex hit > confident classifier > fuzzy threshold
    if regex_hit:
        # High-precision regex matched → always flag as vulnerable
        is_vulnerable = True
        if clf_vulnerable is not None and clf_conf > 0.60:
            confidence = clf_conf
        else:
            confidence = max(fuzzy_score, 0.65)   # regex hit → at least 65% conf
    elif clf_vulnerable is not None and clf_conf > 0.60:
        is_vulnerable = clf_vulnerable
        confidence    = clf_conf
    else:
        # Pure fuzzy fallback
        is_vulnerable = fuzzy_score >= 0.55
        confidence    = fuzzy_score

    # 6. Premium SWE Agent Referee
    # Always invoke the AI Referee for the premium experience
    logger.info("Invoking Premium Agent Referee...")
    ref_vuln, reason, ref_conf = request_referee_review(code, confidence, cwe)
    is_vulnerable = ref_vuln
    confidence = max(confidence, ref_conf) # Take the higher confidence
    llm_reason_str = reason
        
    # 7. CWE label
    if not is_vulnerable:
        cwe = "N/A"
    cwe_name  = config.CWE_DESCRIPTIONS.get(cwe, "")
    vuln_type = f"{cwe} – {cwe_name}" if cwe not in ("N/A", "Unknown") and cwe_name else cwe

    # 8. Affected lines
    affected = _detect_affected_lines(code, cwe) if is_vulnerable and cwe != "N/A" else "N/A"

    return (
        "⚠️  Yes" if is_vulnerable else "✅  No",
        vuln_type,
        _fuzzy_severity(fuzzy_score),
        f"{confidence * 100:.1f}%",
        affected,
        llm_reason_str
    )


# ─── Gradio UI ────────────────────────────────────────────────────────────────

_DESCRIPTION = """
# 🛡️ Premium AI Security Auditor
Paste any C / C++ / Java code snippet and click **Analyze** to detect
security vulnerabilities. Powered by GraphCodeBERT & GPT-4 Security Agent.
"""

_EXAMPLE_VULN = """\
void login(char *username, char *password) {
    char query[256];
    sprintf(query, "SELECT * FROM users WHERE user='%s' AND pass='%s'",
            username, password);
    db_execute(query);
}"""

_EXAMPLE_SAFE = """\
void connect_db() {
    char *password = getenv("DB_PASSWORD");
    if (!password) { fprintf(stderr, "Missing DB_PASSWORD\\n"); exit(1); }
    db_connect("localhost", "root", password);
}"""

with gr.Blocks(
    title="VulnDetect – Premium Security Auditor",
    theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="blue"),
) as demo:

    gr.Markdown(_DESCRIPTION)

    with gr.Row():
        with gr.Column(scale=2):
            code_input = gr.Code(
                label="📝 Paste code here",
                language="c",
                lines=20,
                elem_id="code_input",
            )
            language_sel = gr.Dropdown(
                choices=["C", "C++", "Java", "Python"],
                value="C",
                label="🌐 Language",
                elem_id="language_selector",
            )
            with gr.Row():
                analyze_btn = gr.Button("🔍 Analyze", variant="primary",  elem_id="analyze_btn")
                clear_btn   = gr.Button("🗑️  Clear",  variant="secondary", elem_id="clear_btn")

        with gr.Column(scale=1):
            gr.Markdown("### 📊 Analysis Results")
            out_vulnerable = gr.Textbox(label="Vulnerable",        interactive=False, elem_id="out_vulnerable")
            out_type       = gr.Textbox(label="Vulnerability Type", interactive=False, elem_id="out_type")
            out_severity   = gr.Textbox(label="Severity",           interactive=False, elem_id="out_severity")
            out_confidence = gr.Textbox(label="Confidence",         interactive=False, elem_id="out_confidence")
            out_lines      = gr.Textbox(label="Affected Lines",     interactive=False, elem_id="out_lines")
            
    with gr.Row():
        out_reasoning = gr.Markdown("### 🤖 Premium AI Security Audit\nWaiting for analysis...", elem_id="out_reasoning")

    with gr.Row():
        gr.Examples(
            examples=[[_EXAMPLE_VULN, "C"], [_EXAMPLE_SAFE, "C"]],
            inputs=[code_input, language_sel],
            label="Example snippets",
        )

    analyze_btn.click(
        fn=analyze_code,
        inputs=[code_input, language_sel],
        outputs=[out_vulnerable, out_type, out_severity, out_confidence, out_lines, out_reasoning],
    )

    clear_btn.click(
        fn=lambda: ("", "C", "—", "—", "—", "—", "—", "### 🤖 Premium AI Security Audit\nWaiting for analysis..."),
        inputs=[],
        outputs=[
            code_input, language_sel,
            out_vulnerable, out_type, out_severity, out_confidence, out_lines, out_reasoning
        ],
    )

# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("Launching VulnDetect Gradio app …")
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
    )
