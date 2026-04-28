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

from dotenv import load_dotenv
load_dotenv()

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
from modules.pdf_generator import generate_pdf_report
from modules.syntax_checker import check_syntax

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

def guess_language(code: str, current_lang: str) -> str:
    """Simple heuristic to guess the programming language of the snippet."""
    if not code.strip():
        return current_lang
    
    java_patterns = [
        r"public\s+(?:final\s+)?class\b", 
        r"System\.out\.print", 
        r"import\s+java\.", 
        r"public\s+static\s+void\s+main",
        r"String\s+\[\]\s+args"
    ]
    cpp_patterns = [
        r"#include\s*<iostream>", 
        r"std::", 
        r"cout\s*<<", 
        r"cin\s*>>", 
        r"using\s+namespace\s+std;"
    ]
    c_patterns = [
        r"#include\s*<stdio\.h>", 
        r"\bprintf\s*\(", 
        r"\bmalloc\s*\(", 
        r"\bfree\s*\("
    ]
    
    # Check Java
    for p in java_patterns:
        if re.search(p, code):
            return "Java"
            
    # Check C++
    for p in cpp_patterns:
        if re.search(p, code):
            return "C++"
            
    # Check C
    for p in c_patterns:
        if re.search(p, code):
            return "C"
            
    return current_lang # keep current if no strong signal

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

def analyze_code(code: str, language: str) -> tuple[str, str, str, str, str, str, str]:
    """
    Run the full detection pipeline on user-supplied code.

    Returns
    -------
    (vulnerable, vuln_type, severity, risk_score, confidence, affected_lines, llm_reasoning)  - all str.
    """
    if not code.strip():
        return "—", "—", "—", "—", "—", "—", "—"

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
                if config.DEVICE == "cuda":
                    with torch.amp.autocast('cuda'):
                        outputs = _finetuned_model(b_input_ids, attention_mask=b_attn_mask)
                        probs = torch.softmax(outputs.logits, dim=1)[0].cpu().numpy()
                else:
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
    elif clf_vulnerable is not None and clf_conf > 0.70:
        is_vulnerable = clf_vulnerable
        confidence    = clf_conf
    else:
        # Pure fuzzy fallback
        is_vulnerable = fuzzy_score >= 0.65
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
        f"{fuzzy_score * 100:.1f}/100",
        f"{confidence * 100:.1f}%",
        affected,
        llm_reason_str
    )


# ─── Gradio UI ────────────────────────────────────────────────────────────────

_CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* ══════════════════════════════════════════════════════
   BASE RESET & BACKGROUND
══════════════════════════════════════════════════════ */
*, *::before, *::after { box-sizing: border-box; margin: 0; }

html, body { height: 100%; }

body, .gradio-container, .gradio-container > .main {
    background: #080c14 !important;
    font-family: 'Inter', system-ui, sans-serif !important;
    color: #c9d1d9 !important;
}

.gradio-container {
    max-width: 1400px !important;
    margin: 0 auto !important;
    padding: 20px 24px 40px !important;
}

/* ══════════════════════════════════════════════════════
   TOP NAV BAR
══════════════════════════════════════════════════════ */
#topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 14px 28px;
    background: rgba(13,17,28,0.95);
    border: 1px solid rgba(88,101,242,0.25);
    border-radius: 14px;
    margin-bottom: 20px;
    backdrop-filter: blur(20px);
    box-shadow: 0 1px 0 rgba(88,101,242,0.1), 0 8px 32px rgba(0,0,0,0.5);
}

/* ══════════════════════════════════════════════════════
   HERO HEADER
══════════════════════════════════════════════════════ */
#hero-section {
    position: relative;
    background: linear-gradient(135deg, #0d1321 0%, #111827 40%, #0c1220 100%);
    border: 1px solid rgba(88,101,242,0.3);
    border-radius: 18px;
    padding: 36px 44px 32px;
    margin-bottom: 24px;
    overflow: hidden;
}
#hero-section::before {
    content: '';
    position: absolute;
    top: -80px; right: -80px;
    width: 360px; height: 360px;
    background: radial-gradient(circle, rgba(88,101,242,0.18) 0%, transparent 65%);
    pointer-events: none;
}
#hero-section::after {
    content: '';
    position: absolute;
    bottom: -60px; left: 10%;
    width: 280px; height: 280px;
    background: radial-gradient(circle, rgba(139,92,246,0.1) 0%, transparent 65%);
    pointer-events: none;
}
#hero-section h1 {
    font-size: 2rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.03em !important;
    color: #f0f6fc !important;
    margin-bottom: 10px !important;
}
#hero-section p {
    color: #8b949e !important;
    font-size: 0.95rem !important;
    line-height: 1.65 !important;
    max-width: 680px;
}
#hero-section strong { color: #a5b4fc !important; }
.badge-row {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    margin-top: 18px;
}
.badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.04em;
    border: 1px solid;
}
.badge-blue  { background: rgba(88,101,242,0.12); border-color: rgba(88,101,242,0.35); color: #818cf8; }
.badge-green { background: rgba(16,185,129,0.1);  border-color: rgba(16,185,129,0.3);  color: #34d399; }
.badge-amber { background: rgba(245,158,11,0.1);  border-color: rgba(245,158,11,0.3);  color: #fbbf24; }

/* ══════════════════════════════════════════════════════
   MAIN PANEL CARDS
══════════════════════════════════════════════════════ */
.card {
    background: rgba(13,17,28,0.95) !important;
    border: 1px solid rgba(88,101,242,0.18) !important;
    border-radius: 16px !important;
    box-shadow: 0 4px 24px rgba(0,0,0,0.45) !important;
    overflow: hidden;
}

/* ══════════════════════════════════════════════════════
   CODE EDITOR
══════════════════════════════════════════════════════ */
#code_input {
    background: transparent !important;
    border: none !important;
}
#code_input > .label-wrap {
    padding: 14px 18px 8px !important;
    border-bottom: 1px solid rgba(88,101,242,0.15) !important;
    margin-bottom: 0 !important;
}
#code_input .codemirror-wrapper,
#code_input .cm-editor,
#code_input textarea {
    background: #070b13 !important;
    font-family: 'JetBrains Mono', 'Fira Code', monospace !important;
    font-size: 13px !important;
    line-height: 1.7 !important;
    color: #e2e8f0 !important;
    border: none !important;
    border-radius: 0 !important;
    min-height: 340px !important;
}
#code_input .cm-gutters {
    background: #080c14 !important;
    border-right: 1px solid rgba(88,101,242,0.15) !important;
    color: #374151 !important;
}
#code_input .cm-activeLineGutter { background: rgba(88,101,242,0.08) !important; }
#code_input .cm-activeLine { background: rgba(88,101,242,0.06) !important; }

/* ══════════════════════════════════════════════════════
   LANGUAGE DROPDOWN
══════════════════════════════════════════════════════ */
#language_selector {
    background: transparent !important;
    border: none !important;
}
#language_selector > .label-wrap { display: none !important; }
#language_selector select,
#language_selector .wrap-inner,
#language_selector input {
    background: rgba(15,22,38,0.9) !important;
    border: 1px solid rgba(88,101,242,0.25) !important;
    border-radius: 8px !important;
    color: #a5b4fc !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 8px 12px !important;
    height: 38px !important;
}
#language_selector .wrap { background: transparent !important; border: none !important; }

/* ══════════════════════════════════════════════════════
   BUTTONS
══════════════════════════════════════════════════════ */
#analyze_btn {
    background: linear-gradient(135deg, #5865f2 0%, #7c3aed 100%) !important;
    border: 1px solid rgba(88,101,242,0.4) !important;
    border-radius: 10px !important;
    color: #ffffff !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 700 !important;
    font-size: 14px !important;
    letter-spacing: 0.03em !important;
    padding: 11px 24px !important;
    box-shadow: 0 0 0 0 rgba(88,101,242,0.5), 0 4px 16px rgba(88,101,242,0.35) !important;
    transition: all 0.22s cubic-bezier(0.4,0,0.2,1) !important;
    text-transform: uppercase !important;
}
#analyze_btn:hover {
    box-shadow: 0 0 0 3px rgba(88,101,242,0.25), 0 8px 28px rgba(88,101,242,0.5) !important;
    transform: translateY(-1px) !important;
    background: linear-gradient(135deg, #6875f5 0%, #8b5cf6 100%) !important;
}
#analyze_btn:active { transform: translateY(0) !important; }

#clear_btn {
    background: rgba(15,22,38,0.8) !important;
    border: 1px solid rgba(88,101,242,0.2) !important;
    border-radius: 10px !important;
    color: #6b7280 !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    transition: all 0.2s ease !important;
}
#clear_btn:hover {
    border-color: rgba(88,101,242,0.4) !important;
    color: #9ca3af !important;
    background: rgba(25,32,54,0.9) !important;
}

/* ══════════════════════════════════════════════════════
   RESULTS PANEL
══════════════════════════════════════════════════════ */
.results-header {
    padding: 14px 20px;
    border-bottom: 1px solid rgba(88,101,242,0.15);
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #6875f5;
}

#out_vulnerable, #out_type, #out_severity, #out_risk_score, #out_confidence, #out_lines, #out_syntax {
    background: transparent !important;
    border: none !important;
    border-bottom: 1px solid rgba(88,101,242,0.1) !important;
    border-radius: 0 !important;
    padding: 0 !important;
}
#out_vulnerable textarea, #out_type textarea,
#out_severity textarea, #out_risk_score textarea, #out_confidence textarea, #out_lines textarea, #out_syntax textarea {
    background: rgba(9,13,22,0.6) !important;
    border: 1px solid rgba(88,101,242,0.15) !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    padding: 10px 14px !important;
    transition: border-color 0.2s ease !important;
    min-height: 44px !important;
    resize: none !important;
}
#out_vulnerable textarea:focus,
#out_type textarea:focus { border-color: rgba(88,101,242,0.4) !important; outline: none !important; }

/* ══════════════════════════════════════════════════════
   ALL LABELS (universal)
══════════════════════════════════════════════════════ */
.gradio-container label,
.gradio-container .label-wrap span,
.block > .label-wrap > span {
    color: #4b5563 !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    font-family: 'Inter', sans-serif !important;
}

/* ══════════════════════════════════════════════════════
   AI REPORT PANEL
══════════════════════════════════════════════════════ */
#out_reasoning {
    background: rgba(13,17,28,0.95) !important;
    border: 1px solid rgba(88,101,242,0.18) !important;
    border-radius: 16px !important;
    padding: 28px 36px !important;
    color: #9ca3af !important;
    font-size: 14px !important;
    line-height: 1.8 !important;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4) !important;
    width: 100% !important;
}
#out_reasoning h3 {
    color: #f0f6fc !important;
    font-size: 15px !important;
    font-weight: 700 !important;
    letter-spacing: 0.04em !important;
    margin-bottom: 16px !important;
    padding-bottom: 12px !important;
    border-bottom: 1px solid rgba(88,101,242,0.15) !important;
}
#out_reasoning p, #out_reasoning li { color: #8b949e !important; line-height: 1.8 !important; }
#out_reasoning strong { color: #c9d1d9 !important; }
#out_reasoning code {
    background: rgba(88,101,242,0.12) !important;
    color: #a5b4fc !important;
    padding: 2px 6px !important;
    border-radius: 4px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 12px !important;
}
#out_reasoning blockquote {
    border-left: 3px solid rgba(88,101,242,0.4) !important;
    padding-left: 16px !important;
    color: #6b7280 !important;
    font-style: italic !important;
    margin: 12px 0 !important;
}

/* ══════════════════════════════════════════════════════
   EXAMPLES TABLE — DARK OVERRIDE (fixes white table)
══════════════════════════════════════════════════════ */
.examples-holder,
.examples table,
.gr-samples-table,
table.gr-samples-table {
    background: rgba(13,17,28,0.95) !important;
    border: 1px solid rgba(88,101,242,0.15) !important;
    border-radius: 12px !important;
    overflow: hidden !important;
    color: #8b949e !important;
}
.examples-holder thead th,
.examples table thead th,
.gr-samples-table thead th {
    background: rgba(9,13,22,0.8) !important;
    color: #4b5563 !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    border-bottom: 1px solid rgba(88,101,242,0.15) !important;
    padding: 10px 16px !important;
}
.examples-holder tbody tr,
.examples table tbody tr,
.gr-samples-table tbody tr {
    background: transparent !important;
    border-bottom: 1px solid rgba(88,101,242,0.08) !important;
    transition: background 0.15s ease !important;
    cursor: pointer !important;
}
.examples-holder tbody tr:hover,
.examples table tbody tr:hover,
.gr-samples-table tbody tr:hover {
    background: rgba(88,101,242,0.07) !important;
}
.examples-holder tbody td,
.examples table tbody td,
.gr-samples-table tbody td {
    color: #6b7280 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 12px !important;
    padding: 10px 16px !important;
    border: none !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    max-width: 420px !important;
}
/* The examples accordion header */
.examples > .label-wrap,
details > summary {
    background: rgba(13,17,28,0.9) !important;
    border: 1px solid rgba(88,101,242,0.15) !important;
    border-radius: 10px !important;
    color: #4b5563 !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    padding: 10px 16px !important;
    margin-bottom: 6px !important;
    cursor: pointer !important;
}

/* ══════════════════════════════════════════════════════
   TOOLBAR ROW (language + buttons strip)
══════════════════════════════════════════════════════ */
#toolbar-row {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 12px 16px;
    background: rgba(9,13,22,0.6);
    border-top: 1px solid rgba(88,101,242,0.12);
}

/* ══════════════════════════════════════════════════════
   DIVIDER
══════════════════════════════════════════════════════ */
.section-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(88,101,242,0.3), transparent);
    margin: 20px 0;
    border: none;
}

/* ══════════════════════════════════════════════════════
   SCROLLBAR
══════════════════════════════════════════════════════ */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #080c14; }
::-webkit-scrollbar-thumb { background: rgba(88,101,242,0.3); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(88,101,242,0.55); }

/* ══════════════════════════════════════════════════════
   BLOCK OVERRIDES — kill Gradio's default white cards
══════════════════════════════════════════════════════ */
.gradio-container .block,
.gradio-container .form,
.gradio-container fieldset {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

/* Row gaps */
.gradio-container .gap { gap: 16px !important; }

/* Footer */
footer { display: none !important; }
"""

_HEADER_MD = """
<div id="hero-section">

# 🛡️ VulnDetect — AI Security Auditor

**Hybrid Fuzzy-Transformer static analysis engine** powered by **GraphCodeBERT** + **Groq LLaMA-3.3 Agent Referee**.  
Paste any C, C++, or Java snippet and receive a precise, line-level security audit in seconds.

<div class="badge-row">
  <span class="badge badge-blue">⚡ GraphCodeBERT</span>
  <span class="badge badge-blue">🤖 LLaMA-3.3 Referee</span>
  <span class="badge badge-green">✓ C &nbsp;/&nbsp; C++ &nbsp;/&nbsp; Java</span>
  <span class="badge badge-amber">🔬 Fuzzy Risk Scoring</span>
</div>

</div>
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

_EXAMPLE_JAVA_VULN = """\
public void getUserData(String userId) throws Exception {
    String query = "SELECT * FROM users WHERE id = " + userId;
    Statement stmt = conn.createStatement();
    stmt.execute(query);
}"""

_EXAMPLE_CPP_VULN = """\
void copyInput(char *src) {
    char dest[64];
    strcpy(dest, src);  // no bounds check!
    printf("Copied: %s\\n", dest);
}"""

_THEME = gr.themes.Base(
    primary_hue="indigo",
    secondary_hue="violet",
    neutral_hue="gray",
    font=gr.themes.GoogleFont("Inter"),
    font_mono=gr.themes.GoogleFont("JetBrains Mono"),
).set(
    body_background_fill="#080c14",
    body_text_color="#c9d1d9",
    block_background_fill="transparent",
    block_border_width="0px",
    block_label_text_color="#4b5563",
    block_label_text_size="11px",
    input_background_fill="rgba(9,13,22,0.6)",
    input_border_color="rgba(88,101,242,0.2)",
    input_border_width="1px",
    button_primary_background_fill="linear-gradient(135deg,#5865f2,#7c3aed)",
    button_primary_text_color="#ffffff",
    button_secondary_background_fill="rgba(15,22,38,0.8)",
    button_secondary_text_color="#6b7280",
    button_secondary_border_color="rgba(88,101,242,0.2)",
)

with gr.Blocks(title="VulnDetect – AI Security Auditor") as demo:

    # ── Hero header ──────────────────────────────────────────────────
    gr.Markdown(_HEADER_MD)

    # ── Main two-column workspace ────────────────────────────────────
    with gr.Row(equal_height=False):

        # Left column — code editor
        with gr.Column(scale=5, elem_classes=["card"]):
            code_input = gr.Code(
                label="SOURCE CODE",
                language="c",
                lines=24,
                elem_id="code_input",
            )
            with gr.Row():
                language_sel = gr.Dropdown(
                    choices=["C", "C++", "Java"],
                    value="C",
                    label="Language",
                    elem_id="language_selector",
                    scale=1,
                    show_label=False,
                    info="Select language",
                )
                analyze_btn = gr.Button(
                    "🔍  ANALYZE",
                    variant="primary",
                    elem_id="analyze_btn",
                    scale=3,
                )
                clear_btn = gr.Button(
                    "✕  Clear",
                    variant="secondary",
                    elem_id="clear_btn",
                    scale=1,
                )

        # Right column — scan results
        with gr.Column(scale=3, elem_classes=["card"]):
            with gr.Row(elem_id="results_header_row", variant="compact"):
                gr.Markdown(
                    "<div class='results-header' style='border-bottom:none; padding-bottom:0;'>📊 &nbsp; SCAN RESULTS</div>"
                )
                export_btn = gr.DownloadButton("📥 EXPORT PDF", visible=False, size="sm", elem_id="export_btn")
                
            with gr.Row():
                out_vulnerable = gr.Textbox(
                    label="VERDICT",
                    interactive=False,
                    elem_id="out_vulnerable",
                    placeholder="—",
                )
                out_syntax = gr.Textbox(
                    label="SYNTAX STATUS",
                    interactive=False,
                    elem_id="out_syntax",
                    placeholder="—",
                )
            with gr.Row():
                out_severity = gr.Textbox(
                    label="SEVERITY",
                    interactive=False,
                    elem_id="out_severity",
                    placeholder="—",
                    scale=1,
                )
                out_risk_score = gr.Textbox(
                    label="RISK SCORE",
                    interactive=False,
                    elem_id="out_risk_score",
                    placeholder="—",
                    scale=1,
                )
                out_confidence = gr.Textbox(
                    label="CONFIDENCE",
                    interactive=False,
                    elem_id="out_confidence",
                    placeholder="—",
                    scale=1,
                )
            out_type = gr.Textbox(
                label="VULNERABILITY TYPE",
                interactive=False,
                elem_id="out_type",
                placeholder="—",
            )
            out_lines = gr.Textbox(
                label="AFFECTED LINES",
                interactive=False,
                elem_id="out_lines",
                placeholder="—",
            )

    # ── AI Audit report (full width) ────────────────────────────────
    with gr.Row():
        out_reasoning = gr.Markdown(
            """### 🤖 AI Security Audit Report

> Paste your code snippet above, select the language, and click **ANALYZE** to generate a detailed audit report.""",
            elem_id="out_reasoning",
        )

    # ── Example snippets ─────────────────────────────────────────────
    with gr.Row():
        gr.Examples(
            examples=[
                [_EXAMPLE_VULN,      "C"],
                [_EXAMPLE_SAFE,      "C"],
                [_EXAMPLE_CPP_VULN,  "C++"],
                [_EXAMPLE_JAVA_VULN, "Java"],
            ],
            inputs=[code_input, language_sel],
            label="EXAMPLE SNIPPETS — click any row to load",
        )

    # ── Events ──────────────────────────────────────────────────────
    code_input.change(
        fn=guess_language,
        inputs=[code_input, language_sel],
        outputs=[language_sel]
    )

    def _handle_analyze(code, lang):
        is_ok, syn_msg = check_syntax(code)
        results = analyze_code(code, lang)
        
        if code.strip():
            pdf_path = generate_pdf_report(code, lang, syn_msg, *results)
            return (results[0], syn_msg) + results[1:] + (gr.update(value=pdf_path, visible=True),)
        else:
            return (results[0], syn_msg) + results[1:] + (gr.update(visible=False),)

    analyze_btn.click(
        fn=_handle_analyze,
        inputs=[code_input, language_sel],
        outputs=[out_vulnerable, out_syntax, out_type, out_severity, out_risk_score, out_confidence, out_lines, out_reasoning, export_btn],
    )

    clear_btn.click(
        fn=lambda: (
            "", "C",
            "", "", "", "", "", "", "",
            """### 🤖 AI Security Audit Report\n\n> Paste your code snippet above, select the language, and click **ANALYZE** to generate a detailed audit report.""",
            gr.update(visible=False, value=None)
        ),
        inputs=[],
        outputs=[
            code_input, language_sel,
            out_vulnerable, out_syntax, out_type, out_severity, out_risk_score, out_confidence, out_lines,
            out_reasoning, export_btn
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
        theme=_THEME,
        css=_CUSTOM_CSS,
    )
