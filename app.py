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
    # Path Traversal
    "CWE22":  [r"File\s*\([^)]*\+\s*[a-zA-Z0-9_]+\)", r"\.\./", r"open\s*\([^)]*\+\s*[a-zA-Z0-9_]+\)"],
    # OS Command Injection
    "CWE78":  [r"Runtime\.getRuntime\(\)\.exec\s*\(", r"\bsystem\s*\(", r"\bpopen\s*\("],
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


def _detect_all_cwes(code: str) -> list:
    """Return ALL CWE IDs whose heuristic patterns match in the given code."""
    found = []
    for cwe, patterns in _CWE_PATTERNS.items():
        hits = sum(1 for p in patterns if re.search(p, code, re.IGNORECASE | re.DOTALL))
        if cwe == "CWE401" and hits > 0:
            # Only flag as a leak when there is NO matching free/delete
            if re.search(r"\bfree\s*\(", code, re.IGNORECASE) or \
               re.search(r"\bdelete\b", code, re.IGNORECASE):
                hits = 0
        if hits > 0:
            found.append(cwe)
    return found


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
    found_cwes = _detect_all_cwes(code)
    regex_hit  = len(found_cwes) > 0
    cwe        = found_cwes[0] if found_cwes else "Unknown"   # primary CWE for back-compat

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
    if regex_hit:
        is_vulnerable = True
        if clf_vulnerable is not None and clf_conf > 0.60:
            confidence = clf_conf
        else:
            confidence = max(fuzzy_score, 0.65)
    elif clf_vulnerable is not None and clf_conf > 0.70:
        is_vulnerable = clf_vulnerable
        confidence    = clf_conf
    else:
        is_vulnerable = fuzzy_score >= 0.65
        confidence    = fuzzy_score

    # 6. Premium SWE Agent Referee
    cwe_guess_str = ", ".join(found_cwes) if found_cwes else "Unknown"
    logger.info("Invoking Premium Agent Referee...")
    ref_vuln, reason, ref_conf = request_referee_review(code, confidence, cwe_guess_str)
    is_vulnerable = ref_vuln
    confidence = max(confidence, ref_conf)
    llm_reason_str = reason

    # 7. CWE label — build multi-CWE display string
    if not is_vulnerable:
        vuln_type = "N/A"
        affected  = "N/A"
    else:
        if found_cwes:
            parts = []
            for c in found_cwes:
                name = config.CWE_DESCRIPTIONS.get(c, "")
                parts.append(f"{c} – {name}" if name else c)
            vuln_type = "\n".join(parts)
        else:
            vuln_type = "Unknown"

        # 8. Affected lines — union across all matched CWEs
        all_lines: set = set()
        for c in found_cwes:
            raw = _detect_affected_lines(code, c)
            if raw != "N/A":
                all_lines.update(raw.split(", "))
        affected = ", ".join(sorted(all_lines, key=lambda x: int(x))) if all_lines else "N/A"

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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; }
html, body { height: 100%; }

body, .gradio-container, .gradio-container > .main {
    background: #000000 !important;
    font-family: 'Inter', system-ui, sans-serif !important;
    color: #ededed !important;
}

.gradio-container { max-width: 1400px !important; padding: 40px 20px !important; }

/* HERO */
#hero-section {
    background: linear-gradient(180deg, #0a0a0a 0%, #000000 100%);
    border: 1px solid #262626;
    border-radius: 12px;
    padding: 32px 40px;
    margin-bottom: 32px;
    text-align: center;
    box-shadow: 0 4px 30px rgba(0, 0, 0, 0.5);
}
#hero-section h1 { 
    font-size: 2.2rem !important; 
    font-weight: 700 !important; 
    letter-spacing: -0.04em !important;
    background: linear-gradient(to right, #ffffff, #a3a3a3);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 12px !important; 
}
#hero-section p { color: #a3a3a3 !important; font-size: 1rem !important; margin-bottom: 24px !important; }

.badge-row { display: flex; justify-content: center; gap: 12px; flex-wrap: wrap; }
.badge {
    display: inline-flex; align-items: center; padding: 6px 14px;
    border-radius: 9999px; font-size: 12px; font-weight: 500; letter-spacing: 0.02em;
}
.badge-blue  { background: rgba(59, 130, 246, 0.1); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.2); }
.badge-green { background: rgba(16, 185, 129, 0.1); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.2); }
.badge-amber { background: rgba(245, 158, 11, 0.1); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.2); }

/* CARDS */
.card {
    background: #0a0a0a !important;
    border: 1px solid #262626 !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3) !important;
    transition: border-color 0.3s ease;
}
.card:hover { border-color: #404040 !important; }

/* CODE EDITOR */
#code_input > .label-wrap { padding: 16px 20px 8px !important; border-bottom: 1px solid #262626 !important; }
#code_input textarea, #code_input .cm-editor {
    background: #000000 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 14px !important;
    line-height: 1.6 !important;
    color: #ededed !important;
    border: none !important;
    border-radius: 0 0 12px 12px !important;
    min-height: 450px !important;
}
#code_input .cm-gutters { background: #000000 !important; border-right: 1px solid #262626 !important; color: #525252 !important; }

/* BUTTONS */
#analyze_btn {
    background: #ededed !important;
    border: none !important;
    border-radius: 8px !important;
    color: #000000 !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    padding: 12px 24px !important;
    transition: transform 0.2s ease, opacity 0.2s ease !important;
}
#analyze_btn:hover { opacity: 0.9 !important; transform: scale(1.02) !important; }

#clear_btn {
    background: transparent !important;
    border: 1px solid #262626 !important;
    border-radius: 8px !important;
    color: #a3a3a3 !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
}
#clear_btn:hover { background: #171717 !important; color: #ededed !important; }

/* TEXTBOXES */
.results-header {
    padding: 16px 24px; border-bottom: 1px solid #262626;
    font-size: 13px; font-weight: 600; color: #ededed; text-transform: uppercase; letter-spacing: 0.05em;
}
#out_vulnerable textarea, #out_type textarea, #out_severity textarea, #out_risk_score textarea, #out_confidence textarea, #out_lines textarea, #out_syntax textarea {
    background: #000000 !important;
    border: 1px solid #262626 !important;
    border-radius: 6px !important;
    color: #ededed !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 15px !important;
    font-weight: 500 !important;
    padding: 12px 16px !important;
}

/* AI REPORT */
#out_reasoning {
    background: #0a0a0a !important;
    border: 1px solid #262626 !important;
    border-radius: 12px !important;
    padding: 32px 40px !important;
    font-size: 15px !important;
    line-height: 1.7 !important;
    color: #a3a3a3 !important;
}
#out_reasoning h3 { color: #ededed !important; font-size: 18px !important; margin-bottom: 16px !important; padding-bottom: 12px !important; border-bottom: 1px solid #262626 !important; }
#out_reasoning strong { color: #ffffff !important; }
#out_reasoning code { background: #171717 !important; border: 1px solid #262626 !important; padding: 2px 6px !important; border-radius: 6px !important; color: #60a5fa !important; font-family: 'JetBrains Mono', monospace !important; font-size: 13px !important; }
#out_reasoning blockquote { border-left: 3px solid #525252 !important; padding-left: 20px !important; color: #737373 !important; font-style: italic !important; margin: 16px 0 !important; }

/* LABELS */
.gradio-container label, .gradio-container .label-wrap span {
    color: #737373 !important; font-size: 12px !important; font-weight: 500 !important; text-transform: uppercase !important; letter-spacing: 0.05em !important;
}

/* EXAMPLES TABLE */
.examples-holder, .examples table { background: #0a0a0a !important; border: 1px solid #262626 !important; border-radius: 12px !important; }
.examples-holder thead th { background: #000000 !important; color: #a3a3a3 !important; border-bottom: 1px solid #262626 !important; font-weight: 500 !important; }
.examples-holder tbody tr { border-bottom: 1px solid #262626 !important; transition: background 0.2s ease !important; }
.examples-holder tbody tr:hover { background: #171717 !important; }
.examples-holder tbody td { color: #a3a3a3 !important; font-family: 'JetBrains Mono', monospace !important; font-size: 13px !important; padding: 12px 16px !important; }
.examples > .label-wrap { background: #0a0a0a !important; border: 1px solid #262626 !important; border-radius: 8px !important; margin-bottom: 8px !important; padding: 12px 16px !important; }

/* BLOCK OVERRIDES */
.gradio-container .block, .gradio-container .form, .gradio-container fieldset { background: transparent !important; border: none !important; box-shadow: none !important; }
.gradio-container .gap { gap: 24px !important; }
footer { display: none !important; }
"""

_HEADER_MD = """
<div id="hero-section">
<h1>VulnDetect Security Audit</h1>
<p>Precision-engineered static analysis using GraphCodeBERT and Groq LLaMA-3.3 XAI.</p>
<div class="badge-row">
  <span class="badge badge-blue">GraphCodeBERT</span>
  <span class="badge badge-blue">LLaMA-3.3 Referee</span>
  <span class="badge badge-green">C / C++ / Java</span>
  <span class="badge badge-amber">Fuzzy Logic</span>
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
    primary_hue="zinc",
    secondary_hue="zinc",
    neutral_hue="zinc",
    font=gr.themes.GoogleFont("Inter"),
    font_mono=gr.themes.GoogleFont("JetBrains Mono"),
).set(
    body_background_fill="#000000",
    body_text_color="#ededed",
    block_background_fill="transparent",
    block_border_width="0px",
    block_label_text_color="#737373",
    block_label_text_size="12px",
    input_background_fill="#000000",
    input_border_color="#262626",
    input_border_width="1px",
    button_primary_background_fill="#ededed",
    button_primary_text_color="#000000",
    button_secondary_background_fill="#000000",
    button_secondary_text_color="#ededed",
    button_secondary_border_color="#262626",
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
                label="VULNERABILITIES FOUND",
                interactive=False,
                elem_id="out_type",
                placeholder="—",
                lines=3,
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
