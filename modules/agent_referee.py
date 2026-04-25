"""
agent_referee.py – Phase 11: LLM Agent Referee
Takes highly-confident but potentially false-positive snippets flagged by GraphCodeBERT,
and forces an LLM to step through them line-by-line via Chain of Thought to confirm
if a true vulnerability exists.
"""

import os
import logging
import json
import warnings

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

logger = logging.getLogger(__name__)

# Load API key from environment variable — NEVER hardcode secrets in source code
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

_REFEREE_PROMPT = """You are a World-Class Application Security Engineer.

Task:
Analyze the provided code snippet line-by-line. Determine if there are vulnerabilities (like Memory Leaks, Buffer Overflows, SQL Injections, etc.) or if it is completely secure.

Code Snippet:
```c
{code}
```

ML Baseline Confidence: {confidence}%
Detected CWE Strategy: {cwe_guess}

Provide your complete analysis in JSON format *exactly* like this:
{
  "is_vulnerable": true/false,
  "final_confidence": 99.5,
  "markdown_report": "### 🛡️ AI Security Audit\\n\\n**Status:** 🔴 Vulnerable (or 🟢 Secure)\\n**Details:** Provide a detailed forensic explanation of the vulnerability or why it is safe.\\n\\n### 🛠️ Secure Code Solution\\n```c\\n// Fix goes here...\\n```"
}
"""

def request_referee_review(code: str, confidence: float, cwe_guess: str) -> tuple[bool, str, float]:
    """
    Sends the flagged snippet to the LLM agent for review.
    Returns: (is_vulnerable: bool, reasoning: str, final_confidence: float)
    
    If the API is not set up, degrades gracefully.
    """
    if not OpenAI or not OPENAI_API_KEY:
        logger.warning("LLM Agent Referee invoked but OPENAI_API_KEY is not configured or openai package missing.")
        return True, "API Key missing. Deferred to GraphCodeBERT's initial judgement.", confidence

    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        prompt = _REFEREE_PROMPT.format(
            code=code,
            confidence=round(confidence * 100, 1),
            cwe_guess=cwe_guess
        )
        
        response = client.chat.completions.create(
            model="gpt-4o",  # or gpt-4o-mini
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        
        result_text = response.choices[0].message.content
        data = json.loads(result_text)
        
        final_vuln = bool(data.get("is_vulnerable", True))
        reasoning = str(data.get("markdown_report", "LLM provided no report."))
        final_conf = float(data.get("final_confidence", confidence * 100)) / 100.0
        
        return final_vuln, reasoning, final_conf
        
    except Exception as e:
        logger.error(f"Referee analysis failed: {e}")
        return True, f"LLM Integration failed: {e}. Deferred to GraphCodeBERT.", confidence
