"""
agent_referee.py – LLM Agent Referee powered by Groq (free tier).

Uses llama-3.3-70b-versatile via Groq's OpenAI-compatible API to perform
a Chain-of-Thought security analysis on flagged code snippets, reducing
false positives from the upstream ML pipeline.

Public API
----------
request_referee_review(code, confidence, cwe_guess) -> tuple[bool, str, float]
"""

import os
import json
import logging
import requests

logger = logging.getLogger(__name__)

# ─── Groq API config ──────────────────────────────────────────────────────────
_GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
_GROQ_MODEL   = "llama-3.3-70b-versatile"

# ─── Prompts ──────────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = (
    "You are a senior security engineer specializing in code vulnerability analysis. "
    "Analyze the provided code snippet carefully and determine if it contains a real, "
    "exploitable security vulnerability. Ignore pedantic best-practice issues, such as "
    "unclosed streams or missing null checks, unless they lead to a direct security compromise. "
    "Focus primarily on vulnerabilities like SQL/OS Injection, Buffer Overflows, Memory Corruption, "
    "Use-After-Free, Path Traversal, and Hardcoded Credentials. Be precise and explicitly avoid false positives."
)

_USER_PROMPT_TEMPLATE = """\
Code snippet to analyze:
```
{code}
```

The static analysis pipeline suspects the code may contain MULTIPLE vulnerability types: {cwe_guess}
ML model confidence: {confidence_pct:.1f}%

For each suspected vulnerability type listed above:
1. Determine if it is actually present and exploitable in this specific code.
2. Explain your reasoning concisely.

Finally, provide a consolidated verdict:
- List only the CONFIRMED vulnerabilities (if any).
- Conclude with either VULNERABLE or SAFE on the very last line.
"""


# ─── Public API ───────────────────────────────────────────────────────────────

def request_referee_review(
    code: str,
    confidence: float,
    cwe_guess: str,
) -> tuple[bool, str, float]:
    """
    Send a flagged code snippet to the Groq LLM for Chain-of-Thought review.

    Parameters
    ----------
    code       : Source code string to analyse.
    confidence : ML model confidence score in [0, 1].
    cwe_guess  : Suspected CWE type string (e.g. "CWE89").

    Returns
    -------
    Tuple of (is_vulnerable: bool, reasoning: str, final_confidence: float).
    Degrades gracefully when GROQ_API_KEY is absent or the API call fails.
    """
    api_key = os.environ.get("GROQ_API_KEY", "").strip()

    if not api_key:
        logger.warning("GROQ_API_KEY not set — Agent Referee skipped.")
        return True, "Agent Referee skipped — GROQ_API_KEY not set.", confidence

    try:
        user_prompt = _USER_PROMPT_TEMPLATE.format(
            code=code,
            cwe_guess=cwe_guess,
            confidence_pct=confidence * 100,
        )

        payload = {
            "model": _GROQ_MODEL,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            "max_tokens": 1024,
            "temperature": 0.1,
        }

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type":  "application/json",
        }

        logger.info("Invoking Groq Agent Referee (model=%s) …", _GROQ_MODEL)
        resp = requests.post(
            _GROQ_API_URL,
            headers=headers,
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()

        data     = resp.json()
        reasoning = data["choices"][0]["message"]["content"].strip()

        # Parse verdict from the last non-empty line
        last_line = ""
        for line in reversed(reasoning.splitlines()):
            stripped = line.strip().upper()
            if stripped:
                last_line = stripped
                break

        if "VULNERABLE" in last_line:
            logger.info("Referee verdict: VULNERABLE")
            return True, reasoning, min(confidence + 0.1, 1.0)
        elif "SAFE" in last_line:
            logger.info("Referee verdict: SAFE")
            return False, reasoning, 0.2
        else:
            # Fallback: check the entire reasoning text
            upper_text = reasoning.upper()
            if "SAFE" in upper_text and "VULNERABLE" not in upper_text:
                logger.info("Referee verdict (fallback): SAFE")
                return False, reasoning, 0.2
            elif "VULNERABLE" in upper_text:
                logger.info("Referee verdict (fallback): VULNERABLE")
                return True, reasoning, min(confidence + 0.1, 1.0)
                
            logger.warning("Referee response did not contain VULNERABLE/SAFE — keeping original prediction.")
            return True, reasoning, confidence

    except Exception as e:
        logger.error("Referee analysis failed: %s", e)
        return True, f"Referee error: {str(e)}", confidence
