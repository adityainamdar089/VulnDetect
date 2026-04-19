"""
agent_referee.py – Phase 11: LLM Agent Referee
Takes highly-confident but potentially false-positive snippets flagged by GraphCodeBERT,
and forces an LLM to step through them line-by-line via Chain of Thought to confirm
if a true vulnerability exists.
"""

import os
import logging
import json

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

logger = logging.getLogger(__name__)

# Try to load API key from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

_REFEREE_PROMPT = """You are an Expert Application Security Engineer.
Our initial ML tool (GraphCodeBERT) has flagged the following C/C++ function as potentially containing a vulnerability.

Task:
Analyze this snippet line-by-line. Determine if the ML model is hallucinating (False Positive) or if there is a true vulnerability (True Positive).
Look out specifically for:
- Memory Leaks (malloc without free)
- Buffer Overflows (strcpy over boundaries limits)
- Resource Leaks

Code Snippet:
```c
{code}
```

GraphCodeBERT Confidence: {confidence}%
Detected CWE Strategy: {cwe_guess}

Provide your analysis in JSON format *exactly* like this:
{
  "reasoning": "First, I see the pointer initialized on line X...",
  "is_vulnerable": true/false,
  "final_confidence": 99.5
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
        reasoning = str(data.get("reasoning", "LLM provided no reasoning."))
        final_conf = float(data.get("final_confidence", confidence * 100)) / 100.0
        
        return final_vuln, reasoning, final_conf
        
    except Exception as e:
        logger.error(f"Referee analysis failed: {e}")
        return True, f"LLM Integration failed: {e}. Deferred to GraphCodeBERT.", confidence
