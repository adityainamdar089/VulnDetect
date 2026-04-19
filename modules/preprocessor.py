"""
modules/preprocessor.py – Tree-sitter AST parsing and feature extraction.

Extracts 5 normalised numerical features (each 0–1) from C, C++, and Java
code snippets.  Falls back to regex-only analysis when tree-sitter is
unavailable.

Public API
----------
extract_features(code, language) -> dict
parse_juliet_dataset(raw_dir)    -> pd.DataFrame
"""

import re
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# ─── Tree-sitter import (optional) ────────────────────────────────────────────
try:
    from tree_sitter_languages import get_parser as _ts_get_parser
    _TREE_SITTER_OK = True
    logger.info("tree-sitter-languages is available.")
except ImportError:
    _TREE_SITTER_OK = False
    logger.warning(
        "tree-sitter-languages not found – regex-only feature extraction will be used."
    )

# ─── Helpers ──────────────────────────────────────────────────────────────────

_LANG_MAP: Dict[str, str] = {
    ".c":    "c",
    ".cpp":  "cpp",
    ".cc":   "cpp",
    ".cxx":  "cpp",
    ".java": "java",
}

_EXT_LANG_MAP: Dict[str, str] = {
    "c":    "c",
    "cpp":  "cpp",
    "c++":  "cpp",
    "java": "java",
}


def _get_ts_parser(language: str):
    """Return a tree-sitter Parser for *language*, or None on failure."""
    if not _TREE_SITTER_OK:
        return None
    lang_key = _EXT_LANG_MAP.get(language.lower())
    if lang_key is None:
        return None
    try:
        return _ts_get_parser(lang_key)
    except Exception as exc:
        logger.debug(f"Could not get parser for '{language}': {exc}")
        return None


# ─── Feature Extractors ───────────────────────────────────────────────────────

def _input_validation_score(code: str) -> float:
    """
    Measure how well inputs are validated before use.
    Looks for null-checks, bounds checks, and sanitisation calls.
    Returns 0–1 (higher = more validation = safer).
    """
    patterns = [
        r'\bif\s*\([^)]*(?:input|argv|param|buf|str|val|data|ptr)[^)]*(?:!=|==|<|>|<=|>=)',
        r'\bstrlen\s*\(',
        r'\bstrnlen\s*\(',
        r'\bfgets\s*\(',
        r'\bgetline\s*\(',
        r'\bsscanf\s*\(',
        r'\bvalidat\w*\s*\(',
        r'\bcheck\w*(?:Input|Param|Arg)\s*\(',
        r'\bsanitiz\w*\s*\(',
        r'\bstrncmp\s*\(',
        r'\bsnprintf\s*\(',
        r'\bregex\b',
        r'\bPattern\.\w+',
        r'\bMatcher\b',
        r'\bObjects\.requireNonNull\b',
    ]
    hits = sum(1 for p in patterns if re.search(p, code, re.IGNORECASE))
    return min(hits / len(patterns), 1.0)


def _sensitive_data_exposure(code: str) -> float:
    """
    Detect hardcoded passwords, tokens, and secrets.
    Returns 0–1 (higher = more exposure = worse).
    """
    patterns = [
        r'(?:password|passwd|pwd)\s*=\s*["\'](?!.*getenv)',
        r'(?:secret|api_key|apikey|token|auth_token)\s*=\s*["\']',
        r'(?:private_key|privatekey|secret_key)\s*=\s*["\']',
        r'\bBEGIN\s+(?:RSA|EC|OPENSSH|DSA)\s+PRIVATE\s+KEY\b',
        r'(?:access_key|aws_access|firebase_key)\s*=\s*["\']',
        r'(?:db_pass|database_password|db_password)\s*=\s*["\']',
        r'"[A-Za-z0-9+/]{40,}={0,2}"',   # long base64-like literals
        r"'[A-Za-z0-9+/]{40,}={0,2}'",
        r'\bhardcoded\b',
    ]
    hits = sum(1 for p in patterns if re.search(p, code, re.IGNORECASE))
    return min(hits / len(patterns), 1.0)


def _access_control_strength(code: str) -> float:
    """
    Check for auth/permission checks before sensitive operations.
    Returns 0–1 (higher = stronger access control = safer).
    """
    patterns = [
        r'\bauthentic(?:ate|ated|ation)?\s*\(',
        r'\bauthorize?\s*\(',
        r'\bcheckPermission\s*\(',
        r'\bisAuthorized\s*\(',
        r'\bhasRole\s*\(',
        r'\brole\s*==\s*["\']',
        r'\bpermission\b',
        r'\bAccessControl\b',
        r'\bsetuid\s*\(',
        r'\bgetCurrentUser\s*\(',
        r'\bsession\s*\[',
        r'\brequest\.user\b',
        r'\bSecurityContext\b',
        r'\b@PreAuthorize\b',
    ]
    hits = sum(1 for p in patterns if re.search(p, code, re.IGNORECASE))
    return min(hits / len(patterns), 1.0)


def _resource_management_score(code: str) -> float:
    """
    Assess whether allocated resources are released/closed.
    Returns 0–1 (higher = better resource mgmt = safer).
    """
    alloc_patterns   = [
        r'\bmalloc\s*\(', r'\bcalloc\s*\(', r'\brealloc\s*\(',
        r'\bnew\s+\w', r'\bfopen\s*\(', r'\bopen\s*\(', r'\bsocket\s*\(',
    ]
    dealloc_patterns = [
        r'\bfree\s*\(', r'\bdelete\s+', r'\bdelete\[\]',
        r'\bfclose\s*\(', r'\bclose\s*\(',
        r'\bfinally\s*\{', r'\btry\s*\(', r'AutoCloseable',
    ]
    allocs   = sum(1 for p in alloc_patterns   if re.search(p, code, re.IGNORECASE))
    deallocs = sum(1 for p in dealloc_patterns if re.search(p, code, re.IGNORECASE))
    if allocs == 0:
        return 1.0  # nothing allocated → no leak risk
    return min(deallocs / allocs, 1.0)


def _control_flow_complexity(code: str) -> float:
    """
    Normalised cyclomatic complexity (decision-point count + 1).
    Returns 0–1 (higher = more complex = riskier).
    Scores above cyclomatic-complexity 30 saturate at 1.0.
    """
    patterns = [
        r'\bif\b', r'\belse\s+if\b', r'\bfor\b', r'\bwhile\b',
        r'\bdo\b', r'\bswitch\b', r'\bcase\b', r'\bcatch\b',
        r'&&', r'\|\|', r'\?[^?]',
    ]
    count = sum(len(re.findall(p, code)) for p in patterns)
    complexity = count + 1
    return min(complexity / 30.0, 1.0)


# ─── Public API ───────────────────────────────────────────────────────────────

def extract_features(code: str, language: str) -> Dict[str, float]:
    """
    Extract 5 normalised numerical features from a code snippet.

    Parameters
    ----------
    code     : Source code string.
    language : One of "c", "cpp", "c++", "java".

    Returns
    -------
    dict with keys:
        input_validation_score, sensitive_data_exposure,
        access_control_strength, resource_management_score,
        control_flow_complexity
    """
    # Build tree-sitter parse tree if available (future: walk AST for accuracy)
    parser = _get_ts_parser(language)
    tree: Optional[Any] = None
    if parser is not None:
        try:
            tree = parser.parse(bytes(code, "utf-8"))
        except Exception as exc:
            logger.debug(f"AST parse error ({language}): {exc}")

    features: Dict[str, float] = {
        "input_validation_score":    _input_validation_score(code),
        "sensitive_data_exposure":   _sensitive_data_exposure(code),
        "access_control_strength":   _access_control_strength(code),
        "resource_management_score": _resource_management_score(code),
        "control_flow_complexity":   _control_flow_complexity(code),
    }
    logger.debug("Features: %s", features)
    return features


def parse_juliet_dataset(raw_dir: str) -> pd.DataFrame:
    """
    Parse the Juliet Test Suite from *raw_dir* and extract AST features.

    Expected directory layout::

        raw_dir/
        └── CWE89/
            ├── CWE89_SQL_Injection__char_connect_socket_01.c
            └── CWE89_SQL_Injection__char_connect_socket_01g.c  (good / benign)

    Juliet convention: filenames containing "good" are benign (label=0);
    all others are vulnerable (label=1).

    Parameters
    ----------
    raw_dir : Path to the raw Juliet data directory.

    Returns
    -------
    pd.DataFrame with columns:
        snippet_id, code, label, cwe_type, language,
        input_validation_score, sensitive_data_exposure,
        access_control_strength, resource_management_score,
        control_flow_complexity
    """
    records = []
    raw_path = Path(raw_dir)
    snippet_id = 0

    if not raw_path.exists():
        logger.warning(
            "Raw directory not found: %s. Returning empty DataFrame.", raw_dir
        )
        return _empty_df()

    for cwe_dir in sorted(raw_path.iterdir()):
        if not cwe_dir.is_dir():
            continue
        cwe_type = cwe_dir.name  # e.g. "CWE89"

        for src_file in sorted(cwe_dir.rglob("*")):
            ext = src_file.suffix.lower()
            if ext not in _LANG_MAP:
                continue
            language = _LANG_MAP[ext]

            try:
                code = src_file.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                logger.warning("Cannot read %s: %s", src_file, exc)
                continue

            # Juliet naming: "good" in stem → benign
            label = 0 if "good" in src_file.stem.lower() else 1
            features = extract_features(code, language)

            records.append({
                "snippet_id": snippet_id,
                "code":       code,
                "label":      label,
                "cwe_type":   cwe_type,
                "language":   language,
                **features,
            })
            snippet_id += 1

    logger.info("Parsed %d snippets from %s", len(records), raw_dir)
    return pd.DataFrame(records) if records else _empty_df()


def _empty_df() -> pd.DataFrame:
    """Return an empty DataFrame with the expected schema."""
    return pd.DataFrame(columns=[
        "snippet_id", "code", "label", "cwe_type", "language",
        "input_validation_score", "sensitive_data_exposure",
        "access_control_strength", "resource_management_score",
        "control_flow_complexity",
    ])
