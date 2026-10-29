"""Request normalization and deterministic, explainable feature extraction."""
from __future__ import annotations
import math
import re
from collections import Counter
from urllib.parse import unquote

SQL_TOKENS = (
    "select", "union", "insert", "update", "delete", "drop", "alter",
    "sleep", "benchmark", "or", "and", "having", "information_schema",
    "waitfor", "cast", "char", "concat"
)
XSS_TOKENS = (
    "<script", "</script", "javascript:", "onerror", "onload", "onclick",
    "onmouseover", "alert(", "document.cookie", "svg", "iframe"
)
TRAVERSAL_TOKENS = ("../", "..\\", "%2e%2e", "/etc/passwd", "boot.ini")
CMD_TOKENS = (";id", "|id", "&&", "||", "/bin/sh", "powershell", "cmd.exe")
PERCENT_RE = re.compile(r"%[0-9a-fA-F]{2}")
WORD_RE = re.compile(r"[a-zA-Z_]+")

def normalize_request(value: str) -> str:
    """Return a bounded, lowercase normalized representation.

    Repeated decoding is deliberately limited to two passes; unbounded decoding
    can be expensive and can produce behavior different from the target server.
    """
    text = str(value or "")[:20000]
    for _ in range(2):
        decoded = unquote(text)
        if decoded == text:
            break
        text = decoded
    return text.lower()

def shannon_entropy(text: str) -> float:
    if not text:
        return 0.0
    counts = Counter(text)
    n = len(text)
    return -sum((c/n) * math.log2(c/n) for c in counts.values())

def _count_terms(text: str, terms) -> int:
    return sum(len(re.findall(r"(?<![a-z0-9_])" + re.escape(t) + r"(?![a-z0-9_])", text, re.I)) for t in terms)

def extract_features(request_text: str) -> dict:
    """Feature names are stable and shared by training and online inference."""
    raw = str(request_text or "")[:20000]
    norm = normalize_request(raw)
    tokens = WORD_RE.findall(norm)
    length = max(len(norm), 1)
    return {
        "request_length": len(raw),
        "normalized_length": len(norm),
        "word_count": len(tokens),
        "unique_word_ratio": len(set(tokens)) / max(len(tokens), 1),
        "entropy": shannon_entropy(norm),
        "url_encoded_count": len(PERCENT_RE.findall(raw)),
        "percent_char_count": raw.count("%"),
        "special_char_ratio": sum(not c.isalnum() and not c.isspace() for c in norm) / length,
        "quote_count": norm.count("'") + norm.count('"'),
        "semicolon_count": norm.count(";"),
        "comment_marker_count": norm.count("--") + norm.count("/*") + norm.count("#"),
        "parenthesis_count": norm.count("(") + norm.count(")"),
        "equals_count": norm.count("="),
        "angle_bracket_count": norm.count("<") + norm.count(">"),
        "sql_keyword_hits": _count_terms(norm, SQL_TOKENS),
        "xss_token_hits": sum(norm.count(t) for t in XSS_TOKENS),
        "traversal_token_hits": sum(norm.count(t) for t in TRAVERSAL_TOKENS),
        "command_token_hits": sum(norm.count(t) for t in CMD_TOKENS),
        "has_union_select": int("union" in norm and "select" in norm),
        "has_tautology_like_pattern": int(bool(re.search(r"(['\"]?\s*(?:or|and)\s+['\"]?\d+['\"]?\s*=\s*['\"]?\d+)", norm))),
        "has_script_tag": int("<script" in norm or "</script" in norm),
        "has_event_handler": int(bool(re.search(r"\bon[a-z]{3,}\s*=", norm))),
        "has_javascript_scheme": int("javascript:" in norm),
        "has_path_traversal": int(any(t in norm for t in TRAVERSAL_TOKENS)),
        "has_command_indicator": int(any(t in norm for t in CMD_TOKENS)),
        "has_null_byte": int("\\x00" in raw or "%00" in raw.lower()),
        "has_repeated_encoding": int(bool(re.search(r"%25(?:[0-9a-f]{2})", raw.lower()))),
        "url_entropy": shannon_entropy(re.split(r"\s+", norm)[0]),
    }
