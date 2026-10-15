"""Redact sensitive data from tool output before passing it to the LLM or the UI.

All redaction is regex-based and conservative: it's better to over-redact than
to leak an internal hostname, token, or connection string.
"""
import re
from typing import Any

# Each entry: (compiled pattern, replacement)
_RULES: list[tuple[re.Pattern, str]] = [
    # JWT-shaped strings (check before generic hex so the whole token is replaced)
    (
        re.compile(r"eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+"),
        "<JWT>",
    ),
    # Connection strings: mongodb://, postgres://, redis://, etc.
    (
        re.compile(r"\b\w+://[^\s\"'<>]+"),
        "<CONNECTION_STRING>",
    ),
    # key=value or key: value patterns for secrets
    (
        re.compile(
            r"(?i)\b(token|password|passwd|secret|apikey|api[_-]key"
            r"|credential|auth|authorization|bearer)\b\s*[=:]\s*\S+",
            re.IGNORECASE,
        ),
        r"<REDACTED_CREDENTIAL>",
    ),
    # Kubernetes internal FQDNs: *.svc.cluster.local
    (
        re.compile(r"\b[\w-]+\.[\w-]+\.svc\.cluster\.local\b"),
        "<INTERNAL_HOST>",
    ),
    # Email addresses
    (
        re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
        "<EMAIL>",
    ),
    # IPv4 addresses (after connection strings so URLs aren't double-redacted)
    (
        re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
        "<IP>",
    ),
    # Long hex strings that look like tokens/hashes (>= 20 hex chars, word-bounded)
    (
        re.compile(r"\b[0-9a-fA-F]{20,}\b"),
        "<HEX_REDACTED>",
    ),
]


def redact(text: str) -> str:
    """Apply all redaction rules to *text* and return the sanitised result."""
    for pattern, replacement in _RULES:
        text = pattern.sub(replacement, text)
    return text


def redact_obj(obj: Any) -> Any:
    """Recursively redact all string values in a dict/list/str structure."""
    if isinstance(obj, str):
        return redact(obj)
    if isinstance(obj, dict):
        return {k: redact_obj(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [redact_obj(item) for item in obj]
    return obj
