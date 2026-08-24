from __future__ import annotations

import hashlib
import html
import re
import unicodedata

STRICT_VERSION = "strict_v1"
GROUP_VERSION = "template_v1"

_URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_HANDLE_RE = re.compile(r"(?<!\w)@[A-Za-z0-9_]{1,15}\b")
_LEADING_RT_RE = re.compile(r"^(?:[!?.:;\-_*#\s]+)?rt\s+", re.IGNORECASE)
_WHITESPACE_RE = re.compile(r"\s+")


def strict_normalize(text: str) -> str:
    """Normalize representation-only differences without changing tokens."""
    value = unicodedata.normalize("NFKC", html.unescape(text)).casefold()
    return _WHITESPACE_RE.sub(" ", value).strip()


def group_normalize(text: str) -> str:
    """Collapse retweet/template variants that can leak across random splits."""
    value = strict_normalize(text)
    value = _URL_RE.sub("<url>", value)
    value = _HANDLE_RE.sub("@user", value)
    value = _LEADING_RT_RE.sub("", value)
    return _WHITESPACE_RE.sub(" ", value).strip()


def versioned_hash(normalized_text: str, version: str) -> str:
    payload = f"{version}\0{normalized_text}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def strict_group_hash(text: str) -> str:
    return versioned_hash(strict_normalize(text), STRICT_VERSION)


def template_group_hash(text: str) -> str:
    return versioned_hash(group_normalize(text), GROUP_VERSION)

