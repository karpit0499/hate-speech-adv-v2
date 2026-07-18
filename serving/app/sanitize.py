import re

def sanitize(text: str) -> str:
    # Defense in depth: strip control chars, collapse whitespace, cap length.
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:5000]