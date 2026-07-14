"""Text normalisation utilities for legal contracts."""

import re
import unicodedata


def normalize_text(text: str) -> str:
    """Clean and normalise raw contract text for LLM consumption.

    Steps
    -----
    1. Unicode normalisation (NFKC)
    2. Smart-quote / dash replacement
    3. Whitespace collapsing
    4. Page-number removal
    5. Strip leading/trailing whitespace
    """
    # 1. Canonical unicode form
    text = unicodedata.normalize("NFKC", text)

    # 2. Replace typographic characters with ASCII equivalents
    _REPLACEMENTS = {
        "\u2018": "'",   # left single quote
        "\u2019": "'",   # right single quote
        "\u201c": '"',   # left double quote
        "\u201d": '"',   # right double quote
        "\u2013": "-",   # en-dash
        "\u2014": "--",  # em-dash
        "\u2026": "...", # ellipsis
        "\xa0": " ",     # non-breaking space
    }
    for old, new in _REPLACEMENTS.items():
        text = text.replace(old, new)

    # 3. Collapse horizontal whitespace (preserve line breaks)
    text = re.sub(r"[ \t]+", " ", text)
    # Collapse 3+ consecutive blank lines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 4. Remove standalone page numbers (e.g. "\n  - 12 -\n" or "\n  12  \n")
    text = re.sub(r"\n\s*-?\s*\d{1,3}\s*-?\s*\n", "\n", text)

    # 5. Final trim
    text = text.strip()

    return text


def truncate_for_context(text: str, max_chars: int = 900_000) -> str:
    """Truncate text to fit within model context limits.

    Gemini 2.0 Flash supports ~1M tokens, but we leave headroom for the
    prompt template and response.  900k chars ≈ 225k tokens.
    """
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[… truncated for context length …]"
