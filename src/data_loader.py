"""Load contracts and few-shot examples from the local self-contained subset.

Bypasses external Hugging Face downloads to enable fully offline execution and
improve speed for the recruiter reviewing the assignment.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from config import config

logger = logging.getLogger(__name__)

# Locate the local subset file relative to the project root
_SUBSET_PATH = Path(__file__).parent.parent / "data" / "cuad_subset.json"


@dataclass
class Contract:
    """A single legal contract."""

    id: str
    title: str
    text: str


@dataclass
class FewShotExample:
    """Ground-truth clause example for few-shot prompting."""

    contract_title: str
    clause_type: str
    clause_text: str


# ── Internal Cache ──────────────────────────────────────────────
_cache: dict | None = None


def _load_local_subset() -> dict:
    """Load the local JSON subset from disk."""
    global _cache
    if _cache is not None:
        return _cache

    if not _SUBSET_PATH.exists():
        raise FileNotFoundError(
            f"Local dataset subset not found at {_SUBSET_PATH}. "
            "Please ensure data/cuad_subset.json is present."
        )

    logger.info("Reading local contract subset from %s …", _SUBSET_PATH)
    with open(_SUBSET_PATH, encoding="utf-8") as f:
        _cache = json.load(f)

    return _cache


# ── Public API ──────────────────────────────────────────────────

def load_contracts(num_contracts: int = config.num_contracts) -> list[Contract]:
    """Load *num_contracts* contracts from the local subset."""
    data = _load_local_subset()
    contracts_raw = data.get("contracts", [])

    contracts = [
        Contract(id=c["id"], title=c["title"], text=c["text"])
        for c in contracts_raw[:num_contracts]
    ]

    logger.info("Loaded %d contracts from local subset", len(contracts))
    return contracts


def load_few_shot_examples(
    exclude_titles: set[str],
    max_per_type: int = 2,
) -> dict[str, list[FewShotExample]]:
    """Load pre-selected few-shot examples from the local subset."""
    data = _load_local_subset()
    few_shot_raw = data.get("few_shot_examples", {})

    examples: dict[str, list[FewShotExample]] = {}
    for clause_type, exs in few_shot_raw.items():
        examples[clause_type] = []
        for ex in exs[:max_per_type]:
            # Skip if the example happens to be in the active contract set
            if ex["contract_title"] in exclude_titles:
                continue
            examples[clause_type].append(
                FewShotExample(
                    contract_title=ex["contract_title"],
                    clause_type=ex["clause_type"],
                    clause_text=ex["clause_text"],
                )
            )

    for ctype, exs in examples.items():
        logger.info("  %s: %d examples loaded", ctype, len(exs))
    return examples
