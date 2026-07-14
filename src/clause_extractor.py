"""Extract termination, confidentiality, and liability clauses via Gemini."""

import logging
from typing import Any

from src.llm_engine import LLMEngine
from src.text_preprocessor import truncate_for_context
from prompts.clause_extraction import (
    CLAUSE_EXTRACTION_PROMPT,
    build_few_shot_prompt,
)

logger = logging.getLogger(__name__)

_NOT_FOUND = "Not found in this contract"
_CLAUSE_KEYS = ("termination_clause", "confidentiality_clause", "liability_clause")


class ClauseExtractor:
    """Extracts key legal clauses from contract text using an LLM."""

    def __init__(
        self,
        llm: LLMEngine,
        few_shot_examples: dict[str, list] | None = None,
    ) -> None:
        self.llm = llm
        self.few_shot_examples = few_shot_examples

    def extract(self, contract_text: str) -> dict[str, str]:
        """Return a dict with termination, confidentiality, and liability clauses."""
        text = truncate_for_context(contract_text)

        # Prefer few-shot if examples are available
        if self.few_shot_examples and any(self.few_shot_examples.values()):
            prompt = build_few_shot_prompt(text, self.few_shot_examples)
        else:
            prompt = CLAUSE_EXTRACTION_PROMPT.format(contract_text=text)

        try:
            result = self.llm.generate_json(prompt)
            return self._validate(result)
        except Exception as exc:
            logger.error("Clause extraction failed: %s", exc)
            return {k: _NOT_FOUND for k in _CLAUSE_KEYS}

    @staticmethod
    def _validate(raw: dict[str, Any]) -> dict[str, str]:
        """Ensure all expected keys are present and values are strings."""
        validated: dict[str, str] = {}
        for key in _CLAUSE_KEYS:
            value = raw.get(key, _NOT_FOUND)
            if not isinstance(value, str) or not value.strip():
                value = _NOT_FOUND
            validated[key] = value.strip()
        return validated
