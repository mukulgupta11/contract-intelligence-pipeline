"""Generate concise contract summaries via Gemini."""

import logging

from src.llm_engine import LLMEngine
from src.text_preprocessor import truncate_for_context
from prompts.summarization import SUMMARIZATION_PROMPT
from config import config

logger = logging.getLogger(__name__)


class ContractSummarizer:
    """Produces a 100-150 word summary highlighting purpose, obligations, and risks."""

    def __init__(self, llm: LLMEngine) -> None:
        self.llm = llm

    def summarize(self, contract_text: str) -> str:
        """Generate and return a contract summary."""
        text = truncate_for_context(contract_text)
        prompt = SUMMARIZATION_PROMPT.format(contract_text=text)

        try:
            summary = self.llm.generate(prompt).strip()
            summary = self._clean(summary)
            self._log_word_count(summary)
            return summary
        except Exception as exc:
            logger.error("Summarization failed: %s", exc)
            return "Summary generation failed."

    @staticmethod
    def _clean(summary: str) -> str:
        """Remove markdown wrappers the model sometimes adds."""
        # Strip wrapping quotes
        if summary.startswith('"') and summary.endswith('"'):
            summary = summary[1:-1]
        # Strip markdown bold/header artefacts at start
        for prefix in ("**Summary:**", "**Summary**", "Summary:", "## Summary"):
            if summary.startswith(prefix):
                summary = summary[len(prefix) :].strip()
        return summary

    @staticmethod
    def _log_word_count(summary: str) -> None:
        wc = len(summary.split())
        level = (
            logging.INFO
            if config.summary_min_words <= wc <= config.summary_max_words
            else logging.WARNING
        )
        logger.log(level, "Summary word count: %d (target: %d–%d)",
                    wc, config.summary_min_words, config.summary_max_words)
