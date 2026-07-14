"""Gemini API wrapper with rate-limiting, retries, and structured output.

Uses the current ``google-genai`` SDK (replaces the deprecated
``google-generativeai`` package).
"""

import json
import logging
import time
from typing import Any

from google import genai
from google.genai import types

from config import config

logger = logging.getLogger(__name__)


class LLMEngine:
    """Thin wrapper around Gemini that handles rate-limiting and retries."""

    def __init__(self) -> None:
        if not config.gemini_api_key:
            raise ValueError(
                "GEMINI_API_KEY not set. Copy .env.example → .env and add your key."
            )
        self.client = genai.Client(api_key=config.gemini_api_key)
        self._min_interval = 60.0 / config.requests_per_minute
        self._last_call: float = 0.0

    # ── Rate limiting ───────────────────────────────────────────
    def _wait(self) -> None:
        """Block until the minimum interval between requests has elapsed."""
        elapsed = time.time() - self._last_call
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_call = time.time()

    # ── Core generation ─────────────────────────────────────────
    def generate(self, prompt: str, *, json_mode: bool = False) -> str:
        """Send a prompt and return the response text.

        Parameters
        ----------
        prompt : str
            The full prompt (system instructions + user content).
        json_mode : bool
            If *True*, instruct Gemini to return valid JSON.
        """
        gen_config: dict[str, Any] = {}
        if json_mode:
            gen_config["response_mime_type"] = "application/json"

        for attempt in range(1, config.max_retries + 1):
            self._wait()
            try:
                response = self.client.models.generate_content(
                    model=config.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(**gen_config) if gen_config else None,
                )
                return response.text
            except Exception as exc:
                logger.warning(
                    "Gemini call failed (attempt %d/%d): %s",
                    attempt,
                    config.max_retries,
                    str(exc)[:200],
                )
                if attempt == config.max_retries:
                    raise
                time.sleep(config.retry_base_delay * (2 ** (attempt - 1)))

        return ""  # unreachable, satisfies type checker

    def generate_json(self, prompt: str) -> dict:
        """Generate and parse a JSON response."""
        raw = self.generate(prompt, json_mode=True)
        return json.loads(raw)

    # ── Embeddings ──────────────────────────────────────────────
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return embedding vectors for a list of texts.

        Batches automatically to respect API limits.
        """
        all_embeddings: list[list[float]] = []
        batch_size = config.embedding_batch_size

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            
            response_embeddings = None
            for attempt in range(1, config.max_retries + 1):
                self._wait()
                try:
                    result = self.client.models.embed_content(
                        model=config.embedding_model,
                        contents=batch,
                    )
                    response_embeddings = result.embeddings
                    break
                except Exception as exc:
                    logger.warning(
                        "Embedding call failed (attempt %d/%d): %s",
                        attempt,
                        config.max_retries,
                        str(exc)[:200],
                    )
                    if attempt == config.max_retries:
                        raise
                    time.sleep(config.retry_base_delay * (2 ** (attempt - 1)))
            
            if response_embeddings:
                all_embeddings.extend([e.values for e in response_embeddings])

        return all_embeddings
