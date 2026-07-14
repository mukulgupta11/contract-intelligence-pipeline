"""Semantic search over extracted clauses using Gemini embeddings + cosine similarity."""

import logging
from dataclasses import dataclass

import numpy as np
from google import genai

from config import config

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A single semantic search hit."""

    contract_id: str
    contract_title: str
    clause_type: str
    clause_text: str
    score: float


class SemanticSearch:
    """Build an in-memory embedding index and query it with natural language."""

    def __init__(self, llm) -> None:
        """Accept an LLMEngine instance (used for document embeddings)."""
        self.llm = llm
        self._client = genai.Client(api_key=config.gemini_api_key)
        self._embeddings: np.ndarray | None = None
        self._metadata: list[dict] = []

    @property
    def is_indexed(self) -> bool:
        return self._embeddings is not None and len(self._metadata) > 0

    def index_clauses(self, clauses: list[dict]) -> None:
        """Embed all clauses and store for later querying.

        Parameters
        ----------
        clauses : list[dict]
            Each dict must have keys: contract_id, contract_title, clause_type, text.
        """
        if not clauses:
            logger.warning("No clauses to index.")
            return

        texts = [c["text"] for c in clauses]
        logger.info("Embedding %d clauses …", len(texts))

        vectors = self.llm.embed(texts)
        self._embeddings = np.array(vectors, dtype=np.float32)
        self._metadata = clauses

        logger.info(
            "Semantic index built — %d vectors, dim=%d",
            self._embeddings.shape[0],
            self._embeddings.shape[1],
        )

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Find the most semantically similar clauses to *query*."""
        if not self.is_indexed:
            logger.warning("Index is empty — call index_clauses() first.")
            return []

        # Embed the query
        result = self._client.models.embed_content(
            model=config.embedding_model,
            contents=[query],
        )
        query_vec = np.array(result.embeddings[0].values, dtype=np.float32)

        # Cosine similarity
        norms = np.linalg.norm(self._embeddings, axis=1) * np.linalg.norm(query_vec)
        norms = np.where(norms == 0, 1e-10, norms)  # avoid division by zero
        similarities = np.dot(self._embeddings, query_vec) / norms

        top_indices = np.argsort(similarities)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            meta = self._metadata[idx]
            results.append(
                SearchResult(
                    contract_id=meta["contract_id"],
                    contract_title=meta["contract_title"],
                    clause_type=meta["clause_type"],
                    clause_text=meta["text"],
                    score=float(similarities[idx]),
                )
            )

        return results
