"""Central configuration for the contract processing pipeline."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent


@dataclass(frozen=True)
class Config:
    """Immutable pipeline configuration."""

    # ── Gemini API ──────────────────────────────────────────────
    gemini_api_key: str = field(
        default_factory=lambda: os.getenv("GEMINI_API_KEY", "")
    )
    model_name: str = "gemini-3.1-flash-lite"
    embedding_model: str = "gemini-embedding-001"

    # ── Dataset ─────────────────────────────────────────────────
    num_contracts: int = 50
    hf_dataset: str = "theatticusproject/cuad-qa"

    # ── Rate Limiting (free tier: 15 RPM, 1M TPM) ──────────────
    requests_per_minute: int = 15
    max_retries: int = 5
    retry_base_delay: float = 3.0

    # ── Output ──────────────────────────────────────────────────
    output_dir: Path = field(default_factory=lambda: BASE_DIR / "output")

    # ── Summary Constraints ─────────────────────────────────────
    summary_min_words: int = 100
    summary_max_words: int = 150

    # ── Semantic Search ─────────────────────────────────────────
    embedding_batch_size: int = 20
    search_top_k: int = 5


config = Config()
