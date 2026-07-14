"""
Contract Processing Pipeline — Main Entry Point

Analyses 50 legal contracts from the CUAD dataset using Google Gemini:
  1. Load & preprocess contracts
  2. Extract termination, confidentiality, and liability clauses
  3. Generate 100-150 word summaries
  4. Build a semantic search index over extracted clauses
  5. Export results to CSV + JSON

Usage:
    python main.py                     # Run full pipeline
    python main.py --contracts 10      # Process only 10 contracts
    python main.py --no-search         # Skip semantic search (faster)
"""

import argparse
import json
import logging
import sys
import time

from tqdm import tqdm

from config import config
from src.data_loader import load_contracts, load_few_shot_examples
from src.text_preprocessor import normalize_text
from src.llm_engine import LLMEngine
from src.clause_extractor import ClauseExtractor
from src.contract_summarizer import ContractSummarizer
from src.semantic_search import SemanticSearch
from src.output_generator import save_results

_NOT_FOUND = "Not found in this contract"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(name)s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CUAD Contract Processing Pipeline")
    parser.add_argument(
        "--contracts", type=int, default=config.num_contracts,
        help=f"Number of contracts to process (default: {config.num_contracts})",
    )
    parser.add_argument(
        "--no-search", action="store_true",
        help="Skip semantic search indexing (faster run)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_time = time.time()

    # ── Step 1: Load contracts ──────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 1 — Loading %d contracts from CUAD", args.contracts)
    logger.info("=" * 60)

    contracts = load_contracts(num_contracts=args.contracts)
    contract_titles = {c.title for c in contracts}

    # ── Step 2: Load few-shot examples ──────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 2 — Loading few-shot examples (bonus)")
    logger.info("=" * 60)

    few_shot_examples = load_few_shot_examples(exclude_titles=contract_titles)

    # ── Step 3: Initialize LLM components ───────────────────────
    logger.info("=" * 60)
    logger.info("STEP 3 — Initializing Gemini LLM engine")
    logger.info("=" * 60)

    llm = LLMEngine()
    extractor = ClauseExtractor(llm, few_shot_examples=few_shot_examples)
    summarizer = ContractSummarizer(llm)
    search_engine = SemanticSearch(llm)

    # ── Step 4: Process each contract ───────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 4 — Processing contracts (extraction + summarization)")
    logger.info("=" * 60)

    results: list[dict] = []
    all_clauses: list[dict] = []

    # Try to load existing results from disk to support resuming
    existing_results = []
    processed_ids = set()
    results_file = config.output_dir / "results.json"
    if results_file.exists():
        try:
            with open(results_file, encoding="utf-8") as f:
                existing_results = json.load(f)
                if isinstance(existing_results, list):
                    processed_ids = {r["contract_id"] for r in existing_results if "contract_id" in r}
                    logger.info("Found %d already processed contracts in %s", len(processed_ids), results_file)
                else:
                    existing_results = []
                    logger.warning("Existing results file %s is not a list. Restarting fresh.", results_file)
        except Exception as e:
            logger.warning("Failed to load existing results: %s. Restarting fresh.", e)

    for contract in tqdm(contracts, desc="Processing contracts", unit="contract"):
        if contract.id in processed_ids:
            # Reconstruct the result and clause list from loaded file
            for row in existing_results:
                if row.get("contract_id") == contract.id:
                    results.append(row)
                    # Collect non-empty clauses for semantic search
                    for clause_type in ("termination_clause", "confidentiality_clause", "liability_clause"):
                        text = row.get(clause_type, _NOT_FOUND)
                        if text and text != _NOT_FOUND:
                            all_clauses.append({
                                "contract_id": contract.id,
                                "contract_title": contract.title,
                                "clause_type": clause_type.replace("_clause", ""),
                                "text": text,
                            })
                    break
            logger.info("✓ %s — Loaded from disk (skipped)", contract.id)
            continue

        clean_text = normalize_text(contract.text)

        # 4a — Clause extraction
        clauses = extractor.extract(clean_text)

        # 4b — Summarization
        summary = summarizer.summarize(clean_text)

        # Assemble result row
        result = {
            "contract_id": contract.id,
            "contract_title": contract.title,
            "summary": summary,
            "termination_clause": clauses.get("termination_clause", _NOT_FOUND),
            "confidentiality_clause": clauses.get("confidentiality_clause", _NOT_FOUND),
            "liability_clause": clauses.get("liability_clause", _NOT_FOUND),
        }
        results.append(result)

        # Collect non-empty clauses for semantic search
        for clause_type in ("termination_clause", "confidentiality_clause", "liability_clause"):
            text = clauses.get(clause_type, _NOT_FOUND)
            if text != _NOT_FOUND:
                all_clauses.append({
                    "contract_id": contract.id,
                    "contract_title": contract.title,
                    "clause_type": clause_type.replace("_clause", ""),
                    "text": text,
                })

        logger.info("✓ %s — %s", contract.id, contract.title[:50])

        # Progressive save after each contract processing
        save_results(results, config.output_dir)

    # ── Step 5: Semantic search index (bonus) ───────────────────
    if not args.no_search and all_clauses:
        logger.info("=" * 60)
        logger.info("STEP 5 — Building semantic search index (%d clauses)", len(all_clauses))
        logger.info("=" * 60)

        search_engine.index_clauses(all_clauses)

        # Demo queries
        demo_queries = [
            "early termination penalty",
            "non-disclosure of trade secrets",
            "limitation of liability for indirect damages",
        ]
        for query in demo_queries:
            hits = search_engine.search(query, top_k=3)
            logger.info("─" * 40)
            logger.info("Search: \"%s\"", query)
            for hit in hits:
                logger.info(
                    "  [%.3f] %s (%s): %s…",
                    hit.score,
                    hit.contract_id,
                    hit.clause_type,
                    hit.clause_text[:80],
                )
    else:
        logger.info("Semantic search skipped.")

    # ── Step 6: Export results ──────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 6 — Exporting results")
    logger.info("=" * 60)

    save_results(results, config.output_dir)

    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info("✅ Pipeline complete — %d contracts in %.1f seconds", len(results), elapsed)
    logger.info("   Output: %s/results.csv  •  %s/results.json", config.output_dir, config.output_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
