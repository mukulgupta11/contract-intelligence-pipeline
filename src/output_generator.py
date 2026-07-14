"""Export pipeline results to CSV and JSON."""

import csv
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_CSV_COLUMNS = [
    "contract_id",
    "contract_title",
    "summary",
    "termination_clause",
    "confidentiality_clause",
    "liability_clause",
]


def save_results(results: list[dict], output_dir: Path) -> None:
    """Write results to both CSV and JSON files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    _write_csv(results, output_dir / "results.csv")
    _write_json(results, output_dir / "results.json")


def _write_csv(results: list[dict], path: Path) -> None:
    """Write results to a CSV file with proper encoding."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    logger.info("CSV saved → %s (%d rows)", path, len(results))


def _write_json(results: list[dict], path: Path) -> None:
    """Write results to a pretty-printed JSON file."""
    # Filter to only the expected columns for clean output
    clean = [
        {k: row.get(k, "") for k in _CSV_COLUMNS}
        for row in results
    ]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(clean, f, indent=2, ensure_ascii=False)
    logger.info("JSON saved → %s (%d records)", path, len(results))
