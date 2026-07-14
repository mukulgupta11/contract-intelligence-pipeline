# Legal Contract Intelligence Pipeline (CUAD and Gemini LLM)

A modular, production-grade Python pipeline that leverages Google Gemini models to analyze legal contracts, extract key clauses verbatim, generate structured summaries, and perform semantic search over legal text. 

The pipeline is pre-packaged with a clean, self-contained local subset of the Atticus Project's CUAD dataset to ensure instant replication, zero-setup data loading, and fully offline-capable operations.

---

## Pipeline Architecture

The system is designed with a clean separation of concerns, decoupled into preprocessing, LLM interface, semantic indexing, and output serialization layers.

```
┌─────────────────────────────────────────────────────────────────┐
│                      main.py (Orchestrator)                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────────┐    ┌───────────────┐  │
│  │  Data Loader  │───▶│ Text Preprocessor│───▶│  LLM Engine   │  │
│  │  (Local JSON) │    │  (Normalization) │    │  (Gemini API) │  │
│  └──────────────┘    └──────────────────┘    └──────┬────────┘  │
│                                                      │          │
│                              ┌───────────────────────┼────────┐ │
│                              │                       │        │ │
│                    ┌─────────▼──────┐    ┌───────────▼──────┐ │ │
│                    │Clause Extractor│    │  Summarizer      │ │ │
│                    │(Few-shot/Zero) │    │ (100-150 words)  │ │ │
│                    └────────┬───────┘    └────────┬─────────┘ │ │
│                             │                     │           │ │
│                    ┌────────▼─────────────────────▼─────────┐ │ │
│                    │         Output Generator                │ │ │
│                    │        (CSV + JSON export)              │ │ │
│                    └────────────────┬────────────────────────┘ │ │
│                                     │                         │ │
│                    ┌────────────────▼────────────────────────┐ │ │
│                    │     Semantic Search                      │ │ │
│                    │  (Gemini embeddings + cosine similarity) │ │ │
│                    └─────────────────────────────────────────┘ │ │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Document Processing Workflow

```
Local Dataset (data/cuad_subset.json)
        │
        ▼
  Load 50 contracts  ──────▶  Load few-shot examples
        │                     (harvested from remaining contracts)
        ▼
  Normalize text
  (Unicode canonicalization, horizontal/vertical spacing cleanup)
        │
        ├──────────────────────────────────┐
        ▼                                  ▼
  Clause Extraction                   Summarization
  (Structured JSON + Few-shot)        (Word-count targeted summary)
        │                                  │
        ▼                                  │
  Semantic Search Index                    │
  (Embedding vector computation)           │
        │                                  │
        └──────────────┬───────────────────┘
                       ▼
               CSV + JSON Output
```

---

## Project Structure

```
├── config.py                    # Global configuration parameters
├── main.py                      # Core pipeline execution orchestrator
├── requirements.txt             # Minimal, pinned Python dependencies
├── .env.example                 # Environment variables template
├── data/
│   └── cuad_subset.json         # Self-contained subset of 50 contracts + annotations
├── src/
│   ├── data_loader.py           # Pre-packaged local data reader
│   ├── text_preprocessor.py     # Text cleaning & spacing normalizer
│   ├── llm_engine.py            # Unified interface for Gemini SDK
│   ├── clause_extractor.py      # LLM handler for clause extraction
│   ├── contract_summarizer.py   # LLM handler for structured summaries
│   ├── semantic_search.py       # In-memory vector database and cosine search
│   └── output_generator.py      # CSV and JSON file writers
└── prompts/
    ├── clause_extraction.py     # Few-shot and zero-shot legal prompts
    └── summarization.py         # Summary instructions and constraints
```

---

## Detailed Pipeline Design

### 1. Preprocessing and Text Normalization
Legal documents from SEC filings contain irregular formatting, smart quote variances, and unicode artifacts. The `text_preprocessor.py` cleanses the raw data before passing it to the LLM:
*   Applies **NFKC unicode normalization** for character consistency.
*   Converts typographic smart quotes and dashes to standard ASCII equivalents.
*   Collapses duplicate horizontal whitespace and redundant line breaks.
*   Removes stray pagination headers and standalone page numbers.
*   Safe truncation ensures the text fits comfortably inside the LLM context window.

### 2. LLM Engine and Quota Engineering
The pipeline implements a production-ready wrapper (`llm_engine.py`) around the current `google-genai` SDK:
*   **Rate Limiting**: Restricts requests to 15 Requests Per Minute (RPM) to align with standard free-tier thresholds, avoiding transient rate limit issues.
*   **Structured Outputs**: Utilizes native JSON output mode (`response_mime_type="application/json"`) to guarantee clean, machine-parseable data.
*   **Retry Mechanisms**: Implements exponential-backoff retry handling to handle network drops or API hiccups.

### 3. Prompt Engineering and In-Context Learning
*   **Structured Clause Extraction**: Prompts enforce the extraction of verbatim, un-paraphrased legal text from the contracts. The target clauses are:
    1.  *Termination Clauses*: Convenience/cause terms, notice periods, and post-termination services.
    2.  *Confidentiality Clauses*: Scope of confidential information, non-disclosure terms, and duration.
    3.  *Liability Clauses*: Limitations of liability, caps, indemnifications, and carve-outs.
*   **Few-Shot Support**: The data loader automatically harvests actual annotated ground-truth examples from other contracts in the dataset, using them as in-context examples to significantly improve extraction accuracy.
*   **Targeted Summarization**: Prompts enforce a strict 100-150 word summary covering the contract's purpose, key obligations, and notable risks/penalties.

### 4. Embedded Semantic Search
Rather than introducing heavy database dependencies (like FAISS or Pinecone), the system uses an elegant, lightweight vector index in `semantic_search.py`:
*   Generates 3072-dimensional embeddings of all extracted clauses using `gemini-embedding-001`.
*   Uses `numpy` to calculate cosine similarities across vectors.
*   Enables querying the processed clauses using natural language queries (e.g., "limitation of liability for indirect damages") with real-time similarity scoring.

---

## Technical Performance and Quota Selection

| Model | Category | Quota Limits | Rationale for Project |
|---|---|---|---|
| **Gemini 3.1 Flash Lite** | Text Generation | 15 RPM / 250K TPM / 500 RPD | Selected as the primary model. High daily quota (500 RPD) allows processing the entire 50-contract set in a single run. |
| **Gemini Embedding 001** | Embeddings | 100 RPM / 30K TPM / 1K RPD | High rate limits make it perfect for embedding all extracted clauses to build the semantic search index. |

---

## Setup and Installation

### Prerequisites
*   Python 3.10 or higher
*   A Google Gemini API key (obtainable for free on Google AI Studio)

### Installation Steps

1.  **Clone the Repository**
    ```bash
    git clone <repo-url>
    cd <repo-name>
    ```

2.  **Create a Virtual Environment**
    ```bash
    python -m venv .venv
    source .venv/bin/activate      # On Linux/macOS
    # .venv\Scripts\activate       # On Windows
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Environment Variables**
    Create a `.env` file based on `.env.example` and insert your Gemini API Key:
    ```bash
    cp .env.example .env
    # Edit the file and paste your key: GEMINI_API_KEY=your_key
    ```

5.  **Run the Pipeline**
    ```bash
    python main.py
    ```

### Command Line Options
*   Run on a custom subset (e.g., only 5 contracts):
    ```bash
    python main.py --contracts 5
    ```
*   Disable semantic search indexing to speed up the run:
    ```bash
    python main.py --no-search
    ```

---

## Output Formats

### CSV Output (output/results.csv)
Saves results into a flat tabular format. Columns match the assignment schema:
`[contract_id, contract_title, summary, termination_clause, confidentiality_clause, liability_clause]`

### JSON Output (output/results.json)
Generates structured JSON data:
```json
[
  {
    "contract_id": "CUAD_001",
    "contract_title": "LIMEENERGYCO_09_09_1999-EX-10-DISTRIBUTOR AGREEMENT",
    "summary": "This Distributor Agreement establishes an exclusive partnership...",
    "termination_clause": "Either party may terminate this Agreement upon 30 days...",
    "confidentiality_clause": "None of the parties hereto shall during the term...",
    "liability_clause": "Company and Distributor agree to indemnify..."
  }
]
```

---

## Key Design Choices

*   **Local Package vs. Network Fetch**: Bundling the 50-contract subset into a local JSON (`data/cuad_subset.json`) eliminates the 50MB runtime dataset download. This guarantees instant start-up and fully offline compatibility for the data loader.
*   **Native SDK and JSON Mode**: Relying on the new `google-genai` SDK and using the API's native JSON schema parsing removes brittle regular expression parsing and ensures outputs conform exactly to the required schemas.
*   **Vector Search without Server Bloat**: Using clean `numpy` matrix calculations for cosine similarity avoids installing heavy native binaries (like FAISS) while delivering instantaneous search results on small-to-medium datasets.
*   **Rate Limits and Fail-Fast Strategy**: The global configuration enables configuring RPM and retries. Defaults are tuned to fail fast on hard API blocks rather than locking the terminal.
