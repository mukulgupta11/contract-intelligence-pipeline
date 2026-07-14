"""Prompt templates for clause extraction (zero-shot and few-shot)."""

# ── Zero-shot prompt ────────────────────────────────────────────

CLAUSE_EXTRACTION_PROMPT = """\
You are an expert legal contract analyst. Carefully read the contract below and \
extract the following three clause types.

### Clause Types

1. **Termination Clause** — Conditions under which either party may terminate \
the agreement, notice periods, termination for cause or convenience, and any \
post-termination obligations.

2. **Confidentiality Clause** — Obligations regarding confidential or \
proprietary information, non-disclosure terms, duration of confidentiality, \
and any exceptions.

3. **Liability Clause** — Limitations or caps on liability, indemnification \
obligations, warranty disclaimers, and any carve-outs (e.g. for IP \
infringement or wilful misconduct).

### Instructions

- Extract the **exact text** from the contract. Do NOT paraphrase.
- If multiple relevant sections exist for a clause type, combine the most \
important ones (max ~500 words each).
- If a clause type is not present in the contract, return the string \
"Not found in this contract".

### Output Format (strict JSON)

{{
  "termination_clause": "<extracted text or 'Not found in this contract'>",
  "confidentiality_clause": "<extracted text or 'Not found in this contract'>",
  "liability_clause": "<extracted text or 'Not found in this contract'>"
}}

### Contract Text

{contract_text}
"""


# ── Few-shot prompt ─────────────────────────────────────────────

FEW_SHOT_HEADER = """\
You are an expert legal contract analyst. Below are examples of correctly \
extracted clauses from other contracts, followed by a new contract for you to \
analyse.

"""

FEW_SHOT_EXAMPLE_TEMPLATE = """\
**Example — {clause_type} clause** (from "{contract_title}"):
\"\"\"{clause_text}\"\"\"

"""

FEW_SHOT_TASK = """\
Now extract clauses from the following contract using the same approach.

### Clause Types

1. **Termination Clause** — Conditions under which either party may terminate \
the agreement, notice periods, termination for cause or convenience.

2. **Confidentiality Clause** — Obligations regarding confidential or \
proprietary information, non-disclosure terms, duration of confidentiality.

3. **Liability Clause** — Limitations or caps on liability, indemnification \
obligations, warranty disclaimers.

### Instructions

- Extract the **exact text** from the contract. Do NOT paraphrase.
- If a clause type is not present, return "Not found in this contract".
- Keep each clause under ~500 words.

### Output Format (strict JSON)

{{
  "termination_clause": "<extracted text>",
  "confidentiality_clause": "<extracted text>",
  "liability_clause": "<extracted text>"
}}

### Contract Text

{contract_text}
"""


def build_few_shot_prompt(
    contract_text: str,
    examples: dict[str, list],
) -> str:
    """Assemble a few-shot clause-extraction prompt from ground-truth examples."""
    parts = [FEW_SHOT_HEADER]

    for clause_type, example_list in examples.items():
        for ex in example_list:
            parts.append(
                FEW_SHOT_EXAMPLE_TEMPLATE.format(
                    clause_type=clause_type.replace("_", " ").title(),
                    contract_title=ex.contract_title,
                    clause_text=ex.clause_text[:800],  # cap example length
                )
            )

    parts.append(FEW_SHOT_TASK.format(contract_text=contract_text))
    return "".join(parts)
