"""Prompt template for contract summarisation."""

SUMMARIZATION_PROMPT = """\
You are an expert legal analyst. Read the contract below and produce a concise \
summary of **exactly 100–150 words**.

Your summary **must** cover all three areas:

1. **Purpose** — What is the agreement about? What type of contract is it?
2. **Key Obligations** — What are the main obligations of each party?
3. **Risks / Penalties** — What are the notable risks, penalties, or \
liabilities mentioned?

### Rules
- Be precise and factual.
- Do NOT invent information not present in the contract.
- Avoid unnecessary legal jargon; keep it readable.
- Your response must be ONLY the summary text (no JSON, no headers).

### Contract Text

{contract_text}
"""
