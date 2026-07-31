#!/usr/bin/env python3
"""
Structured JSON Extractor (OpenAI, single-file, end-to-end)
=============================================================

Extracts entities from invoice/contract text into schema-validated JSON.

Enforcement has two layers:
  1. Generation-time constraint: the Pydantic schema below is passed
     directly to OpenAI's Structured Outputs (`client.responses.parse`,
     `text_format=ExtractionResult`), which constrains token generation so
     the model can only emit JSON matching that schema exactly (correct
     types, no missing/extra fields).
  2. Validation-time enforcement: the SDK already deserializes the result
     into a Pydantic object for us, but a few rules (e.g. "if
     document_type is 'invoice' then `invoice` must be populated") are
     cross-field checks that structured outputs can't express, so they're
     enforced again as Pydantic validators. If that check fails, or the
     model refuses / returns nothing, the script automatically retries
     with the error fed back to the model.

Usage:
    pip install openai pydantic
    export OPENAI_API_KEY=sk-...
    python structured_extractor.py examples/sample_invoice.txt
    python structured_extractor.py examples/sample_contract.txt --out result.json

    # or import and use directly:
    from structured_extractor import extract_document
    result = extract_document(open("invoice.txt").read())
    print(result.invoice.total_amount)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import date
from enum import Enum
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field, ValidationError, field_validator

logger = logging.getLogger("structured_extractor")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

MODEL = "gpt-5.6"
MAX_RETRIES = 2

SYSTEM_PROMPT = """You are a precise document-extraction engine for business documents \
(invoices and contracts).

Rules:
- Extract only information that is actually present in the document. Never invent values.
- If a field is not present or not determinable, leave it null (or an empty list for lists).
- Dates must be normalized to ISO 8601 (YYYY-MM-DD) whenever the source date is unambiguous.
  If the format is ambiguous (e.g. 03/04/2025), keep the original string and add a note to
  extraction_warnings instead of guessing.
- Numbers (amounts, quantities) must be plain numbers, not strings, and must not include
  currency symbols or thousands separators.
- For contracts, do NOT copy long verbatim clauses. Summarize termination_clause and
  key_obligations concisely in your own words (this avoids reproducing large blocks of the
  source text).
- Classify document_type as "invoice", "contract", or "unknown" and report your confidence.
- Record anything uncertain, illegible, contradictory, or missing-but-expected in
  extraction_warnings so a human reviewer can follow up.
"""


# ==========================================================================
# 1. SCHEMA — the contract the LLM output must satisfy
# ==========================================================================

class Party(BaseModel):
    """A person or organization referenced in a document."""
    name: str = Field(..., description="Full legal or display name")
    role: Optional[str] = Field(
        None, description="Role in the document, e.g. 'Vendor', 'Buyer', 'Lessor'"
    )
    address: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    tax_id: Optional[str] = Field(
        None, description="VAT/EIN/GSTIN or other tax identifier, if present"
    )


class LineItem(BaseModel):
    description: str
    quantity: Optional[float] = None
    unit_price: Optional[float] = None
    line_total: Optional[float] = None


class Invoice(BaseModel):
    document_type: str = Field("invoice", frozen=True)
    invoice_number: Optional[str] = None
    issue_date: Optional[str] = Field(None, description="ISO 8601 date (YYYY-MM-DD) if determinable")
    due_date: Optional[str] = None
    vendor: Optional[Party] = None
    customer: Optional[Party] = None
    line_items: List[LineItem] = Field(default_factory=list)
    subtotal: Optional[float] = None
    tax_amount: Optional[float] = None
    total_amount: Optional[float] = None
    currency: Optional[str] = None
    payment_terms: Optional[str] = None
    payment_due_in_days: Optional[int] = None

    @field_validator("issue_date", "due_date")
    @classmethod
    def _basic_date_shape(cls, v):
        if v is None:
            return v
        try:
            date.fromisoformat(v)
        except ValueError:
            pass  # keep the raw string rather than hard-failing extraction
        return v


class Signature(BaseModel):
    name: Optional[str] = None
    title: Optional[str] = None
    date: Optional[str] = None
    signed: bool = Field(False, description="True if a signature/initials appear to be present")


class Contract(BaseModel):
    document_type: str = Field("contract", frozen=True)
    title: Optional[str] = None
    contract_type: Optional[str] = Field(
        None, description="e.g. 'NDA', 'MSA', 'Lease', 'Employment Agreement'"
    )
    parties: List[Party] = Field(default_factory=list)
    effective_date: Optional[str] = None
    expiration_date: Optional[str] = None
    term_length: Optional[str] = None
    payment_terms: Optional[str] = None
    total_contract_value: Optional[float] = None
    currency: Optional[str] = None
    termination_clause: Optional[str] = Field(
        None, description="One or two sentence summary, not a verbatim quote"
    )
    governing_law: Optional[str] = None
    confidentiality: Optional[bool] = None
    auto_renewal: Optional[bool] = None
    key_obligations: List[str] = Field(
        default_factory=list, description="Short bullet summaries of key obligations, not verbatim text"
    )
    signatures: List[Signature] = Field(default_factory=list)


class DocumentType(str, Enum):
    invoice = "invoice"
    contract = "contract"
    unknown = "unknown"


class ExtractionResult(BaseModel):
    """
    Top-level object the model must return. Exactly one of `invoice` /
    `contract` should be populated, matching `document_type`.
    """
    document_type: DocumentType
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in document_type classification")
    invoice: Optional[Invoice] = None
    contract: Optional[Contract] = None
    extraction_warnings: List[str] = Field(
        default_factory=list,
        description="Notes about missing/ambiguous/illegible fields, low-confidence guesses, etc.",
    )

    @field_validator("contract")
    @classmethod
    def _contract_matches_type(cls, v, info):
        if info.data.get("document_type") == DocumentType.contract and v is None:
            raise ValueError("document_type is 'contract' but no contract object was returned")
        return v

    @field_validator("invoice")
    @classmethod
    def _invoice_matches_type(cls, v, info):
        if info.data.get("document_type") == DocumentType.invoice and v is None:
            raise ValueError("document_type is 'invoice' but no invoice object was returned")
        return v


# ==========================================================================
# 2. EXTRACTOR — calls OpenAI with the schema, validates, retries
# ==========================================================================

class ExtractionError(Exception):
    """Raised when the model fails to produce schema-valid output after all retries."""


class StructuredExtractor:
    def __init__(self, api_key: Optional[str] = None, model: str = MODEL, max_retries: int = MAX_RETRIES):
        from openai import OpenAI  # lazy import so schema-only usage doesn't need the package

        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        self.model = model
        self.max_retries = max_retries

    def extract(self, document_text: str) -> ExtractionResult:
        """
        Extract structured entities from raw document text (invoice or
        contract). Returns a validated ExtractionResult, or raises
        ExtractionError if the model cannot produce valid output within
        max_retries attempts.
        """
        input_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Extract the structured data from this document:\n\n---\n{document_text}\n---",
            },
        ]

        last_error: Optional[str] = None
        for attempt in range(self.max_retries + 1):
            if last_error:
                input_messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Your previous output failed validation with this error:\n"
                            f"{last_error}\n"
                            "Return corrected data matching the schema exactly."
                        ),
                    }
                )

            try:
                response = self.client.responses.parse(
                    model=self.model,
                    input=input_messages,
                    text_format=ExtractionResult,
                )
            except Exception as e:  # network/API errors, refusals surfaced as exceptions in some SDK versions
                last_error = f"API call failed: {e}"
                logger.warning("Attempt %d: %s", attempt, last_error)
                continue

            # The SDK already parses/validates against the schema, but the
            # model can still "refuse" (e.g. safety refusal), in which case
            # output_parsed is None even though the call succeeded.
            refusal = getattr(response, "output_refusal", None) or getattr(response, "refusal", None)
            if response.output_parsed is None:
                last_error = refusal or "Model returned no parsed output (possible refusal or empty response)."
                logger.warning("Attempt %d: %s", attempt, last_error)
                continue

            # Re-validate independently. This re-runs our cross-field
            # validators (invoice/contract <-> document_type) even though
            # the SDK already built the object, and protects us if we ever
            # swap in a provider/model that *doesn't* enforce the schema
            # at generation time.
            try:
                return ExtractionResult.model_validate(response.output_parsed.model_dump())
            except ValidationError as e:
                last_error = str(e)
                logger.warning("Attempt %d: post-hoc validation failed: %s", attempt, last_error)
                continue

        raise ExtractionError(
            f"Failed to extract schema-valid JSON after {self.max_retries + 1} attempts. "
            f"Last error: {last_error}"
        )


def extract_document(document_text: str, api_key: Optional[str] = None) -> ExtractionResult:
    """Convenience one-shot function."""
    return StructuredExtractor(api_key=api_key).extract(document_text)


# ==========================================================================
# 3. CLI — end-to-end runner
# ==========================================================================

def main() -> int:
    parser = argparse.ArgumentParser(description="Extract structured JSON from an invoice/contract.")
    parser.add_argument("input_file", type=Path, help="Path to a .txt file with the document content")
    parser.add_argument("--out", type=Path, default=None, help="Write JSON result to this file (default: stdout)")
    parser.add_argument("--model", default=MODEL, help="OpenAI model to use")
    parser.add_argument("--max-retries", type=int, default=MAX_RETRIES)
    args = parser.parse_args()

    if not args.input_file.exists():
        print(f"error: {args.input_file} not found", file=sys.stderr)
        return 1

    text = args.input_file.read_text(encoding="utf-8")

    extractor = StructuredExtractor(model=args.model, max_retries=args.max_retries)
    try:
        result = extractor.extract(text)
    except ExtractionError as e:
        print(f"extraction failed: {e}", file=sys.stderr)
        return 2

    output = json.dumps(result.model_dump(mode="json"), indent=2)
    if args.out:
        args.out.write_text(output, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
