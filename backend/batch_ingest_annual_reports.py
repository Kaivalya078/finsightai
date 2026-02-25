"""
FinSight AI - Batch Ingestion: NIFTY 50 Annual Reports
========================================================
Deterministic batch ingestion script for annual reports.

Walks backend/data/<SYMBOL>/<YEAR>.pdf and ingests each PDF
via the existing CorpusManager pipeline.

Usage:
    python batch_ingest_annual_reports.py

Author: FinSight AI Team
Stage: 7 (Batch Ingestion)
"""

import os
import re
import sys

from dotenv import load_dotenv
load_dotenv()

from retriever_pipeline import RetrieverPipeline
from corpus_manager import CorpusManager
from lookup_index import ImmutableRangeError
from cache_utils import has_leftover_tmp, clean_cache


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
CACHE_DIR = os.getenv("INDEX_CACHE_DIR", "index_cache")
YEAR_PDF_RE = re.compile(r"^\d{4}\.pdf$")


def main():
    # ── Step 1: Validate data directory ──────────────────────────────
    if not os.path.isdir(DATA_DIR):
        print(f"Data directory not found: {DATA_DIR}")
        sys.exit(1)

    # ── Step 2: Init pipeline + corpus manager ───────────────────────
    pipeline = RetrieverPipeline()
    corpus = CorpusManager(pipeline)

    # ── Step 3: Load existing index + registry (if any) ──────────────
    if os.path.exists(CACHE_DIR):
        if has_leftover_tmp(CACHE_DIR):
            clean_cache(CACHE_DIR)

        index_loaded = pipeline.load_index(CACHE_DIR)

        if index_loaded:
            registry_loaded = corpus.load_registry(CACHE_DIR)

            if registry_loaded:
                integrity_ok = corpus.validate_cache_integrity(
                    pipeline.index.ntotal
                )
                if integrity_ok:
                    corpus.init_lookup_index(CACHE_DIR, pipeline.index.ntotal)
                else:
                    print("Cache integrity check failed. Cannot proceed.")
                    sys.exit(1)
            else:
                print("Failed to load registry. Cannot proceed.")
                sys.exit(1)

    # ── Step 4: Discover company folders (sorted alphabetically) ─────
    companies = sorted(
        entry
        for entry in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, entry))
    )

    # ── Step 5: Ingest each PDF ──────────────────────────────────────
    for company in companies:
        company_dir = os.path.join(DATA_DIR, company)

        # Collect only valid year PDFs, sorted numerically
        pdfs = sorted(
            f
            for f in os.listdir(company_dir)
            if YEAR_PDF_RE.match(f)
        )

        for pdf_name in pdfs:
            year = pdf_name.replace(".pdf", "")
            pdf_path = os.path.join(company_dir, pdf_name)

            print(f"Ingesting {company} - {year}")

            try:
                corpus.add_document(
                    pdf_path=pdf_path,
                    company=company,
                    document_type="Annual_Report",
                    year=year,
                )
            except (ValueError, ImmutableRangeError) as e:
                print(f"Skipped (duplicate): {e}")
                continue
            except Exception as e:
                print(f"Error ingesting {company}/{year}: {e}")
                sys.exit(1)

            # Three-phase commit after each successful ingestion
            corpus.save_registry(CACHE_DIR)
            pipeline.save_index(CACHE_DIR, pdf_path)
            corpus.save_lookup_index(CACHE_DIR)

    print("Batch ingestion complete.")


if __name__ == "__main__":
    main()
