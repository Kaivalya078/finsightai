"""
FinSight AI — Query Orchestrator (Stage 6)
==========================================
Pure orchestration layer that connects:
    parse_query  →  build_plan  →  CorpusManager.execute_plan

No retrieval logic, no parsing logic, no embedding logic.
"""

from typing import Callable, List

import numpy as np

from .query_understanding import parse_query
from .search_plan_builder import build_plan
from .search_plan import SearchPlan
from core.corpus_manager import CorpusManager
from core.metadata_schema import RetrievalResult


def retrieve_context(
    raw_query: str,
    corpus_manager: CorpusManager,
    embed_query: Callable[[str], np.ndarray],
    default_top_k: int = 5,
) -> List[RetrievalResult]:
    entities = corpus_manager.list_available_entities()
    companies = entities.get("companies", [])
    parsed = parse_query(raw_query, companies)
    plan = build_plan(parsed, default_top_k)
    results = corpus_manager.execute_plan(plan, embed_query)
    return results
