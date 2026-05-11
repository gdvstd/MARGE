"""Expert-only live medical web search tool.

The orchestrator never imports this module. It is wired only inside
`MedicalExpertAgent`, so any `search_medical_web` call in the trace proves the
expert model initiated the retrieval step.

Source restriction:
- Default whitelist: MedlinePlus (medical encyclopedia, easy-to-read clinical
  summaries) and PubMed (peer-reviewed paper archive).
- Override via `MEDICAL_WEB_SEARCH_INCLUDE_DOMAINS` (comma- or semicolon-
  separated host list).
- Override result count via `MARGE_WEB_RAG_MAX_RESULTS` or
  `MEDICAL_WEB_SEARCH_MAX_RESULTS` (clamped to 1..5).
"""

from __future__ import annotations

import os
from typing import Any

from pydantic import BaseModel, Field

from packages.schemas.retrieval import RetrievedDocument

TOOL_NAME = "search_medical_web"

DEFAULT_MAX_RESULTS = 3
DEFAULT_INCLUDE_DOMAINS: tuple[str, ...] = (
    "medlineplus.gov",
    "pubmed.ncbi.nlm.nih.gov",
)

TOOL_DESCRIPTION = (
    "Search authoritative medical sources for current, citable guidance. By "
    "default the search is scoped to MedlinePlus (medical encyclopedia) and "
    "PubMed (peer-reviewed paper archive). Use this when a clinical claim "
    "needs to be grounded in current literature — guideline thresholds, "
    "diagnostic criteria, treatment recommendations, or quantitative effect "
    "sizes. You decide when to search; if you do search, the documents you "
    "retrieve are automatically attached to your response as citations and "
    "your reasoning MUST reference them. Domain coverage and the per-query "
    "result count are configurable via environment variables — see module "
    "docstring."
)


class ToolInput(BaseModel):
    query: str = Field(
        description=(
            "Focused medical search query, including the condition, value, "
            "threshold, or guideline body when relevant."
        )
    )
    max_results: int = Field(
        default=DEFAULT_MAX_RESULTS,
        ge=1,
        le=5,
        description="Maximum number of web results to return.",
    )


def medical_web_max_results(requested: int | None = None) -> int:
    """Resolve the effective max_results, honoring env overrides.

    Precedence: function arg → MARGE_WEB_RAG_MAX_RESULTS →
    MEDICAL_WEB_SEARCH_MAX_RESULTS → DEFAULT_MAX_RESULTS. Always clamped to
    [1, 5] regardless of source.
    """
    raw = os.getenv("MARGE_WEB_RAG_MAX_RESULTS") or os.getenv(
        "MEDICAL_WEB_SEARCH_MAX_RESULTS"
    )
    try:
        configured = int(raw) if raw else DEFAULT_MAX_RESULTS
    except (TypeError, ValueError):
        configured = DEFAULT_MAX_RESULTS

    configured = max(1, min(5, configured))
    if requested is None:
        return configured
    return max(1, min(configured, requested))


def _medical_web_include_domains() -> list[str]:
    """Resolve the include-domain whitelist from env, falling back to default."""
    raw = os.getenv("MEDICAL_WEB_SEARCH_INCLUDE_DOMAINS")
    if not raw:
        return list(DEFAULT_INCLUDE_DOMAINS)

    domains = [part.strip() for part in raw.replace(";", ",").split(",")]
    return [domain for domain in domains if domain]


async def search_medical_web(query: str, max_results: int = 3) -> dict[str, Any]:
    """Run a Tavily-backed web search and return normalized retrieval docs.

    Missing optional setup is returned as a warning instead of raising so the
    expert consultation can still complete while the trace clearly shows that
    web search was attempted but not configured.
    """

    api_key = os.getenv("TAVILY_API_KEY") or os.getenv("MEDICAL_WEB_SEARCH_API_KEY")
    if not api_key:
        return {
            "query": query,
            "documents": [],
            "warning": "TAVILY_API_KEY is not set; live web search was not executed.",
        }

    try:
        from tavily import TavilyClient
    except ImportError:
        return {
            "query": query,
            "documents": [],
            "warning": (
                "tavily-python is not installed. Install the medical-kb extra "
                "to enable live web search."
            ),
        }

    effective_max_results = medical_web_max_results(max_results)
    include_domains = _medical_web_include_domains()

    client = TavilyClient(api_key=api_key)
    raw = client.search(
        query=query,
        max_results=effective_max_results,
        search_depth="basic",
        include_answer=False,
        include_domains=include_domains,
    )

    documents: list[dict[str, Any]] = []
    for idx, item in enumerate(
        raw.get("results", [])[:effective_max_results], start=1
    ):
        score = item.get("score")
        try:
            score_value = float(score) if score is not None else float(idx)
        except (TypeError, ValueError):
            score_value = float(idx)

        doc = RetrievedDocument(
            title=str(item.get("title") or "Untitled medical web result"),
            snippet=str(item.get("content") or item.get("snippet") or ""),
            source_url=item.get("url"),
            retrieval_source="web",
            score=score_value,
        )
        documents.append(doc.model_dump(mode="json"))

    return {"query": query, "documents": documents, "warning": None}


def search_web(query: str) -> list:
    """Sync alias used by unit tests (calls search_medical_web via asyncio)."""
    import asyncio
    try:
        result = asyncio.run(search_medical_web(query))
        return result.get("documents", [])
    except Exception:
        return []
