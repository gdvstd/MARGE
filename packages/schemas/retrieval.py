"""Schemas for the retrieval/search layer.

Used by:
- The medical expert agent's search tools (`search_local_kb`, `search_web`)
- The deep_medical_research subagent (returns `ResearchReport`)
- The `enforce_citation` middleware on the medical expert agent

The `MedicalExpertResponse` shape is the structural enforcement of the
"no medical claim without a source" rule from architecture.md §3.4.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


RetrievalSource = Literal["local_kb", "web", "deep_research"]


class RetrievedDocument(BaseModel):
    """A single search result returned by any retriever.

    Same shape across local RAG and web search so downstream code does not
    need to branch on source — but `retrieval_source` is preserved for
    citation provenance and source weighting.
    """

    title: str
    snippet: str
    source_url: str | None = None
    retrieval_source: RetrievalSource
    retrieved_at: datetime = Field(default_factory=datetime.utcnow)
    score: float | None = Field(
        default=None, description="Retrieval score (cosine similarity for RAG, rank for web)"
    )


class Citation(BaseModel):
    """A document cited in support of a clinical claim.

    `supporting_quote` is the exact passage being cited — useful for the UI
    to render an expandable "show source" panel.
    """

    document: RetrievedDocument
    supporting_quote: str | None = None


class ResearchReport(BaseModel):
    """Output of the deep_medical_research sub-sub-agent.

    Returned to the medical expert when it spawns deep research.
    """

    question: str
    findings: str
    citations: list[Citation]


class MedicalExpertResponse(BaseModel):
    """The Medical Expert's response shape.

    The Expert always answers — there is no `abstained` field. When the
    question is uncertain or under-specified, the Expert hedges in
    `reasoning` ("differential includes A and B; needs lab X to narrow")
    rather than declining. The decision to abstain from the user's
    overall request is the Chat Agent's responsibility, not the Expert's.

    `citations` is auto-populated by the agent harness from the Expert's
    `search_medical_web` tool calls. If the Expert did not call the
    search tool, `citations` is empty and `reasoning` should hedge
    accordingly rather than make claims that require a primary source.
    """

    reasoning: str
    citations: list[Citation]
