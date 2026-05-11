"""Tests for the JSON-tail parser in services/ml_orchestrator_agent/agent.py.

`_split_reasoning_and_needed_features` extracts a trailing fenced JSON
block (the Phase 2 self-review's structured needed_features output) from
the LLM's free-form response. Robustness matters because the LLM is
allowed to omit, malform, or place the block anywhere — only a strict
trailing block should ever populate `needed_features`.
"""

from packages.schemas.ml import NeededFeature
from services.ml_orchestrator_agent.agent import (
    _split_reasoning_and_needed_features,
)


class TestSplitReasoningAndNeededFeatures:
    def test_no_json_block_returns_text_unchanged(self):
        text = "Diabetes risk is 0.92. Prediction is credible."
        reasoning, needed = _split_reasoning_and_needed_features(text)
        assert reasoning == text
        assert needed is None

    def test_well_formed_trailing_json_block_extracted(self):
        text = (
            "Diabetes risk is 0.62 — not yet credible. We need more inputs.\n"
            "\n"
            "```json\n"
            '{"needed_features":[{"name":"plas","reason":"top SHAP, missing"},'
            '{"name":"insu","reason":"second largest, missing"}]}\n'
            "```"
        )
        reasoning, needed = _split_reasoning_and_needed_features(text)
        assert "needed_features" not in reasoning  # JSON block stripped
        assert reasoning.endswith("more inputs.")
        assert needed is not None
        assert len(needed) == 2
        assert all(isinstance(n, NeededFeature) for n in needed)
        assert needed[0].name == "plas"
        assert "top SHAP" in needed[0].reason

    def test_malformed_json_left_in_text(self):
        """Bad JSON → keep block in reasoning, return needed_features=None."""
        text = (
            "Some prose.\n\n```json\n{this is not valid json}\n```"
        )
        reasoning, needed = _split_reasoning_and_needed_features(text)
        assert reasoning == text  # nothing stripped
        assert needed is None

    def test_empty_features_list_strips_block_returns_none(self):
        text = (
            "Both predictions are credible enough.\n\n"
            "```json\n"
            '{"needed_features":[]}\n'
            "```"
        )
        reasoning, needed = _split_reasoning_and_needed_features(text)
        # The block was well-formed but declared no features — strip from
        # user-facing reasoning so machine-only payload doesn't leak.
        assert "needed_features" not in reasoning
        assert needed is None

    def test_block_must_be_trailing(self):
        """A JSON block in the middle of the prose is NOT extracted."""
        text = (
            "First a JSON example: ```json\n"
            '{"needed_features":[{"name":"plas","reason":"x"}]}\n'
            "```\n\n"
            "Then more prose follows that describes the prediction normally."
        )
        reasoning, needed = _split_reasoning_and_needed_features(text)
        assert reasoning == text
        assert needed is None

    def test_unfenced_json_not_extracted(self):
        """Only fenced ```json ... ``` blocks are recognized."""
        text = (
            "Prediction summary.\n\n"
            '{"needed_features":[{"name":"plas","reason":"x"}]}'
        )
        reasoning, needed = _split_reasoning_and_needed_features(text)
        assert reasoning == text
        assert needed is None

    def test_individual_invalid_entries_skipped(self):
        text = (
            "Prose.\n\n```json\n"
            '{"needed_features":[{"name":"plas","reason":"ok"},'
            '{"BADKEY":"x"},'
            '{"name":"insu","reason":"ok2"}]}\n'
            "```"
        )
        reasoning, needed = _split_reasoning_and_needed_features(text)
        assert reasoning == "Prose."
        assert needed is not None
        # The middle entry is missing required fields → skipped, keep two.
        assert len(needed) == 2
        assert {n.name for n in needed} == {"plas", "insu"}
