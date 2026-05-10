"""Unit tests for services/ml_orchestrator_agent/agent.py.

Tests:
1. _result_text — extracts text from BeeAI result in priority order:
   output_structured.response  > answer.text  > str(result)
2. ml_orchestrator_agent — context-manager shape and tool surface

All tests are unit-level: no real LLM is invoked.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers — lightweight fake result objects
# ---------------------------------------------------------------------------


def _make_result(
    *,
    structured_response: str | None = None,
    answer_text: str | None = None,
    fallback_str: str | None = None,
) -> Any:
    """Build a fake BeeAI-style result object."""
    obj = SimpleNamespace()

    if structured_response is not None:
        obj.output_structured = SimpleNamespace(response=structured_response)
    else:
        obj.output_structured = None

    if answer_text is not None:
        obj.answer = SimpleNamespace(text=answer_text)
    else:
        obj.answer = None

    if fallback_str is not None:
        obj.__str__ = lambda self: fallback_str

    return obj


# ---------------------------------------------------------------------------
# Tests for _result_text
# ---------------------------------------------------------------------------


class TestResultText:
    """Unit tests for the _result_text helper."""

    def test_prefers_output_structured_response(self):
        """output_structured.response is the highest-priority extraction path."""
        from services.ml_orchestrator_agent.agent import _result_text

        result = _make_result(
            structured_response="ML structured answer",
            answer_text="fallback answer text",
        )
        assert _result_text(result) == "ML structured answer"

    def test_falls_back_to_answer_text_when_no_structured(self):
        """Falls back to answer.text when output_structured is absent or None."""
        from services.ml_orchestrator_agent.agent import _result_text

        result = _make_result(answer_text="answer text fallback")
        assert _result_text(result) == "answer text fallback"

    def test_falls_back_to_str_when_both_absent(self):
        """Falls back to str(result) when neither path yields text."""
        from services.ml_orchestrator_agent.agent import _result_text

        sentinel = object()
        # str(sentinel) will be something like "<object object at 0x...>"
        text = _result_text(sentinel)
        assert isinstance(text, str)
        assert len(text) > 0

    def test_falls_back_to_str_when_structured_response_is_none(self):
        """output_structured exists but response is None — must fall through."""
        from services.ml_orchestrator_agent.agent import _result_text

        result = SimpleNamespace(
            output_structured=SimpleNamespace(response=None),
            answer=SimpleNamespace(text="answer fallback"),
        )
        assert _result_text(result) == "answer fallback"

    def test_falls_back_to_str_when_answer_text_is_none(self):
        """Both paths present but both are None — falls to str()."""
        from services.ml_orchestrator_agent.agent import _result_text

        result = SimpleNamespace(
            output_structured=SimpleNamespace(response=None),
            answer=SimpleNamespace(text=None),
        )
        text = _result_text(result)
        assert isinstance(text, str)

    def test_falls_back_to_str_when_structured_response_is_empty_string(self):
        """Empty string is falsy — must NOT be returned as the answer."""
        from services.ml_orchestrator_agent.agent import _result_text

        result = SimpleNamespace(
            output_structured=SimpleNamespace(response=""),
            answer=SimpleNamespace(text="non-empty answer"),
        )
        # empty string is falsy, so we expect the answer.text fallback
        assert _result_text(result) == "non-empty answer"

    def test_handles_missing_output_structured_attribute(self):
        """Objects without output_structured at all must not raise."""
        from services.ml_orchestrator_agent.agent import _result_text

        result = SimpleNamespace(answer=SimpleNamespace(text="from answer"))
        assert _result_text(result) == "from answer"

    def test_handles_missing_answer_attribute(self):
        """Objects without answer at all must not raise."""
        from services.ml_orchestrator_agent.agent import _result_text

        result = SimpleNamespace(output_structured=SimpleNamespace(response="from structured"))
        assert _result_text(result) == "from structured"

    def test_handles_none_input(self):
        """None is a valid edge-case — str(None) == 'None'."""
        from services.ml_orchestrator_agent.agent import _result_text

        assert _result_text(None) == "None"

    def test_handles_plain_string_input(self):
        """A plain str passed as result is returned as-is (str(str) = str)."""
        from services.ml_orchestrator_agent.agent import _result_text

        assert _result_text("direct string") == "direct string"

    def test_returns_str_type(self):
        """Return type must always be str."""
        from services.ml_orchestrator_agent.agent import _result_text

        for val in (
            _make_result(structured_response="ok"),
            _make_result(answer_text="ok"),
            "ok",
            None,
            42,
        ):
            assert isinstance(_result_text(val), str)


# ---------------------------------------------------------------------------
# Tests for ml_orchestrator_agent context-manager shape
# ---------------------------------------------------------------------------


class TestMlOrchestratorAgentContextManager:
    """Verify ml_orchestrator_agent yields a RequirementAgent-like object
    with the expected tool surface (predict_* tools from ML MCP server).

    We use real in-process FastMCP server (same as integration tests) but
    patch out only the LLM so no API calls are made.
    """

    @pytest.mark.asyncio
    async def test_yields_something(self):
        """Context manager must yield a non-None object."""
        from services.ml_orchestrator_agent.agent import ml_orchestrator_agent

        class FakeLLM:
            model_id = "fake"
            provider_id = "fake"
            parameters = None
            tool_choice_support = ["auto"]
            middlewares = ()

        async with ml_orchestrator_agent(llm=FakeLLM()) as agent:
            assert agent is not None

    @pytest.mark.asyncio
    async def test_yields_requirement_agent(self):
        """Yielded object must be a BeeAI RequirementAgent."""
        from beeai_framework.agents.requirement import RequirementAgent

        from services.ml_orchestrator_agent.agent import ml_orchestrator_agent

        class FakeLLM:
            model_id = "fake"
            provider_id = "fake"
            parameters = None
            tool_choice_support = ["auto"]
            middlewares = ()

        async with ml_orchestrator_agent(llm=FakeLLM()) as agent:
            assert isinstance(agent, RequirementAgent)

    @pytest.mark.asyncio
    async def test_agent_has_predict_diabetes_risk_tool(self):
        """The ML MCP server must expose predict_diabetes_risk."""
        from services.ml_orchestrator_agent.agent import ml_orchestrator_agent

        class FakeLLM:
            model_id = "fake"
            provider_id = "fake"
            parameters = None
            tool_choice_support = ["auto"]
            middlewares = ()

        async with ml_orchestrator_agent(llm=FakeLLM()) as agent:
            tool_names = {t.name for t in agent._tools}
        assert "predict_diabetes_risk" in tool_names

    @pytest.mark.asyncio
    async def test_agent_has_predict_breast_cancer_tool(self):
        """The ML MCP server must expose predict_breast_cancer_malignancy."""
        from services.ml_orchestrator_agent.agent import ml_orchestrator_agent

        class FakeLLM:
            model_id = "fake"
            provider_id = "fake"
            parameters = None
            tool_choice_support = ["auto"]
            middlewares = ()

        async with ml_orchestrator_agent(llm=FakeLLM()) as agent:
            tool_names = {t.name for t in agent._tools}
        assert "predict_breast_cancer_malignancy" in tool_names

    @pytest.mark.asyncio
    async def test_agent_has_no_patient_tools(self):
        """ML Orchestrator must NOT have patient_data MCP tools."""
        from services.ml_orchestrator_agent.agent import ml_orchestrator_agent

        class FakeLLM:
            model_id = "fake"
            provider_id = "fake"
            parameters = None
            tool_choice_support = ["auto"]
            middlewares = ()

        patient_tool_names = {"get_patient", "update_patient", "list_patients"}

        async with ml_orchestrator_agent(llm=FakeLLM()) as agent:
            tool_names = {t.name for t in agent._tools}

        assert tool_names.isdisjoint(patient_tool_names), (
            f"ML Orchestrator should not have patient tools but found: "
            f"{tool_names & patient_tool_names}"
        )

    @pytest.mark.asyncio
    async def test_agent_has_no_local_consult_tools(self):
        """ML Orchestrator must NOT have consult_medical_expert or other Chat Agent tools."""
        from services.ml_orchestrator_agent.agent import ml_orchestrator_agent

        class FakeLLM:
            model_id = "fake"
            provider_id = "fake"
            parameters = None
            tool_choice_support = ["auto"]
            middlewares = ()

        chat_agent_tools = {
            "consult_medical_expert",
            "consult_ml_orchestrator",
            "clinical_report",
            "abstain",
            "request_more_info",
        }

        async with ml_orchestrator_agent(llm=FakeLLM()) as agent:
            tool_names = {t.name for t in agent._tools}

        assert tool_names.isdisjoint(chat_agent_tools), (
            f"ML Orchestrator should not have Chat Agent tools but found: "
            f"{tool_names & chat_agent_tools}"
        )

    @pytest.mark.asyncio
    async def test_agent_name_is_ml_orchestrator(self):
        """Agent name must be 'ML Orchestrator'."""
        from services.ml_orchestrator_agent.agent import ml_orchestrator_agent

        class FakeLLM:
            model_id = "fake"
            provider_id = "fake"
            parameters = None
            tool_choice_support = ["auto"]
            middlewares = ()

        async with ml_orchestrator_agent(llm=FakeLLM()) as agent:
            assert agent.meta.name == "ML Orchestrator"

    @pytest.mark.asyncio
    async def test_agent_final_answer_as_tool_disabled(self):
        """BeeAI's auto-final-answer tool must be off."""
        from services.ml_orchestrator_agent.agent import ml_orchestrator_agent

        class FakeLLM:
            model_id = "fake"
            provider_id = "fake"
            parameters = None
            tool_choice_support = ["auto"]
            middlewares = ()

        async with ml_orchestrator_agent(llm=FakeLLM()) as agent:
            tool_names = {t.name for t in agent._tools}
        assert "final_answer" not in tool_names
