"""Unit tests for apps/orchestrator/tools/consult_ml_orchestrator.py.

Tests:
1. TOOL_NAME constant is "consult_ml_orchestrator"
2. ToolInput schema validates correctly (required field, optional field)
3. make_consult_ml_orchestrator returns an async callable
4. Callable records TOOL_NAME in the enforcer
5. Callable appends patient_features to prompt as JSON when provided
6. Callable handles None patient_features gracefully (prompt = bare request)
7. Callable returns a string (the result text)

The ML Orchestrator sub-agent is fully mocked — no real LLM or MCP calls.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from apps.orchestrator.middleware.enforce_protocol import ProtocolEnforcer
from apps.orchestrator.tools.consult_ml_orchestrator import (
    TOOL_NAME,
    ToolInput,
    make_consult_ml_orchestrator,
)


# ---------------------------------------------------------------------------
# Constant and schema tests
# ---------------------------------------------------------------------------


class TestToolNameConstant:
    def test_tool_name_is_correct_string(self):
        assert TOOL_NAME == "consult_ml_orchestrator"

    def test_tool_name_is_a_string(self):
        assert isinstance(TOOL_NAME, str)


class TestToolInputSchema:
    def test_accepts_request_only(self):
        inp = ToolInput(request="predict diabetes risk")
        assert inp.request == "predict diabetes risk"
        assert inp.patient_features is None

    def test_accepts_request_and_patient_features(self):
        features = {"preg": 6, "plas": 148, "age": 50}
        inp = ToolInput(request="predict", patient_features=features)
        assert inp.patient_features == features

    def test_request_is_required(self):
        with pytest.raises(ValidationError):
            ToolInput()  # type: ignore[call-arg]

    def test_patient_features_defaults_to_none(self):
        inp = ToolInput(request="some request")
        assert inp.patient_features is None

    def test_patient_features_accepts_nested_dict(self):
        features = {"meta": {"key": "value"}, "age": 30}
        inp = ToolInput(request="r", patient_features=features)
        assert inp.patient_features["meta"]["key"] == "value"

    def test_patient_features_accepts_none_explicitly(self):
        inp = ToolInput(request="r", patient_features=None)
        assert inp.patient_features is None


# ---------------------------------------------------------------------------
# make_consult_ml_orchestrator factory tests
# ---------------------------------------------------------------------------


def _make_fake_agent(return_text: str = "ML result text") -> Any:
    """Return a mock agent whose run() returns a result with answer.text."""
    result = SimpleNamespace(
        output_structured=None,
        answer=SimpleNamespace(text=return_text),
    )
    fake_agent = MagicMock()
    fake_agent.run = AsyncMock(return_value=result)
    return fake_agent


class _FakeLLM:
    """Minimal stand-in for a ChatModel. Never actually called."""
    model_id = "fake"


class TestMakeConsultMlOrchestrator:
    def test_returns_callable(self):
        enforcer = ProtocolEnforcer()
        fn = make_consult_ml_orchestrator(llm=_FakeLLM(), enforcer=enforcer)
        assert callable(fn)

    def test_returns_async_callable(self):
        """The returned function must be a coroutine function."""
        import inspect

        enforcer = ProtocolEnforcer()
        fn = make_consult_ml_orchestrator(llm=_FakeLLM(), enforcer=enforcer)
        assert inspect.iscoroutinefunction(fn)

    @pytest.mark.asyncio
    async def test_records_tool_name_in_enforcer(self):
        """Calling the tool must record TOOL_NAME in the enforcer."""
        enforcer = ProtocolEnforcer()
        fn = make_consult_ml_orchestrator(llm=_FakeLLM(), enforcer=enforcer)

        fake_agent = _make_fake_agent("diabetes: low risk")
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=fake_agent)
        ctx.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "apps.orchestrator.tools.consult_ml_orchestrator.ml_orchestrator_agent",
            return_value=ctx,
        ):
            await fn(request="Predict diabetes risk")

        assert enforcer.has_called(TOOL_NAME)

    @pytest.mark.asyncio
    async def test_prompt_contains_request_text(self):
        """The prompt sent to the sub-agent must include the request string."""
        enforcer = ProtocolEnforcer()
        fn = make_consult_ml_orchestrator(llm=_FakeLLM(), enforcer=enforcer)

        fake_agent = _make_fake_agent("result")
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=fake_agent)
        ctx.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "apps.orchestrator.tools.consult_ml_orchestrator.ml_orchestrator_agent",
            return_value=ctx,
        ):
            await fn(request="Predict breast cancer malignancy")

        call_args = fake_agent.run.call_args
        prompt_sent = call_args[0][0]  # first positional arg
        assert "Predict breast cancer malignancy" in prompt_sent

    @pytest.mark.asyncio
    async def test_prompt_appends_patient_features_as_json(self):
        """When patient_features is provided it must be appended as JSON."""
        enforcer = ProtocolEnforcer()
        fn = make_consult_ml_orchestrator(llm=_FakeLLM(), enforcer=enforcer)

        features = {"preg": 6, "plas": 148, "age": 50}
        fake_agent = _make_fake_agent("result")
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=fake_agent)
        ctx.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "apps.orchestrator.tools.consult_ml_orchestrator.ml_orchestrator_agent",
            return_value=ctx,
        ):
            await fn(request="run prediction", patient_features=features)

        prompt_sent = fake_agent.run.call_args[0][0]
        # The JSON of features must appear somewhere in the prompt
        assert json.dumps(features, default=str) in prompt_sent

    @pytest.mark.asyncio
    async def test_prompt_without_patient_features_has_no_json_suffix(self):
        """When patient_features is None, no JSON block is appended."""
        enforcer = ProtocolEnforcer()
        fn = make_consult_ml_orchestrator(llm=_FakeLLM(), enforcer=enforcer)

        fake_agent = _make_fake_agent("result")
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=fake_agent)
        ctx.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "apps.orchestrator.tools.consult_ml_orchestrator.ml_orchestrator_agent",
            return_value=ctx,
        ):
            await fn(request="list available models", patient_features=None)

        prompt_sent = fake_agent.run.call_args[0][0]
        assert "Patient features:" not in prompt_sent
        assert "{" not in prompt_sent

    @pytest.mark.asyncio
    async def test_returns_ml_orchestrator_response(self):
        """Tool now returns MLOrchestratorResponse (Pydantic, not str).

        The legacy fallback path (when ml_orchestrator instance is not
        provided, only `llm`) wraps the agent's text into the same
        response shape with `needed_features=None`.
        """
        from packages.schemas.ml import MLOrchestratorResponse

        enforcer = ProtocolEnforcer()
        fn = make_consult_ml_orchestrator(llm=_FakeLLM(), enforcer=enforcer)

        fake_agent = _make_fake_agent("Diabetes risk: 0.72 (high risk)")
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=fake_agent)
        ctx.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "apps.orchestrator.tools.consult_ml_orchestrator.ml_orchestrator_agent",
            return_value=ctx,
        ):
            result = await fn(request="run prediction", patient_features={"age": 50})

        assert isinstance(result, MLOrchestratorResponse)
        assert "Diabetes risk: 0.72" in result.reasoning
        # Legacy fallback path never populates needed_features (no Phase 2).
        assert result.needed_features is None

    @pytest.mark.asyncio
    async def test_uses_structured_response_when_available(self):
        """Prefer output_structured.response over answer.text in the
        wrapped MLOrchestratorResponse.reasoning."""
        from packages.schemas.ml import MLOrchestratorResponse

        enforcer = ProtocolEnforcer()
        fn = make_consult_ml_orchestrator(llm=_FakeLLM(), enforcer=enforcer)

        beeai_result = SimpleNamespace(
            output_structured=SimpleNamespace(response="structured ML text"),
            answer=SimpleNamespace(text="fallback text"),
        )
        fake_agent = MagicMock()
        fake_agent.run = AsyncMock(return_value=beeai_result)

        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=fake_agent)
        ctx.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "apps.orchestrator.tools.consult_ml_orchestrator.ml_orchestrator_agent",
            return_value=ctx,
        ):
            result = await fn(request="predict")

        assert isinstance(result, MLOrchestratorResponse)
        assert result.reasoning == "structured ML text"

    @pytest.mark.asyncio
    async def test_enforcer_not_yet_has_tool_before_call(self):
        """Enforcer state starts clean before any tool call."""
        enforcer = ProtocolEnforcer()
        assert not enforcer.has_called(TOOL_NAME)

    @pytest.mark.asyncio
    async def test_multiple_calls_all_recorded(self):
        """Each call records the tool name (trajectory grows)."""
        enforcer = ProtocolEnforcer()
        fn = make_consult_ml_orchestrator(llm=_FakeLLM(), enforcer=enforcer)

        fake_agent = _make_fake_agent("result")
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=fake_agent)
        ctx.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "apps.orchestrator.tools.consult_ml_orchestrator.ml_orchestrator_agent",
            return_value=ctx,
        ):
            await fn(request="call one")
            await fn(request="call two")

        assert enforcer.trajectory.count(TOOL_NAME) == 2
