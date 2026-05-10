"""Integration: assemble the BeeAI RequirementAgent via the orchestrator_agent
context manager.

3-agent architecture tests:
  - Chat Agent has patient MCP tools + local tools (no predict_* directly)
  - consult_ml_orchestrator present in local tools when llm is provided
  - consult_ml_orchestrator absent from MCP-discovered tools (it's local)
  - predict_* tools are NOT directly on the Chat Agent (those are ML Orchestrator's)

We pass a fake ChatModel (no real LLM call) — these tests only verify that
the agent has the expected tool surface and configuration. Live LLM calls
happen in scripts/orchestrator_smoke.py.
"""

import pytest

from apps.orchestrator.agent import build_bundle, orchestrator_agent


class _FakeChatModel:
    """Stand-in for BeeAI ChatModel — never invoked, just stored on the agent."""

    model_id = "fake-model"
    provider_id = "fake"
    parameters = None
    tool_choice_support = ["auto", "single"]
    middlewares = ()


# ---------------------------------------------------------------------------
# Basic shape tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_yields_requirement_agent():
    from beeai_framework.agents.requirement import RequirementAgent

    bundle = build_bundle()
    async with orchestrator_agent(bundle=bundle, llm=_FakeChatModel()) as agent:
        assert isinstance(agent, RequirementAgent)


@pytest.mark.asyncio
async def test_agent_has_base_local_tools():
    """All 4 base local tools must be on the Chat Agent."""
    bundle = build_bundle()
    async with orchestrator_agent(bundle=bundle, llm=_FakeChatModel()) as agent:
        tool_names = {t.name for t in agent._tools}
    expected_local = {
        "consult_medical_expert",
        "request_more_info",
        "clinical_report",
        "abstain",
    }
    assert expected_local <= tool_names


@pytest.mark.asyncio
async def test_agent_has_patient_tools_when_db_provided(tmp_path):
    from services.patient_data_mcp_server.sources.csv_ingest import seed_demo_db

    db = tmp_path / "test.db"
    seed_demo_db(db)
    bundle = build_bundle()
    async with orchestrator_agent(bundle=bundle, llm=_FakeChatModel(), patient_db_path=db) as agent:
        tool_names = {t.name for t in agent._tools}
    assert {"list_patients", "get_patient", "update_patient"} <= tool_names


# ---------------------------------------------------------------------------
# 3-agent architecture: Chat Agent must NOT have predict_* tools directly
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chat_agent_has_no_predict_tools_without_llm():
    """Without llm, predict_* tools must NOT appear on the Chat Agent.
    They belong exclusively to the ML Orchestrator sub-agent."""
    bundle = build_bundle()  # no llm
    async with orchestrator_agent(bundle=bundle, llm=_FakeChatModel()) as agent:
        tool_names = {t.name for t in agent._tools}
    predict_tools = {n for n in tool_names if n.startswith("predict_")}
    assert predict_tools == set(), (
        f"Chat Agent should not expose predict_* tools directly; found: {predict_tools}"
    )


@pytest.mark.asyncio
async def test_chat_agent_has_no_predict_tools_with_llm():
    """Even with llm provided, the Chat Agent must not have predict_* tools."""
    bundle = build_bundle(llm=_FakeChatModel())
    async with orchestrator_agent(bundle=bundle, llm=_FakeChatModel()) as agent:
        tool_names = {t.name for t in agent._tools}
    predict_tools = {n for n in tool_names if n.startswith("predict_")}
    assert predict_tools == set(), (
        f"Chat Agent should not expose predict_* tools directly; found: {predict_tools}"
    )


# ---------------------------------------------------------------------------
# consult_ml_orchestrator on Chat Agent when llm is provided
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chat_agent_has_consult_ml_orchestrator_when_llm_provided():
    """When build_bundle(llm=...) is used, consult_ml_orchestrator appears in tools."""
    bundle = build_bundle(llm=_FakeChatModel())
    async with orchestrator_agent(bundle=bundle, llm=_FakeChatModel()) as agent:
        tool_names = {t.name for t in agent._tools}
    assert "consult_ml_orchestrator" in tool_names


@pytest.mark.asyncio
async def test_chat_agent_does_not_have_consult_ml_orchestrator_without_llm():
    """Without llm, consult_ml_orchestrator must be absent (cannot call sub-agent)."""
    bundle = build_bundle()  # no llm
    async with orchestrator_agent(bundle=bundle, llm=_FakeChatModel()) as agent:
        tool_names = {t.name for t in agent._tools}
    assert "consult_ml_orchestrator" not in tool_names


# ---------------------------------------------------------------------------
# Tool counts
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agent_has_four_tools_without_patient_db_or_llm():
    """Without patient_db_path and without llm: exactly 4 local tools."""
    bundle = build_bundle()  # no llm
    async with orchestrator_agent(bundle=bundle, llm=_FakeChatModel()) as agent:
        # 4 local tools, no MCP ML tools, no patient tools
        assert len(agent._tools) == 4


@pytest.mark.asyncio
async def test_agent_has_five_tools_with_llm_but_no_patient_db():
    """With llm but no patient DB: 4 base local + consult_ml_orchestrator = 5."""
    bundle = build_bundle(llm=_FakeChatModel())
    async with orchestrator_agent(bundle=bundle, llm=_FakeChatModel()) as agent:
        assert len(agent._tools) == 5


# ---------------------------------------------------------------------------
# Protocol and meta tests (carried over from old architecture)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agent_has_marge_protocol_requirement():
    from apps.orchestrator.requirements.marge_protocol import MARGEProtocolRequirement

    bundle = build_bundle()
    async with orchestrator_agent(bundle=bundle, llm=_FakeChatModel()) as agent:
        types = [type(r).__name__ for r in agent._requirements]
        assert "MARGEProtocolRequirement" in types


@pytest.mark.asyncio
async def test_agent_uses_provided_llm():
    fake = _FakeChatModel()
    bundle = build_bundle()
    async with orchestrator_agent(bundle=bundle, llm=fake) as agent:
        assert agent._llm is fake


@pytest.mark.asyncio
async def test_final_answer_as_tool_disabled():
    """BeeAI's auto-final-answer tool must be off."""
    bundle = build_bundle()
    async with orchestrator_agent(bundle=bundle, llm=_FakeChatModel()) as agent:
        tool_names = {t.name for t in agent._tools}
        assert "final_answer" not in tool_names


@pytest.mark.asyncio
async def test_system_prompt_contains_ml_catalog_content():
    """System prompt should carry ML catalog content (no raw placeholder)."""
    bundle = build_bundle()
    async with orchestrator_agent(bundle=bundle, llm=_FakeChatModel()) as agent:
        assert "predict_diabetes_risk" in bundle.system_prompt
        assert "predict_breast_cancer_malignancy" in bundle.system_prompt
        assert "{ML_CATALOG}" not in bundle.system_prompt
        assert agent.meta.name == "MARGE Orchestrator"


@pytest.mark.asyncio
async def test_mcp_tool_invocations_not_recorded_in_chat_agent_enforcer():
    """In the 3-agent architecture, predict_* tools are NOT on the Chat Agent.
    So calling them does not flow through the Chat Agent's enforcer directly.
    The enforcer is only for the Chat Agent's own local tools."""
    bundle = build_bundle()
    async with orchestrator_agent(bundle=bundle, llm=_FakeChatModel()) as agent:
        tool_names = {t.name for t in agent._tools}
    # Confirm no predict_* are present
    assert not any(n.startswith("predict_") for n in tool_names)
