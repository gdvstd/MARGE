"""BeeAI Chat Agent assembly (3-agent architecture).

Two layers:

`build_bundle(expert=None, llm=None)` returns deterministic dependencies
(enforcer, local tools, system prompt with ML catalog injected). No DB
dependency — tests reach for this directly.

`orchestrator_agent(bundle, llm, patient_db_path, memory)` is an async
context manager:
- Opens an in-process MCP client for the patient-data server (backed by
  the session SQLite DB).
- Discovers patient tools from the patient MCP session.
- Wires the MARGE protocol Requirement (final_answer gating + ordering).
- Yields a fully wired `RequirementAgent` (Chat Agent).
- Closes the MCP session on exit.

3-agent boundary (CRITICAL):
- The Chat Agent does NOT hold predict_* tools directly.
- ML predictions are delegated to the ML Orchestrator sub-agent via the
  `consult_ml_orchestrator` local tool (only wired when `llm` is passed
  to `build_bundle`).
- Patient data tools come from the patient-data MCP server.

Memory: the caller may pass a `BaseMemory` instance to persist conversation
history across user turns (Streamlit reuses one per session). Defaults to
fresh `UnconstrainedMemory` when omitted.

Usage:
    bundle = build_bundle(llm=llm)
    async with orchestrator_agent(bundle, llm, patient_db_path=db,
                                   memory=session_memory) as agent:
        result = await agent.run("Analyse patient csv-42.")
"""

from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, AsyncIterator

from apps.orchestrator.middleware.enforce_protocol import ProtocolEnforcer
from apps.orchestrator.requirements.marge_protocol import (
    build_marge_protocol_requirement,
)
from apps.orchestrator.tools.abstain import make_abstain
from apps.orchestrator.tools.clinical_report import make_clinical_report
from apps.orchestrator.tools.consult_expert import make_consult_expert
from apps.orchestrator.tools.request_more_info import make_request_more_info
from services.medical_expert_agent.agent import StubMedicalExpert

if TYPE_CHECKING:
    from beeai_framework.agents.requirement import RequirementAgent
    from beeai_framework.backend.chat import ChatModel
    from beeai_framework.memory.base_memory import BaseMemory

_SYSTEM_PROMPT_PATH = Path(__file__).parent / "system_prompt.md"


@dataclass
class OrchestratorBundle:
    """Deterministic dependencies of the Chat Agent (no DB)."""

    enforcer: ProtocolEnforcer
    system_prompt: str
    local_tools: dict[str, object]


def _build_ml_catalog() -> str:
    """Discover registered ML models and return a formatted catalog string.

    The catalog is injected into the system prompt so the Chat Agent knows
    which conditions it can request predictions for and what features each
    model needs. The ML Orchestrator owns the actual predict_* tools; the
    Chat Agent only sees this textual description.
    """
    from services.ml_mcp_server.registry import discover_models

    models = discover_models()
    if not models:
        return "No ML models are currently registered."

    lines: list[str] = []
    for model in models:
        # Feature names live on model.config for DynamicMLAgent subclasses;
        # fall back gracefully for any other MLModel implementation.
        feature_names: list[str] = getattr(
            getattr(model, "config", None), "feature_names", []
        ) or []
        feature_list = ", ".join(feature_names) if feature_names else "see model documentation"
        lines.append(
            f"- **{model.name}** — {model.metadata.description}\n"
            f"  Features: {feature_list}"
        )
    return "\n".join(lines)


def build_bundle(expert: Any = None, llm: Any = None) -> OrchestratorBundle:
    """Build the Chat Agent's deterministic dependencies.

    Args:
        expert: Optional MedicalExpert implementation. Defaults to
            `StubMedicalExpert` (deterministic test stub). Production
            use should pass the live BeeAI sub-agent expert.
        llm: Optional ChatModel. When provided, `consult_ml_orchestrator`
            is added to local_tools, enabling the Chat Agent to delegate
            ML predictions to the ML Orchestrator sub-agent.
    """
    enforcer = ProtocolEnforcer()
    if expert is None:
        expert = StubMedicalExpert()

    local_tools: dict[str, Any] = {
        "consult_medical_expert": make_consult_expert(expert, enforcer),
        "request_more_info": make_request_more_info(enforcer),
        "clinical_report": make_clinical_report(enforcer),
        "abstain": make_abstain(enforcer),
    }

    if llm is not None:
        from apps.orchestrator.tools.consult_ml_orchestrator import (
            make_consult_ml_orchestrator,
        )
        local_tools["consult_ml_orchestrator"] = make_consult_ml_orchestrator(
            llm=llm, enforcer=enforcer
        )

    ml_catalog = _build_ml_catalog()
    raw_prompt = _SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    system_prompt = raw_prompt.replace("{ML_CATALOG}", ml_catalog)

    return OrchestratorBundle(
        enforcer=enforcer,
        system_prompt=system_prompt,
        local_tools=local_tools,
    )


@asynccontextmanager
async def orchestrator_agent(
    bundle: OrchestratorBundle,
    llm: "ChatModel",
    patient_db_path: Path | None = None,
    memory: "BaseMemory | None" = None,
) -> AsyncIterator["RequirementAgent"]:
    """Build and yield a fully wired Chat Agent (RequirementAgent).

    The Chat Agent does NOT open the ML MCP server directly. ML predictions
    are delegated to the ML Orchestrator sub-agent via the
    `consult_ml_orchestrator` local tool.

    Opens an in-process MCP session for the patient-data server when
    `patient_db_path` is provided. The session stays alive for the duration
    of agent.run() — closing it early causes MCPTool to raise ToolError.

    Args:
        bundle: Deterministic dependencies from `build_bundle()`.
        llm: Chat model instance.
        patient_db_path: Path to the session SQLite DB. If None, the patient
            MCP server is not attached.
        memory: Optional `BaseMemory` to persist conversation across turns.
            Defaults to a fresh `UnconstrainedMemory`.
    """
    from beeai_framework.agents.requirement import RequirementAgent
    from beeai_framework.memory import UnconstrainedMemory
    from beeai_framework.tools.mcp import MCPTool
    from fastmcp import Client

    from apps.orchestrator.tools._adapter import local_tools_as_beeai

    local_tools = local_tools_as_beeai(bundle)

    if memory is None:
        memory = UnconstrainedMemory()

    def _make_recorder(tool_name: str):
        def _record(data: Any, event: Any) -> None:
            if getattr(event, "name", None) == "start":
                bundle.enforcer.record(tool_name)
        return _record

    if patient_db_path is not None:
        from services.patient_data_mcp_server.server import build_patient_server

        patient_server = build_patient_server(patient_db_path)
        async with Client(patient_server) as patient_client:
            patient_tools = await MCPTool.from_client(patient_client.session)
            for t in patient_tools:
                t.emitter.match("*", _make_recorder(t.name))

            agent = RequirementAgent(
                llm=llm,
                memory=memory,
                tools=[*local_tools, *patient_tools],
                requirements=[build_marge_protocol_requirement()],
                name="MARGE Orchestrator",
                description=(
                    "Clinical ML head researcher: coordinates with the ML Orchestrator "
                    "sub-agent, manages patient data, and consults a medical expert."
                ),
                instructions=bundle.system_prompt,
                final_answer_as_tool=False,
            )
            yield agent
    else:
        agent = RequirementAgent(
            llm=llm,
            memory=memory,
            tools=local_tools,
            requirements=[build_marge_protocol_requirement()],
            name="MARGE Orchestrator",
            description=(
                "Clinical ML head researcher: coordinates with the ML Orchestrator "
                "sub-agent and consults a medical expert."
            ),
            instructions=bundle.system_prompt,
            final_answer_as_tool=False,
        )
        yield agent
