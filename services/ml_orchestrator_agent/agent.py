"""ML Orchestrator Agent — professional ML researcher sub-agent.

Called by the Chat Agent's `consult_ml_orchestrator` tool. Receives a
natural-language request (which conditions to predict + patient feature
values), routes to the appropriate ML MCP tools, runs predictions, and
returns structured results with SHAP-based interpretation.

Two surfaces:

- `ml_orchestrator_agent` async context manager — fresh per-call agent
  with a fresh memory. Used by older callsites and tests.
- `MLOrchestratorAgent` class — session-persistent. Holds its own
  `ChatModel` and `UnconstrainedMemory` so that across multiple
  consultations within a session it remembers prior context. Mirrors
  `MedicalExpertAgent`: exposes `set_event_sink` so the UI can stream
  the sub-agent's own tool calls live (without exposing them to the
  Chat Agent's LLM context).
"""

import json
import re
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, AsyncIterator

from packages.schemas.ml import MLOrchestratorResponse, NeededFeature

if TYPE_CHECKING:
    from beeai_framework.backend.chat import ChatModel

_SYSTEM_PROMPT_PATH = Path(__file__).parent / "system_prompt.md"

# Phase 2 trigger — short, content-free instruction. All credibility-judgment
# logic lives in the system prompt so the workflow remains portable (the prompt
# can be lifted into a generic skill without depending on this harness).
_PHASE_2_TRIGGER = (
    "Phase 2 trigger from harness: you just produced a Phase 1 draft with at "
    "least one prediction. Perform Phase 2 self-review now exactly as defined "
    "in your role description. Replace your previous answer with a single "
    "integrated revised response that includes a credibility verdict for each "
    "prediction and, if applicable, a prioritized list of features to collect "
    "for higher confidence. Do not call additional prediction tools unless you "
    "have a specific, justified reason."
)


def _result_text(result: Any) -> str:
    structured = getattr(result, "output_structured", None)
    text = getattr(structured, "response", None)
    if text:
        return text
    answer = getattr(result, "answer", None)
    text = getattr(answer, "text", None)
    if text:
        return text
    return str(result)


# Match a trailing ```json ... ``` fenced block. Tolerates trailing whitespace
# and is anchored to the end of the response so we never accidentally pick up
# a JSON example from earlier in the prose.
_TRAILING_JSON_BLOCK = re.compile(
    r"```(?:json)?\s*(\{.*?\})\s*```\s*\Z",
    re.DOTALL | re.IGNORECASE,
)


def _split_reasoning_and_needed_features(
    text: str,
) -> tuple[str, list[NeededFeature] | None]:
    """Pull a trailing JSON block out of the Phase 2 response, if present.

    Returns (reasoning_without_json, needed_features). On any parse failure
    the JSON block is left in the reasoning unchanged and `needed_features`
    is None — Phase 2's prose still flows through to the user.
    """
    match = _TRAILING_JSON_BLOCK.search(text)
    if match is None:
        return text, None

    raw = match.group(1)
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return text, None

    if not isinstance(payload, dict):
        return text, None

    raw_features = payload.get("needed_features")
    if not isinstance(raw_features, list) or not raw_features:
        # Block is well-formed JSON but does not declare any needed features
        # (the spec says: omit the block entirely in that case). Strip it
        # anyway so users do not see machine-only output.
        cleaned = text[: match.start()].rstrip()
        return cleaned, None

    parsed: list[NeededFeature] = []
    for entry in raw_features:
        if not isinstance(entry, dict):
            continue
        try:
            parsed.append(NeededFeature.model_validate(entry))
        except Exception:
            continue

    cleaned = text[: match.start()].rstrip()
    return cleaned, parsed or None


@asynccontextmanager
async def ml_orchestrator_agent(llm: "ChatModel | None" = None) -> AsyncIterator[Any]:
    """Yield a fully wired ML Orchestrator agent backed by the ML MCP server."""
    from beeai_framework.agents.requirement import RequirementAgent
    from beeai_framework.memory import UnconstrainedMemory
    from beeai_framework.tools.mcp import MCPTool
    from fastmcp import Client

    from packages.llm_provider.client import build_chat_model_for_role
    from packages.llm_provider.settings import Role
    from services.ml_mcp_server.server import build_server

    if llm is None:
        llm = build_chat_model_for_role(Role.ML_ORCHESTRATOR)

    ml_server = build_server()
    async with Client(ml_server) as client:
        ml_tools = await MCPTool.from_client(client.session)

        agent = RequirementAgent(
            llm=llm,
            memory=UnconstrainedMemory(),
            tools=ml_tools,
            requirements=[],
            name="ML Orchestrator",
            description=(
                "Professional ML researcher: routes patient features to "
                "clinical ML models and interprets predictions with SHAP."
            ),
            instructions=_SYSTEM_PROMPT_PATH.read_text(encoding="utf-8"),
            final_answer_as_tool=False,
        )
        yield agent


# ---------------- Session-persistent agent (mirrors MedicalExpertAgent) ----------------


class MLOrchestratorAgent:
    """LLM-backed ML Orchestrator with session-persistent context.

    - Holds its own `ChatModel` and `UnconstrainedMemory`. Memory persists
      across `run()` calls within the same Python session so the agent can
      remember prior ML consultations.
    - Each `run()` opens a short-lived in-process MCP client to the ML
      server and builds a `RequirementAgent` around the persistent memory.
    - `set_event_sink` lets the UI stream this sub-agent's own tool
      events (predict_*, …) separately from the Chat Agent's trace.

    Usage:
        ml = MLOrchestratorAgent.from_env()
        text = await ml.run("Predict diabetes risk for ...")
    """

    def __init__(
        self,
        llm: "ChatModel",
        system_prompt: str | None = None,
    ) -> None:
        from beeai_framework.memory import UnconstrainedMemory

        self._llm = llm
        self._system_prompt = system_prompt or _SYSTEM_PROMPT_PATH.read_text(
            encoding="utf-8"
        )
        self._memory = UnconstrainedMemory()
        self._event_sink: Callable[[dict[str, Any]], None] | None = None

    @classmethod
    def from_env(cls) -> "MLOrchestratorAgent":
        """Build with the LLM configured for `Role.ML_ORCHESTRATOR`."""
        from packages.llm_provider.client import build_chat_model_for_role
        from packages.llm_provider.settings import Role

        return cls(llm=build_chat_model_for_role(Role.ML_ORCHESTRATOR))

    @property
    def llm(self) -> "ChatModel":
        return self._llm

    def set_event_sink(
        self, sink: Callable[[dict[str, Any]], None] | None
    ) -> None:
        """Attach a per-turn trace sink for ML-internal tool events."""
        self._event_sink = sink

    def _emit_event(self, event: dict[str, Any]) -> None:
        if self._event_sink is None:
            return
        self._event_sink(event)

    @staticmethod
    def _to_jsonable(obj: Any) -> Any:
        if obj is None or isinstance(obj, (bool, int, float, str)):
            return obj
        if isinstance(obj, dict):
            return {str(k): MLOrchestratorAgent._to_jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [MLOrchestratorAgent._to_jsonable(x) for x in obj]
        if hasattr(obj, "to_json_safe"):
            try:
                return MLOrchestratorAgent._to_jsonable(obj.to_json_safe())
            except Exception:
                pass
        if hasattr(obj, "model_dump"):
            try:
                return MLOrchestratorAgent._to_jsonable(obj.model_dump(mode="json"))
            except Exception:
                pass
        return repr(obj)[:1000]

    def _wire_tool_logging(
        self,
        tools: list[Any],
        on_tool_call: Callable[[str], None] | None = None,
    ) -> None:
        """Hook tool emitters so we can stream events to the UI and (optionally)
        count tool invocations.

        Args:
            tools: BeeAI tool list to instrument.
            on_tool_call: Optional callback invoked with the tool name on every
                tool `start` event. Used by `run()` to decide whether Phase 2
                self-review should be triggered (purely structural — fired
                whenever a prediction tool ran, regardless of confidence).
        """
        for tool in tools:

            def make_recorder(tool_name: str):
                pending: dict[str, Any] = {}

                def on_evt(data: Any, event: Any) -> None:
                    event_name = getattr(event, "name", "")
                    if event_name == "start":
                        if on_tool_call is not None:
                            try:
                                on_tool_call(tool_name)
                            except Exception:
                                pass
                        pending["input"] = self._to_jsonable(getattr(data, "input", None))
                        self._emit_event(
                            {
                                "kind": "tool_call",
                                "agent": "ml",
                                "name": tool_name,
                                "input": pending["input"],
                            }
                        )
                    elif event_name == "success":
                        output = self._to_jsonable(getattr(data, "output", None))
                        self._emit_event(
                            {
                                "kind": "tool_output",
                                "agent": "ml",
                                "name": tool_name,
                                "input": pending.get("input"),
                                "output": output,
                                "success": True,
                            }
                        )
                        pending.clear()
                    elif event_name == "error":
                        err = getattr(data, "error", None)
                        self._emit_event(
                            {
                                "kind": "tool_output",
                                "agent": "ml",
                                "name": tool_name,
                                "input": pending.get("input"),
                                "error": f"{type(err).__name__}: {err}",
                                "success": False,
                            }
                        )
                        pending.clear()

                return on_evt

            try:
                tool.emitter.match("*", make_recorder(tool.name))
            except Exception:
                pass

    async def run(
        self,
        request: str,
        patient_features: dict[str, Any] | None = None,
    ) -> MLOrchestratorResponse:
        """Two-phase consultation:

        - Phase 1: run the user/Chat-Agent request. The model selects ML tools,
          runs predictions, and produces a draft answer.
        - Phase 2 (triggered iff at least one ML tool was invoked in Phase 1):
          send a fixed trigger message to the same agent + memory. The system
          prompt defines the actual self-review logic (credibility judgment,
          missing-feature recommendations). The harness does NOT decide what
          "confidence is enough" — the LLM does, guided by the prompt. This
          keeps the workflow portable into a standalone skill.

        Returns the Phase 2 text when Phase 2 fired, otherwise Phase 1 text.
        """
        from beeai_framework.agents.requirement import RequirementAgent
        from beeai_framework.tools.mcp import MCPTool
        from fastmcp import Client

        from services.ml_mcp_server.server import build_server

        prompt = request
        if patient_features:
            prompt += (
                "\n\nPatient features: "
                f"{json.dumps(patient_features, default=str)}"
            )

        # Structural trigger for Phase 2: did any ML tool run during Phase 1?
        # The ML Orchestrator's tool surface only contains prediction tools
        # (no patient / consult tools), so counting any tool call is equivalent
        # to counting prediction calls.
        phase1_tool_calls = [0]

        def _count(_tool_name: str) -> None:
            phase1_tool_calls[0] += 1

        ml_server = build_server()
        async with Client(ml_server) as client:
            ml_tools = await MCPTool.from_client(client.session)
            self._wire_tool_logging(ml_tools, on_tool_call=_count)

            agent = RequirementAgent(
                llm=self._llm,
                memory=self._memory,
                tools=ml_tools,
                requirements=[],
                name="ML Orchestrator",
                description=(
                    "Professional ML researcher with persistent session memory: "
                    "routes patient features to clinical ML models, interprets "
                    "predictions with SHAP, and performs structural self-review."
                ),
                instructions=self._system_prompt,
                final_answer_as_tool=False,
            )

            phase1_result = await agent.run(prompt)

            if phase1_tool_calls[0] == 0:
                # No prediction tool ran (e.g. meta question). Phase 1 is the
                # final answer — no self-review needed and no needed_features.
                return MLOrchestratorResponse(
                    reasoning=_result_text(phase1_result),
                    needed_features=None,
                )

            # Phase 2 — same agent, same memory. The trigger is content-free;
            # the system prompt carries the actual self-review framework.
            phase2_result = await agent.run(_PHASE_2_TRIGGER)
            phase2_text = _result_text(phase2_result)
            reasoning, needed = _split_reasoning_and_needed_features(phase2_text)
            return MLOrchestratorResponse(
                reasoning=reasoning,
                needed_features=needed,
            )


def build_ml_orchestrator_agent() -> "MLOrchestratorAgent | None":
    """Factory: build a session-persistent MLOrchestratorAgent from env.

    Returns None if the role's LLM cannot be constructed (e.g. missing env).
    Callers can then fall back to the per-call context-manager path.
    """
    try:
        return MLOrchestratorAgent.from_env()
    except (ValueError, KeyError):
        return None
