"""Local tool: consult the ML Orchestrator sub-agent.

The Chat Agent calls this when it wants to:
- Ask what inputs a specific predictor needs
- Run one or more predictions for given patient features
- Get SHAP-based interpretation of a prediction

The ML Orchestrator selects the right `predict_*` tool(s), runs them,
and returns results with plain-language interpretation.

Two backing modes:

- If a session-persistent `MLOrchestratorAgent` instance is provided via
  the `ml_orchestrator` kwarg of `make_consult_ml_orchestrator`, all
  consultations go through that instance (memory persists across calls,
  trace events flow through its event sink).
- Otherwise the older per-call `ml_orchestrator_agent` context manager
  is used (fresh agent + fresh memory each call). This keeps unit tests
  that patch `ml_orchestrator_agent` working without changes.
"""

import json
from typing import Any

from pydantic import BaseModel, Field

from apps.orchestrator.middleware.enforce_protocol import ProtocolEnforcer
from packages.schemas.ml import MLOrchestratorResponse
from services.ml_orchestrator_agent.agent import _result_text, ml_orchestrator_agent

TOOL_NAME = "consult_ml_orchestrator"
TOOL_DESCRIPTION = (
    "Ask the ML Orchestrator to run clinical predictions or explain what "
    "models are available and what inputs they need. Provide patient feature "
    "values when requesting predictions. Returns prediction results with "
    "confidence scores, SHAP feature contributions, and plain-language "
    "interpretation."
)


class ToolInput(BaseModel):
    request: str = Field(
        description=(
            "Natural language request. Examples: "
            "'What inputs does the diabetes predictor need?', "
            "'Predict diabetes risk with: plas=148, mass=33.6, age=50, pedi=0.627'."
        )
    )
    patient_features: dict[str, Any] | None = Field(
        default=None,
        description="Feature name → value dict to attach alongside the request.",
    )


def make_consult_ml_orchestrator(
    llm: Any,
    enforcer: ProtocolEnforcer,
    on_response: Any = None,
    ml_orchestrator: Any = None,
):
    """Build the consult_ml_orchestrator tool callable.

    Args:
        llm: ChatModel used when no `ml_orchestrator` instance is given —
            falls back to the per-call `ml_orchestrator_agent` context
            manager (fresh agent + fresh memory per call).
        enforcer: ProtocolEnforcer to record the tool invocation in the
            Chat Agent's trajectory.
        on_response: Optional `(agent_name: str, response: str) -> None`
            callback invoked once the sub-agent returns its final text.
        ml_orchestrator: Optional session-persistent `MLOrchestratorAgent`
            instance. When provided, all consultations go through this
            instance — memory persists across calls and tool events flow
            through its event sink.
    """

    async def consult_ml_orchestrator(
        request: str,
        patient_features: dict[str, Any] | None = None,
    ) -> MLOrchestratorResponse:
        enforcer.record(TOOL_NAME)

        if ml_orchestrator is not None:
            # Session-persistent agent path — returns a structured response
            # with optional `needed_features` list parsed from Phase 2's
            # JSON tail. The Chat Agent reads `reasoning` for prose and
            # `needed_features` for follow-up data collection.
            response = await ml_orchestrator.run(
                request, patient_features=patient_features
            )
        else:
            # Per-call fallback (used by tests that patch ml_orchestrator_agent).
            # No session memory, no Phase 2 — just wrap the agent's text output
            # into the same response shape so callers stay uniform.
            prompt = request
            if patient_features:
                prompt += (
                    "\n\nPatient features: "
                    f"{json.dumps(patient_features, default=str)}"
                )
            async with ml_orchestrator_agent(llm=llm) as _agent:
                result = await _agent.run(prompt)
            response = MLOrchestratorResponse(
                reasoning=_result_text(result),
                needed_features=None,
            )

        if on_response is not None:
            on_response("ML Expert", response.reasoning)
        return response

    consult_ml_orchestrator.__doc__ = TOOL_DESCRIPTION
    return consult_ml_orchestrator
