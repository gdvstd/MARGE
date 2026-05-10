"""Local tool: consult the ML Orchestrator sub-agent.

The Chat Agent calls this when it wants to:
- Ask what inputs a specific predictor needs
- Run one or more predictions for given patient features
- Get SHAP-based interpretation of a prediction

The ML Orchestrator selects the right `predict_*` tool(s), runs them,
and returns results with plain-language interpretation.
"""

import json
from typing import Any

from pydantic import BaseModel, Field

from apps.orchestrator.middleware.enforce_protocol import ProtocolEnforcer
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
):
    async def consult_ml_orchestrator(
        request: str,
        patient_features: dict[str, Any] | None = None,
    ) -> str:
        enforcer.record(TOOL_NAME)

        prompt = request
        if patient_features:
            prompt += f"\n\nPatient features: {json.dumps(patient_features, default=str)}"

        async with ml_orchestrator_agent(llm=llm) as _agent:
            result = await _agent.run(prompt)

        response = _result_text(result)
        if on_response is not None:
            on_response("ML Expert", response)
        return response

    consult_ml_orchestrator.__doc__ = TOOL_DESCRIPTION
    return consult_ml_orchestrator
