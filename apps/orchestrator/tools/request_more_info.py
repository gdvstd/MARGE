"""Local tool: request_more_info — terminal asking the user for ML model inputs only.

ONLY asks for features that are exact input column names of a registered ML model.
Clinical interview questions (symptoms, history, medications) are NOT allowed here —
those belong in consult_medical_expert.

This is a terminal because it ends the agent loop and waits for the user's
next message. It is NOT gated — the orchestrator may request more info at
any point in the conversation, including before any expert consultation.
"""

from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel, Field
from apps.orchestrator.tools._schema import StrictToolInput

from apps.orchestrator.middleware.enforce_protocol import ProtocolEnforcer

TOOL_NAME = "request_more_info"
TOOL_DESCRIPTION = (
    "Terminal: ask the user for specific ML model input features that are missing. "
    "Every item in `needed` MUST use an exact feature name from the ML model catalog "
    "(e.g., 'plas', 'mass', 'age', 'ejection_fraction'). "
    "DO NOT ask for symptoms, medications, history, or any field not in the ML catalog. "
    "Run consult_ml_orchestrator first to get SHAP scores, then ask only for the "
    "highest-SHAP missing features. Always allowed (no protocol prerequisites)."
)


class NeededField(BaseModel):
    name: str = Field(
        description=(
            "Exact feature column name from the ML model catalog "
            "(e.g., 'plas', 'mass', 'age'). "
            "MUST match a catalog feature exactly — clinical terms like "
            "'symptoms' or 'medical_history' are NOT valid."
        )
    )
    why: str = Field(description="One-line reason this ML feature would shift the prediction.")
    field_type: Literal["number", "text", "category", "yes_no"] = Field(
        default="text",
        description="Hint to the UI for input rendering.",
    )
    unit: str | None = Field(
        default=None,
        description="Optional unit if numeric (e.g., 'mg/dL', 'kg/m^2').",
    )


class ToolInput(StrictToolInput):
    needed: list[NeededField] = Field(
        description=(
            "ML model input features to ask the user for. Each name MUST be an exact "
            "column from the ML catalog. Keep the list short (1–4 items). "
            "Run consult_ml_orchestrator first to identify which missing features "
            "have the highest SHAP scores."
        ),
    )
    rationale: str = Field(
        description=(
            "One- or two-sentence framing for the user explaining why these "
            "specific data points matter for the ML prediction."
        )
    )


def make_request_more_info(
    enforcer: ProtocolEnforcer,
    valid_feature_names: frozenset[str] | None = None,
) -> Callable[..., dict[str, Any]]:
    """Build request_more_info tool.

    Args:
        enforcer: Protocol enforcer instance.
        valid_feature_names: Frozenset of all valid ML model feature names.
            When provided, any `needed[].name` not in this set is rejected
            and an error is returned to the agent.
    """

    def request_more_info(
        needed: list[dict[str, Any]],
        rationale: str,
    ) -> dict[str, Any]:
        if valid_feature_names is not None:
            invalid = [
                item.get("name", "")
                for item in needed
                if item.get("name", "") not in valid_feature_names
            ]
            if invalid:
                invalid_str = ", ".join(f'"{n}"' for n in invalid)
                valid_str = ", ".join(sorted(valid_feature_names))
                return {
                    "error": (
                        f"Invalid feature name(s): {invalid_str}. "
                        f"request_more_info only accepts exact ML model feature names. "
                        f"Valid features: {valid_str}. "
                        f"For clinical questions (symptoms, history, medications), "
                        f"use consult_medical_expert instead."
                    )
                }

        enforcer.record(TOOL_NAME)
        return {
            "needs_more_info": True,
            "needed": needed,
            "rationale": rationale,
        }

    request_more_info.__doc__ = TOOL_DESCRIPTION
    return request_more_info
