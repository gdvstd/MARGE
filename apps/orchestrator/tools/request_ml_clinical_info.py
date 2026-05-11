"""Local tool: request_ml_clinical_info — ML-driven structured inquiry to the user.

This is the *only* terminal the Chat Agent uses to gather more clinical input
data. It is purposely narrow:

- The features asked for MUST be exact column names of a registered ML model
  (whitelist enforced). Free-form clinical questions ("how long have you had
  this?", "are you on medication?") belong in a plain natural-language reply,
  not in this tool.
- The Chat Agent populates the payload from two sources only — the ML
  Orchestrator's `needed_features` (which gives the bare names) and the
  `describe_ml_features` MCP tool (which gives author-written labels, units,
  and plain-language descriptions). The Chat Agent does NOT invent any of
  those texts itself.

The UI renders this as a structured "Clinical inquiry" card distinct from
ordinary chat messages: it shows what the agent already knows about the
patient, what it still needs, and why each missing field matters.
"""

from collections.abc import Callable
from typing import Any, Literal

from pydantic import Field

from apps.orchestrator.middleware.enforce_protocol import ProtocolEnforcer
from apps.orchestrator.tools._schema import StrictToolInput

TOOL_NAME = "request_ml_clinical_info"
TOOL_DESCRIPTION = (
    "Terminal: ask the user for the specific ML-model input features needed "
    "to run a credible prediction for a target condition. Pass the values "
    "you already know in `known_features`, the missing features in "
    "`needed_features` (each `name` MUST be an exact ML catalog feature "
    "name — use describe_ml_features to look up label / unit / description), "
    "and a one- or two-sentence `rationale` explaining why these specific "
    "values matter. The UI will render this as a structured Clinical inquiry "
    "card. Always allowed (no protocol prerequisites). For ordinary "
    "clarifying questions, reply in natural language without using this tool."
)


class KnownFeature(StrictToolInput):
    """A clinical value the agent has already learned (from chat or records)."""

    label: str = Field(
        description="Human-readable name (e.g., 'Plasma glucose', 'BMI')."
    )
    value: str = Field(
        description="Stringified value (numbers as strings keep formatting)."
    )
    unit: str | None = Field(
        default=None,
        description="Optional unit string, e.g. 'mg/dL', 'kg/m²'.",
    )


class NeededFeatureRequest(StrictToolInput):
    """One missing feature the user is being asked to supply."""

    name: str = Field(
        description=(
            "Exact feature column name from a registered ML model's input_schema. "
            "MUST match a catalog feature exactly — clinical terms like "
            "'symptoms' or 'medical_history' are NOT valid here."
        )
    )
    label: str = Field(
        description=(
            "Human-readable label authored by the model owner (look up via "
            "describe_ml_features). Example: 'Blood sugar' for `plas`."
        )
    )
    why: str = Field(
        description=(
            "Short ML rationale forwarded from the ML Orchestrator's "
            "`needed_features[*].reason` (e.g. 'top SHAP driver, currently null')."
        )
    )
    explanation: str = Field(
        description=(
            "Plain-language description of what the feature is and how the "
            "user can find or measure it. Source from describe_ml_features's "
            "`description` field — do NOT invent."
        )
    )
    field_type: Literal["number", "text", "category", "yes_no"] = Field(
        default="number",
        description="UI hint for input rendering. Source from describe_ml_features.",
    )
    unit: str | None = Field(
        default=None,
        description="Optional unit, e.g. 'mg/dL'. Source from describe_ml_features.",
    )


class ToolInput(StrictToolInput):
    target_condition: str = Field(
        description=(
            "Plain-language name of the condition being investigated (e.g., "
            "'type-2 diabetes risk', 'breast cancer malignancy'). Used as "
            "the inquiry card header."
        )
    )
    known_features: list[KnownFeature] = Field(
        default_factory=list,
        description=(
            "Clinical values already known about the patient — from prior "
            "messages or the patient record. Shown in the inquiry card so "
            "the user does not have to repeat them."
        ),
    )
    needed_features: list[NeededFeatureRequest] = Field(
        description=(
            "Missing ML-feature inputs to ask the user for, in priority order. "
            "Source from the most recent consult_ml_orchestrator response's "
            "`needed_features`, then look up each name's display metadata via "
            "describe_ml_features."
        )
    )
    rationale: str = Field(
        description=(
            "One- or two-sentence framing for the user explaining why these "
            "specific data points matter (typically a brief restatement of "
            "the ML Orchestrator's credibility verdict)."
        )
    )


def make_request_ml_clinical_info(
    enforcer: ProtocolEnforcer,
    valid_feature_names: frozenset[str] | None = None,
) -> Callable[..., dict[str, Any]]:
    """Build the request_ml_clinical_info tool.

    Args:
        enforcer: Protocol enforcer instance.
        valid_feature_names: Frozenset of all valid ML model feature names.
            When provided, any `needed_features[].name` not in this set is
            rejected with an error message that lists the valid names — this
            blocks the Chat Agent from asking for non-ML feature data via
            this tool.
    """

    def request_ml_clinical_info(
        target_condition: str,
        needed_features: list[dict[str, Any]],
        rationale: str,
        known_features: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        if valid_feature_names is not None:
            invalid = [
                item.get("name", "")
                for item in needed_features
                if item.get("name", "") not in valid_feature_names
            ]
            if invalid:
                invalid_str = ", ".join(f'"{n}"' for n in invalid)
                valid_str = ", ".join(sorted(valid_feature_names))
                return {
                    "error": (
                        f"Invalid feature name(s): {invalid_str}. "
                        f"request_ml_clinical_info only accepts exact ML model "
                        f"feature names. Valid features: {valid_str}. "
                        f"For free-form clinical questions (symptoms, history, "
                        f"medications), reply in natural language without "
                        f"calling this tool."
                    )
                }

        enforcer.record(TOOL_NAME)
        return {
            "needs_more_info": True,
            "target_condition": target_condition,
            "known_features": known_features or [],
            "needed_features": needed_features,
            "rationale": rationale,
        }

    request_ml_clinical_info.__doc__ = TOOL_DESCRIPTION
    return request_ml_clinical_info
