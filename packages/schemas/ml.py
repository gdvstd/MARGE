"""Schemas for ML model feature descriptions and ML Orchestrator responses.

These types live one level above `prediction.py`: they describe *how the
agents talk about features and prediction credibility*, rather than the
raw output of a single model. They are intentionally minimal so that the
same shapes can survive being lifted into a portable skill / external MCP
without dragging in clinical or trainer-side concerns.
"""

from typing import Literal

from pydantic import BaseModel, Field


class FeatureDescription(BaseModel):
    """Documentation for one ML model input feature.

    Source of truth: the model's Pydantic `input_schema` (auto-populated
    from the model's `feature_metadata` dict — see
    `services/ml_mcp_server/models/_agent_factory.py:_build_dynamic_schema`).

    Both the ML Orchestrator and the Chat Agent retrieve these via the
    `describe_ml_features` MCP tool. Neither agent invents these texts —
    they originate in the model definition file authored by the model
    owner.
    """

    name: str = Field(description="Exact feature column name in the model's input_schema.")
    label: str = Field(
        description=(
            "Human-readable display name authored by the model owner. "
            "Falls back to `name` when no label was provided."
        )
    )
    description: str = Field(
        description=(
            "Plain-language explanation of what this feature is and how a user "
            "might find or measure it (e.g., where to look on a lab report). "
            "Authored by the model owner in `feature_metadata[*].detail`."
        )
    )
    unit: str | None = Field(
        default=None,
        description="Optional unit string (e.g., 'mg/dL', 'kg/m^2'). Author-supplied.",
    )
    field_type: Literal["number", "text", "category", "yes_no"] = Field(
        default="number",
        description="UI hint for input rendering.",
    )
    aliases: list[str] = Field(
        default_factory=list,
        description=(
            "Alternative names a user might say for this feature (e.g., 'BMI', "
            "'체질량', '혈당'). Authored alongside the feature metadata."
        ),
    )
    model_name: str | None = Field(
        default=None,
        description=(
            "Name of the model this description came from. Useful when the same "
            "underlying clinical concept appears in multiple models with slightly "
            "different feature names."
        ),
    )


class NeededFeature(BaseModel):
    """One feature the ML Orchestrator wants collected to strengthen a prediction.

    Intentionally minimal — only `name` (which must be an exact ML catalog
    feature name) and `reason` (the ML Orchestrator's own justification,
    typically SHAP-driven). The display-side metadata (label, unit, plain
    description) is looked up separately via `describe_ml_features` by
    whichever agent renders the request to the user.
    """

    name: str = Field(
        description=(
            "Exact feature column name from an ML model's input_schema. "
            "MUST match a registered ML model field; otherwise the Chat "
            "Agent's whitelist will reject the downstream request."
        )
    )
    reason: str = Field(
        description=(
            "One- or two-sentence ML rationale for why this feature would "
            "strengthen the prediction (e.g., 'top SHAP driver and currently "
            "null'). Audience: another agent, not the end user."
        )
    )


class MLOrchestratorResponse(BaseModel):
    """The ML Orchestrator's structured response to a consultation.

    Shape mirrors `MedicalExpertResponse` (reasoning + an optional structured
    list field). `needed_features` is populated only when Phase 2 self-review
    judged that the predictions are not yet credible and named specific
    features the user should be asked for. When predictions are strong enough
    to report directly, this stays None.
    """

    reasoning: str = Field(
        description=(
            "Final user-facing prose from Phase 2 self-review: predictions, "
            "confidence verdict, recommended next steps. Free-form Markdown."
        )
    )
    needed_features: list[NeededFeature] | None = Field(
        default=None,
        description=(
            "When Phase 2's verdict for at least one prediction was 'not yet "
            "credible', this lists the features to collect from the user, in "
            "priority order. None means no follow-up data collection is "
            "needed and the answer can be reported as-is."
        ),
    )
