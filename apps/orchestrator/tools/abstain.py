"""Local tool: abstain — structured out-of-scope guidance.

Use ONLY when the user's clinical concern does not map to any predictor
in the ML catalog, *even after* the Medical Expert has confirmed the
question is reasonable. abstain is the system saying:

  "Given the patient's situation, the current ML predictor stack cannot
   give a meaningful insight. Based on the Expert's reasoning the concern
   might be A, B, or C — please see {appropriate clinician} for an
   in-person assessment."

This is **not** a refusal. It is a structured, hedged hand-off — the
output is rendered as a card analogous to `clinical_report`, with
explicit fields for what the Expert thought the concern might be and
which clinical setting to escalate to.

Do NOT use abstain for:
- "I'm not confident in my prediction" — that is `request_ml_clinical_info`
  driven by the ML Orchestrator's needed_features (or by hedging in chat).
- "Expert hedged and ML confidence was mediocre but a predictor exists" —
  synthesize and report, or ask for more data.
- "I don't have enough info to answer a clarifying question" — just reply
  in natural language.

Gating: requires at least one consult_medical_expert in the trajectory
(enforced by MARGEProtocolRequirement). The Expert's hedged differential
is the source of `possible_directions` — Chat Agent forwards, never
invents.
"""

from collections.abc import Callable
from typing import Any

from pydantic import Field

from apps.orchestrator.middleware.enforce_protocol import ProtocolEnforcer
from apps.orchestrator.tools._schema import StrictToolInput

TOOL_NAME = "abstain"
TOOL_DESCRIPTION = (
    "Terminal: structured out-of-scope guidance. Use when the user's concern "
    "does not map to any predictor in the ML catalog. Provide a concrete "
    "`reason` (why current predictors do not help), a hedged "
    "`possible_directions` list (clinical possibilities sourced from a prior "
    "consult_medical_expert call — never invented), and a `recommended_action` "
    "that names the appropriate clinical setting or specialist for in-person "
    "evaluation. Do NOT use for low confidence, mixed signals, or missing "
    "data — those go to `request_ml_clinical_info` or natural-language "
    "replies. Requires at least one consult_medical_expert in the trajectory."
)


class ToolInput(StrictToolInput):
    reason: str = Field(
        description=(
            "Why the current ML predictor stack cannot give a meaningful "
            "insight for this patient. Be specific about the scope gap, e.g. "
            "'User describes a dermatologic rash; the catalog covers "
            "cardiometabolic and oncologic predictors only.'"
        )
    )
    possible_directions: list[str] = Field(
        default_factory=list,
        description=(
            "Hedged list of clinical possibilities worth considering, sourced "
            "from a prior consult_medical_expert response. Each item is one "
            "short phrase the user could mention to the clinician (e.g., "
            "'Could be contact dermatitis from a new product', 'Could be a "
            "fungal infection in moist skin folds'). Do NOT invent these — "
            "forward only what the Expert proposed. 2–5 items is typical."
        ),
    )
    recommended_action: str = Field(
        default="Please consult a qualified clinician for in-person evaluation.",
        description=(
            "Concrete next step naming the clinical setting or specialist. "
            "Examples: 'See a dermatologist for skin examination', 'Visit "
            "urgent care if the rash is spreading or accompanied by fever', "
            "'Schedule an appointment with a general practitioner'."
        ),
    )


def make_abstain(enforcer: ProtocolEnforcer) -> Callable[..., dict[str, Any]]:
    def abstain(
        reason: str,
        possible_directions: list[str] | None = None,
        recommended_action: str = "Please consult a qualified clinician for in-person evaluation.",
    ) -> dict[str, Any]:
        enforcer.record(TOOL_NAME)
        return {
            "abstained": True,
            "reason": reason,
            "possible_directions": possible_directions or [],
            "recommended_action": recommended_action,
        }

    abstain.__doc__ = TOOL_DESCRIPTION
    return abstain
