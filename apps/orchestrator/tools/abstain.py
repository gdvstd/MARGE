"""Local tool: abstain — terminal for "this query is outside our ML scope".

Use ONLY for scope mismatch: the user's concern does not map to any
predictor in the ML catalog, even after the Medical Expert has confirmed
the clinical question is reasonable. Examples: a user asks about a
dermatologic rash, a musculoskeletal symptom, or a pediatric condition,
and the catalog only carries cardiometabolic / oncology / infectious
predictors.

Do NOT use abstain for:
- "I'm not confident in my prediction" — that is `request_ml_clinical_info`
  driven by the ML Orchestrator's needed_features (or by hedging in chat).
- "Expert hedged and ML confidence was mediocre" — synthesize and report,
  or ask for more data.
- "I don't have enough info to answer this clarifying question" — just
  reply in natural language.

Abstain is the system saying "this question is outside what MARGE is built
to analyze" — not "I'm uncertain". The UI renders it as an explicit
scope-mismatch warning with a referral.

Gating: requires at least one consult_medical_expert in the trajectory
(enforced by MARGEProtocolRequirement) so the system has *attempted* a
clinical assessment before declaring scope mismatch.
"""

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field
from apps.orchestrator.tools._schema import StrictToolInput

from apps.orchestrator.middleware.enforce_protocol import ProtocolEnforcer

TOOL_NAME = "abstain"
TOOL_DESCRIPTION = (
    "Terminal: declare that the user's query is outside the scope of MARGE's "
    "ML catalog (no available predictor maps to the clinical concern, even "
    "after the medical expert confirms the question is reasonable). Do NOT "
    "use this for low confidence, mixed signals, or missing data — those are "
    "handled by `request_ml_clinical_info` or by natural-language replies. "
    "Requires at least one consult_medical_expert in the trajectory."
)


class ToolInput(StrictToolInput):
    reason: str = Field(
        description=(
            "Concrete scope-mismatch reason. The user's clinical concern does "
            "not map to any registered ML predictor. Example: 'User describes "
            "a dermatologic rash; the ML catalog covers cardiometabolic and "
            "oncologic predictors only and contains no skin model.'"
        )
    )
    fallback_recommendation: str = Field(
        default="Please consult a qualified clinician for evaluation.",
        description="What the user should do instead — concrete next step.",
    )


def make_abstain(enforcer: ProtocolEnforcer) -> Callable[..., dict[str, Any]]:
    def abstain(
        reason: str,
        fallback_recommendation: str = "Please consult a qualified clinician for evaluation.",
    ) -> dict[str, Any]:
        enforcer.record(TOOL_NAME)
        return {
            "abstained": True,
            "reason": reason,
            "fallback_recommendation": fallback_recommendation,
        }

    abstain.__doc__ = TOOL_DESCRIPTION
    return abstain
