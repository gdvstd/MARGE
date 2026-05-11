"""MARGE Protocol Requirement (custom BeeAI Requirement).

3-agent architecture rules (Chat Agent + ML Orchestrator + Expert):

  B. clinical_report (terminal) needs consult_ml_orchestrator AND
     consult_medical_expert in the trajectory. Order is free — the two
     experts may be called interleaved multiple times.
  C. abstain (terminal) needs consult_medical_expert at least once.
  D. request_ml_clinical_info (terminal) is always allowed.

No ordering constraint between consult_ml_orchestrator and
consult_medical_expert — the chat agent decides dynamically which to
call next based on what insight is needed.

Natural-language chat (no tool call) remains a valid turn ending for
casual conversation. The prompt steers the model toward structured
terminals for clinical analysis.
"""

from typing import Any

from beeai_framework.agents.requirement.requirements.requirement import (
    Requirement,
    Rule,
    run_with_context,
)


_ML_PREDICTION_PREFIX = "predict_"
_ML_ORCHESTRATOR_TOOL = "consult_ml_orchestrator"
_EXPERT_TOOL_NAME = "consult_medical_expert"


def _successful_tool_names(state: Any) -> list[str]:
    return [
        s.tool.name
        for s in state.steps
        if s.tool is not None and not s.error and getattr(s.tool, "name", None)
    ]


def has_any_ml_prediction(state: Any) -> bool:
    """True if consult_ml_orchestrator or any predict_* tool has succeeded."""
    return any(
        name == _ML_ORCHESTRATOR_TOOL or name.startswith(_ML_PREDICTION_PREFIX)
        for name in _successful_tool_names(state)
    )


def has_consulted_expert(state: Any) -> bool:
    """True if consult_medical_expert has succeeded in the trajectory."""
    return _EXPERT_TOOL_NAME in _successful_tool_names(state)


def has_pre_ml_expert_consult(state: Any) -> bool:
    """True if an expert consult succeeded before a successful ML prediction."""
    seen_expert = False
    for name in _successful_tool_names(state):
        if name == _EXPERT_TOOL_NAME:
            seen_expert = True
        elif name.startswith(_ML_PREDICTION_PREFIX) and seen_expert:
            return True
    return False


def has_post_ml_expert_consult(state: Any) -> bool:
    """True if an expert consult succeeded after a successful ML prediction."""
    seen_ml = False
    for name in _successful_tool_names(state):
        if name.startswith(_ML_PREDICTION_PREFIX):
            seen_ml = True
        elif name == _EXPERT_TOOL_NAME and seen_ml:
            return True
    return False


def has_expert_ml_expert_sequence(state: Any) -> bool:
    """True if the successful trajectory contains expert -> ML -> expert."""
    seen_pre_expert = False
    seen_ml_after_pre_expert = False

    for name in _successful_tool_names(state):
        if name == _EXPERT_TOOL_NAME:
            if seen_ml_after_pre_expert:
                return True
            seen_pre_expert = True
        elif name.startswith(_ML_PREDICTION_PREFIX) and seen_pre_expert:
            seen_ml_after_pre_expert = True

    return False


class MARGEProtocolRequirement(Requirement):
    """Single Requirement encoding the four MARGE protocol rules (A–D above)."""

    TERMINALS = frozenset(
        {"clinical_report", "abstain", "request_ml_clinical_info"}
    )

    def __init__(self) -> None:
        super().__init__()
        # BeeAI's Requirement base treats `name` as a required attribute —
        # the framework reads it (via `to_safe_word(name)`) when creating
        # the per-requirement emitter group. Set it explicitly here.
        self.name = "marge_protocol"

    @property
    def priority(self) -> int:
        return 50

    async def init(self, *, tools, ctx) -> None:
        await super().init(tools=tools, ctx=ctx)
        self._predict_tools = []
        self._terminal_tools = [t for t in tools if t.name in self.TERMINALS]
        # Rule E: track ML orchestrator tool to enforce consult after expert
        self._ml_orchestrator_tool = next(
            (t for t in tools if t.name == _ML_ORCHESTRATOR_TOOL), None
        )

    @run_with_context
    async def run(self, state: Any, context: Any) -> list[Rule]:  # noqa: ARG002
        return self._compute_rules(state)

    def _compute_rules(self, state: Any) -> list[Rule]:
        """Pure-sync rule evaluation. Public for tests."""
        called = _successful_tool_names(state)
        has_expert = _EXPERT_TOOL_NAME in called
        has_ml = has_any_ml_prediction(state)

        rules: list[Rule] = []

        # Rules B / C / D (terminals — gating only, no prevent_stop)
        for tool in getattr(self, "_terminal_tools", []):
            if tool.name == "clinical_report":
                allowed = has_ml and has_expert
                reason = (
                    None
                    if allowed
                    else (
                        "clinical_report needs at least one predict_* run AND "
                        "one consult_medical_expert in the trajectory."
                    )
                )
            elif tool.name == "abstain":
                allowed = has_expert
                reason = (
                    None
                    if allowed
                    else (
                        "abstain may only be used after consulting the medical "
                        "expert at least once."
                    )
                )
            else:  # request_ml_clinical_info — free
                allowed = True
                reason = None

            rules.append(
                Rule(
                    target=tool.name,
                    allowed=allowed,
                    prevent_stop=False,
                    hidden=False,
                    forced=False,
                    reason=reason,
                )
            )

        # Rule E: after expert consult, prevent turn ending until ML orchestrator
        # is also consulted. Models accept null features — run with partial data
        # and use XAI scores to identify the most important missing features.
        ml_tool = getattr(self, "_ml_orchestrator_tool", None)
        if has_expert and not has_ml and ml_tool is not None:
            rules.append(
                Rule(
                    target=ml_tool.name,
                    allowed=True,
                    prevent_stop=True,
                    hidden=False,
                    forced=False,
                    reason=(
                        "After consulting the medical expert you MUST also call "
                        "consult_ml_orchestrator before ending this turn. "
                        "Pass available patient features (use null for unknowns) — "
                        "the ML Orchestrator handles missing values and returns "
                        "XAI scores so you know exactly which features matter most."
                    ),
                )
            )

        return rules


def build_marge_protocol_requirement() -> MARGEProtocolRequirement:
    """Factory used by `apps/orchestrator/agent.py`."""
    return MARGEProtocolRequirement()
