"""Tests for the orchestrator's local tool factories.

Four local tools after the hybrid refactor:
- consult_expert     — sub-agent invocation; records call
- request_ml_clinical_info  — terminal, free; records call
- clinical_report    — terminal, gated by Requirement; records call (no in-tool gate)
- abstain            — terminal, gated by Requirement; records call

Casual chat is plain natural-language `content` from the LLM (no tool
call) — there is no longer an `update_user` or `conversational_reply`
tool. Gating is enforced LLM-side by `MARGEProtocolRequirement` (tested
in test_marge_requirement.py).
"""

import pytest

from apps.orchestrator.middleware.enforce_protocol import ProtocolEnforcer
from apps.orchestrator.tools.abstain import make_abstain
from apps.orchestrator.tools.clinical_report import make_clinical_report
from apps.orchestrator.tools.consult_expert import make_consult_expert
from apps.orchestrator.tools.request_ml_clinical_info import make_request_ml_clinical_info
from packages.schemas.retrieval import MedicalExpertResponse
from services.medical_expert_agent.agent import StubMedicalExpert


class TestConsultExpertTool:
    @pytest.mark.asyncio
    async def test_returns_medical_expert_response(self):
        enforcer = ProtocolEnforcer()
        consult = make_consult_expert(StubMedicalExpert(), enforcer)
        response = await consult(question="What does this suggest?", findings={"a": 1})
        assert isinstance(response, MedicalExpertResponse)

    @pytest.mark.asyncio
    async def test_records_consult_medical_expert_call(self):
        enforcer = ProtocolEnforcer()
        consult = make_consult_expert(StubMedicalExpert(), enforcer)
        await consult(question="?", findings={})
        assert enforcer.has_called("consult_medical_expert")


class TestRequestMlClinicalInfoTool:
    def test_returns_structured_payload(self):
        enforcer = ProtocolEnforcer()
        ask = make_request_ml_clinical_info(enforcer)
        out = ask(
            target_condition="type-2 diabetes risk",
            needed_features=[
                {
                    "name": "plas",
                    "label": "Blood sugar",
                    "why": "top SHAP driver, missing",
                    "explanation": "Recent fasting glucose result.",
                    "field_type": "number",
                    "unit": "mg/dL",
                }
            ],
            known_features=[
                {"label": "BMI", "value": "24.1", "unit": "kg/m²"}
            ],
            rationale="Insulin and plasma glucose would lift confidence.",
        )
        assert out["needs_more_info"] is True
        assert out["target_condition"] == "type-2 diabetes risk"
        assert out["needed_features"][0]["name"] == "plas"
        assert out["known_features"][0]["label"] == "BMI"
        assert "lift confidence" in out["rationale"]

    def test_records_call(self):
        enforcer = ProtocolEnforcer()
        ask = make_request_ml_clinical_info(enforcer)
        ask(
            target_condition="x",
            needed_features=[],
            rationale="x",
        )
        assert enforcer.has_called("request_ml_clinical_info")

    def test_whitelist_rejects_non_catalog_feature(self):
        enforcer = ProtocolEnforcer()
        ask = make_request_ml_clinical_info(
            enforcer, valid_feature_names=frozenset({"plas", "mass"})
        )
        out = ask(
            target_condition="diabetes",
            needed_features=[
                {
                    "name": "symptoms",  # not in whitelist
                    "label": "Symptoms",
                    "why": "?",
                    "explanation": "?",
                }
            ],
            rationale="?",
        )
        assert "error" in out
        assert "symptoms" in out["error"]
        # Tool was rejected — enforcer should NOT have recorded the call.
        assert not enforcer.has_called("request_ml_clinical_info")


class TestClinicalReportTool:
    def test_returns_structured_report(self):
        enforcer = ProtocolEnforcer()
        report = make_clinical_report(enforcer)
        out = report(
            summary="High diabetes risk.",
            recommendation="See PCP for HbA1c repeat.",
            confidence="high",
            evidence=[{"model": "predict_diabetes_risk",
                       "predicted_class": "diabetic_risk",
                       "confidence": 0.85, "top_features": []}],
            expert_quote="HbA1c 6.5% meets ADA criteria.",
        )
        assert out["summary"] == "High diabetes risk."
        assert out["confidence"] == "high"
        assert out["evidence"][0]["model"] == "predict_diabetes_risk"
        assert "clinician" in out["safety_note"]

    def test_records_call(self):
        enforcer = ProtocolEnforcer()
        report = make_clinical_report(enforcer)
        report(summary="x", recommendation="y", confidence="medium")
        assert enforcer.has_called("clinical_report")


class TestAbstainTool:
    def test_returns_abstention_payload(self):
        enforcer = ProtocolEnforcer()
        abst = make_abstain(enforcer)
        out = abst(
            reason="Symptoms outside ML scope — catalog has no dermatologic model.",
            possible_directions=[
                "Could be contact dermatitis from a new product",
                "Could be a fungal infection in moist skin folds",
            ],
            recommended_action="See a dermatologist for skin examination.",
        )
        assert out["abstained"] is True
        assert "scope" in out["reason"]
        assert len(out["possible_directions"]) == 2
        assert "dermatologist" in out["recommended_action"]

    def test_defaults_when_minimal_payload(self):
        enforcer = ProtocolEnforcer()
        abst = make_abstain(enforcer)
        out = abst(reason="Out of scope.")
        # possible_directions defaults to empty list (Chat Agent may have
        # nothing to forward if Expert was not consulted with hedging).
        assert out["possible_directions"] == []
        # recommended_action has a sensible default referral string.
        assert "clinician" in out["recommended_action"].lower()

    def test_records_call(self):
        enforcer = ProtocolEnforcer()
        abst = make_abstain(enforcer)
        abst(reason="x")
        assert enforcer.has_called("abstain")
