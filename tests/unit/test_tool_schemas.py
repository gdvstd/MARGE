"""Tests for the per-tool Pydantic input schemas (4 local tools).

Casual chat is natural-language content with no tool call (hybrid
pattern — see apps/orchestrator/system_prompt.md), so there is no
update_user / conversational_reply schema to verify.
"""

import pytest
from pydantic import ValidationError

from apps.orchestrator.tools import (
    abstain as ab_mod,
    clinical_report as cr_mod,
    consult_expert as ce_mod,
    request_ml_clinical_info as rmi_mod,
)


class TestConsultExpertSchema:
    def test_constants_exposed(self):
        assert ce_mod.TOOL_NAME == "consult_medical_expert"
        assert ce_mod.TOOL_DESCRIPTION

    def test_accepts_question_and_findings(self):
        obj = ce_mod.ToolInput(question="why?", findings={"a": 1})
        assert obj.question == "why?"
        assert obj.findings == {"a": 1}

    def test_findings_defaults_to_empty(self):
        obj = ce_mod.ToolInput(question="?")
        assert obj.findings == {}


class TestRequestMlClinicalInfoSchema:
    def test_constants_exposed(self):
        assert rmi_mod.TOOL_NAME == "request_ml_clinical_info"
        assert rmi_mod.TOOL_DESCRIPTION

    def test_accepts_full_payload(self):
        obj = rmi_mod.ToolInput(
            target_condition="type-2 diabetes risk",
            known_features=[
                {"label": "BMI", "value": "24.1", "unit": "kg/m²"}
            ],
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
            rationale="Plasma glucose would lift confidence.",
        )
        assert obj.target_condition == "type-2 diabetes risk"
        assert obj.needed_features[0].name == "plas"
        assert obj.needed_features[0].field_type == "number"
        assert obj.known_features[0].label == "BMI"

    def test_known_features_defaults_to_empty(self):
        obj = rmi_mod.ToolInput(
            target_condition="x",
            needed_features=[],
            rationale="r",
        )
        assert obj.known_features == []

    def test_rejects_missing_required_fields(self):
        with pytest.raises(ValidationError):
            rmi_mod.ToolInput(needed_features=[])
        with pytest.raises(ValidationError):
            rmi_mod.ToolInput(target_condition="x", rationale="r")


class TestClinicalReportSchema:
    def test_constants_exposed(self):
        assert cr_mod.TOOL_NAME == "clinical_report"
        assert cr_mod.TOOL_DESCRIPTION

    def test_accepts_full_payload(self):
        obj = cr_mod.ToolInput(
            summary="High diabetes risk.",
            recommendation="Refer for confirmation.",
            confidence="high",
            evidence=[{
                "model": "predict_diabetes_risk",
                "predicted_class": "diabetic_risk",
                "confidence": 0.85,
                "top_features": [],
            }],
            expert_quote="HbA1c at threshold.",
        )
        assert obj.confidence == "high"
        assert obj.evidence[0].model == "predict_diabetes_risk"

    def test_confidence_must_be_one_of_three(self):
        with pytest.raises(ValidationError):
            cr_mod.ToolInput(summary="s", recommendation="r", confidence="absolute")


class TestAbstainSchema:
    def test_constants_exposed(self):
        assert ab_mod.TOOL_NAME == "abstain"
        assert ab_mod.TOOL_DESCRIPTION

    def test_accepts_minimal_reason(self):
        obj = ab_mod.ToolInput(reason="Out of scope.")
        assert obj.reason == "Out of scope."
        # possible_directions defaults to empty list.
        assert obj.possible_directions == []
        # recommended_action has a default referral string.
        assert obj.recommended_action
        assert "clinician" in obj.recommended_action.lower()

    def test_accepts_full_structured_payload(self):
        obj = ab_mod.ToolInput(
            reason="Dermatologic concern — catalog has no skin model.",
            possible_directions=[
                "Could be contact dermatitis",
                "Could be a fungal infection",
            ],
            recommended_action="See a dermatologist.",
        )
        assert len(obj.possible_directions) == 2
        assert obj.recommended_action == "See a dermatologist."
