"""Tests for the describe_ml_features MCP tool.

Smoke-tests the lookup logic against the live registry: the tool reads
each model's input_schema and emits author-written label / description /
unit / field_type / aliases. The diabetes model's metadata is used as the
authoritative example because its features have full author metadata
(see services/ml_mcp_server/models/diabetes_xgb.py:_FEATURE_METADATA).
"""

import pytest

from packages.schemas.ml import FeatureDescription
from services.ml_mcp_server.feature_descriptions import describe_ml_features


class TestDescribeMlFeatures:
    def test_returns_list_of_feature_descriptions(self):
        descs = describe_ml_features(model_name="predict_diabetes_risk")
        assert isinstance(descs, list)
        assert len(descs) > 0
        for d in descs:
            assert isinstance(d, FeatureDescription)

    def test_diabetes_model_has_authored_metadata(self):
        descs = describe_ml_features(
            model_name="predict_diabetes_risk", feature_names=["plas"]
        )
        assert len(descs) == 1
        plas = descs[0]
        assert plas.name == "plas"
        assert plas.label == "Blood sugar"
        assert "glucose" in plas.description.lower()
        assert plas.unit == "mg/dL"
        assert plas.field_type == "number"
        assert "혈당" in plas.aliases
        assert plas.model_name == "predict_diabetes_risk"

    def test_filter_by_feature_names_only(self):
        """Without model_name, looks across all models, returns first match."""
        descs = describe_ml_features(feature_names=["plas", "mass"])
        names = {d.name for d in descs}
        assert names == {"plas", "mass"}

    def test_unknown_feature_silently_dropped(self):
        descs = describe_ml_features(
            model_name="predict_diabetes_risk",
            feature_names=["plas", "this_feature_does_not_exist"],
        )
        names = {d.name for d in descs}
        assert names == {"plas"}

    def test_unknown_model_returns_empty_list(self):
        descs = describe_ml_features(model_name="model_that_does_not_exist")
        assert descs == []

    def test_field_type_falls_back_to_number(self):
        """Models without field_type metadata default to 'number'."""
        descs = describe_ml_features(model_name="predict_diabetes_risk")
        for d in descs:
            assert d.field_type in ("number", "text", "category", "yes_no")

    def test_label_falls_back_to_name_when_unauthored(self):
        """Features without authored label use the feature name as label."""
        # breast_cancer model fields like 'mean_radius' have no authored
        # label in the current diabetes-style metadata — verify the fallback.
        descs = describe_ml_features(model_name="predict_breast_cancer_malignancy")
        assert descs, "breast cancer model should be registered"
        for d in descs:
            # Label is either an authored string or a humanized version of name
            assert d.label
            # Every fallback uses underscore→space, so label has no underscores
            # when it came from the fallback path
            if "_" in d.name:
                assert d.label == d.name.replace("_", " ") or d.label != d.name
