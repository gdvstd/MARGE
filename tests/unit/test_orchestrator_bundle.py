"""Tests for the OrchestratorBundle assembly.

Bundle wires local tools and the protocol enforcer.
Patient data and ML tools are attached via MCP at runtime and are not tested here.

ARCHITECTURE (3-agent):
  Chat Agent (orchestrator) has:
  - consult_ml_orchestrator   (if llm provided)
  - consult_medical_expert
  - request_more_info
  - clinical_report
  - abstain
  No direct predict_* tools (those belong to ML Orchestrator).

The system prompt has {ML_CATALOG} replaced at build time with real model names.
"""

import pytest

from apps.orchestrator.agent import OrchestratorBundle, _build_ml_catalog, build_bundle
from apps.orchestrator.middleware.enforce_protocol import ProtocolEnforcer


# ---------------------------------------------------------------------------
# Original tools expected without llm (baseline set: 4 pure-local tools)
# ---------------------------------------------------------------------------

_BASE_LOCAL_TOOLS = {
    "consult_medical_expert",
    "request_more_info",
    "clinical_report",
    "abstain",
}

# When llm is provided, consult_ml_orchestrator is also included
_FULL_LOCAL_TOOLS = _BASE_LOCAL_TOOLS | {"consult_ml_orchestrator"}


# ---------------------------------------------------------------------------
# _build_ml_catalog unit tests
# ---------------------------------------------------------------------------


class TestBuildMlCatalog:
    def test_returns_non_empty_string(self):
        catalog = _build_ml_catalog()
        assert isinstance(catalog, str)
        assert len(catalog.strip()) > 0

    def test_contains_diabetes_model_name(self):
        catalog = _build_ml_catalog()
        assert "predict_diabetes_risk" in catalog

    def test_contains_breast_cancer_model_name(self):
        catalog = _build_ml_catalog()
        assert "predict_breast_cancer_malignancy" in catalog

    def test_contains_at_least_one_feature_name(self):
        """Catalog should mention features so Chat Agent knows what inputs to collect."""
        catalog = _build_ml_catalog()
        # Any one of these feature names appearing confirms feature listing
        feature_clues = ["preg", "plas", "mass", "age", "radius", "texture"]
        assert any(f in catalog for f in feature_clues), (
            f"Expected at least one feature name in catalog, got:\n{catalog}"
        )

    def test_is_deterministic(self):
        """Multiple calls must return identical content (no randomness)."""
        assert _build_ml_catalog() == _build_ml_catalog()

    def test_returns_fallback_when_no_models_registered(self):
        """When discover_models returns empty, a clear fallback string is returned.

        _build_ml_catalog imports discover_models lazily inside the function body,
        so we monkey-patch the registry module directly then restore it.
        """
        import services.ml_mcp_server.registry as _registry

        real_discover = _registry.discover_models
        try:
            _registry.discover_models = lambda: []
            catalog = _build_ml_catalog()
        finally:
            _registry.discover_models = real_discover

        assert isinstance(catalog, str)
        assert len(catalog.strip()) > 0
        # Must communicate that nothing is registered
        assert (
            "no" in catalog.lower()
            or "none" in catalog.lower()
            or "registered" in catalog.lower()
        )


# ---------------------------------------------------------------------------
# build_bundle — base shape (no llm)
# ---------------------------------------------------------------------------


class TestBuildBundle:
    def test_returns_orchestrator_bundle(self):
        bundle = build_bundle()
        assert isinstance(bundle, OrchestratorBundle)

    def test_bundle_has_enforcer(self):
        bundle = build_bundle()
        assert isinstance(bundle.enforcer, ProtocolEnforcer)

    def test_bundle_has_four_local_tools_without_llm(self):
        """Without llm, exactly the 4 base local tools are present."""
        bundle = build_bundle()
        assert set(bundle.local_tools.keys()) == _BASE_LOCAL_TOOLS

    @pytest.mark.asyncio
    async def test_bundle_local_tools_share_one_enforcer(self):
        bundle = build_bundle()
        await bundle.local_tools["consult_medical_expert"](question="?", findings={})
        assert bundle.enforcer.has_called("consult_medical_expert")

    def test_system_prompt_loaded_and_not_empty(self):
        bundle = build_bundle()
        assert bundle.system_prompt
        assert len(bundle.system_prompt.strip()) > 0

    def test_system_prompt_contains_consult_medical_expert(self):
        """System prompt must reference the consult_medical_expert tool."""
        bundle = build_bundle()
        assert "consult_medical_expert" in bundle.system_prompt

    def test_system_prompt_does_not_contain_raw_placeholder(self):
        """The {ML_CATALOG} placeholder must be substituted at build time."""
        bundle = build_bundle()
        assert "{ML_CATALOG}" not in bundle.system_prompt

    def test_system_prompt_contains_ml_catalog_content(self):
        """Model names from the catalog appear in the system prompt."""
        bundle = build_bundle()
        assert "predict_diabetes_risk" in bundle.system_prompt
        assert "predict_breast_cancer_malignancy" in bundle.system_prompt


# ---------------------------------------------------------------------------
# build_bundle — with llm (adds consult_ml_orchestrator)
# ---------------------------------------------------------------------------


class _FakeLLM:
    model_id = "fake"
    provider_id = "fake"
    parameters = None
    tool_choice_support = ["auto"]
    middlewares = ()


class TestBuildBundleWithLlm:
    def test_has_five_local_tools_when_llm_provided(self):
        """With llm, consult_ml_orchestrator is added → 5 tools total."""
        bundle = build_bundle(llm=_FakeLLM())
        assert set(bundle.local_tools.keys()) == _FULL_LOCAL_TOOLS

    def test_consult_ml_orchestrator_is_callable(self):
        bundle = build_bundle(llm=_FakeLLM())
        assert callable(bundle.local_tools["consult_ml_orchestrator"])

    def test_consult_ml_orchestrator_is_async(self):
        import inspect

        bundle = build_bundle(llm=_FakeLLM())
        fn = bundle.local_tools["consult_ml_orchestrator"]
        assert inspect.iscoroutinefunction(fn)

    def test_base_tools_still_present_when_llm_provided(self):
        """The 4 base tools must still be present when llm is given."""
        bundle = build_bundle(llm=_FakeLLM())
        assert _BASE_LOCAL_TOOLS <= set(bundle.local_tools.keys())

    def test_system_prompt_unchanged_by_llm_argument(self):
        """Adding llm must not change system prompt content."""
        bundle_no_llm = build_bundle()
        bundle_with_llm = build_bundle(llm=_FakeLLM())
        assert bundle_no_llm.system_prompt == bundle_with_llm.system_prompt
