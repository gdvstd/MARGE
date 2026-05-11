"""Feature documentation lookup for registered ML models.

Both the ML Orchestrator and the Chat Agent retrieve feature metadata via
this single source. The information is the model author's authored text in
`feature_metadata` (label / detail / unit / field_type / aliases), surfaced
through the model's `input_schema` Pydantic Field objects.

Exposed as the MCP tool `describe_ml_features`. The ML Orchestrator sees
this alongside the `predict_*` tools; the Chat Agent sees a filtered tool
list that contains *only* this tool (no predict_* leakage).
"""

from typing import Any

from packages.schemas.ml import FeatureDescription
from services.ml_mcp_server.registry import discover_models


def _field_to_description(
    feature_name: str,
    field: Any,
    model_name: str,
) -> FeatureDescription:
    """Project a Pydantic FieldInfo to a FeatureDescription."""
    extra = field.json_schema_extra or {}
    return FeatureDescription(
        name=feature_name,
        label=extra.get("label") or feature_name.replace("_", " "),
        description=(
            field.description or f"Feature measurement: {feature_name}."
        ),
        unit=extra.get("unit"),
        field_type=extra.get("field_type", "number"),
        aliases=list(extra.get("aliases") or []),
        model_name=model_name,
    )


def describe_ml_features(
    model_name: str | None = None,
    feature_names: list[str] | None = None,
) -> list[FeatureDescription]:
    """Return author-written documentation for ML model input features.

    Behavior:
    - `model_name` only: return every feature of that model.
    - `feature_names` only: search across all models, return one descriptor
      per requested name (the first model that defines it wins).
    - both: filter `model_name`'s features by the requested names.
    - neither: return every feature of every model (verbose — prefer one
      of the filtered modes when possible).

    Names not found are silently dropped from the result; callers should
    check by length / by name match in the returned list.
    """
    models = discover_models()
    if model_name:
        models = [m for m in models if m.name == model_name]

    requested = set(feature_names or [])
    results: list[FeatureDescription] = []
    seen_global: set[str] = set()

    for model in models:
        for fname, field in model.input_schema.model_fields.items():
            if requested and fname not in requested:
                continue
            if (
                not model_name
                and feature_names
                and fname in seen_global
            ):
                continue
            results.append(_field_to_description(fname, field, model.name))
            if not model_name and feature_names:
                seen_global.add(fname)

    return results
