"""FastMCP server that exposes every registered MLModel as a tool.

Run as a stdio MCP server: `python -m services.ml_mcp_server.server`

The orchestrator (BeeAI) connects to this over MCP and discovers the tools
dynamically — adding a new model never requires orchestrator changes.
"""

from typing import Any

from fastmcp import FastMCP

from packages.schemas.prediction import Prediction
from services.ml_mcp_server.feature_descriptions import (
    describe_ml_features as _describe_ml_features,
)
from services.ml_mcp_server.models._base import MLModel
from services.ml_mcp_server.registry import discover_models


def _register(mcp: FastMCP, model: MLModel) -> None:
    """Register one MLModel as an MCP tool with proper input/output schemas."""
    input_cls = model.input_schema

    def tool_fn(inputs: input_cls) -> Any:  # type: ignore[valid-type]
        from pydantic import TypeAdapter
        result = model.predict(inputs)
        # Ensure NaNs are converted to nulls for valid MCP JSON output
        return TypeAdapter(Prediction).dump_python(result, mode="json")

    tool_fn.__name__ = model.name
    tool_fn.__doc__ = model.metadata.description
    mcp.tool(tool_fn)


def _register_describe_features(mcp: FastMCP) -> None:
    """Register the documentation-lookup tool.

    Distinct from `predict_*`: this is read-only metadata and is the only
    ML MCP tool the Chat Agent ever sees (filtered on the orchestrator
    side). The ML Orchestrator sees both this and the `predict_*` family.
    """

    def describe_ml_features(
        model_name: str | None = None,
        feature_names: list[str] | None = None,
    ) -> list[dict]:
        """Look up author-written documentation for ML model input features.

        Use this when you have a feature name (e.g., 'plas') and need to
        present it to a user in plain language: returns label, plain
        description, unit, field_type, and any aliases the author noted.
        Both filters are optional; use `feature_names` to look up specific
        features by name and `model_name` to scope the lookup to one model.
        """
        descs = _describe_ml_features(model_name, feature_names)
        return [d.model_dump(mode="json") for d in descs]

    mcp.tool(describe_ml_features)


def build_server() -> FastMCP:
    mcp = FastMCP("ml-models")
    for model in discover_models():
        _register(mcp, model)
    _register_describe_features(mcp)
    return mcp


def main() -> None:
    server = build_server()
    server.run()  # stdio transport


if __name__ == "__main__":
    main()
