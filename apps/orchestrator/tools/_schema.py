"""Shared Pydantic base for tool input schemas.

OpenAI strict mode requirements for function tool schemas:
1. Every key in `properties` must appear in `required`
2. `additionalProperties` must be `false` on every object schema
3. These rules apply recursively to nested schemas

`StrictToolInput` patches the generated JSON schema to satisfy both rules
while keeping Python-level defaults so agents can omit optional fields.
"""

from pydantic import BaseModel, ConfigDict


def _make_strict(schema: dict) -> dict:
    """Recursively enforce OpenAI strict mode on a JSON schema dict."""
    if schema.get("type") == "object":
        schema["additionalProperties"] = False
        props = schema.get("properties", {})
        if props:
            schema["required"] = list(props.keys())
        schema["properties"] = {k: _make_strict(v) for k, v in props.items()}
    elif schema.get("type") == "array":
        if "items" in schema:
            schema["items"] = _make_strict(schema["items"])
    # Handle anyOf / oneOf / allOf
    for key in ("anyOf", "oneOf", "allOf"):
        if key in schema:
            schema[key] = [_make_strict(s) for s in schema[key]]
    return schema


class StrictToolInput(BaseModel):
    """BaseModel whose JSON schema is fully OpenAI strict-mode compliant."""

    model_config = ConfigDict(
        json_schema_extra=lambda s: _make_strict(s)
    )
