"""Shared Pydantic base for tool input schemas.

OpenAI strict mode requires that every key in `properties` also appears in
`required`. Pydantic omits fields that have defaults from `required`, which
causes OpenAI to reject the tool schema.

`StrictToolInput` patches the generated JSON schema to include all property
keys in `required` while still allowing Python-level defaults so the agent
can omit optional fields in its tool call arguments.
"""

from pydantic import BaseModel, ConfigDict


class StrictToolInput(BaseModel):
    """BaseModel that forces all properties into JSON schema 'required'."""

    model_config = ConfigDict(
        json_schema_extra=lambda s: s.update(
            {"required": list(s.get("properties", {}).keys())}
        )
        or s
    )
