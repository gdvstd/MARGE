"""Wrap expert-only Python callables as BeeAI tools."""

from __future__ import annotations

import functools
from collections.abc import Callable
from types import ModuleType
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from services.medical_expert_agent.tools import search_web as _search_web

if TYPE_CHECKING:
    from beeai_framework.tools import Tool


EXPERT_TOOL_MODULES: tuple[ModuleType, ...] = (_search_web,)


def _to_tool_output(result: Any) -> Any:
    from beeai_framework.tools import JSONToolOutput

    if isinstance(result, BaseModel):
        return JSONToolOutput(result.model_dump(mode="json"))
    return JSONToolOutput(result)


def to_beeai_tool(
    fn: Callable[..., Any],
    *,
    name: str,
    description: str,
    input_schema: type[BaseModel],
) -> "Tool":
    import inspect as _inspect

    from beeai_framework.tools import tool

    if _inspect.iscoroutinefunction(fn):

        @functools.wraps(fn)
        async def output_wrapped(*args: Any, **kwargs: Any) -> Any:
            return _to_tool_output(await fn(*args, **kwargs))

    else:

        @functools.wraps(fn)
        def output_wrapped(*args: Any, **kwargs: Any) -> Any:
            return _to_tool_output(fn(*args, **kwargs))

    decorator = tool(name=name, description=description, input_schema=input_schema)
    return decorator(output_wrapped)


def _limited(fn: Callable[..., Any], max_calls: int) -> Callable[..., Any]:
    """Wrap `fn` so it can be called at most `max_calls` times per session."""
    count = 0

    import inspect as _inspect

    if _inspect.iscoroutinefunction(fn):
        @functools.wraps(fn)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            nonlocal count
            if count >= max_calls:
                return {"documents": [], "warning": f"Web search limit ({max_calls}) reached for this turn."}
            count += 1
            return await fn(*args, **kwargs)
        return async_wrapper
    else:
        @functools.wraps(fn)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            nonlocal count
            if count >= max_calls:
                return {"documents": [], "warning": f"Web search limit ({max_calls}) reached for this turn."}
            count += 1
            return fn(*args, **kwargs)
        return sync_wrapper


def expert_tools_as_beeai(max_web_searches: int = 1) -> list["Tool"]:
    """Build BeeAI tools for the expert agent.

    Args:
        max_web_searches: Maximum number of `search_medical_web` calls allowed
            per `consult()` invocation. Defaults to 1.
    """
    tools: list[Tool] = []
    for mod in EXPERT_TOOL_MODULES:
        impl = getattr(mod, mod.TOOL_NAME)
        if mod.TOOL_NAME == "search_medical_web":
            impl = _limited(impl, max_web_searches)
        tools.append(
            to_beeai_tool(
                impl,
                name=mod.TOOL_NAME,
                description=mod.TOOL_DESCRIPTION,
                input_schema=mod.ToolInput,
            )
        )
    return tools
