"""ML Orchestrator Agent — professional ML researcher sub-agent.

Called by the Chat Agent's `consult_ml_orchestrator` tool. Receives a
natural-language request (which conditions to predict + patient feature
values), routes to the appropriate ML MCP tools, runs predictions, and
returns structured results with SHAP-based interpretation.
"""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, AsyncIterator

if TYPE_CHECKING:
    from beeai_framework.backend.chat import ChatModel

_SYSTEM_PROMPT_PATH = Path(__file__).parent / "system_prompt.md"


def _result_text(result: Any) -> str:
    structured = getattr(result, "output_structured", None)
    text = getattr(structured, "response", None)
    if text:
        return text
    answer = getattr(result, "answer", None)
    text = getattr(answer, "text", None)
    if text:
        return text
    return str(result)


@asynccontextmanager
async def ml_orchestrator_agent(llm: "ChatModel | None" = None) -> AsyncIterator[Any]:
    """Yield a fully wired ML Orchestrator agent backed by the ML MCP server."""
    from beeai_framework.agents.requirement import RequirementAgent
    from beeai_framework.memory import UnconstrainedMemory
    from beeai_framework.tools.mcp import MCPTool
    from fastmcp import Client

    from packages.llm_provider.client import build_chat_model_for_role
    from packages.llm_provider.settings import Role
    from services.ml_mcp_server.server import build_server

    if llm is None:
        llm = build_chat_model_for_role(Role.ML_ORCHESTRATOR)

    ml_server = build_server()
    async with Client(ml_server) as client:
        ml_tools = await MCPTool.from_client(client.session)

        agent = RequirementAgent(
            llm=llm,
            memory=UnconstrainedMemory(),
            tools=ml_tools,
            requirements=[],
            name="ML Orchestrator",
            description=(
                "Professional ML researcher: routes patient features to "
                "clinical ML models and interprets predictions with SHAP."
            ),
            instructions=_SYSTEM_PROMPT_PATH.read_text(encoding="utf-8"),
            final_answer_as_tool=False,
        )
        yield agent
