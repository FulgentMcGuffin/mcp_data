"""Agentic (multi-step) LLM query mode backed by LangGraph.

Where :class:`~mcp_data.client.planner.LLMPlanner` performs a single
tool-selection step, the agent here runs a full ReAct loop: it can introspect
the schema, run SQL, observe the results or errors, and self-correct over
several turns before producing a natural-language answer.

Tools are pulled live from the connected MCP session via
``langchain-mcp-adapters`` (:func:`load_mcp_tools`), so the agent always stays
in sync with whatever the server exposes -- no re-declared stubs.

Requires ``ANTHROPIC_API_KEY`` (or an equivalent for a substituted model).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent

from mcp_data.client.session import DBClient

DEFAULT_MODEL = "claude-sonnet-4-5"

_AGENT_SYSTEM_PROMPT = """\
You are an expert data analyst with access to tools for querying a database.

Work step by step:
1. Use the dataset description (provided below, if any) to understand the tables,
   columns, and the vocabulary that maps business terms to stored values/columns.
2. If you are unsure of the exact schema, call list_tables / get_schema.
3. Construct a single read-only SQL query and run it with run_sql.
4. If the query errors or returns unexpected results, inspect the schema and
   try again -- do not give up after one attempt.
5. When you have the answer, reply in clear natural language. Briefly state the
   SQL you ran so the user can verify it.

Only read-only SQL (SELECT / WITH / PRAGMA / EXPLAIN) is permitted.\
"""


def _build_model(model: object | None):
    if model is not None:
        return model
    from langchain_anthropic import ChatAnthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY is not set. "
            "Add it to your .env file or export it before using --llm."
        )
    return ChatAnthropic(model=DEFAULT_MODEL, api_key=api_key, temperature=0)


def _compose_system_prompt(profile_prompt: str | None) -> str:
    if profile_prompt:
        return f"{profile_prompt}\n\n{_AGENT_SYSTEM_PROMPT}"
    return _AGENT_SYSTEM_PROMPT


@dataclass
class AgentResult:
    """Outcome of an agent run."""

    answer: str
    messages: list


class SQLAgent:
    """A LangGraph ReAct agent that answers questions over the MCP database.

    The agent is bound to an *already-connected* :class:`DBClient`; its tools are
    loaded from that client's MCP session.
    """

    def __init__(
        self,
        client: DBClient,
        *,
        model: object | None = None,
        profile_prompt: str | None = None,
    ) -> None:
        self._client = client
        self._model = _build_model(model)
        self._profile_prompt = profile_prompt
        self._agent = None

    async def _ensure_agent(self):
        if self._agent is None:
            tools = await load_mcp_tools(self._client.session)
            self._agent = create_react_agent(
                self._model,
                tools,
                prompt=SystemMessage(
                    content=_compose_system_prompt(self._profile_prompt)
                ),
            )
        return self._agent

    async def run(self, query: str) -> AgentResult:
        """Run the agent on a single query and return its final answer."""
        agent = await self._ensure_agent()
        state = await agent.ainvoke(
            {"messages": [HumanMessage(content=query)]}
        )
        messages = state.get("messages", [])
        answer = ""
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                answer = _message_text(msg)
                if answer:
                    break
        return AgentResult(answer=answer, messages=messages)


def _message_text(message: AIMessage) -> str:
    """Extract plain text from an AIMessage (content may be a list of blocks)."""
    content = message.content
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "\n".join(p for p in parts if p).strip()
    return str(content).strip()


__all__ = ["SQLAgent", "AgentResult", "DEFAULT_MODEL"]
