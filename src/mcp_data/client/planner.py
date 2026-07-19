"""Turn a user's query into a sequence of MCP tool calls.

Two planners are provided:

* :class:`RuleBasedPlanner` – deterministic keyword parser, no LLM required.
* :class:`LLMPlanner` – uses LangChain + Claude claude-sonnet-4-5 to translate free-form
  natural language into tool calls.  Requires ``ANTHROPIC_API_KEY`` to be set.
  The LangChain abstraction makes it straightforward to swap in any other
  LangChain-supported chat model (GPT-4o, Gemini, …) by changing one line.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from mcp_data.client import _tracing  # noqa: F401 — disable LangSmith before LangChain

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool as lc_tool


@dataclass(frozen=True)
class ToolCall:
    """A single tool invocation the client should perform."""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Planner(Protocol):
    """Maps a free-form user query to an ordered list of tool calls."""

    def plan(self, query: str, available_tools: list[str]) -> list[ToolCall]:
        ...


HELP_TEXT = (
    "Commands (rule-based mode):\n"
    "  tables                 -> list tables\n"
    "  schema <table>         -> show a table's schema\n"
    "  sql: <query>           -> run a read-only SQL query\n"
    "  help                   -> show this help\n"
    "  quit / exit            -> leave\n"
    "\n"
    "In --llm mode any natural-language question is accepted."
)


# ---------------------------------------------------------------------------
# Tool schemas shared between planners
# ---------------------------------------------------------------------------

# These LangChain tool stubs are never actually *called*; they exist solely to
# give the model an accurate JSON schema so it can emit well-formed tool-use
# blocks. The real execution happens via the MCP session in cli.py.

@lc_tool
def list_tables() -> list[str]:
    """List the names of all queryable tables in the database."""
    ...  # pragma: no cover


@lc_tool
def get_schema(table: str) -> dict:
    """Return the column schema (name, type, nullability, primary key) for a table.

    Args:
        table: Name of the table whose schema should be returned.
    """
    ...  # pragma: no cover


@lc_tool
def run_sql(sql: str) -> dict:
    """Execute a read-only SQL query (SELECT / WITH / PRAGMA / EXPLAIN) and return rows.

    Only single, read-only statements are allowed. Mutating or multi-statement
    SQL is rejected by the server.

    Args:
        sql: The SQL statement to execute.
    """
    ...  # pragma: no cover


_ALL_LC_TOOLS = {t.name: t for t in [list_tables, get_schema, run_sql]}

_SYSTEM_PROMPT = """\
You are a helpful database assistant. You have access to a set of tools that \
let you query a SQLite database. When the user asks a question, decide which \
tool(s) to call (in order) to answer it. You may call multiple tools if needed \
— for example, first list tables to discover the schema, then run a query.

Always prefer running SQL directly when the question can be answered with a \
single query. Only call list_tables or get_schema when you genuinely need that \
information to construct the SQL.

Reply ONLY with tool calls — do not add any prose before or after them.\
"""


# ---------------------------------------------------------------------------
# Planners
# ---------------------------------------------------------------------------

class RuleBasedPlanner(Planner):
    """Deterministic command parser (no LLM required)."""

    def plan(self, query: str, available_tools: list[str]) -> list[ToolCall]:
        calls = self._parse(query)
        # Only emit calls for tools the server actually exposes. This keeps the
        # Planner contract honest (callers can trust every returned call is
        # runnable) and guards future planners (e.g. an LLM-driven one) from
        # inventing tool names. The CLI still reports availability per call.
        return [call for call in calls if call.name in available_tools]

    def _parse(self, query: str) -> list[ToolCall]:
        text = query.strip()
        lowered = text.lower()

        if lowered in ("tables", "list tables"):
            return [ToolCall("list_tables")]

        if lowered.startswith("schema"):
            parts = text.split(None, 1)
            if len(parts) == 2:
                return [ToolCall("get_schema", {"table": parts[1].strip()})]
            return []

        if lowered.startswith("sql:"):
            sql = text[len("sql:"):].strip()
            if sql:
                return [ToolCall("run_sql", {"sql": sql})]
            return []

        # Unrecognized input: no tool calls (CLI prints help).
        return []


class LLMPlanner(Planner):
    """LangChain-backed planner that translates natural language into tool calls.

    Uses Claude claude-sonnet-4-5 by default via ``langchain-anthropic``.  Any other
    LangChain chat model that supports tool-calling can be substituted::

        from langchain_openai import ChatOpenAI
        planner = LLMPlanner(model=ChatOpenAI(model="gpt-4o"))

    Requires the ``ANTHROPIC_API_KEY`` environment variable (or equivalent for
    other providers) to be set.
    """

    DEFAULT_MODEL = "claude-sonnet-4-5"

    def __init__(
        self,
        model=None,
        profile_prompt: str | None = None,
        extra_tools: list | None = None,
    ) -> None:
        if model is None:
            # Imported here so the rest of the module loads without the SDK.
            from langchain_anthropic import ChatAnthropic

            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError(
                    "ANTHROPIC_API_KEY is not set. "
                    "Add it to your .env file or export it before using --llm."
                )
            model = ChatAnthropic(
                model=self.DEFAULT_MODEL,
                api_key=api_key,
                temperature=0,
            )
        self._model = model
        self._profile_prompt = profile_prompt
        self._extra_tools = list(extra_tools) if extra_tools else []

    @property
    def system_prompt(self) -> str:
        """The system prompt, with the dataset semantic profile prepended."""
        if self._profile_prompt:
            return (
                f"{self._profile_prompt}\n\n"
                "Use the dataset description above (table/column meaning, the "
                "vocabulary mapping business terms to stored values/columns, and "
                "the example queries) to build correct SQL.\n\n"
                f"{_SYSTEM_PROMPT}"
            )
        return _SYSTEM_PROMPT

    def plan(self, query: str, available_tools: list[str]) -> list[ToolCall]:
        # Restrict built-in SQL tools to what the server actually exposes;
        # caller-supplied extra_tools (e.g. custom Python-side actions) are
        # always offered regardless of the server's tool list.
        extra_by_name = {t.name: t for t in self._extra_tools}
        lc_tools = [
            t for name, t in _ALL_LC_TOOLS.items() if name in available_tools
        ]
        lc_tools.extend(extra_by_name.values())
        if not lc_tools:
            return []

        bound = self._model.bind_tools(lc_tools, tool_choice="any")
        response = bound.invoke(
            [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=query),
            ]
        )

        allowed = set(available_tools) | set(extra_by_name)
        calls: list[ToolCall] = []
        for tc in response.tool_calls:
            name = tc["name"]
            if name in allowed:
                calls.append(ToolCall(name=name, arguments=tc.get("args", {})))
        return calls
