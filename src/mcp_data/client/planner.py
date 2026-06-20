"""Turn a user's query into a sequence of MCP tool calls.

The :class:`Planner` protocol is the seam where a future LLM-driven natural
language to SQL layer plugs in: an ``LLMPlanner`` would implement the same
``plan`` method by asking a model to choose tools/generate SQL from the tool
list. For now :class:`RuleBasedPlanner` maps a few simple commands.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


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
    "Commands:\n"
    "  tables                 -> list tables\n"
    "  schema <table>         -> show a table's schema\n"
    "  sql: <query>           -> run a read-only SQL query\n"
    "  help                   -> show this help\n"
    "  quit / exit            -> leave"
)


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
