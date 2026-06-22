"""Interactive CLI that accepts user queries and runs them via the MCP server.

Usage::

    db-mcp-client                 # interactive REPL
    db-mcp-client "sql: select 1" # one-shot query

Transport (stdio vs HTTP) and the database come from the same environment
settings as the server.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from typing import TYPE_CHECKING, Any

import polars as pl

if TYPE_CHECKING:
    from mcp_data.client.agent import SQLAgent

from mcp_data.client.planner import HELP_TEXT, LLMPlanner, Planner, RuleBasedPlanner, ToolCall
from mcp_data.client.session import DBClient
from mcp_data.config import get_settings


def _ensure_utf8_stdout() -> None:
    # polars table rendering uses box-drawing glyphs that crash on legacy
    # Windows code pages; force UTF-8 so results print cleanly.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8")
            except Exception:
                pass


def _render(result: Any) -> str:
    if isinstance(result, dict) and "error" in result:
        return f"Error: {result['error']}"

    # run_sql result
    if isinstance(result, dict) and "rows" in result:
        columns = result.get("columns", [])
        rows = result.get("rows", [])
        if not rows:
            return f"(0 rows) columns={columns}"
        df = pl.DataFrame(rows)
        return f"{df}\n({result.get('row_count', len(rows))} rows)"

    # get_schema result
    if isinstance(result, dict) and "columns" in result and "table" in result:
        df = pl.DataFrame(result["columns"])
        return f"Schema for {result['table']}:\n{df}"

    if isinstance(result, list):
        return "\n".join(f"- {item}" for item in result) or "(none)"

    return str(result)


async def _run_calls(client: DBClient, calls: list[ToolCall]) -> None:
    tools = await client.list_tools()
    for call in calls:
        if call.name not in tools:
            print(f"Tool {call.name!r} not available (have: {tools})")
            continue
        result = await client.call_tool(call.name, call.arguments)
        print(_render(result))


async def _interactive(client: DBClient, planner: Planner) -> None:
    print(HELP_TEXT)
    print()
    loop = asyncio.get_event_loop()
    while True:
        try:
            query = await loop.run_in_executor(None, input, "db> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        query = query.strip()
        if not query:
            continue
        if query.lower() in ("quit", "exit"):
            break
        if query.lower() == "help":
            print(HELP_TEXT)
            continue
        calls = planner.plan(query, await client.list_tools())
        if not calls:
            print("Could not interpret that. Type 'help' for commands.")
            continue
        await _run_calls(client, calls)


async def _interactive_agent(agent: "SQLAgent") -> None:
    print("Agentic LLM mode. Ask a question in natural language; type 'quit' to leave.")
    print()
    loop = asyncio.get_event_loop()
    while True:
        try:
            query = await loop.run_in_executor(None, input, "db> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        query = query.strip()
        if not query:
            continue
        if query.lower() in ("quit", "exit"):
            break
        result = await agent.run(query)
        print(result.answer or "(no answer)")


async def _amain(one_shot: str | None, mode: str) -> None:
    settings = get_settings()
    async with DBClient(settings) as client:
        # Fetch the dataset semantic profile once so both LLM modes get domain
        # context (table/column meaning, vocabulary, examples).
        profile_prompt: str | None = None
        if mode in ("agent", "single_shot"):
            description = await client.describe_dataset()
            profile_prompt = description.get("prompt") or None

        if mode == "agent":
            from mcp_data.client.agent import SQLAgent

            agent = SQLAgent(client, profile_prompt=profile_prompt)
            if one_shot:
                result = await agent.run(one_shot)
                print(result.answer or "(no answer)")
            else:
                await _interactive_agent(agent)
            return

        if mode == "single_shot":
            planner: Planner = LLMPlanner(profile_prompt=profile_prompt)
        else:
            planner = RuleBasedPlanner()

        if one_shot:
            calls = planner.plan(one_shot, await client.list_tools())
            if not calls:
                print("Could not interpret that. Type 'help' for commands.")
                return
            await _run_calls(client, calls)
        else:
            await _interactive(client, planner)


def main() -> None:
    _ensure_utf8_stdout()
    parser = argparse.ArgumentParser(description="MCP database client.")
    parser.add_argument(
        "query",
        nargs="?",
        default=None,
        help="Optional one-shot query (e.g. \"sql: select * from customers\").",
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        default=False,
        help=(
            "Use the agentic LLM mode: a LangGraph ReAct agent (Claude "
            "claude-sonnet-4-5 via LangChain) that introspects the schema, runs "
            "SQL, and self-corrects over multiple steps. Requires "
            "ANTHROPIC_API_KEY to be set."
        ),
    )
    parser.add_argument(
        "--llm-single-shot",
        action="store_true",
        default=False,
        help=(
            "Use the single-shot LLM planner instead of the agentic loop: the "
            "model picks tool calls in one step. Requires ANTHROPIC_API_KEY."
        ),
    )
    args = parser.parse_args()
    if args.llm and args.llm_single_shot:
        parser.error("Use only one of --llm / --llm-single-shot.")
    if args.llm:
        mode = "agent"
    elif args.llm_single_shot:
        mode = "single_shot"
    else:
        mode = "rule"
    asyncio.run(_amain(args.query, mode=mode))


if __name__ == "__main__":
    main()
