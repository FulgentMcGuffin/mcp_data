"""Transport-agnostic MCP client session wrapper.

Connects either by spawning the server over ``stdio`` or by talking to a running
Streamable HTTP server, selected from :class:`Settings`. The rest of the client
(planner, CLI) only sees :meth:`DBClient.list_tools` and
:meth:`DBClient.call_tool`, so swapping transports (or adding OAuth to the HTTP
path later) does not ripple outward.
"""

from __future__ import annotations

import os
import sys
from contextlib import AsyncExitStack
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client

from mcp_data.config import Settings, get_settings


class DBClient:
    """Async context manager wrapping an MCP ``ClientSession``."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._stack = AsyncExitStack[bool | None]()
        self._session: ClientSession | None = None

    @property
    def session(self) -> ClientSession:
        if self._session is None:
            raise RuntimeError("DBClient is not connected; use 'async with'.")
        return self._session

    async def __aenter__(self) -> "DBClient":
        if self._settings.transport == "http":
            read, write, _ = await self._stack.enter_async_context(
                streamablehttp_client(self._settings.http_url)
            )
        else:
            # Spawn the server as a subprocess in stdio mode.
            env = dict(os.environ)
            env["MCP_TRANSPORT"] = "stdio"
            env["MCP_DB_PATH"] = str(self._settings.db_path)
            params = StdioServerParameters(
                command=sys.executable,
                args=["-m", "mcp_data.server"],
                env=env,
            )
            read, write = await self._stack.enter_async_context(
                stdio_client(params)
            )

        self._session = await self._stack.enter_async_context(
            ClientSession(read, write)
        )
        await self._session.initialize()
        return self

    async def __aexit__(self, *exc: object) -> None:
        self._session = None
        await self._stack.aclose()

    async def list_tools(self) -> list[str]:
        result = await self.session.list_tools()
        return [tool.name for tool in result.tools]

    async def describe_dataset(self) -> dict[str, Any]:
        """Fetch the dataset semantic profile from the server.

        Returns a dict with ``profile`` (the structured profile),
        ``prompt`` (a rendered LLM-friendly text block) and
        ``has_curated_profile``. Returns an empty dict if the server does not
        expose the ``describe_dataset`` tool.
        """
        if "describe_dataset" not in await self.list_tools():
            return {}
        result = await self.call_tool("describe_dataset", {})
        return result if isinstance(result, dict) else {}

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """Call a tool and return its structured result.

        Prefers ``structuredContent`` (our tools return dicts/lists); falls back
        to concatenated text content.
        """
        result = await self.session.call_tool(name, arguments)
        if result.structuredContent is not None:
            # FastMCP wraps non-object returns (e.g. lists) under "result".
            structured = result.structuredContent
            if isinstance(structured, dict) and set(structured.keys()) == {"result"}:
                return structured["result"]
            return structured
        texts = [
            block.text
            for block in result.content
            if getattr(block, "type", None) == "text"
        ]
        return "\n".join(texts)
