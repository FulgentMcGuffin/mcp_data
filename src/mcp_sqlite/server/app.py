"""FastMCP server definition and FastAPI host for the HTTP transport.

The same set of tools is exposed regardless of transport. For ``stdio`` the
``FastMCP`` instance is run directly; for ``http`` the FastMCP Streamable HTTP
ASGI app is mounted inside a FastAPI application, which is where OAuth
middleware/routes will be added in the remote phase.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from mcp.server.fastmcp import FastMCP

from mcp_sqlite.backends import QueryError, create_backend
from mcp_sqlite.backends.base import DataBackend
from mcp_sqlite.config import Settings, get_settings
from mcp_sqlite.pipeline import run_sql_pipeline


def _register_tools(mcp: FastMCP, backend: DataBackend) -> None:
    """Attach the database tools/resources, closing over ``backend``."""

    @mcp.tool()
    def list_tables() -> list[str]:
        """List the names of all queryable tables in the database."""
        return backend.list_tables()

    @mcp.tool()
    def get_schema(table: str) -> dict[str, Any]:
        """Return the column schema (name, type, nullability, PK) for a table."""
        try:
            return backend.get_schema(table).to_dict()
        except QueryError as exc:
            return {"error": str(exc)}

    @mcp.tool()
    def run_sql(sql: str) -> dict[str, Any]:
        """Run a read-only SQL query (SELECT/WITH/PRAGMA/EXPLAIN) and return rows.

        Returns a dict with ``columns``, ``dtypes``, ``row_count`` and ``rows``.
        Mutating or multi-statement SQL is rejected.
        """
        try:
            return run_sql_pipeline(backend, sql)
        except QueryError as exc:
            return {"error": str(exc)}

    @mcp.resource("schema://{table}")
    def schema_resource(table: str) -> dict[str, Any]:
        """Expose a table schema as an addressable MCP resource."""
        try:
            return backend.get_schema(table).to_dict()
        except QueryError as exc:
            return {"error": str(exc)}


def create_server(settings: Settings | None = None) -> tuple[FastMCP, DataBackend]:
    """Build a configured FastMCP server and its backend."""

    settings = settings or get_settings()
    backend = create_backend(settings)
    mcp = FastMCP(
        settings.server_name,
        host=settings.host,
        port=settings.port,
        json_response=True,
        stateless_http=True,
    )
    _register_tools(mcp, backend)
    return mcp, backend


def create_http_app(settings: Settings | None = None) -> FastAPI:
    """Build a FastAPI app hosting the MCP server over Streamable HTTP."""

    settings = settings or get_settings()
    mcp, backend = create_server(settings)

    # Must be created before the session manager is accessed in the lifespan.
    mcp_app = mcp.streamable_http_app()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async with mcp.session_manager.run():
            try:
                yield
            finally:
                backend.close()

    app = FastAPI(title=f"{settings.server_name} (MCP over HTTP)", lifespan=lifespan)

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok", "backend": backend.name}

    # Mount the MCP ASGI app; its route lives at settings.transport path "/mcp".
    app.mount("/", mcp_app)
    return app
