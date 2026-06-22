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

from mcp_data.backends import QueryError, create_backend
from mcp_data.backends.base import DataBackend
from mcp_data.config import Settings, get_settings
from mcp_data.pipeline import run_sql_pipeline
from mcp_data.semantics import (
    ColumnSemantics,
    SemanticProfile,
    TableSemantics,
    load_profile,
    render_profile_prompt,
)


def _build_dataset_description(
    backend: DataBackend, settings: Settings
) -> dict[str, Any]:
    """Return the semantic description of the dataset for LLM consumption.

    Loads the curated YAML profile if one exists for ``settings.dataset``;
    otherwise synthesises a minimal profile from live schema introspection so
    clients always get a usable (if sparse) context block.
    """
    profile = load_profile(settings.dataset, settings.semantics_dir)
    if profile is None:
        tables: list[TableSemantics] = []
        for table in backend.list_tables():
            schema = backend.get_schema(table)
            columns = [
                ColumnSemantics(name=col.name, type=col.type, description="")
                for col in schema.columns
            ]
            tables.append(TableSemantics(name=table, columns=columns))
        profile = SemanticProfile(
            dataset=settings.dataset,
            backend=backend.name,
            description="(No curated semantic profile; schema introspected live.)",
            tables=tables,
        )

    return {
        "profile": profile.to_dict(),
        "prompt": render_profile_prompt(profile),
        "has_curated_profile": load_profile(settings.dataset, settings.semantics_dir)
        is not None,
    }


def _register_tools(mcp: FastMCP, backend: DataBackend, settings: Settings) -> None:
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

    @mcp.tool()
    def describe_dataset() -> dict[str, Any]:
        """Return the semantic profile (business meaning) of this dataset.

        Includes table/column descriptions, a controlled vocabulary mapping
        business terms to stored values/columns (e.g. country names to source
        codes), conventions, and example natural-language -> SQL pairs. Use this
        to understand the data before writing SQL.
        """
        return _build_dataset_description(backend, settings)

    @mcp.resource("schema://{table}")
    def schema_resource(table: str) -> dict[str, Any]:
        """Expose a table schema as an addressable MCP resource."""
        try:
            return backend.get_schema(table).to_dict()
        except QueryError as exc:
            return {"error": str(exc)}

    @mcp.resource("semantics://dataset")
    def semantics_resource() -> dict[str, Any]:
        """Expose the dataset semantic profile as an addressable MCP resource."""
        return _build_dataset_description(backend, settings)


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
    _register_tools(mcp, backend, settings)
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
