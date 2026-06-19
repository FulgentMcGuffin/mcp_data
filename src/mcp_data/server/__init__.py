"""MCP server: exposes database tools over stdio or Streamable HTTP."""

from mcp_sqlite.server.app import create_http_app, create_server

__all__ = ["create_server", "create_http_app"]
