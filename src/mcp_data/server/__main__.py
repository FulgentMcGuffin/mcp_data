"""Server entrypoint: dispatches to stdio or Streamable HTTP based on config."""

from __future__ import annotations

import sys

from mcp_data.config import get_settings


def main() -> None:
    settings = get_settings()

    if settings.transport == "stdio":
        from mcp_data.server.app import create_server

        mcp, backend = create_server(settings)
        try:
            mcp.run(transport="stdio")
        finally:
            backend.close()
        return

    # HTTP transport: serve the FastAPI host with uvicorn.
    import uvicorn

    from mcp_data.server.app import create_http_app

    app = create_http_app(settings)
    print(
        f"Serving MCP ({settings.server_name}) over Streamable HTTP at "
        f"{settings.http_url}",
        file=sys.stderr,
    )
    uvicorn.run(app, host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
