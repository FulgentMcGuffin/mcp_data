"""Environment-driven configuration for the MCP server and client.

Keeping all settings in one place makes it easy to flip between transports
(``stdio`` vs Streamable HTTP) and to point at a different database without
code changes. The same ``Settings`` object is consumed by both the server and
the client so they stay in sync.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

Transport = Literal["stdio", "http"]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "example.db"

# Path component the Streamable HTTP transport is mounted under.
MCP_PATH = "/mcp"

# Load variables from a .env file at the project root if it exists. Missing
# files are ignored, and existing OS environment variables take precedence so
# that explicit overrides keep working.
load_dotenv(PROJECT_ROOT / ".env", override=False)


def _env_transport(default: Transport = "stdio") -> Transport:
    value = os.environ.get("MCP_TRANSPORT", default).strip().lower()
    if value in ("stdio", "http"):
        return value  # type: ignore[return-value]
    raise ValueError(
        f"Invalid MCP_TRANSPORT={value!r}; expected 'stdio' or 'http'."
    )


@dataclass(frozen=True)
class Settings:
    """Resolved runtime settings."""

    transport: Transport = "stdio"
    db_path: Path = DEFAULT_DB_PATH
    host: str = "127.0.0.1"
    port: int = 8000
    server_name: str = "db-mcp"

    @property
    def http_url(self) -> str:
        """Full URL a client should connect to for the HTTP transport."""
        return f"http://{self.host}:{self.port}{MCP_PATH}"

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            transport=_env_transport(),
            db_path=Path(os.environ.get("MCP_DB_PATH", str(DEFAULT_DB_PATH))),
            host=os.environ.get("MCP_HOST", "127.0.0.1"),
            port=int(os.environ.get("MCP_PORT", "8000")),
            server_name=os.environ.get("MCP_SERVER_NAME", "db-mcp"),
        )


def get_settings() -> Settings:
    return Settings.from_env()
