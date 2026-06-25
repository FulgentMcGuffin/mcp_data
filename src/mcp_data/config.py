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
DEFAULT_SEMANTICS_DIR = PROJECT_ROOT / "semantics"

# Path component the Streamable HTTP transport is mounted under.
MCP_PATH = "/mcp"

# Load variables from .env and .secrets files at the project root if they exist.
# Missing files are ignored. Both files are treated as extensions of each other,
# with .secrets loaded second so it can override .env if needed (but OS environment
# variables take precedence over both, so explicit overrides keep working).
load_dotenv(PROJECT_ROOT / ".env", override=False)
load_dotenv(PROJECT_ROOT / ".secrets", override=False)


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
    dataset: str = DEFAULT_DB_PATH.stem
    semantics_dir: Path = DEFAULT_SEMANTICS_DIR

    @property
    def http_url(self) -> str:
        """Full URL a client should connect to for the HTTP transport."""
        return f"http://{self.host}:{self.port}{MCP_PATH}"

    @classmethod
    def from_env(cls) -> "Settings":
        db_path = Path(os.environ.get("MCP_DB_PATH", str(DEFAULT_DB_PATH)))
        # The dataset name keys the semantic profile; defaults to the DB file
        # stem (e.g. input_data.db -> "input_data"), overridable via MCP_DATASET.
        dataset = os.environ.get("MCP_DATASET", db_path.stem)
        return cls(
            transport=_env_transport(),
            db_path=db_path,
            host=os.environ.get("MCP_HOST", "127.0.0.1"),
            port=int(os.environ.get("MCP_PORT", "8000")),
            server_name=os.environ.get("MCP_SERVER_NAME", "db-mcp"),
            dataset=dataset,
            semantics_dir=Path(
                os.environ.get("MCP_SEMANTICS_DIR", str(DEFAULT_SEMANTICS_DIR))
            ),
        )


def get_settings() -> Settings:
    return Settings.from_env()
