# mcp-data

A generic **Model Context Protocol (MCP) server and client** for querying databases.
It starts with a local **SQLite** backend behind a generic `DataBackend` abstraction so
the storage layer can later be swapped (e.g. a **Redis** cache) without touching the
server tools or client. Transport is selectable between **stdio** and **Streamable HTTP**
(hosted via FastAPI, ready for OAuth in the remote phase).

- Package/dependency management: **uv**
- DataFrames: **polars** (never pandas)
- Query pipeline: **Apache Hamilton**
- Server framework: **FastMCP** (official `mcp` SDK) + **FastAPI** for the HTTP host

## Architecture

```
client (CLI REPL) -> Planner -> MCP ClientSession --(stdio | HTTP)--> FastMCP server
                                                                         |
                                                            tools: list_tables,
                                                            get_schema, run_sql
                                                                         |
                                                          Hamilton dataflow (validate
                                                          -> execute -> polars -> JSON)
                                                                         |
                                                          DataBackend (SQLiteBackend now,
                                                          RedisBackend later)
```

Layout (`src/mcp_sqlite/`):

| Module | Responsibility |
| --- | --- |
| `config.py` | Env-driven `Settings` (transport, db path, host, port) |
| `backends/base.py` | `DataBackend` protocol + `is_read_only_sql` guard |
| `backends/sqlite_backend.py` | Read-only SQLite backend producing polars frames |
| `backends/__init__.py` | `create_backend()` factory (future backend switch) |
| `data/seed.py` | Creates/seeds the example database |
| `pipeline/dataflow.py` | Hamilton nodes: validate -> execute -> frame -> serialize |
| `pipeline/runner.py` | Builds the Hamilton driver and runs the dataflow |
| `server/app.py` | FastMCP tools/resource + FastAPI HTTP host |
| `server/__main__.py` | Entrypoint dispatching on transport |
| `client/session.py` | Transport-agnostic `DBClient` (stdio + HTTP) |
| `client/planner.py` | `Planner` protocol + `RuleBasedPlanner` (LLM drop-in later) |
| `client/cli.py` | Interactive / one-shot CLI |

## Setup

```bash
uv sync
uv run db-mcp-seed        # create the example SQLite database
```

## Configuration (environment variables)

| Variable | Default | Meaning |
| --- | --- | --- |
| `MCP_TRANSPORT` | `stdio` | `stdio` or `http` |
| `MCP_DB_PATH` | `data/example.db` | SQLite file to query |
| `MCP_HOST` | `127.0.0.1` | HTTP host |
| `MCP_PORT` | `8000` | HTTP port |
| `MCP_SERVER_NAME` | `db-mcp` | Server display name |

## Running

### stdio (client spawns the server)

```bash
# one-shot
MCP_TRANSPORT=stdio uv run db-mcp-client "tables"
MCP_TRANSPORT=stdio uv run db-mcp-client "schema customers"
MCP_TRANSPORT=stdio uv run db-mcp-client "sql: select * from customers limit 5"

# interactive REPL
MCP_TRANSPORT=stdio uv run db-mcp-client
```

On PowerShell, set env vars first, e.g. `$env:MCP_TRANSPORT="stdio"`.

### Streamable HTTP (separate server + client)

```bash
# terminal 1
MCP_TRANSPORT=http uv run db-mcp-server      # serves http://127.0.0.1:8000/mcp

# terminal 2
MCP_TRANSPORT=http uv run db-mcp-client "tables"
```

Health check: `GET http://127.0.0.1:8000/healthz`.

## CLI commands

```
tables                 list tables
schema <table>         show a table's schema
sql: <query>           run a read-only SQL query
help                   show help
quit / exit            leave
```

## Safety

`run_sql` accepts only a single read-only statement (`SELECT` / `WITH` / `PRAGMA` /
`EXPLAIN`); multi-statement and mutating SQL is rejected, and the SQLite connection is
opened read-only.

## Tests

```bash
uv run pytest -q
```

## Roadmap (designed for, not yet built)

- **Redis cache backend**: add `RedisBackend(DataBackend)` and select it in
  `create_backend()`; tools, pipeline and client are unchanged. A cache node can slot
  into the Hamilton dataflow between `validated_sql` and `result_frame`.
- **LLM natural-language-to-SQL**: implement `LLMPlanner(Planner)` in the client; the
  CLI and session layer stay the same.
- **Remote + OAuth**: add auth middleware/routes to the FastAPI host in
  `server/app.py`; the HTTP client gains an OAuth provider.
