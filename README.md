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
                                                          DataBackend (SQLiteSource now,
                                                          RedisBackend later)
```

Layout (`src/mcp_data/`):

| Module | Responsibility |
| --- | --- |
| `config.py` | Env-driven `Settings` (transport, db path, host, port) |
| `backends/base.py` | `DataBackend` (read) + `DataSink` (write) contracts + `is_read_only_sql` guard |
| `backends/sqlite_backend.py` | `SQLiteSource`: unified read/write SQLite store (`read_only` flag) |
| `backends/__init__.py` | `create_backend()` factory (future backend switch) |
| `data/seed.py` | Creates/seeds the example database |
| `pipeline/dataflow.py` | Hamilton nodes: validate -> execute -> frame -> serialize |
| `pipeline/runner.py` | Builds the Hamilton driver and runs the dataflow |
| `server/app.py` | FastMCP tools/resource + FastAPI HTTP host |
| `server/__main__.py` | Entrypoint dispatching on transport |
| `client/session.py` | Transport-agnostic `DBClient` (stdio + HTTP) |
| `client/planner.py` | `Planner` protocol + `RuleBasedPlanner` + `LLMPlanner` (Claude via LangChain) |
| `client/cli.py` | Interactive / one-shot CLI (`--llm` flag selects `LLMPlanner`) |

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
| `ANTHROPIC_API_KEY` | *(required for `--llm`)* | Anthropic API key for the LLM planner |

Variables are loaded from a `.env` file at the project root (if present) via `python-dotenv`. OS environment variables take precedence over `.env` values.

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

### Rule-based planner (default)

Fixed keyword syntax, no API key required:

```
tables                 list tables
schema <table>         show a table's schema
sql: <query>           run a read-only SQL query
help                   show help
quit / exit            leave
```

```bash
# one-shot
uv run db-mcp-client "schema customers"
uv run db-mcp-client "sql: select * from customers limit 5"

# interactive REPL
uv run db-mcp-client
```

### LLM planner (`--llm`)

Free-form natural language translated to tool calls by **Claude claude-sonnet-4-5** via
**LangChain** (`langchain-anthropic`). Requires `ANTHROPIC_API_KEY`.

```bash
# one-shot natural language questions
uv run db-mcp-client --llm "what tables are in this database?"
uv run db-mcp-client --llm "show me the schema of the customers table"
uv run db-mcp-client --llm "how many customers are there per country?"
uv run db-mcp-client --llm "list the top 3 orders by amount"

# interactive REPL with LLM
uv run db-mcp-client --llm
```

The LLM model is swappable — pass any LangChain chat model directly in code:

```python
from langchain_openai import ChatOpenAI
from mcp_data.client.planner import LLMPlanner

planner = LLMPlanner(model=ChatOpenAI(model="gpt-4o"))
```

## Safety

`run_sql` accepts only a single read-only statement (`SELECT` / `WITH` / `PRAGMA` /
`EXPLAIN`); multi-statement and mutating SQL is rejected, and the SQLite connection is
opened read-only.

## Tests

```bash
uv run pytest -q
```

## Roadmap

- ✅ **LLM natural-language planner**: `LLMPlanner` (Claude claude-sonnet-4-5 via LangChain)
  translates free-form questions into tool calls. Toggle with `--llm`; swap the model
  by passing any LangChain chat model to `LLMPlanner(model=...)`.
- **Redis cache backend**: add `RedisBackend(DataBackend)` and select it in
  `create_backend()`; tools, pipeline and client are unchanged. A cache node can slot
  into the Hamilton dataflow between `validated_sql` and `result_frame`.
- **Remote + OAuth**: add auth middleware/routes to the FastAPI host in
  `server/app.py`; the HTTP client gains an OAuth provider.
