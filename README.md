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
client (CLI REPL) -> Planner / Agent -> MCP ClientSession --(stdio | HTTP)--> FastMCP server
                                                                                 |
                                                            tools: list_tables, get_schema,
                                                            run_sql, describe_dataset
                                                                                 |
                                                          Hamilton dataflow (validate
                                                          -> execute -> polars -> JSON)
                                                                                 |
                                                          DataBackend (SQLiteSource now,
                                                          RedisBackend later)
                                                                                 |
                                                          semantic profile (semantics/<dataset>.yaml)
                                                          served via describe_dataset
```

A per-dataset **semantic layer** (see [Semantic layer](#semantic-layer)) is co-located
with the data source and served by the MCP server, giving the LLM the business meaning of
tables/columns, a vocabulary mapping domain terms to stored values, and example queries.

Layout (`src/mcp_data/`):

| Module | Responsibility |
| --- | --- |
| `config.py` | Env-driven `Settings` (transport, db path, host, port, dataset, semantics dir) |
| `backends/base.py` | `DataBackend` (read) + `DataSink` (write) contracts + `is_read_only_sql` guard |
| `backends/sqlite_backend.py` | `SQLiteSource`: unified read/write SQLite store (`read_only` flag) |
| `backends/__init__.py` | `create_backend()` factory (future backend switch) |
| `semantics/__init__.py` | `SemanticProfile`, `load_profile`, `render_profile_prompt` (the semantic layer) |
| `data/seed.py` | Creates/seeds the example database |
| `pipeline/dataflow.py` | Hamilton nodes: validate -> execute -> frame -> serialize |
| `pipeline/runner.py` | Builds the Hamilton driver and runs the dataflow |
| `server/app.py` | FastMCP tools/resources (incl. `describe_dataset`) + FastAPI HTTP host |
| `server/__main__.py` | Entrypoint dispatching on transport |
| `client/session.py` | Transport-agnostic `DBClient` (stdio + HTTP) + `describe_dataset()` |
| `client/planner.py` | `Planner` protocol + `RuleBasedPlanner` + `LLMPlanner` (Claude via LangChain) |
| `client/agent.py` | `SQLAgent`: LangGraph ReAct agent over live MCP tools (agentic LLM mode) |
| `client/cli.py` | Interactive / one-shot CLI (`--llm` agentic, `--llm-single-shot` planner) |

Profiles live under `semantics/` at the project root (e.g. `semantics/input_data.yaml`,
`semantics/example.yaml`).

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
| `MCP_DATASET` | *(db file stem)* | Dataset name keying the semantic profile (`semantics/<dataset>.yaml`) |
| `MCP_SEMANTICS_DIR` | `semantics/` | Directory holding semantic profile YAML files |
| `ANTHROPIC_API_KEY` | *(required for `--llm`)* | Anthropic API key for the LLM modes |

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

### Agentic LLM mode (`--llm`)

The default LLM mode: a **LangGraph ReAct agent** (`SQLAgent`) driven by **Claude
claude-sonnet-4-5** via **LangChain**. It runs a multi-step loop — introspect the schema,
run SQL, observe the results or errors, and self-correct — before replying in natural
language with the SQL it used. Its tools are loaded live from the MCP session
(`langchain-mcp-adapters`), and the dataset's semantic profile is injected into its context
so it understands domain terms. Requires `ANTHROPIC_API_KEY`.

```bash
# one-shot natural language questions
uv run db-mcp-client --llm "how many customers are there per country?"
uv run db-mcp-client --llm "what is the highest difference between the 5Y and 10Y zero rate for Italy between 2010 and 2015?"

# interactive REPL (agentic)
uv run db-mcp-client --llm
```

### Single-shot LLM planner (`--llm-single-shot`)

A lighter mode: the model picks tool calls in a single step (no self-correction loop),
also with the semantic profile injected. Useful when you want one round-trip.

```bash
uv run db-mcp-client --llm-single-shot "show me the schema of the customers table"
uv run db-mcp-client --llm-single-shot "list the top 3 orders by amount"
```

The LLM model is swappable — pass any LangChain chat model directly in code:

```python
from langchain_openai import ChatOpenAI
from mcp_data.client.planner import LLMPlanner

planner = LLMPlanner(model=ChatOpenAI(model="gpt-4o"))
```

## Semantic layer

Each dataset can have a declarative **semantic profile** — a data dictionary describing
table/column meaning, a controlled vocabulary (business term → stored value/column), date
and naming conventions, and curated natural-language → SQL examples. This is the knowledge
the LLM uses to translate domain language (e.g. *"Italy"*, *"5Y"*) into correct SQL.

Profiles are YAML files named `semantics/<dataset>.yaml`, where `<dataset>` defaults to the
database file stem (override with `MCP_DATASET`). The MCP server exposes them via the
`describe_dataset` tool and the `semantics://dataset` resource; if no profile exists, the
server synthesises a minimal one from live schema introspection.

```yaml
dataset: input_data
backend: sqlite
description: Yield-curve and FX time series (zero rates, par rates, spot FX).
vocabulary:
  countries:          # values for the `source` column
    Italy: ITA
    Germany: DEU
  tenors:             # natural-language tenor -> column name
    "5Y": Y005p0
    "10Y": Y010p0
conventions:
  date: "TEXT in 'YYYY-MM-DD'; filter with date >= '<start>' AND date <= '<end>'"
tables:
  zero_rates:
    description: Zero-coupon spot rates; one row per (source, date).
    columns:
      date:   {type: TEXT, description: "Observation date (YYYY-MM-DD)."}
      source: {type: TEXT, description: "Curve/country code, e.g. ITA=Italy."}
      Y005p0: {type: REAL, description: "5-year zero rate."}
      Y010p0: {type: REAL, description: "10-year zero rate."}
examples:
  - q: "highest absolute difference in zero rate between 5Y and 10Y for Italy between 2010 and 2015"
    sql: "SELECT MAX(ABS(Y010p0 - Y005p0)) FROM zero_rates WHERE source='ITA' AND date >= '2010-01-01' AND date <= '2015-12-31'"
```

Shipped profiles: `semantics/input_data.yaml` (the real yield-curve dataset) and
`semantics/example.yaml` (the seeded demo database). Verify the `source` codes in
`input_data.yaml` against your actual database (`SELECT DISTINCT source FROM zero_rates`).

## Safety

`run_sql` accepts only a single read-only statement (`SELECT` / `WITH` / `PRAGMA` /
`EXPLAIN`); multi-statement and mutating SQL is rejected, and the SQLite connection is
opened read-only.

## Tests

```bash
uv run pytest -q
```

## Roadmap

- ✅ **LLM natural-language modes**: an agentic LangGraph ReAct agent (`--llm`) and a
  single-shot planner (`--llm-single-shot`), both Claude claude-sonnet-4-5 via LangChain.
  Swap the model by passing any LangChain chat model to `LLMPlanner(model=...)` /
  `SQLAgent(..., model=...)`.
- ✅ **Semantic layer**: per-dataset YAML profiles (`semantics/<dataset>.yaml`) served via
  `describe_dataset`, injected into both LLM modes for domain-aware NL→SQL.
- **Redis cache backend**: add `RedisBackend(DataBackend)` and select it in
  `create_backend()`; tools, pipeline and client are unchanged. A cache node can slot
  into the Hamilton dataflow between `validated_sql` and `result_frame`.
- **Remote + OAuth**: add auth middleware/routes to the FastAPI host in
  `server/app.py`; the HTTP client gains an OAuth provider.
