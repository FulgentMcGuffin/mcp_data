# mcp-data

A **Model Context Protocol (MCP) server and client** for querying databases with natural
language. SQLite is the first backend, hidden behind a generic `DataBackend` abstraction so
the storage layer can be swapped (e.g. a Redis cache) without touching the server tools or
client. Transport is selectable between **stdio** and **Streamable HTTP** (hosted via
FastAPI).

Key libraries:

| Concern | Library |
| --- | --- |
| Package / dependency management | **uv** |
| DataFrames | **polars** |
| Query pipeline | **Apache Hamilton** |
| MCP server | **FastMCP** (official `mcp` SDK) + **FastAPI** for HTTP |
| LLM integration | **LangChain** + **langchain-anthropic** (Claude) |
| Agentic loop | **LangGraph** (`create_react_agent`) |
| MCP ↔ LangChain bridge | **langchain-mcp-adapters** |

## Architecture

```
user query
    │
    ├── RuleBasedPlanner  (keyword syntax, no API key)
    ├── LLMPlanner        (single-shot, --llm-single-shot)
    └── SQLAgent          (ReAct loop, --llm)
            │
            ▼
    MCP ClientSession ─── stdio or Streamable HTTP ───► FastMCP server
                                                              │
                                          tools: list_tables, get_schema,
                                                 run_sql, describe_dataset
                                                              │
                                          Hamilton dataflow (validate →
                                          execute → polars frame → JSON)
                                                              │
                                          SQLiteSource (DataBackend, read-only)
                                                              │
                                          semantics/<dataset>.yaml
                                          (served via describe_dataset)
```

A per-dataset **semantic layer** (see [Semantic layer](#semantic-layer)) is co-located
with the database and exposed by the MCP server. Both LLM modes fetch it on startup so
they understand domain vocabulary (e.g. "Italy" → `ITA`, "5Y" → `Y005p0`) before
generating SQL.

## Module layout (`src/mcp_data/`)

| Module | Responsibility |
| --- | --- |
| `config.py` | `Settings` dataclass; loads all config from env / `.env` |
| `backends/base.py` | `DataBackend` (read) + `DataSink` (write) contracts; `is_read_only_sql` guard |
| `backends/sqlite_backend.py` | `SQLiteSource`: unified read/write SQLite store (`read_only` flag) |
| `backends/__init__.py` | `create_backend()` factory |
| `semantics/__init__.py` | `SemanticProfile`, `load_profile`, `render_profile_prompt` |
| `data/seed.py` | Creates and seeds the example SQLite database |
| `data/download_ycs.py` | Downloads raw yield-curve data from external sources |
| `data/populate_ycs.py` | Parses and loads yield-curve data into SQLite |
| `pipeline/dataflow.py` | Hamilton nodes: validate → execute → frame → serialize |
| `pipeline/runner.py` | Builds the Hamilton driver and calls the dataflow |
| `server/app.py` | MCP tools + resources + FastAPI HTTP host |
| `server/__main__.py` | Entry point; dispatches on `MCP_TRANSPORT` |
| `client/session.py` | `DBClient`: transport-agnostic MCP session (stdio + HTTP) |
| `client/planner.py` | `RuleBasedPlanner` + `LLMPlanner` (single-shot Claude) |
| `client/agent.py` | `SQLAgent`: LangGraph ReAct agent over live MCP tools |
| `client/cli.py` | CLI entry point (`db-mcp-client`) |

Semantic profiles live under `semantics/` at the project root:
`semantics/input_data.yaml` (real yield-curve dataset) and `semantics/example.yaml`
(seeded demo database).

---

## Setup (development)

Clone the repo and sync the environment (creates `.venv` and installs the project
editably with all dev dependencies):

```bash
uv sync
uv run db-mcp-seed        # create data/example.db with sample customers + orders
```

## Installing as a package

`mcp-data` is a standard [PEP 621](https://peps.python.org/pep-0621/) project built with
**hatchling**. It exposes three console scripts: `db-mcp-server`, `db-mcp-client`,
`db-mcp-seed`.

### With uv

```bash
# Add to another uv project
uv add /path/to/mcp_data
uv add "git+https://github.com/<org>/mcp_data.git"

# Install as an isolated CLI tool (scripts on PATH)
uv tool install /path/to/mcp_data
```

### With pip

```bash
pip install .                                             # from a local checkout
pip install -e .                                          # editable / develop mode
pip install "git+https://github.com/<org>/mcp_data.git"  # directly from Git
```

### Build distributables

```bash
uv build       # writes dist/mcp_data-<version>-py3-none-any.whl and .tar.gz
pip install dist/mcp_data-*.whl
```

### After installing

```bash
db-mcp-seed                             # create the example database
db-mcp-client "tables"                  # run a one-shot query
```

```python
from mcp_data.client.session import DBClient  # use as a library
```

> **Note — runtime data.** The example database (`data/`) and curated semantic profiles
> (`semantics/`) are not bundled in the wheel. When running from an installed package,
> set `MCP_DB_PATH` and `MCP_SEMANTICS_DIR` to point at your own files. Without a
> profile, the server falls back to live schema introspection automatically.

---

## Configuration (environment variables)

| Variable | Default | Meaning |
| --- | --- | --- |
| `MCP_TRANSPORT` | `stdio` | `stdio` or `http` |
| `MCP_DB_PATH` | `data/example.db` | Path to the SQLite file |
| `MCP_HOST` | `127.0.0.1` | HTTP server host |
| `MCP_PORT` | `8000` | HTTP server port |
| `MCP_SERVER_NAME` | `db-mcp` | Server display name reported to clients |
| `MCP_DATASET` | *(db file stem)* | Dataset name that keys the semantic profile (`semantics/<dataset>.yaml`) |
| `MCP_SEMANTICS_DIR` | `semantics/` | Directory holding semantic profile YAML files |
| `ANTHROPIC_API_KEY` | *(required for `--llm` / `--llm-single-shot`)* | Anthropic API key |

Variables are loaded from `.env` and `.secrets` files (if they exist) at the project root
via `python-dotenv`. Both files are treated as extensions of each other: `.env` is loaded
first, then `.secrets`, so `.secrets` can override `.env` if needed. OS environment
variables take precedence over both files.

---

## Running

### stdio (client spawns the server automatically)

```bash
# One-shot queries
uv run db-mcp-client "tables"
uv run db-mcp-client "schema customers"
uv run db-mcp-client "sql: select * from customers limit 5"

# Interactive REPL
uv run db-mcp-client
```

### Streamable HTTP (long-lived server)

```bash
# Terminal 1 — start the server
MCP_TRANSPORT=http uv run db-mcp-server      # http://127.0.0.1:8000/mcp

# Terminal 2 — run the client
MCP_TRANSPORT=http uv run db-mcp-client "tables"
```

Health check: `GET http://127.0.0.1:8000/healthz`

> On PowerShell use `$env:MCP_TRANSPORT = "http"` instead of the inline prefix.

---

## MCP tools and resources

The server exposes four tools and two resources regardless of transport.

### Tools

| Tool | Arguments | Returns |
| --- | --- | --- |
| `list_tables` | — | `list[str]` of table names |
| `get_schema` | `table: str` | Column schema (name, type, nullable, primary key) |
| `run_sql` | `sql: str` | `{columns, dtypes, row_count, rows}` — read-only SQL only |
| `describe_dataset` | — | `{profile, prompt, has_curated_profile}` — semantic description of the dataset |

### Resources

| URI | Content |
| --- | --- |
| `schema://{table}` | Column schema for `{table}` |
| `semantics://dataset` | Full semantic profile for the active dataset |

---

## CLI commands

### Rule-based planner (default — no API key needed)

```
tables                 list tables
schema <table>         show a table's schema
sql: <query>           run a read-only SQL query
help                   show this help
quit / exit            leave
```

```bash
uv run db-mcp-client "schema customers"
uv run db-mcp-client "sql: select country, count(*) from customers group by country"
```

### Agentic LLM mode (`--llm`)

A **LangGraph ReAct agent** (`SQLAgent`) backed by Claude claude-sonnet-4-5. It runs a
multi-step reasoning loop: introspect schema → build SQL → execute → observe result or
error → self-correct. Tools are loaded live from the MCP session via
`langchain-mcp-adapters`. The dataset's semantic profile is injected on startup.
Requires `ANTHROPIC_API_KEY`.

```bash
# One-shot
uv run db-mcp-client --llm "how many customers are there per country?"
uv run db-mcp-client --llm "highest 5Y–10Y zero rate spread for Italy between 2010 and 2015"

# Interactive REPL
uv run db-mcp-client --llm
```

### Single-shot LLM planner (`--llm-single-shot`)

The LLM picks tool calls in a single step (no agentic loop). Lighter and cheaper;
the semantic profile is still injected. Requires `ANTHROPIC_API_KEY`.

```bash
uv run db-mcp-client --llm-single-shot "show me the schema of the customers table"
uv run db-mcp-client --llm-single-shot "top 3 orders by amount"
```

### Swapping the LLM model

Both LLM modes accept any LangChain chat model:

```python
from langchain_openai import ChatOpenAI
from mcp_data.client.planner import LLMPlanner
from mcp_data.client.agent import SQLAgent

planner = LLMPlanner(model=ChatOpenAI(model="gpt-4o"))
# agent: SQLAgent(client, model=ChatOpenAI(model="gpt-4o"))
```

---

## Semantic layer

Each dataset can have a declarative **semantic profile** — a YAML data dictionary that
tells the LLM:

- what each table and column means,
- a **controlled vocabulary** mapping business terms to stored values/columns
  (e.g. `Italy → ITA`, `5Y → Y005p0`),
- date/naming conventions,
- curated natural-language → SQL examples.

Profiles are files named `semantics/<dataset>.yaml`. The dataset name defaults to the
database file stem (e.g. `input_data.db` → `input_data`); override with `MCP_DATASET`.
The server exposes the profile via `describe_dataset` and `semantics://dataset`; when no
profile file exists, it synthesises a minimal one from live schema introspection.

```yaml
dataset: input_data
backend: sqlite
description: Yield-curve and FX time series (zero rates, par rates, spot FX).
vocabulary:
  countries:
    Italy: ITA
    Germany: DEU
  tenors:
    "5Y": Y005p0
    "10Y": Y010p0
conventions:
  date: "TEXT 'YYYY-MM-DD'; filter with date >= '<start>' AND date <= '<end>'"
tables:
  zero_rates:
    description: Zero-coupon spot rates; one row per (source, date).
    columns:
      date:   {type: TEXT, description: "Observation date (YYYY-MM-DD)."}
      source: {type: TEXT, description: "Curve/country code, e.g. ITA=Italy."}
      Y005p0: {type: REAL, description: "5-year zero rate."}
      Y010p0: {type: REAL, description: "10-year zero rate."}
examples:
  - q: "Highest absolute 5Y–10Y spread for Italy 2010–2015"
    sql: >
      SELECT MAX(ABS(Y010p0 - Y005p0)) AS max_abs_diff
      FROM zero_rates
      WHERE source = 'ITA'
        AND date >= '2010-01-01' AND date <= '2015-12-31'
```

Shipped profiles:

- `semantics/input_data.yaml` — real yield-curve dataset (`zero_rates`, `par_rates`,
  `spotfx`, `window_corr`). Verify the `source` codes against your database:
  `SELECT DISTINCT source FROM zero_rates`.
- `semantics/example.yaml` — seeded demo database (`customers`, `orders`).

---

## Safety

`run_sql` only accepts a single read-only statement (`SELECT` / `WITH` / `PRAGMA` /
`EXPLAIN`). Multi-statement and mutating SQL is rejected before reaching the database,
and the SQLite connection is opened with `mode=ro` so it physically cannot write.

---

## Tests

```bash
uv run pytest -q
```

The suite covers the SQLite backend, Hamilton pipeline, `DataSink` write operations,
the semantics module (load, render, roundtrip, fallback), `describe_dataset` (curated
and live-introspection paths), and `LLMPlanner` profile injection (mocked model, no
live API call required).

---

## Roadmap

- **Redis cache backend** — add `RedisBackend(DataBackend)` and wire it into
  `create_backend()`; the server tools, Hamilton pipeline, and client are unchanged.
  A cache node could slot into the Hamilton dataflow between `validated_sql` and
  `result_frame`.
- **Remote transport + OAuth** — add auth middleware/routes to the FastAPI host in
  `server/app.py`; the HTTP client gains an OAuth provider.
- **`db-mcp-describe` bootstrap command** — introspect a database and emit a starter
  `semantics/<dataset>.yaml` (descriptions blank) to cut profile-authoring effort.
- **Dynamic few-shot retrieval** — vector-search example queries at inference time
  instead of inlining all examples in the profile.
