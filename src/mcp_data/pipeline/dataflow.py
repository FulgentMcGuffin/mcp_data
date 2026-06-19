"""Hamilton dataflow for processing a SQL query into a serialized result.

Each function below is a node in the DAG. Hamilton wires nodes together by
matching a function's parameter names to other function names (or to runtime
inputs). The graph is::

    sql ------> validated_sql ---+
                                 v
    backend ----------------> result_frame --> row_count
                                 |                 |
                                 +-----------------+--> serialized_result

Keeping validation, execution and formatting as discrete nodes makes each step
independently testable and gives the future Redis phase a natural place to slot
in a caching node between ``validated_sql`` and ``result_frame``.
"""

from __future__ import annotations

from typing import Any

import polars as pl

from mcp_data.backends.base import QueryError, is_read_only_sql


def validated_sql(sql: str) -> str:
    """Reject anything that is not a single read-only statement."""

    if not is_read_only_sql(sql):
        raise QueryError(
            "Only single read-only statements are allowed "
            "(SELECT / WITH / PRAGMA / EXPLAIN)."
        )
    return sql.strip().rstrip(";").strip()


# ``backend`` is annotated ``Any`` because it is supplied as a runtime input and
# Hamilton's strict input type-check does not resolve ``DataBackend`` (a
# ``Protocol``) against concrete implementations. The runner signature keeps the
# precise type for callers.
def result_frame(validated_sql: str, backend: Any) -> pl.DataFrame:
    """Execute the validated query against the backend."""

    return backend.run_query(validated_sql)


def row_count(result_frame: pl.DataFrame) -> int:
    """Number of rows returned by the query."""

    return result_frame.height


def serialized_result(
    result_frame: pl.DataFrame, row_count: int
) -> dict[str, Any]:
    """JSON-serializable view of the result, ready to hand back to the client."""

    return {
        "columns": result_frame.columns,
        "dtypes": [str(dt) for dt in result_frame.dtypes],
        "row_count": row_count,
        "rows": result_frame.to_dicts(),
    }
