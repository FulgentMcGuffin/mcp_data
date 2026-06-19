"""Thin wrapper that builds the Hamilton driver once and runs the dataflow."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from hamilton import driver

from mcp_sqlite.backends.base import DataBackend
from mcp_sqlite.pipeline import dataflow


@lru_cache(maxsize=1)
def _build_driver() -> driver.Driver:
    # The driver only references stateless module functions, so a single
    # instance can be reused across requests; per-call state (sql, backend)
    # is passed as inputs at execute time.
    return driver.Builder().with_modules(dataflow).build()


def run_sql_pipeline(backend: DataBackend, sql: str) -> dict[str, Any]:
    """Run the validate -> execute -> format dataflow and return the result."""

    dr = _build_driver()
    outputs = dr.execute(
        ["serialized_result"],
        inputs={"sql": sql, "backend": backend},
    )
    return outputs["serialized_result"]
