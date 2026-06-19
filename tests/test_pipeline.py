"""Tests for the Hamilton query-processing dataflow."""

from __future__ import annotations

import pytest

from mcp_sqlite.backends import SQLiteBackend
from mcp_sqlite.backends.base import QueryError
from mcp_sqlite.pipeline import run_sql_pipeline


def test_pipeline_serializes_result(backend: SQLiteBackend) -> None:
    result = run_sql_pipeline(backend, "select id, name from customers order by id")
    assert result["columns"] == ["id", "name"]
    assert result["row_count"] == 5
    assert result["rows"][0] == {"id": 1, "name": "Ada Lovelace"}
    assert "dtypes" in result


def test_pipeline_rejects_mutation(backend: SQLiteBackend) -> None:
    with pytest.raises(QueryError):
        run_sql_pipeline(backend, "delete from customers")
