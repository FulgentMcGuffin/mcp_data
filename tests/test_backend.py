"""Tests for the SQLite backend and the read-only guard."""

from __future__ import annotations

import polars as pl
import pytest

from mcp_sqlite.backends import SQLiteBackend
from mcp_sqlite.backends.base import QueryError, is_read_only_sql


def test_list_tables(backend: SQLiteBackend) -> None:
    assert backend.list_tables() == ["customers", "orders"]


def test_get_schema(backend: SQLiteBackend) -> None:
    schema = backend.get_schema("customers")
    assert schema.table == "customers"
    names = [c.name for c in schema.columns]
    assert names == ["id", "name", "email", "country"]
    assert schema.columns[0].primary_key is True


def test_get_schema_unknown_table(backend: SQLiteBackend) -> None:
    with pytest.raises(QueryError):
        backend.get_schema("does_not_exist")


def test_run_query_returns_polars(backend: SQLiteBackend) -> None:
    df = backend.run_query("select * from customers")
    assert isinstance(df, pl.DataFrame)
    assert df.height == 5


def test_run_query_empty_result_keeps_columns(backend: SQLiteBackend) -> None:
    df = backend.run_query("select * from customers where id = -1")
    assert df.height == 0
    assert df.columns == ["id", "name", "email", "country"]


@pytest.mark.parametrize(
    "sql",
    [
        "drop table customers",
        "delete from customers",
        "update customers set name='x'",
        "select 1; drop table customers",
        "",
    ],
)
def test_run_query_rejects_non_readonly(backend: SQLiteBackend, sql: str) -> None:
    with pytest.raises(QueryError):
        backend.run_query(sql)


@pytest.mark.parametrize(
    ("sql", "expected"),
    [
        ("select 1", True),
        ("WITH t AS (SELECT 1) SELECT * FROM t", True),
        ("pragma table_info(customers)", True),
        ("explain select 1", True),
        ("insert into customers values (9, 'a', 'b', 'c')", False),
        ("select 1; select 2", False),
        ("", False),
    ],
)
def test_is_read_only_sql(sql: str, expected: bool) -> None:
    assert is_read_only_sql(sql) is expected
