"""Tests for the write side of the unified SQLiteSource (mcp_data.backends).

``MCP_DB_PATH`` is resolved lazily (only when an env-derived default location is
actually needed), so importing the module and using an explicit ``db_path`` never
requires the variable to be set. We still provide a deterministic value here so
the default-location tests assert against a known directory.

The read side (list_tables / get_schema / run_query) is covered in
``test_backend.py``; this module focuses on writes and connection lifecycle.
"""

from __future__ import annotations

import datetime as dt
import os
import tempfile

_DEFAULT_DB_PATH = os.path.join(tempfile.gettempdir(), "mcp_data_ds_tests", "input_data.db")
os.environ.setdefault("MCP_DB_PATH", _DEFAULT_DB_PATH)

import polars as pl  # noqa: E402
import pytest  # noqa: E402

from mcp_data.backends import DataSink, QueryError, SQLiteSource  # noqa: E402
from mcp_data.backends.base import DataBackend, _polars_dtype_to_sql  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def db():
    """A connected, writable in-memory SQLiteSource (single shared connection)."""
    with SQLiteSource(":memory:", read_only=False) as source:
        yield source


@pytest.fixture()
def people(db: SQLiteSource) -> SQLiteSource:
    """An in-memory db pre-populated with a small ``people`` table."""
    db.create_table(
        "people",
        {"id": "INTEGER PRIMARY KEY", "name": "TEXT", "age": "INTEGER"},
    )
    db.insert(
        "people",
        [
            {"id": 1, "name": "Ada", "age": 36},
            {"id": 2, "name": "Alan", "age": 41},
            {"id": 3, "name": "Grace", "age": 36},
        ],
    )
    return db


# ---------------------------------------------------------------------------
# _polars_dtype_to_sql
# ---------------------------------------------------------------------------


def test_polars_dtype_to_sql_integers_and_bool() -> None:
    frame = pl.DataFrame(
        {
            "i8": pl.Series([1], dtype=pl.Int8),
            "i16": pl.Series([1], dtype=pl.Int16),
            "i32": pl.Series([1], dtype=pl.Int32),
            "i64": pl.Series([1], dtype=pl.Int64),
            "u8": pl.Series([1], dtype=pl.UInt8),
            "u32": pl.Series([1], dtype=pl.UInt32),
            "u64": pl.Series([1], dtype=pl.UInt64),
            "flag": pl.Series([True], dtype=pl.Boolean),
        }
    )
    for col in frame.columns:
        assert _polars_dtype_to_sql(frame.schema[col]) == "INTEGER"


def test_polars_dtype_to_sql_floats() -> None:
    frame = pl.DataFrame(
        {
            "f32": pl.Series([1.0], dtype=pl.Float32),
            "f64": pl.Series([1.0], dtype=pl.Float64),
        }
    )
    assert _polars_dtype_to_sql(frame.schema["f32"]) == "REAL"
    assert _polars_dtype_to_sql(frame.schema["f64"]) == "REAL"


def test_polars_dtype_to_sql_text_fallback() -> None:
    frame = pl.DataFrame(
        {
            "s": pl.Series(["x"], dtype=pl.Utf8),
            "d": pl.Series([dt.date(2020, 1, 1)]),
        }
    )
    assert _polars_dtype_to_sql(frame.schema["s"]) == "TEXT"
    assert _polars_dtype_to_sql(frame.schema["d"]) == "TEXT"


# ---------------------------------------------------------------------------
# Connection management
# ---------------------------------------------------------------------------


def test_methods_require_active_connection() -> None:
    # The source connects eagerly; after close() the connection guard kicks in.
    source = SQLiteSource(":memory:", read_only=False)
    source.close()
    with pytest.raises(RuntimeError, match="No active connection"):
        source.create_table("t", {"id": "INTEGER"})


def test_context_manager_opens_and_closes() -> None:
    source = SQLiteSource(":memory:", read_only=False)
    assert source._connection is not None  # connected eagerly on construction
    with source as opened:
        assert opened is source
        assert source._connection is not None
    assert source._connection is None


def test_close_is_idempotent() -> None:
    source = SQLiteSource(":memory:", read_only=False)
    source.close()
    source.close()  # second close must not raise
    assert source._connection is None


def test_connect_creates_file_and_parent_dirs(tmp_path) -> None:
    db_path = tmp_path / "nested" / "dir" / "test.db"
    source = SQLiteSource(str(db_path), read_only=False)
    try:
        assert db_path.exists()
        assert db_path.parent.is_dir()
    finally:
        source.close()


def test_row_factory_returns_mapping_rows(db: SQLiteSource) -> None:
    db.create_table("t", {"id": "INTEGER", "name": "TEXT"})
    db.insert("t", {"id": 1, "name": "Ada"})
    rows = db.select("t")
    assert rows == [{"id": 1, "name": "Ada"}]


# ---------------------------------------------------------------------------
# create_table
# ---------------------------------------------------------------------------


def test_create_table_returns_true_on_creation(db: SQLiteSource) -> None:
    assert db.create_table("t", {"id": "INTEGER PRIMARY KEY", "name": "TEXT"}) is True


def test_create_table_existing_no_overwrite_returns_false(db: SQLiteSource) -> None:
    db.create_table("t", {"id": "INTEGER"})
    db.insert("t", {"id": 1})
    assert db.create_table("t", {"id": "INTEGER"}) is False
    # Existing data is preserved when the table is left unchanged.
    assert db.select("t") == [{"id": 1}]


def test_create_table_overwrite_recreates(db: SQLiteSource) -> None:
    db.create_table("t", {"id": "INTEGER", "name": "TEXT"})
    db.insert("t", {"id": 1, "name": "Ada"})
    assert db.create_table("t", {"id": "INTEGER"}, overwrite_if_exists=True) is True
    # Table was dropped and recreated: old rows gone, schema is the new one.
    assert db.select("t") == []
    cols = [r["name"] for r in db.execute("PRAGMA table_info(t)")]
    assert cols == ["id"]


# ---------------------------------------------------------------------------
# select
# ---------------------------------------------------------------------------


def test_select_all_columns(people: SQLiteSource) -> None:
    rows = people.select("people")
    assert len(rows) == 3
    assert set(rows[0].keys()) == {"id", "name", "age"}


def test_select_specific_columns(people: SQLiteSource) -> None:
    rows = people.select("people", columns=["name"])
    assert rows[0] == {"name": "Ada"}
    assert all(set(r.keys()) == {"name"} for r in rows)


def test_select_with_where(people: SQLiteSource) -> None:
    rows = people.select("people", where={"age": 36})
    names = sorted(r["name"] for r in rows)
    assert names == ["Ada", "Grace"]


def test_select_with_multiple_where_conditions(people: SQLiteSource) -> None:
    rows = people.select("people", where={"age": 36, "name": "Ada"})
    assert rows == [{"id": 1, "name": "Ada", "age": 36}]


# ---------------------------------------------------------------------------
# insert
# ---------------------------------------------------------------------------


def test_insert_single_dict_returns_one(db: SQLiteSource) -> None:
    db.create_table("t", {"id": "INTEGER", "name": "TEXT"})
    assert db.insert("t", {"id": 1, "name": "Ada"}) == 1


def test_insert_list_returns_count(db: SQLiteSource) -> None:
    db.create_table("t", {"id": "INTEGER"})
    assert db.insert("t", [{"id": 1}, {"id": 2}, {"id": 3}]) == 3
    assert len(db.select("t")) == 3


def test_insert_empty_list_returns_zero(db: SQLiteSource) -> None:
    db.create_table("t", {"id": "INTEGER"})
    assert db.insert("t", []) == 0


# ---------------------------------------------------------------------------
# update / delete
# ---------------------------------------------------------------------------


def test_update_returns_affected_rowcount(people: SQLiteSource) -> None:
    affected = people.update("people", {"age": 99}, where={"age": 36})
    assert affected == 2
    assert all(r["age"] == 99 for r in people.select("people", where={"age": 99}))


def test_update_no_match_returns_zero(people: SQLiteSource) -> None:
    assert people.update("people", {"age": 1}, where={"name": "Nobody"}) == 0


def test_delete_returns_deleted_rowcount(people: SQLiteSource) -> None:
    assert people.delete("people", where={"age": 36}) == 2
    assert [r["name"] for r in people.select("people")] == ["Alan"]


def test_delete_no_match_returns_zero(people: SQLiteSource) -> None:
    assert people.delete("people", where={"name": "Nobody"}) == 0


# ---------------------------------------------------------------------------
# execute
# ---------------------------------------------------------------------------


def test_execute_select_returns_dicts(people: SQLiteSource) -> None:
    rows = people.execute("SELECT name FROM people WHERE age = ?", (41,))
    assert rows == [{"name": "Alan"}]


def test_execute_non_select_returns_empty_list(db: SQLiteSource) -> None:
    assert db.execute("CREATE TABLE t (id INTEGER)") == []
    assert db.execute("INSERT INTO t (id) VALUES (1)") == []
    assert db.select("t") == [{"id": 1}]


# ---------------------------------------------------------------------------
# create_table_from_polars (defined on the DataSource base class)
# ---------------------------------------------------------------------------


def test_create_table_from_polars_infers_schema_and_inserts(db: SQLiteSource) -> None:
    frame = pl.DataFrame(
        {
            "id": pl.Series([1, 2], dtype=pl.Int64),
            "score": pl.Series([1.5, 2.5], dtype=pl.Float64),
            "name": pl.Series(["a", "b"], dtype=pl.Utf8),
        }
    )
    created = db.create_table_from_polars("metrics", frame)
    assert created is True

    info = {r["name"]: r["type"] for r in db.execute("PRAGMA table_info(metrics)")}
    assert info == {"id": "INTEGER", "score": "REAL", "name": "TEXT"}

    rows = db.select("metrics")
    assert len(rows) == 2
    assert rows[0] == {"id": 1, "score": 1.5, "name": "a"}


def test_create_table_from_polars_empty_frame_creates_without_rows(db: SQLiteSource) -> None:
    frame = pl.DataFrame({"id": pl.Series([], dtype=pl.Int64)})
    created = db.create_table_from_polars("empty_t", frame)
    assert created is True
    assert db.select("empty_t") == []


def test_create_table_from_polars_skips_when_exists(db: SQLiteSource) -> None:
    frame = pl.DataFrame({"id": pl.Series([1], dtype=pl.Int64)})
    assert db.create_table_from_polars("t", frame) is True
    # Second call with a different frame should be a no-op (table already exists).
    other = pl.DataFrame({"id": pl.Series([2, 3], dtype=pl.Int64)})
    assert db.create_table_from_polars("t", other) is False
    assert db.select("t") == [{"id": 1}]


def test_create_table_from_polars_overwrite_replaces_rows(db: SQLiteSource) -> None:
    db.create_table_from_polars("t", pl.DataFrame({"id": pl.Series([1], dtype=pl.Int64)}))
    new_frame = pl.DataFrame({"id": pl.Series([7, 8], dtype=pl.Int64)})
    created = db.create_table_from_polars("t", new_frame, overwrite_if_exists=True)
    assert created is True
    assert sorted(r["id"] for r in db.select("t")) == [7, 8]


def test_create_table_from_polars_large_frame_splits_inserts(db: SQLiteSource) -> None:
    # Exceed the 1e6 row threshold so the chunked insert path is exercised.
    n = 1_000_001
    frame = pl.DataFrame({"v": pl.arange(0, n, eager=True)})
    created = db.create_table_from_polars("big", frame, num_splits=5)
    assert created is True
    count = db.execute("SELECT COUNT(*) AS c FROM big")[0]["c"]
    assert count == n


# ---------------------------------------------------------------------------
# Class-level helpers and MCP_DB_PATH validation
# ---------------------------------------------------------------------------


def test_get_full_db_path_appends_db_extension() -> None:
    expected = os.path.join(SQLiteSource.default_db_dir(), "foo.db")
    assert SQLiteSource.get_full_db_path("foo") == expected
    assert SQLiteSource.get_full_db_path("foo.db") == expected


def test_get_full_db_path_default_uses_default_name() -> None:
    expected = os.path.join(SQLiteSource.default_db_dir(), SQLiteSource.default_db_name())
    assert SQLiteSource.get_full_db_path() == expected


def test_init_default_db_path_uses_class_defaults() -> None:
    source = SQLiteSource(read_only=False)
    try:
        expected = os.path.join(
            SQLiteSource.default_db_dir(), SQLiteSource.default_db_name()
        )
        assert source._db_path == expected
    finally:
        source.close()


def test_sqlitesource_satisfies_read_and_write_contracts() -> None:
    assert issubclass(SQLiteSource, DataSink)
    with SQLiteSource(":memory:", read_only=False) as source:
        assert isinstance(source, DataBackend)


def test_explicit_path_does_not_require_mcp_db_path(monkeypatch) -> None:
    # The whole point of the lazy resolution: an explicit db_path must work even
    # with MCP_DB_PATH completely absent (importing the module already succeeded).
    monkeypatch.delenv("MCP_DB_PATH", raising=False)
    with SQLiteSource(":memory:", read_only=False) as source:
        source.create_table("t", {"id": "INTEGER"})
        assert source.select("t") == []


def test_default_path_without_mcp_db_path_raises(monkeypatch) -> None:
    monkeypatch.delenv("MCP_DB_PATH", raising=False)
    with pytest.raises(ValueError, match="not set"):
        SQLiteSource(read_only=False)
    with pytest.raises(ValueError, match="not set"):
        SQLiteSource.get_full_db_path("x")


def test_non_db_extension_raises(monkeypatch) -> None:
    monkeypatch.setenv(
        "MCP_DB_PATH", os.path.join(tempfile.gettempdir(), "bad", "data.sqlite")
    )
    with pytest.raises(ValueError, match="must end with .db"):
        SQLiteSource(read_only=False)


# ---------------------------------------------------------------------------
# read_only enforcement (the safety guarantee for the serving path)
# ---------------------------------------------------------------------------


def test_read_only_rejects_writes_but_allows_reads(tmp_path) -> None:
    db_path = tmp_path / "ro.db"
    with SQLiteSource(str(db_path), read_only=False) as writer:
        writer.create_table("t", {"id": "INTEGER", "name": "TEXT"})
        writer.insert("t", {"id": 1, "name": "Ada"})

    reader = SQLiteSource(str(db_path))  # read_only=True by default
    try:
        assert reader.list_tables() == ["t"]
        assert reader.run_query("select * from t").height == 1
        assert reader.select("t") == [{"id": 1, "name": "Ada"}]
        for write in (
            lambda: reader.create_table("x", {"id": "INTEGER"}),
            lambda: reader.insert("t", {"id": 2, "name": "Bob"}),
            lambda: reader.update("t", {"name": "z"}, {"id": 1}),
            lambda: reader.delete("t", {"id": 1}),
        ):
            with pytest.raises(QueryError):
                write()
    finally:
        reader.close()


def test_read_only_missing_file_raises(tmp_path) -> None:
    with pytest.raises(QueryError, match="not found"):
        SQLiteSource(str(tmp_path / "does_not_exist.db"))
