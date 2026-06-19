"""SQLite implementation of :class:`DataBackend`.

The connection is opened in read-only mode using a SQLite URI, so the backend
physically cannot mutate the database even if a mutating statement slipped past
the :func:`is_read_only_sql` guard. Results are returned as polars DataFrames.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import polars as pl

from mcp_sqlite.backends.base import (
    ColumnInfo,
    DataBackend,
    QueryError,
    TableSchema,
    is_read_only_sql,
)


class SQLiteBackend(DataBackend):
    """Read-only SQLite backend producing polars frames."""

    name = "sqlite"

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        if not self._db_path.exists():
            raise QueryError(
                f"SQLite database not found at {self._db_path}. "
                "Run the seeder (db-mcp-seed) first."
            )
        # mode=ro opens the existing file read-only; uri=True enables the syntax.
        uri = f"file:{self._db_path.as_posix()}?mode=ro"
        self._conn = sqlite3.connect(uri, uri=True, check_same_thread=False)

    def list_tables(self) -> list[str]:
        cursor = self._conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' "
            "ORDER BY name"
        )
        return [row[0] for row in cursor.fetchall()]

    def get_schema(self, table: str) -> TableSchema:
        if table not in self.list_tables():
            raise QueryError(f"Unknown table: {table!r}")
        # PRAGMA does not support parameter binding for the table name; the name
        # is validated against the table list above, so this is safe.
        cursor = self._conn.execute(f'PRAGMA table_info("{table}")')
        columns = [
            ColumnInfo(
                name=row[1],
                type=row[2] or "UNKNOWN",
                nullable=not bool(row[3]),
                primary_key=bool(row[5]),
            )
            for row in cursor.fetchall()
        ]
        return TableSchema(table=table, columns=columns)

    def run_query(self, sql: str) -> pl.DataFrame:
        if not is_read_only_sql(sql):
            raise QueryError(
                "Only single read-only statements are allowed "
                "(SELECT / WITH / PRAGMA / EXPLAIN)."
            )
        try:
            cursor = self._conn.execute(sql)
        except sqlite3.Error as exc:
            raise QueryError(f"SQL error: {exc}") from exc

        if cursor.description is None:
            return pl.DataFrame()

        column_names = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        if not rows:
            return pl.DataFrame(schema=column_names)
        return pl.DataFrame(rows, schema=column_names, orient="row")

    def close(self) -> None:
        self._conn.close()
