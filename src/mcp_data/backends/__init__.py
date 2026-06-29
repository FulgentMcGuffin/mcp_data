"""Database backends.

``base`` defines the generic :class:`DataBackend` protocol; concrete backends
(SQLite, DuckDB) implement it so the server tools and query pipeline never depend
on a specific storage engine.
"""

from mcp_data.backends.base import (
    ColumnInfo,
    DataBackend,
    DataSink,
    QueryError,
    TableSchema,
    is_read_only_sql,
)
from mcp_data.backends.sqlite_backend import SQLiteSource
from mcp_data.backends.duckdb_backend import DuckDBSource
from mcp_data.config import Settings

__all__ = [
    "ColumnInfo",
    "DataBackend",
    "DataSink",
    "QueryError",
    "TableSchema",
    "SQLiteSource",
    "DuckDBSource",
    "create_backend",
    "is_read_only_sql",
]


def create_backend(settings: Settings) -> DataBackend:
    """Construct the configured backend (read-only, for the serving path).

    Selects between SQLite and DuckDB based on ``settings.db_type``. Both satisfy
    the :class:`DataBackend` protocol, so the server and client code is unchanged
    when switching backends.
    """

    if settings.db_type == "duckdb":
        return DuckDBSource(settings.db_path, read_only=True)
    else:  # default to sqlite
        return SQLiteSource(settings.db_path, read_only=True)
