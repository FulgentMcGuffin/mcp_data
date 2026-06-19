"""Database backends.

``base`` defines the generic :class:`DataBackend` protocol; concrete backends
(SQLite now, Redis later) implement it so the server tools and query pipeline
never depend on a specific storage engine.
"""

from mcp_sqlite.backends.base import (
    ColumnInfo,
    DataBackend,
    QueryError,
    TableSchema,
    is_read_only_sql,
)
from mcp_sqlite.backends.sqlite_backend import SQLiteBackend
from mcp_sqlite.config import Settings

__all__ = [
    "ColumnInfo",
    "DataBackend",
    "QueryError",
    "TableSchema",
    "SQLiteBackend",
    "create_backend",
    "is_read_only_sql",
]


def create_backend(settings: Settings) -> DataBackend:
    """Construct the configured backend.

    Today only SQLite is supported. A future Redis cache backend would be
    selected here (e.g. via a ``settings.backend`` field) without changing any
    server or client code.
    """

    return SQLiteBackend(settings.db_path)
