"""Database backends.

``base`` defines the generic :class:`DataBackend` protocol; concrete backends
(SQLite now, Redis later) implement it so the server tools and query pipeline
never depend on a specific storage engine.
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
from mcp_data.config import Settings

__all__ = [
    "ColumnInfo",
    "DataBackend",
    "DataSink",
    "QueryError",
    "TableSchema",
    "SQLiteSource",
    "create_backend",
    "is_read_only_sql",
]


def create_backend(settings: Settings) -> DataBackend:
    """Construct the configured backend (read-only, for the serving path).

    Today only SQLite is supported. A future Redis cache backend would be
    selected here (e.g. via a ``settings.backend`` field) without changing any
    server or client code, since both satisfy the :class:`DataBackend` protocol.
    """

    return SQLiteSource(settings.db_path, read_only=True)
