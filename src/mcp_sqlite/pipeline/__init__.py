"""Query-processing pipeline expressed as an Apache Hamilton dataflow."""

from mcp_sqlite.pipeline.runner import run_sql_pipeline

__all__ = ["run_sql_pipeline"]
