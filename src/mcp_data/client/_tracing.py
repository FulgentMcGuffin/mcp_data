"""LangSmith / LangChain tracing defaults for the MCP client."""

from __future__ import annotations

import logging
import os

_TRUTHY = frozenset({"1", "true", "yes", "on"})


def disable_langsmith_tracing(*, force: bool = False) -> None:
    """Disable LangSmith trace export unless explicitly opted in.

    LangChain may POST traces when ``LANGCHAIN_TRACING_V2`` is enabled but
    credentials are missing or invalid, which spams stderr with 403 errors.
    """
    explicit_on = os.environ.get("LANGCHAIN_TRACING_V2", "").strip().lower() in _TRUTHY
    if force or not explicit_on:
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        os.environ["LANGSMITH_TRACING"] = "false"
        os.environ["LANGCHAIN_TRACING"] = "false"

    for name in ("langsmith", "langsmith.client", "langchain"):
        logging.getLogger(name).setLevel(logging.ERROR)


# Disable by default when this module is imported (before LangChain loads).
disable_langsmith_tracing()
