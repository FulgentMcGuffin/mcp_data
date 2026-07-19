"""Shared Anthropic chat-model construction for client LLM paths."""

from __future__ import annotations

import os


def create_anthropic_chat(model: str, *, api_key: str | None = None):
    """Build a :class:`ChatAnthropic` instance for tool-calling planners/agents.

    Omits ``temperature`` — recent Claude model families reject that parameter
    (400 ``invalid_request_error: temperature is deprecated for this model``).
    """
    from langchain_anthropic import ChatAnthropic

    resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not resolved_key:
        raise ValueError(
            "ANTHROPIC_API_KEY is not set. "
            "Add it to your .env file or export it before using --llm."
        )
    return ChatAnthropic(model=model, api_key=resolved_key)
