"""Launch the MCP Data desktop GUI (`python -m mcp_data.gui`)."""

from mcp_data.client._tracing import disable_langsmith_tracing
from mcp_data.gui.chat_dialog import main


def _entry() -> None:
    disable_langsmith_tracing(force=True)
    main()


if __name__ == "__main__":
    _entry()
