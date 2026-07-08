"""Smoke tests for the optional desktop GUI package."""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")


def test_gui_module_imports() -> None:
    from mcp_data.gui import main
    from mcp_data.gui.chat_dialog import ChatDialog

    assert callable(main)
    assert ChatDialog is not None
