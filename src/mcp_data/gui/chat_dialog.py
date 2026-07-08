from __future__ import annotations

import asyncio
import json
import re
import sys
import warnings
from datetime import datetime
from pathlib import Path

import polars as pl
from matplotlib.ticker import AutoMinorLocator
from plotnine.exceptions import PlotnineError, PlotnineWarning
from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot
from PySide6.QtGui import QCloseEvent, QShowEvent, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTableWidgetItem,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from mcp_data.gui.chat_settings_dialog import ChatSettingsDialog
from mcp_data.gui.gui_constants import Theme, get_plot_bg, get_theme, window_style
from mcp_data.gui.gui_settings import load_settings, save_settings
from mcp_data.gui.gui_utils import FramelessResizeHelper, PanelFrame, RoundedShell, TitleBar
from mcp_data.gui.plotnine_wrapper import (
    coerce_date_columns,
    create_ggplot_from_df,
    format_table_cell,
    infer_plot_columns,
    process_df_for_ggplot,
)
from mcp_data.gui.table_and_plot_widget import PlotWidget, TableWidget

_SQL_FENCE_PATTERNS = (
    re.compile(r"```sql\s*\n(.*?)```", re.IGNORECASE | re.DOTALL),
    re.compile(
        r"```\s*\n((?:SELECT|WITH|PRAGMA|EXPLAIN)\b.*?)```",
        re.IGNORECASE | re.DOTALL,
    ),
)
_READ_ONLY_SQL_RE = re.compile(
    r"^\s*(SELECT|WITH|PRAGMA|EXPLAIN)\b",
    re.IGNORECASE | re.DOTALL,
)


def run_sql_result_to_dataframe(result: object) -> pl.DataFrame | None:
    """Convert a run_sql tool payload into a Polars DataFrame with column names."""
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except json.JSONDecodeError:
            return None
    if not isinstance(result, dict) or "error" in result:
        return None

    rows = result.get("rows")
    columns = result.get("columns")
    if not isinstance(rows, list):
        return None

    if not rows:
        if isinstance(columns, list) and columns:
            return pl.DataFrame({str(col): [] for col in columns})
        return None

    df = pl.DataFrame(rows)
    if isinstance(columns, list) and columns:
        col_names = [str(col) for col in columns]
        if all(name in df.columns for name in col_names):
            df = df.select(col_names)
    return df


def extract_sql_from_text(text: str) -> str | None:
    """Extract the last read-only SQL statement from markdown code fences."""
    if not text:
        return None

    for pattern in _SQL_FENCE_PATTERNS:
        for candidate in reversed(pattern.findall(text)):
            sql = candidate.strip().rstrip(";").strip()
            if sql and _READ_ONLY_SQL_RE.match(sql):
                return sql

    return None


def extract_dataframe_from_messages(messages) -> pl.DataFrame | None:
    """Return the most recent SQL result as a Polars DataFrame, if present."""
    for msg in reversed(messages):
        if type(msg).__name__ != "ToolMessage":
            continue

        content = msg.content
        payloads: list[object] = []
        if isinstance(content, dict):
            payloads = [content]
        elif isinstance(content, str):
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                payloads = [parsed]
            elif isinstance(parsed, list):
                payloads = parsed
        elif isinstance(content, list):
            payloads = content

        for data in payloads:
            df = run_sql_result_to_dataframe(data)
            if df is not None:
                return df
    return None


async def fetch_dataframe_for_answer(client, answer: str, messages) -> pl.DataFrame | None:
    """Build a dataframe from tool messages or by running SQL found in the answer."""
    dataframe = extract_dataframe_from_messages(messages)
    if dataframe is not None and not dataframe.is_empty():
        return dataframe

    sql = extract_sql_from_text(answer)
    if not sql:
        return dataframe

    tools = await client.list_tools()
    if "run_sql" not in tools:
        return dataframe

    result = await client.call_tool("run_sql", {"sql": sql})
    df = run_sql_result_to_dataframe(result)
    if df is not None:
        return df
    return dataframe


class LlmWorker(QObject):
    """Runs mcp_data LLM agent queries on a background thread."""

    query_requested = Signal(str)
    finished = Signal(str, object)
    error = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.query_requested.connect(self.run_query)

    @Slot(str)
    def run_query(self, query: str) -> None:
        from mcp_data.client._tracing import disable_langsmith_tracing

        disable_langsmith_tracing(force=True)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            answer, dataframe = loop.run_until_complete(self._execute(query))
            self.finished.emit(answer, dataframe)
        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            loop.close()

    async def _execute(self, query: str) -> tuple[str, pl.DataFrame | None]:
        from mcp_data.client.agent import SQLAgent
        from mcp_data.client.session import DBClient

        async with DBClient() as client:
            profile = await client.describe_dataset()
            profile_prompt = profile.get("prompt") if isinstance(profile, dict) else None
            agent = SQLAgent(client, profile_prompt=profile_prompt)
            result = await agent.run(query)
            answer = result.answer or "(no answer)"
            dataframe = await fetch_dataframe_for_answer(
                client, answer, result.messages
            )
            return answer, dataframe


class ChatDialog(QMainWindow):
    """Interactive LLM chat with data table and plot output."""

    def __init__(self, parent=None):
        super().__init__(parent)

        self._settings = load_settings()
        self._theme = get_theme(self._settings.get("ui_theme"))
        self._last_dataframe: pl.DataFrame | None = None
        self._chat_log: list[str] = []

        self.setWindowTitle("MCP Data Chat")
        self.resize(1100, 984)
        self.setWindowFlags(
            self.windowFlags() | Qt.WindowType.FramelessWindowHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        outer = QWidget()
        outer.setObjectName("outerRoot")
        outer.setAttribute(Qt.WA_StyledBackground, True)
        self.setCentralWidget(outer)
        outer_layout = QVBoxLayout(outer)
        outer_layout.setContentsMargins(16, 16, 16, 16)

        shell = RoundedShell()
        shell_layout = QVBoxLayout(shell)
        shell_layout.setContentsMargins(0, 0, 0, 0)
        shell_layout.setSpacing(0)

        self._settings_btn = QPushButton("Settings")
        self._settings_btn.setObjectName("titleSettingsBtn")
        self._settings_btn.clicked.connect(self._open_settings)
        shell_layout.addWidget(
            TitleBar(
                self,
                "MCP Data Chat",
                shell,
                trailing=self._settings_btn,
                subtitle="Natural language data queries",
            )
        )

        content = QWidget()
        content.setObjectName("dialogContent")
        content.setAttribute(Qt.WA_StyledBackground, True)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(20, 18, 20, 20)
        layout.setSpacing(14)

        self._status_label = QLabel("Ready")
        self._status_label.setObjectName("statusPill")
        self._status_label.setProperty("busy", False)
        layout.addWidget(self._status_label, alignment=Qt.AlignmentFlag.AlignLeft)

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.setObjectName("mainSplitter")

        chat_panel = PanelFrame("Assistant")
        chat_inner = QWidget()
        chat_inner.setObjectName("chatInner")
        chat_layout = QVBoxLayout(chat_inner)
        chat_layout.setContentsMargins(0, 0, 0, 0)
        chat_layout.setSpacing(10)

        prompt_row = QHBoxLayout()
        prompt_row.setSpacing(10)
        self._prompt_input = QLineEdit()
        self._prompt_input.setObjectName("promptInput")
        self._prompt_input.setPlaceholderText("Ask a question about your data…")
        self._prompt_input.returnPressed.connect(self._send_prompt)
        prompt_row.addWidget(self._prompt_input, stretch=1)

        self._send_btn = QPushButton("Send")
        self._send_btn.setObjectName("sendBtn")
        self._send_btn.clicked.connect(self._send_prompt)
        prompt_row.addWidget(self._send_btn)
        chat_layout.addLayout(prompt_row)

        self._chat_view = QTextBrowser()
        self._chat_view.setObjectName("chatView")
        self._chat_view.setOpenExternalLinks(True)
        self._chat_view.setPlaceholderText("LLM responses appear here…")
        chat_layout.addWidget(self._chat_view, stretch=1)

        chat_panel.set_content(chat_inner)
        splitter.addWidget(chat_panel)

        data_splitter = QSplitter(Qt.Orientation.Horizontal)
        data_splitter.setObjectName("dataSplitter")
        self._data_splitter = data_splitter

        table_panel = PanelFrame("Results")
        self._table = TableWidget()
        self._table.setObjectName("dataTable")
        table_panel.set_content(self._table)

        plot_panel = PanelFrame("Visualization")
        self._plot = PlotWidget()
        self._plot.setObjectName("dataPlot")
        plot_panel.set_content(self._plot)

        expanding = QSizePolicy.Policy.Expanding
        for widget in (self._table, self._plot):
            widget.setSizePolicy(expanding, expanding)

        data_splitter.addWidget(table_panel)
        data_splitter.addWidget(plot_panel)
        data_splitter.setStretchFactor(0, 1)
        data_splitter.setStretchFactor(1, 1)
        data_splitter.setChildrenCollapsible(False)

        splitter.addWidget(data_splitter)

        # Chat panel gets 60% of vertical space (1.5× the previous 40% share).
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        layout.addWidget(splitter, stretch=1)

        # Action buttons row
        actions_row = QHBoxLayout()
        actions_row.setSpacing(8)
        actions_row.addStretch()

        self._download_btn = QPushButton("Download Data")
        self._download_btn.setObjectName("actionBtn")
        self._download_btn.clicked.connect(self._download_data)
        self._download_btn.setEnabled(False)
        actions_row.addWidget(self._download_btn)

        self._copy_data_btn = QPushButton("Copy Data")
        self._copy_data_btn.setObjectName("actionBtn")
        self._copy_data_btn.clicked.connect(self._copy_table_data)
        self._copy_data_btn.setEnabled(False)
        self._copy_data_btn.setShortcut(QKeySequence("Ctrl+Shift+C"))
        actions_row.addWidget(self._copy_data_btn)

        self._copy_chart_btn = QPushButton("Copy Chart")
        self._copy_chart_btn.setObjectName("actionBtn")
        self._copy_chart_btn.clicked.connect(self._copy_chart_image)
        self._copy_chart_btn.setEnabled(False)
        actions_row.addWidget(self._copy_chart_btn)

        actions_row.addStretch()
        layout.addLayout(actions_row)

        shell_layout.addWidget(content)
        outer_layout.addWidget(shell)

        self._apply_theme()

        self._append_system_message(
            "Connected to the MCP Data LLM agent. Ask a question in natural language."
        )

        self._worker_thread = QThread(self)
        self._llm_worker = LlmWorker()
        self._llm_worker.moveToThread(self._worker_thread)
        self._llm_worker.finished.connect(self._on_llm_response)
        self._llm_worker.error.connect(self._on_llm_error)
        self._worker_thread.start()

        self._resize_helper = FramelessResizeHelper(self)
        self._resize_helper.install()

    def eventFilter(self, watched, event):
        if self._resize_helper.handle_event_filter(watched, event):
            return True
        return super().eventFilter(watched, event)

    def _apply_theme(self) -> None:
        self._theme = get_theme(self._settings.get("ui_theme"))
        self.setStyleSheet(window_style(self._theme))
        if self._plot.has_plot:
            plot_bg = get_plot_bg(self._theme.name)
            self._plot.figure.patch.set_facecolor(plot_bg)
            if self._plot.ax is not None:
                self._plot.ax.set_facecolor(plot_bg)
            self._plot.canvas.draw_idle()
        self.update()
        QApplication.processEvents()

    def _equalize_data_splitter(self) -> None:
        width = self._data_splitter.width()
        if width <= 0:
            return
        half = width // 2
        self._data_splitter.setSizes([half, width - half])

    def showEvent(self, event: QShowEvent) -> None:
        super().showEvent(event)
        self._equalize_data_splitter()

    def _preview_ui_theme(self, theme_name: str) -> None:
        self._settings["ui_theme"] = theme_name
        self._apply_theme()
        self._refresh_chat_view()

    def _open_settings(self) -> None:
        original_theme = self._settings.get("ui_theme")
        original_plot_settings = dict(self._settings.get("plot_settings", {}))
        original_download_settings = dict(self._settings.get("download_settings", {}))

        dlg = ChatSettingsDialog(
            plot_settings=self._settings.get("plot_settings"),
            download_settings=self._settings.get("download_settings"),
            parent=self,
            theme=self._theme.name,
        )
        dlg.theme_changed.connect(self._preview_ui_theme)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            self._settings["ui_theme"] = original_theme
            self._settings["plot_settings"] = original_plot_settings
            self._settings["download_settings"] = original_download_settings
            self._apply_theme()
            self._refresh_chat_view()
            return

        new_plot_settings = dict(dlg.plot_settings)
        plot_settings_changed = new_plot_settings != original_plot_settings

        self._settings["ui_theme"] = dlg.ui_theme_name
        self._settings["plot_settings"] = new_plot_settings
        self._settings["download_settings"] = dlg.download_settings
        save_settings(self._settings)
        self._apply_theme()
        self._refresh_chat_view()
        if (
            plot_settings_changed
            and self._last_dataframe is not None
            and not self._last_dataframe.is_empty()
        ):
            self._update_plot(self._last_dataframe)

    def _set_busy(self, busy: bool) -> None:
        self._send_btn.setDisabled(busy)
        self._prompt_input.setDisabled(busy)
        self._settings_btn.setDisabled(busy)
        self._status_label.setText("Thinking…" if busy else "Ready")
        self._status_label.setProperty("busy", busy)
        self._status_label.style().unpolish(self._status_label)
        self._status_label.style().polish(self._status_label)

    def _append_system_message(self, text: str) -> None:
        self._append_chat_block("System", text, muted=True)

    def _append_user_message(self, text: str) -> None:
        self._append_chat_block("You", text)

    def _append_assistant_message(self, text: str) -> None:
        timestamp = datetime.now().strftime("%H:%M")
        self._chat_log.append(f"**Assistant** · {timestamp}\n\n{text}")
        self._refresh_chat_view()

    def _append_chat_block(self, speaker: str, text: str, *, muted: bool = False) -> None:
        timestamp = datetime.now().strftime("%H:%M")
        prefix = f"*{speaker}* · {timestamp}" if muted else f"**{speaker}** · {timestamp}"
        self._chat_log.append(f"{prefix}\n\n{text}")
        self._refresh_chat_view()

    def _refresh_chat_view(self) -> None:
        scrollbar = self._chat_view.verticalScrollBar()
        at_bottom = scrollbar.value() >= scrollbar.maximum() - 20
        self._chat_view.setMarkdown("\n\n---\n\n".join(self._chat_log))
        if at_bottom:
            scrollbar.setValue(scrollbar.maximum())

    def _send_prompt(self) -> None:
        query = self._prompt_input.text().strip()
        if not query:
            return

        self._prompt_input.clear()
        self._append_user_message(query)
        self._set_busy(True)
        self._llm_worker.query_requested.emit(query)

    def _on_llm_response(self, answer: str, dataframe: object) -> None:
        self._append_assistant_message(answer)
        if isinstance(dataframe, pl.DataFrame):
            dataframe = coerce_date_columns(dataframe)
            self._last_dataframe = dataframe
            self._update_table(dataframe)
            self._update_plot(dataframe)
        self._set_busy(False)

    def _on_llm_error(self, message: str) -> None:
        self._append_system_message(f"Error: {message}")
        self._set_busy(False)

    def _update_table(self, df: pl.DataFrame) -> None:
        self._table.setSortingEnabled(False)
        self._table.clearContents()
        self._table.setRowCount(0)
        self._table.setColumnCount(0)

        if df.is_empty() and len(df.columns) == 0:
            return

        # Convert to pandas for easier cell access
        df_pd = df.to_pandas()
        
        self._table.setRowCount(len(df_pd))
        self._table.setColumnCount(len(df_pd.columns))
        self._table.setHorizontalHeaderLabels([str(c) for c in df_pd.columns])

        for row_idx in range(len(df_pd)):
            for col_idx, col_name in enumerate(df_pd.columns):
                value = df_pd.iloc[row_idx, col_idx]
                display_text = format_table_cell(value)
                item = QTableWidgetItem(display_text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                item.setData(Qt.ItemDataRole.UserRole, row_idx)
                self._table.setItem(row_idx, col_idx, item)

        self._table.resizeColumnsToContents()
        self._table.setSortingEnabled(True)

    def _update_plot(self, df: pl.DataFrame) -> None:
        if df.is_empty():
            self._plot.clear_plot()
            return

        x_col, y_cols = infer_plot_columns(df)
        if not y_cols:
            self._plot.clear_plot()
            return

        df_pd, y_cols, has_multiple_series = process_df_for_ggplot(df.to_pandas(), x_col)
        plot_settings = self._settings.get("plot_settings", {})

        plot = create_ggplot_from_df(
            df_pd,
            x_col,
            y_cols,
            "",
            None,
            has_multiple_series,
            plot_settings.get("geoms", "LP"),
            plot_settings.get("color", True),
            plot_settings.get("alpha", 1.0),
            plot_settings.get("size", False),
            plot_settings.get("shape", False),
            plot_settings.get("group", False),
            plot_settings.get("theme_str", "538"),
            plot_settings.get("flip_coord", False),
            plot_settings.get("y_scale", "linear"),
            plot_settings.get("pos_legend", "bottom"),
        )

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", PlotnineWarning)
                figure = plot.draw()
        except (IndexError, ValueError, TypeError, PlotnineError):
            plot = create_ggplot_from_df(
                df_pd,
                x_col,
                y_cols,
                "",
                None,
                has_multiple_series,
                plot_settings.get("geoms", "LP"),
                plot_settings.get("color", True),
                plot_settings.get("alpha", 1.0),
                plot_settings.get("size", False),
                plot_settings.get("shape", False),
                plot_settings.get("group", False),
                plot_settings.get("theme_str", "538"),
                plot_settings.get("flip_coord", False),
                "linear",
                plot_settings.get("pos_legend", "bottom"),
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", PlotnineWarning)
                try:
                    figure = plot.draw()
                except PlotnineError as exc:
                    self._status_label.setText(f"Plot error: {exc}")
                    return

        for axis in figure.axes:
            axis.xaxis.set_minor_locator(AutoMinorLocator(3))
            axis.yaxis.set_minor_locator(AutoMinorLocator(3))

        self._plot.display_figure(
            figure,
            series_names=y_cols,
            flip_coord=bool(plot_settings.get("flip_coord", False)),
            x_values=df_pd[x_col].tolist(),
        )

        # Enable action buttons now that we have data
        self._download_btn.setEnabled(True)
        self._copy_data_btn.setEnabled(True)
        self._copy_chart_btn.setEnabled(True)

    def _download_data(self) -> None:
        """Download the current dataframe to file based on settings."""
        if self._last_dataframe is None or self._last_dataframe.is_empty():
            QMessageBox.information(self, "No Data", "No data available to download.")
            return

        download_settings = self._settings.get("download_settings", {})
        file_format = download_settings.get("format", "csv")
        include_headers = download_settings.get("headers", True)
        directory = download_settings.get("directory") or str(Path.home())

        # Generate default filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"mcp_data_{timestamp}.{file_format}"

        # Determine file filter based on format
        if file_format == "csv":
            file_filter = "CSV Files (*.csv);;All Files (*.*)"
        else:
            file_filter = "Parquet Files (*.parquet);;All Files (*.*)"

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Data",
            str(Path(directory) / default_name),
            file_filter,
        )

        if not file_path:
            return

        try:
            if file_format == "csv":
                self._last_dataframe.write_csv(
                    file_path,
                    include_header=include_headers,
                )
            else:
                self._last_dataframe.write_parquet(file_path)

            QMessageBox.information(
                self,
                "Download Complete",
                f"Data saved to:\n{file_path}",
            )
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Download Failed",
                f"Could not save file:\n{exc}",
            )

    def _copy_table_data(self) -> None:
        """Copy table data to clipboard, respecting headers setting."""
        if self._last_dataframe is None or self._last_dataframe.is_empty():
            return

        download_settings = self._settings.get("download_settings", {})
        include_headers = download_settings.get("headers", True)

        # Convert to pandas for easy CSV export
        df_pd = self._last_dataframe.to_pandas()

        # Generate CSV string (excluding row numbers)
        csv_string = df_pd.to_csv(
            index=False,
            header=include_headers,
            sep="\t",
        )

        clipboard = QApplication.clipboard()
        clipboard.setText(csv_string)

    def _copy_chart_image(self) -> None:
        """Copy the current chart to clipboard as an image."""
        if not self._plot.has_plot or self._plot.figure is None:
            return

        from io import BytesIO

        import matplotlib.pyplot as plt
        from PySide6.QtGui import QImage

        # Save figure to buffer
        buffer = BytesIO()
        self._plot.figure.savefig(
            buffer,
            format="png",
            dpi=150,
            bbox_inches="tight",
            facecolor=self._plot.figure.get_facecolor(),
            edgecolor="none",
        )
        buffer.seek(0)

        # Convert to QImage for clipboard
        image = QImage()
        image.loadFromData(buffer.getvalue())

        clipboard = QApplication.clipboard()
        clipboard.setImage(image)

    def closeEvent(self, event: QCloseEvent) -> None:
        save_settings(self._settings)
        if self._worker_thread.isRunning():
            self._worker_thread.quit()
            self._worker_thread.wait(2000)
        super().closeEvent(event)


def main() -> None:
    from PySide6.QtGui import QFont

    app = QApplication(sys.argv)
    font = QFont("Segoe UI", 10)
    font.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)
    app.setFont(font)
    window = ChatDialog()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
