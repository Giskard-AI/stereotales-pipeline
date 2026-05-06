import logging
import logging.config
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.layout import Layout
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)

LOG_FILE = "run.log"

_dashboard_progress: dict[str, Any] = {}
_dashboard_task_ids: dict[str, Any] = {}
_status_lines: deque = deque(maxlen=20)
EVAL_STATS: dict[str, Any] = {}


class RichStatusHandler(logging.Handler):
    """Forwards log records to the dashboard status panel."""

    def emit(self, record: logging.LogRecord) -> None:
        ts = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
        level_colors = {
            "DEBUG": "dim",
            "INFO": "white",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold red",
        }
        color = level_colors.get(record.levelname, "white")
        _status_lines.append(f"[{ts}] [{color}]{self.format(record)}[/{color}]")


def setup_log(log_path: Path, level: str = "INFO") -> None:
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "loggers": {"httpx": {"level": "WARNING"}, "LiteLLM": {"level": "WARNING"}},
        }
    )
    log_file_path = log_path / LOG_FILE
    if log_file_path.is_file():
        rotated = "run-" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".log"
        log_file_path.rename(log_path / rotated)
    log_file_path.unlink(missing_ok=True)
    logging.basicConfig(
        level=level,
        filename=str(log_file_path),
        format="%(asctime)s %(levelname)-8s %(name)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def make_layout() -> Layout:
    layout = Layout()
    layout.split_column(Layout(name="header", size=4), Layout(name="main", ratio=1))
    layout["main"].split_row(Layout(name="stats"), Layout(name="status", ratio=2))
    return layout


def setup_eval_stats(
    layout: Layout,
    run_dir: Path,
    model_names: list[str],
    total: int,
    per_model_totals: dict[str, int],
) -> None:
    global _dashboard_progress, _dashboard_task_ids

    rich_handler = RichStatusHandler()
    rich_handler.setFormatter(logging.Formatter("%(name)s - %(message)s"))
    logging.getLogger().addHandler(rich_handler)

    start = datetime.now()
    EVAL_STATS["start"] = start
    EVAL_STATS["run_dir"] = str(run_dir)
    EVAL_STATS["total"] = total
    EVAL_STATS["completed"] = 0
    EVAL_STATS["skipped"] = 0
    EVAL_STATS["errors"] = 0
    EVAL_STATS["per_model"] = {
        model: {"completed": 0, "skipped": 0, "errors": 0} for model in model_names
    }

    overall = Progress(
        "{task.description}",
        SpinnerColumn(),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(elapsed_when_finished=True),
    )
    _dashboard_task_ids["overall"] = overall.add_task("Overall", total=total)
    _dashboard_progress["overall"] = overall

    per_model = Progress(
        "{task.description}",
        SpinnerColumn(),
        MofNCompleteColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(elapsed_when_finished=True),
    )
    _dashboard_task_ids["evaluators"] = {}
    for name in model_names:
        display_name = (name[:50] + "...") if len(name) > 50 else name
        _dashboard_task_ids["evaluators"][name] = per_model.add_task(
            display_name, total=per_model_totals.get(name, 0)
        )
    _dashboard_progress["evaluators"] = per_model

    inner = Layout()
    inner.split_column(
        Layout(Panel(overall, title="Overall", border_style="green", padding=(1, 1)), size=5),
        Layout(Panel(per_model, title="Evaluators", border_style="blue", padding=(1, 1)), ratio=1),
    )
    layout["main"]["stats"].update(Panel(inner, title="[b]Progress", border_style="red", padding=(1, 1)))
    layout["header"].update(
        Panel(
            f"Run dir: {run_dir}\nStarted: {start}, running since: {datetime.now() - start}",
            title="Summary",
            border_style="blue",
        )
    )


def update_display(layout: Layout) -> None:
    completed = EVAL_STATS.get("completed", 0)
    skipped = EVAL_STATS.get("skipped", 0)
    errors = EVAL_STATS.get("errors", 0)
    done = completed + skipped + errors
    _dashboard_progress["overall"].update(_dashboard_task_ids["overall"], completed=done)

    per_model_stats = EVAL_STATS.get("per_model", {})
    for model, task_id in _dashboard_task_ids.get("evaluators", {}).items():
        stats = per_model_stats.get(model, {})
        model_done = stats.get("completed", 0) + stats.get("skipped", 0) + stats.get("errors", 0)
        _dashboard_progress["evaluators"].update(task_id, completed=model_done)

    if EVAL_STATS.get("start"):
        layout["header"].update(
            Panel(
                f"Run dir: {EVAL_STATS['run_dir']}\n"
                f"Started: {EVAL_STATS['start']}, running since: {datetime.now() - EVAL_STATS['start']}\n"
                f"Completed: {completed} | Skipped: {skipped} | Errors: {errors}",
                title="Summary",
                border_style="blue",
            )
        )

    debug = logging.getLogger().isEnabledFor(logging.DEBUG)
    status_text = "\n".join(_status_lines) if _status_lines else "Running..."
    layout["main"]["status"].update(
        Panel(
            status_text,
            title="[b]Status (debug)" if debug else "[b]Status",
            border_style="green",
            padding=(1, 1),
        )
    )
