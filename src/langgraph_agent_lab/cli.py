"""CLI for the lab."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Annotated, Any

import typer
import yaml

from .graph import build_graph
from .metrics import MetricsReport, metric_from_state, summarize_metrics, write_metrics
from .persistence import build_checkpointer
from .report import write_report
from .scenarios import load_scenarios
from .state import initial_state

app = typer.Typer(no_args_is_help=True)


def _verify_resume(
    checkpointer_kind: str,
    database_url: str | None,
    sample_run_config: dict[str, Any],
    expected_final_answer: str | None,
) -> bool:
    """Simulate a process restart: rebuild graph from the same checkpointer and read state back.

    Returns True if the fresh graph can recover the prior thread's final state — proves the
    checkpointer durably persisted it. Only meaningful for non-memory checkpointers, since
    MemorySaver state lives only inside the current process.
    """
    if checkpointer_kind == "memory" or checkpointer_kind == "none":
        return False
    fresh_cp = build_checkpointer(checkpointer_kind, database_url)
    fresh_graph = build_graph(checkpointer=fresh_cp)
    snapshot = fresh_graph.get_state(sample_run_config)
    if snapshot is None or not snapshot.values:
        return False
    recovered = snapshot.values.get("final_answer") or snapshot.values.get("pending_question")
    return bool(recovered) and recovered == expected_final_answer


@app.command("run-scenarios")
def run_scenarios(
    config: Annotated[Path, typer.Option("--config")],
    output: Annotated[Path, typer.Option("--output")],
) -> None:
    """Run all grading scenarios and write metrics JSON."""
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    checkpointer_kind = cfg.get("checkpointer", "memory")
    database_url = cfg.get("database_url")

    # Reset SQLite DB at the start of each run so metrics are reproducible.
    if checkpointer_kind == "sqlite" and database_url:
        db_path = Path(database_url)
        if db_path.exists():
            db_path.unlink()

    scenarios = load_scenarios(cfg["scenarios_path"])
    checkpointer = build_checkpointer(checkpointer_kind, database_url)
    graph = build_graph(checkpointer=checkpointer)

    metrics = []
    sample_run_config: dict[str, Any] | None = None
    sample_final_answer: str | None = None
    for scenario in scenarios:
        state = initial_state(scenario)
        run_config = {"configurable": {"thread_id": state["thread_id"]}}
        t0 = time.perf_counter()
        final_state = graph.invoke(state, config=run_config)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        metrics.append(
            metric_from_state(
                final_state,
                scenario.expected_route.value,
                scenario.requires_approval,
                latency_ms=elapsed_ms,
            )
        )
        if sample_run_config is None:
            sample_run_config = run_config
            sample_final_answer = final_state.get("final_answer") or final_state.get("pending_question")

    report = summarize_metrics(metrics)
    if sample_run_config is not None:
        report.resume_success = _verify_resume(
            checkpointer_kind, database_url, sample_run_config, sample_final_answer
        )
    write_metrics(report, output)
    if cfg.get("report_path"):
        write_report(report, cfg["report_path"])
    typer.echo(f"Wrote metrics to {output}")


@app.command("validate-metrics")
def validate_metrics(metrics: Annotated[Path, typer.Option("--metrics")]) -> None:
    """Validate metrics JSON schema for grading."""
    payload = json.loads(metrics.read_text(encoding="utf-8"))
    report = MetricsReport.model_validate(payload)
    if report.total_scenarios < 6:
        raise typer.BadParameter("Expected at least 6 scenarios")
    typer.echo(f"Metrics valid. success_rate={report.success_rate:.2%}")


if __name__ == "__main__":
    app()
