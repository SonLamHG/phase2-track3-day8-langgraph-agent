"""Checkpointer adapter.

Memory: zero-config, in-process — fine for tests and dev.
SQLite: durable across process restarts — used for crash-resume / time-travel demos.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any


def build_checkpointer(kind: str = "memory", database_url: str | None = None) -> Any | None:
    if kind == "none":
        return None
    if kind == "memory":
        from langgraph.checkpoint.memory import MemorySaver

        return MemorySaver()
    if kind == "sqlite":
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver
        except ImportError as exc:
            raise RuntimeError("SQLite checkpointer requires: pip install langgraph-checkpoint-sqlite") from exc
        db_path = database_url or "checkpoints.db"
        # Note: SqliteSaver.from_conn_string() returns a context manager, not a checkpointer.
        # Construct directly with a connection so the checkpointer is usable outside `with`.
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        return SqliteSaver(conn=conn)
    if kind == "postgres":
        try:
            from langgraph.checkpoint.postgres import PostgresSaver
        except ImportError as exc:
            raise RuntimeError("Postgres checkpointer requires: pip install langgraph-checkpoint-postgres") from exc
        return PostgresSaver.from_conn_string(database_url or "")
    raise ValueError(f"Unknown checkpointer kind: {kind}")
