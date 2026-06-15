"""SQLite-хранилище трейсов агента.

Модуль сохраняет каждый прогон агента в локальную SQLite-базу. Он намеренно отделён
от бизнес/клиентских данных: трейсы - это observability-данные разработчика, а не память
диалога и не источник истины для ответов клиенту.
"""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_DB_PATH = _PROJECT_ROOT / "logs" / "traces.sqlite"


def get_trace_db_path() -> Path:
    """Возвращает путь к базе трейсов.

    Можно переопределить через TRACE_DB_PATH=/custom/path/traces.sqlite.
    Относительные пути считаются от корня проекта.
    """
    raw = os.getenv("TRACE_DB_PATH")
    if not raw:
        return _DEFAULT_DB_PATH
    p = Path(raw)
    return p if p.is_absolute() else (_PROJECT_ROOT / p)


def trace_db_enabled() -> bool:
    """Позволяет отключить персистентность в тестах или чувствительных окружениях."""
    return os.getenv("TRACE_DB_ENABLED", "1") != "0"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _connect(db_path: Optional[Path] = None) -> sqlite3.Connection:
    path = db_path or get_trace_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_trace_db(db_path: Optional[Path] = None) -> None:
    """Создаёт таблицы трейсов, если их ещё нет."""
    with _connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS trace_runs (
                trace_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                question TEXT,
                client_id TEXT,
                channel TEXT,
                intent TEXT,
                outcome_type TEXT,
                escalation INTEGER NOT NULL DEFAULT 0,
                security_flag TEXT,
                total_duration_ms REAL,
                steps INTEGER,
                answer_preview TEXT,
                sources_json TEXT,
                tool_result_json TEXT,
                trace_summary_json TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS trace_spans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trace_id TEXT NOT NULL,
                span_index INTEGER NOT NULL,
                name TEXT NOT NULL,
                started_at TEXT,
                duration_ms REAL,
                status TEXT,
                input_json TEXT,
                output_json TEXT,
                error TEXT,
                FOREIGN KEY(trace_id) REFERENCES trace_runs(trace_id) ON DELETE CASCADE
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_trace_spans_trace_id ON trace_spans(trace_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_trace_runs_created_at ON trace_runs(created_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_trace_runs_outcome ON trace_runs(outcome_type)")


def save_trace_run(
    *,
    trace_id: str,
    question: str,
    client_id: Optional[str],
    channel: str,
    intent: Optional[str],
    outcome_type: Optional[str],
    escalation: bool,
    security_flag: Optional[str],
    answer: str,
    sources: Iterable[Any] | None,
    tool_result: Any,
    trace: List[Dict[str, Any]] | None,
    trace_summary: Dict[str, Any] | None,
    db_path: Optional[Path] = None,
) -> None:
    """Сохраняет полный прогон трейса и его spans.

    Функция идемпотентна по trace_id: при повторном сохранении старые spans
    заменяются на актуальный список.
    """
    if not trace_db_enabled():
        return

    trace = trace or []
    trace_summary = trace_summary or {}
    init_trace_db(db_path)

    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO trace_runs (
                trace_id, created_at, question, client_id, channel, intent,
                outcome_type, escalation, security_flag, total_duration_ms,
                steps, answer_preview, sources_json, tool_result_json,
                trace_summary_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                trace_id,
                _utc_now(),
                question,
                client_id,
                channel,
                intent,
                outcome_type,
                1 if escalation else 0,
                security_flag,
                trace_summary.get("total_duration_ms"),
                trace_summary.get("steps", len(trace)),
                (answer or "")[:500],
                _json(list(sources or [])),
                _json(tool_result),
                _json(trace_summary),
            ),
        )
        conn.execute("DELETE FROM trace_spans WHERE trace_id = ?", (trace_id,))
        for i, span in enumerate(trace):
            conn.execute(
                """
                INSERT INTO trace_spans (
                    trace_id, span_index, name, started_at, duration_ms,
                    status, input_json, output_json, error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trace_id,
                    i,
                    span.get("name", "unknown"),
                    span.get("started_at"),
                    span.get("duration_ms"),
                    span.get("status"),
                    _json(span.get("input", {})),
                    _json(span.get("output", {})),
                    span.get("error"),
                ),
            )


def load_trace(trace_id: str, db_path: Optional[Path] = None) -> Dict[str, Any] | None:
    """Читает один сохранённый трейс с его spans. Полезно для локальной отладки."""
    init_trace_db(db_path)
    with _connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        run = conn.execute("SELECT * FROM trace_runs WHERE trace_id = ?", (trace_id,)).fetchone()
        if not run:
            return None
        spans = conn.execute(
            "SELECT * FROM trace_spans WHERE trace_id = ? ORDER BY span_index",
            (trace_id,),
        ).fetchall()
    return {
        "run": dict(run),
        "spans": [dict(s) for s in spans],
    }
