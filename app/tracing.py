"""Лёгкая трассировка агента кредитования МСБ.

Трейсер без внешних зависимостей: каждый узел возвращает свой span в state['trace'],
поэтому трейсы доступны и в локальном демо, и в eval-прогонах без LangSmith/OpenTelemetry.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from time import perf_counter
from typing import Any, Dict, Iterator, List, Optional
from uuid import uuid4


def new_trace_id() -> str:
    return uuid4().hex[:12]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _safe_value(value: Any, max_len: int = 800) -> Any:
    """Делает значения безопасными для трейса: короткими, JSON-сериализуемыми, без больших payload."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, str) and len(value) > max_len:
            return value[:max_len] + "..."
        return value
    if isinstance(value, (list, tuple)):
        return [_safe_value(v, max_len=max_len) for v in value[:20]]
    if isinstance(value, dict):
        return {str(k): _safe_value(v, max_len=max_len) for k, v in list(value.items())[:40]}
    return str(value)[:max_len]


@dataclass
class Span:
    name: str
    started_at: str = field(default_factory=_utc_now)
    duration_ms: Optional[float] = None
    status: str = "ok"
    input: Dict[str, Any] = field(default_factory=dict)
    output: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "name": self.name,
            "started_at": self.started_at,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "input": _safe_value(self.input),
            "output": _safe_value(self.output),
        }
        if self.error:
            data["error"] = self.error
        return data


@contextmanager
def span(name: str, *, input: Optional[Dict[str, Any]] = None) -> Iterator[Span]:
    s = Span(name=name, input=input or {})
    t0 = perf_counter()
    try:
        yield s
    except Exception as exc:
        s.status = "error"
        s.error = f"{exc.__class__.__name__}: {exc}"
        raise
    finally:
        s.duration_ms = round((perf_counter() - t0) * 1000, 2)


def append_span(state: Dict[str, Any], span_obj: Span, updates: Dict[str, Any]) -> Dict[str, Any]:
    """Возвращает обновления узла с добавленным span в state['trace']."""
    span_obj.output = _safe_value(updates)
    trace = list(state.get("trace") or [])
    trace.append(span_obj.to_dict())
    enriched = dict(updates)
    enriched["trace"] = trace
    enriched["trace_id"] = state.get("trace_id") or new_trace_id()
    return enriched


def trace_summary(trace: List[Dict[str, Any]] | None) -> Dict[str, Any]:
    trace = trace or []
    return {
        "steps": len(trace),
        "total_duration_ms": round(sum(float(s.get("duration_ms") or 0) for s in trace), 2),
        "errors": [s for s in trace if s.get("status") == "error"],
        "nodes": [s.get("name") for s in trace],
    }


def format_trace(trace: List[Dict[str, Any]] | None) -> str:
    trace = trace or []
    if not trace:
        return "trace: empty"
    lines = ["trace:"]
    for s in trace:
        status = s.get("status", "ok")
        dur = s.get("duration_ms", 0)
        name = s.get("name", "unknown")
        out = s.get("output") or {}
        compact = []
        for key in ("intent", "security_flag", "outcome_type", "escalation"):
            if key in out and out[key] not in (None, ""):
                compact.append(f"{key}={out[key]}")
        suffix = "  " + ", ".join(compact) if compact else ""
        lines.append(f"- {name}: {status}, {dur} ms{suffix}")
    return "\n".join(lines)
