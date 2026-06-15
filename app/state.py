from typing import TypedDict, Optional, List, Dict, Any

from .tracing import new_trace_id


class AgentState(TypedDict, total=False):
    question: str
    client_id: Optional[str]
    channel: str
    chat_history: List[Dict[str, str]]

    intent: str
    needs_rag: bool
    needs_tools: bool
    needs_escalation: bool
    security_flag: Optional[str]
    requested_client_id: Optional[str]

    rag_context: str
    rag_sources: List[str]
    tool_result: Dict[str, Any]
    draft_answer: str

    final_answer: str
    outcome_type: str
    escalation: bool
    escalation_reason: str
    sources: List[str]

    trace_id: str
    trace: List[Dict[str, Any]]
    trace_summary: Dict[str, Any]


def make_initial_state(
    question: str,
    client_id: Optional[str] = None,
    channel: str = "chat_site",
    chat_history: Optional[List[Dict[str, str]]] = None,
) -> AgentState:
    """Собирает стартовый state из входных данных. Историю по умолчанию делаем пустой."""
    return {
        "question": question,
        "client_id": client_id,
        "channel": channel,
        "chat_history": chat_history or [],
        "trace_id": new_trace_id(),
        "trace": [],
    }


def current_question(state) -> str:
    """Текущий вопрос клиента.

    В single-turn это поле question. В multi-turn у части кейсов question пустой,
    а реальная реплика лежит последней в history (role=client). Тогда берём её.
    История может использовать ключ 'text' или 'content' - учитываем оба.
    """
    q = (state.get("question") or "").strip()
    if q:
        return q
    for msg in reversed(state.get("chat_history") or []):
        if msg.get("role") in ("client", "user"):
            return (msg.get("text") or msg.get("content") or "").strip()
    return ""
