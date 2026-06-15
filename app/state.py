# app/state.py — общее состояние агента, которое LangGraph передаёт между узлами.
#
# В LangGraph каждый узел получает state (словарь) и возвращает словарь с теми
# полями, которые он поменял. LangGraph сам сливает их в общий state.
# Мы описываем форму этого словаря через TypedDict, чтобы было понятно,
# какие поля есть и что они значат. total=False означает «все поля
# необязательны» — узлы заполняют их постепенно.

from typing import TypedDict, Optional, List, Dict, Any


class AgentState(TypedDict, total=False):
    # ---- вход (приходит от бота / eval / qa.jsonl) ----
    question: str                 # текущая реплика клиента
    client_id: Optional[str]      # ID авторизованного клиента из канала; None для анонима
    channel: str                  # chat_intern / chat_site / mobile / contact_center
    chat_history: List[Dict[str, str]]   # история диалога [{"role": ..., "content": ...}]

    # ---- заполняет classify_node ----
    intent: str                   # info / transactional / escalation_sales / escalation_negative /
                                  #   edge_no_data / edge_manipulation / offtopic
    needs_rag: bool               # нужен ли поиск по документам
    needs_tools: bool             # нужен ли доступ к БД клиента
    needs_escalation: bool        # нужно ли передавать оператору
    security_flag: Optional[str]  # причина блокировки, если сработала защита (см. security.py)

    # ---- заполняют рабочие узлы (info / transactional / escalation / rejection) ----
    rag_context: str              # текст найденных фрагментов документов
    rag_sources: List[str]        # ссылки-источники вида "01_credit_products.md#2.1.2"
    tool_result: Dict[str, Any]   # что вернули tools к БД (см. client_db.py)
    draft_answer: str             # черновой ответ узла (до финальной сборки)

    # ---- финал (собирает answer_node) ----
    final_answer: str             # текст ответа клиенту
    outcome_type: str             # info / calculation / escalation / rejection / clarification
    escalation: bool              # был ли это перевод на оператора
    escalation_reason: str        # причина эскалации (заполняет escalation_node)
    sources: List[str]            # источники для финального ответа


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
    }


def current_question(state) -> str:
    """Текущий вопрос клиента.

    В single-turn это поле question. В multi-turn у части кейсов question пустой,
    а реальная реплика лежит последней в history (role=client). Тогда берём её.
    История может использовать ключ 'text' или 'content' — учитываем оба.
    """
    q = (state.get("question") or "").strip()
    if q:
        return q
    for msg in reversed(state.get("chat_history") or []):
        if msg.get("role") in ("client", "user"):
            return (msg.get("text") or msg.get("content") or "").strip()
    return ""
