from . import llm, security, rag_bridge, client_db, tracing
from .state import current_question


_CAPABILITY_WORDS = [
    "что ты умеешь", "на что способен", "твои возможности", "твои способности",
    "что можешь", "как тебя использовать", "твои функции", "расскажи о себе",
    "кто ты", "представься", "твоя роль",
]
_CAPABILITY_ANSWER = (
    "Я виртуальный ассистент Банка по кредитованию малого и микробизнеса. Помогаю по таким вопросам:\n"
    "- условия кредитных продуктов, требования и ставки;\n"
    "- порядок подачи и статус заявок;\n"
    "- действующие кредиты, платежи и досрочное погашение;\n"
    "- реструктуризация.\n"
    "Вопросы по вашим данным доступны после авторизации. Спорные и нестандартные случаи передаю оператору."
)


def _is_capability_question(question):
    q = (question or "").lower()
    return any(w in q for w in _CAPABILITY_WORDS)


def classify_node(state):
    question = current_question(state)
    client_id = state.get("client_id")
    history = state.get("chat_history", [])

    with tracing.span("classify", input={"question": question, "client_id": client_id, "channel": state.get("channel")}) as sp:
        reason = security.detect_manipulation(question, authorized_client_id=client_id)
        if reason:
            updates = {
                "intent": "edge_manipulation",
                "security_flag": reason,
                "needs_rag": False, "needs_tools": False, "needs_escalation": False,
            }
            return tracing.append_span(state, sp, updates)

        auth = security.validate_authorization(question, authorized_client_id=client_id)
        if not auth.get("allowed"):
            updates = {
                "intent": "auth_required" if auth.get("reason") == "auth_required" else "edge_manipulation",
                "security_flag": auth.get("reason"),
                "requested_client_id": auth.get("requested_client_id"),
                "needs_rag": False, "needs_tools": False, "needs_escalation": False,
            }
            return tracing.append_span(state, sp, updates)

        intent = llm.classify(question, history=history, client_id=client_id)
        if not client_id and intent == "transactional":
            intent = "auth_required"

        updates = {
            "intent": intent,
            "security_flag": None,
            "requested_client_id": auth.get("requested_client_id"),
            "needs_rag": intent in ("info", "transactional"),
            "needs_tools": intent == "transactional",
            "needs_escalation": intent in ("escalation_sales", "escalation_negative"),
        }
        return tracing.append_span(state, sp, updates)


def route_after_classify(state):
    intent = state.get("intent", "info")
    if intent == "info":
        return "info"
    if intent == "transactional":
        return "transactional"
    if intent in ("escalation_sales", "escalation_negative"):
        return "escalation"
    if intent == "auth_required":
        return "rejection"
    if intent == "edge_manipulation":
        flag = state.get("security_flag")
        if flag == "third_party_data":
            return "rejection"
        if flag == "false_status":
            return "escalation"
        q = current_question(state).lower()
        wants_product = any(w in q for w in ["одобри", "кредит", "оформ", "исключени", "реструктуризац"])
        if flag == "prompt_injection" and wants_product:
            return "escalation"
        return "rejection"
    return "rejection"


def info_node(state):
    question = current_question(state)
    history = state.get("chat_history", [])
    with tracing.span("rag.info", input={"question": question}) as sp:
        if _is_capability_question(question):
            updates = {
                "draft_answer": _CAPABILITY_ANSWER,
                "sources": [],
                "rag_sources": [],
                "outcome_type": "info",
                "escalation": False,
            }
            return tracing.append_span(state, sp, updates)
        answer, sources = rag_bridge.rag_answer(question, history=history)
        updates = {
            "draft_answer": answer,
            "sources": sources,
            "rag_sources": sources,
            "outcome_type": "info",
            "escalation": False,
        }
        return tracing.append_span(state, sp, updates)


def _money(n):
    """Форматирует число с пробелами-разделителями: 3000000 -> '3 000 000'."""
    try:
        return f"{float(n):,.0f}".replace(",", " ")
    except Exception:
        return str(n)


def _is_repayment_question(q):
    """Это вопрос про полное досрочное погашение (нужен расчёт)?"""
    return any(w in q for w in [
        "досрочн", "полное погашение", "полностью погасить", "закрыть кредит",
        "сколько нужно для погаш", "погасить полностью", "погасить весь",
    ])


def _is_about_own_loan(q):
    """Вопрос именно про СВОЙ кредит (а не общий вопрос про правила досрочки)."""
    return any(w in q for w in ["я ", "мой", "моего", "мою", "моё", "свой", "своего", "мне", "у меня", "закрою", "погашу"])


def _format_transactional(kind, q, tool):
    """Собирает текст ответа и outcome_type из данных тула. Без LLM - детерминированно."""
    status = tool.get("status")
    if status == "need_auth":
        return ("Чтобы ответить по вашим данным, нужно авторизоваться через личный кабинет "
                "или мобильное приложение.", "clarification")
    if status == "access_denied":
        return ("Я не могу предоставлять данные другого клиента.", "rejection")
    if status == "not_found":
        if kind == "application":
            return ("По данным БД у вас нет поданных заявок.", "info")
        if kind == "credit":
            return ("По данным БД у вас нет действующих кредитов.", "info")
        return ("По данным БД информация не найдена.", "info")

    data = tool["data"]

    if kind == "application":
        name = client_db.product_name(data.get("product_code"))
        days = data.get("days_since_submitted")
        when = f", подана {days} дн. назад" if days is not None else ""
        return (
            f"По данным БД: заявка по продукту «{name}» на сумму {_money(data.get('amount_requested'))} ₽, "
            f"статус «{data.get('status')}»{when}.",
            "info",
        )

    if kind == "eligible":
        age = data.get("business_age_months")
        return (
            f"По данным БД: {data.get('legal_form')}, выручка {_money(data.get('annual_revenue'))} ₽/год, "
            f"скоринг {data.get('credit_score')}, бизнес ~{age} мес. "
            f"Доступность зависит от требований по каждому продукту (см. источники). "
            f"Точный персональный подбор подтвердит менеджер.",
            "info",
        )

    credits = data.get("credits", [])
    if not credits:
        return ("По данным БД у вас нет действующих кредитов.", "info")
    c = credits[0]
    name = c.get("product_name") or client_db.product_name(c.get("product_code"))
    outstanding = c.get("principal_outstanding")
    rate = c.get("interest_rate")

    if _is_repayment_question(q) and _is_about_own_loan(q):
        er = c.get("early_repayment") or {}
        return (
            f"По данным БД: {name}, остаток основного долга {_money(outstanding)} ₽, ставка {rate}%. "
            f"Для полного досрочного погашения на сегодня потребуется примерно {_money(er.get('total_to_repay'))} ₽ "
            f"(основной долг {_money(outstanding)} ₽ + проценты за {er.get('days_accrued')} дн. ≈ "
            f"{_money(er.get('accrued_interest'))} ₽). Точную сумму на дату погашения подтвердит банк.",
            "calculation",
        )

    return (
        f"По данным БД: действующий {name}, следующий платёж {c.get('next_payment_date')}, "
        f"сумма {_money(c.get('next_payment_amount'))} ₽, остаток основного долга {_money(outstanding)} ₽.",
        "info",
    )


def transactional_node(state):
    question = current_question(state)
    client_id = state.get("client_id")
    history = state.get("chat_history", [])

    if _is_capability_question(question):
        return info_node(state)

    with tracing.span("tools.transactional", input={"question": question, "client_id": client_id, "requested_client_id": state.get("requested_client_id")}) as sp:
        auth = security.validate_authorization(question, authorized_client_id=client_id)
        if not auth.get("allowed"):
            updates = {
                "draft_answer": security.safe_refusal_text(auth.get("reason")),
                "outcome_type": "clarification" if auth.get("reason") == "auth_required" else "rejection",
                "security_flag": auth.get("reason"),
                "escalation": False,
                "sources": [],
            }
            return tracing.append_span(state, sp, updates)

        q = question.lower()
        requested_client_id = auth.get("requested_client_id")

        if any(w in q for w in ["статус", "заявк", "рассматр", "одобрили", "решение по", "подал", "моя заявка", "заявку"]):
            tool = client_db.get_application_status(client_id, requested_client_id=requested_client_id)
            kind = "application"
        elif any(w in q for w in ["доступн", "подойд", "какие кредиты мне", "могу взять", "мне подход", "на что могу", "подобрать", "какой продукт"]):
            tool = client_db.get_client_profile(client_id, requested_client_id=requested_client_id)
            kind = "eligible"
        elif any(w in q for w in ["платёж", "платеж", "погаш", "остаток", "долг", "досрочн", "закрыть кредит", "сколько должен", "выплатить", "оставшуюся сумму"]):
            if "правила" in q and "досрочн" in q:
                updates = {
                    "draft_answer": "Этот вопрос относится к общим правилам досрочного погашения. Я могу ответить по нормативным документам, но для расчёта по вашему кредиту уточните, пожалуйста, авторизацию.",
                    "outcome_type": "info",
                    "escalation": False,
                    "sources": [],
                }
                return tracing.append_span(state, sp, updates)
            tool = client_db.get_active_credit_summary(client_id, requested_client_id=requested_client_id)
            kind = "credit"
        else:
            tool = client_db.get_active_credit_summary(client_id, requested_client_id=requested_client_id)
            kind = "credit"

        _, sources = rag_bridge.rag_answer(question, history=history)

        draft, outcome = _format_transactional(kind, q, tool)
        updates = {
            "draft_answer": draft,
            "outcome_type": outcome,
            "tool_result": tool,
            "sources": sources,
            "escalation": False,
        }
        return tracing.append_span(state, sp, updates)


def escalation_node(state):
    with tracing.span("escalation", input={"intent": state.get("intent"), "security_flag": state.get("security_flag")}) as sp:
        intent = state.get("intent", "")
        flag = state.get("security_flag")
        if intent == "edge_manipulation":
            reason = "запрос требует проверки оператором (" + str(flag) + ")"
            text = (
                "Этот запрос я не могу обработать автоматически. Передаю обращение оператору - "
                "он проверит и поможет в рамках регламента."
            )
        elif intent == "escalation_sales":
            reason = "намерение оформить продукт"
            text = (
                "Передаю ваше обращение менеджеру для оформления - он свяжется с вами. "
                "Я не оформляю продукты сам, но подготовлю контекст вашего запроса."
            )
        else:
            reason = "негатив / просьба оператора"
            text = (
                "Понимаю вас. Передаю обращение оператору - он подключится и поможет разобраться."
            )
        updates = {
            "draft_answer": text,
            "outcome_type": "escalation",
            "escalation": True,
            "escalation_reason": reason,
            "sources": [],
        }
        return tracing.append_span(state, sp, updates)


def rejection_node(state):
    with tracing.span("rejection", input={"intent": state.get("intent"), "security_flag": state.get("security_flag")}) as sp:
        intent = state.get("intent", "")
        flag = state.get("security_flag")

        if intent == "auth_required" or flag == "auth_required":
            text = security.safe_refusal_text("auth_required")
            outcome = "clarification"
        elif intent == "offtopic":
            text = (
                "Это вне моей компетенции - я консультирую по вопросам кредитования "
                "малого и микробизнеса. Готов помочь по кредитам, заявкам и условиям."
            )
            outcome = "info"
        elif intent == "edge_no_data":
            text = (
                "В нормативных документах Банка нет данных по этому вопросу. "
                "Уточните детали или обратитесь к оператору - подскажу, что в моей компетенции."
            )
            outcome = "info"
        elif flag:
            text = security.safe_refusal_text(flag)
            outcome = "rejection"
        else:
            text = "Этот запрос вне моей компетенции. Помогу по вопросам кредитования МСБ."
            outcome = "rejection"

        updates = {
            "draft_answer": text,
            "outcome_type": outcome,
            "escalation": False,
            "sources": [],
        }
        return tracing.append_span(state, sp, updates)


def answer_node(state):
    with tracing.span("answer", input={"outcome_type": state.get("outcome_type"), "sources": state.get("sources", [])}) as sp:
        trace = state.get("trace", [])
        updates = {
            "final_answer": state.get("draft_answer", ""),
            "outcome_type": state.get("outcome_type", "info"),
            "escalation": state.get("escalation", False),
            "sources": state.get("sources", []),
            "trace_summary": tracing.trace_summary(trace),
        }
        return tracing.append_span(state, sp, updates)
