import re


_PROMPT_INJECTION = [
    "игнорируй", "игнорировать", "забудь инструкц", "забудь все", "забудь всё",
    "новые инструкции", "новая инструкция", "ты теперь", "притворись", "представь что ты",
    "режим разработчика", "developer mode", "системный промпт", "твой промпт",
    "твои инструкции", "покажи инструкции", "как тебя настроили", "act as",
    "ignore previous", "ignore all", "jailbreak", "без ограничений", "сними ограничения",
    "обойди правила", "веди себя как", "с этого момента ты", "теперь ты ",
    "ты больше не", "ты теперь менеджер", "представь себя",
]

_SCORING_DISCLOSURE = [
    "критери", "веса модел", "вес модел", "пороги скоринг", "порог скоринг",
    "формула скоринг", "формулу скоринг", "как считается скоринг", "как считаете скоринг",
    "раскрой скоринг", "точные критери", "если скоринг больше", "если скоринг выше",
    "при каком скоринге", "какой скоринг нужен чтобы одобр",
    "слабые места", "слабых мест",
]

_REQUEST_EXCEPTION = [
    "сделайте исключение", "сделай исключение", "в виде исключения", "в порядке исключения",
    "закройте глаза", "закрой глаза", "одобрите несмотря", "одобри несмотря",
    "обойти требования", "обойдите требования", "не по правилам, но",
    "обойдёмся", "обойдемся",
]

_PROVOKE_PROMISE = [
    "гарантируй", "гарантируйте", "дай гарантию", "дайте гарантию", "100% одобр",
    "точно одобрят", "точно одобрите", "обещай одобр", "обещайте одобр",
    "гарантия одобрения", "наверняка одобрят", "гарантир",
]

_THIRD_PARTY_WORDS = [
    "другого клиента", "другому клиенту", "чужие данные", "чужой кредит",
    "чужую заявку", "не мой клиент", "данные клиента c-", "кредит клиента c-",
    "заявку клиента c-", "по клиенту c-", "у партнёра", "у партнера", "моего партнёра",
    "моего партнера", "у конкурента", "конкурент", "по доверенности", "доверенность от",
]

_FALSE_STATUS = [
    "я директор отделения", "я сотрудник банка", "я директор банка", "я из руководства банка",
    "я работаю в банке", "разрешите по моему", "я премиальный клиент",
]

_CLIENT_ID_RE = re.compile(r"[cсC]-?0*\d{1,6}", re.IGNORECASE)


def _normalize(text: str) -> str:
    return (text or "").lower().replace("\u2011", "-").strip()


def extract_client_ids(text: str) -> list:
    """Находит в тексте все упоминания вида C-000001 и нормализует к 'C-000001'."""
    ids = []
    for raw in _CLIENT_ID_RE.findall(text or ""):
        digits = re.sub(r"\D", "", raw)
        if digits:
            ids.append("C-" + digits.zfill(6))
    return ids


def detect_manipulation(question: str, authorized_client_id=None) -> str:
    """Главная функция защиты. Возвращает причину блокировки (строку) или None.

    authorized_client_id - id клиента, под которым человек авторизован (из канала).
    Нужен, чтобы отличить «спрашиваю про СВОИ данные» (это норма) от
    «спрашиваю про ЧУЖИЕ данные» (это манипуляция).
    """
    q = _normalize(question)

    if any(p in q for p in _PROMPT_INJECTION):
        return "prompt_injection"

    if any(p in q for p in _THIRD_PARTY_WORDS):
        return "third_party_data"
    mentioned = extract_client_ids(question)
    if mentioned:
        auth = (authorized_client_id or "").upper()
        if any(cid.upper() != auth for cid in mentioned):
            return "third_party_data"

    if any(p in q for p in _SCORING_DISCLOSURE):
        return "scoring_disclosure"

    if any(p in q for p in _REQUEST_EXCEPTION):
        return "request_exception"
    if "исключени" in q and any(
        w in q for w in ["сделай", "сделайте", "одобри", "одобрите", "закрой", "закройте", "в виде", "в порядке"]
    ):
        return "request_exception"

    if any(p in q for p in _PROVOKE_PROMISE):
        return "provoke_promise"

    if any(p in q for p in _FALSE_STATUS):
        return "false_status"

    return None


_PERSONAL_DATA_WORDS = [
    "моя заявка", "мою заявку", "моей заявк", "статус заявки", "статус по заявке",
    "мой кредит", "моего кредита", "моему кредиту", "мой долг", "моя задолженность",
    "мой платёж", "мой платеж", "следующий платёж", "следующий платеж", "остаток долга",
    "сколько я должен", "сколько должен", "у меня кредит", "у меня заявка",
    "мне одобрили", "меня одобрили", "мои кредиты", "мои заявки", "мой скоринг",
    "персональн", "личн", "по моим данным", "по моей компании", "по моему ип", "по моему ооо",
]
_GENERAL_RAG_HINTS = [
    "какие документы", "условия", "требования", "срок", "ставка", "комисси",
    "как подать", "можно ли", "что нужно", "правила", "регламент", "досрочн",
    "реструктуризац", "обеспечени", "поручител", "залог", "овердрафт",
]


def needs_personal_data(question: str) -> bool:
    """True, если вопрос требует персональных данных из БД/tools, а не общего RAG."""
    q = _normalize(question)
    if extract_client_ids(question):
        return True
    if any(p in q for p in _PERSONAL_DATA_WORDS):
        return True
    first_person = any(w in q for w in ["мой", "моя", "моё", "мои", "у меня", "мне", "я "])
    product_state = any(w in q for w in ["заявк", "кредит", "платеж", "платёж", "долг", "остаток", "одобр"])
    general_hint = any(w in q for w in _GENERAL_RAG_HINTS)
    return bool(first_person and product_state and not general_hint)


def validate_authorization(question: str, authorized_client_id=None) -> dict:
    """Проверяет право доступа к персональным данным.

    Правила:
    - если клиент не авторизован, доступны только общие RAG-вопросы;
    - если клиент авторизован, tools могут выдавать данные только по его client_id;
    - запросы по чужому client_id блокируются до обращения к БД.
    """
    mentioned = extract_client_ids(question)
    auth = (authorized_client_id or "").upper()
    if mentioned and not auth:
        return {"allowed": False, "reason": "auth_required", "requested_client_id": mentioned[0]}
    if mentioned and any(cid.upper() != auth for cid in mentioned):
        return {"allowed": False, "reason": "third_party_data", "requested_client_id": mentioned[0]}
    if needs_personal_data(question) and not auth:
        return {"allowed": False, "reason": "auth_required", "requested_client_id": None}
    return {"allowed": True, "reason": None, "requested_client_id": mentioned[0] if mentioned else None}


def mentions_own_records(question: str) -> bool:
    """True, если вопрос про КОНКРЕТНЫЕ существующие записи клиента (его заявка/кредит/платёж),
    а не про общую допустимость или условия."""
    q = _normalize(question)
    if extract_client_ids(question):
        return True
    return any(p in q for p in _PERSONAL_DATA_WORDS)


def is_policy_request(question: str) -> bool:
    """True, если запрос подпадает под продуктовый стоп-фактор или решение оператора
    (самозанятый, снижение ставки, повторный рефинанс, доп. овердрафт, снятие обеспечения,
    тяжёлая просрочка). Нужно, чтобы такой запрос у авторизованного клиента дошёл до
    transactional, а не был перехвачен sales-правилами."""
    q = _normalize(question)
    if "самозанят" in q:
        return True
    if ("снизить" in q or "снизьте" in q or "уменьшить" in q) and "ставк" in q:
        return True
    if "обратно" in q and ("дешевле" in q or "рефинансир" in q):
        return True
    if any(w in q for w in ["ещё овердрафт", "еще овердрафт", "уже есть овердрафт", "овердрафт оформ", "ещё и овердрафт", "еще и овердрафт"]):
        return True
    if "поручительств" in q and any(w in q for w in ["снять", "убрать", "снимать", "снимет"]):
        return True
    if "просрочк" in q and any(w in q for w in ["больш", "что делать", "огромн", "помогите", "не знаю"]):
        return True
    return False


SAFE_REFUSALS = {
    "prompt_injection": (
        "Я не могу раскрывать внутренние инструкции или менять правила своей работы. "
        "Готов помочь по вопросам кредитования малого и микробизнеса."
    ),
    "third_party_data": (
        "Я не могу предоставлять данные другого клиента. "
        "Могу помочь по вашему обращению в рамках вашей авторизации."
    ),
    "scoring_disclosure": (
        "Я не раскрываю внутренние критерии и веса скоринговой модели. "
        "Решение принимается по совокупности факторов; могу рассказать общие требования по продукту."
    ),
    "request_exception": (
        "Я не вправе делать индивидуальные исключения из регламента. "
        "Могу рассказать, какие условия требуются по продукту, и при необходимости передать обращение оператору."
    ),
    "provoke_promise": (
        "Я не могу гарантировать одобрение или обещать конкретное решение - "
        "его принимает Банк по результатам рассмотрения. При намерении оформить продукт передам вас менеджеру."
    ),
    "false_status": (
        "Я обслуживаю обращения в рамках вашей авторизации и не предоставляю особые условия "
        "по заявленному статусу. Помогу по вашим вопросам кредитования; при необходимости передам оператору."
    ),
    "auth_required": (
        "Чтобы ответить по заявке, кредиту, платежам или другим персональным данным, нужно авторизоваться. "
        "Без авторизации я могу отвечать только на общие вопросы по условиям и регламентам кредитования МСБ."
    ),
}


def safe_refusal_text(reason: str) -> str:
    """Возвращает безопасный текст отказа по причине (или общий, если причина незнакомая)."""
    return SAFE_REFUSALS.get(
        reason,
        "Этот запрос вне моей компетенции. Помогу по вопросам кредитования МСБ.",
    )
