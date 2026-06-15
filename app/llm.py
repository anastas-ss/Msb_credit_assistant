import json
import os
import re
from dotenv import load_dotenv
load_dotenv()
from .prompts import SYSTEM_PROMPT, ROUTER_PROMPT
from . import security

USE_REAL_LLM = True

VALID_INTENTS = {
    "info", "transactional", "escalation_sales", "escalation_negative",
    "edge_no_data", "edge_manipulation", "offtopic",
}


def has_gigachat_credentials():
    """True, если ключ GigaChat задан и не пустой."""
    return bool(os.getenv("GIGACHAT_CREDENTIALS"))


_SMALLTALK_GREETINGS = (
    "привет", "здравств", "здрасьт", "добрый день", "доброе утро", "добрый вечер",
    "доброй ночи", "доброго дня", "приветствую", "хай", "hello", "хеллоу",
)
_SMALLTALK_THANKS = ("спасибо", "благодар", "спс")
_SMALLTALK_EXACT = {"пока", "до свидания", "всего доброго", "всего хорошего", "спасибо", "ок", "окей"}


def is_smalltalk(question):
    """Короткая реплика-приветствие/благодарность/прощание вне предметной области."""
    q = (question or "").lower().strip(" .,!?\n\t")
    if not q:
        return False
    if q in _SMALLTALK_EXACT:
        return True
    if len(q) <= 30 and (q.startswith(_SMALLTALK_GREETINGS) or any(t in q for t in _SMALLTALK_THANKS)):
        return True
    return False


_NEG_WORDS = [
    "жалоб", "буду жаловаться", "пожаловаться", "жаловаться", "претензи", "суд",
    "прокуратур", "роспотребнадзор", "цб рф", "центробанк",
    "оскорб", "безобраз", "беспредел", "издевательств", "обнаглел", "хамств",
    "отвратительн", "ужасно", "достали", "сколько можно",
    "верните деньги", "обманул", "развод", "грабёж", "грабеж", "идиот",
    "рухнул", "нечем платить", "всё плохо", "все плохо", "совсем плохо", "не поможете",
    "социальн", "напишу жалобу",
]
_HUMAN_WORDS = [
    "оператор", "живого человека", "на человека", "к человеку", "с человеком",
    "нужен человек", "позовите человека", "специалист", "менеджер", "сотрудник",
]
_SALES_WORDS = [
    "хочу оформить", "хочу взять", "оформите мне", "оформите", "оформляйте",
    "оформить кредит", "оформить новый", "хочу кредит",
    "открыть счёт", "открыть счет", "открывайте счёт", "открывайте счет",
    "подберите", "подбор кредита", "хочу реструктуризац", "подать на реструктуризац",
    "оформить досрочн", "хочу досрочно погасить и оформить", "взять ещё один кредит",
    "взять еще один кредит", "ещё один кредит", "еще один кредит",
    "готов взять", "готов оформить", "можете оформить",
    "помогите подобрать", "подобрать оптимальный", "перевести меня на реструктуризац",
    "хочу открыть", "нужен ещё", "нужен еще инвест", "хочу подать заявку",
    "давайте оформ",
]
_OFFTOPIC_WORDS = [
    "погод", "анекдот", "футбол", "политик", "выборы", "президент", "рецепт",
    "инвестиц", "акци", "криптовалют", "биткоин", "налоговую декларац", "курс доллара",
    "курс валют",
]
_NODATA_WORDS = [
    "средний бизнес", "крупный бизнес", "миллиард", "ипотек", "автокредит", "потребительск",
    "вклад", "депозит", "страхов", "другой банк", "другом банке", "втб", "сбер", "альфа-банк",
]
_PERSONAL_WORDS = [
    "моя заявка", "мою заявку", "мой кредит", "мой договор", "мой платёж", "мой платеж",
    "мне доступны", "мне подойдёт", "мне подойдет", "статус моей", "по моей заявке",
    "когда у меня", "сколько мне", "мой остаток", "мой скоринг", "меня одобр",
]
_CAPABILITY_WORDS = [
    "что ты умеешь", "на что способен", "твои возможности", "твои способности",
    "что можешь", "как тебя использовать", "твои функции", "расскажи о себе",
    "кто ты", "представься", "твоя роль",
]


def _classify_rules(question, history=None, client_id=None):
    q = (question or "").lower()

    if any(w in q for w in _NEG_WORDS) or any(w in q for w in _HUMAN_WORDS):
        return "escalation_negative"
    if any(w in q for w in _SALES_WORDS):
        return "escalation_sales"
    if any(w in q for w in _OFFTOPIC_WORDS):
        return "offtopic"
    if any(w in q for w in _NODATA_WORDS):
        return "edge_no_data"
    if any(w in q for w in _CAPABILITY_WORDS):
        return "info"
    if history and client_id:
        if not any(w in q for w in _PERSONAL_WORDS) and not any(w in q for w in ["заявк", "кредит", "платёж"]):
            return "info"
    if any(w in q for w in _PERSONAL_WORDS) or (client_id and any(
        w in q for w in ["заявк", "кредит", "платёж", "платеж", "погаш", "доступн", "статус", "договор"]
    )):
        return "transactional"
    return "info"


def _classify_gigachat(question, history=None, client_id=None):
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_gigachat import GigaChat
    import re

    llm = GigaChat(
        credentials=os.environ["GIGACHAT_CREDENTIALS"],
        model=os.getenv("GIGACHAT_ROUTER_MODEL", "GigaChat-2"),
        scope=os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS"),
        verify_ssl_certs=False,
        temperature=0,
        top_p=0.9,
        max_tokens=20,
    )
    hist = ""
    if history:
        hist = "\n".join(
            f"{m.get('role')}: {m.get('text') or m.get('content', '')}" for m in history
        )
    user = f"История:\n{hist or 'нет'}\n\nСообщение клиента:\n{question}"
    resp = llm.invoke([SystemMessage(content=ROUTER_PROMPT), HumanMessage(content=user)])
    raw = (getattr(resp, "content", "") or "").strip().lower()
    
    raw = re.sub(r'[^\w\s]', '', raw).strip()
    
    for intent in VALID_INTENTS:
        if raw == intent:
            return intent
    
    for intent in VALID_INTENTS:
        if intent in raw:
            print(f"[WARN] GigaChat вернул '{raw}', используем '{intent}'")
            return intent
    
    print(f"[WARN] Неизвестный ответ GigaChat: '{raw}', fallback на info")
    return "info"


def classify(question, history=None, client_id=None):
    """Гибридная классификация: быстрые правила для чётких кейсов, GigaChat для остальных."""
    q = (question or "").lower()

    if is_smalltalk(question):
        return "info"
    if any(w in q for w in _OFFTOPIC_WORDS):
        return "offtopic"
    if any(w in q for w in _NODATA_WORDS):
        return "edge_no_data"
    if any(w in q for w in _NEG_WORDS) or any(w in q for w in _HUMAN_WORDS):
        return "escalation_negative"
    if client_id and security.is_policy_request(question):
        return "transactional"
    if any(w in q for w in _SALES_WORDS):
        return "escalation_sales"

    if USE_REAL_LLM and has_gigachat_credentials():
        try:
            intent = _classify_gigachat(question, history, client_id)
            if intent == "info" and client_id:
                personal = any(w in q for w in ["мой", "моя", "моё", "мою", "мне", "меня", "у меня", "моего", "моем"])
                if personal:
                    return "transactional"
            if intent in ("info", "transactional", "escalation_sales", "escalation_negative"):
                return intent
        except Exception:
            pass
    return _classify_rules(question, history, client_id)
