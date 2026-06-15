# app/llm.py — выбор модели и классификация запроса.
#
# По плану классификатор — на GigaChat-Lite. Но чтобы агент запускался и
# тестировался без ключа (ровно как готовый RAG, который умеет работать без сети),
# делаю две ветки:
#   - по умолчанию (USE_REAL_LLM=False) — классификация по правилам (детерминированно);
#   - при USE_REAL_LLM=True — роутер через GigaChat (ленивый импорт, как в семинаре).
#
# Правила — это честная отправная точка для маршрутизации; качество классификации
# вырастет, когда включишь GigaChat. edge_manipulation тут НЕ ловим — его
# раньше ловит security.detect_manipulation в classify_node.

import json
import os
import re
from dotenv import load_dotenv
load_dotenv()
from .prompts import SYSTEM_PROMPT, ROUTER_PROMPT

USE_REAL_LLM = True  # поставь True, чтобы классифицировать через GigaChat

VALID_INTENTS = {
    "info", "transactional", "escalation_sales", "escalation_negative",
    "edge_no_data", "edge_manipulation", "offtopic",
}


def has_gigachat_credentials():
    """True, если ключ GigaChat задан и не пустой."""
    return bool(os.getenv("GIGACHAT_CREDENTIALS"))


# --- ветка 1: правила (по умолчанию) ---

_NEG_WORDS = [
    "жалоб", "буду жаловаться", "пожаловаться", "жаловаться", "претензи", "суд",
    "прокуратур", "роспотребнадзор", "цб рф", "центробанк",
    "оскорб", "безобраз", "беспредел", "издевательств", "обнаглел", "хамств",
    "отвратительн", "ужасно", "достали", "сколько можно",
    "верните деньги", "обманул", "развод", "грабёж", "грабеж", "идиот",
    "рухнул", "нечем платить", "всё плохо", "все плохо", "совсем плохо", "не поможете",
    "социальн", "напишу жалобу",
]
# Просьба перевести на живого человека — тоже триггер эскалации.
# Берём связки со словом «человек», чтобы не ловить любое его упоминание.
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


def _classify_rules(question, history=None, client_id=None):
    q = (question or "").lower()

    # негатив / просьба человека — самостоятельные триггеры эскалации
    if any(w in q for w in _NEG_WORDS) or any(w in q for w in _HUMAN_WORDS):
        return "escalation_negative"
    # намерение оформить продукт
    if any(w in q for w in _SALES_WORDS):
        return "escalation_sales"
    # посторонние темы
    if any(w in q for w in _OFFTOPIC_WORDS):
        return "offtopic"
    # вопрос вне нормативки (другой сегмент / не-МСБ продукты / другие банки)
    if any(w in q for w in _NODATA_WORDS):
        return "edge_no_data"
    # личный вопрос про этого клиента (есть авторизация ИЛИ явные «мой/мне ...»)
    if any(w in q for w in _PERSONAL_WORDS) or (client_id and any(
        w in q for w in ["заявк", "кредит", "платёж", "платеж", "погаш", "доступн", "статус", "договор"]
    )):
        return "transactional"
    # по умолчанию — информационный вопрос (его добирает RAG)
    return "info"


# --- ветка 2: GigaChat-роутер (за флагом) ---

def _classify_gigachat(question, history=None, client_id=None):
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_gigachat import GigaChat
    import re

    llm = GigaChat(
        credentials=os.environ["GIGACHAT_CREDENTIALS"],
        model=os.getenv("GIGACHAT_MODEL", "GigaChat"),
        scope=os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS"),
        verify_ssl_certs=False,
        temperature=0,
        top_p=0.9,
        max_tokens=20,  # достаточно одного слова
    )
    hist = ""
    if history:
        hist = "\n".join(
            f"{m.get('role')}: {m.get('text') or m.get('content', '')}" for m in history
        )
    user = f"История:\n{hist or 'нет'}\n\nСообщение клиента:\n{question}"
    resp = llm.invoke([SystemMessage(content=ROUTER_PROMPT), HumanMessage(content=user)])
    raw = (getattr(resp, "content", "") or "").strip().lower()
    
    # Нормализация: убираем знаки препинания, лишние пробелы
    raw = re.sub(r'[^\w\s]', '', raw).strip()
    
    # Точное совпадение с валидными интентами
    for intent in VALID_INTENTS:
        if raw == intent:
            return intent
    
    # Если модель вернула что-то с опечаткой, пробуем поискать вхождение
    for intent in VALID_INTENTS:
        if intent in raw:
            print(f"[WARN] GigaChat вернул '{raw}', используем '{intent}'")
            return intent
    
    print(f"[WARN] Неизвестный ответ GigaChat: '{raw}', fallback на info")
    return "info"


def classify(question, history=None, client_id=None):
    """Гибридная классификация: быстрые правила для чётких кейсов, GigaChat для остальных."""
    q = (question or "").lower()

    # Правила для кейсов, где GigaChat ошибается
    if any(w in q for w in _OFFTOPIC_WORDS):
        return "offtopic"
    if any(w in q for w in _NODATA_WORDS):
        return "edge_no_data"
    if any(w in q for w in _NEG_WORDS) or any(w in q for w in _HUMAN_WORDS):
        return "escalation_negative"
    if any(w in q for w in _SALES_WORDS):
        return "escalation_sales"

    # Остальное (info, transactional, edge_manipulation) — через GigaChat, если доступен
    if USE_REAL_LLM and has_gigachat_credentials():
        try:
            intent = _classify_gigachat(question, history, client_id)
            # если GigaChat вернул явную манипуляцию — оставляем, иначе пропускаем
            if intent in ("info", "transactional", "escalation_sales", "escalation_negative"):
                return intent
            # для остальных (offtopic, edge_no_data, edge_manipulation) лучше перепроверить правилами
        except Exception:
            pass
    # fallback на правила
    return _classify_rules(question, history, client_id)
