# app/rag_bridge.py — тонкая обёртка над готовым RAG-пакетом msb_rag.
#
# Готовый RAG ТРОГАТЬ НЕЛЬЗЯ. Здесь мы только подключаемся к нему через его
# публичный API (load_document_chunks -> RagRetriever -> RagAssistant) и отдаём
# узлам агента ровно то, что им нужно: текст ответа и список источников.
#
# Чанки и ретривер тяжело строить, поэтому строим их один раз и кешируем
# в модульных переменных (ленивая инициализация).

import os
import sys

# msb_rag лежит в src/ соседнего проекта — добавим src в пути импорта.
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_APP_DIR)
_SRC_DIR = os.path.join(_PROJECT_ROOT, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

_DOCS_DIR = os.path.join(_PROJECT_ROOT, "data", "documents")

_assistant = None  # кеш RagAssistant


def _get_assistant():
    """Лениво строит RagAssistant поверх готового msb_rag и кеширует его."""
    global _assistant
    if _assistant is None:
        from msb_rag.documents import load_document_chunks
        from msb_rag.retriever import RagRetriever
        from msb_rag.answering import RagAssistant

        chunks = load_document_chunks(_DOCS_DIR)
        retriever = RagRetriever(chunks)
        # use_llm=None: RagAssistant сам включит GigaChat, если задан ключ,
        # иначе вернёт extractive-ответ. То есть работает и без сети.
        _assistant = RagAssistant(retriever, use_llm=None)
    return _assistant


def rag_answer(question, history=None):
    """Возвращает (текст_ответа, список_источников) из готового RAG."""
    result = _get_assistant().answer(question, history=history)
    return result.answer, list(result.sources)
