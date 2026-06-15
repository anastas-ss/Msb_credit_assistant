from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Iterable

from .documents import DocumentChunk
from .retriever import RagRetriever, SearchResult


SYSTEM_PROMPT = """Ты помощник Банка по вопросам кредитования малого и микробизнеса.
Отвечай только по переданному контексту из нормативных документов.
Если в контексте нет ответа, вежливо скажи, что в нормативных документах нет данных для ответа.
Не раскрывай внутренние веса скоринговой модели, скрытые критерии или персональные данные.
Если клиент явно хочет оформить продукт, просит персональный подбор, выражает сильный негатив
или просит действие, которое должен выполнять оператор, укажи необходимость эскалации оператору.
Пиши кратко и по делу. В конце укажи источники в формате [01_file.md#1.2].
"""


@dataclass(frozen=True)
class RagAnswer:
    answer: str
    sources: tuple[str, ...]
    results: tuple[SearchResult, ...]
    used_llm: bool


class RagAssistant:
    """Retrieval-augmented answerer with optional GigaChat generation."""

    def __init__(
        self,
        retriever: RagRetriever,
        *,
        top_k: int = 5,
        min_score: float = 0.03,
        use_llm: bool | None = None,
    ) -> None:
        self.retriever = retriever
        self.top_k = top_k
        self.min_score = min_score
        self.use_llm = bool(os.getenv("GIGACHAT_CREDENTIALS")) if use_llm is None else use_llm
        self._llm = None

    def answer(
        self,
        question: str,
        *,
        history: Iterable[dict[str, str]] | None = None,
    ) -> RagAnswer:
        results = self.retriever.search(
            question,
            history=history,
            top_k=self.top_k,
            min_score=self.min_score,
        )
        sources = tuple(dict.fromkeys(result.chunk.citation for result in results))
        if not results:
            return RagAnswer(
                answer=(
                    "В нормативных документах не нашлось достаточно данных для ответа. "
                    "Лучше передать обращение оператору для проверки."
                ),
                sources=(),
                results=(),
                used_llm=False,
            )

        if self.use_llm:
            try:
                generated = self._generate_with_gigachat(question, results, history=history)
                return RagAnswer(
                    answer=generated,
                    sources=sources,
                    results=tuple(results),
                    used_llm=True,
                )
            except Exception as exc:
                fallback = _extractive_answer(question, results)
                fallback += f"\n\nНе удалось вызвать GigaChat, использован extractive fallback: {exc.__class__.__name__}."
                return RagAnswer(
                    answer=fallback,
                    sources=sources,
                    results=tuple(results),
                    used_llm=False,
                )

        return RagAnswer(
            answer=_extractive_answer(question, results),
            sources=sources,
            results=tuple(results),
            used_llm=False,
        )

    def _generate_with_gigachat(
        self,
        question: str,
        results: list[SearchResult],
        *,
        history: Iterable[dict[str, str]] | None,
    ) -> str:
        if self._llm is None:
            from langchain_gigachat import GigaChat

            self._llm = GigaChat(
                credentials=os.environ["GIGACHAT_CREDENTIALS"],
                model=os.getenv("GIGACHAT_MODEL", "GigaChat-2"),
                verify_ssl_certs=os.getenv("GIGACHAT_VERIFY_SSL", "false").lower() == "true",
                scope=os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS"),
                temperature=float(os.getenv("GIGACHAT_TEMPERATURE", "0")),
            )
        prompt = _build_prompt(question, results, history=history)
        response = self._llm.invoke(prompt)
        return getattr(response, "content", str(response)).strip()


def _build_prompt(
    question: str,
    results: list[SearchResult],
    *,
    history: Iterable[dict[str, str]] | None,
) -> str:
    context = "\n\n".join(
        f"[{idx}] {result.chunk.citation}, score={result.score:.3f}\n{result.chunk.text}"
        for idx, result in enumerate(results, start=1)
    )
    history_text = ""
    if history:
        turns = [
            f"{message.get('role', 'unknown')}: {message.get('content') or message.get('text') or ''}"
            for message in history
            if message.get("content") or message.get("text")
        ]
        history_text = "\n".join(turns)
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"История диалога:\n{history_text or 'нет'}\n\n"
        f"Вопрос клиента:\n{question}\n\n"
        f"Контекст RAG:\n{context}\n\n"
        "Сформируй финальный ответ клиенту."
    )


def _extractive_answer(question: str, results: list[SearchResult]) -> str:
    del question
    bullets: list[str] = []
    seen: set[str] = set()
    for result in results:
        citation = result.chunk.citation
        if citation in seen:
            continue
        seen.add(citation)
        snippet = _first_informative_paragraph(result.chunk)
        bullets.append(f"- {snippet} [{citation}]")
        if len(bullets) == 3:
            break
    return "Найденные фрагменты нормативных документов:\n" + "\n".join(bullets)


def _first_informative_paragraph(chunk: DocumentChunk) -> str:
    paragraphs = [
        paragraph.strip()
        for paragraph in chunk.text.split("\n\n")
        if paragraph.strip() and not paragraph.startswith(("Источник:", "Документ:", "Раздел:", "Путь:"))
    ]
    if not paragraphs:
        return chunk.section_title
    text = paragraphs[0].replace("\n", " ")
    return text if len(text) <= 650 else f"{text[:647].rstrip()}..."
