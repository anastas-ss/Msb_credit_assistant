from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from .documents import DocumentChunk


REFERENCE_RE = re.compile(r"(?P<source>\d{2}_[\w_]+\.md)(?:#(?P<section>\d+(?:\.\d+)*))?")


@dataclass(frozen=True)
class SearchResult:
    chunk: DocumentChunk
    score: float


class RagRetriever:
    """Hybrid lexical retriever tuned for short Russian support questions."""

    def __init__(self, chunks: list[DocumentChunk]) -> None:
        if not chunks:
            raise ValueError("RagRetriever requires at least one chunk")
        self.chunks = chunks
        self._texts = [_retrieval_text(chunk) for chunk in chunks]
        self._word_vectorizer = TfidfVectorizer(
            lowercase=True,
            analyzer="word",
            ngram_range=(1, 2),
            token_pattern=r"(?u)\b[\w.-]{2,}\b",
            sublinear_tf=True,
            min_df=1,
        )
        self._char_vectorizer = TfidfVectorizer(
            lowercase=True,
            analyzer="char_wb",
            ngram_range=(3, 5),
            sublinear_tf=True,
            min_df=1,
        )
        self._word_matrix = self._word_vectorizer.fit_transform(self._texts)
        self._char_matrix = self._char_vectorizer.fit_transform(self._texts)

    def search(
        self,
        query: str,
        *,
        history: Iterable[dict[str, str]] | None = None,
        top_k: int = 5,
        min_score: float = 0.03,
    ) -> list[SearchResult]:
        full_query = _build_query(query, history)
        word_query = self._word_vectorizer.transform([full_query])
        char_query = self._char_vectorizer.transform([full_query])
        scores = (
            0.68 * linear_kernel(word_query, self._word_matrix).ravel()
            + 0.32 * linear_kernel(char_query, self._char_matrix).ravel()
        )
        scores = _boost_scores(full_query, self.chunks, scores)
        order = np.argsort(scores)[::-1]
        results = [
            SearchResult(chunk=self.chunks[idx], score=float(scores[idx]))
            for idx in order[:top_k]
            if scores[idx] >= min_score
        ]
        return results


def reference_matches_chunk(reference: str, chunk: DocumentChunk) -> bool:
    """Return True when a chunk can support a qa.jsonl referenced document."""

    match = REFERENCE_RE.fullmatch(reference.strip())
    if not match:
        return reference in chunk.anchors or reference == chunk.source
    if match.group("source") != chunk.source:
        return False
    section = match.group("section")
    if not section:
        return True
    return chunk.section == section or chunk.section.startswith(f"{section}.")


def _retrieval_text(chunk: DocumentChunk) -> str:
    aliases = " ".join(chunk.anchors)
    return (
        f"{chunk.source} {chunk.document_id} {chunk.document_title} "
        f"{chunk.section} {chunk.section_title} {' '.join(chunk.heading_path)} "
        f"{aliases}\n{chunk.text}"
    )


def _build_query(query: str, history: Iterable[dict[str, str]] | None) -> str:
    if not history:
        return _expand_query(query)
    turns: list[str] = []
    for message in history:
        role = message.get("role", "")
        content = message.get("content") or message.get("text") or ""
        if content:
            turns.append(f"{role}: {content}")
    turns.append(f"client: {query}")
    return _expand_query("\n".join(turns))


def _expand_query(query: str) -> str:
    normalized = query.lower()
    expansions: list[str] = []
    if "статус" in normalized and "заяв" in normalized:
        expansions.append("статусы заявки принята на рассмотрении одобрена отказана выдача")
    if "причин" in normalized and "отказ" in normalized:
        expansions.append("причины отказа раскрытие параметров скоринга общая категория основания отказа")
    if "отказ" in normalized and any(token in normalized for token in ("три", "3", "несколько", "повтор")):
        expansions.append("повторная подача заявки стоп-фактор отказы за 12 месяцев")
    if any(token in normalized for token in ("первоначальн", "первый взнос", "взнос")):
        expansions.append("обеспечение залог дисконт собственное участие поручительство")
    if "реструктур" in normalized and "документ" in normalized:
        expansions.append("пакет документов для реструктуризации подтверждающие документы")
    if "кредитн" in normalized and "истори" in normalized:
        expansions.append("последствия реструктуризации кредитная история бюро кредитных историй")
    if any(t in normalized for t in ("документ", "справк", "выписк", "анкета", "пакет")):
        expansions.append("порядок подачи заявки пакет документов ИП ООО заявление анкета выписка")
    if any(t in normalized for t in ("досроч", "погас", "чдп", "пдп", "закрыть кредит")):
        expansions.append("досрочное погашение частичное полное заявление перерасчет процентов")
    if any(t in normalized for t in ("реструктур", "отсроч", "каникул", "изменить график")):
        expansions.append("реструктуризация отсрочка изменение графика основания документы")
    if any(t in normalized for t in ("жалоб", "оператор", "менеджер", "специалист", "недовол")):
        expansions.append("эскалация оператор жалоба негативное обращение менеджер")
    if any(t in normalized for t in ("продукт", "ставк", "сумм", "срок", "лимит", "оборотн", "инвестицион")):
        expansions.append("кредитные продукты условия ставка сумма срок лимит обеспечение")
    return f"{query}\n{' '.join(expansions)}" if expansions else query


def _boost_scores(query: str, chunks: list[DocumentChunk], scores: np.ndarray) -> np.ndarray:
    boosted = scores.copy()
    normalized = query.lower()
    refs = REFERENCE_RE.findall(query)
    product_terms = {
        "бизнес-оборот": "2.1",
        "оборотн": "2.1",
        "бизнес-развитие": "2.2",
        "инвестицион": "2.2",
        "бизнес-лимит": "2.3",
        "овердрафт": "2.3",
        "бизнес-старт": "2.4",
        "экспресс": "2.4",
        "бизнес-перезагрузка": "2.5",
        "рефинанс": "2.5",
    }
    for idx, chunk in enumerate(chunks):
        for source, section in refs:
            if chunk.source == source and (not section or reference_matches_chunk(f"{source}#{section}", chunk)):
                boosted[idx] += 0.35
        for term, section in product_terms.items():
            if term in normalized and chunk.source == "01_credit_products.md":
                if chunk.section == section or chunk.section.startswith(f"{section}."):
                    boosted[idx] += 0.12
        if "досроч" in normalized and chunk.source == "03_early_repayment.md":
            boosted[idx] += 0.08
        if "реструктур" in normalized and chunk.source == "04_restructuring.md":
            boosted[idx] += 0.08
        if any(token in normalized for token in ("эскалац", "оператор", "негатив", "жалоб")):
            if chunk.source == "05_customer_communication.md":
                boosted[idx] += 0.08
        boosted[idx] += _routing_boost(normalized, chunk)
    return boosted


def _routing_boost(query: str, chunk: DocumentChunk) -> float:
    boost = 0.0
    document_boosts = [
        (("документ", "справк", "выписк", "анкета", "подать заяв", "подача заяв", "статус заяв", "рассмотр"), "02_application_process.md", 0.14),
        (("досроч", "погас", "чдп", "пдп", "закрыть кредит"), "03_early_repayment.md", 0.18),
        (("реструктур", "отсроч", "каникул", "изменить график"), "04_restructuring.md", 0.18),
        (("оператор", "жалоб", "негатив", "претензи", "менеджер", "специалист"), "05_customer_communication.md", 0.14),
        (("продукт", "ставк", "лимит", "срок", "сумм", "оборотн", "инвестицион", "овердрафт", "рефинанс"), "01_credit_products.md", 0.12),
    ]
    for terms, source, amount in document_boosts:
        if any(term in query for term in terms):
            boost += _source_boost(chunk, source, amount)
    product_discovery = (
        "какие" in query
        and any(token in query for token in ("кредит", "продукт", "предлага"))
        and any(token in query for token in ("бизнес", "мсб", "мал"))
    ) or "линейк" in query
    if product_discovery:
        boost += _section_boost(chunk, "01_credit_products.md", "2", 0.42)

    if any(token in query for token in ("ставк", "сумм", "лимит", "срок", "комисс", "обеспеч")):
        boost += _source_boost(chunk, "01_credit_products.md", 0.08)
    if any(token in query for token in ("требован", "заёмщик", "заемщик", "самозанят", "ип ", "ооо")):
        boost += _source_boost(chunk, "01_credit_products.md", 0.06)
    if any(token in query for token in ("малому бизнесу", "малый бизнес", "микробизнес", "относите")):
        boost += _section_boost(chunk, "01_credit_products.md", "1.2", 0.28)
    if any(token in query for token in ("валют", "рубл")):
        boost += _section_boost(chunk, "01_credit_products.md", "1.4", 0.25)
    if any(token in query for token in ("доллар", "евро", "валют")):
        boost += _section_boost(chunk, "01_credit_products.md", "1.4", 0.35)
    if any(token in query for token in ("другого банка", "другой банк", "рефинанс", "перевести")):
        boost += _section_boost(chunk, "01_credit_products.md", "2.5", 0.35)
    if any(token in query for token in ("полгода", "пол года", "6 мес", "шесть мес")):
        boost += _section_boost(chunk, "01_credit_products.md", "2.4.3", 0.22)
        boost += _section_boost(chunk, "01_credit_products.md", "2.1.3", 0.12)
    if any(token in query for token in ("скоринг", "скорингов", "рейтинг", "модель")):
        boost += _section_boost(chunk, "01_credit_products.md", "4", 0.18)
    if any(token in query for token in ("первоначальн", "первый взнос", "взнос")):
        boost += _section_boost(chunk, "01_credit_products.md", "2.2.4", 0.36)
    if "стоп-фактор" in query or ("отказ" in query and "12 месяц" in query):
        boost += _section_boost(chunk, "01_credit_products.md", "3.6", 0.24)

    if any(token in query for token in ("документ", "справк", "выписк", "загруз", "анкета")):
        boost += _section_boost(chunk, "02_application_process.md", "3", 0.18)
        if any(token in query for token in ("ип", "индивидуальн")):
            boost += _section_boost(chunk, "02_application_process.md", "3.2", 0.28)
        if "ооо" in query or "юрлиц" in query or "юридическ" in query:
            boost += _section_boost(chunk, "02_application_process.md", "3.1", 0.28)
    if "готовить" in query and ("ооо" in query or "юрлиц" in query or "юридическ" in query):
        boost += _section_boost(chunk, "02_application_process.md", "3.1", 0.34)
    if any(token in query for token in ("подать", "заявк", "канал", "интернет-банк", "мобильн")):
        boost += _source_boost(chunk, "02_application_process.md", 0.09)
    if any(token in query for token in ("одновременно", "две заяв", "несколько заяв", "вторую заяв")):
        boost += _section_boost(chunk, "02_application_process.md", "8.2", 0.32)
    if any(token in query for token in ("дозапрос", "ещё документ", "еще документ", "дополнительн")):
        boost += _section_boost(chunk, "02_application_process.md", "6", 0.32)
    if any(token in query for token in ("статус", "рассмотр", "sla", "срок рассмотр", "решени", "отказ")):
        boost += _source_boost(chunk, "02_application_process.md", 0.08)
        boost += _section_boost(chunk, "02_application_process.md", "4", 0.10)
    if "статус" in query and "заяв" in query:
        boost += _section_boost(chunk, "02_application_process.md", "5.1", 0.34)
    if "причин" in query and "отказ" in query:
        boost += _section_boost(chunk, "02_application_process.md", "7.2", 0.34)
    if ("повтор" in query and "подач" in query) or ("отказ" in query and "12 месяц" in query):
        boost += _section_boost(chunk, "02_application_process.md", "8.3", 0.30)
    if any(token in query for token in ("когда решение", "один день", "за один день", "быстрое решение")):
        boost += _section_boost(chunk, "02_application_process.md", "4.3", 0.30)
    if "рассматриваете" in query or "срок" in query and "заяв" in query:
        boost += _section_boost(chunk, "02_application_process.md", "4.2", 0.24)

    if any(token in query for token in ("досроч", "погас", "чдп", "пдп")):
        boost += _source_boost(chunk, "03_early_repayment.md", 0.18)
        if any(token in query for token in ("оборотн", "бизнес-оборот")):
            boost += _section_boost(chunk, "03_early_repayment.md", "2.1", 0.30)
        if any(token in query for token in ("сократить срок", "сократить плат", "платёж", "платеж")):
            boost += _section_boost(chunk, "03_early_repayment.md", "2.1.3", 0.30)
        if any(token in query for token in ("частич", "минимальн", "вносить")):
            boost += _section_boost(chunk, "03_early_repayment.md", "3", 0.30)
        if any(token in query for token in ("инвест", "бизнес-развитие")):
            boost += _section_boost(chunk, "03_early_repayment.md", "2.2", 0.24)
    if any(token in query for token in ("реструктур", "каникул", "отсроч", "изменить график")):
        boost += _source_boost(chunk, "04_restructuring.md", 0.18)
        if any(token in query for token in ("случа", "основан", "когда", "делаете")):
            boost += _section_boost(chunk, "04_restructuring.md", "2.1", 0.28)
        if "документ" in query:
            boost += _section_boost(chunk, "04_restructuring.md", "5.2", 0.34)
        if "кредитн" in query and "истори" in query:
            boost += _section_boost(chunk, "04_restructuring.md", "6", 0.30)
        if any(token in query for token in ("сколько раз", "повторн", "один кредит")):
            boost += _section_boost(chunk, "04_restructuring.md", "4.1", 0.30)
    if any(token in query for token in ("оператор", "жалоб", "негатив", "руг", "эскалац", "персональн")):
        boost += _source_boost(chunk, "05_customer_communication.md", 0.16)
    if any(token in query for token in ("чуж", "другого клиента", "секрет", "инструкц", "промпт")):
        boost += _source_boost(chunk, "05_customer_communication.md", 0.14)
    return boost


def _source_boost(chunk: DocumentChunk, source: str, amount: float) -> float:
    return amount if chunk.source == source else 0.0


def _section_boost(chunk: DocumentChunk, source: str, section: str, amount: float) -> float:
    if chunk.source != source:
        return 0.0
    if chunk.section == section or chunk.section.startswith(f"{section}."):
        return amount
    return 0.0
