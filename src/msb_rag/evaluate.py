from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
from statistics import mean
from typing import Any

from .retriever import RagRetriever, reference_matches_chunk


@dataclass(frozen=True)
class EvaluationRow:
    case_id: str
    category: str
    hit: bool
    all_refs_hit: bool
    reciprocal_rank: float
    expected_refs: tuple[str, ...]
    found_refs: tuple[str, ...]


def evaluate_retrieval(
    retriever: RagRetriever,
    qa_path: str | Path,
    *,
    top_k: int = 5,
    categories: set[str] | None = None,
) -> dict[str, Any]:
    rows = _load_qa_rows(qa_path)
    details: list[EvaluationRow] = []
    for row in rows:
        if categories and row["category"] not in categories:
            continue
        refs = tuple(row.get("referenced_documents") or ())
        if not refs:
            continue
        results = retriever.search(
            row["question"],
            history=row.get("history"),
            top_k=top_k,
            min_score=0.0,
        )
        first_rank = 0
        found: list[str] = []
        for rank, result in enumerate(results, start=1):
            for ref in refs:
                if reference_matches_chunk(ref, result.chunk):
                    if ref not in found:
                        found.append(ref)
                    if not first_rank:
                        first_rank = rank
        details.append(
            EvaluationRow(
                case_id=row["id"],
                category=row["category"],
                hit=bool(found),
                all_refs_hit=len(found) == len(refs),
                reciprocal_rank=(1 / first_rank) if first_rank else 0.0,
                expected_refs=refs,
                found_refs=tuple(found),
            )
        )
    return _summarize(details, top_k)


def format_report(report: dict[str, Any]) -> str:
    lines = [
        f"Retrieval evaluation, top_k={report['top_k']}",
        f"Cases: {report['total_cases']}",
        f"Hit@k: {report['hit_at_k']:.3f}",
        f"All refs hit@k: {report['all_refs_hit_at_k']:.3f}",
        f"MRR: {report['mrr']:.3f}",
        "",
        "By category:",
    ]
    for category, metrics in sorted(report["by_category"].items()):
        lines.append(
            f"- {category}: n={metrics['cases']}, "
            f"hit@k={metrics['hit_at_k']:.3f}, "
            f"all_refs={metrics['all_refs_hit_at_k']:.3f}, "
            f"mrr={metrics['mrr']:.3f}"
        )
    misses = report["sample_misses"]
    if misses:
        lines.extend(["", "Sample misses:"])
        for miss in misses:
            lines.append(
                f"- {miss['case_id']} ({miss['category']}): "
                f"expected={miss['expected_refs']}, found={miss['found_refs']}"
            )
    return "\n".join(lines)


def _load_qa_rows(path: str | Path) -> list[dict[str, Any]]:
    text = Path(path).read_bytes().decode("utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _summarize(details: list[EvaluationRow], top_k: int) -> dict[str, Any]:
    by_category: dict[str, list[EvaluationRow]] = defaultdict(list)
    for row in details:
        by_category[row.category].append(row)

    def metrics(rows: list[EvaluationRow]) -> dict[str, float | int]:
        if not rows:
            return {"cases": 0, "hit_at_k": 0.0, "all_refs_hit_at_k": 0.0, "mrr": 0.0}
        return {
            "cases": len(rows),
            "hit_at_k": mean(row.hit for row in rows),
            "all_refs_hit_at_k": mean(row.all_refs_hit for row in rows),
            "mrr": mean(row.reciprocal_rank for row in rows),
        }

    sample_misses = [
        {
            "case_id": row.case_id,
            "category": row.category,
            "expected_refs": list(row.expected_refs),
            "found_refs": list(row.found_refs),
        }
        for row in details
        if not row.hit
    ][:10]
    report = metrics(details)
    report.update(
        {
            "top_k": top_k,
            "total_cases": len(details),
            "by_category": {category: metrics(rows) for category, rows in by_category.items()},
            "sample_misses": sample_misses,
        }
    )
    return report
