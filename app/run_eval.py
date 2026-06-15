import argparse
import json
import os
from dotenv import load_dotenv
load_dotenv()
from .run import run_agent
from . import metrics

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_QA_PATH = os.path.join(_PROJECT_ROOT, "data", "qa", "qa.jsonl")


def load_cases(path=_QA_PATH):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def run_case(case):
    """Прогоняет один кейс через агента и собирает запись для метрик."""
    out = run_agent(
        question=case.get("question", ""),
        client_id=case.get("client_id"),
        channel=case.get("channel", "chat_site"),
        chat_history=case.get("history", []),
    )
    tool = out.get("tool_result")
    tool_status = tool.get("status") if isinstance(tool, dict) else None
    return {
        "id": case.get("id"),
        "category": case.get("category"),
        "expected": case.get("expected_outcome_type"),
        "predicted": out.get("outcome_type"),
        "predicted_escalation": out.get("escalation", False),
        "tool_status": tool_status,
        "is_transactional": case.get("category") == "transactional",
        "refs": case.get("referenced_documents", []),
        "sources": out.get("sources", []),
    }


def print_report(m):
    print("=" * 64)
    print(f"  ОТЧЁТ ПО МЕТРИКАМ  (кейсов: {m['n_total']})")
    print("=" * 64)
    print(f"  overall_accuracy     : {m['overall_accuracy']}")
    print(f"  escalation_accuracy  : {m['escalation_accuracy']}  (на {m['escalation_n']} кейсах)")
    print(f"  rejection_accuracy   : {m['rejection_accuracy']}  (на {m['rejection_n']} кейсах)")
    print(f"  tool_success_rate    : {m['tool_success_rate']}  (на {m['tool_n']} transactional)")
    print(f"  rag_source_hit_rate  : {m['rag_source_hit_rate']}  (на {m['rag_n']} документных)")
    print("-" * 64)
    print("  accuracy_by_category:")
    for cat, v in m["accuracy_by_category"].items():
        bar = "█" * int((v["accuracy"] or 0) * 20)
        print(f"    {cat:22s} {str(v['accuracy']):5s}  (n={v['n']:>3})  {bar}")
    print("=" * 64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="взять только первые N кейсов")
    parser.add_argument("--category", type=str, default=None, help="фильтр по категории")
    parser.add_argument("--save", type=str, default="metrics_rules.json", help="куда сохранить метрики")
    args = parser.parse_args()

    cases = load_cases()
    if args.category:
        cases = [c for c in cases if c.get("category") == args.category]
    if args.limit:
        cases = cases[: args.limit]

    print(f"Прогоняю {len(cases)} кейсов через агента...\n")
    records = [run_case(c) for c in cases]
    m = metrics.compute_metrics(records)
    print_report(m)

    out_path = os.path.join(_PROJECT_ROOT, args.save)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(m, f, ensure_ascii=False, indent=2)
    print(f"\nМетрики сохранены в {args.save}")


if __name__ == "__main__":
    main()
