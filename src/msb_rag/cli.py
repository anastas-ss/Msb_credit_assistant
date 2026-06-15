from __future__ import annotations

import argparse
import json
from pathlib import Path

from .answering import RagAssistant
from .documents import load_document_chunks
from .evaluate import evaluate_retrieval, format_report
from .retriever import RagRetriever


DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DOCS_DIR = DEFAULT_PROJECT_ROOT / "data" / "documents"
DEFAULT_QA_PATH = DEFAULT_PROJECT_ROOT / "data" / "qa" / "qa.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG for SMB lending regulations")
    parser.add_argument("--docs-dir", type=Path, default=DEFAULT_DOCS_DIR)
    subparsers = parser.add_subparsers(dest="command", required=True)

    ask_parser = subparsers.add_parser("ask", help="answer a question")
    ask_parser.add_argument("question")
    ask_parser.add_argument("--top-k", type=int, default=5)
    ask_parser.add_argument("--no-llm", action="store_true")
    ask_parser.add_argument("--json", action="store_true")

    retrieve_parser = subparsers.add_parser("retrieve", help="show retrieved chunks")
    retrieve_parser.add_argument("question")
    retrieve_parser.add_argument("--top-k", type=int, default=5)

    eval_parser = subparsers.add_parser("eval", help="evaluate retrieval against qa.jsonl")
    eval_parser.add_argument("--qa-path", type=Path, default=DEFAULT_QA_PATH)
    eval_parser.add_argument("--top-k", type=int, default=5)
    eval_parser.add_argument("--category", action="append")
    eval_parser.add_argument("--json", action="store_true")

    args = parser.parse_args()
    chunks = load_document_chunks(args.docs_dir)
    retriever = RagRetriever(chunks)

    if args.command == "ask":
        assistant = RagAssistant(retriever, top_k=args.top_k, use_llm=not args.no_llm)
        answer = assistant.answer(args.question)
        if args.json:
            print(
                json.dumps(
                    {
                        "answer": answer.answer,
                        "sources": list(answer.sources),
                        "used_llm": answer.used_llm,
                        "scores": [
                            {"source": result.chunk.citation, "score": result.score}
                            for result in answer.results
                        ],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        else:
            print(answer.answer)
        return

    if args.command == "retrieve":
        for idx, result in enumerate(retriever.search(args.question, top_k=args.top_k), start=1):
            print(f"{idx}. {result.chunk.citation} score={result.score:.3f}")
            print(result.chunk.section_title)
        return

    if args.command == "eval":
        categories = set(args.category) if args.category else None
        report = evaluate_retrieval(
            retriever,
            args.qa_path,
            top_k=args.top_k,
            categories=categories,
        )
        if args.json:
            print(json.dumps(report, ensure_ascii=False, indent=2))
        else:
            print(format_report(report))
        return


if __name__ == "__main__":
    main()
