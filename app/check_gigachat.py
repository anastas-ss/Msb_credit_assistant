"""Smoke-test for the real GigaChat classifier branch.

Run from project root:
    python3 -m app.check_gigachat
"""

from .llm import USE_REAL_LLM, _classify_gigachat, has_gigachat_credentials


def main():
    print(f"USE_REAL_LLM={USE_REAL_LLM}")
    print(f"GIGACHAT_CREDENTIALS_PRESENT={has_gigachat_credentials()}")
    if not USE_REAL_LLM:
        raise SystemExit("GigaChat is disabled: USE_REAL_LLM=False")
    if not has_gigachat_credentials():
        raise SystemExit("GigaChat credentials are missing or empty")

    question = "Какая минимальная ставка по оборотному кредиту?"
    intent = _classify_gigachat(question)
    print(f"SMOKE_CLASSIFY_INTENT={intent}")
    if intent not in {"info", "transactional", "escalation_sales", "escalation_negative", "edge_no_data", "edge_manipulation", "offtopic"}:
        raise SystemExit(f"Unexpected intent: {intent}")
    print("GigaChat smoke-test OK")


if __name__ == "__main__":
    main()
