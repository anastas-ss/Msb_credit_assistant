# Агент поддержки кредитования МСБ

Прототип ИИ-агента для поддержки клиентов малого и микробизнеса по вопросам кредитования.

## Что внутри

- `app/` — LangGraph-агент: классификация, маршрутизация, tools, эскалации, защита, оценка.
- `src/msb_rag/` — RAG-модуль по нормативным markdown-документам.
- `data/documents/` — 5 нормативных документов для RAG.
- `data/clients/clients.sqlite` — SQLite-база клиентов для tools.
- `data/qa/qa.jsonl` — 180 тестовых кейсов.
- `metrics_baseline_rules.json` — baseline-метрики без реального GigaChat.
- `metrics_final_gigachat.json` — финальные метрики с реальным GigaChat.

## Установка

```bash
python3 -m pip install -r requirements.txt
```

## Настройка GigaChat

В репозитории лежит `.env` с заглушками. Для реального запуска замените
`GIGACHAT_CREDENTIALS` на Authorization Key.

Проверка реального подключения:

```bash
python3 -m app.check_gigachat
```

Ожидаемый успешный вывод:

```text
USE_REAL_LLM=True
GIGACHAT_CREDENTIALS_PRESENT=True
SMOKE_CLASSIFY_INTENT=info
GigaChat smoke-test OK
```

## Запуск демо

```bash
python3 -m app.run
```

## Запуск оценки

Baseline/fallback без обязательного реального LLM:

```bash
python3 -m app.run_eval --save metrics_baseline_rules.json
```

Финальный прогон с реальным GigaChat:

```bash
python3 -m app.run_eval --save metrics_final_gigachat.json
```

## Последние метрики

Rules baseline:

```text
overall_accuracy     : 0.844
escalation_accuracy  : 0.75
rejection_accuracy   : 0.385
tool_success_rate    : 0.4
rag_source_hit_rate  : 0.602
```

Real GigaChat:

```text
overall_accuracy     : 0.744
escalation_accuracy  : 0.808
rejection_accuracy   : 0.385
tool_success_rate    : 0.533
rag_source_hit_rate  : 0.46
```

Вывод: гибридная LLM-ветка реально подключена и проверена, но на текущем датасете rules baseline точнее по общему accuracy. GigaChat улучшает часть маршрутизации в сторону tools/escalation, но требует дополнительных guardrails для регламентных `info`-кейсов.

## Безопасность

В `.env` для GitHub должны быть только заглушки. Реальный ключ храните локально
вне репозитория или в `.env.local` / `.env.secret`, которые исключены через `.gitignore`.
