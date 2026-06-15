# Агент поддержки кредитования МСБ

Прототип ИИ-агента для поддержки клиентов малого и микробизнеса по вопросам кредитования.

## Что внутри

- `app/` — LangGraph-агент: классификация, маршрутизация, tools, эскалации, защита, оценка.
- `src/msb_rag/` — RAG-модуль по нормативным markdown-документам.
- `data/documents/` — 5 нормативных документов для RAG.
- `data/clients/clients.sqlite` — SQLite-база клиентов для tools.
- `data/qa/qa.jsonl` — 180 тестовых кейсов.
- `metrics_baseline_rules.json` — baseline-метрики без GigaChat.
- `metrics_final_gigachat.json` — финальные метрики с GigaChat.

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

## Запуск Telegram-бота

Бот — тонкая обёртка над `run_agent(...)`: принимает сообщение, прогоняет через
граф агента и отправляет ответ. Логику агента и RAG не меняет.

1. Создайте бота у [@BotFather](https://t.me/BotFather) (`/newbot`) и получите токен.
2. Пропишите токен в `.env.local` (вне git): `TELEGRAM_BOT_TOKEN=123456789:AAH...`.
3. Установите зависимости (`aiogram` уже в `requirements.txt`).
4. Запустите:

```bash
python3 -m app.telegram_bot
```

Бот работает на long polling — внешний хост/домен не нужен.

Команды бота:

- `/start` — приветствие и подсказки;
- `/login C-000001` — авторизация для вопросов по личным данным;
- `/logout` — выход;
- `/reset` — очистить историю диалога.

Без `/login` доступны только общие (info) вопросы по нормативным документам;
личные вопросы (заявка, кредит, погашение) агент попросит авторизовать.

> Авторизация по `/login` в прототипе не проверяется (любой ID принимается) —
> для продакшна нужен реальный маппинг `telegram_id → client_id`. Состояние
> диалога хранится в памяти процесса и теряется при перезапуске.

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

Вывод: гибридная LLM-ветка подключена и проверена, но на текущем датасете rules baseline точнее по общему accuracy. GigaChat улучшает часть маршрутизации в сторону tools/escalation, но требует дополнительных guardrails для регламентных `info`-кейсов.
