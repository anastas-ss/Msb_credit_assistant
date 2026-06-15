import asyncio
import os
import re
from dotenv import load_dotenv
load_dotenv()

from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command, CommandStart
from aiogram.types import Message
from aiogram.enums import ChatAction

from .run import run_agent
from . import rag_bridge, client_db

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

MAX_HISTORY = 10
_CLIENT_ID_RE = re.compile(r"C-\d{6}")

SESSIONS: dict[int, dict] = {}

dp = Dispatcher()


def _session(user_id: int) -> dict:
    """Возвращает (создавая при необходимости) сессию пользователя."""
    return SESSIONS.setdefault(user_id, {"client_id": None, "history": []})


@dp.message(CommandStart())
async def on_start(message: Message):
    _session(message.from_user.id)
    await message.answer(
        "Здравствуйте! Я ассистент по кредитованию малого и микробизнеса.\n\n"
        "Задайте вопрос по продуктам, заявкам или условиям.\n"
        "Для вопросов по вашим данным авторизуйтесь: /login C-000001\n"
        "Сбросить диалог - /reset, выйти - /logout."
    )


@dp.message(Command("login"))
async def on_login(message: Message):
    parts = (message.text or "").split(maxsplit=1)
    candidate = parts[1].strip().upper() if len(parts) > 1 else ""
    if not _CLIENT_ID_RE.fullmatch(candidate):
        await message.answer("Укажите ID в формате: /login C-000001")
        return
    if not await asyncio.to_thread(client_db.client_exists, candidate):
        await message.answer(f"Клиент с ID {candidate} не найден. Проверьте номер и попробуйте снова.")
        return
    s = _session(message.from_user.id)
    s["client_id"] = candidate
    await message.answer(
        f"Вы авторизованы как {candidate}. Теперь доступны вопросы по вашим данным."
    )


@dp.message(Command("logout"))
async def on_logout(message: Message):
    s = _session(message.from_user.id)
    s["client_id"] = None
    await message.answer("Вы вышли. Личные данные больше недоступны.")


@dp.message(Command("reset"))
async def on_reset(message: Message):
    _session(message.from_user.id)["history"] = []
    await message.answer("История диалога очищена.")


@dp.message(F.text)
async def on_text(message: Message):
    s = _session(message.from_user.id)
    await message.bot.send_chat_action(message.chat.id, ChatAction.TYPING)

    out = await asyncio.to_thread(
        run_agent,
        message.text,
        client_id=s["client_id"],
        channel="telegram",
        chat_history=s["history"],
    )

    answer = out.get("answer") or "Извините, не удалось сформировать ответ."

    s["history"].append({"role": "client", "content": message.text})
    s["history"].append({"role": "assistant", "content": out.get("answer", "")})
    s["history"] = s["history"][-MAX_HISTORY:]

    await message.answer(answer[:4096])


async def main():
    if not TOKEN:
        raise SystemExit(
            "TELEGRAM_BOT_TOKEN не задан. Получите токен у @BotFather и пропишите "
            "его в .env.local (TELEGRAM_BOT_TOKEN=...)."
        )
    rag_bridge._get_assistant()

    bot = Bot(TOKEN)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
