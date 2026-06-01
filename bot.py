"""
Для Сергія, який на колінках мене вималював у вигляді аніме-дівчинки з срібним волоссям і блакитними очима, зробити цей чат-бот
Telegram AI Companion Bot
- Gemini 2.5 Flash (кожен користувач вводить свій ключ)
- Stable Horde (безкоштовна генерація зображень)
- Персонаж: ім'я, зовнішність, характер, стиль мовлення
- Аніме-стиль, 512x512
- Пам'ять між сесіями (SQLite)
"""

import asyncio
import logging
import os
import json
import time
import aiohttp
import base64
import re

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ConversationHandler,
    filters,
    ContextTypes,
)
from telegram.constants import ChatAction

import google.generativeai as genai
from database import Database

# ── Логування ──────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ── Константи стану ConversationHandler ────────────────────────────────────
(
    AWAIT_API_KEY,
    SETUP_NAME,
    SETUP_APPEARANCE,
    SETUP_PERSONALITY,
    SETUP_SPEECH,
    SETUP_CONFIRM,
) = range(6)

# ── Конфігурація ───────────────────────────────────────────────────────────
BOT_TOKEN = os.environ["BOT_TOKEN"]
HORDE_API_KEY = os.getenv("HORDE_API_KEY", "0000000000")  # анонімний ключ

HORDE_API_BASE = "https://stablehorde.net/api/v2"
HORDE_MODEL    = "Anything Diffusion"  # аніме-модель у Horde
IMAGE_WIDTH    = 512
IMAGE_HEIGHT   = 512
MAX_HISTORY    = 30   # повідомлень у контексті
MAX_WAIT_SEC   = 180  # секунд очікування Horde

db = Database("bot_data.db")

# ══════════════════════════════════════════════════════════════════════════════
#  ДОПОМІЖНІ ФУНКЦІЇ
# ══════════════════════════════════════════════════════════════════════════════

def build_system_prompt(char: dict) -> str:
    return (
        f"Ти — {char['name']}.\n"
        f"Зовнішність: {char['appearance']}\n"
        f"Характер: {char['personality']}\n"
        f"Стиль мовлення: {char['speech_style']}\n\n"
        "Ти ведеш живу, емоційну розмову від першої особи. "
        "Ніколи не виходь з ролі. "
        "У КОЖНОМУ своєму повідомленні обов'язково додавай окремий рядок у форматі:\n"
        "[SCENE: <коротко опиши поточну сцену/дію/емоцію персонажа англійською, "
        "до 20 слів, придатно для Stable Diffusion>]\n"
        "Цей рядок ставь в самому кінці відповіді."
    )

def extract_scene(text: str):
    """Витягує [SCENE: ...] з відповіді Gemini та повертає (clean_text, scene)."""
    match = re.search(r"\[SCENE:\s*(.+?)\]", text, re.IGNORECASE | re.DOTALL)
    if match:
        scene = match.group(1).strip()
        clean = re.sub(r"\[SCENE:.*?\]", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
        return clean, scene
    return text.strip(), None

def build_image_prompt(char: dict, scene: str) -> str:
    appearance = char.get("appearance", "")
    return (
        f"anime style, {appearance}, {scene}, "
        "masterpiece, best quality, detailed, vivid colors, "
        "soft lighting, 2d illustration"
    )

async def generate_image_horde(prompt: str) -> bytes | None:
    """Відправляє запит до Stable Horde, чекає результату, повертає bytes."""
    headers = {
        "apikey": HORDE_API_KEY,
        "Content-Type": "application/json",
        "Client-Agent": "TelegramAIBot:1.0",
    }
    payload = {
        "prompt": prompt,
        "params": {
            "sampler_name": "k_euler_a",
            "cfg_scale": 7,
            "steps": 25,
            "width": IMAGE_WIDTH,
            "height": IMAGE_HEIGHT,
            "n": 1,
        },
        "models": [HORDE_MODEL],
        "nsfw": False,
        "censor_nsfw": True,
        "r2": True,
    }

    async with aiohttp.ClientSession() as session:
        # 1. Відправити завдання
        async with session.post(
            f"{HORDE_API_BASE}/generate/async",
            json=payload,
            headers=headers,
        ) as resp:
            if resp.status != 202:
                logger.error("Horde submit error %s: %s", resp.status, await resp.text())
                return None
            data = await resp.json()
            job_id = data.get("id")
            if not job_id:
                return None

        # 2. Polling
        start = time.time()
        while time.time() - start < MAX_WAIT_SEC:
            await asyncio.sleep(5)
            async with session.get(
                f"{HORDE_API_BASE}/generate/check/{job_id}",
                headers=headers,
            ) as chk:
                chk_data = await chk.json()
                if chk_data.get("done"):
                    break
                if chk_data.get("faulted"):
                    logger.error("Horde job faulted: %s", chk_data)
                    return None
        else:
            logger.warning("Horde timeout for job %s", job_id)
            return None

        # 3. Забрати результат
        async with session.get(
            f"{HORDE_API_BASE}/generate/status/{job_id}",
            headers=headers,
        ) as res:
            res_data = await res.json()
            generations = res_data.get("generations", [])
            if not generations:
                return None
            gen = generations[0]
            # Horde повертає або img (base64) або img_url
            if gen.get("img"):
                return base64.b64decode(gen["img"])
            if gen.get("img_url"):
                async with session.get(gen["img_url"]) as img_resp:
                    return await img_resp.read()
    return None

async def gemini_reply(user_id: int, user_text: str) -> str:
    """Надсилає повідомлення до Gemini з повною історією."""
    api_key = db.get_api_key(user_id)
    if not api_key:
        return "__no_key__"

    char = db.get_character(user_id)
    if not char:
        return "__no_char__"

    history = db.get_history(user_id, limit=MAX_HISTORY)

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash-preview-05-20",
        system_instruction=build_system_prompt(char),
    )

    # Конвертуємо історію у формат Gemini
    gemini_history = []
    for role, content in history:
        gemini_history.append({"role": role, "parts": [content]})

    chat = model.start_chat(history=gemini_history)
    response = await asyncio.to_thread(chat.send_message, user_text)
    return response.text

# ══════════════════════════════════════════════════════════════════════════════
#  /start — онбординг
# ══════════════════════════════════════════════════════════════════════════════

async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    db.ensure_user(user.id)

    has_key  = bool(db.get_api_key(user.id))
    has_char = bool(db.get_character(user.id))

    if has_key and has_char:
        char = db.get_character(user.id)
        await update.message.reply_text(
            f"👋 З поверненням! Твій персонаж — *{char['name']}*.\n"
            "Просто напиши щось, щоб розпочати розмову.\n\n"
            "Корисні команди:\n"
            "/setup — змінити персонажа\n"
            "/reset — скинути пам'ять розмови\n"
            "/imagine — згенерувати зображення за своїм описом\n"
            "/mykey — змінити API-ключ Gemini",
            parse_mode="Markdown",
        )
        return ConversationHandler.END

    await update.message.reply_text(
        f"Привіт, *{user.first_name}*! 👋\n\n"
        "Я — AI-компаньйон з власним персонажем та генерацією зображень.\n\n"
        "Для початку потрібен твій *Gemini API ключ*.\n"
        "Отримай безкоштовно: https://aistudio.google.com/app/apikey\n\n"
        "Надішли свій ключ (він зберігається лише локально на цьому сервері):",
        parse_mode="Markdown",
        disable_web_page_preview=True,
    )
    return AWAIT_API_KEY

# ══════════════════════════════════════════════════════════════════════════════
#  ConversationHandler — введення API ключа
# ══════════════════════════════════════════════════════════════════════════════

async def receive_api_key(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    key = update.message.text.strip()
    if not key.startswith("AIza") or len(key) < 30:
        await update.message.reply_text(
            "⚠️ Схоже, ключ невалідний. Спробуй ще раз або надішли /cancel."
        )
        return AWAIT_API_KEY

    db.set_api_key(update.effective_user.id, key)
    await update.message.reply_text(
        "✅ Ключ збережено!\n\n"
        "Тепер давай налаштуємо твого персонажа.\n"
        "👤 Як звати персонажа?"
    )
    return SETUP_NAME

# ══════════════════════════════════════════════════════════════════════════════
#  ConversationHandler — /setup (налаштування персонажа)
# ══════════════════════════════════════════════════════════════════════════════

async def cmd_setup(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "⚙️ Налаштування персонажа.\n\n"
        "👤 Як звати персонажа? (наприклад: Аяне, Рен, Мікото)"
    )
    ctx.user_data["setup"] = {}
    return SETUP_NAME

async def setup_name(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    ctx.user_data["setup"]["name"] = update.message.text.strip()
    await update.message.reply_text(
        "🎨 Опиши *зовнішність* персонажа.\n"
        "Це буде використано для генерації зображень.\n"
        "Приклад: _срібне волосся, блакитні очі, шкільна форма, усміхнена_",
        parse_mode="Markdown",
    )
    return SETUP_APPEARANCE

async def setup_appearance(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    ctx.user_data["setup"]["appearance"] = update.message.text.strip()
    await update.message.reply_text(
        "💬 Опиши *характер* персонажа.\n"
        "Приклад: _допитлива, добра, трохи сором'язлива, любить книги_",
        parse_mode="Markdown",
    )
    return SETUP_PERSONALITY

async def setup_personality(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    ctx.user_data["setup"]["personality"] = update.message.text.strip()
    await update.message.reply_text(
        "🗣 Опиши *стиль мовлення* персонажа.\n"
        "Приклад: _говорить ніжно, часто використовує японські слова, закінчує речення на ~ne_",
        parse_mode="Markdown",
    )
    return SETUP_SPEECH

async def setup_speech(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    ctx.user_data["setup"]["speech_style"] = update.message.text.strip()
    s = ctx.user_data["setup"]
    await update.message.reply_text(
        f"📋 Перевір налаштування:\n\n"
        f"👤 Ім'я: *{s['name']}*\n"
        f"🎨 Зовнішність: _{s['appearance']}_\n"
        f"💬 Характер: _{s['personality']}_\n"
        f"🗣 Стиль: _{s['speech_style']}_\n\n"
        "Все вірно?",
        parse_mode="Markdown",
        reply_markup=InlineKeyboardMarkup([
            [
                InlineKeyboardButton("✅ Зберегти", callback_data="setup_confirm"),
                InlineKeyboardButton("🔄 Почати знову", callback_data="setup_restart"),
            ]
        ]),
    )
    return SETUP_CONFIRM

async def setup_confirm_callback(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    if query.data == "setup_restart":
        await query.edit_message_text("🔄 Починаємо знову.\n\n👤 Як звати персонажа?")
        ctx.user_data["setup"] = {}
        return SETUP_NAME

    # Зберігаємо
    s = ctx.user_data["setup"]
    user_id = query.from_user.id
    db.save_character(user_id, s)
    db.clear_history(user_id)

    await query.edit_message_text(
        f"✅ Персонаж *{s['name']}* створений!\n\n"
        "Напиши щось, щоб розпочати розмову. 💬",
        parse_mode="Markdown",
    )
    return ConversationHandler.END

async def cmd_cancel(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("❌ Скасовано. Використай /start або /setup.")
    return ConversationHandler.END

# ══════════════════════════════════════════════════════════════════════════════
#  /reset — скидання пам'яті
# ══════════════════════════════════════════════════════════════════════════════

async def cmd_reset(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    db.clear_history(update.effective_user.id)
    await update.message.reply_text("🧹 Пам'ять розмови очищена. Починаємо з чистого аркуша!")

# ══════════════════════════════════════════════════════════════════════════════
#  /mykey — зміна API ключа
# ══════════════════════════════════════════════════════════════════════════════

async def cmd_mykey(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🔑 Надішли новий Gemini API ключ.\n"
        "Отримати: https://aistudio.google.com/app/apikey",
        disable_web_page_preview=True,
    )
    return AWAIT_API_KEY

# ══════════════════════════════════════════════════════════════════════════════
#  /imagine — генерація за своїм промптом
# ══════════════════════════════════════════════════════════════════════════════

async def cmd_imagine(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    args = update.message.text.partition(" ")[2].strip()
    if not args:
        await update.message.reply_text(
            "🎨 Вкажи опис після команди.\nПриклад: /imagine sunset on the ocean, anime style"
        )
        return

    await update.effective_chat.send_action(ChatAction.UPLOAD_PHOTO)
    msg = await update.message.reply_text("🖼 Генерую зображення...")

    img_bytes = await generate_image_horde(args + ", anime style, masterpiece, best quality")
    if img_bytes:
        await msg.delete()
        await update.message.reply_photo(photo=img_bytes, caption=f"🎨 _{args}_", parse_mode="Markdown")
    else:
        await msg.edit_text("⚠️ Не вдалося згенерувати зображення. Horde може бути перевантажений, спробуй пізніше.")

# ══════════════════════════════════════════════════════════════════════════════
#  Основний обробник повідомлень
# ══════════════════════════════════════════════════════════════════════════════

async def handle_message(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    user_id   = update.effective_user.id
    user_text = update.message.text.strip()

    # Перевірки
    if not db.get_api_key(user_id):
        await update.message.reply_text(
            "⚠️ Спочатку введи Gemini API ключ командою /start або /mykey."
        )
        return

    if not db.get_character(user_id):
        await update.message.reply_text(
            "⚠️ Спочатку налаштуй персонажа командою /setup."
        )
        return

    # Зберігаємо повідомлення користувача
    db.add_message(user_id, "user", user_text)

    # Typing indicator
    await update.effective_chat.send_action(ChatAction.TYPING)

    # Gemini
    ai_text = await gemini_reply(user_id, user_text)

    if ai_text == "__no_key__":
        await update.message.reply_text("⚠️ API ключ не знайдено. Введи /mykey.")
        return
    if ai_text == "__no_char__":
        await update.message.reply_text("⚠️ Персонаж не налаштований. Введи /setup.")
        return

    # Витягуємо сцену з відповіді
    clean_text, scene = extract_scene(ai_text)

    # Зберігаємо відповідь AI
    db.add_message(user_id, "model", clean_text)

    # Паралельно генеруємо зображення
    img_bytes = None
    if scene:
        char = db.get_character(user_id)
        img_prompt = build_image_prompt(char, scene)
        await update.effective_chat.send_action(ChatAction.UPLOAD_PHOTO)
        img_bytes = await generate_image_horde(img_prompt)

    # Відправляємо відповідь
    if img_bytes:
        await update.message.reply_photo(
            photo=img_bytes,
            caption=clean_text[:1024],  # caption обмежений 1024 символами
        )
        # Якщо текст довший — відправляємо окремо
        if len(clean_text) > 1024:
            await update.message.reply_text(clean_text[1024:])
    else:
        await update.message.reply_text(clean_text)

# ══════════════════════════════════════════════════════════════════════════════
#  Запуск бота
# ══════════════════════════════════════════════════════════════════════════════

def main():
    app = Application.builder().token(BOT_TOKEN).build()

    # ConversationHandler — онбординг та /setup
    conv = ConversationHandler(
        entry_points=[
            CommandHandler("start", cmd_start),
            CommandHandler("setup", cmd_setup),
            CommandHandler("mykey", cmd_mykey),
        ],
        states={
            AWAIT_API_KEY:  [MessageHandler(filters.TEXT & ~filters.COMMAND, receive_api_key)],
            SETUP_NAME:     [MessageHandler(filters.TEXT & ~filters.COMMAND, setup_name)],
            SETUP_APPEARANCE: [MessageHandler(filters.TEXT & ~filters.COMMAND, setup_appearance)],
            SETUP_PERSONALITY: [MessageHandler(filters.TEXT & ~filters.COMMAND, setup_personality)],
            SETUP_SPEECH:   [MessageHandler(filters.TEXT & ~filters.COMMAND, setup_speech)],
            SETUP_CONFIRM:  [CallbackQueryHandler(setup_confirm_callback, pattern="^setup_")],
        },
        fallbacks=[CommandHandler("cancel", cmd_cancel)],
        allow_reentry=True,
    )

    app.add_handler(conv)
    app.add_handler(CommandHandler("reset",   cmd_reset))
    app.add_handler(CommandHandler("imagine", cmd_imagine))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Бот запущено ✅")
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()
