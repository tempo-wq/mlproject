"""
Telegram-бот для транскрибации, суммаризации и оценки важности голосовых сообщений.
Транскрибация: faster-whisper (speech-to-text)
Суммаризация: классический ML (TF-IDF + TextRank)
Важность: Zero-Shot NLP (ruBERT-tiny)
"""

import os
import logging
import asyncio
import tempfile
from pathlib import Path

from telegram import Update, Message
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)
from telegram.constants import ParseMode

# Импорты локальных ML модулей
from transcriber import Transcriber
from summarizer import Summarizer
from importance import ImportanceScorer

# СНАЧАЛА создаем папку для логов:
Path("logs").mkdir(exist_ok=True)

# ПОТОМ настраиваем логи:
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/bot.log"),
    ],
)
logger = logging.getLogger(__name__)

# Инициализация моделей (один раз при старте бота)
logger.info("Инициализация ML-моделей...")
transcriber = Transcriber(model_size="small")
summarizer = Summarizer()
importance_scorer = ImportanceScorer()
logger.info("Все модели успешно загружены!")


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Приветственное сообщение."""
    text = (
        "👋 *Привет!* Я бот для анализа голосовых сообщений.\n\n"
        "📩 *Что умею:*\n"
        "• Расшифрую аудио в текст (Whisper)\n"
        "• Сделаю краткую выжимку главного (TextRank)\n"
        "• Оценю важность и срочность сообщения (ruBERT-tiny)\n\n"
        "🎙️ Просто перешли мне голосовое или аудиофайл!"
    )
    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Справка."""
    text = (
        "ℹ️ *Справка*\n\n"
        "1. Перешли боту голосовое сообщение\n"
        "2. Бот обработает его через каскад ML-моделей без блокировки интерфейса\n\n"
        "Команды:\n"
        "/start — начало работы\n"
        "/help — эта справка\n"
        "/stats — статистика обработанных сообщений"
    )
    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)


async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Статистика."""
    processed = context.bot_data.get("processed", 0)
    total_words = context.bot_data.get("total_words", 0)
    await update.message.reply_text(
        f"📊 *Статистика бота*\n\n"
        f"Обработано сообщений: {processed}\n"
        f"Расшифровано слов: {total_words}",
        parse_mode=ParseMode.MARKDOWN,
    )


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Основной обработчик голосовых и аудио сообщений."""
    message: Message = update.message

    # Определяем тип — голосовое или аудиофайл
    if message.voice:
        file_obj = message.voice
        file_ext = ".ogg"
    elif message.audio:
        file_obj = message.audio
        file_ext = ".mp3"
    else:
        return

    status_msg = await message.reply_text("⏳ Обрабатываю аудио...")

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = Path(tmpdir) / f"audio{file_ext}"
            file = await context.bot.get_file(file_obj.file_id)
            await file.download_to_drive(audio_path)

            # 1. ТРАНСКРИБАЦИЯ (в отдельном потоке)
            await status_msg.edit_text("🔊 Расшифровываю речь...")
            transcript, language, duration = await asyncio.to_thread(
                transcriber.transcribe, str(audio_path)
            )

            if not transcript.strip():
                await status_msg.edit_text("❌ Не удалось распознать речь. Попробуй другое аудио.")
                return

            # 2. СУММАРИЗАЦИЯ (в отдельном потоке)
            await status_msg.edit_text("📝 Делаю выжимку главного...")
            summary = await asyncio.to_thread(summarizer.summarize, transcript)

            # 3. ОЦЕНКА ВАЖНОСТИ (в отдельном потоке)
            await status_msg.edit_text("🔍 Оцениваю важность текста...")
            importance_data = await asyncio.to_thread(importance_scorer.score, transcript)

            # Формируем метаданные для ответа
            word_count = len(transcript.split())
            compress_ratio = len(summary.split()) / max(word_count, 1)
            
            is_important = importance_data["is_important"]
            imp_score = importance_data["score"]
            if imp_score >= 0.70:
                alert_icon = "🔴 Ваш собеседник уже красный"
            elif imp_score >= 0.40:
                # Зона неуверенности модели (от 40% до 70%)
                alert_icon = "🟡 Неоднозначно (стоит прослушать на x2)"
            else:
                alert_icon = "🟢 Заблокируйте собеседника(воздух гоняет) "

            # Сборка финального сообщения
            response = (
                f"🎙️ *Расшифровка* (язык: {language.upper()}, {duration:.0f} сек):\n"
                f"```\n{transcript}\n```\n\n"
                f"📌 *Краткая выжимка* (~{compress_ratio:.0%} от оригинала):\n"
                f"{summary}\n\n"
                f"📊 *Статус*: {alert_icon} (Уверенность: {imp_score:.0%})"
            )

            await status_msg.edit_text(response, parse_mode=ParseMode.MARKDOWN)

            # Обновляем статистику бота
            context.bot_data["processed"] = context.bot_data.get("processed", 0) + 1
            context.bot_data["total_words"] = context.bot_data.get("total_words", 0) + word_count

            logger.info(f"Успешно обработано: {word_count} слов, важность={is_important}")

    except Exception as e:
        logger.error(f"Ошибка при обработке аудио: {e}", exc_info=True)
        await status_msg.edit_text(
            "❌ Произошла ошибка при обработке. Попробуй ещё раз."
        )


async def handle_forwarded(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик для пересланных голосовых."""
    await handle_voice(update, context)


def main() -> None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise ValueError("Переменная окружения TELEGRAM_BOT_TOKEN не задана!")

    app = Application.builder().token(token).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("stats", stats_command))
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, handle_voice))

    logger.info("Бот запущен и готов к работе...")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
