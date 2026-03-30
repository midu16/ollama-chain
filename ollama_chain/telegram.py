import os
import sys
import asyncio
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters
from .chains import chain_auto
from .models import discover_models, model_names
from .sysinfo import collect_snapshot
from dotenv import load_dotenv
load_dotenv()

_READONLY_INSTRUCTION = (
    "You are a READ-ONLY assistant. You have access to live system data shown "
    "below. Use it to answer the user's question accurately. You MUST NOT "
    "suggest, generate, or perform any command that modifies system state "
    "(no writes, installs, service restarts, config changes, etc.). "
    "Only provide information and analysis."
)

def _build_enriched_query(query: str) -> str:
    """Prepend live system context and read-only instructions to the query."""
    snapshot = collect_snapshot()
    return (
        f"{_READONLY_INSTRUCTION}\n\n"
        f"=== LIVE SYSTEM DATA ===\n{snapshot}\n\n"
        f"User question: {query}"
    )


async def start(update: Update, context):
    await update.message.reply_text('Hello! Send me a query to process.')

_TG_MAX_LEN = 4096

async def _send_long(message, text: str):
    """Split text into chunks that fit Telegram's message length limit."""
    while text:
        if len(text) <= _TG_MAX_LEN:
            await message.reply_text(text)
            break
        split = text.rfind("\n", 0, _TG_MAX_LEN)
        if split <= 0:
            split = _TG_MAX_LEN
        await message.reply_text(text[:split])
        text = text[split:].lstrip("\n")

async def handle_message(update: Update, context):
    query = update.message.text
    enriched = _build_enriched_query(query)
    print(f"[telegram] System snapshot attached to query", file=sys.stderr)
    all_models = model_names(discover_models())
    response = await asyncio.to_thread(chain_auto, enriched, all_models)
    await _send_long(update.message, response)

def main():
    token = os.getenv('TELEGRAM_BOT_TOKEN')
    if not token:
        raise ValueError('TELEGRAM_BOT_TOKEN environment variable not set')
    
    application = Application.builder().token(token).build()
    
    application.add_handler(CommandHandler('start', start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    application.run_polling()

if __name__ == '__main__':
    main()