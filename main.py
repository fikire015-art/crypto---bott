import os
import time
import threading
import logging
from typing import Optional

import requests
import pandas as pd

from flask import Flask, jsonify

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
)

# =========================================================
# CONFIG
# =========================================================

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
TWELVE_DATA_KEY = os.getenv("TWELVE_DATA_KEY", "").strip()

TIMEFRAME = "15min"

# Keep this small because Twelve Data has request limits.
DEFAULT_SYMBOLS = [
    "BTC/USD",
    "ETH/USD",
    "SOL/USD",
]

TWELVE_DATA_URL = "https://api.twelvedata.com/time_series"

# =========================================================
# LOGGING
# =========================================================

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)

# =========================================================
# FLASK
# =========================================================

flask_app = Flask(__name__)


@flask_app.route("/")
def home():
    return jsonify({
        "status": "online",
        "service": "Crypto Market Telegram Bot",
        "timeframe": TIMEFRAME
    })


@flask_app.route("/health")
def health():
    return jsonify({"status": "healthy"})


def run_flask():
    port = int(os.getenv("PORT", "10000"))

    flask_app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
        use_reloader=False
    )


# =========================================================
# AUTHORIZATION
# =========================================================

def authorized(update: Update) -> bool:

    if not TELEGRAM_CHAT_ID:
        return True

    if not update.effective_chat:
        return False

    return str(update.effective_chat.id) == TELEGRAM_CHAT_ID


async def deny(update: Update):

    if update.message:
        await update.message.reply_text(
            "⛔ This Telegram chat is not authorized."
        )


# =========================================================
# API CACHE / RATE CONTROL
# =========================================================

CACHE = {}

CACHE_SECONDS = 65

API_LOCK = threading.Lock()

LAST_API_REQUEST = 0.0

MIN_SECONDS_BETWEEN_REQUESTS = 9


def get_cached(symbol: str):

    item = CACHE.get(symbol)

    if not item:
        return None

    timestamp, dataframe = item

    if time.time() - timestamp < CACHE_SECONDS:
        return dataframe.copy()

    return None


def save_cache(symbol: str, dataframe: pd.DataFrame):

    CACHE[symbol] = (
        time.time(),
        dataframe.copy()
    )


def wait_for_api_slot():

    global LAST_API_REQUEST

    with API_LOCK:

        now = time.time()

        elapsed = now - LAST_API_REQUEST

        if elapsed < MIN_SECONDS_BETWEEN_REQUESTS:

            wait_time = (
                MIN_SECONDS_BETWEEN_REQUESTS
                - elapsed
            )

            logger.info(
                "Waiting %.1f seconds before API request",
                wait_time
            )

            time.sleep
