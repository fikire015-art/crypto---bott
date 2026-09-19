import os
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
# CONFIGURATION
# =========================================================

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
TWELVE_DATA_KEY = os.getenv("TWELVE_DATA_KEY", "").strip()

TIMEFRAME = "15min"

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
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)

# =========================================================
# FLASK HEALTH SERVER
# =========================================================

flask_app = Flask(__name__)


@flask_app.route("/")
def home():
    return jsonify(
        {
            "status": "online",
            "service": "Crypto Market Telegram Bot",
            "timeframe": TIMEFRAME,
        }
    )


@flask_app.route("/health")
def health():
    return jsonify({"status": "healthy"})


def run_flask():
    port = int(os.getenv("PORT", "10000"))

    flask_app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
        use_reloader=False,
    )


# =========================================================
# TELEGRAM SECURITY
# =========================================================

def authorized(update: Update) -> bool:
    """
    If TELEGRAM_CHAT_ID is configured, only that Telegram
    chat can use the bot.
    """

    if not TELEGRAM_CHAT_ID:
        return True

    if not update.effective_chat:
        return False

    current_chat_id = str(update.effective_chat.id)

    return current_chat_id == TELEGRAM_CHAT_ID


async def deny(update: Update):
    if update.message:
        await update.message.reply_text(
            "⛔ This Telegram chat is not authorized."
        )


# =========================================================
# TWELVE DATA
# =========================================================

def get_market_data(symbol: str) -> Optional[pd.DataFrame]:
    if not TWELVE_DATA_KEY:
        logger.error("TWELVE_DATA_KEY is missing.")
        return None

    params = {
        "symbol": symbol,
        "interval": TIMEFRAME,
        "outputsize": 100,
        "apikey": TWELVE_DATA_KEY,
    }

    try:
        response = requests.get(
            TWELVE_DATA_URL,
            params=params,
            timeout=20,
        )

        response.raise_for_status()

        data = response.json()

        if "status" in data and data["status"] == "error":
            logger.error("Twelve Data error: %s", data)
            return None

        values = data.get("values")

        if not values:
            logger.error("No market data returned for %s", symbol)
            return None

        df = pd.DataFrame(values)

        required_columns = [
            "datetime",
            "open",
            "high",
            "low",
            "close",
        ]

        for column in required_columns:
            if column not in df.columns:
                logger.error(
                    "Missing column %s for %s",
                    column,
                    symbol,
                )
                return None

        for column in ["open", "high", "low", "close"]:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

        df["datetime"] = pd.to_datetime(
            df["datetime"],
            errors="coerce",
        )

        df = df.dropna(
            subset=[
                "datetime",
                "open",
                "high",
                "low",
                "close",
            ]
        )

        df = df.sort_values("datetime").reset_index(drop=True)

        return df

    except requests.RequestException as exc:
        logger.error(
            "Network error for %s: %s",
            symbol,
            exc,
        )
        return None

    except Exception as exc:
        logger.exception(
            "Unexpected market data error: %s",
            exc,
        )
        return None


# =========================================================
# TECHNICAL ANALYSIS
# =========================================================

def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    average_gain = gain.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False,
    ).mean()

    average_loss = loss.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False,
    ).mean()

    rs = average_gain / average_loss.replace(0, pd.NA)

    rsi = 100 - (100 / (1 + rs))

    return rsi.fillna(50)


def analyze_market(
    df: pd.DataFrame,
    symbol: str,
) -> dict:

    if len(df) < 30:
        raise Value
