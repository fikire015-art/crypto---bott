# ============================================================
# CRYPTO FLOW BOT
# Gemini + Groq + Technical Analysis + Telegram
# ============================================================

import os
import re
import io
import json
import base64
import logging
import threading
from datetime import datetime, timezone

import requests
import pandas as pd
from PIL import Image

from flask import Flask
from groq import Groq
from google import genai

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

# ============================================================
# CONFIG
# ============================================================

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
TWELVE_DATA_KEY = os.getenv("TWELVE_DATA_KEY", "").strip()

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.8-flash"
)

TIMEFRAME = "15min"
CANDLE_COUNT = 150

CHECK_SECONDS = 60

MIN_TARGET_PIPS = 50

GROQ_VISION_MODELS = [
    "qwen/qwen3.6-27b",
    "qwen/qwen3.8-27b",
]

GROQ_MAX_TOKENS = 350
VISION_MAX_TOKENS = 400

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

logger = logging.getLogger("CryptoFlowBot")


# ============================================================
# CLIENTS
# ============================================================

groq_client = None
gemini_client = None

if GROQ_API_KEY:
    try:
        groq_client = Groq(api_key=GROQ_API_KEY)
    except Exception as e:
        logger.error("Groq client error: %s", e)

if GEMINI_API_KEY:
    try:
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception as e:
        logger.error("Gemini client error: %s", e)


# ============================================================
# FLASK SERVER FOR RENDER
# ============================================================

app = Flask(__name__)


@app.route("/")
def home():
    return "Crypto Flow Bot is running."


@app.route("/health")
def health():
    return {
        "status": "ok",
        "bot": "Crypto Flow Bot",
        "time": datetime.now(timezone.utc).isoformat(),
    }


def run_web():
    port = int(os.getenv("PORT", "10000"))
    app.run(
        host="0.0.0.0",
        port=port,
        use_reloader=False,
    )


# ============================================================
# SYMBOL NORMALIZATION
# ============================================================

SYMBOL_ALIASES = {
    "GOLD": "XAUUSD",
    "XAU": "XAUUSD",
    "XAU/USD": "XAUUSD",

    "BTC": "BTCUSDT",
    "BTCUSD": "BTCUSDT",

    "ETH": "ETHUSDT",
    "ETHUSD": "ETHUSDT",

    "BNB": "BNBUSDT",
    "BNBUSD": "BNBUSDT",

    "SOL": "SOLUSDT",
    "SOLUSD": "SOLUSDT",

    "XRP": "XRPUSDT",
    "XRPUSD": "XRPUSDT",

    "DOGE": "DOGEUSDT",
    "DOGEUSD": "DOGEUSDT",

    "ADA": "ADAUSDT",
    "ADAUSD": "ADAUSDT",

    "EURUSD": "EUR/USD",
    "EUR/USD": "EUR/USD",

    "GBPUSD": "GBP/USD",
    "GBP/USD": "GBP/USD",

    "USDJPY": "USD/JPY",
    "USD/JPY": "USD/JPY",

    "AUDUSD": "AUD/USD",
    "AUD/USD": "AUD/USD",

    "USDCAD": "USD/CAD",
    "USD/CAD": "USD/CAD",

    "USDCHF": "USD/CHF",
    "USD/CHF": "USD/CHF",

    "NZDUSD": "NZD/USD",
    "NZD/USD": "NZD/USD",
}


def normalize_symbol(text):
    text = text.upper().strip()
    text = text.replace(" ", "")

    return SYMBOL_ALIASES.get(text, text)


def extract_symbol(text):
    text = text.upper()

    patterns = [
        r"XAUUSD",
        r"XAU/USD",
        r"GOLD",

        r"BTCUSDT",
        r"BTCUSD",
        r"BTC",

        r"ETHUSDT",
        r"ETHUSD",
        r"ETH",

        r"BNBUSDT",
        r"SOLUSDT",
        r"XRPUSDT",
        r"ADAUSDT",
        r"DOGEUSDT",

        r"EURUSD",
        r"EUR/USD",

        r"GBPUSD",
        r"GBP/USD",

        r"USDJPY",
        r"USD/JPY",

        r"AUDUSD",
        r"AUD/USD",

        r"USDCAD",
        r"USD/CAD",

        r"USDCHF",
        r"USD/CHF",

        r"NZDUSD",
        r"NZD/USD",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)

        if match:
            return normalize_symbol(match.group(0))

    return None


# ============================================================
# MARKET DATA
# ============================================================

def get_binance_data(symbol):
    url = "https://api.binance.com/api/v3/klines"

    params = {
        "symbol": symbol,
        "interval": "15m",
        "limit": CANDLE_COUNT,
    }

    response = requests.get(
        url,
        params=params,
        timeout=15,
    )

    response.raise_for_status()

    data = response.json()

    if not data:
        return None

    rows = []

    for candle in data:
        rows.append(
            {
                "timestamp": candle[0],
                "open": float(candle[1]),
                "high": float(candle[2]),
                "low": float(candle[3]),
                "close": float(candle[4]),
                "volume": float(candle[5]),
            }
        )

    return pd.DataFrame(rows)


def get_twelve_data(symbol):
    if not TWELVE_DATA_KEY:
        return None

    url = "https://api.twelvedata.com/time_series"

    params = {
        "symbol": symbol,
        "interval": TIMEFRAME,
        "outputsize": CANDLE_COUNT,
        "apikey": TWELVE_DATA_KEY,
    }

    response = requests.get(
        url,
        params=params,
        timeout=20,
    )

    response.raise_for_status()

    data = response.json()

    if "values" not in data:
        raise Exception(
            data.get("message", "Twelve Data error")
        )

    df = pd.DataFrame(data["values"])

    df = df.rename(
        columns={
            "datetime": "timestamp",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
        }
    )

    for column in [
        "open",
        "high",
        "low",
        "close",
    ]:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df.dropna()

    df = df.iloc[::-1].reset_index(drop=True)

    return df


def get_market_data(symbol):
    symbol = normalize_symbol(symbol)

    crypto_symbols = [
        "BTCUSDT",
        "ETHUSDT",
        "BNBUSDT",
        "SOLUSDT",
        "XRPUSDT",
        "ADAUSDT",
        "DOGEUSDT",
    ]

    if symbol in crypto_symbols:
        return get_binance_data(symbol)

    return get_twelve_data(symbol)


# ============================================================
# TECHNICAL INDICATORS
# ============================================================

def calculate_indicators(df):

    df = df.copy()

    df["EMA20"] = (
        df["close"]
        .ewm(span=20, adjust=False)
        .mean()
    )

    df["EMA50"] = (
        df["close"]
        .ewm(span=50, adjust=False)
        .mean()
    )

    df["EMA200"] = (
        df["close"]
        .ewm(span=200, adjust=False)
        .mean()
    )

    delta = df["close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss.replace(0, pd.NA)

    df["RSI"] = 100 - (
        100 / (1 + rs)
    )

    high_low = (
        df["high"] - df["low"]
    )

    high_close = (
        df["high"] - df["close"].shift()
    ).abs()

    low_close = (
        df["low"] - df["close"].shift()
    ).abs()

    tr = pd.concat(
        [
            high_low,
            high_close,
            low_close,
        ],
        axis=1,
    ).max(axis=1)

    df["ATR"] = tr.rolling(14).mean()

    ema12 = (
        df["close"]
        .ewm(span=12, adjust=False)
        .mean()
    )

    ema26 = (
        df["close"]
        .ewm(span=26, adjust=False)
        .mean()
    )

    df["MACD"] = ema12 - ema26

    df["MACD_SIGNAL"] = (
        df["MACD"]
        .ewm(span=9, adjust=False)
        .mean()
    )

    df["MOMENTUM"] = (
        df["close"].pct_change(5) * 100
    )

    return df.dropna()


# ============================================================
# PIP / PRICE HELPERS
# ============================================================

def pip_size(symbol):

    symbol = normalize_symbol(symbol)

    if symbol == "XAUUSD":
        return 0.10

    if symbol.endswith("JPY"):
        return 0.01

    if "/" in symbol:
        return 0.0001

    return 0.0001


def format_price(price, symbol):

    symbol = normalize_symbol(symbol)

    if price is None:
        return "Not available"

    if symbol == "XAUUSD":
        return f"{price:.2f}"

    if "JPY" in symbol:
        return f"{price:.3f}"

    if symbol.endswith("USDT"):
        return f"{price:.4f}"

    return f"{price:.5f}"


# ============================================================
# TECHNICAL MARKET ANALYSIS
# ============================================================

def analyze_market(df, symbol):

    df = calculate_indicators(df)

    
