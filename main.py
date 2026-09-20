import os
import base64
import asyncio
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import requests
import pandas as pd

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)


# =========================================================
# CONFIG
# =========================================================

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

TWELVE_DATA_KEY = os.getenv("TWELVE_DATA_KEY")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

PORT = int(os.getenv("PORT", "10000"))


# =========================================================
# SYMBOLS
# =========================================================

SYMBOLS = [
    "BTC/USD",
    "ETH/USD",
    "BNB/USD",
    "SOL/USD",
    "XRP/USD",
    "DOGE/USD",
    "ADA/USD",
    "AVAX/USD",
    "LINK/USD",
    "TRX/USD",
]


# =========================================================
# TIMEFRAMES
# =========================================================

TIMEFRAMES = [
    "1min",
    "5min",
    "15min",
    "30min",
    "45min",
    "1h",
    "2h",
    "4h",
    "8h",
    "1day",
    "1week",
    "1month",
]


# =========================================================
# RENDER HEALTH SERVER
# =========================================================

class HealthHandler(BaseHTTPRequestHandler):

    def do_GET(self):

        self.send_response(200)

        self.send_header(
            "Content-Type",
            "text/plain"
        )

        self.end_headers()

        self.wfile.write(
            b"Crypto Market Bot is running."
        )

    def log_message(self, format, *args):
        return


def start_health_server():

    server = HTTPServer(
        ("0.0.0.0", PORT),
        HealthHandler
    )

    print(
        f"Health server running on port {PORT}"
    )

    server.serve_forever()


# =========================================================
# TWELVE DATA
# =========================================================

def get_market_data(
    symbol,
    interval,
    outputsize=100
):

    if not TWELVE_DATA_KEY:

        print(
            "ERROR: TWELVE_DATA_KEY is missing."
        )

        return None

    url = (
        "https://api.twelvedata.com/"
        "time_series"
    )

    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": outputsize,
        "apikey": TWELVE_DATA_KEY,
        "format": "JSON",
    }

    try:

        response = requests.get(
            url,
            params=params,
            timeout=30
        )

        data = response.json()

        if "values" not in data:

            print(
                f"Twelve Data error "
                f"{symbol} {interval}: "
                f"{data}"
            )

            return None

        df = pd.DataFrame(
            data["values"]
        )

        numeric_columns = [
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]

        for column in numeric_columns:

            if column in df.columns:

                df[column] = pd.to_numeric(
                    df[column],
                    errors="coerce"
                )

        df = df.sort_values(
            "datetime"
        )

        df = df.reset_index(
            drop=True
        )

        return df

    except Exception as e:

        print(
            f"Market data error "
            f"{symbol} {interval}: {e}"
        )

        return None


# =========================================================
# TECHNICAL INDICATORS
# =========================================================

def calculate_indicators(df):

    df = df.copy()

    # EMA 9
    df["EMA9"] = (
        df["close"]
        .ewm(
            span=9,
            adjust=False
        )
        .mean()
    )

    # EMA 21
    df["EMA21"] = (
        df["close"]
        .ewm(
            span=21,
            adjust=False
        )
        .mean()
    )

    # RSI
    delta = df["close"].diff()

    gain = delta.clip(
        lower=0
    )

    loss = -delta.clip(
        upper=0
    )

    avg_gain = (
        gain
        .rolling(14)
        .mean()
    )

    avg_loss = (
        loss
        .rolling(14)
        .mean()
    )

    rs = (
        avg_gain /
        avg_loss.replace(
            0,
            pd.NA
        )
    )

    df["RSI14"] = (
        100 -
        (
            100 /
            (1 + rs)
        )
    )

    # Support / resistance
    df["HIGH20"] = (
        df["high"]
        .rolling(20)
        .max()
    )

    df["LOW20"] = (
        df["low"]
        .rolling(20)
        .min()
    )

    return df


# =========================================================
# MARKET ANALYSIS
# =========================================================

def analyze_market(
    df,
    symbol,
    timeframe
):

    if df is None:

        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "signal": "NO DATA",
            "confidence": 0,
            "price": 0,
            "rsi": 0,
            "ema9": 0,
            "ema21": 0,
            "support": 0,
            "resistance": 0,
        }

    if len(df) < 30:

        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "signal": "NO DATA",
            "confidence": 0,
            "price": 0,
            "rsi": 0,
            "ema9": 0,
            "ema21": 0,
            "support": 0,
            "resistance": 0,
        }

    df = calculate_indicators(df)

    last = df.iloc[-1]

    price = float(
        last["close"]
    )

    ema9 = float(
        last["EMA9"]
    )

    ema21 = float(
        last["EMA21"]
    )

    rsi_value = last["RSI14"]

    if pd.isna(rsi_value):

        rsi = 50.0

    else:

        rsi = float(rsi_value)

    support_value = last["LOW20"]

    resistance_value = last["HIGH20"]

    if pd.isna(support_value):

        support = 0

    else:

        support = float(
            support_value
        )

    if pd.isna(resistance_value):

        resistance = 0

    else:

        resistance = float(
            resistance_value
        )

    score = 0

    # -----------------------------------------------------
    # EMA TREND
    # -----------------------------------------------------

    if ema9 > ema21:

        score += 1

    elif ema9 < ema21:

        score -= 1

    # -----------------------------------------------------
    # RSI
    # -----------------------------------------------------

    if rsi >= 55:

        score += 1

    elif rsi <= 45:

        score
