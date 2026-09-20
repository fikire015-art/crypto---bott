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


# Twelve Data supported intervals
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
        print("TWELVE_DATA_KEY missing")
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
# INDICATORS
# =========================================================

def calculate_indicators(df):

    df = df.copy()

    df["EMA9"] = (
        df["close"]
        .ewm(
            span=9,
            adjust=False
        )
        .mean()
    )

    df["EMA21"] = (
        df["close"]
        .ewm(
            span=21,
            adjust=False
        )
        .mean()
    )

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

    if df is None or len(df) < 30:

        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "signal": "NO DATA",
            "confidence": 0,
            "price": 0,
            "rsi": 0,
            "ema9": 0,
            "ema21": 0,
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

    rsi = float(
        last["RSI14"]
    )

    score = 0

    # EMA trend
    if ema9 > ema21:
        score += 1

    elif ema9 < ema21:
        score -= 1

    # RSI
    if rsi >= 55:
        score += 1

    elif rsi <= 45:
        score -= 1

    # Price vs EMA
    if price > ema9:
        score += 1

    elif price < ema9:
        score -= 1

    if score >= 2:

        signal = "BUY"

    elif score <= -2:

        signal = "SELL"

    else:

        signal = "WAIT"

    confidence = min(
        95,
        50 + abs(score) * 15
    )

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "signal": signal,
        "confidence": confidence,
        "price": price,
        "rsi": rsi,
        "ema9": ema9,
        "ema21": ema21,
    }


# =========================================================
# FORMAT RESULT
# =========================================================

def format_result(result):

    signal = result["signal"]

    if signal == "BUY":
        emoji = "🟢"

    elif signal == "SELL":
        emoji = "🔴"

    elif signal == "WAIT":
        emoji = "🟡"

    else:
        emoji = "⚪"

    return (
        f"{emoji} {result['symbol']} "
        f"| {result['timeframe']}\n"
        f"Signal: {signal}\n"
        f"Confidence: "
        f"{result['confidence']}%\n"
        f"Price: {result['price']:.6f}\n"
        f"RSI: {result['rsi']:.2f}\n"
        f"EMA9: {result['ema9']:.6f}\n"
        f"EMA21: {result['ema21']:.6f}"
    )


# =========================================================
# SINGLE ANALYSIS
# =========================================================

def run_analysis(
    symbol,
    timeframe
):

    df = get_market_data(
        symbol,
        timeframe,
        100
    )

    return analyze_market(
        df,
        symbol,
        timeframe
    )


# =========================================================
# /START
# =========================================================

async def start_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    await update.message.reply_text(
        "🤖 CRYPTO MARKET BOT\n\n"
        "Commands:\n\n"
        "/analyze BTC/USD 15min\n"
        "/scan BTC/USD 1h\n"
        "/all\n"
        "/symbols\n"
        "/timeframes\n\n"
        "📸 Send a chart screenshot "
        "for Photo Scan."
    )


# =========================================================
# /SYMBOLS
# =========================================================

async def symbols_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    text = (
        "🪙 SYMBOLS\n\n" +
        "\n".join(
            f"• {x}"
            for x in SYMBOLS
        )
    )

    await update.message.reply_text(
        text
    )


# =========================================================
# /TIMEFRAMES
# =========================================================

async def timeframes_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    text = (
        "⏱ TIMEFRAMES\n\n" +
        "\n".join(
            f"• {x}"
            for x in TIMEFRAMES
        )
    )

    await update.message.reply_text(
        text
    )


# =========================================================
# /ANALYZE
# =========================================================

async def analyze_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    if len(context.args) < 2:

        await update.message.reply_text(
            "Usage:\n"
            "/analyze BTC/USD 15min"
        )

        return

    symbol = context.args[0].upper()
    timeframe = context.args[1]

    if timeframe not in TIMEFRAMES:

        await update.message.reply_text(
            "Invalid timeframe.\n\n"
            + ", ".join(TIMEFRAMES)
        )

        return

    await update.message.reply_text(
        f"🔎 Analyzing "
        f"{symbol} {timeframe}..."
    )

    result = await asyncio.to_thread(
        run_analysis,
        symbol,
        timeframe
    )

    await update.message.reply_text(
        format_result(result)
    )


# =========================================================
# /SCAN
# =========================================================

async def scan_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    if len(context.args) < 2:

        await update.message.reply_text(
            "Usage:\n"
            "/scan BTC/USD 1h"
        )

        return

    symbol = context.args[0].upper()
    timeframe = context.args[1]

    await update.message.reply_text(
        f"📊 Scanning "
        f"{symbol} {timeframe}..."
    )

    result = await asyncio.to_thread(
        run_analysis,
        symbol,
        timeframe
    )

    await update.message.reply_text(
        format_result(result)
    )


# =========================================================
# /ALL
# =========================================================

async def all_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    await update.message.reply_text(
        "🚀 ALL SCAN STARTED\n\n"
        f"Symbols: {len(SYMBOLS)}\n"
        f"Timeframes: {len(TIMEFRAMES)}\n"
        f"Total scans: "
        f"{len(SYMBOLS) * len(TIMEFRAMES)}\n\n"
        "This may take some time."
    )

    results = []

    total = (
        len(SYMBOLS)
        * len(TIMEFRAMES)
    )

    counter = 0

    for symbol in SYMBOLS:

        for timeframe in TIMEFRAMES:

            counter += 1

            result = await asyncio.to_thread(
                run_analysis,
                symbol,
                timeframe
            )

            results.append(
                result
            )

            # Small delay to reduce
            # API pressure
            await asyncio.sleep(
                0.25
            )

            if counter % 20 == 0:

                await update.message.reply_text(
                    f"⏳ Progress: "
                    f"{counter}/{total}"
                )

    # Send results in chunks
    lines = []

    for result in results:

        lines.append(
            f"{result['symbol']} "
            f"{result['timeframe']} "
            f"→ {result['signal']} "
            f"({result['confidence']}%)"
        )

    chunk = []

    for line in lines:

        chunk.append(line)

        if len(chunk) >= 25:

            await update.message.reply_text(
                "📊 ALL RESULTS\n\n"
                + "\n".join(chunk)
            )

            chunk = []

    if chunk:

        await update.message.reply_text(
            "📊 ALL RESULTS\n\n"
            + "\n".join(chunk)
        )

    await update.message.reply_text(
        "✅ ALL SCAN COMPLETE"
    )


# =========================================================
# GROQ PHOTO SCAN
# =========================================================

def groq_photo_scan(
    image_bytes
):

    if not GROQ_API_KEY:

        return (
            "GROQ_API_KEY is missing."
        )

    try:

        encoded = base64.b64encode(
            image_bytes
        ).decode("utf-8")

        url = (
            "https://api.groq.com/"
            "openai/v1/chat/completions"
        )

        headers = {
            "Authorization":
                f"Bearer {GROQ_API_KEY}",
            "Content-Type":
                "application/json",
        }

        prompt = """
Analyze this trading chart screenshot.

Identify, if visible:
- Symbol
- Timeframe
- Trend
- Support
- Resistance
- EMA
- RSI
- Candlestick structure
- Possible BUY / SELL / WAIT setup

Do not guarantee profit.
If information cannot be read from the image,
say so clearly.
"""

        payload = {
            "model":
                "qwen/qwen3.6-27b",

            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type":
                                "image_url",
                            "image_url": {
                                "url":
                                    "data:image/jpeg;"
                                    f"base64,{encoded}"
                            }
                        }
                    ]
                }
            ],

            "temperature": 0.2,

            "max_completion_tokens":
                1200
        }

        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=60
        )

        data = response.json()

        if "choices" not in data:

            return (
                "Groq error:\n"
                + str(data)
            )

        return (
            data["choices"][0]
            ["message"]
            ["content"]
        )

    except Exception as e:

        return (
            "Groq Photo Scan error: "
            + str(e)
        )


# =========================================================
# GEMINI PHOTO SCAN
# =========================================================

def gemini_photo_scan(
    image_bytes
):

    if not GEMINI_API_KEY:

        return (
            "GEMINI_API_KEY is missing."
        )

    try:

        encoded = base64.b64encode(
            image_bytes
        ).decode("utf-8")

        url = (
            "https://generativelanguage.googleapis.com/"
            "v1beta/models/gemini-2.5-flash:generateContent"
        )

        params = {
            "key": GEMINI_API_KEY
        }

        prompt = """
Analyze this trading chart screenshot.

Report:
1. Symbol
2. Timeframe
3. Trend
4. Support
5. Resistance
6. Candlestick pattern
7. RSI/EMA if visible
8. BUY / SELL / WAIT setup
9. Main reason
10. Risk warning

Do not guarantee profit.
Only use information visible in the image.
"""

        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt
                        },
                        {
                            "inline_data": {
                                "mime_type":
                                    "image/jpeg",
                                "data":
                                    encoded
                            }
                        }
                    ]
                }
            ]
        }

        response = requests.post(
            url,
            params=params,
            json=payload,
            timeout=60
        )

        data = response.json()

        if "candidates" not in data:

            return (
                "Gemini error:\n"
                + str(data)
            )

        return (
            data["candidates"][0]
            ["content"]
            ["parts"][0]
            ["text"]
        )

    except Exception as e:

        return (
            "Gemini Photo Scan error: "
            + str(e)
        )


# =========================================================
# PHOTO HANDLER
# =========================================================

async def photo_handler(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    if not update.message:

        return

    photo = update.message.photo

    if not photo:

        return

    await update.message.reply_text(
        "📸 PHOTO SCAN STARTED...\n\n"
        "🤖 Checking chart with AI..."
    )

    try:

        largest = photo[-1]

        telegram_file = (
            await context.bot
            .get_file(
                largest.file_id
            )
        )

        image_bytes = (
            await telegram_file
            .download_as_bytearray()
        )

        image_bytes
