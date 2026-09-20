import os
import base64
import asyncio
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

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

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

TWELVE_DATA_KEY = os.getenv("TWELVE_DATA_KEY", "").strip()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

PORT = int(os.getenv("PORT", "10000"))

# Current vision models
GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "qwen/qwen3.6-27b"
).strip()

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.6-flash"
).strip()


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
            "text/plain; charset=utf-8"
        )

        self.end_headers()

        self.wfile.write(
            b"Crypto Market Bot is running."
        )

    def do_HEAD(self):

        self.send_response(200)

        self.send_header(
            "Content-Type",
            "text/plain; charset=utf-8"
        )

        self.end_headers()

    def log_message(self, format, *args):
        return


def start_health_server():

    try:

        server = ThreadingHTTPServer(
            ("0.0.0.0", PORT),
            HealthHandler
        )

        print(
            f"Health server running on 0.0.0.0:{PORT}",
            flush=True
        )

        server.serve_forever()

    except Exception as e:

        print(
            f"Health server error: {e}",
            flush=True
        )


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
            "WARNING: TWELVE_DATA_KEY is missing.",
            flush=True
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

        response.raise_for_status()

        data = response.json()

        if "values" not in data:

            print(
                f"Twelve Data error "
                f"{symbol} {interval}: "
                f"{data}",
                flush=True
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

        if "datetime" in df.columns:

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
            f"{symbol} {interval}: {e}",
            flush=True
        )

        return None


# =========================================================
# TECHNICAL INDICATORS
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

    # RSI 14
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

    avg_loss = avg_loss.replace(
        0,
        float("nan")
    )

    rs = (
        avg_gain /
        avg_loss
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

    empty_result = {
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

    if df is None:
        return empty_result

    if len(df) < 30:
        return empty_result

    try:

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

            rsi = float(
                rsi_value
            )

        support_value = last["LOW20"]

        resistance_value = last["HIGH20"]

        support = (
            0
            if pd.isna(support_value)
            else float(support_value)
        )

        resistance = (
            0
            if pd.isna(resistance_value)
            else float(resistance_value)
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

        # Price vs EMA9
        if price > ema9:
            score += 1

        elif price < ema9:
            score -= 1

        # Signal
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
            "support": support,
            "resistance": resistance,
        }

    except Exception as e:

        print(
            f"Analysis error: {e}",
            flush=True
        )

        return empty_result


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
        f"| {result['timeframe']}\n\n"

        f"Signal: {signal}\n"

        f"Confidence: "
        f"{result['confidence']}%\n\n"

        f"Price: "
        f"{result['price']:.6f}\n"

        f"RSI: "
        f"{result['rsi']:.2f}\n"

        f"EMA9: "
        f"{result['ema9']:.6f}\n"

        f"EMA21: "
        f"{result['ema21']:.6f}\n\n"

        f"Support: "
        f"{result['support']:.6f}\n"

        f"Resistance: "
        f"{result['resistance']:.6f}"
    )


# =========================================================
# RUN ANALYSIS
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

    if not update.message:
        return

    text = (
        "🤖 CRYPTO MARKET BOT\n\n"

        "📊 Commands:\n\n"

        "/analyze BTC/USD 15min\n"
        "/scan BTC/USD 1h\n"
        "/all\n"
        "/symbols\n"
        "/timeframes\n\n"

        "📸 PHOTO SCAN\n"
        "Send a trading chart screenshot.\n\n"

        "🤖 Gemini + Groq will analyze it."
    )

    await update.message.reply_text(
        text
    )


# =========================================================
# /PING
# =========================================================

async def ping_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    if update.message:

        await update.message.reply_text(
            "🟢 BOT IS ONLINE"
        )


# =========================================================
# /SYMBOLS
# =========================================================

async def symbols_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    if not update.message:
        return

    text = (
        "🪙 AVAILABLE SYMBOLS\n\n"
        +
        "\n".join(
            f"• {symbol}"
            for symbol in SYMBOLS
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

    if not update.message:
        return

    text = (
        "⏱ AVAILABLE TIMEFRAMES\n\n"
        +
        "\n".join(
            f"• {timeframe}"
            for timeframe in TIMEFRAMES
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

    if not update.message:
        return

    if len(context.args) < 2:

        await update.message.reply_text(
            "Usage:\n\n"
            "/analyze BTC/USD 15min"
        )

        return

    symbol = (
        context.args[0]
        .upper()
    )

    timeframe = (
        context.args[1]
        .lower()
    )

    if timeframe not in TIMEFRAMES:

        await update.message.reply_text(
            "❌ Invalid timeframe.\n\n"
            "Available:\n"
            +
            ", ".join(TIMEFRAMES)
        )

        return

    await update.message.reply_text(
        f"🔎 Analyzing "
        f"{symbol} "
        f"{timeframe}..."
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

    if not update.message:
        return

    if len(context.args) < 2:

        await update.message.reply_text(
            "Usage:\n\n"
            "/scan BTC/USD 1h"
        )

        return

    symbol = (
        context.args[0]
        .upper()
    )

    timeframe = (
        context.args[1]
        .lower()
    )

    if timeframe not in TIMEFRAMES:

        await update.message.reply_text(
            "❌ Invalid timeframe."
        )

        return

    await update.message.reply_text(
        f"📊 Scanning "
        f"{symbol} "
        f"{timeframe}..."
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

    if not update.message:
        return

    total = (
        len(SYMBOLS)
        *
        len(TIMEFRAMES)
    )

    await update.message.reply_text(
        "🚀 ALL SCAN STARTED\n\n"

        f"🪙 Symbols: {len(SYMBOLS)}\n"
        f"⏱ Timeframes: {len(TIMEFRAMES)}\n"
        f"📊 Total scans: {total}\n\n"

        "Please wait..."
    )

    results = []

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

            await asyncio.sleep(
                0.25
            )

            if counter % 20 == 0:

                await update.message.reply_text(
                    f"⏳ Progress "
                    f"{counter}/{total}"
                )

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
                +
                "\n".join(chunk)
            )

            chunk = []

    if chunk:

        await update.message.reply_text(
            "📊 ALL RESULTS\n\n"
            +
            "\n".join(chunk)
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
            "❌ GROQ_API_KEY is missing.\n\n"
            "Add it in Render → Environment."
        )

    try:

        encoded_image = (
            base64.b64encode(
                image_bytes
            )
            .decode("utf-8")
        )

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
You are a technical chart analysis assistant.

Analyze ONLY what is visible in the trading
chart screenshot.

Return:

1. Symbol
2. Timeframe
3. Current trend
4. Support
5. Resistance
6. Candlestick structure
7. EMA if visible
8. RSI if visible
9. BUY / SELL / WAIT setup
10. Main technical reasons
11. Risk warning

If something is not visible, write:
"Not visible".

Do not invent prices.
Do not guarantee profit.
"""

        payload = {

            "model": GROQ_MODEL,

            "messages": [

                {
                    "role": "user",

                    "content": [

                        {
                            "type": "text",
                            "text": prompt
                        },

                        {
                            "type": "image_url",

                            "image_url": {

                                "url":
                                    "data:image/jpeg;"
                                    "base64,"
                                    f"{encoded_image}"
                            }
                        }
                    ]
                }
            ],

            "temperature": 0.2,

            "max_completion_tokens": 1200
        }

        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=90
        )

        try:

            data = response.json()

        except Exception:

            return (
                "❌ Groq returned invalid response.\n\n"
                f"HTTP {response.status_code}\n"
                f"{response.text[:1000]}"
            )

        if response.status_code != 200:

            return (
                "❌ Groq error\n\n"
                f"HTTP: {response.status_code}\n"
                f"Model: {GROQ_MODEL}\n\n"
                f"{data}"
            )

        if "choices" not in data:

            return (
                "❌ Groq response error:\n"
                +
                str(data)
            )

        content = (
            data["choices"][0]
            ["message"]
            .get("content", "")
        )

        if not content:

            return (
                "❌ Groq returned empty result."
            )

        return content

    except Exception as e:

        return (
            "❌ Groq Photo Scan error:\n"
            +
            str(e)
        )


# =========================================================
# GEMINI PHOTO SCAN
# =========================================================

def gemini_photo_scan(
    image_bytes
):

    if not GEMINI_API_KEY:

        return (
            "❌ GEMINI_API_KEY is missing.\n\n"
            "Add it in Render → Environment."
        )

    try:

        encoded_image = (
            base64.b64encode(
                image_bytes
            )
            .decode("utf-8")
        )

        url = (
            "https://generativelanguage.googleapis.com/"
            f"v1beta/models/{GEMINI_MODEL}:generateContent"
        )

        headers = {
            "x-goog-api-key":
                GEMINI_API_KEY,

            "Content-Type":
                "application/json",
        }

        prompt = """
You are a technical chart analysis assistant.

Analyze ONLY what is visible in the trading
chart screenshot.

Return:

1. Symbol
2. Timeframe
3. Current trend
4. Support
5. Resistance
6. Candlestick structure
7. EMA if visible
8. RSI if visible
9. BUY / SELL / WAIT setup
10. Main technical reasons
11. Risk warning

If something is not visible, write:
"Not visible".

Do not invent prices.
Do not guarantee profit.
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
                                    encoded_image
                            }
                        }
                    ]
                }
            ]
        }

        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=90
        )

        try:

            data = response.json()

        except Exception:

            return (
                "❌ Gemini returned invalid response.\n\n"
                f"HTTP {response.status_code}\n"
                f"{response.text[:1000]}"
            )

        if response.status_code != 200:

            return (
                "❌ Gemini error\n\n"
                f"HTTP: {response.status_code}\n"
                f"Model: {GEMINI_MODEL}\n\n"
                f"{data}"
            )

        candidates = data.get(
            "candidates"
        )

        if not candidates:

            return (
                "❌ Gemini returned no candidates.\n\n"
                +
                str(data)
            )

        parts = (
            candidates[0]
            .get("content", {})
            .get("parts", [])
        )

        texts = []

        for part in parts:

            if isinstance(part, dict):

                text = part.get("text")

                if text:

                    texts.append(text)

        if not texts:

            return (
                "❌ Gemini returned empty result."
            )

        return "\n".join(texts)

    except Exception as e:

        return (
            "❌ Gemini Photo Scan error:\n"
            +
            str(e)
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

    if not update.message.photo:
        return

    await update.message.reply_text(
        "📸 PHOTO SCAN STARTED...\n\n"
        "🤖 Gemini + Groq are analyzing "
        "the chart..."
    )

    try:

        # Highest quality Telegram photo
        photo = update.message.photo[-1]

        telegram_file = (
            await context.bot.get_file(
                photo.file_id
            )
        )

        image_bytes = (
            await telegram_file
            .download_as_bytearray()
        )

        image_bytes = bytes(
            image_bytes
        )

        if not image_bytes:

            await update.message.reply_text(
                "❌ Could not download image."
            )

            return

        # Run both AI scans
        groq_task = asyncio.to_thread(
            groq_photo_scan,
            image_bytes
        )

        gemini_task = asyncio.to_thread(
            gemini_photo_scan,
            image_bytes
        )

        groq_result, gemini_result = (
            await asyncio.gather(
                groq_task,
                gemini_task
            )
        )

        # Groq
        await update.message.reply_text(
            "👁️ GROQ PHOTO SCAN\n\n"
            +
            groq_result
        )

        # Gemini
        await update.message.reply_text(
            "✨ GEMINI PHOTO SCAN\n\n"
            +
            gemini_result
        )

        await update.message.reply_text(
            "⚠️ NOTE\n\n"
            "Chart analysis is informational "
            "and does not guarantee profit."
        )

    except Exception as e:

        print(
            "Photo handler error:",
            e,
            flush=True
        )

        await update.message.reply_text(
            "❌ PHOTO SCAN ERROR\n\n"
            +
            str(e)
        )


# =========================================================
# ERROR HANDLER
# =========================================================

async def error_handler(
    update,
    context
):

    print(
        "Telegram error:",
        context.error,
        flush=True
    )


# =========================================================
# STARTUP CHECK
# =========================================================

def startup_check():

    print(
        "========================================",
        flush=True
    )

    print(
        "CRYPTO MARKET BOT",
        flush=True
    )

    print(
        "========================================",
        flush=True
    )

    print(
        f"PORT: {PORT}",
        flush=True
    )

    print(
        f"Telegram token: "
        f"{'OK' if TELEGRAM_BOT_TOKEN else 'MISSING'}",
        flush=True
    )

    print(
        f"Twelve Data key: "
        f"{'OK' if TWELVE_DATA_KEY else 'MISSING'}",
        flush=True
    )

    print(
        f"Groq key: "
        f"{'OK' if GROQ_API_KEY else 'MISSING'}",
        flush=True
    )

    print(
        f"Gemini key: "
        f"{'OK' if GEMINI_API_KEY else 'MISSING'}",
        flush=True
    )

    print(
        f"Groq model: {GROQ_MODEL}",
        flush=True
    )

    print(
        f"Gemini model: {GEMINI_MODEL}",
        flush=True
    )

    print(
        "========================================",
        flush=True
    )

    if not TELEGRAM_BOT_TOKEN:

        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN is missing. "
            "Add it in Render Environment."
        )


# =========================================================
# MAIN
# =========================================================

def main():

    try:

        print(
            "Starting Crypto Market Bot...",
            flush=True
        )

        # Health server first
        health_thread = threading.Thread(
            target=start_health_server,
            daemon=True
        )

        health_thread.start()

        # Startup checks
        startup_check()

        # Create Telegram application
        application = (
            Application
            .builder()
            .token(
                TELEGRAM_BOT_TOKEN
            )
            .build()
        )

        # Commands
        application.add_handler(
            CommandHandler(
                "start",
                start_command
            )
        )

        application.add_handler(
            CommandHandler(
                "ping",
                ping_command
            )
        )

        application.add_handler(
            CommandHandler(
                "analyze",
                analyze_command
            )
        )

        application.add_handler(
            CommandHandler(
                "scan",
                scan_command
            )
        )

        application.add_handler(
            CommandHandler(
                "all",
                all_command
            )
        )

        application.add_handler(
            CommandHandler(
                "symbols",
                symbols_command
            )
        )

        application.add_handler(
            CommandHandler(
                "timeframes",
                timeframes_command
            )
        )

        # Photo
        application.add_handler(
            MessageHandler(
                filters.PHOTO,
                photo_handler
            )
        )

        # Errors
        application.add_error_handler(
            error_handler
        )

        print(
            "Telegram bot is starting...",
            flush=True
        )

        # Run forever
        application.run_polling(
            drop_pending_updates=True
        )

    except Exception as e:

        print(
            "========================================",
            flush=True
        )

        print(
            "FATAL ERROR",
            flush=True
        )

        print(
            str(e),
            flush=True
        )

        traceback.print_exc()

        print(
            "========================================",
            flush=True
        )

        raise


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":

    main()
