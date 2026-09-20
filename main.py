import os
import base64
import asyncio
import threading
import traceback
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
# HELPER
# =========================================================

def safe_text(text, max_length=3900):

    if text is None:
        return "No response."

    text = str(text)

    if len(text) <= max_length:
        return text

    return (
        text[:max_length]
        + "\n\n...[message shortened]"
    )


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

        server = HTTPServer(
            ("0.0.0.0", PORT),
            HealthHandler
        )

        print(
            f"Health server running on port {PORT}"
        )

        server.serve_forever()

    except Exception as e:

        print(
            "Health server error:",
            str(e)
        )

        traceback.print_exc()


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

        response.raise_for_status()

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

        required_columns = [
            "datetime",
            "open",
            "high",
            "low",
            "close",
        ]

        for column in required_columns:

            if column not in df.columns:

                print(
                    f"Missing column: {column}"
                )

                return None

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

        df = df.dropna(
            subset=[
                "open",
                "high",
                "low",
                "close",
            ]
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

    # -----------------------------------------------------
    # EMA 9
    # -----------------------------------------------------

    df["EMA9"] = (
        df["close"]
        .ewm(
            span=9,
            adjust=False
        )
        .mean()
    )

    # -----------------------------------------------------
    # EMA 21
    # -----------------------------------------------------

    df["EMA21"] = (
        df["close"]
        .ewm(
            span=21,
            adjust=False
        )
        .mean()
    )

    # -----------------------------------------------------
    # RSI 14
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # SUPPORT
    # -----------------------------------------------------

    df["LOW20"] = (
        df["low"]
        .rolling(20)
        .min()
    )

    # -----------------------------------------------------
    # RESISTANCE
    # -----------------------------------------------------

    df["HIGH20"] = (
        df["high"]
        .rolling(20)
        .max()
    )

    return df


# =========================================================
# NO DATA RESULT
# =========================================================

def no_data_result(
    symbol,
    timeframe
):

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "signal": "NO DATA",
        "confidence": 0,
        "price": 0.0,
        "rsi": 0.0,
        "ema9": 0.0,
        "ema21": 0.0,
        "support": 0.0,
        "resistance": 0.0,
    }


# =========================================================
# MARKET ANALYSIS
# =========================================================

def analyze_market(
    df,
    symbol,
    timeframe
):

    if df is None:

        return no_data_result(
            symbol,
            timeframe
        )

    if len(df) < 30:

        return no_data_result(
            symbol,
            timeframe
        )

    try:

        df = calculate_indicators(
            df
        )

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

        if pd.isna(
            support_value
        ):

            support = 0.0

        else:

            support = float(
                support_value
            )

        if pd.isna(
            resistance_value
        ):

            resistance = 0.0

        else:

            resistance = float(
                resistance_value
            )

        # -------------------------------------------------
        # SCORE
        # -------------------------------------------------

        score = 0

        # EMA TREND

        if ema9 > ema21:

            score += 1

        elif ema9 < ema21:

            score -= 1

        # RSI

        if rsi >= 55:

            score += 1

        elif rsi <= 45:

            score -= 1

        # PRICE VS EMA9

        if price > ema9:

            score += 1

        elif price < ema9:

            score -= 1

        # -------------------------------------------------
        # SIGNAL
        # -------------------------------------------------

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
            f"Analysis error "
            f"{symbol} {timeframe}: {e}"
        )

        traceback.print_exc()

        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "signal": "ERROR",
            "confidence": 0,
            "price": 0.0,
            "rsi": 0.0,
            "ema9": 0.0,
            "ema21": 0.0,
            "support": 0.0,
            "resistance": 0.0,
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

        "📸 Photo Scan:\n"
        "Send a trading chart screenshot."
    )

    await update.message.reply_text(
        text
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
            ", ".join(
                TIMEFRAMES
            )
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
        f"🪙 Symbols: "
        f"{len(SYMBOLS)}\n"
        f"⏱ Timeframes: "
        f"{len(TIMEFRAMES)}\n"
        f"📊 Total scans: "
        f"{total}\n\n"
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

        if len(chunk) >= 20:

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
            "⚠️ GROQ PHOTO SCAN UNAVAILABLE\n\n"
            "GROQ_API_KEY is missing."
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
Analyze this trading chart screenshot.

Only report information that can
reasonably be seen in the image.

Report:

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

Do not guarantee profit.

If something is not visible,
say "not visible".
"""

        payload = {

            "model":
                "meta-llama/"
                "llama-4-scout-17b-16e-instruct",

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
                                    f"base64,"
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
            timeout=60
        )

        data = response.json()

        if "choices" not in data:

            return (
                "❌ Groq error:\n\n"
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
                "❌ Groq returned an empty response."
            )

        return content

    except Exception as e:

        return (
            "❌ Groq Photo Scan error:\n\n"
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
            "⚠️ GEMINI PHOTO SCAN UNAVAILABLE\n\n"
            "GEMINI_API_KEY is missing."
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
            "v1beta/models/gemini-2.5-flash:"
            "generateContent"
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
6. Candlestick structure
7. EMA if visible
8. RSI if visible
9. BUY / SELL / WAIT setup
10. Main technical reason
11. Risk warning

Only use information visible in the image.

Do not guarantee profit.

If information is not visible,
say "not visible".
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
            params=params,
            json=payload,
            timeout=60
        )

        data = response.json()

        if "candidates" not in data:

            return (
                "❌ Gemini error:\n\n"
                +
                str(data)
            )

        parts = (
            data["candidates"][0]
            ["content"]
            .get("parts", [])
        )

        if not parts:

            return (
                "❌ Gemini returned an empty response."
            )

        return parts[0].get(
            "text",
            "❌ Gemini returned no text."
        )

    except Exception as e:

        return (
            "❌ Gemini Photo Scan error:\n\n"
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

        # -------------------------------------------------
        # HIGHEST QUALITY PHOTO
        # -------------------------------------------------

        photo = update.message.photo[-1]

        telegram_file = (
            await context.bot.get_file(
                photo.file_id
            )
        )

        # -------------------------------------------------
        # DOWNLOAD PHOTO
        # -------------------------------------------------

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

        # -------------------------------------------------
        # GROQ
        # -------------------------------------------------

        groq_result = await asyncio.to_thread(
            groq_photo_scan,
            image_bytes
        )

        # -------------------------------------------------
        # GEMINI
        # -------------------------------------------------

        gemini_result = await asyncio.to_thread(
            gemini_photo_scan,
            image_bytes
        )

        # -------------------------------------------------
        # SEND GROQ
        # -------------------------------------------------

        await update.message.reply_text(
            safe_text(
                "👁️ GROQ PHOTO SCAN\n\n"
                +
                groq_result
            )
        )

        # -------------------------------------------------
        # SEND GEMINI
        # -------------------------------------------------

        await update.message.reply_text(
            safe_text(
                "✨ GEMINI PHOTO SCAN\n\n"
                +
                gemini_result
            )
        )

        # -------------------------------------------------
        # WARNING
        # -------------------------------------------------

        await update.message.reply_text(
            "⚠️ NOTE\n\n"
            "Chart analysis is informational "
            "and does not guarantee profit."
        )

    except Exception as e:

        print(
            "Photo handler error:",
            str(e)
        )

        traceback.print_exc()

        try:

            await update.message.reply_text(
                safe_text(
                    "❌ PHOTO SCAN ERROR\n\n"
                    +
                    str(e)
                )
            )

        except Exception:

            pass


# =========================================================
# ERROR HANDLER
# =========================================================

async def error_handler(
    update,
    context
):

    print(
        "Telegram error:",
        str(context.error)
    )

    if context.error:

        traceback.print_exception(
            type(context.error),
            context.error,
            context.error.__traceback__
        )


# =========================================================
# STARTUP CHECK
# =========================================================

def startup_check():

    print("================================")
    print("ENVIRONMENT CHECK")
    print("================================")

    print(
        "TELEGRAM_BOT_TOKEN:",
        "OK"
        if TELEGRAM_BOT_TOKEN
        else "MISSING"
    )

    print(
        "TWELVE_DATA_KEY:",
        "OK"
        if TWELVE_DATA_KEY
        else "MISSING"
    )

    print(
        "GROQ_API_KEY:",
        "OK"
        if GROQ_API_KEY
        else "MISSING"
    )

    print(
        "GEMINI_API_KEY:",
        "OK"
        if GEMINI_API_KEY
        else "MISSING"
    )

    print(
        "PORT:",
        PORT
    )

    print("================================")

    # Telegram token is required

    if not TELEGRAM_BOT_TOKEN:

        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN is missing. "
            "Add TELEGRAM_BOT_TOKEN in "
            "Render Environment Variables."
        )


# =========================================================
# MAIN
# =========================================================

def main():

    print("================================")
    print("Starting Crypto Market Bot...")
    print("================================")

    # -----------------------------------------------------
    # STARTUP CHECK
    # -----------------------------------------------------

    startup_check()

    # -----------------------------------------------------
    # START RENDER HEALTH SERVER
    # -----------------------------------------------------

    health_thread = threading.Thread(
        target=start_health_server,
        daemon=True
    )

    health_thread.start()

    print(
        "Render health server started."
    )

    # -----------------------------------------------------
    # CREATE TELEGRAM APPLICATION
    # -----------------------------------------------------

    try:

        application = (
            Application
            .builder()
            .token(
                TELEGRAM_BOT_TOKEN
            )
            .build()
        )

        print(
            "Telegram application created."
        )

    except Exception as e:

        print(
            "Telegram application creation failed:"
        )

        print(
            str(e)
        )

        traceback.print_exc()

        raise

    # -----------------------------------------------------
    # COMMANDS
    # -----------------------------------------------------

    application.add_handler(
        CommandHandler(
            "start",
            start_command
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

    # -----------------------------------------------------
    # PHOTO
    # -----------------------------------------------------

    application.add_handler(
        MessageHandler(
            filters.PHOTO,
            photo_handler
        )
    )

    # -----------------------------------------------------
    # ERROR HANDLER
    # -----------------------------------------------------

    application.add_error_handler(
        error_handler
    )

    print("================================")
    print("Telegram bot is ready.")
    print("Starting polling...")
    print("================================")

    # -----------------------------------------------------
    # RUN BOT
    # -----------------------------------------------------

    try:

        application.run_polling(
            drop_pending_updates=True
        )

    except Exception as e:

        print("================================")
        print("BOT CRASHED")
        print("================================")

        print(
            str(e)
        )

        traceback.print_exc()

        raise


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print(
            "Bot stopped."
        )

    except Exception as e:

        print("================================")
        print("FATAL ERROR")
        print("================================")

        print(
            str(e)
        )

        traceback.print_exc()

        raise
