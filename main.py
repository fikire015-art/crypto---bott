# ============================================================
# ETHIO TRADE BOT
# Telegram + Twelve Data + Groq Vision + Gemini Vision
# Render Ready
# ============================================================

import os
import asyncio
import threading
import traceback
import base64
import time
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


# ============================================================
# CONFIG
# ============================================================

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

TWELVE_DATA_KEY = os.getenv("TWELVE_DATA_KEY", "").strip()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

PORT = int(os.getenv("PORT", "10000"))

# Groq models are tried in this order.
# The code automatically falls back if the first model
# is not available for your API key.
GROQ_VISION_MODELS = [
    os.getenv("GROQ_VISION_MODEL", "").strip(),
    "qwen/qwen3.8-27b",
    "qwen/qwen3.6-27b",
]

# Remove duplicates / empty values
GROQ_VISION_MODELS = list(
    dict.fromkeys(
        model for model in GROQ_VISION_MODELS
        if model
    )
)

# Current stable Gemini Flash model
GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.6-flash"
).strip()


# ============================================================
# SYMBOLS
# ============================================================

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


# ============================================================
# RENDER HEALTH SERVER
# ============================================================

class HealthHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.end_headers()
        self.wfile.write(
            b"ETHIO TRADE BOT is running"
        )

    def log_message(self, format, *args):
        return


def start_health_server():
    try:
        server = ThreadingHTTPServer(
            ("0.0.0.0", PORT),
            HealthHandler
        )

        print(
            f"Health server listening on "
            f"0.0.0.0:{PORT}"
        )

        thread = threading.Thread(
            target=server.serve_forever,
            daemon=True
        )

        thread.start()

    except Exception:
        print("Health server error:")
        traceback.print_exc()


# ============================================================
# STARTUP DIAGNOSTICS
# ============================================================

def print_startup_status():

    print("")
    print("=" * 60)
    print("STARTING ETHIO TRADE BOT")
    print("=" * 60)

    print(
        "TELEGRAM_BOT_TOKEN:",
        "OK" if TELEGRAM_BOT_TOKEN else "MISSING"
    )

    print(
        "TWELVE_DATA_KEY:",
        "OK" if TWELVE_DATA_KEY else "MISSING"
    )

    print(
        "GROQ_API_KEY:",
        "OK" if GROQ_API_KEY else "MISSING"
    )

    print(
        "GEMINI_API_KEY:",
        "OK" if GEMINI_API_KEY else "MISSING"
    )

    print(
        "GROQ MODELS:",
        ", ".join(GROQ_VISION_MODELS)
    )

    print(
        "GEMINI MODEL:",
        GEMINI_MODEL
    )

    print("=" * 60)
    print("")


# ============================================================
# TWELVE DATA
# ============================================================

def get_market_data(symbol, interval="5min", outputsize=100):

    if not TWELVE_DATA_KEY:
        return None

    url = "https://api.twelvedata.com/time_series"

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
            timeout=20
        )

        if response.status_code != 200:
            print(
                "Twelve Data HTTP:",
                response.status_code
            )
            return None

        data = response.json()

        if "values" not in data:
            print(
                "Twelve Data error:",
                data.get("message", data)
            )
            return None

        df = pd.DataFrame(data["values"])

        if df.empty:
            return None

        for column in [
            "open",
            "high",
            "low",
            "close"
        ]:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

        df = df.sort_values("datetime")
        df = df.reset_index(drop=True)

        return df

    except Exception as e:

        print(
            "Twelve Data exception:",
            str(e)
        )

        return None


# ============================================================
# INDICATORS
# ============================================================

def calculate_indicators(df):

    if df is None or len(df) < 30:
        return None

    df = df.copy()

    # EMA
    df["EMA9"] = (
        df["close"]
        .ewm(span=9, adjust=False)
        .mean()
    )

    df["EMA21"] = (
        df["close"]
        .ewm(span=21, adjust=False)
        .mean()
    )

    # RSI
    delta = df["close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = (
        gain
        .rolling(
            window=14,
            min_periods=14
        )
        .mean()
    )

    avg_loss = (
        loss
        .rolling(
            window=14,
            min_periods=14
        )
        .mean()
    )

    avg_loss = avg_loss.replace(0, float("nan"))

    rs = avg_gain / avg_loss

    df["RSI14"] = (
        100 - (100 / (1 + rs))
    )

    # Support / Resistance
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


# ============================================================
# MARKET ANALYSIS
# ============================================================

def analyze_market(symbol, timeframe):

    df = get_market_data(
        symbol,
        timeframe,
        100
    )

    if df is None:
        return (
            f"❌ NO DATA\n"
            f"Symbol: {symbol}\n"
            f"Timeframe: {timeframe}"
        )

    df = calculate_indicators(df)

    if df is None:
        return (
            f"❌ Not enough market data\n"
            f"Symbol: {symbol}\n"
            f"Timeframe: {timeframe}"
        )

    last = df.iloc[-1]

    close = float(last["close"])
    ema9 = float(last["EMA9"])
    ema21 = float(last["EMA21"])
    rsi = float(last["RSI14"]) if pd.notna(last["RSI14"]) else 50
    high20 = float(last["HIGH20"])
    low20 = float(last["LOW20"])

    score = 0

    # EMA
    if ema9 > ema21:
        score += 1
    elif ema9 < ema21:
        score -= 1

    # RSI
    if rsi < 35:
        score += 1
    elif rsi > 65:
        score -= 1

    # Price position
    if close > high20 * 0.995:
        score += 1

    if close < low20 * 1.005:
        score -= 1

    if score >= 2:
        signal = "BUY"

    elif score <= -2:
        signal = "SELL"

    else:
        signal = "WAIT"

    return (
        f"📊 MARKET ANALYSIS\n\n"
        f"Symbol: {symbol}\n"
        f"Timeframe: {timeframe}\n\n"
        f"💰 Price: {close:.5f}\n"
        f"📈 EMA9: {ema9:.5f}\n"
        f"📉 EMA21: {ema21:.5f}\n"
        f"RSI14: {rsi:.2f}\n\n"
        f"🔺 Resistance: {high20:.5f}\n"
        f"🔻 Support: {low20:.5f}\n\n"
        f"🎯 Signal: {signal}\n"
        f"Score: {score}\n\n"
        f"⚠️ Educational analysis only."
    )


# ============================================================
# VALIDATION
# ============================================================

def valid_symbol(symbol):

    return symbol.upper() in [
        s.upper() for s in SYMBOLS
    ]


def valid_timeframe(timeframe):

    return timeframe in TIMEFRAMES


# ============================================================
# /START
# ============================================================

async def start_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    text = (
        "🤖 ETHIO TRADE BOT\n\n"
        "Welcome!\n\n"

        "📊 Commands:\n"
        "/symbols - Show symbols\n"
        "/timeframes - Show timeframes\n"
        "/analyze BTC/USD 5min\n"
        "/scan BTC/USD 15min\n"
        "/all\n"
        "/ping\n\n"

        "📷 Send a chart screenshot "
        "for AI analysis."
    )

    await update.message.reply_text(text)


# ============================================================
# /PING
# ============================================================

async def ping_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    await update.message.reply_text(
        "🟢 ETHIO TRADE BOT is online."
    )


# ============================================================
# /SYMBOLS
# ============================================================

async def symbols_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    text = "📊 AVAILABLE SYMBOLS\n\n"

    for symbol in SYMBOLS:
        text += f"• {symbol}\n"

    await update.message.reply_text(text)


# ============================================================
# /TIMEFRAMES
# ============================================================

async def timeframes_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    text = "⏱ AVAILABLE TIMEFRAMES\n\n"

    for timeframe in TIMEFRAMES:
        text += f"• {timeframe}\n"

    await update.message.reply_text(text)


# ============================================================
# /ANALYZE
# ============================================================

async def analyze_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    if len(context.args) < 2:

        await update.message.reply_text(
            "❌ Usage:\n\n"
            "/analyze BTC/USD 5min"
        )

        return

    symbol = context.args[0].upper()
    timeframe = context.args[1]

    if not valid_symbol(symbol):

        await update.message.reply_text(
            "❌ Invalid symbol.\n\n"
            "Use /symbols"
        )

        return

    if not valid_timeframe(timeframe):

        await update.message.reply_text(
            "❌ Invalid timeframe.\n\n"
            "Use /timeframes"
        )

        return

    await update.message.reply_text(
        "⏳ Analyzing..."
    )

    result = await asyncio.to_thread(
        analyze_market,
        symbol,
        timeframe
    )

    await update.message.reply_text(
        result
    )


# ============================================================
# /SCAN
# ============================================================

async def scan_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    if len(context.args) < 2:

        await update.message.reply_text(
            "❌ Usage:\n\n"
            "/scan BTC/USD 15min"
        )

        return

    symbol = context.args[0].upper()
    timeframe = context.args[1]

    if not valid_symbol(symbol):

        await update.message.reply_text(
            "❌ Invalid symbol."
        )

        return

    if not valid_timeframe(timeframe):

        await update.message.reply_text(
            "❌ Invalid timeframe."
        )

        return

    await update.message.reply_text(
        "🔎 Scanning market..."
    )

    result = await asyncio.to_thread(
        analyze_market,
        symbol,
        timeframe
    )

    await update.message.reply_text(
        result
    )


# ============================================================
# /ALL
# ============================================================

async def all_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    await update.message.reply_text(
        "🚀 Starting full market scan...\n\n"
        "This may take some time because "
        "multiple API requests are required."
    )

    results = []

    for symbol in SYMBOLS:

        for timeframe in TIMEFRAMES:

            result = await asyncio.to_thread(
                analyze_market,
                symbol,
                timeframe
            )

            results.append(
                f"{symbol} | {timeframe}\n"
                f"{result}\n"
            )

            # Avoid hammering the API
            await asyncio.sleep(0.15)

    # Telegram message size protection
    chunk = ""
    chunks = []

    for result in results:

        if len(chunk) + len(result) > 3800:

            chunks.append(chunk)
            chunk = ""

        chunk += result + "\n"

    if chunk:
        chunks.append(chunk)

    for part in chunks:

        await update.message.reply_text(
            part
        )


# ============================================================
# GROQ MODEL DISCOVERY
# ============================================================

def get_groq_available_models():

    if not GROQ_API_KEY:
        return []

    url = (
        "https://api.groq.com/openai/v1/models"
    )

    headers = {
        "Authorization":
            f"Bearer {GROQ_API_KEY}",
        "Content-Type":
            "application/json",
    }

    try:

        response = requests.get(
            url,
            headers=headers,
            timeout=15
        )

        if response.status_code != 200:

            print(
                "Groq models HTTP:",
                response.status_code
            )

            return []

        data = response.json()

        models = []

        for item in data.get("data", []):

            model_id = item.get("id")

            if model_id:
                models.append(model_id)

        return models

    except Exception as e:

        print(
            "Groq model discovery error:",
            str(e)
        )

        return []


# ============================================================
# GROQ PHOTO SCAN
# ============================================================

def groq_photo_scan(image_bytes):

    if not GROQ_API_KEY:

        return (
            "❌ Groq error\n\n"
            "GROQ_API_KEY is missing "
            "in Render Environment Variables."
        )

    image_b64 = base64.b64encode(
        image_bytes
    ).decode("utf-8")

    url = (
        "https://api.groq.com/openai/v1/"
        "chat/completions"
    )

    headers = {
        "Authorization":
            f"Bearer {GROQ_API_KEY}",
        "Content-Type":
            "application/json",
    }

    prompt = """
Analyze this trading chart screenshot.

Return a concise technical analysis.

Identify if visible:
- Symbol
- Timeframe
- Current price
- Trend
- Support
- Resistance
- EMA
- RSI
- Candlestick structure
- BUY / SELL / WAIT setup

Important:
Do not invent values that are not visible.
If an indicator is not visible, say "Not visible".

For the setup:
Use BUY, SELL, or WAIT based only on the visible chart structure.

Include:
1. Symbol
2. Timeframe
3. Current trend
4. Support
5. Resistance
6. Candlestick structure
7. EMA
8. RSI
9. Setup
10. Reasons
11. Risk warning

This is educational technical analysis,
not financial advice.
"""

    # --------------------------------------------------------
    # First use models configured above.
    # Then discover active models if necessary.
    # --------------------------------------------------------

    models_to_try = list(GROQ_VISION_MODELS)

    available_models = get_groq_available_models()

    # Prefer known vision models if account exposes them
    for model in [
        "qwen/qwen3.8-27b",
        "qwen/qwen3.6-27b",
    ]:

        if model in available_models:
            if model not in models_to_try:
                models_to_try.append(model)

    # --------------------------------------------------------
    # Try each model
    # --------------------------------------------------------

    errors = []

    for model in models_to_try:

        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url":
                                    "data:image/jpeg;base64,"
                                    + image_b64
                            },
                        },
                    ],
                }
            ],
            "temperature": 0.2,
            "max_tokens": 1800,
        }

        try:

            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=90
            )

            # Success
            if response.status_code == 200:

                data = response.json()

                choices = data.get(
                    "choices",
                    []
                )

                if choices:

                    message = choices[0].get(
                        "message",
                        {}
                    )

                    content = message.get(
                        "content"
                    )

                    if content:

                        return (
                            "👁️ GROQ PHOTO SCAN\n\n"
                            f"Model: {model}\n\n"
                            f"{content}"
                        )

                errors.append(
                    f"{model}: empty response"
                )

                continue

            # Model not found/access
            if response.status_code == 404:

                errors.append(
                    f"{model}: model not available"
                )

                continue

            # Other API error
            try:
                error_data = response.json()
            except Exception:
                error_data = response.text

            errors.append(
                f"{model}: HTTP "
                f"{response.status_code} - "
                f"{error_data}"
            )

        except Exception as e:

            errors.append(
                f"{model}: {str(e)}"
            )

    # --------------------------------------------------------
    # Nothing worked
    # --------------------------------------------------------

    return (
        "👁️ GROQ PHOTO SCAN\n\n"
        "❌ Groq could not access a Vision model.\n\n"
        "Models tried:\n"
        + "\n".join(
            f"• {x}" for x in models_to_try
        )
        + "\n\n"
        "Details:\n"
        + "\n".join(
            f"• {x}" for x in errors
        )
    )


# ============================================================
# GEMINI PHOTO SCAN
# ============================================================

def gemini_photo_scan(image_bytes):

    if not GEMINI_API_KEY:

        return (
            "✨ GEMINI PHOTO SCAN\n\n"
            "❌ GEMINI_API_KEY is missing "
            "in Render Environment Variables."
        )

    image_b64 = base64.b64encode(
        image_bytes
    ).decode("utf-8")

    url = (
        "https://generativelanguage.googleapis.com/"
        f"v1beta/models/{GEMINI_MODEL}:generateContent"
    )

    params = {
        "key": GEMINI_API_KEY
    }

    headers = {
        "Content-Type":
            "application/json"
    }

    prompt = """
Analyze this trading chart screenshot carefully.

Give a concise technical analysis.

Identify:
1. Symbol
2. Timeframe
3. Current trend
4. Current price if visible
5. Support
6. Resistance
7. Candlestick structure
8. EMA if visible
9. RSI if visible
10. BUY / SELL / WAIT setup
11. Main technical reasons
12. Risk warning

Do NOT invent information.
If something is not visible, say "Not visible".

If the chart says market closed,
mention that.

Use only the visible chart information.

This is educational technical analysis,
not financial advice or a guarantee.
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
                                image_b64
                        }
                    }
                ]
            }
        ]
    }

    try:

        response = requests.post(
            url,
            params=params,
            headers=headers,
            json=payload,
            timeout=90
        )

        if response.status_code != 200:

            try:
                error_data = response.json()
            except Exception:
                error_data = response.text

            return (
                "✨ GEMINI PHOTO SCAN\n\n"
                f"❌ Gemini error\n\n"
                f"HTTP: {response.status_code}\n"
                f"Model: {GEMINI_MODEL}\n\n"
                f"{error_data}"
            )

        data = response.json()

        candidates = data.get(
            "candidates",
            []
        )

        if not candidates:

            return (
                "✨ GEMINI PHOTO SCAN\n\n"
                "❌ Gemini returned no candidates."
            )

        parts = (
            candidates[0]
            .get("content", {})
            .get("parts", [])
        )

        text_parts = []

        for part in parts:

            text = part.get("text")

            if text:
                text_parts.append(text)

        if not text_parts:

            return (
                "✨ GEMINI PHOTO SCAN\n\n"
                "❌ Gemini returned empty text."
            )

        return (
            "✨ GEMINI PHOTO SCAN\n\n"
            + "\n".join(text_parts)
        )

    except Exception as e:

        return (
            "✨ GEMINI PHOTO SCAN\n\n"
            "❌ Gemini exception\n\n"
            f"{str(e)}"
        )


# ============================================================
# PHOTO HANDLER
# ============================================================

async def photo_handler(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    if not update.message:
        return

    if not update.message.photo:
        return

    await update.message.reply_text(
        "📷 Image received.\n\n"
        "⏳ Running Groq + Gemini analysis..."
    )

    try:

        photo = update.message.photo[-1]

        file = await photo.get_file()

        image_bytes = (
            await file.download_as_bytearray()
        )

        image_bytes = bytes(image_bytes)

        # Run both AI scans simultaneously
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

        # Send Groq result
        await update.message.reply_text(
            groq_result
        )

        # Send Gemini result
        await update.message.reply_text(
            gemini_result
        )

        # Risk note
        await update.message.reply_text(
            "⚠️ RISK WARNING\n\n"
            "Chart analysis is educational only. "
            "Markets can move unexpectedly. "
            "Do not risk money you cannot afford to lose."
        )

    except Exception as e:

        print("Photo handler error:")
        traceback.print_exc()

        await update.message.reply_text(
            "❌ Photo processing error:\n\n"
            f"{str(e)}"
        )


# ============================================================
# TEXT HANDLER
# ============================================================

async def text_handler(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    if not update.message:
        return

    text = update.message.text

    if not text:
        return

    await update.message.reply_text(
        "🤖 I received your message.\n\n"
        "Use:\n"
        "/start\n"
        "/symbols\n"
        "/timeframes\n"
        "/analyze BTC/USD 5min\n"
        "/scan BTC/USD 15min\n\n"
        "Or send a chart screenshot 📷"
    )


# ============================================================
# ERROR HANDLER
# ============================================================

async def error_handler(
    update: object,
    context: ContextTypes.DEFAULT_TYPE
):

    print("Telegram error:")

    try:
        print(context.error)
        traceback.print_exception(
            type(context.error),
            context.error,
            context.error.__traceback__
        )

    except Exception:
        print("Unknown Telegram error")


# ============================================================
# MAIN
# ============================================================

def main():

    try:

        print_startup_status()

        # ----------------------------------------------------
        # Telegram token is REQUIRED
        # ----------------------------------------------------

        if not TELEGRAM_BOT_TOKEN:

            raise RuntimeError(
                "TELEGRAM_BOT_TOKEN is missing. "
                "Add it to Render Environment Variables."
            )

        # ----------------------------------------------------
        # Start Render health server
        # ----------------------------------------------------

        start_health_server()

        # ----------------------------------------------------
        # Build Telegram application
        # ----------------------------------------------------

        application = (
            Application
            .builder()
            .token(TELEGRAM_BOT_TOKEN)
            .build()
        )

        # ----------------------------------------------------
        # Commands
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Photo
        # ----------------------------------------------------

        application.add_handler(
            MessageHandler(
                filters.PHOTO,
                photo_handler
            )
        )

        # ----------------------------------------------------
        # Text
        # ----------------------------------------------------

        application.add_handler(
            MessageHandler(
                filters.TEXT
                & ~filters.COMMAND,
                text_handler
            )
        )

        # ----------------------------------------------------
        # Error handler
        # ----------------------------------------------------

        application.add_error_handler(
            error_handler
        )

        print("")
        print("=" * 60)
        print("TELEGRAM BOT IS RUNNING")
        print("=" * 60)
        print("")

        # ----------------------------------------------------
        # Start polling
        # ----------------------------------------------------

        application.run_polling(
            drop_pending_updates=True
        )

    except Exception:

        print("")
        print("=" * 60)
        print("FATAL ERROR")
        print("=" * 60)

        traceback.print_exc()

        print("=" * 60)
        print("")

        raise


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
