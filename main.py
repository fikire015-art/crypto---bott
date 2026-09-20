import os
import json
import base64
import asyncio
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import requests
import pandas as pd

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
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

GROQ_MODEL = os.getenv(
    "GROQ_VISION_MODEL",
    "qwen/qwen3.8-27b"
)

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.6-flash"
)

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
    "XAU/USD",
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
# PIP SIZE
# =========================================================

PIP_SIZE = {
    "XAU/USD": 0.01,

    "BTC/USD": 0.01,
    "ETH/USD": 0.01,
    "BNB/USD": 0.01,
    "SOL/USD": 0.01,
    "AVAX/USD": 0.01,
    "LINK/USD": 0.01,

    "XRP/USD": 0.0001,
    "DOGE/USD": 0.0001,
    "ADA/USD": 0.0001,
    "TRX/USD": 0.0001,
}

MIN_PIPS = 101

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
            b"ETHIO TRADE BOT OK"
        )

    def log_message(self, format, *args):
        return


def start_health_server():

    server = ThreadingHTTPServer(
        ("0.0.0.0", PORT),
        HealthHandler
    )

    print(
        f"Health server running on port {PORT}"
    )

    server.serve_forever()


# =========================================================
# NORMALIZE SYMBOL
# =========================================================

def normalize_symbol(symbol):

    symbol = symbol.strip().upper()

    aliases = {
        "BTCUSD": "BTC/USD",
        "ETHUSD": "ETH/USD",
        "BNBUSD": "BNB/USD",
        "SOLUSD": "SOL/USD",
        "XRPUSD": "XRP/USD",
        "DOGEUSD": "DOGE/USD",
        "ADAUSD": "ADA/USD",
        "AVAXUSD": "AVAX/USD",
        "LINKUSD": "LINK/USD",
        "TRXUSD": "TRX/USD",
        "XAUUSD": "XAU/USD",
    }

    return aliases.get(symbol, symbol)


# =========================================================
# NORMALIZE TIMEFRAME
# =========================================================

def normalize_timeframe(timeframe):

    timeframe = timeframe.strip().lower()

    aliases = {
        "1m": "1min",
        "5m": "5min",
        "15m": "15min",
        "30m": "30min",
        "45m": "45min",

        "1h": "1h",
        "2h": "2h",
        "4h": "4h",
        "8h": "8h",

        "d": "1day",
        "1d": "1day",

        "w": "1week",
        "1w": "1week",

        "mo": "1month",
        "1mo": "1month",
    }

    return aliases.get(
        timeframe,
        timeframe
    )


# =========================================================
# VALIDATION
# =========================================================

def valid_symbol(symbol):
    return symbol in SYMBOLS


def valid_timeframe(timeframe):
    return timeframe in TIMEFRAMES


# =========================================================
# TWELVE DATA
# =========================================================

def get_market_data(
    symbol,
    interval="30min",
    outputsize=100
):

    if not TWELVE_DATA_KEY:
        raise Exception(
            "TWELVE_DATA_KEY is missing"
        )

    url = (
        "https://api.twelvedata.com/time_series"
    )

    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": outputsize,
        "apikey": TWELVE_DATA_KEY,
        "format": "JSON",
    }

    response = requests.get(
        url,
        params=params,
        timeout=30
    )

    response.raise_for_status()

    data = response.json()

    if "values" not in data:

        raise Exception(
            data.get(
                "message",
                "No market data returned"
            )
        )

    df = pd.DataFrame(
        data["values"]
    )

    if df.empty:
        raise Exception(
            "Empty market data"
        )

    df["datetime"] = pd.to_datetime(
        df["datetime"]
    )

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

    df = df.sort_values(
        "datetime"
    ).reset_index(drop=True)

    # =====================================================
    # EMA
    # =====================================================

    df["EMA9"] = df["close"].ewm(
        span=9,
        adjust=False
    ).mean()

    df["EMA21"] = df["close"].ewm(
        span=21,
        adjust=False
    ).mean()

    # =====================================================
    # RSI
    # =====================================================

    delta = df["close"].diff()

    gain = delta.clip(
        lower=0
    )

    loss = -delta.clip(
        upper=0
    )

    avg_gain = gain.rolling(
        14
    ).mean()

    avg_loss = loss.rolling(
        14
    ).mean()

    rs = avg_gain / avg_loss.replace(
        0,
        pd.NA
    )

    df["RSI14"] = (
        100 -
        (
            100 /
            (1 + rs)
        )
    )

    # =====================================================
    # SUPPORT / RESISTANCE
    # =====================================================

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

    df = df.dropna(
        subset=[
            "EMA9",
            "EMA21",
            "RSI14",
            "HIGH20",
            "LOW20"
        ]
    )

    if df.empty:
        raise Exception(
            "Not enough market data"
        )

    return df


# =========================================================
# TECHNICAL ANALYSIS
# =========================================================

def technical_analysis(
    symbol,
    timeframe
):

    df = get_market_data(
        symbol,
        timeframe,
        100
    )

    last = df.iloc[-1]

    price = float(last["close"])
    ema9 = float(last["EMA9"])
    ema21 = float(last["EMA21"])
    rsi = float(last["RSI14"])

    support = float(last["LOW20"])
    resistance = float(last["HIGH20"])

    # =====================================================
    # SIGNAL
    # =====================================================

    if (
        ema9 > ema21
        and 52 <= rsi <= 70
    ):

        signal = "BUY"

        confidence = 70

        if ema9 > ema21:
            confidence += 5

        if 55 <= rsi <= 65:
            confidence += 5

    elif (
        ema9 < ema21
        and 30 <= rsi <= 48
    ):

        signal = "SELL"

        confidence = 70

        if ema9 < ema21:
            confidence += 5

        if 35 <= rsi <= 45:
            confidence += 5

    else:

        signal = "WAIT"

        confidence = 55

    confidence = min(
        confidence,
        95
    )

    # =====================================================
    # PIP DISTANCE
    # =====================================================

    pip = PIP_SIZE.get(
        symbol,
        0.01
    )

    min_distance = (
        MIN_PIPS * pip
    )

    # =====================================================
    # LEVELS
    # =====================================================

    if signal == "BUY":

        entry = price

        tp = max(
            price + min_distance,
            resistance
        )

        sl = min(
            price - min_distance,
            support
        )

        buy_limit = price - min_distance

        sell_limit = max(
            price + min_distance,
            resistance
        )

    elif signal == "SELL":

        entry = price

        tp = min(
            price - min_distance,
            support
        )

        sl = max(
            price + min_distance,
            resistance
        )

        sell_limit = price + min_distance

        buy_limit = min(
            price - min_distance,
            support
        )

    else:

        entry = price

        tp = price + min_distance

        sl = price - min_distance

        buy_limit = price - min_distance

        sell_limit = price + min_distance

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "signal": signal,
        "entry": entry,
        "tp": tp,
        "sl": sl,
        "buy_limit": buy_limit,
        "sell_limit": sell_limit,
        "confidence": confidence,
        "price": price,
        "ema9": ema9,
        "ema21": ema21,
        "rsi": rsi,
        "support": support,
        "resistance": resistance,
    }


# =========================================================
# FORMAT NUMBER
# =========================================================

def format_price(value):

    try:

        value = float(value)

        if abs(value) >= 1000:
            return f"{value:.2f}"

        if abs(value) >= 1:
            return f"{value:.4f}".rstrip("0").rstrip(".")

        return f"{value:.6f}".rstrip("0").rstrip(".")

    except Exception:

        return str(value)


# =========================================================
# FORMAT TECHNICAL RESULT
# =========================================================

def format_technical_result(result):

    signal = result["signal"]

    if signal == "BUY":
        emoji = "🟢"

    elif signal == "SELL":
        emoji = "🔴"

    else:
        emoji = "⚪"

    return (
        f"📊 {result['symbol']} | "
        f"{result['timeframe']}\n\n"

        f"🤖 BOT: {emoji} {signal}\n"
        f"🔥 Confidence: "
        f"{result['confidence']}%\n\n"

        f"🟢 Entry: "
        f"{format_price(result['entry'])}\n"

        f"🎯 TP: "
        f"{format_price(result['tp'])}\n"

        f"🛑 SL: "
        f"{format_price(result['sl'])}\n\n"

        f"🟢 BUY LIMIT: "
        f"{format_price(result['buy_limit'])}\n"

        f"🔴 SELL LIMIT: "
        f"{format_price(result['sell_limit'])}"
    )


# =========================================================
# GROQ MODEL DISCOVERY
# =========================================================

def get_groq_models():

    if not GROQ_API_KEY:
        return []

    url = (
        "https://api.groq.com/openai/v1/models"
    )

    headers = {
        "Authorization":
        f"Bearer {GROQ_API_KEY}"
    }

    try:

        response = requests.get(
            url,
            headers=headers,
            timeout=20
        )

        response.raise_for_status()

        data = response.json()

        return [
            item.get("id")
            for item in data.get("data", [])
            if item.get("id")
        ]

    except Exception as e:

        print(
            "Groq model discovery error:",
            e
        )

        return []


# =========================================================
# IMAGE BASE64
# =========================================================

def image_to_base64(
    image_path
):

    with open(
        image_path,
        "rb"
    ) as file:

        return base64.b64encode(
            file.read()
        ).decode("utf-8")


# =========================================================
# CLEAN JSON
# =========================================================

def clean_json(text):

    text = text.strip()

    if text.startswith("```"):

        text = text.replace(
            "```json",
            ""
        )

        text = text.replace(
            "```",
            ""
        )

        text = text.strip()

    start = text.find("{")
    end = text.rfind("}")

    if start >= 0 and end >= 0:

        text = text[
            start:end + 1
        ]

    return text


# =========================================================
# TRADING PROMPT
# =========================================================

def trading_prompt():

    return """
Analyze the trading chart carefully.

Return ONLY valid JSON.
NO markdown.
NO explanation.
NO extra text.

Use exactly these fields:

{
  "symbol": "XAUUSD",
  "timeframe": "M30",
  "signal": "BUY",
  "entry": "4375",
  "tp": "4395",
  "sl": "4365",
  "buy_limit": "4368",
  "sell_limit": "4400",
  "confidence": "82%",
  "reason": "short reason"
}

Rules:

signal must be BUY, SELL, or WAIT.

If BUY:
- entry should be realistic
- tp must be above entry
- sl must be below entry
- buy_limit must be below entry
- sell_limit must be above entry

If SELL:
- entry should be realistic
- tp must be below entry
- sl must be above entry
- sell_limit must be above entry
- buy_limit must be below entry

If WAIT:
- still provide reasonable levels
- confidence should represent setup strength

Confidence must be between 0% and 100%.

Do not guarantee profit.
Do not invent certainty.

Keep reason under 8 words.
"""


# =========================================================
# GROQ PHOTO ANALYSIS
# =========================================================

def groq_photo_analysis(
    image_path
):

    if not GROQ_API_KEY:

        return {
            "error":
            "GROQ_API_KEY missing"
        }

    image_b64 = image_to_base64(
        image_path
    )

    url = (
        "https://api.groq.com/openai/"
        "v1/chat/completions"
    )

    headers = {
        "Authorization":
        f"Bearer {GROQ_API_KEY}",

        "Content-Type":
        "application/json",
    }

    payload = {

        "model": GROQ_MODEL,

        "messages": [

            {
                "role": "system",
                "content":
                trading_prompt()
            },

            {
                "role": "user",

                "content": [

                    {
                        "type": "text",
                        "text":
                        "Analyze this chart."
                    },

                    {
                        "type": "image_url",

                        "image_url": {
                            "url":
                            (
                                "data:image/jpeg;base64,"
                                f"{image_b64}"
                            )
                        }
                    }

                ]
            }

        ],

        "temperature": 0.1,

        "max_tokens": 300
    }

    response = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=120
    )

    response.raise_for_status()

    data = response.json()

    text = (
        data["choices"][0]
        ["message"]["content"]
    )

    try:

        return json.loads(
            clean_json(text)
        )

    except Exception:

        return {
            "error":
            "Groq returned invalid JSON",
            "raw": text
        }


# =========================================================
# GEMINI PHOTO ANALYSIS
# =========================================================

def gemini_photo_analysis(
    image_path
):

    if not GEMINI_API_KEY:

        return {
            "error":
            "GEMINI_API_KEY missing"
        }

    image_b64 = image_to_base64(
        image_path
    )

    url = (
        "https://generativelanguage.googleapis.com/"
        f"v1beta/models/{GEMINI_MODEL}:generateContent"
        f"?key={GEMINI_API_KEY}"
    )

    payload = {

        "contents": [

            {
                "parts": [

                    {
                        "text":
                        trading_prompt()
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

        ],

        "generationConfig": {

            "temperature": 0.1,

            "maxOutputTokens": 300,

            "responseMimeType":
            "application/json"
        }
    }

    response = requests.post(
        url,
        json=payload,
        timeout=120
    )

    response.raise_for_status()

    data = response.json()

    text = (
        data["candidates"][0]
        ["content"]["parts"][0]["text"]
    )

    try:

        return json.loads(
            clean_json(text)
        )

    except Exception:

        return {
            "error":
            "Gemini returned invalid JSON",
            "raw": text
        }


# =========================================================
# GET VALUE
# =========================================================

def get_value(
    data,
    key
):

    if not isinstance(
        data,
        dict
    ):
        return "-"

    value = data.get(
        key,
        "-"
    )

    if value is None:
        return "-"

    return str(value)


# =========================================================
# FINAL SIGNAL
# =========================================================

def final_signal(
    bot,
    groq,
    gemini
):

    signals = []

    for data in [
        bot,
        groq,
        gemini
    ]:

        if not isinstance(
            data,
            dict
        ):
            continue

        signal = data.get(
            "signal"
        )

        if signal in [
            "BUY",
            "SELL",
            "WAIT"
        ]:

            signals.append(
                signal
            )

    if not signals:
        return "WAIT"

    buys = signals.count(
        "BUY"
    )

    sells = signals.count(
        "SELL"
    )

    if buys >= 2:
        return "BUY"

    if sells >= 2:
        return "SELL"

    return "WAIT"


# =========================================================
# FINAL CONFIDENCE
# =========================================================

def extract_confidence(data):

    try:

        value = str(
            data.get(
                "confidence",
                "0"
            )
        )

        value = (
            value
            .replace("%", "")
            .strip()
        )

        return float(value)

    except Exception:

        return 0


def calculate_confidence(
    bot,
    groq,
    gemini
):

    values = []

    for data in [
        bot,
        groq,
        gemini
    ]:

        if isinstance(
            data,
            dict
        ):

            values.append(
                extract_confidence(
                    data
                )
            )

    if not values:
        return 0

    return round(
        sum(values) / len(values)
    )


# =========================================================
# COMBINED PHOTO OUTPUT
# =========================================================

def combined_output(
    bot,
    groq,
    gemini
):

    final = final_signal(
        bot,
        groq,
        gemini
    )

    confidence = calculate_confidence(
        bot,
        groq,
        gemini
    )

    # =====================================================
    # SELECT LEVELS
    # =====================================================

    source = None

    if final == "BUY":

        for data in [
            bot,
            groq,
            gemini
        ]:

            if (
                isinstance(data, dict)
                and data.get("signal")
                == "BUY"
            ):

                source = data
                break

    elif final == "SELL":

        for data in [
            bot,
            groq,
            gemini
        ]:

            if (
                isinstance(data, dict)
                and data.get("signal")
                == "SELL"
            ):

                source = data
                break

    if source is None:

        for data in [
            bot,
            groq,
            gemini
        ]:

            if isinstance(
                data,
                dict
            ):

                source = data
                break

    if source is None:
        source = {}

    symbol = (
        get_value(
            bot,
            "symbol"
        )
    )

    timeframe = (
        get_value(
            bot,
            "timeframe"
        )
    )

    return (
        f"📊 {symbol} | "
        f"{timeframe}\n\n"

        f"🤖 BOT: "
        f"{get_value(bot, 'signal')}\n"

        f"👁️ GROQ: "
        f"{get_value(groq, 'signal')}\n"

        f"✨ GEMINI: "
        f"{get_value(gemini, 'signal')}\n\n"

        f"🎯 FINAL: {final}\n"
        f"🔥 Confidence: "
        f"{confidence}%\n\n"

        f"🟢 Entry: "
        f"{get_value(source, 'entry')}\n"

        f"🎯 TP: "
        f"{get_value(source, 'tp')}\n"

        f"🛑 SL: "
        f"{get_value(source, 'sl')}\n\n"

        f"🟢 BUY LIMIT: "
        f"{get_value(source, 'buy_limit')}\n"

        f"🔴 SELL LIMIT: "
        f"{get_value(source, 'sell_limit')}"
    )


# =========================================================
# /START
# =========================================================

async def start_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    text = (
        "🤖 ETHIO TRADE BOT\n\n"

        "Use:\n"
        "/analyze BTC/USD 5min\n"
        "/analyze XAU/USD 30min\n\n"

        "/scan BTC/USD\n\n"

        "📸 Send a chart screenshot "
        "for BOT + GROQ + GEMINI analysis."
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

    await update.message.reply_text(
        "🟢 BOT ONLINE"
    )


# =========================================================
# /SYMBOLS
# =========================================================

async def symbols_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    text = (
        "📊 Available Symbols:\n\n"
        + "\n".join(
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

    text = (
        "⏱ Available Timeframes:\n\n"
        + "\n".join(
            f"• {tf}"
            for tf in TIMEFRAMES
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
            "❌ Use:\n"
            "/analyze BTC/USD 5min"
        )

        return

    symbol = normalize_symbol(
        context.args[0]
    )

    timeframe = normalize_timeframe(
        context.args[1]
    )

    if not valid_symbol(symbol):

        await update.message.reply_text(
            "❌ Invalid symbol.\n\n"
            "Use /symbols"
        )

        return

    if not valid_timeframe(
        timeframe
    ):

        await update.message.reply_text(
            "❌ Invalid timeframe.\n\n"
            "Use /timeframes"
        )

        return

    message = await update.message.reply_text(
        f"⏳ Analyzing {symbol} | "
        f"{timeframe}..."
    )

    try:

        result = await asyncio.to_thread(
            technical_analysis,
            symbol,
            timeframe
        )

        output = format_technical_result(
            result
        )

        await message.edit_text(
            output
        )

    except Exception as e:

        await message.edit_text(
            "❌ Analysis error:\n"
            f"{str(e)}"
        )


# =========================================================
# /SCAN SYMBOL
# =========================================================

async def scan_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    if len(context.args) < 1:

        await update.message.reply_text(
            "❌ Use:\n"
            "/scan BTC/USD"
        )

        return

    symbol = normalize_symbol(
        context.args[0]
    )

    if not valid_symbol(symbol):

        await update.message.reply_text(
            "❌ Invalid symbol.\n\n"
            "Use /symbols"
        )

        return

    status = await update.message.reply_text(
        f"⏳ Scanning {symbol}..."
    )

    results = []

    for timeframe in TIMEFRAMES:

        try:

            result = await asyncio.to_thread(
                technical_analysis,
                symbol,
                timeframe
            )

            signal = result["signal"]

            if signal == "BUY":
                emoji = "🟢"

            elif signal == "SELL":
                emoji = "🔴"

            else:
                emoji = "⚪"

            results.append(
                f"{emoji} {timeframe} | "
                f"{signal} | "
                f"{result['confidence']}%"
            )

        except Exception as e:

            results.append(
                f"⚠️ {timeframe} | ERROR"
            )

    output = (
        f"📊 {symbol} SCAN\n\n"
        + "\n".join(results)
    )

    await status.edit_text(
        output
    )


# =========================================================
# PHOTO HANDLER
# =========================================================

async def photo_handler(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    photo = update.message.photo[-1]

    file = await photo.get_file()

    image_path = "/tmp/chart.jpg"

    await file.download_to_drive(
        image_path
    )

    status = await update.message.reply_text(
        "🔎 Analyzing chart...\n"
        "🤖 BOT + GROQ + GEMINI"
    )

    try:

        # =================================================
        # AI IMAGE ANALYSIS
        # =================================================

        groq_task = asyncio.to_thread(
            groq_photo_analysis,
            image_path
        )

        gemini_task = asyncio.to_thread(
            gemini_photo_analysis,
            image_path
        )

        groq, gemini = await asyncio.gather(
            groq_task,
            gemini_task
        )

        # =================================================
        # BOT ANALYSIS
        # =================================================

        # Default screenshot technical analysis.
        # AI models identify the chart symbol/timeframe.

        bot = None

        detected_symbol = None
        detected_timeframe = None

        for data in [
            groq,
            gemini
        ]:

            if isinstance(
                data,
                dict
            ):

                detected_symbol = data.get(
                    "symbol"
                )

                detected_timeframe = data.get(
                    "timeframe"
                )

                if (
                    detected_symbol
                    and detected_timeframe
                ):
                    break

        if detected_symbol:

            detected_symbol = normalize_symbol(
                detected_symbol
            )

        timeframe_map = {
            "M1": "1min",
            "M5": "5min",
            "M15": "15min",
            "M30": "30min",
            "M45": "45min",
            "H1": "1h",
            "H2": "2h",
            "H4": "4h",
            "H8": "8h",
            "D1": "1day",
            "W1": "1week",
            "MN1": "1month",
            "1M": "1month",
        }

        if detected_timeframe:

            detected_timeframe = (
                detected_timeframe
                .upper()
                .strip()
            )

            detected_timeframe = (
                timeframe_map.get(
                    detected_timeframe,
                    detected_timeframe.lower()
                )
            )

        # =================================================
        # RUN BOT ONLY FOR DETECTED SYMBOL/TIMEFRAME
        # =================================================

        if (
            detected_symbol
            in SYMBOLS
            and
            detected_timeframe
            in TIMEFRAMES
        ):

            try:

                bot = await asyncio.to_thread(
                    technical_analysis,
                    detected_symbol,
                    detected_timeframe
                )

            except Exception as e:

                bot = {
                    "symbol":
                    detected_symbol,

                    "timeframe":
                    detected_timeframe,

                    "signal":
                    "WAIT",

                    "confidence":
                    "0",

                    "error":
                    str(e)
                }

        else:

            bot = {
                "symbol":
                get_value(
                    groq,
                    "symbol"
                ),

                "timeframe":
                get_value(
                    groq,
                    "timeframe"
                ),

                "signal":
                "WAIT",

                "confidence":
                "0"
            }

        # =================================================
        # OUTPUT
        # =================================================

        output = combined_output(
            bot,
            groq,
            gemini
        )

        await status.edit_text(
            output
        )

    except Exception as e:

        await status.edit_text(
            "❌ Photo analysis error:\n"
            f"{str(e)}"
        )

    finally:

        try:

            if os.path.exists(
                image_path
            ):

                os.remove(
                    image_path
                )

        except Exception:
            pass


# =========================================================
# TEXT HANDLER
# =========================================================

async def text_handler(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    text = update.message.text.strip()

    if text.startswith("/"):
        return

    await update.message.reply_text(
        "🤖 Use:\n\n"
        "/analyze BTC/USD 5min\n"
        "/scan BTC/USD\n"
        "/symbols\n"
        "/timeframes\n\n"
        "📸 Or send a chart screenshot."
    )


# =========================================================
# ERROR HANDLER
# =========================================================

async def error_handler(
    update: object,
    context: ContextTypes.DEFAULT_TYPE
):

    print(
        "Telegram error:",
        context.error
    )


# =========================================================
# MAIN
# =========================================================

def main():

    if not TELEGRAM_BOT_TOKEN:

        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN is missing"
        )

    # =====================================================
    # START RENDER HEALTH SERVER
    # =====================================================

    health_thread = threading.Thread(
        target=start_health_server,
        daemon=True
    )

    health_thread.start()

    # =====================================================
    # TELEGRAM APP
    # =====================================================

    application = (
        Application.builder()
        .token(
            TELEGRAM_BOT_TOKEN
        )
        .build()
    )

    # =====================================================
    # COMMANDS
    # =====================================================

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

    # =====================================================
    # PHOTO
    # =====================================================

    application.add_handler(
        MessageHandler(
            filters.PHOTO,
            photo_handler
        )
    )

    # =====================================================
    # TEXT
    # =====================================================

    application.add_handler(
        MessageHandler(
            filters.TEXT
            & ~filters.COMMAND,
            text_handler
        )
    )

    # =====================================================
    # ERROR
    # =====================================================

    application.add_error_handler(
        error_handler
    )

    print(
        "ETHIO TRADE BOT STARTED"
    )

    application.run_polling(
        drop_pending_updates=True
    )


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":
    main()
