import os
import json
import base64
import asyncio
import threading
import time
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

GROQ_MODEL = os.getenv("GROQ_VISION_MODEL", "qwen/qwen3.8-27b")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
GEMINI_FALLBACK_MODELS = [
    GEMINI_MODEL,
    "gemini-3.7-flash",
]

# Minimum confidence for a technical BUY/SELL signal.
MIN_CONFIDENCE = 80

# Internal target-distance filter. This is intentionally NOT shown to users.
INTERNAL_MIN_TARGET_PIPS = 101

# A BUY/SELL is published only when at least 2 of the 3
# analysis sources agree and the averaged confidence reaches this level.
# This is a filter, not a guarantee of profit.

# =========================================================
# SYMBOLS
# =========================================================
# "All symbols" here means all symbols configured for this bot.
# Twelve Data/broker availability still depends on the API/account.

SYMBOLS = [
    # Forex
    "EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF", "AUD/USD",
    "NZD/USD", "USD/CAD", "EUR/GBP", "EUR/JPY", "GBP/JPY",
    "AUD/JPY", "NZD/JPY", "CAD/JPY", "CHF/JPY", "EUR/CHF",
    "GBP/CHF", "AUD/CAD", "AUD/CHF", "CAD/CHF", "NZD/CAD",
    "NZD/CHF",

    # Metals
    "XAU/USD", "XAG/USD",

    # Crypto
    "BTC/USD", "ETH/USD", "BNB/USD", "SOL/USD", "XRP/USD",
    "DOGE/USD", "ADA/USD", "AVAX/USD", "LINK/USD", "TRX/USD",
    "DOT/USD", "LTC/USD", "BCH/USD", "ATOM/USD", "ETC/USD",
    "FIL/USD", "NEAR/USD", "APT/USD", "ARB/USD", "OP/USD",
]

# =========================================================
# TIMEFRAMES
# =========================================================

TIMEFRAMES = [
    "1min", "5min", "15min", "30min", "45min",
    "1h", "2h", "4h", "8h",
    "1day", "1week", "1month",
]

# =========================================================
# SYMBOL ALIASES
# =========================================================

ALIASES = {
    "BTCUSD": "BTC/USD", "BTC/USDT": "BTC/USD",
    "ETHUSD": "ETH/USD", "ETH/USDT": "ETH/USD",
    "BNBUSD": "BNB/USD", "BNB/USDT": "BNB/USD",
    "SOLUSD": "SOL/USD", "SOL/USDT": "SOL/USD",
    "XRPUSD": "XRP/USD", "XRP/USDT": "XRP/USD",
    "DOGEUSD": "DOGE/USD", "DOGE/USDT": "DOGE/USD",
    "ADAUSD": "ADA/USD", "ADA/USDT": "ADA/USD",
    "AVAXUSD": "AVAX/USD", "AVAX/USDT": "AVAX/USD",
    "LINKUSD": "LINK/USD", "LINK/USDT": "LINK/USD",
    "TRXUSD": "TRX/USD", "TRX/USDT": "TRX/USD",
    "DOTUSD": "DOT/USD", "DOT/USDT": "DOT/USD",
    "LTCUSD": "LTC/USD", "LTC/USDT": "LTC/USD",
    "BCHUSD": "BCH/USD", "BCH/USDT": "BCH/USD",
    "ATOMUSD": "ATOM/USD", "ATOM/USDT": "ATOM/USD",
    "ETCUSD": "ETC/USD", "ETC/USDT": "ETC/USD",
    "FILUSD": "FIL/USD", "FIL/USDT": "FIL/USD",
    "NEARUSD": "NEAR/USD", "NEAR/USDT": "NEAR/USD",
    "APTUSD": "APT/USD", "APT/USDT": "APT/USD",
    "ARBUSD": "ARB/USD", "ARB/USDT": "ARB/USD",
    "OPUSD": "OP/USD", "OP/USDT": "OP/USD",
    "XAUUSD": "XAU/USD", "GOLD": "XAU/USD",
    "XAGUSD": "XAG/USD", "SILVER": "XAG/USD",
}

# =========================================================
# NORMALIZE
# =========================================================

def normalize_symbol(symbol):
    symbol = symbol.strip().upper()
    return ALIASES.get(symbol, symbol)


def normalize_timeframe(timeframe):
    timeframe = timeframe.strip().lower()
    aliases = {
        "1m": "1min", "5m": "5min", "15m": "15min",
        "30m": "30min", "45m": "45min",
        "h1": "1h", "h2": "2h", "h4": "4h", "h8": "8h",
        "d": "1day", "1d": "1day",
        "w": "1week", "1w": "1week",
        "mo": "1month", "1mo": "1month",
        "mn1": "1month",
    }
    return aliases.get(timeframe, timeframe)


def valid_symbol(symbol):
    return symbol in SYMBOLS


def valid_timeframe(timeframe):
    return timeframe in TIMEFRAMES


# =========================================================
# PIP SIZE
# =========================================================

def get_pip_size(symbol):
    """Return the pip size used by this bot."""
    symbol = normalize_symbol(symbol)

    # Gold/silver: 0.01 price movement = 1 pip
    if symbol in ("XAU/USD", "XAG/USD"):
        return 0.01

    # JPY forex pairs: 0.01 price movement = 1 pip
    if "JPY" in symbol:
        return 0.01

    # Low-priced crypto tokens
    if symbol in {
        "XRP/USD", "DOGE/USD", "ADA/USD", "TRX/USD",
        "ARB/USD", "OP/USD", "SHIB/USD"
    }:
        return 0.0001

    # Other crypto
    if symbol in {
        "BTC/USD", "ETH/USD", "BNB/USD", "SOL/USD",
        "AVAX/USD", "LINK/USD", "DOT/USD", "LTC/USD",
        "BCH/USD", "ATOM/USD", "ETC/USD", "FIL/USD",
        "NEAR/USD", "APT/USD"
    }:
        return 0.01

    # Normal forex
    if "/" in symbol:
        return 0.0001

    return 0.01


# =========================================================
# RENDER HEALTH SERVER
# =========================================================

class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(b"ETHIO TRADE BOT OK")

    def log_message(self, format, *args):
        return


def start_health_server():
    server = ThreadingHTTPServer(("0.0.0.0", PORT), HealthHandler)
    print(f"Health server running on port {PORT}")
    server.serve_forever()


# =========================================================
# TWELVE DATA
# =========================================================

def get_market_data(symbol, interval="30min", outputsize=150):
    if not TWELVE_DATA_KEY:
        raise Exception("TWELVE_DATA_KEY is missing")

    url = "https://api.twelvedata.com/time_series"

    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": outputsize,
        "apikey": TWELVE_DATA_KEY,
        "format": "JSON",
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()

    data = response.json()

    if "values" not in data:
        raise Exception(data.get("message", "No market data returned"))

    df = pd.DataFrame(data["values"])

    if df.empty:
        raise Exception("Empty market data")

    df["datetime"] = pd.to_datetime(df["datetime"])

    for column in ["open", "high", "low", "close"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.sort_values("datetime").reset_index(drop=True)

    # EMA
    df["EMA9"] = df["close"].ewm(span=9, adjust=False).mean()
    df["EMA21"] = df["close"].ewm(span=21, adjust=False).mean()

    # RSI
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss.replace(0, pd.NA)
    df["RSI14"] = 100 - (100 / (1 + rs))

    # ATR
    previous_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - previous_close).abs(),
        (df["low"] - previous_close).abs(),
    ], axis=1).max(axis=1)

    df["ATR14"] = tr.rolling(14).mean()

    # Support / resistance
    df["HIGH20"] = df["high"].rolling(20).max()
    df["LOW20"] = df["low"].rolling(20).min()

    df = df.dropna(
        subset=["EMA9", "EMA21", "RSI14", "ATR14", "HIGH20", "LOW20"]
    )

    if df.empty:
        raise Exception("Not enough market data")

    return df


# =========================================================
# TECHNICAL ANALYSIS
# =========================================================

def target_distance_pips(symbol, entry, target):
    try:
        pip_size = float(get_pip_size(symbol))
        if pip_size <= 0:
            return 0.0
        return abs(float(target) - float(entry)) / pip_size
    except Exception:
        return 0.0


def target_meets_minimum(symbol, entry, target):
    return target_distance_pips(symbol, entry, target) >= INTERNAL_MIN_TARGET_PIPS


def technical_analysis(symbol, timeframe):
    df = get_market_data(symbol, timeframe, 150)
    last = df.iloc[-1]

    price = float(last["close"])
    ema9 = float(last["EMA9"])
    ema21 = float(last["EMA21"])
    rsi = float(last["RSI14"])
    atr = float(last["ATR14"])

    support = float(last["LOW20"])
    resistance = float(last["HIGH20"])

    # Signal
    if ema9 > ema21 and 52 <= rsi <= 70:
        signal = "BUY"
        confidence = 70

        if 55 <= rsi <= 65:
            confidence += 5

        if price > ema9:
            confidence += 5

    elif ema9 < ema21 and 30 <= rsi <= 48:
        signal = "SELL"
        confidence = 70

        if 35 <= rsi <= 45:
            confidence += 5

        if price < ema9:
            confidence += 5

    else:
        signal = "WAIT"
        confidence = 55

    confidence = min(confidence, 95)

    # Use live market structure and ATR to derive levels.
    # There is no fixed pip target restriction.
    target_distance = max(atr * 1.5, get_pip_size(symbol))

    if signal == "BUY":
        entry = price
        tp = resistance if resistance > price else price + target_distance
        sl = support if support < price else price - target_distance
        buy_limit = support if support < price else price - target_distance
        sell_limit = resistance if resistance > price else price + target_distance

    elif signal == "SELL":
        entry = price
        tp = support if support < price else price - target_distance
        sl = resistance if resistance > price else price + target_distance
        sell_limit = resistance if resistance > price else price + target_distance
        buy_limit = support if support < price else price - target_distance

    else:
        entry = price
        tp = price + target_distance
        sl = price - target_distance
        buy_limit = price - target_distance
        sell_limit = price + target_distance

    # Only publish a directional signal when the live market-structure
    # target is at least the internal minimum distance. Do not force a
    # larger TP just to satisfy the filter; use WAIT when the market
    # does not support the required target.
    pip_size = get_pip_size(symbol)
    if signal in ("BUY", "SELL") and pip_size:
        target_pips = abs(float(tp) - float(entry)) / float(pip_size)
        if target_pips < INTERNAL_MIN_TARGET_PIPS:
            signal = "WAIT"
            confidence = min(confidence, 55)

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
        "atr": atr,
        "support": support,
        "resistance": resistance,
        "pip_size": get_pip_size(symbol),
    }


# =========================================================
# FORMAT
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


def format_technical_result(result):
    signal = result["signal"]

    emoji = (
        "🟢" if signal == "BUY"
        else "🔴" if signal == "SELL"
        else "⚪"
    )

    return (
        f"📊 {result['symbol']} | {result['timeframe']}\n\n"
        f"🤖 BOT: {emoji} {signal}\n"
        f"🔥 Confidence: {result['confidence']}%\n"
        f"🟢 Entry: {format_price(result['entry'])}\n"
        f"🎯 TP: {format_price(result['tp'])}\n"
        f"🛑 SL: {format_price(result['sl'])}\n\n"
        f"🟢 BUY LIMIT: {format_price(result['buy_limit'])}\n"
        f"🔴 SELL LIMIT: {format_price(result['sell_limit'])}"
    )


# =========================================================
# AI HELPERS
# =========================================================

def image_to_base64(image_path):
    with open(image_path, "rb") as file:
        return base64.b64encode(file.read()).decode("utf-8")


def clean_json(text):
    text = text.strip()

    if text.startswith("```"):
        text = text.replace("```json", "")
        text = text.replace("```", "")
        text = text.strip()

    start = text.find("{")
    end = text.rfind("}")

    if start >= 0 and end >= 0:
        text = text[start:end + 1]

    return text


def trading_prompt():
    return f"""
Analyze the trading chart carefully.

Return ONLY valid JSON.
NO markdown.
NO explanation.
NO extra text.

Use exactly these fields:
{{
  "symbol": "XAUUSD",
  "timeframe": "M15",
  "signal": "BUY",
  "entry": "4375",
  "tp": "4380",
  "sl": "4365",
  "buy_limit": "4370",
  "sell_limit": "4380",
  "confidence": "82%",
  "reason": "short reason"
}}

Rules:
- signal must be BUY, SELL, or WAIT.
- Confidence must be 0% to 100%.
- Set TP from the live market structure and visible target area.
- Do not force a fixed pip distance.
- If the chart does not support a clear target, use WAIT.
- BUY: SL below entry.
- SELL: SL above entry.
- Do not guarantee profit.
- Do not invent certainty.
- Keep reason under 8 words.
"""


def groq_photo_analysis(image_path):
    if not GROQ_API_KEY:
        return {"error": "GROQ_API_KEY missing"}

    image_b64 = image_to_base64(image_path)

    url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": trading_prompt()},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze this chart."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64}"
                        },
                    },
                ],
            },
        ],
        "temperature": 0.1,
        "max_tokens": 300,
    }

    response = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=25,
    )
    response.raise_for_status()

    data = response.json()
    text = data["choices"][0]["message"]["content"]

    try:
        return json.loads(clean_json(text))
    except Exception:
        return {"error": "Groq returned invalid JSON", "raw": text}


def gemini_photo_analysis(image_path):
    """Analyze a chart with Gemini, with retry/fallback protection.

    A temporary Gemini 503 must NOT stop the whole Telegram photo analysis.
    """
    if not GEMINI_API_KEY:
        return {
            "signal": "WAIT",
            "confidence": "0",
            "error": "GEMINI_API_KEY missing",
        }

    image_b64 = image_to_base64(image_path)

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": trading_prompt()},
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": image_b64,
                        }
                    },
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 300,
            "responseMimeType": "application/json",
        },
    }

    last_error = "Gemini unavailable"

    # Google recommends exponential backoff for transient 503/429/5xx errors.
    for model in GEMINI_FALLBACK_MODELS:
        url = (
            "https://generativelanguage.googleapis.com/"
            f"v1beta/models/{model}:generateContent"
            f"?key={GEMINI_API_KEY}"
        )

        for attempt in range(2):
            try:
                response = requests.post(
                    url,
                    json=payload,
                    timeout=25,
                )

                # Retry transient server/rate-limit errors.
                if response.status_code in (408, 429, 500, 502, 503, 504):
                    last_error = (
                        f"Gemini {model}: HTTP {response.status_code}"
                    )
                    if attempt < 1:
                        time.sleep(1)
                        continue
                    break

                response.raise_for_status()
                data = response.json()

                candidates = data.get("candidates", [])
                if not candidates:
                    last_error = f"Gemini {model}: no candidates returned"
                    break

                parts = candidates[0].get("content", {}).get("parts", [])
                text = "".join(
                    part.get("text", "")
                    for part in parts
                    if isinstance(part, dict)
                ).strip()

                if not text:
                    last_error = f"Gemini {model}: empty response"
                    break

                try:
                    result = json.loads(clean_json(text))
                    if isinstance(result, dict):
                        result["model"] = model
                        return result
                    last_error = f"Gemini {model}: invalid JSON object"
                except Exception:
                    last_error = f"Gemini {model}: invalid JSON"
                    break

            except requests.RequestException as exc:
                last_error = f"Gemini {model}: {exc}"
                if attempt < 1:
                    time.sleep(1)
                    continue
                break
            except Exception as exc:
                last_error = f"Gemini {model}: {exc}"
                break

    # IMPORTANT: Do not fail the entire Telegram photo analysis if Gemini is down.
    return {
        "signal": "WAIT",
        "confidence": "0",
        "error": last_error,
    }


# =========================================================
# FINAL SIGNAL
# =========================================================

def final_signal(bot, groq, gemini):
    # Only count BUY/SELL opinions that meet the minimum confidence.
    # This prevents a weak AI opinion from helping create a trade signal.
    signals = []

    for data in [bot, groq, gemini]:
        if not isinstance(data, dict):
            continue

        signal = str(data.get("signal", "WAIT")).upper()
        confidence = extract_confidence(data)

        if signal in ["BUY", "SELL"] and confidence >= MIN_CONFIDENCE:
            signals.append(signal)

    if not signals:
        return "WAIT"

    buys = signals.count("BUY")
    sells = signals.count("SELL")

    # Require at least 2 confident sources to agree.
    if buys >= 2:
        return "BUY"

    if sells >= 2:
        return "SELL"

    return "WAIT"


def extract_confidence(data):
    try:
        value = str(data.get("confidence", "0"))
        value = value.replace("%", "").strip()
        return float(value)
    except Exception:
        return 0


def calculate_confidence(bot, groq, gemini):
    values = []

    for data in [bot, groq, gemini]:
        if isinstance(data, dict):
            values.append(extract_confidence(data))

    if not values:
        return 0

    return round(sum(values) / len(values))


# =========================================================
# VALIDATE AI LEVELS
# =========================================================

def validate_ai_levels(data, final, symbol):
    """Keep AI levels consistent with the selected direction without forcing a fixed target size."""
    if not isinstance(data, dict):
        return data

    try:
        entry = float(data.get("entry"))
    except Exception:
        return data

    try:
        tp = float(data.get("tp", entry))
    except Exception:
        tp = entry

    try:
        sl = float(data.get("sl", entry))
    except Exception:
        sl = entry

    step = get_pip_size(symbol)

    if final == "BUY":
        if tp <= entry:
            tp = entry + step
        if sl >= entry:
            sl = entry - step
    elif final == "SELL":
        if tp >= entry:
            tp = entry - step
        if sl <= entry:
            sl = entry + step

    data["entry"] = entry
    data["tp"] = tp
    data["sl"] = sl
    data["buy_limit"] = entry - step
    data["sell_limit"] = entry + step
    return data


# =========================================================
# COMBINED PHOTO OUTPUT
# =========================================================

def combined_output(bot, groq, gemini):
    final = final_signal(bot, groq, gemini)
    confidence = calculate_confidence(bot, groq, gemini)

    source = None

    if final in ["BUY", "SELL"]:
        for data in [bot, groq, gemini]:
            if (
                isinstance(data, dict)
                and data.get("signal") == final
            ):
                source = data
                break

    if source is None:
        for data in [bot, groq, gemini]:
            if isinstance(data, dict) and data:
                source = data
                break

    if source is None:
        source = {}

    symbol = get_value(bot, "symbol")
    timeframe = get_value(bot, "timeframe")

    # Prefer the technical bot's real market price when available.
    if isinstance(bot, dict) and bot.get("price") is not None:
        try:
            source = dict(source)
            source["entry"] = float(bot["price"])
        except Exception:
            pass

    source = validate_ai_levels(source, final, symbol)

    # Final target-distance safety filter. Do not force a larger TP;
    # if the selected setup does not support the internal minimum, use WAIT.
    if final in ["BUY", "SELL"]:
        if not target_meets_minimum(symbol, source.get("entry"), source.get("tp")):
            final = "WAIT"

    # A final BUY/SELL is only shown when confidence is acceptable.
    if final in ["BUY", "SELL"] and confidence < MIN_CONFIDENCE:
        final = "WAIT"

    if final == "WAIT":
        # Show levels, but clearly mark that there is no accepted trade.
        try:
            entry = float(source.get("entry", bot.get("price")))
            distance = get_pip_size(symbol)
            source["entry"] = entry
            source["tp"] = entry + distance
            source["sl"] = entry - distance
            source["buy_limit"] = entry - distance
            source["sell_limit"] = entry + distance
        except Exception:
            pass

    return (
        f"📊 {symbol} | {timeframe}\n\n"
        f"🤖 BOT: {get_value(bot, 'signal')}\n"
        f"👁️ GROQ: {get_value(groq, 'signal')}\n"
        f"✨ GEMINI: {get_value(gemini, 'signal')}\n\n"
        f"🎯 FINAL: {final}\n"
        f"🔥 Confidence: {confidence}%\n"
        f"🟢 Entry: {get_value(source, 'entry')}\n"
        f"🎯 TP: {get_value(source, 'tp')}\n"
        f"🛑 SL: {get_value(source, 'sl')}\n\n"
        f"🟢 BUY LIMIT: {get_value(source, 'buy_limit')}\n"
        f"🔴 SELL LIMIT: {get_value(source, 'sell_limit')}"
    )


def get_value(data, key):
    if not isinstance(data, dict):
        return "-"

    value = data.get(key, "-")
    if value is None:
        return "-"

    return format_price(value) if key in {
        "entry", "tp", "sl", "buy_limit", "sell_limit"
    } else str(value)


# =========================================================
# COMMANDS
# =========================================================

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 ETHIO TRADE BOT\n\n"
        "🎯 Live market-structure targets\n"
        "📊 All configured symbols\n"
        "⏱ All configured timeframes\n\n"
        "Use:\n"
        "/analyze XAU/USD 15min\n"
        "/analyze BTC/USD 1h\n"
        "/scan XAU/USD\n"
        "/scan BTC/USD\n\n"
        "📸 Send a chart screenshot for AI analysis."
    )


async def ping_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🟢 BOT ONLINE")


async def symbols_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📊 Available Symbols:\n\n"
        + "\n".join(f"• {symbol}" for symbol in SYMBOLS)
    )


async def timeframes_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "⏱ Available Timeframes:\n\n"
        + "\n".join(f"• {tf}" for tf in TIMEFRAMES)
    )


async def analyze_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if len(context.args) < 2:
        await update.message.reply_text(
            "❌ Use:\n/analyze XAU/USD 15min"
        )
        return

    symbol = normalize_symbol(context.args[0])
    timeframe = normalize_timeframe(context.args[1])

    if not valid_symbol(symbol):
        await update.message.reply_text(
            "❌ Invalid symbol.\nUse /symbols"
        )
        return

    if not valid_timeframe(timeframe):
        await update.message.reply_text(
            "❌ Invalid timeframe.\nUse /timeframes"
        )
        return

    message = await update.message.reply_text(
        f"⏳ Analyzing {symbol} | {timeframe}..."
    )

    try:
        result = await asyncio.to_thread(
            technical_analysis, symbol, timeframe
        )

        await message.edit_text(
            format_technical_result(result)
        )

    except Exception as e:
        await message.edit_text(
            "❌ Analysis error:\n" + str(e)
        )


async def scan_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if len(context.args) < 1:
        await update.message.reply_text(
            "❌ Use:\n/scan XAU/USD"
        )
        return

    symbol = normalize_symbol(context.args[0])

    if not valid_symbol(symbol):
        await update.message.reply_text(
            "❌ Invalid symbol.\nUse /symbols"
        )
        return

    status = await update.message.reply_text(
        f"⏳ Scanning ALL timeframes for {symbol}..."
    )

    results = []

    for timeframe in TIMEFRAMES:
        try:
            result = await asyncio.to_thread(
                technical_analysis, symbol, timeframe
            )

            signal = result["signal"]
            emoji = (
                "🟢" if signal == "BUY"
                else "🔴" if signal == "SELL"
                else "⚪"
            )

            results.append(
                f"{emoji} {timeframe} | {signal} | "
                f"{result['confidence']}% | TP={format_price(result['tp'])}"
            )

        except Exception:
            results.append(f"⚠️ {timeframe} | ERROR")

    await status.edit_text(
        f"📊 {symbol} ALL-TIMEFRAME SCAN\n\n"
        + "\n".join(results)
    )


# =========================================================
# PHOTO
# =========================================================

async def photo_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    photo = update.message.photo[-1]
    file = await photo.get_file()

    image_path = "/tmp/chart.jpg"
    await file.download_to_drive(image_path)

    status = await update.message.reply_text(
        "🔎 Analyzing chart...\n"
        "🤖 BOT + GROQ + GEMINI\n"
        "🎯 Checking live market structure..."
    )

    try:
        groq_task = asyncio.to_thread(
            groq_photo_analysis, image_path
        )
        gemini_task = asyncio.to_thread(
            gemini_photo_analysis, image_path
        )

        groq, gemini = await asyncio.gather(
            groq_task, gemini_task
        )

        bot = None
        detected_symbol = None
        detected_timeframe = None

        for data in [groq, gemini]:
            if isinstance(data, dict):
                if data.get("symbol"):
                    detected_symbol = data.get("symbol")
                if data.get("timeframe"):
                    detected_timeframe = data.get("timeframe")

                if detected_symbol and detected_timeframe:
                    break

        if detected_symbol:
            detected_symbol = normalize_symbol(detected_symbol)

        if detected_timeframe:
            detected_timeframe = normalize_timeframe(detected_timeframe)

        if (
            detected_symbol in SYMBOLS
            and detected_timeframe in TIMEFRAMES
        ):
            try:
                bot = await asyncio.to_thread(
                    technical_analysis,
                    detected_symbol,
                    detected_timeframe,
                )
            except Exception as e:
                bot = {
                    "symbol": detected_symbol,
                    "timeframe": detected_timeframe,
                    "signal": "WAIT",
                    "confidence": "0",
                    "error": str(e),
                }
        else:
            bot = {
                "symbol": detected_symbol or get_value(groq, "symbol"),
                "timeframe": detected_timeframe or get_value(groq, "timeframe"),
                "signal": "WAIT",
                "confidence": "0",
            }

        output = combined_output(bot, groq, gemini)
        await status.edit_text(output)

    except Exception as e:
        await status.edit_text(
            "❌ Photo analysis error:\n" + str(e)
        )

    finally:
        try:
            if os.path.exists(image_path):
                os.remove(image_path)
        except Exception:
            pass


async def text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 Use:\n\n"
        "/analyze XAU/USD 15min\n"
        "/scan XAU/USD\n"
        "/symbols\n"
        "/timeframes\n\n"
        "📸 Or send a chart screenshot."
    )


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    print("Telegram error:", context.error)


# =========================================================
# MAIN
# =========================================================

def main():
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is missing")

    health_thread = threading.Thread(
        target=start_health_server,
        daemon=True,
    )
    health_thread.start()

    application = (
        Application.builder()
        .token(TELEGRAM_BOT_TOKEN)
        .build()
    )

    application.add_handler(
        CommandHandler("start", start_command)
    )
    application.add_handler(
        CommandHandler("ping", ping_command)
    )
    application.add_handler(
        CommandHandler("symbols", symbols_command)
    )
    application.add_handler(
        CommandHandler("timeframes", timeframes_command)
    )
    application.add_handler(
        CommandHandler("analyze", analyze_command)
    )
    application.add_handler(
        CommandHandler("scan", scan_command)
    )

    application.add_handler(
        MessageHandler(filters.PHOTO, photo_handler)
    )

    application.add_handler(
        MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            text_handler,
        )
    )

    application.add_error_handler(error_handler)

    print("ETHIO TRADE BOT STARTED")
    print("Live target mode: market structure + ATR")
    print(f"Symbols: {len(SYMBOLS)}")
    print(f"Timeframes: {len(TIMEFRAMES)}")

    application.run_polling(
        drop_pending_updates=True
    )


if __name__ == "__main__":
    main()

