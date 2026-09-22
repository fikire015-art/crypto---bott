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
TWELVE_DATA_KEY = os.getenv("TWELVE_DATA_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

PORT = int(os.getenv("PORT", "10000"))

GROQ_MODEL = os.getenv("GROQ_VISION_MODEL", "qwen/qwen3.8-27b")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

# A directional signal needs at least two agreeing sources.
MIN_CONFIDENCE = 70

# Do NOT use a fixed 101-pip target filter.
# A setup is accepted when its target has a reasonable ATR/R:R structure.
MIN_RR = 1.5

# =========================================================
# SYMBOLS
# =========================================================

SYMBOLS = [
    "EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF", "AUD/USD",
    "NZD/USD", "USD/CAD", "EUR/GBP", "EUR/JPY", "GBP/JPY",
    "AUD/JPY", "NZD/JPY", "CAD/JPY", "CHF/JPY", "EUR/CHF",
    "GBP/CHF", "AUD/CAD", "AUD/CHF", "CAD/CHF", "NZD/CAD",
    "NZD/CHF",
    "XAU/USD", "XAG/USD",
    "BTC/USD", "ETH/USD", "BNB/USD", "SOL/USD", "XRP/USD",
    "DOGE/USD", "ADA/USD", "AVAX/USD", "LINK/USD", "TRX/USD",
    "DOT/USD", "LTC/USD", "BCH/USD", "ATOM/USD", "ETC/USD",
    "FIL/USD", "NEAR/USD", "APT/USD", "ARB/USD", "OP/USD",
]

TIMEFRAMES = [
    "1min", "5min", "15min", "30min", "45min",
    "1h", "2h", "4h", "8h",
    "1day", "1week", "1month",
]

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


def normalize_symbol(symbol):
    return ALIASES.get(str(symbol).strip().upper(), str(symbol).strip().upper())


def normalize_timeframe(timeframe):
    timeframe = str(timeframe).strip().lower()
    aliases = {
        "1m": "1min", "5m": "5min", "15m": "15min",
        "30m": "30min", "45m": "45min",
        "h1": "1h", "h2": "2h", "h4": "4h", "h8": "8h",
        "d": "1day", "1d": "1day",
        "w": "1week", "1w": "1week",
        "mo": "1month", "1mo": "1month", "mn1": "1month",
        "m15": "15min", "m30": "30min", "h1": "1h",
    }
    return aliases.get(timeframe, timeframe)


def valid_symbol(symbol):
    return symbol in SYMBOLS


def valid_timeframe(timeframe):
    return timeframe in TIMEFRAMES


# =========================================================
# MARKET DATA
# =========================================================

def get_pip_size(symbol):
    symbol = normalize_symbol(symbol)
    if symbol in ("XAU/USD", "XAG/USD"):
        return 0.01
    if "JPY" in symbol:
        return 0.01
    if symbol in {
        "XRP/USD", "DOGE/USD", "ADA/USD", "TRX/USD",
        "ARB/USD", "OP/USD"
    }:
        return 0.0001
    if symbol in {
        "BTC/USD", "ETH/USD", "BNB/USD", "SOL/USD",
        "AVAX/USD", "LINK/USD", "DOT/USD", "LTC/USD",
        "BCH/USD", "ATOM/USD", "ETC/USD", "FIL/USD",
        "NEAR/USD", "APT/USD"
    }:
        return 0.01
    if "/" in symbol:
        return 0.0001
    return 0.01


def get_market_data(symbol, interval="15min", outputsize=200):
    if not TWELVE_DATA_KEY:
        raise RuntimeError("TWELVE_DATA_KEY is missing")

    response = requests.get(
        "https://api.twelvedata.com/time_series",
        params={
            "symbol": symbol,
            "interval": interval,
            "outputsize": outputsize,
            "apikey": TWELVE_DATA_KEY,
            "format": "JSON",
        },
        timeout=12,
    )
    response.raise_for_status()
    data = response.json()

    if "values" not in data:
        raise RuntimeError(data.get("message", "No market data returned"))

    df = pd.DataFrame(data["values"])
    if df.empty:
        raise RuntimeError("Empty market data")

    df["datetime"] = pd.to_datetime(df["datetime"])
    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values("datetime").reset_index(drop=True)

    # Indicators
    df["EMA9"] = df["close"].ewm(span=9, adjust=False).mean()
    df["EMA21"] = df["close"].ewm(span=21, adjust=False).mean()

    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    df["RSI14"] = 100 - (100 / (1 + rs))

    previous_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - previous_close).abs(),
            (df["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["ATR14"] = tr.rolling(14).mean()

    # Exclude the current incomplete candle where possible.
    if len(df) > 25:
        df = df.iloc[:-1].copy()

    df["HIGH20"] = df["high"].rolling(20).max().shift(1)
    df["LOW20"] = df["low"].rolling(20).min().shift(1)

    df = df.dropna(
        subset=["EMA9", "EMA21", "RSI14", "ATR14", "HIGH20", "LOW20"]
    )
    if df.empty:
        raise RuntimeError("Not enough market data")

    return df


# =========================================================
# TECHNICAL ANALYSIS
# =========================================================

def technical_analysis(symbol, timeframe):
    df = get_market_data(symbol, timeframe, 200)
    last = df.iloc[-1]

    price = float(last["close"])
    ema9 = float(last["EMA9"])
    ema21 = float(last["EMA21"])
    rsi = float(last["RSI14"])
    atr = float(last["ATR14"])
    support = float(last["LOW20"])
    resistance = float(last["HIGH20"])

    # Trend + momentum
    if ema9 > ema21:
        signal = "BUY"
        confidence = 65
        if price > ema9:
            confidence += 8
        if 52 <= rsi <= 68:
            confidence += 7
        elif rsi > 75:
            confidence -= 10
    elif ema9 < ema21:
        signal = "SELL"
        confidence = 65
        if price < ema9:
            confidence += 8
        if 32 <= rsi <= 48:
            confidence += 7
        elif rsi < 25:
            confidence -= 10
    else:
        signal = "WAIT"
        confidence = 50

    confidence = max(0, min(95, confidence))

    # Structure-based levels.
    if signal == "BUY":
        entry = price
        structural_tp = resistance if resistance > entry else entry + 1.5 * atr
        risk_distance = max(entry - support, 0.8 * atr)
        sl = entry - risk_distance
        tp = structural_tp

        # If structure is too close, use an ATR target rather than forcing
        # a fixed number of pips.
        if tp <= entry or (tp - entry) / max(entry - sl, 1e-9) < MIN_RR:
            tp = entry + max(1.5 * atr, MIN_RR * (entry - sl))

        buy_limit = max(support, entry - 0.5 * atr)
        sell_limit = tp

    elif signal == "SELL":
        entry = price
        structural_tp = support if support < entry else entry - 1.5 * atr
        risk_distance = max(resistance - entry, 0.8 * atr)
        sl = entry + risk_distance
        tp = structural_tp

        if tp >= entry or (entry - tp) / max(sl - entry, 1e-9) < MIN_RR:
            tp = entry - max(1.5 * atr, MIN_RR * (sl - entry))

        sell_limit = min(resistance, entry + 0.5 * atr)
        buy_limit = tp

    else:
        entry = price
        tp = price + atr
        sl = price - atr
        buy_limit = price - 0.5 * atr
        sell_limit = price + 0.5 * atr

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "signal": signal,
        "entry": entry,
        "tp": tp,
        "sl": sl,
        "buy_limit": buy_limit,
        "sell_limit": sell_limit,
        "confidence": round(confidence),
        "price": price,
        "ema9": ema9,
        "ema21": ema21,
        "rsi": rsi,
        "atr": atr,
        "support": support,
        "resistance": resistance,
        "pip_size": get_pip_size(symbol),
    }


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
    emoji = "🟢" if signal == "BUY" else "🔴" if signal == "SELL" else "⚪️"

    return (
        f"📊 {result['symbol']} | {result['timeframe']}\n\n"
        f"🤖 BOT: {emoji} {signal}\n"
        f"🔥 Confidence: {result['confidence']}%\n"
        f"💵 Price: {format_price(result['price'])}\n"
        f"📈 EMA9: {format_price(result['ema9'])}\n"
        f"📉 EMA21: {format_price(result['ema21'])}\n"
        f"RSI14: {result['rsi']:.1f}\n"
        f"ATR14: {format_price(result['atr'])}\n\n"
        f"🟢 Entry: {format_price(result['entry'])}\n"
        f"🎯 TP: {format_price(result['tp'])}\n"
        f"🛑 SL: {format_price(result['sl'])}\n"
    )


# =========================================================
# AI HELPERS
# =========================================================

def image_to_base64(image_path):
    with open(image_path, "rb") as file:
        return base64.b64encode(file.read()).decode("utf-8")


def clean_json(text):
    text = str(text).strip()
    if text.startswith("```"):
        text = text.replace("```json", "").replace("```", "").strip()
    start = text.find("{")
    end = text.rfind("}")
    return text[start:end + 1] if start >= 0 and end >= 0 else text


def trading_prompt():
    return """
Analyze this trading chart carefully.

Return ONLY valid JSON:
{
  "symbol": "XAU/USD",
  "timeframe": "15min",
  "signal": "BUY",
  "entry": 0,
  "tp": 0,
  "sl": 0,
  "buy_limit": 0,
  "sell_limit": 0,
  "confidence": 0,
  "reason": "short reason"
}

Rules:
- signal must be BUY, SELL, or WAIT.
- confidence must be a number from 0 to 100.
- Read the visible symbol and timeframe from the chart.
- Do not invent a symbol/timeframe.
- Use visible market structure, trend, momentum and support/resistance.
- BUY: TP above entry and SL below entry.
- SELL: TP below entry and SL above entry.
- If the chart is unclear, use WAIT.
- No profit guarantees.
"""


def groq_photo_analysis(image_path):
    if not GROQ_API_KEY:
        return {"signal": "WAIT", "confidence": 0, "error": "GROQ_API_KEY missing"}

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
                            "url": "data:image/jpeg;base64,"
                            + image_to_base64(image_path)
                        },
                    },
                ],
            },
        ],
        "temperature": 0.1,
        "max_completion_tokens": 500,
        "response_format": {"type": "json_object"},
    }

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=20,
        )
        response.raise_for_status()
        text = response.json()["choices"][0]["message"]["content"]
        result = json.loads(clean_json(text))
        result["model"] = GROQ_MODEL
        return result
    except Exception as exc:
        return {"signal": "WAIT", "confidence": 0, "error": f"Groq: {exc}"}


def gemini_photo_analysis(image_path):
    if not GEMINI_API_KEY:
        return {"signal": "WAIT", "confidence": 0, "error": "GEMINI_API_KEY missing"}

    payload = {
        "contents": [{
            "parts": [
                {"text": trading_prompt()},
                {
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": image_to_base64(image_path),
                    }
                },
            ]
        }],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 500,
            "responseMimeType": "application/json",
        },
    }

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    )

    try:
        response = requests.post(url, json=payload, timeout=20)
        response.raise_for_status()
        data = response.json()
        parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in parts if isinstance(p, dict))
        result = json.loads(clean_json(text))
        result["model"] = GEMINI_MODEL
        return result
    except Exception as exc:
        return {"signal": "WAIT", "confidence": 0, "error": f"Gemini: {exc}"}


# =========================================================
# FINAL DECISION
# =========================================================

def extract_confidence(data):
    if not isinstance(data, dict):
        return 0
    try:
        return float(str(data.get("confidence", 0)).replace("%", "").strip())
    except Exception:
        return 0


def final_signal(bot, groq, gemini):
    sources = [bot, groq, gemini]
    votes = {"BUY": 0, "SELL": 0}

    for data in sources:
        if not isinstance(data, dict):
            continue
        signal = str(data.get("signal", "WAIT")).upper()
        confidence = extract_confidence(data)
        if signal in votes and confidence >= MIN_CONFIDENCE:
            votes[signal] += 1

    if votes["BUY"] >= 2:
        return "BUY"
    if votes["SELL"] >= 2:
        return "SELL"
    return "WAIT"


def calculate_confidence(bot, groq, gemini):
    # Average only valid responding sources.
    values = []
    for data in [bot, groq, gemini]:
        if not isinstance(data, dict):
            continue
        if data.get("error"):
            continue
        if "confidence" in data:
            values.append(extract_confidence(data))

    return round(sum(values) / len(values)) if values else 0


def get_value(data, key):
    if not isinstance(data, dict):
        return "-"
    value = data.get(key)
    if value is None:
        return "-"
    if key in {"entry", "tp", "sl", "buy_limit", "sell_limit"}:
        return format_price(value)
    return str(value)


def choose_source(final, bot, groq, gemini):
    candidates = [bot, groq, gemini]

    matching = [
        d for d in candidates
        if isinstance(d, dict)
        and str(d.get("signal", "WAIT")).upper() == final
        and not d.get("error")
    ]

    if matching:
        return max(matching, key=extract_confidence)

    valid = [d for d in candidates if isinstance(d, dict) and not d.get("error")]
    return max(valid, key=extract_confidence) if valid else {}


def combined_output(bot, groq, gemini):
    final = final_signal(bot, groq, gemini)
    confidence = calculate_confidence(bot, groq, gemini)
    source = choose_source(final, bot, groq, gemini)

    symbol = get_value(bot, "symbol")
    timeframe = get_value(bot, "timeframe")

    # If technical analysis has a real live price, use it as the entry.
    if isinstance(bot, dict) and bot.get("price") is not None:
        source = dict(source)
        source["entry"] = bot["price"]

    if final == "WAIT":
        reason = "Sources do not agree or setup is not strong enough."
    else:
        reason = get_value(source, "reason")

    return (
        f"📊 {symbol} | {timeframe}\n\n"
        f"🤖 BOT: {get_value(bot, 'signal')} "
        f"({extract_confidence(bot):.0f}%)\n"
        f"👁 GROQ: {get_value(groq, 'signal')} "
        f"({extract_confidence(groq):.0f}%)\n"
        f"✨ GEMINI: {get_value(gemini, 'signal')} "
        f"({extract_confidence(gemini):.0f}%)\n\n"
        f"🎯 FINAL: {final}\n"
        f"🔥 Confidence: {confidence}%\n"
        f"📝 {reason}\n\n"
        f"🟢 Entry: {get_value(source, 'entry')}\n"
        f"🎯 TP: {get_value(source, 'tp')}\n"
        f"🛑 SL: {get_value(source, 'sl')}\n\n"
        f"🟢 BUY LIMIT: {get_value(source, 'buy_limit')}\n"
        f"🔴 SELL LIMIT: {get_value(source, 'sell_limit')}"
    )


# =========================================================
# HEALTH SERVER
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
# TELEGRAM COMMANDS
# =========================================================

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 ETHIO TRADE BOT\n\n"
        "📊 Live technical analysis + AI chart analysis\n\n"
        "Commands:\n"
        "/analyze XAU/USD 15min\n"
        "/scan XAU/USD\n"
        "/symbols\n"
        "/timeframes\n\n"
        "📸 Send a chart screenshot for Groq + Gemini analysis."
    )


async def ping_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🟢 BOT ONLINE")


async def symbols_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📊 Available Symbols:\n\n" + "\n".join(f"• {s}" for s in SYMBOLS)
    )


async def timeframes_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "⏱️ Available Timeframes:\n\n" + "\n".join(f"• {t}" for t in TIMEFRAMES)
    )


async def analyze_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if len(context.args) < 2:
        await update.message.reply_text("❌ Use: /analyze XAU/USD 15min")
        return

    symbol = normalize_symbol(context.args[0])
    timeframe = normalize_timeframe(context.args[1])

    if not valid_symbol(symbol):
        await update.message.reply_text("❌ Invalid symbol. Use /symbols")
        return

    if not valid_timeframe(timeframe):
        await update.message.reply_text("❌ Invalid timeframe. Use /timeframes")
        return

    status = await update.message.reply_text(
        f"⏳ Analyzing {symbol} | {timeframe}..."
    )

    try:
        result = await asyncio.to_thread(technical_analysis, symbol, timeframe)
        await status.edit_text(format_technical_result(result))
    except Exception as exc:
        await status.edit_text("❌ Analysis error:\n" + str(exc))


async def scan_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if len(context.args) < 1:
        await update.message.reply_text("❌ Use: /scan XAU/USD")
        return

    symbol = normalize_symbol(context.args[0])

    if not valid_symbol(symbol):
        await update.message.reply_text("❌ Invalid symbol. Use /symbols")
        return

    status = await update.message.reply_text(
        f"⏳ Scanning all timeframes for {symbol}..."
    )

    results = []
    for timeframe in TIMEFRAMES:
        try:
            result = await asyncio.to_thread(
                technical_analysis, symbol, timeframe
            )
            results.append(
                f"{'🟢' if result['signal']=='BUY' else '🔴' if result['signal']=='SELL' else '⚪️'} "
                f"{timeframe} | {result['signal']} | {result['confidence']}%"
            )
        except Exception as exc:
            results.append(f"⚠️ {timeframe} | ERROR")

    await status.edit_text(
        f"📊 {symbol} ALL-TIMEFRAME SCAN\n\n" + "\n".join(results)
    )


# =========================================================
# PHOTO ANALYSIS
# =========================================================

async def photo_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    photo = update.message.photo[-1]
    file = await photo.get_file()

    image_path = f"/tmp/chart_{update.message.message_id}.jpg"
    await file.download_to_drive(image_path)

    status = await update.message.reply_text(
        "🔎 Analyzing chart...\n"
        "🤖 Groq + Gemini + live technical data..."
    )

    try:
        # Run BOTH vision models at the same time.
        groq_task = asyncio.to_thread(groq_photo_analysis, image_path)
        gemini_task = asyncio.to_thread(gemini_photo_analysis, image_path)

        groq, gemini = await asyncio.gather(groq_task, gemini_task)

        detected_symbol = None
        detected_timeframe = None

        for data in [groq, gemini]:
            if not isinstance(data, dict):
                continue

            if not detected_symbol and data.get("symbol"):
                detected_symbol = normalize_symbol(data["symbol"])

            if not detected_timeframe and data.get("timeframe"):
                detected_timeframe = normalize_timeframe(data["timeframe"])

        # For your XAUUSD screenshots, use XAU/USD + 15min only if
        # one of the vision models actually identified it.
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
            except Exception as exc:
                bot = {
                    "symbol": detected_symbol,
                    "timeframe": detected_timeframe,
                    "signal": "WAIT",
                    "confidence": 0,
                    "error": f"Technical data: {exc}",
                }
        else:
            bot = {
                "symbol": detected_symbol or "XAU/USD",
                "timeframe": detected_timeframe or "15min",
                "signal": "WAIT",
                "confidence": 0,
                "error": "Could not confidently detect symbol/timeframe",
            }

        output = combined_output(bot, groq, gemini)
        await status.edit_text(output)

    except Exception as exc:
        await status.edit_text("❌ Photo analysis error:\n" + str(exc))

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

    threading.Thread(
        target=start_health_server,
        daemon=True,
    ).start()

    application = (
        Application.builder()
        .token(TELEGRAM_BOT_TOKEN)
        .connect_timeout(15)
        .read_timeout(30)
        .write_timeout(30)
        .pool_timeout(30)
        .build()
    )

    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("ping", ping_command))
    application.add_handler(CommandHandler("symbols", symbols_command))
    application.add_handler(CommandHandler("timeframes", timeframes_command))
    application.add_handler(CommandHandler("analyze", analyze_command))
    application.add_handler(CommandHandler("scan", scan_command))
    application.add_handler(MessageHandler(filters.PHOTO, photo_handler))
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, text_handler)
    )
    application.add_error_handler(error_handler)

    print("ETHIO TRADE BOT STARTED")
    print("Live technical mode: market structure + EMA + RSI + ATR")
    print(f"Symbols: {len(SYMBOLS)}")
    print(f"Timeframes: {len(TIMEFRAMES)}")
    print(f"Groq model: {GROQ_MODEL}")
    print(f"Gemini model: {GEMINI_MODEL}")

    application.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()

