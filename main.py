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

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

# ============================================================
# CRYPTO FLOW BOT - FULL REPAIRED VERSION
# ============================================================
# Environment variables required on Render:
# TELEGRAM_BOT_TOKEN = your Telegram bot token
# GROQ_API_KEY       = your Groq API key
# TWELVE_DATA_KEY    = your Twelve Data API key (for FX/XAUUSD)
#
# Optional:
# PORT = Render port (Render normally sets this automatically)
#
# This bot:
# - analyzes ONE requested symbol
# - accepts text symbols such as XAUUSD, EUR/USD, BTC/USD
# - accepts chart screenshots for AI vision analysis
# - gives BUY / SELL / WAIT
# - gives confidence percentage
# - gives entry, buy limit, sell limit, SL, TP1, TP2
# - targets at least 50 pips when market range/ATR supports it
# - uses short Groq output to avoid TPM/token errors
# - sends plain Telegram text to avoid Markdown entity errors
# - includes a small HTTP health server for Render
#
# Educational analysis only. No automatic order execution.
# ============================================================

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("crypto_flow_bot")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
TWELVE_DATA_KEY = os.getenv("TWELVE_DATA_KEY", "").strip()

# Current Groq vision model. Keep this exact model ID.
VISION_MODEL = "qwen/qwen3.6-27b"

TIMEFRAME = "15min"
CANDLE_COUNT = 150

# Minimum desired move in "pips" for FX/gold style instruments.
# For crypto, the bot reports points/move instead of pretending every
# dollar move is an FX pip.
MIN_TARGET_PIPS = 50

# Keep Groq output short. This is important because the user's previous
# account returned a 1000 output-tokens-per-minute limit.
VISION_MAX_TOKENS = 650

# ============================================================
# Render health server
# ============================================================

app = Flask(__name__)


@app.get("/")
def home():
    return "Crypto Flow Bot is running."


@app.get("/health")
def health():
    return {"status": "ok", "bot": "crypto-flow-bot"}


def run_web_server():
    port = int(os.getenv("PORT", "10000"))
    app.run(host="0.0.0.0", port=port, use_reloader=False)


# ============================================================
# Symbol helpers
# ============================================================

CRYPTO_BASES = {
    "BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "DOGE", "TRX",
    "AVAX", "LINK", "DOT", "MATIC", "LTC", "BCH", "UNI", "ATOM",
    "ETC", "FIL", "NEAR", "APT", "ARB", "OP", "SUI", "PEPE",
    "SHIB", "TON", "INJ", "AAVE", "MKR", "ALGO", "VET",
}

FOREX_BASES = {
    "EUR", "GBP", "USD", "JPY", "CHF", "CAD", "AUD", "NZD",
}


def normalize_symbol(raw: str) -> str:
    s = raw.strip().upper()
    s = s.replace(" ", "")
    s = s.replace("-", "/")
    s = s.replace("_", "/")

    aliases = {
        "XAUUSD": "XAU/USD",
        "GOLD": "XAU/USD",
        "XAU/USD": "XAU/USD",
        "BTCUSD": "BTC/USD",
        "BTCUSDT": "BTC/USDT",
        "ETHUSD": "ETH/USD",
        "ETHUSDT": "ETH/USDT",
        "EURUSD": "EUR/USD",
        "GBPUSD": "GBP/USD",
        "USDJPY": "USD/JPY",
        "USDCHF": "USD/CHF",
        "USDCAD": "USD/CAD",
        "AUDUSD": "AUD/USD",
        "NZDUSD": "NZD/USD",
    }

    if s in aliases:
        return aliases[s]

    if "/" in s:
        return s

    # Common six-letter FX pair, e.g. EURUSD.
    if len(s) == 6 and s[:3] in FOREX_BASES and s[3:] in FOREX_BASES:
        return f"{s[:3]}/{s[3:]}"

    # Common crypto suffixes.
    for suffix in ("USDT", "USDC", "USD"):
        if s.endswith(suffix) and len(s) > len(suffix):
            base = s[:-len(suffix)]
            if base in CRYPTO_BASES:
                return f"{base}/{suffix}"

    return s


def is_crypto(symbol: str) -> bool:
    s = symbol.replace("/", "")
    return any(s.endswith(q) for q in ("USDT", "USDC", "USD")) and any(
        s.startswith(b) for b in CRYPTO_BASES
    )


def is_forex_or_gold(symbol: str) -> bool:
    return symbol == "XAU/USD" or (
        "/" in symbol
        and symbol.split("/")[0] in FOREX_BASES
        and symbol.split("/")[1] in FOREX_BASES
    )


def pip_size(symbol: str) -> float:
    """Approximate conventional pip size for FX/gold."""
    if symbol == "XAU/USD":
        return 0.10
    if symbol.endswith("/JPY"):
        return 0.01
    if is_forex_or_gold(symbol):
        return 0.0001
    return 0.01


def fmt_price(value: float) -> str:
    if value >= 1000:
        return f"{value:,.2f}"
    if value >= 100:
        return f"{value:,.3f}"
    if value >= 1:
        return f"{value:,.4f}"
    return f"{value:,.6f}"


# ============================================================
# Market data
# ============================================================

def fetch_binance(symbol: str) -> pd.DataFrame:
    pair = symbol.replace("/", "").upper()

    # Binance commonly uses USDT. If user requests BTC/USD, use BTCUSDT.
    if pair.endswith("USD") and not pair.endswith("USDT"):
        pair = pair[:-3] + "USDT"

    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": pair,
        "interval": "15m",
        "limit": CANDLE_COUNT,
    }

    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()

    if not isinstance(data, list) or len(data) < 50:
        raise ValueError(f"No usable Binance data for {symbol}")

    df = pd.DataFrame(
        data,
        columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades",
            "taker_buy_base", "taker_buy_quote", "ignore",
        ],
    )

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df[["open", "high", "low", "close", "volume"]].dropna()


def fetch_twelve_data(symbol: str) -> pd.DataFrame:
    if not TWELVE_DATA_KEY:
        raise RuntimeError(
            "TWELVE_DATA_KEY is missing. Add it in Render Environment."
        )

    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,
        "interval": TIMEFRAME,
        "outputsize": CANDLE_COUNT,
        "apikey": TWELVE_DATA_KEY,
        "format": "JSON",
    }

    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()

    if "values" not in data:
        raise ValueError(data.get("message", f"No Twelve Data data for {symbol}"))

    df = pd.DataFrame(data["values"])

    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "volume" not in df.columns:
        df["volume"] = 0.0
    else:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)

    df = df[["open", "high", "low", "close", "volume"]].dropna()
    df = df.iloc[::-1].reset_index(drop=True)

    if len(df) < 50:
        raise ValueError("Not enough candles returned by Twelve Data.")

    return df


def fetch_market_data(symbol: str) -> pd.DataFrame:
    if is_crypto(symbol):
        try:
            return fetch_binance(symbol)
        except Exception as e:
            logger.warning("Binance failed for %s: %s", symbol, e)

    return fetch_twelve_data(symbol)


# ============================================================
# Technical indicators
# ============================================================

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["EMA20"] = out["close"].ewm(span=20, adjust=False).mean()
    out["EMA50"] = out["close"].ewm(span=50, adjust=False).mean()
    out["EMA200"] = out["close"].ewm(span=200, adjust=False).mean()

    delta = out["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    out["RSI"] = 100 - (100 / (1 + rs))
    out["RSI"] = out["RSI"].fillna(50)

    prev_close = out["close"].shift(1)
    tr = pd.concat(
        [
            out["high"] - out["low"],
            (out["high"] - prev_close).abs(),
            (out["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    out["ATR"] = tr.rolling(14).mean()

    ema12 = out["close"].ewm(span=12, adjust=False).mean()
    ema26 = out["close"].ewm(span=26, adjust=False).mean()
    out["MACD"] = ema12 - ema26
    out["MACD_SIGNAL"] = out["MACD"].ewm(span=9, adjust=False).mean()

    out["MOM"] = out["close"].pct_change(5) * 100

    return out.dropna().reset_index(drop=True)


# ============================================================
# Signal engine
# ============================================================

def analyze_market(df: pd.DataFrame, symbol: str) -> dict:
    d = calculate_indicators(df)

    last = d.iloc[-1]
    prev = d.iloc[-2]

    price = float(last["close"])
    atr = max(float(last["ATR"]), price * 0.001)
    ema20 = float(last["EMA20"])
    ema50 = float(last["EMA50"])
    ema200 = float(last["EMA200"])
    rsi = float(last["RSI"])
    macd = float(last["MACD"])
    macd_signal = float(last["MACD_SIGNAL"])
    momentum = float(last["MOM"])

    buy_score = 0
    sell_score = 0

    # Trend
    if ema20 > ema50:
        buy_score += 25
    elif ema20 < ema50:
        sell_score += 25

    if price > ema200:
        buy_score += 15
    elif price < ema200:
        sell_score += 15

    # RSI
    if 52 <= rsi <= 70:
        buy_score += 15
    elif 30 <= rsi <= 48:
        sell_score += 15

    # MACD
    if macd > macd_signal:
        buy_score += 15
    elif macd < macd_signal:
        sell_score += 15

    # Momentum
    if momentum > 0:
        buy_score += 10
    elif momentum < 0:
        sell_score += 10

    # Recent candle direction
    if float(last["close"]) > float(last["open"]):
        buy_score += 10
    elif float(last["close"]) < float(last["open"]):
        sell_score += 10

    # Recent structure
    recent = d.tail(20)
    recent_high = float(recent["high"].max())
    recent_low = float(recent["low"].min())

    # Make the signal explicit.
    total = max(buy_score, sell_score)
    confidence = int(min(95, max(50, 50 + total * 0.45)))

    if buy_score >= sell_score + 10:
        signal = "BUY"
    elif sell_score >= buy_score + 10:
        signal = "SELL"
    else:
        signal = "WAIT"

    # Pullback entries.
    if signal == "BUY":
        entry = price
        buy_limit = min(price - atr * 0.35, ema20)
        sell_limit = None

        sl = min(buy_limit - atr * 0.75, recent_low - atr * 0.10)

        # Ensure meaningful target distance.
        target_distance = max(atr * 1.8, pip_size(symbol) * MIN_TARGET_PIPS)

        tp1 = buy_limit + target_distance
        tp2 = buy_limit + target_distance * 1.8

    elif signal == "SELL":
        entry = price
        buy_limit = None
        sell_limit = max(price + atr * 0.35, ema20)

        sl = max(sell_limit + atr * 0.75, recent_high + atr * 0.10)

        target_distance = max(atr * 1.8, pip_size(symbol) * MIN_TARGET_PIPS)

        tp1 = sell_limit - target_distance
        tp2 = sell_limit - target_distance * 1.8

    else:
        entry = price
        buy_limit = min(price - atr * 0.35, ema20)
        sell_limit = max(price + atr * 0.35, ema20)

        sl = None
        target_distance = max(atr * 1.5, pip_size(symbol) * MIN_TARGET_PIPS)
        tp1 = None
        tp2 = None

    # Convert target distance to pips for FX/gold.
    if is_forex_or_gold(symbol):
        target_pips = int(round(target_distance / pip_size(symbol)))
        tp2_pips = int(round((target_distance * 1.8) / pip_size(symbol)))
        move_label = f"{target_pips}-{tp2_pips} pips"
    else:
        target_pips = None
        move_label = f"{fmt_price(target_distance)} price move"

    # Market condition
    if confidence >= 75 and signal in ("BUY", "SELL"):
        condition = "GOOD"
    elif confidence < 60 or signal == "WAIT":
        condition = "BAD / UNCERTAIN"
    else:
        condition = "MIXED"

    return {
        "symbol": symbol,
        "price": price,
        "signal": signal,
        "condition": condition,
        "confidence": confidence,
        "entry": entry,
        "buy_limit": buy_limit,
        "sell_limit": sell_limit,
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,
        "target_pips": target_pips,
        "move_label": move_label,
        "rsi": rsi,
        "atr": atr,
        "ema20": ema20,
        "ema50": ema50,
        "ema200": ema200,
        "macd": macd,
        "macd_signal": macd_signal,
        "momentum": momentum,
        "support": recent_low,
        "resistance": recent_high,
        "buy_score": buy_score,
        "sell_score": sell_score,
    }


# ============================================================
# Telegram formatting
# ============================================================

def price_line(label: str, value):
    if value is None:
        return f"{label}: N/A"
    return f"{label}: {fmt_price(float(value))}"


def format_market_result(a: dict) -> str:
    signal = a["signal"]

    if signal == "BUY":
        action = "🟢 BUY"
    elif signal == "SELL":
        action = "🔴 SELL"
    else:
        action = "🟡 WAIT"

    lines = [
        "📊 CRYPTO FLOW ANALYSIS",
        "",
        f"SYMBOL: {a['symbol']}",
        f"PRICE: {fmt_price(a['price'])}",
        f"SIGNAL: {action}",
        f"MARKET: {a['condition']}",
        f"CONFIDENCE: {a['confidence']}%",
        "",
        price_line("ENTRY", a["entry"]),
        price_line("BUY LIMIT", a["buy_limit"]),
        price_line("SELL LIMIT", a["sell_limit"]),
        price_line("SL", a["sl"]),
        price_line("TP1", a["tp1"]),
        price_line("TP2", a["tp2"]),
        f"TARGET: {a['move_label']}",
        "",
        f"RSI: {a['rsi']:.1f}",
        f"EMA20: {fmt_price(a['ema20'])}",
        f"EMA50: {fmt_price(a['ema50'])}",
        f"EMA200: {fmt_price(a['ema200'])}",
        f"SUPPORT: {fmt_price(a['support'])}",
        f"RESISTANCE: {fmt_price(a['resistance'])}",
        "",
        "⚠️ Educational analysis only. No automatic trade is placed.",
    ]

    return "\n".join(lines)


# ============================================================
# Groq vision
# ============================================================

def get_groq_client() -> Groq:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is missing in Render Environment.")
    return Groq(api_key=GROQ_API_KEY)


def resize_image_bytes(image_bytes: bytes) -> bytes:
    """
    Resize/compress screenshots before sending them to Groq.
    This reduces request size while preserving chart readability.
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    max_width = 1800
    if image.width > max_width:
        ratio = max_width / image.width
        image = image.resize(
            (max_width, int(image.height * ratio)),
            Image.LANCZOS,
        )

    out = io.BytesIO()
    image.save(out, format="JPEG", quality=82, optimize=True)
    return out.getvalue()


def vision_prompt(caption: str) -> str:
    return f"""
Analyze this trading chart screenshot for {caption or 'the visible symbol'}.

Return ONLY a short plain-text report. No Markdown, no tables, no long explanation.

Use exactly these labels:
SYMBOL:
TIMEFRAME:
TREND:
SIGNAL:
CONFIDENCE:
SUPPORT:
RESISTANCE:
ENTRY:
BUY LIMIT:
SELL LIMIT:
SL:
TP1:
TP2:
TARGET:
REASON:

Rules:
- SIGNAL must be exactly BUY, SELL, or WAIT.
- CONFIDENCE must be a number from 0 to 100 followed by %.
- If the chart does not clearly show an indicator, say "Not visible".
- Do not invent EMA/RSI/MACD values that are not visible.
- Estimate levels from visible price structure only.
- For XAUUSD/forex, try to give a target of at least 50 pips when the visible chart range supports it.
- If a 50+ pip target is not supported by the visible chart, say so instead of inventing one.
- Keep the entire answer under 500 words.
- This is educational market analysis, not an automatic trading instruction.
""".strip()


def analyze_chart_with_groq(image_bytes: bytes, caption: str) -> str:
    client = get_groq_client()
    compressed = resize_image_bytes(image_bytes)

    encoded = base64.b64encode(compressed).decode("utf-8")

    response = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": vision_prompt(caption),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encoded}",
                        },
                    },
                ],
            }
        ],
        temperature=0.2,
        max_completion_tokens=VISION_MAX_TOKENS,
        stream=False,
    )

    text = response.choices[0].message.content or ""
    return text.strip()


# ============================================================
# Telegram handlers
# ============================================================

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "🤖 Crypto Flow Bot\n\n"
        "Send a symbol to analyze it.\n\n"
        "Examples:\n"
        "XAUUSD\n"
        "EUR/USD\n"
        "GBP/USD\n"
        "BTC/USD\n"
        "ETH/USD\n\n"
        "You can also send a chart screenshot 📸 "
        "for AI chart analysis."
    )
    await update.message.reply_text(text)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "📘 HELP\n\n"
        "Send one symbol, for example:\n"
        "XAUUSD\n"
        "EUR/USD\n"
        "BTC/USD\n\n"
        "Or send a chart screenshot.\n\n"
        "The bot returns:\n"
        "BUY / SELL / WAIT\n"
        "Confidence %\n"
        "Entry\n"
        "Buy Limit / Sell Limit\n"
        "SL\n"
        "TP1 / TP2\n"
        "Target movement\n"
        "Support / Resistance"
    )
    await update.message.reply_text(text)


async def analyze_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text(
            "Send a symbol after /analyze.\nExample: /analyze XAUUSD"
        )
        return

    raw = context.args[0]
    await analyze_symbol_message(update, raw)


async def analyze_symbol_message(update: Update, raw: str):
    symbol = normalize_symbol(raw)

    # Basic validation.
    if len(symbol) < 5:
        await update.message.reply_text(
            "❌ Invalid symbol.\nExample: XAUUSD or EUR/USD or BTC/USD"
        )
        return

    status = await update.message.reply_text(
        f"⏳ Analyzing {symbol}..."
    )

    try:
        df = fetch_market_data(symbol)
        result = analyze_market(df, symbol)
        text = format_market_result(result)

        await status.edit_text(text)

    except Exception as e:
        logger.exception("Market analysis error")
        await status.edit_text(
            "❌ Market analysis failed.\n\n"
            f"Reason: {str(e)[:700]}"
        )


async def text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return

    raw = update.message.text.strip()

    # Ignore ordinary sentences; symbols normally contain a slash or are
    # short uppercase tokens.
    if len(raw) > 20 or " " in raw:
        await update.message.reply_text(
            "Send one symbol, for example XAUUSD, EUR/USD, or BTC/USD."
        )
        return

    await analyze_symbol_message(update, raw)


async def photo_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.photo:
        return

    status = await update.message.reply_text(
        "🖼️ Reading chart... please wait."
    )

    try:
        photo = update.message.photo[-1]
        tg_file = await photo.get_file()
        image_bytes = bytes(await tg_file.download_as_bytearray())

        caption = update.message.caption or ""

        result = analyze_chart_with_groq(image_bytes, caption)

        if not result:
            raise RuntimeError("Groq returned an empty analysis.")

        await status.edit_text(
            "📊 AI CHART ANALYSIS\n\n" + result +
            "\n\n⚠️ Educational analysis only."
        )

    except Exception as e:
        logger.exception("Chart analysis error")

        # The previous bot used Telegram Markdown parsing and failed on
        # special characters. This version deliberately sends plain text.
        error_text = str(e)

        if "rate_limit_exceeded" in error_text.lower():
            message = (
                "❌ Chart analysis failed.\n\n"
                "Groq output-token limit was reached. "
                "This version already uses a short output limit. "
                "Wait a little and send the screenshot again."
            )
        elif "model_not_found" in error_text.lower():
            message = (
                "❌ Chart analysis failed.\n\n"
                "The Groq vision model ID is invalid or unavailable. "
                "Check GROQ_API_KEY and use the current vision model."
            )
        else:
            message = (
                "❌ Chart analysis failed.\n\n"
                f"Reason: {error_text[:700]}"
            )

        await status.edit_text(message)


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    logger.exception("Unhandled Telegram error", exc_info=context.error)


# ============================================================
# Main
# ============================================================

def main():
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is missing.")

    # Start Render health server in the background.
    web_thread = threading.Thread(
        target=run_web_server,
        daemon=True,
    )
    web_thread.start()

    logger.info("Starting Crypto Flow Bot...")
    logger.info("Vision model: %s", VISION_MODEL)

    application = Application.builder().token(
        TELEGRAM_BOT_TOKEN
    ).build()

    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("analyze", analyze_command))

    # Photo handler first.
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

    logger.info("Bot is running.")
    application.run_polling(
        allowed_updates=Update.ALL_TYPES,
        drop_pending_updates=True,
    )


if __name__ == "__main__":
    main()

