import os
import re
import io
import base64
import logging
import threading
from typing import Optional

import requests
import pandas as pd
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
# CONFIG
# ============================================================

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
TWELVE_DATA_KEY = os.getenv("TWELVE_DATA_KEY", "")

# Current Groq vision model
GROQ_VISION_MODEL = "qwen/qwen3.6-27b"

# Keep this small to avoid output-token rate-limit problems.
MAX_COMPLETION_TOKENS = 700

TIMEFRAME = "15min"

# Minimum requested target
MIN_TARGET_PIPS = 50

# Minimum confidence for BUY/SELL
MIN_CONFIDENCE = 75

PORT = int(os.getenv("PORT", "10000"))


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)


# ============================================================
# GROQ CLIENT
# ============================================================

groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None


# ============================================================
# FLASK HEALTH SERVER FOR RENDER
# ============================================================

flask_app = Flask(__name__)


@flask_app.route("/")
def home():
    return "Crypto Flow Bot is running."


@flask_app.route("/health")
def health():
    return "OK"


def run_web_server():
    flask_app.run(
        host="0.0.0.0",
        port=PORT,
        use_reloader=False,
    )


# ============================================================
# HELP
# ============================================================

HELP_TEXT = """
🤖 Crypto Flow Bot

Send a symbol to analyze it.

Examples:
XAUUSD
EUR/USD
GBP/USD
USD/JPY
BTC/USD
ETH/USD

📸 You can also send a chart screenshot.

The bot returns:

🟢 BUY
🔴 SELL
🟡 WAIT

Confidence %
Entry
Buy Limit
Sell Limit
SL
TP1
TP2
Target Pips
Support
Resistance
"""


# ============================================================
# SYMBOL NORMALIZATION
# ============================================================

def normalize_symbol(symbol: str) -> str:
    symbol = symbol.strip().upper()

    symbol = symbol.replace(" ", "")
    symbol = symbol.replace("-", "/")
    symbol = symbol.replace("_", "/")

    aliases = {
        "XAUUSD": "XAU/USD",
        "XAGUSD": "XAG/USD",
        "EURUSD": "EUR/USD",
        "GBPUSD": "GBP/USD",
        "USDJPY": "USD/JPY",
        "USDCHF": "USD/CHF",
        "AUDUSD": "AUD/USD",
        "NZDUSD": "NZD/USD",
        "USDCAD": "USD/CAD",

        "BTCUSD": "BTC/USD",
        "BTCUSDT": "BTC/USD",
        "ETHUSD": "ETH/USD",
        "ETHUSDT": "ETH/USD",
        "BNBUSD": "BNB/USD",
        "SOLUSD": "SOL/USD",
        "XRPUSD": "XRP/USD",
        "ADAUSD": "ADA/USD",
        "DOGEUSD": "DOGE/USD",
    }

    return aliases.get(symbol, symbol)


# ============================================================
# PIP SIZE
# ============================================================

def get_pip_size(symbol: str) -> float:
    s = symbol.upper()

    # Gold / Silver
    if "XAU" in s:
        return 0.01

    if "XAG" in s:
        return 0.001

    # JPY forex pairs
    if "JPY" in s:
        return 0.01

    # Most forex
    if "/" in s and len(s.split("/")[0]) == 3:
        return 0.0001

    # Crypto
    if any(x in s for x in ["BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "DOGE"]):
        return 0.01

    return 0.0001


def price_to_pips(price_distance: float, symbol: str) -> float:
    pip_size = get_pip_size(symbol)

    if pip_size <= 0:
        return 0

    return abs(price_distance) / pip_size


def pips_to_price(pips: float, symbol: str) -> float:
    return pips * get_pip_size(symbol)


# ============================================================
# TWELVE DATA
# ============================================================

def get_market_data(symbol: str) -> Optional[pd.DataFrame]:

    if not TWELVE_DATA_KEY:
        return None

    try:
        url = "https://api.twelvedata.com/time_series"

        params = {
            "symbol": symbol,
            "interval": TIMEFRAME,
            "outputsize": 200,
            "apikey": TWELVE_DATA_KEY,
        }

        response = requests.get(
            url,
            params=params,
            timeout=20,
        )

        data = response.json()

        if "values" not in data:
            logger.warning("Twelve Data error: %s", data)
            return None

        df = pd.DataFrame(data["values"])

        if df.empty:
            return None

        for col in ["open", "high", "low", "close"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.sort_values("datetime").reset_index(drop=True)

        df.dropna(
            subset=["open", "high", "low", "close"],
            inplace=True,
        )

        return df

    except Exception:
        logger.exception("Market data error")
        return None


# ============================================================
# TECHNICAL INDICATORS
# ============================================================

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    # EMA
    df["EMA20"] = df["close"].ewm(
        span=20,
        adjust=False,
    ).mean()

    df["EMA50"] = df["close"].ewm(
        span=50,
        adjust=False,
    ).mean()

    df["EMA200"] = df["close"].ewm(
        span=200,
        adjust=False,
    ).mean()

    # RSI
    delta = df["close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss.replace(0, pd.NA)

    df["RSI"] = 100 - (
        100 / (1 + rs)
    )

    # ATR
    previous_close = df["close"].shift(1)

    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - previous_close).abs()
    tr3 = (df["low"] - previous_close).abs()

    tr = pd.concat(
        [tr1, tr2, tr3],
        axis=1,
    ).max(axis=1)

    df["ATR"] = tr.rolling(14).mean()

    # MACD
    ema12 = df["close"].ewm(
        span=12,
        adjust=False,
    ).mean()

    ema26 = df["close"].ewm(
        span=26,
        adjust=False,
    ).mean()

    df["MACD"] = ema12 - ema26

    df["MACD_SIGNAL"] = df["MACD"].ewm(
        span=9,
        adjust=False,
    ).mean()

    return df


# ============================================================
# MARKET SIGNAL
# ============================================================

def calculate_signal(
    df: pd.DataFrame,
    symbol: str,
):

    df = calculate_indicators(df)

    latest = df.iloc[-1]

    price = float(latest["close"])
    ema20 = float(latest["EMA20"])
    ema50 = float(latest["EMA50"])
    ema200 = float(latest["EMA200"])
    rsi = float(latest["RSI"])
    atr = float(latest["ATR"])

    buy_score = 0
    sell_score = 0

    # EMA trend
    if ema20 > ema50:
        buy_score += 25

    if ema20 < ema50:
        sell_score += 25

    # Long-term trend
    if price > ema200:
        buy_score += 15

    if price < ema200:
        sell_score += 15

    # Price location
    if price > ema20:
        buy_score += 15

    if price < ema20:
        sell_score += 15

    # RSI
    if 52 <= rsi <= 68:
        buy_score += 15

    if 32 <= rsi <= 48:
        sell_score += 15

    # MACD
    if latest["MACD"] > latest["MACD_SIGNAL"]:
        buy_score += 15

    if latest["MACD"] < latest["MACD_SIGNAL"]:
        sell_score += 15

    # Momentum
    if df["close"].iloc[-1] > df["close"].iloc[-2]:
        buy_score += 15

    if df["close"].iloc[-1] < df["close"].iloc[-2]:
        sell_score += 15

    # --------------------------------------------------------
    # Confidence
    # --------------------------------------------------------

    if buy_score > sell_score:
        signal = "BUY"
        confidence = min(95, 50 + buy_score / 2)

    elif sell_score > buy_score:
        signal = "SELL"
        confidence = min(95, 50 + sell_score / 2)

    else:
        signal = "WAIT"
        confidence = 50

    confidence = round(confidence)

    # --------------------------------------------------------
    # Support / resistance
    # --------------------------------------------------------

    recent = df.tail(50)

    support = float(recent["low"].min())
    resistance = float(recent["high"].max())

    # --------------------------------------------------------
    # Make target at least 50 pips
    # --------------------------------------------------------

    minimum_price_distance = pips_to_price(
        MIN_TARGET_PIPS,
        symbol,
    )

    # ATR based target
    target_distance = max(
        atr * 2.0,
        minimum_price_distance,
    )

    stop_distance = max(
        atr * 1.0,
        pips_to_price(30, symbol),
    )

    # --------------------------------------------------------
    # Trade levels
    # --------------------------------------------------------

    if signal == "BUY":

        entry = price

        buy_limit = max(
            support,
            price - atr * 0.5,
        )

        sell_limit = None

        sl = entry - stop_distance

        tp1 = entry + target_distance
        tp2 = entry + target_distance * 1.6

    elif signal == "SELL":

        entry = price

        sell_limit = min(
            resistance,
            price + atr * 0.5,
        )

        buy_limit = None

        sl = entry + stop_distance

        tp1 = entry - target_distance
        tp2 = entry - target_distance * 1.6

    else:

        entry = price

        buy_limit = None
        sell_limit = None

        sl = None
        tp1 = None
        tp2 = None

    target_pips = price_to_pips(
        target_distance,
        symbol,
    )

    return {
        "signal": signal,
        "confidence": confidence,
        "price": price,
        "ema20": ema20,
        "ema50": ema50,
        "ema200": ema200,
        "rsi": rsi,
        "atr": atr,
        "support": support,
        "resistance": resistance,
        "entry": entry,
        "buy_limit": buy_limit,
        "sell_limit": sell_limit,
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,
        "target_pips": target_pips,
    }


# ============================================================
# FORMAT NUMBER
# ============================================================

def fmt(value):

    if value is None:
        return "N/A"

    if abs(value) >= 100:
        return f"{value:,.2f}"

    return f"{value:.5f}".rstrip("0").rstrip(".")


# ============================================================
# TEXT MARKET ANALYSIS
# ============================================================

def text_analysis(symbol: str):

    df = get_market_data(symbol)

    if df is None:

        return (
            f"❌ No live market data found for {symbol}.\n\n"
            "Check the symbol format or TWELVE_DATA_KEY."
        )

    if len(df) < 50:

        return (
            f"❌ Not enough market data for {symbol}."
        )

    try:

        result = calculate_signal(
            df,
            symbol,
        )

        signal = result["signal"]
        confidence = result["confidence"]

        if signal == "BUY":
            signal_text = "🟢 BUY"
            market = "🟢 GOOD / BULLISH"

        elif signal == "SELL":
            signal_text = "🔴 SELL"
            market = "🔴 BAD / BEARISH"

        else:
            signal_text = "🟡 WAIT"
            market = "🟡 UNCERTAIN"

        lines = [
            "📊 CRYPTO FLOW SIGNAL",
            "",
            f"SYMBOL: {symbol}",
            f"TIMEFRAME: {TIMEFRAME}",
            "",
            f"SIGNAL: {signal_text}",
            f"CONFIDENCE: {confidence}%",
            f"MARKET: {market}",
            "",
            f"ENTRY: {fmt(result['entry'])}",
        ]

        if signal == "BUY":

            lines.append(
                f"BUY LIMIT: {fmt(result['buy_limit'])}"
            )

            lines.append(
                "SELL LIMIT: N/A"
            )

        elif signal == "SELL":

            lines.append(
                "BUY LIMIT: N/A"
            )

            lines.append(
                f"SELL LIMIT: {fmt(result['sell_limit'])}"
            )

        else:

            lines.append("BUY LIMIT: N/A")
            lines.append("SELL LIMIT: N/A")

        if signal != "WAIT":

            lines.extend([
                "",
                f"SL: {fmt(result['sl'])}",
                f"TP1: {fmt(result['tp1'])}",
                f"TP2: {fmt(result['tp2'])}",
                "",
                f"🎯 TARGET: {result['target_pips']:.0f}+ PIPS",
            ])

        lines.extend([
            "",
            f"SUPPORT: {fmt(result['support'])}",
            f"RESISTANCE: {fmt(result['resistance'])}",
            "",
            f"RSI: {result['rsi']:.1f}",
            f"EMA20: {fmt(result['ema20'])}",
            f"EMA50: {fmt(result['ema50'])}",
            "",
            "⚠️ Educational analysis only.",
        ])

        return "\n".join(lines)

    except Exception as e:

        logger.exception("Text analysis error")

        return (
            "❌ Analysis error.\n"
            f"{str(e)[:500]}"
        )


# ============================================================
# GROQ IMAGE ANALYSIS
# ============================================================

def encode_image(image_bytes: bytes) -> str:

    return base64.b64encode(
        image_bytes
    ).decode("utf-8")


def get_image_mime(image_bytes: bytes) -> str:

    if image_bytes.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"

    if image_bytes.startswith(b"\x89PNG"):
        return "image/png"

    if image_bytes.startswith(b"GIF"):
        return "image/gif"

    if image_bytes.startswith(b"RIFF"):
        return "image/webp"

    return "image/jpeg"


def clean_ai_text(text: str) -> str:

    if not text:
        return "No analysis returned."

    # Remove markdown formatting that can cause Telegram entity errors
    text = text.replace("```", "")
    text = text.replace("**", "")
    text = text.replace("__", "")

    return text.strip()


def analyze_chart_with_groq(
    image_bytes: bytes,
    caption: str = "",
) -> str:

    if not groq_client:
        return "❌ GROQ_API_KEY is missing."

    base64_image = encode_image(
        image_bytes
    )

    mime = get_image_mime(
        image_bytes
    )

    prompt = """
Analyze this trading chart.

Return ONLY this short format.
Do NOT explain the reasoning.
Do NOT use markdown.
Do NOT use tables.

SIGNAL: BUY or SELL or WAIT
CONFIDENCE: number%
SYMBOL: symbol
TIMEFRAME: timeframe
ENTRY: price
BUY LIMIT: price or N/A
SELL LIMIT: price or N/A
SL: price or N/A
TP1: price or N/A
TP2: price or N/A
TARGET PIPS: number
SUPPORT: price
RESISTANCE: price
MARKET: GOOD or BAD or UNCERTAIN
REASON: one short sentence

Important:
- Give a clear BUY, SELL, or WAIT.
- Do not write "Buy on Pullback" instead of BUY.
- Do not write "Sell on Breakout" instead of SELL.
- TARGET PIPS must be at least 50 when a BUY or SELL setup is identified.
- For XAUUSD, 0.01 price movement = 1 pip.
- If indicators are not visible, do not invent them.
- Keep the complete answer under 600 tokens.
"""

    if caption:
        prompt += (
            "\nUser caption/symbol: "
            + caption[:100]
        )

    try:

        response = groq_client.chat.completions.create(
            model=GROQ_VISION_MODEL,

            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a concise chart-analysis assistant. "
                        "Return only the requested fields."
                    ),
                },
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
                                "url": (
                                    f"data:{mime};base64,"
                                    f"{base64_image}"
                                )
                            },
                        },
                    ],
                },
            ],

            temperature=0.2,

            # Important: keep below the user's current 1000-token
            # output-per-minute limit.
            max_completion_tokens=700,

            top_p=0.8,

            stream=False,
        )

        result = response.choices[0].message.content

        return clean_ai_text(result)

    except Exception as e:

        logger.exception("Groq chart analysis error")

        error_text = str(e)

        if "429" in error_text:

            return (
                "❌ Groq rate limit reached.\n\n"
                "The request was too large. "
                "The bot is already configured with a smaller "
                "output limit. Wait a little and send the chart again."
            )

        if "model_not_found" in error_text:

            return (
                "❌ Groq model error.\n\n"
                "Check GROQ_API_KEY and the current Groq model access."
            )

        return (
           
