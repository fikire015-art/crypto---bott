import os
import io
import math
import base64
import threading
import logging
from datetime import datetime

import requests
import pandas as pd
from flask import Flask, jsonify

from groq import Groq

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

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TWELVE_DATA_KEY = os.getenv("TWELVE_DATA_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

PORT = int(os.getenv("PORT", "10000"))

# Main analysis timeframes
TIMEFRAMES = {
    "5m": "5min",
    "15m": "15min",
    "1h": "1h",
}

DEFAULT_TIMEFRAME = "15m"

# Minimum score for "GOOD MARKET CONDITION"
MIN_CONFIDENCE = 70

# Minimum target distance
MIN_TARGET_PIPS = 50

# XAUUSD
# 0.01 price movement = 1 pip
XAU_PIP_SIZE = 0.01

# Market data cache
CACHE_SECONDS = 45
DATA_CACHE = {}

# Groq image output limit
# Your current account showed OTPM limit = 1000.
# Keep this below 1000.
GROQ_MAX_OUTPUT_TOKENS = 800


# =========================================================
# LOGGING
# =========================================================

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)


# =========================================================
# FLASK / RENDER HEALTH SERVER
# =========================================================

app = Flask(__name__)


@app.route("/")
def home():
    return "Crypto Flow Bot is running."


@app.route("/health")
def health():
    return jsonify(
        {
            "status": "online",
            "bot": "Crypto Flow Bot",
            "time": datetime.utcnow().isoformat(),
        }
    )


# =========================================================
# SYMBOL HELPERS
# =========================================================

def normalize_symbol(symbol: str) -> str:

    symbol = symbol.strip().upper()

    replacements = {
        "XAUUSD": "XAU/USD",
        "XAU/USD": "XAU/USD",
        "GOLD": "XAU/USD",

        "BTC": "BTC/USD",
        "BTCUSD": "BTC/USD",

        "ETH": "ETH/USD",
        "ETHUSD": "ETH/USD",

        "BNB": "BNB/USD",
        "SOL": "SOL/USD",
        "XRP": "XRP/USD",
        "ADA": "ADA/USD",
        "DOGE": "DOGE/USD",
    }

    return replacements.get(symbol, symbol)


def pip_size(symbol: str) -> float:

    symbol = symbol.upper()

    if "XAU" in symbol or "GOLD" in symbol:
        return 0.01

    if "JPY" in symbol:
        return 0.01

    # Forex
    if "/" in symbol and not symbol.startswith(
        ("BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "DOGE")
    ):
        return 0.0001

    # Crypto
    return 0.0001


def price_decimals(symbol: str) -> int:

    symbol = symbol.upper()

    if "JPY" in symbol:
        return 3

    if "XAU" in symbol or "GOLD" in symbol:
        return 2

    if symbol.startswith(
        ("BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "DOGE")
    ):
        return 4

    return 5


def fmt_price(value: float, symbol: str) -> str:

    decimals = price_decimals(symbol)

    return f"{value:.{decimals}f}"


def calculate_pips(
    price1: float,
    price2: float,
    symbol: str,
) -> float:

    size = pip_size(symbol)

    return abs(price2 - price1) / size


# =========================================================
# TWELVE DATA
# =========================================================

def get_market_data(
    symbol: str,
    interval: str = "15min",
    outputsize: int = 200,
):

    if not TWELVE_DATA_KEY:
        raise RuntimeError(
            "TWELVE_DATA_KEY is missing."
        )

    cache_key = f"{symbol}_{interval}"

    cached = DATA_CACHE.get(cache_key)

    if cached:

        age = (
            datetime.utcnow()
            - cached["time"]
        ).total_seconds()

        if age < CACHE_SECONDS:
            return cached["data"].copy()

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
            timeout=20,
        )

        response.raise_for_status()

    except requests.RequestException as e:

        raise RuntimeError(
            f"Twelve Data request failed: {e}"
        )

    data = response.json()

    if "values" not in data:

        message = data.get(
            "message",
            "No market data returned."
        )

        raise RuntimeError(message)

    df = pd.DataFrame(data["values"])

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
    ]

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df["datetime"] = pd.to_datetime(
        df["datetime"],
        errors="coerce",
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
    ).reset_index(drop=True)

    if len(df) < 50:

        raise RuntimeError(
            "Not enough market candles returned."
        )

    DATA_CACHE[cache_key] = {
        "time": datetime.utcnow(),
        "data": df.copy(),
    }

    return df


# =========================================================
# INDICATORS
# =========================================================

def calculate_indicators(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df = df.copy()

    # -----------------------------------------------------
    # EMA 20
    # -----------------------------------------------------

    df["EMA20"] = df["close"].ewm(
        span=20,
        adjust=False,
    ).mean()

    # -----------------------------------------------------
    # EMA 50
    # -----------------------------------------------------

    df["EMA50"] = df["close"].ewm(
        span=50,
        adjust=False,
    ).mean()

    # -----------------------------------------------------
    # EMA 200
    # -----------------------------------------------------

    df["EMA200"] = df["close"].ewm(
        span=200,
        adjust=False,
    ).mean()

    # -----------------------------------------------------
    # RSI
    # -----------------------------------------------------

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

    rs = (
        avg_gain /
        avg_loss.replace(
            0,
            math.nan,
        )
    )

    df["RSI"] = 100 - (
        100 /
        (1 + rs)
    )

    # -----------------------------------------------------
    # MACD
    # -----------------------------------------------------

    ema12 = df["close"].ewm(
        span=12,
        adjust=False,
    ).mean()

    ema26 = df["close"].ewm(
        span=26,
        adjust=False,
    ).mean()

    df["MACD"] = (
        ema12 - ema26
    )

    df["MACD_SIGNAL"] = (
        df["MACD"].ewm(
            span=9,
            adjust=False,
        ).mean()
    )

    df["MACD_HIST"] = (
        df["MACD"]
        - df["MACD_SIGNAL"]
    )

    # -----------------------------------------------------
    # ATR
    # -----------------------------------------------------

    previous_close = (
        df["close"].shift(1)
    )

    tr1 = (
        df["high"]
        - df["low"]
    )

    tr2 = (
        df["high"]
        - previous_close
    ).abs()

    tr3 = (
        df["low"]
        - previous_close
    ).abs()

    tr = pd.concat(
        [
            tr1,
            tr2,
            tr3,
        ],
        axis=1,
    ).max(axis=1)

    df["ATR"] = tr.rolling(
        14
    ).mean()

    return df


# =========================================================
# SUPPORT / RESISTANCE
# =========================================================

def support_resistance(
    df: pd.DataFrame,
):

    recent = df.tail(50)

    support = float(
        recent["low"].min()
    )

    resistance = float(
        recent["high"].max()
    )

    return support, resistance


# =========================================================
# MARKET STRUCTURE
# =========================================================

def market_structure(
    df: pd.DataFrame,
):

    recent = df.tail(20)

    highs = recent[
        "high"
    ].values

    lows = recent[
        "low"
    ].values

    if len(highs) < 6:
        return "UNKNOWN"

    last_high = highs[-1]
    previous_high = highs[-4]

    last_low = lows[-1]
    previous_low = lows[-4]

    if (
        last_high > previous_high
        and last_low > previous_low
    ):
        return "HH / HL"

    if (
        last_high < previous_high
        and last_low < previous_low
    ):
        return "LH / LL"

    return "RANGE"


# =========================================================
# CANDLESTICK
# =========================================================

def candle_analysis(
    df: pd.DataFrame,
):

    last = df.iloc[-1]

    body = abs(
        last["close"]
        - last["open"]
    )

    upper_wick = (
        last["high"]
        - max(
            last["open"],
            last["close"],
        )
    )

    lower_wick = (
        min(
            last["open"],
            last["close"],
        )
        - last["low"]
    )

    if body == 0:
        return "DOJI"

    if (
        lower_wick > body * 2
        and last["close"] > last["open"]
    ):
        return "BULLISH REJECTION"

    if (
        upper_wick > body * 2
        and last["close"] < last["open"]
    ):
        return "BEARISH REJECTION"

    if last["close"] > last["open"]:
        return "BULLISH CANDLE"

    return "BEARISH CANDLE"


# =========================================================
# MARKET ANALYSIS
# =========================================================

def analyze_market(
    df: pd.DataFrame,
    symbol: str,
):

    df = calculate_indicators(df)

    last = df.iloc[-1]

    previous = df.iloc[-2]

    price = float(
        last["close"]
    )

    ema20 = float(
        last["EMA20"]
    )

    ema50 = float(
        last["EMA50"]
    )

    ema200 = float(
        last["EMA200"]
    )

    rsi = float(
        last["RSI"]
    )

    macd = float(
        last["MACD"]
    )

    macd_signal = float(
        last["MACD_SIGNAL"]
    )

    atr = float(
        last["ATR"]
    )

    # Safety for NaN
    if math.isnan(rsi):
        rsi = 50.0

    if math.isnan(atr) or atr <= 0:
        atr = abs(price * 0.005)

    buy_score = 0
    sell_score = 0

    # -----------------------------------------------------
    # EMA TREND
    # -----------------------------------------------------

    if ema20 > ema50:
        buy_score += 20

    elif ema20 < ema50:
        sell_score += 20

    # -----------------------------------------------------
    # EMA 200
    # -----------------------------------------------------

    if price > ema200:
        buy_score += 15

    elif price < ema200:
        sell_score += 15

    # -----------------------------------------------------
    # PRICE VS EMA20
    # -----------------------------------------------------

    if price > ema20:
        buy_score += 10

    elif price < ema20:
        sell_score += 10

    # -----------------------------------------------------
    # RSI
    # -----------------------------------------------------

    if 52 <= rsi <= 70:
        buy_score += 15

    elif 30 <= rsi <= 48:
        sell_score += 15

    # -----------------------------------------------------
    # MACD
    # -----------------------------------------------------

    if macd > macd_signal:
        buy_score += 15

    elif macd < macd_signal:
        sell_score += 15

    # -----------------------------------------------------
    # MOMENTUM
    # -----------------------------------------------------

    if price > float(
        previous["close"]
    ):
        buy_score += 10

    elif price < float(
        previous["close"]
    ):
        sell_score += 10

    # -----------------------------------------------------
    # MARKET STRUCTURE
    # -----------------------------------------------------

    structure = market_structure(
        df
    )

    if structure == "HH / HL":
        buy_score += 15

    elif structure == "LH / LL":
        sell_score += 15

    # -----------------------------------------------------
    # DECISION
    # -----------------------------------------------------

    if buy_score > sell_score:

        direction = "BUY"

    elif sell_score > buy_score:

        direction = "SELL"

    else:

        direction = "WAIT"

    confidence = max(
        buy_score,
        sell_score,
    )

    confidence = min(
        confidence,
        100,
    )

    # -----------------------------------------------------
    # SUPPORT / RESISTANCE
    # -----------------------------------------------------

    support, resistance = (
        support_resistance(df)
    )

    candle = candle_analysis(
        df
    )

    # -----------------------------------------------------
    # ENTRY / SL / TP
    # -----------------------------------------------------

    min_distance = (
        MIN_TARGET_PIPS
        * pip_size(symbol)
    )

    atr_distance = (
        atr * 1.5
    )

    distance = max(
        min_distance,
        atr_distance,
    )

    if direction == "BUY":

        entry = price

        sl = (
            entry - distance
        )

        tp1 = (
            entry
            + distance * 1.5
        )

        tp2 = (
            entry
            + distance * 2.5
        )

        buy_limit = max(
            support,
            entry
            - distance * 0.35,
        )

        sell_limit = None

    elif direction == "SELL":

        entry = price

        sl = (
            entry + distance
        )

        tp1 = (
            entry
            - distance * 1.5
        )

        tp2 = (
            entry
            - distance * 2.5
        )

        sell_limit = min(
            resistance,
            entry
            + distance * 0.35,
        )

        buy_limit = None

    else:

        entry = price

        sl = None
        tp1 = None
        tp2 = None

        buy_limit = support
        sell_limit = resistance

    return {
        "symbol": symbol,
        "price": price,
        "direction": direction,
        "confidence": confidence,
        "structure": structure,
        "candle": candle,
        "support": support,
        "resistance": resistance,
        "ema20": ema20,
        "ema50": ema50,
        "ema200": ema200,
        "rsi": rsi,
        "macd": macd,
        "macd_signal": macd_signal,
        "atr": atr,
        "entry": entry,
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,
        "buy_limit": buy_limit,
        "sell_limit": sell_limit,
        "distance": distance,
        "target_pips": (
            distance
            / pip_size(symbol)
        ),
    }


# =========================================================
# FORMAT MARKET SIGNAL
# =========================================================

def format_signal(
    result,
):

    symbol = result["symbol"]

    direction = result["direction"]

    confidence = result["confidence"]

    if direction == "BUY":
        emoji = "🟢"

    elif direction == "SELL":
        emoji = "🔴"

    else:
        emoji = "🟡"

    text = (
        "📊 MARKET ANALYSIS\n\n"
        f"💱 Symbol: {symbol}\n"
        f"{emoji} Signal: {direction}\n"
        f"🎯 Confidence Score: {confidence}%\n\n"
    )

    text += (
        f"📌 Market Structure: "
        f"{result['structure']}\n"
        f"🕯 Candle: "
        f"{result['candle']}\n\n"
    )

    text += (
        f"📈 EMA20: "
        f"{fmt_price(result['ema20'], symbol)}\n"
        f"📈 EMA50: "
        f"{fmt_price(result['ema50'], symbol)}\n"
        f"📈 EMA200: "
        f"{fmt_price(result['ema200'], symbol)}\n"
        f"📊 RSI: "
        f"{result['rsi']:.1f}\n"
        f"📉 MACD: "
        f"{result['macd']:.5f}\n\n"
    )

    text += (
        f"🧱 Support: "
        f"{fmt_price(result['support'], symbol)}\n"
        f"🧱 Resistance: "
        f"{fmt_price(result['resistance'], symbol)}\n\n"
    )

    if direction == "BUY":

        text += (
            f"🎯 Entry: "
            f"{fmt_price(result['entry'], symbol)}\n"
            f"🟢 Buy Limit: "
            f"{fmt_price(result['buy_limit'], symbol)}\n"
            f"🛑 Stop Loss: "
            f"{fmt_price(result['sl'], symbol)}\n"
            f"🎯 TP1: "
            f"{fmt_price(result['tp1'], symbol)}\n"
            f"🎯 TP2: "
            f"{fmt_price(result['tp2'], symbol)}\n\n"
        )

    elif direction == "SELL":

        text += (
            f"🎯 Entry: "
            f"{fmt_price(result['entry'], symbol)}\n"
            f"🔴 Sell Limit: "
            f"{fmt_price(result['sell_limit'], symbol)}\n"
            f"🛑 Stop Loss: "
            f"{fmt_price(result['sl'], symbol)}\n"
            f"🎯 TP1: "
            f"{fmt_price(result['tp1'], symbol)}\n"
            f"🎯 TP2: "
            f"{fmt_price(result['tp2'], symbol)}\n\n"
        )

    else:

        text += (
            "🟡 Market: WAIT\n"
            f"🟢 Buy Limit area: "
            f"{fmt_price(result['buy_limit'], symbol)}\n"
            f"🔴 Sell Limit area: "
            f"{fmt_price(result['sell_limit'], symbol)}\n\n"
        )

    text += (
        f"📏 Estimated target distance: "
        f"{result['target_pips']:.0f} pips\n\n"
    )

    if confidence >= MIN_CONFIDENCE:

        text += (
            "🟢 GOOD MARKET CONDITION\n"
            "Multiple indicators are aligned."
        )

    else:

        text += (
            "🟡 WEAK / UNCERTAIN CONDITION\n"
            "Wait for stronger confirmation."
        )

    text += (
        "\n\n⚠️ Educational market analysis only. "
        "Manage risk carefully."
    )

    return text


# =========================================================
# TELEGRAM START
# =========================================================

async def start(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    message = (
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

    await update.message.reply_text(
        message
    )


# =========================================================
# HELP
# =========================================================

async def help_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    text = (
        "📚 Commands\n\n"
        "/start - Start bot\n"
        "/help - Help\n"
        "/analyze XAUUSD - Analyze symbol\n\n"
        "You can also simply type:\n"
        "XAUUSD\n"
        "EUR/USD\n"
        "BTC/USD"
    )

    await update.message.reply_text(
        text
    )


# =========================================================
# ANALYZE COMMAND
# =========================================================

async def analyze_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    if not context.args:

        await update.message.reply_text(
            "Example: /analyze XAUUSD"
        )

        return

    symbol = normalize_symbol(
        context.args[0]
    )

    await analyze_and_reply(
        update,
        symbol,
    )


# =========================================================
# TEXT SYMBOL MESSAGE
# =========================================================

async def text_message(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    if not update.message:
        return

    text = (
        update.message.text
        or ""
    ).strip()

    if not text:
        return

    # Avoid treating long messages as symbols
    if len(text) > 30:
        return

    symbol = normalize_symbol(
        text
    )

    await analyze_and_reply(
        update,
        symbol,
    )


# =========================================================
# MULTI-TIMEFRAME ANALYSIS
# =========================================================

async def analyze_and_reply(
    update: Update,
    symbol: str,
):

    status = await update.message.reply_text(
        f"🔎 Analyzing {symbol}...\n"
        "5M + 15M + 1H data loading..."
    )

    try:

        # -------------------------------------------------
        # 15M MAIN
        # -------------------------------------------------

        df15 = get_market_data(
            symbol,
            "15min",
            200,
        )

        result = analyze_market(
            df15,
            symbol,
        )

        # -------------------------------------------------
        # 5M CONFIRMATION
        # -------------------------------------------------

        try:

            df5 = get_market_data(
                symbol,
                "5min",
                200,
            )

            result_5m = analyze_market(
                df5,
                symbol,
            )

            if (
                result["direction"]
                == result_5m["direction"]
                and result["direction"]
                != "WAIT"
            ):

                result["confidence"] = min(
                    100,
                    result["confidence"] + 5,
                )

        except Exception as e:

            logger.warning(
                "5M analysis failed: %s",
                e,
            )

        # -------------------------------------------------
        # 1H CONFIRMATION
        # -------------------------------------------------

        try:

            df1h = get_market_data(
                symbol,
                "1h",
                200,
            )

            result_1h = analyze_market(
                df1h,
                symbol,
            )

            if (
                result["direction"]
                == result_1h["direction"]
                and result["direction"]
                != "WAIT"
            ):

                result["confidence"] = min(
                    100,
                    result["confidence"] + 10,
                )

            elif (
                result_1h["direction"]
                != "WAIT"
                and result["direction"]
                != "WAIT"
            ):

                result["confidence"] = max(
                    0,
                    result["confidence"] - 10,
                )

        except Exception as e:

            logger.warning(
                "1H analysis failed: %s",
                e,
            )

        text = format_signal(
            result
        )

        await status.edit_text(
            text
        )

    except Exception as e:

        logger.exception(
            "Analysis error"
        )

        await status.edit_text(
            "❌ Analysis failed.\n\n"
            f"Reason: {str(e)[:500]}\n\n"
            "Check your Twelve Data API key "
            "and symbol format."
        )


# =========================================================
# GROQ AI CHART ANALYSIS
# =========================================================

def groq_analyze_image(
    image_bytes: bytes,
    symbol_hint: str = "",
):

    if not GROQ_API_KEY:

        return (
            "❌ GROQ_API_KEY is not configured."
        )

    client = Groq(
        api_key=GROQ_API_KEY
    )

    encoded = base64.b64encode(
        image_bytes
    ).decode("utf-8")

    prompt = f"""
You are a concise trading chart-analysis assistant.

Analyze the chart screenshot.

Symbol hint:
{symbol_hint or "unknown"}

Return ONLY a concise analysis.

Include:

1. Symbol and timeframe if visible
2. Trend
3. Market structure
4. Support
5. Resistance
6. EMA 20/50/200 if visible
7. RSI if visible
8. MACD if visible
9. Candlestick or breakout observation
10. BUY / SELL / WAIT
11. Possible entry area
12. Stop loss area
13. TP1
14. TP2
15. Confidence score from 0 to 100

IMPORTANT:
- Do not invent exact prices.
- If a value is not readable, say "Not visible".
- Keep the answer under 500 words.
- Use simple plain text.
- Do not use Markdown tables.
- Do not use unusual formatting.
- Do not use long explanations.

This is educational technical analysis only.
"""

    try:

        completion = (
            client.chat.completions.create(
                model="qwen/qwen3.8-27b",

                messages=[
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
                                        "data:image/jpeg;base64,"
                                        + encoded
                                    )
                                },
                            },
                        ],
                    }
                ],

                temperature=0.7,

                top_p=0.80,

                # IMPORTANT:
                # Keep below current 1000 OTPM limit.
                max_completion_tokens=800,

                # Efficient non-thinking mode.
                reasoning_effort="none",

                stream=False,
            )
        )

        result = (
            completion
            .choices[0]
            .message
            .content
        )

        if not result:

            return (
                "❌ Groq returned an empty response."
            )

        return result

    except Exception as e:

        error_text = str(e)

        logger.exception(
            "Groq image analysis failed"
        )

        # Friendly 429 message
        if (
            "429" in error_text
            or "rate_limit" in error_text
            or "rate limit" in error_text.lower()
        ):

            return (
                "❌ Groq rate limit reached.\n\n"
                "The screenshot was received, "
                "but Groq temporarily limited "
                "the request.\n\n"
                "Please wait about one minute "
                "and try the screenshot again."
            )

        return (
            "❌ Groq chart analysis error.\n\n"
            + error_text[:700]
        )


# =========================================================
# TELEGRAM PHOTO HANDLER
# =========================================================

async def photo_handler(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    if not update.message:
        return

    if not update.message.photo:
        return

    wait = await update.message.reply_text(
        "📸 Chart received.\n"
        "🤖 Groq AI is analyzing the chart..."
    )

    try:

        photo = (
            update.message.photo[-1]
        )

        file = await context.bot.get_file(
            photo.file_id
        )

        buffer = io.BytesIO()

        await file.download_to_memory(
            buffer
        )

        image_bytes = (
            buffer.getvalue()
        )

        caption = (
            update.message.caption
            or ""
        )

        result = groq_analyze_image(
            image_bytes,
            caption,
        )

        # IMPORTANT:
        # No parse_mode here.
        # This prevents Telegram
        # "Can't parse entities" errors.

        final_text = (
            "📊 AI CHART ANALYSIS\n\n"
            + result
            + "\n\n"
            "⚠️ Educational analysis only."
        )

        await wait.edit_text(
            final_text
        )

    except Exception as e:

        logger.exception(
            "Image analysis error"
        )

        await wait.edit_text(
            "❌ Chart analysis failed.\n\n"
            f"{str(e)[:700]}"
        )


# =========================================================
# TELEGRAM ERROR HANDLER
# =========================================================

async def error_handler(
    update: object,
    context: ContextTypes.DEFAULT_TYPE,
):

    logger.error(
        "Telegram error: %s",
        context.error,
    )


# =========================================================
# RENDER WEB SERVER
# =========================================================

def run_flask():

    app.run(
        host="0.0.0.0",
        port=PORT,
        use_reloader=False,
    )


# =========================================================
# MAIN
# =========================================================

def main():

    if not TELEGRAM_BOT_TOKEN:

        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN is missing."
        )

    if not TWELVE_DATA_KEY:

        logger.warning(
            "TWELVE_DATA_KEY is missing. "
            "Market analysis will not work."
        )

    if not GROQ_API_KEY:

        logger.warning(
            "GROQ_API_KEY is missing. "
            "Screenshot analysis will not work."
        )

    # -----------------------------------------------------
    # Render health server
    # -----------------------------------------------------

    flask_thread = threading.Thread(
        target=run_flask,
        daemon=True,
    )

    flask_thread.start()

    # -----------------------------------------------------
    # Telegram application
    # -----------------------------------------------------

    application = (
        Application.builder()
        .token(
            TELEGRAM_BOT_TOKEN
        )
        .build()
    )

    # /start
    application.add_handler(
        CommandHandler(
            "start",
            start,
        )
    )

    # /help
    application.add_handler(
        CommandHandler(
            "help",
            help_command,
        )
    )

    # /analyze
    application.add_handler(
        CommandHandler(
            "analyze",
            analyze_command,
        )
    )

    # Photos
    application.add_handler(
        MessageHandler(
            filters.PHOTO,
            photo_handler,
        )
    )

    # Text symbols
    application.add_handler(
        MessageHandler(
            filters.TEXT
            & ~filters.COMMAND,
            text_message,
        )
    )

    # Errors
    application.add_error_handler(
        error_handler
    )

    logger.info(
        "Crypto Flow Bot starting..."
    )

    # Start Telegram polling
    application.run_polling(
        drop_pending_updates=True
    )


# =========================================================
# START
# =========================================================

if __name__ == "__main__":
    main()
