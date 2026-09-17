import os
import time
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import requests
import pandas as pd
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
)

# =========================================================
# CONFIG
# =========================================================

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
TWELVE_DATA_KEY = os.getenv("TWELVE_DATA_KEY")

MIN_TARGET_PIPS = 100
CANDLE_LIMIT = 200
SCAN_DELAY = 60

TIMEFRAMES = {
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1H": "1h",
    "4H": "4h",
}

# =========================================================
# SYMBOLS
# =========================================================

SYMBOLS = [
    "XAU/USD",
    "XAG/USD",

    "EUR/USD",
    "GBP/USD",
    "USD/JPY",
    "USD/CHF",
    "AUD/USD",
    "NZD/USD",
    "USD/CAD",

    "EUR/GBP",
    "EUR/JPY",
    "EUR/CHF",
    "GBP/JPY",
    "GBP/CHF",
    "AUD/JPY",
    "CAD/JPY",
    "CHF/JPY",

    "BTC/USDT",
    "ETH/USDT",
    "BNB/USDT",
    "SOL/USDT",
    "XRP/USDT",
]

# =========================================================
# HEALTH SERVER FOR RENDER
# =========================================================

class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(b"CryptoFlowBot is running")

    def log_message(self, format, *args):
        return


def start_health_server():
    port = int(os.environ.get("PORT", 10000))
    server = HTTPServer(("0.0.0.0", port), HealthHandler)
    server.serve_forever()


# =========================================================
# DATA
# =========================================================

def get_binance_data(symbol, interval, limit=CANDLE_LIMIT):
    try:
        pair = symbol.replace("/", "")

        url = "https://api.binance.com/api/v3/klines"

        params = {
            "symbol": pair,
            "interval": interval,
            "limit": limit
        }

        r = requests.get(url, params=params, timeout=15)
        data = r.json()

        if not isinstance(data, list):
            return None

        df = pd.DataFrame(data, columns=[
            "time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_volume",
            "trades",
            "buy_volume",
            "buy_quote_volume",
            "ignore"
        ])

        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col])

        return df

    except Exception as e:
        print("Binance error:", symbol, e)
        return None


def get_twelve_data(symbol, interval, limit=CANDLE_LIMIT):
    try:
        url = "https://api.twelvedata.com/time_series"

        params = {
            "symbol": symbol,
            "interval": interval,
            "outputsize": limit,
            "apikey": TWELVE_DATA_KEY,
            "format": "JSON"
        }

        r = requests.get(url, params=params, timeout=20)
        data = r.json()

        if "values" not in data:
            print("Twelve Data error:", symbol, data)
            return None

        df = pd.DataFrame(data["values"])

        df["datetime"] = pd.to_datetime(df["datetime"])

        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.sort_values("datetime").reset_index(drop=True)

        return df

    except Exception as e:
        print("Twelve Data error:", symbol, e)
        return None


def get_data(symbol, timeframe):
    interval = TIMEFRAMES[timeframe]

    if symbol.endswith("/USDT"):
        return get_binance_data(symbol, interval)

    return get_twelve_data(symbol, interval)


# =========================================================
# LIVE PRICE
# =========================================================

def get_live_price(symbol):
    try:
        if symbol.endswith("/USDT"):
            pair = symbol.replace("/", "")

            url = "https://api.binance.com/api/v3/ticker/price"

            r = requests.get(
                url,
                params={"symbol": pair},
                timeout=10
            )

            data = r.json()
            return float(data["price"])

        url = "https://api.twelvedata.com/price"

        params = {
            "symbol": symbol,
            "apikey": TWELVE_DATA_KEY
        }

        r = requests.get(url, params=params, timeout=15)
        data = r.json()

        if "price" in data:
            return float(data["price"])

    except Exception as e:
        print("Live price error:", symbol, e)

    return None


# =========================================================
# INDICATORS
# =========================================================

def add_indicators(df):

    df = df.copy()

    df["EMA20"] = df["close"].ewm(
        span=20,
        adjust=False
    ).mean()

    df["EMA50"] = df["close"].ewm(
        span=50,
        adjust=False
    ).mean()

    df["EMA200"] = df["close"].ewm(
        span=200,
        adjust=False
    ).mean()

    delta = df["close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss.replace(0, 1e-10)

    df["RSI"] = 100 - (100 / (1 + rs))

    high_low = df["high"] - df["low"]

    high_close = (
        df["high"] - df["close"].shift()
    ).abs()

    low_close = (
        df["low"] - df["close"].shift()
    ).abs()

    tr = pd.concat(
        [high_low, high_close, low_close],
        axis=1
    ).max(axis=1)

    df["ATR"] = tr.rolling(14).mean()

    ema12 = df["close"].ewm(
        span=12,
        adjust=False
    ).mean()

    ema26 = df["close"].ewm(
        span=26,
        adjust=False
    ).mean()

    df["MACD"] = ema12 - ema26

    df["MACD_SIGNAL"] = df["MACD"].ewm(
        span=9,
        adjust=False
    ).mean()

    return df


# =========================================================
# PIP SIZE
# =========================================================

def pip_size(symbol):

    if symbol.startswith("XAU"):
        return 0.01

    if symbol.startswith("XAG"):
        return 0.01

    if "JPY" in symbol:
        return 0.01

    if symbol.endswith("/USDT"):
        return None

    return 0.0001


# =========================================================
# FORMAT PRICE
# =========================================================

def price_digits(symbol):

    if symbol.startswith("XAU"):
        return 2

    if symbol.startswith("XAG"):
        return 2

    if "JPY" in symbol:
        return 3

    if symbol.endswith("/USDT"):
        return 2

    return 5


def fmt(symbol, value):

    if value is None:
        return "N/A"

    return f"{value:.{price_digits(symbol)}f}"


# =========================================================
# ANALYSIS
# =========================================================

def analyze_market(symbol, timeframe):

    df = get_data(symbol, timeframe)

    if df is None or len(df) < 80:
        return None

    df = add_indicators(df)

    # -----------------------------------------------------
    # IMPORTANT:
    # -1 = current/in-progress candle
    # -2 = last CLOSED candle
    # -3 = previous CLOSED candle
    # -----------------------------------------------------

    last = df.iloc[-2]
    prev = df.iloc[-3]

    live_price = get_live_price(symbol)

    if live_price is None:
        live_price = float(df.iloc[-1]["close"])

    close = float(last["close"])

    ema20 = float(last["EMA20"])
    ema50 = float(last["EMA50"])
    ema200 = float(last["EMA200"])

    rsi = float(last["RSI"])
    atr = float(last["ATR"])

    macd = float(last["MACD"])
    macd_signal = float(last["MACD_SIGNAL"])

    prev_close = float(prev["close"])

    buy_score = 0
    sell_score = 0

    # -----------------------------------------------------
    # TREND
    # -----------------------------------------------------

    if ema20 > ema50:
        buy_score += 20
    else:
        sell_score += 20

    if ema50 > ema200:
        buy_score += 15
    else:
        sell_score += 15

    # -----------------------------------------------------
    # PRICE POSITION
    # -----------------------------------------------------

    if close > ema20:
        buy_score += 10
    else:
        sell_score += 10

    # -----------------------------------------------------
    # RSI
    # -----------------------------------------------------

    if 50 <= rsi <= 68:
        buy_score += 15

    elif 32 <= rsi < 50:
        sell_score += 15

    elif rsi > 70:
        sell_score += 5

    elif rsi < 30:
        buy_score += 5

    # -----------------------------------------------------
    # MACD
    # -----------------------------------------------------

    if macd > macd_signal:
        buy_score += 15
    else:
        sell_score += 15

    # -----------------------------------------------------
    # MOMENTUM
    # -----------------------------------------------------

    if close > prev_close:
        buy_score += 10
    else:
        sell_score += 10

    # -----------------------------------------------------
    # CANDLE DIRECTION
    # -----------------------------------------------------

    if last["close"] > last["open"]:
        buy_score += 5
    else:
        sell_score += 5

    # -----------------------------------------------------
    # DIRECTION
    # -----------------------------------------------------

    if buy_score > sell_score:
        direction = "BUY"
        score = buy_score
    else:
        direction = "SELL"
        score = sell_score

    # -----------------------------------------------------
    # CONFIDENCE
    # -----------------------------------------------------

    confidence = min(int(score), 100)

    if confidence >= 75:
        status = "STRONG"

    elif confidence >= 65:
        status = "GOOD"

    elif confidence >= 55:
        status = "UNCERTAIN"

    else:
        status = "NO TRADE"

    # -----------------------------------------------------
    # ENTRY / SL
    # -----------------------------------------------------

    if direction == "BUY":

        live_entry = live_price

        sl = live_entry - (atr * 1.5)

        future_entry = live_entry - (atr * 0.35)

        future_sl = future_entry - (atr * 1.5)

    else:

        live_entry = live_price

        sl = live_entry + (atr * 1.5)

        future_entry = live_entry + (atr * 0.35)

        future_sl = future_entry + (atr * 1.5)

    # -----------------------------------------------------
    # TARGET
    # -----------------------------------------------------

    # Minimum target = more than 100 pips
    target_distance = atr * 3.0

    pip = pip_size(symbol)

    if pip is not None:

        minimum_distance = pip * (MIN_TARGET_PIPS + 1)

        target_distance = max(
            target_distance,
            minimum_distance
        )

    # -----------------------------------------------------
    # LIVE TP
    # -----------------------------------------------------

    if direction == "BUY":

        live_tp = live_entry + target_distance

        future_target = future_entry + target_distance

    else:

        live_tp = live_entry - target_distance

        future_target = future_entry - target_distance

    # -----------------------------------------------------
    # PIPS
    # -----------------------------------------------------

    if pip is not None:

        target_pips = int(
            target_distance / pip
        )

    else:

        target_pips = None

    # -----------------------------------------------------
    # FUTURE SETUP
    # -----------------------------------------------------

    future_type = "PULLBACK"

    # If price is breaking recent structure,
    # use breakout as the future scenario.

    recent_high = df["high"].iloc[-22:-2].max()
    recent_low = df["low"].iloc[-22:-2].min()

    if direction == "BUY" and close > recent_high:
        future_type = "BREAKOUT"

    elif direction == "SELL" and close < recent_low:
        future_type = "BREAKOUT"

    # -----------------------------------------------------
    # RESULT
    # -----------------------------------------------------

    return {
        "symbol": symbol,
        "timeframe": timeframe,

        "direction": direction,
        "status": status,
        "confidence": confidence,

        "live_price": live_price,

        "live_entry": live_entry,
        "live_sl": sl,
        "live_tp": live_tp,

        "future_type": future_type,
        "future_entry": future_entry,
        "future_sl": future_sl,
        "future_target": future_target,

        "target_distance": target_distance,
        "target_pips": target_pips,

        "rsi": rsi,
        "atr": atr,
    }


# =========================================================
# TELEGRAM MESSAGE
# =========================================================

def signal_message(result):

    if result is None:
        return "❌ Analysis data unavailable."

    symbol = result["symbol"]
    tf = result["timeframe"]

    direction = result["direction"]
    status = result["status"]
    confidence = result["confidence"]

    target_pips = result["target_pips"]

    if target_pips is None:
        target_text = (
            f"{result['target_distance']:.2f} price distance"
        )
    else:
        target_text = f"{target_pips} pips"

    text = (
        f"📊 {symbol} — {tf}\n"
        f"━━━━━━━━━━━━━━\n"
        f"🚨 LIVE {direction}\n"
        f"Entry: {fmt(symbol, result['live_entry'])}\n"
        f"SL: {fmt(symbol, result['live_sl'])}\n"
        f"TP: {fmt(symbol, result['live_tp'])}\n\n"

        f"🔮 FUTURE TARGET\n"
        f"Type: {result['future_type']}\n"
        f"Future Entry: {fmt(symbol, result['future_entry'])}\n"
        f"Future SL: {fmt(symbol, result['future_sl'])}\n"
        f"Future Target: {fmt(symbol, result['future_target'])}\n"
        f"Target: {target_text}\n\n"

        f"🎯 Confidence: {confidence}%\n"
        f"📌 Status: {status}\n"
        f"📈 RSI: {result['rsi']:.1f}\n"
        f"━━━━━━━━━━━━━━\n"
        f"⚠️ Signal is analysis, not a guaranteed prediction."
    )

    return text


# =========================================================
# /START
# =========================================================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):

    message = (
        "🤖 CryptoFlowBot\n\n"
        "Welcome.\n\n"
        "Commands:\n"
        "/analyze XAU/USD\n"
        "/scan\n"
        "/all\n\n"
        "Example:\n"
        "/analyze XAU/USD"
    )

    await update.message.reply_text(message)


# =========================================================
# /ANALYZE
# =========================================================

async def analyze_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    if not context.args:

        await update.message.reply_text(
            "Example:\n/analyze XAU/USD"
        )

        return

    symbol = context.args[0].upper()

    if symbol not in SYMBOLS:

        await update.message.reply_text(
            "❌ Symbol not supported."
        )

        return

    await update.message.reply_text(
        f"🔎 Analyzing {symbol}..."
    )

    messages = []

    for tf in TIMEFRAMES:

        result = analyze_market(
            symbol,
            tf
        )

        if result:

            messages.append(
                signal_message(result)
            )

    if not messages:

        await update.message.reply_text(
            "❌ No analysis data available."
        )

        return

    for msg in messages:

        await update.message.reply_text(
            msg
        )


# =========================================================
# /SCAN
# =========================================================

async def scan_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    await update.message.reply_text(
        "🔎 Scanning all symbols and timeframes..."
    )

    count = 0

    for symbol in SYMBOLS:

        for tf in TIMEFRAMES:

            result = analyze_market(
                symbol,
                tf
            )

            if result is None:
                continue

            # Send only usable setups
            if result["confidence"] >= 65:

                await update.message.reply_text(
                    signal_message(result)
                )

                count += 1

            time.sleep(0.2)

    await update.message.reply_text(
        f"✅ Scan complete.\n"
        f"Signals found: {count}"
    )


# =========================================================
# BACKGROUND SCANNER
# =========================================================

def background_scanner(application):

    while True:

        try:

            if TELEGRAM_CHAT_ID:

                for symbol in SYMBOLS:

                    # Main scanning timeframe
                    result = analyze_market(
                        symbol,
                        "15m"
                    )

                    if result is None:
                        continue

                    if result["confidence"] >= 75:

                        text = signal_message(result)

                        try:

                            asyncio.run(
                                application.bot.send_message(
                                    chat_id=TELEGRAM_CHAT_ID,
                                    text=text
                                )
                            )

                        except Exception as e:

                            print(
                                "Telegram send error:",
                                e
                            )

                    time.sleep(1)

        except Exception as e:

            print(
                "Background scanner error:",
                e
            )

        time.sleep(SCAN_DELAY)


# =========================================================
# MAIN
# =========================================================

def main():

    if not TELEGRAM_BOT_TOKEN:

        raise ValueError(
            "TELEGRAM_BOT_TOKEN is missing"
        )

    if not TWELVE_DATA_KEY:

        print(
            "WARNING: TWELVE_DATA_KEY is missing. "
            "Forex/Gold data will not work."
        )

    # Render health server
    threading.Thread(
        target=start_health_server,
        daemon=True
    ).start()

    # Telegram
    application = (
        Application.builder()
        .token(TELEGRAM_BOT_TOKEN)
        .build()
    )

    application.add_handler(
        CommandHandler(
            "start",
            start
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
            scan_command
        )
    )

    # Background scanner
    threading.Thread(
        target=background_scanner,
        args=(application,),
        daemon=True
    ).start()

    print(
        "🚀 CryptoFlowBot started"
    )

    application.run_polling()


if __name__ == "__main__":
    main()
