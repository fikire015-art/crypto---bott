import os
import time
import threading
from datetime import datetime, timezone

import requests
from flask import Flask, request, jsonify


# =========================================================
# CONFIGURATION
# =========================================================

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
TWELVE_DATA_KEY = os.getenv("TWELVE_DATA_KEY", "").strip()

# Example:
# BTC/USD,ETH/USD
SYMBOLS = [
    s.strip()
    for s in os.getenv("SYMBOLS", "BTC/USD,ETH/USD").split(",")
    if s.strip()
]

TIMEFRAME = os.getenv("TIMEFRAME", "15min")

# Minimum seconds between Twelve Data requests for the same symbol
REQUEST_COOLDOWN = int(os.getenv("REQUEST_COOLDOWN", "65"))

# Cache
cache = {}
cache_lock = threading.Lock()


app = Flask(__name__)


# =========================================================
# BASIC CHECK
# =========================================================

@app.get("/")
def home():
    return jsonify({
        "status": "online",
        "bot": "Crypto Market Bot",
        "time": datetime.now(timezone.utc).isoformat(),
        "symbols": SYMBOLS,
        "timeframe": TIMEFRAME
    })


@app.get("/health")
def health():
    return jsonify({"status": "healthy"})


# =========================================================
# TELEGRAM
# =========================================================

def telegram_api(method, payload=None):
    if not TELEGRAM_BOT_TOKEN:
        return {"ok": False, "description": "TELEGRAM_BOT_TOKEN is missing"}

    url = (
        f"https://api.telegram.org/"
        f"bot{TELEGRAM_BOT_TOKEN}/{method}"
    )

    try:
        response = requests.post(
            url,
            json=payload or {},
            timeout=20
        )

        try:
            return response.json()
        except Exception:
            return {
                "ok": False,
                "description": response.text
            }

    except requests.RequestException as exc:
        return {
            "ok": False,
            "description": str(exc)
        }


def send_message(chat_id, text):
    if not chat_id:
        return

    telegram_api(
        "sendMessage",
        {
            "chat_id": chat_id,
            "text": text
        }
    )


# =========================================================
# TWELVE DATA
# =========================================================

def get_market_data(symbol):
    if not TWELVE_DATA_KEY:
        return None, "TWELVE_DATA_KEY is missing"

    now = time.time()

    # Check cache / cooldown
    with cache_lock:
        old = cache.get(symbol)

        if old:
            age = now - old["time"]

            if age < REQUEST_COOLDOWN:
                return old["data"], None

    url = "https://api.twelvedata.com/time_series"

    params = {
        "symbol": symbol,
        "interval": TIMEFRAME,
        "outputsize": 100,
        "apikey": TWELVE_DATA_KEY
    }

    try:
        response = requests.get(
            url,
            params=params,
            timeout=20
        )

        if response.status_code == 429:
            return None, (
                "Twelve Data API limit reached. "
                "Please wait before trying again."
            )

        if response.status_code != 200:
            return None, (
                f"Twelve Data HTTP error: "
                f"{response.status_code}"
            )

        data = response.json()

        if data.get("status") == "error":
            return None, data.get(
                "message",
                "Twelve Data returned an error."
            )

        values = data.get("values")

        if not values:
            return None, "No market data returned."

        # Twelve Data normally returns newest first.
        values = list(reversed(values))

        with cache_lock:
            cache[symbol] = {
                "time": now,
                "data": values
            }

        return values, None

    except requests.RequestException as exc:
        return None, f"Market request failed: {exc}"


# =========================================================
# TECHNICAL ANALYSIS
# =========================================================

def sma(values, period):
    if len(values) < period:
        return None

    return sum(values[-period:]) / period


def calculate_ema(values, period):
    if len(values) < period:
        return None

    multiplier = 2 / (period + 1)

    ema_value = sum(values[:period]) / period

    for price in values[period:]:
        ema_value = (
            (price - ema_value) * multiplier
        ) + ema_value

    return ema_value


def calculate_rsi(closes, period=14):
    if len(closes) < period + 1:
        return None

    gains = []
    losses = []

    for i in range(1, len(closes)):
        change = closes[i] - closes[i - 1]

        gains.append(max(change, 0))
        losses.append(max(-change, 0))

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    for i in range(period, len(gains)):
        avg_gain = (
            (avg_gain * (period - 1)) +
            gains[i]
        ) / period

        avg_loss = (
            (avg_loss * (period - 1)) +
            losses[i]
        ) / period

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss

    return 100 - (100 / (1 + rs))


def analyze_symbol(symbol):
    values, error = get_market_data(symbol)

    if error:
        return {
            "symbol": symbol,
            "error": error
        }

    try:
        closes = [
            float(x["close"])
            for x in values
        ]

        highs = [
            float(x["high"])
            for x in values
        ]

        lows = [
            float(x["low"])
            for x in values
        ]

        current = closes[-1]

        ema20 = calculate_ema(closes, 20)
        ema50 = calculate_ema(closes, 50)
        rsi = calculate_rsi(closes, 14)

        high20 = max(highs[-20:])
        low20 = min(lows[-20:])

        score = 0

        # Trend
        if ema20 is not None and ema50 is not None:
            if ema20 > ema50:
                score += 2
            elif ema20 < ema50:
                score -= 2

        # Price vs EMA20
        if ema20 is not None:
            if current > ema20:
                score += 1
            elif current < ema20:
                score -= 1

        # RSI
        if rsi is not None:
            if 50 <= rsi <= 70:
                score += 1
            elif 30 <= rsi < 50:
                score -= 1

        # Avoid extreme RSI
        if rsi is not None:
            if rsi > 75:
                score -= 1
            elif rsi < 25:
                score += 1

        if score >= 3:
            signal = "BUY"
        elif score <= -3:
            signal = "SELL"
        else:
            signal = "WAIT"

        return {
            "symbol": symbol,
            "price": current,
            "ema20": ema20,
            "ema50": ema50,
            "rsi": rsi,
            "high20": high20,
            "low20": low20,
            "score": score,
            "signal": signal
        }

    except Exception as exc:
        return {
            "symbol": symbol,
            "error": f"Analysis error: {exc}"
        }


# =========================================================
# FORMAT TELEGRAM SIGNAL
# =========================================================

def format_result(result):
    symbol = result["symbol"]

    if result.get("error"):
        return (
            f"⚠️ {symbol}\n\n"
            f"Error:\n{result['error']}"
        )

    price = result["price"]
    ema20 = result["ema20"]
    ema50 = result["ema50"]
    rsi = result["rsi"]
    score = result["score"]
    signal = result["signal"]

    if signal == "BUY":
        icon = "🟢"
    elif signal == "SELL":
        icon = "🔴"
    else:
        icon = "🟡"

    return (
        f"{icon} {symbol} MARKET ANALYSIS\n\n"
        f"Signal: {signal}\n"
        f"Price: {price:.6f}\n"
        f"EMA20: {ema20:.6f}\n"
        f"EMA50: {ema50:.6f}\n"
        f"RSI(14): {rsi:.2f}\n"
        f"Score: {score}\n\n"
        f"Timeframe: {TIMEFRAME}\n"
        f"⚠️ Signal only — not a guaranteed trade."
    )


# =========================================================
# COMMANDS
# =========================================================

def command_status(chat_id):
    text = (
        "🤖 Crypto Market Bot\n\n"
        "Status: ONLINE ✅\n"
        f"Timeframe: {TIMEFRAME}\n"
        f"Symbols: {', '.join(SYMBOLS)}\n\n"
        "Commands:\n"
        "/status - bot status\n"
        "/analyze BTC/USD - analyze one symbol\n"
        "/scan - scan configured symbols\n"
        "/all - same as scan"
    )

    send_message(chat_id, text)


def command_analyze(chat_id, symbol):
    if not symbol:
        send_message(
            chat_id,
            "Usage:\n/analyze BTC/USD"
        )
        return

    result = analyze_symbol(symbol.upper())

    send_message(
        chat_id,
        format_result(result)
    )


def command_scan(chat_id):
    if not SYMBOLS:
        send_message(
            chat_id,
            "No symbols configured."
        )
        return

    # One request per configured symbol.
    # Cached requests will not hit Twelve Data again.
    results = []

    for symbol in SYMBOLS:
        results.append(
            analyze_symbol(symbol)
        )

    text = "📊 MARKET SCAN\n\n"

    for result in results:
        if result.get("error"):
            text += (
                f"⚠️ {result['symbol']}: "
                f"{result['error']}\n\n"
            )
            continue

        text += (
            f"{result['symbol']}\n"
            f"Signal: {result['signal']}\n"
            f"Price: {result['price']:.6f}\n"
            f"RSI: {result['rsi']:.2f}\n"
            f"Score: {result['score']}\n\n"
        )

    text += (
        f"Timeframe: {TIMEFRAME}\n"
        "⚠️ Analysis is not a guaranteed prediction."
    )

    send_message(chat_id, text)


# =========================================================
# TELEGRAM WEBHOOK
# =========================================================

@app.post("/telegram/webhook")
def telegram_webhook():
    try:
        update = request.get_json(silent=True) or {}

        message = update.get("message")

        if not message:
            return jsonify({"ok": True})

        chat = message.get("chat", {})
        chat_id = str(chat.get("id", ""))

        text = message.get("text", "").strip()

        if not text:
            return jsonify({"ok": True})

        parts = text.split()

        command = parts[0].lower().split("@")[0]

        if command == "/start":
            command_status(chat_id)

        elif command == "/status":
            command_status(chat_id)

        elif command == "/analyze":
            symbol = parts[1] if len(parts) > 1 else ""
            command_analyze(chat_id, symbol)

        elif command in ("/scan", "/all"):
            command_scan(chat_id)

        else:
            send_message(
                chat_id,
                "Unknown command.\n\n"
                "Use /status to see available commands."
            )

        return jsonify({"ok": True})

    except Exception as exc:
        print("Webhook error:", exc)

        return jsonify({
            "ok": False,
            "error": str(exc)
        }), 500


# =========================================================
# SET TELEGRAM WEBHOOK
# =========================================================

def set_webhook():
    render_url = os.getenv("RENDER_EXTERNAL_URL", "").strip()

    if not render_url:
        print("RENDER_EXTERNAL_URL is not available.")
        return

    if not TELEGRAM_BOT_TOKEN:
        print("TELEGRAM_BOT_TOKEN is missing.")
        return

    webhook_url = (
        render_url.rstrip("/")
        + "/telegram/webhook"
    )

    result = telegram_api(
        "setWebhook",
        {
            "url": webhook_url
        }
    )

    print("Telegram webhook:", result)


# =========================================================
# STARTUP
# =========================================================

if __name__ == "__main__":
    print("======================================")
    print("Crypto Market Bot starting...")
    print("======================================")

    if not TELEGRAM_BOT_TOKEN:
        print("WARNING: TELEGRAM_BOT_TOKEN missing")

    if not TELEGRAM_CHAT_ID:
        print("WARNING: TELEGRAM_CHAT_ID missing")

    if not TWELVE_DATA_KEY:
        print("WARNING: TWELVE_DATA_KEY missing")

    print("Symbols:", SYMBOLS)
    print("Timeframe:", TIMEFRAME)

    # Give Render a moment to provide its external URL.
    threading.Thread(
        target=set_webhook,
        daemon=True
    ).start()

    port = int(os.getenv("PORT", "10000"))

    app.run(
        host="0.0.0.0",
        port=port
    )
