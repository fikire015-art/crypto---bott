# =========================================================
# MAIN
# =========================================================

def main():

    print("================================")
    print("Starting Crypto Market Bot...")
    print("================================")

    # -----------------------------------------------------
    # ENVIRONMENT CHECK
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # TELEGRAM TOKEN IS REQUIRED
    # -----------------------------------------------------

    if not TELEGRAM_BOT_TOKEN:

        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN is missing. "
            "Add it in Render → Environment Variables."
        )

    # -----------------------------------------------------
    # START RENDER HEALTH SERVER
    # -----------------------------------------------------

    try:

        health_thread = threading.Thread(
            target=start_health_server,
            daemon=True
        )

        health_thread.start()

        print(
            f"Health server started on port {PORT}"
        )

    except Exception as e:

        print(
            "Health server error:",
            e
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

        print(e)

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
    # PHOTO HANDLER
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
    print("Telegram bot is starting...")
    print("================================")

    # -----------------------------------------------------
    # START POLLING
    # -----------------------------------------------------

    try:

        application.run_polling(
            drop_pending_updates=True
        )

    except Exception as e:

        print("================================")
        print("TELEGRAM BOT CRASHED")
        print("================================")

        print(
            f"Error: {e}"
        )

        raise


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":

    try:

        main()

    except Exception as e:

        print("================================")
        print("FATAL ERROR")
        print("================================")

        print(e)

        raise
