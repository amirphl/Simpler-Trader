from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import uvicorn


class ExtraFormatter(logging.Formatter):
    _reserved = {
        "name",
        "msg",
        "args",
        "levelname",
        "levelno",
        "pathname",
        "filename",
        "module",
        "exc_info",
        "exc_text",
        "stack_info",
        "lineno",
        "funcName",
        "created",
        "msecs",
        "relativeCreated",
        "thread",
        "threadName",
        "processName",
        "process",
        "message",
        "asctime",
    }

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        extras = {
            key: value
            for key, value in record.__dict__.items()
            if key not in self._reserved and not key.startswith("_")
        }
        if extras:
            extra_str = " ".join(f"{key}={value}" for key, value in sorted(extras.items()))
            return f"{base} | {extra_str}"
        return base


def configure_logging(level: str) -> None:
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(ExtraFormatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    logging.basicConfig(level=numeric_level, handlers=[handler])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch the backtest web control panel.")
    parser.add_argument("--host", default=os.getenv("WEB_HOST", "0.0.0.0"), help="Bind address (default: 0.0.0.0)")
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("WEB_PORT", "9092")),
        help="Port to listen on (default: 9092)",
    )
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (for development)")
    parser.add_argument("--log-level", default=os.getenv("WEB_LOG_LEVEL", "info"), help="uvicorn log level")
    parser.add_argument(
        "--postgres-config-file",
        type=Path,
        default=Path(os.getenv("WEB_POSTGRES_CONFIG_FILE", "./configs/postgres.env")),
        help="Path to .env-style Postgres config for candle storage (default: ./configs/postgres.env)",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Run in local-only mode (binds 127.0.0.1:9092 and disables HTTPS enforcement)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    host = args.host
    port = args.port
    if args.local:
        host = "127.0.0.1"
        port = 9092
        os.environ["WEB_FORCE_HTTPS"] = "false"
        os.environ.setdefault("WEB_TRUSTED_HOSTS", "localhost,127.0.0.1")

    configure_logging(args.log_level)
    if args.postgres_config_file:
        os.environ["CANDLE_DB_ENV_FILE"] = str(args.postgres_config_file)
        if args.postgres_config_file.exists():
            logging.info("Using Postgres config file: %s", args.postgres_config_file)
        else:
            logging.info(
                "Postgres config file not found; using defaults/env only: %s",
                args.postgres_config_file,
            )

    config = uvicorn.Config(
        "webserver.app:app",
        host=host,
        port=port,
        reload=args.reload,
        log_level=args.log_level,
        log_config=None,
    )
    server = uvicorn.Server(config)
    try:
        server.run()
    except KeyboardInterrupt:
        logging.info("Web server interrupted by user.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
