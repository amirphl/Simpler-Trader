"""Strategy dispatcher for live trading CLI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from ._shared import load_env_config, resolve_config_file
from .heiken_ashi_main import main as heiken_ashi_main
from .pinbar_magic_v2_main import main as pinbar_magic_v2_main


def _resolve_strategy(argv: Optional[List[str]] = None) -> str:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--strategy-name",
        choices=["heiken_ashi", "pinbar_magic_v2"],
        default=None,
    )
    parser.add_argument("--config-file", type=Path, default=None)

    pre_args, _ = parser.parse_known_args(argv)
    if pre_args.strategy_name:
        return pre_args.strategy_name

    resolved_config_file = resolve_config_file(pre_args)
    env_config = load_env_config(resolved_config_file)
    strategy = str(env_config.get("strategy_name") or "pinbar_magic_v2").strip().lower()
    if strategy == "heiken_ashi":
        return "heiken_ashi"
    return "pinbar_magic_v2"


def main(argv: Optional[List[str]] = None) -> int:
    effective_argv = list(sys.argv[1:] if argv is None else argv)
    strategy = _resolve_strategy(effective_argv)
    if strategy == "heiken_ashi":
        return heiken_ashi_main(effective_argv)
    return pinbar_magic_v2_main(effective_argv)


if __name__ == "__main__":
    sys.exit(main())
