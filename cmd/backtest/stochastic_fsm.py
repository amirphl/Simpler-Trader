from __future__ import annotations

import sys

from cmd.backtest.main import main as unified_backtest_main


def main(argv: list[str] | None = None) -> int:
    forwarded_argv = list(argv) if argv is not None else list(sys.argv[1:])
    if "--strategy" not in forwarded_argv:
        forwarded_argv = ["--strategy", "stochastic_fsm", *forwarded_argv]
    return unified_backtest_main(forwarded_argv)


if __name__ == "__main__":
    sys.exit(main())
