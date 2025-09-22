"""Heiken Ashi live-trading entrypoint."""

from __future__ import annotations

import sys
from typing import List, Optional

from . import _shared


def _force_strategy(argv: Optional[List[str]], strategy: str) -> List[str]:
    args = list(argv or [])
    out: List[str] = []
    skip_next = False
    for token in args:
        if skip_next:
            skip_next = False
            continue
        if token == "--strategy-name":
            skip_next = True
            continue
        if token.startswith("--strategy-name="):
            continue
        out.append(token)
    out.extend(["--strategy-name", strategy])
    return out


def main(argv: Optional[List[str]] = None) -> int:
    effective_argv = list(sys.argv[1:] if argv is None else argv)
    return _shared.main(_force_strategy(effective_argv, "heiken_ashi"))


if __name__ == "__main__":
    sys.exit(main())
