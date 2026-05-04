from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Literal, Sequence, Tuple

from candle_downloader.binance import (
    BinanceClient,
    BinanceClientConfig,
    MAX_BATCH,
    interval_to_milliseconds,
)
from candle_downloader.models import Candle, to_milliseconds
from experiments.candle_csv import read_candles_from_csv

Direction = Literal["UPWARD", "DOWNWARD"]
HuntMode = Literal["wick", "close"]
BosState = Literal["forming", "complete", "confirmed"]


@dataclass(slots=True)
class BOSRecord:
    index: int
    direction: Direction
    anchor_index: (
        int  # start of the formation scan range (previous BOS hunt or initial)
    )
    start_index: int  # first opposite (pullback) candle that triggered BOS formation
    complete_index: int  # first resume candle that completed the BOS
    hunt_index: int  # candle whose extreme crossed the BOS level (confirmed hunt)
    level: float
    state: BosState = "confirmed"
    hunted: bool = True
    label: str = ""


@dataclass(slots=True)
class CHoCHUpdate:
    bos_index: int
    candle_index: int  # the candle that established (or updated) this CHoCH level
    level: float
    reason: str  # "initial-range" | "continuous"


@dataclass(slots=True)
class DirectionState:
    direction: Direction
    since_index: int


@dataclass(slots=True)
class DetectionEvent:
    candle_index: int
    event: str
    direction: Direction
    details: str


@dataclass(slots=True)
class DetectionConfig:
    direction_window: int = 3
    hunt_mode: HuntMode = "wick"
    include_bos_in_choch_range: bool = False
    include_hunt_candle_in_choch_range: bool = True


@dataclass(slots=True)
class BOSCHoCHResult:
    direction_state: DirectionState
    bos_records: List[BOSRecord] = field(default_factory=list)
    choch_levels_by_bos: Dict[int, List[float]] = field(default_factory=dict)
    choch_updates: List[CHoCHUpdate] = field(default_factory=list)
    events: List[DetectionEvent] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Candle helpers
# ---------------------------------------------------------------------------


def _candle_side(candle: Candle) -> int:
    if candle.is_bullish():
        return 1
    if candle.is_bearish():
        return -1
    return 0


def infer_initial_direction(candles: Sequence[Candle], window: int) -> DirectionState:
    if not candles:
        raise ValueError("candles cannot be empty")
    if window <= 0:
        raise ValueError("direction_window must be > 0")

    effective = min(window, len(candles))
    score = sum(_candle_side(candles[i]) for i in range(effective))

    if score > 0:
        direction: Direction = "UPWARD"
    elif score < 0:
        direction = "DOWNWARD"
    else:
        direction = "UPWARD"
        for i in range(effective - 1, -1, -1):
            side = _candle_side(candles[i])
            if side > 0:
                direction = "UPWARD"
                break
            if side < 0:
                direction = "DOWNWARD"
                break

    return DirectionState(direction=direction, since_index=max(0, effective - 1))


def _is_opposite(direction: Direction, candle: Candle) -> bool:
    """True when the candle runs AGAINST the current direction (pullback candle)."""
    return candle.is_bearish() if direction == "UPWARD" else candle.is_bullish()


def _is_resume(direction: Direction, candle: Candle) -> bool:
    """True when the candle resumes the current direction."""
    return candle.is_bullish() if direction == "UPWARD" else candle.is_bearish()


def _extreme(direction: Direction, candle: Candle) -> float:
    """The structural price extreme for a candle in the given direction.

    UPWARD  → HIGH   (we track swing highs as BOS levels)
    DOWNWARD→ LOW    (we track swing lows as BOS levels)
    """
    return candle.high if direction == "UPWARD" else candle.low


# ---------------------------------------------------------------------------
# BUG FIX 1 (Critical):
# The original code used a single _is_hunted / _boundary_value pair for both
# BOS hunting AND CHoCH hunting, but the two operations use OPPOSITE price
# extremes:
#
#   BOS hunt    → price moves WITH the trend and crosses the BOS level
#                 UPWARD  : candle HIGH  (or close) >= UP_BOS  level
#                 DOWNWARD: candle LOW   (or close) <= DOWN_BOS level
#
#   CHoCH hunt  → price moves AGAINST the trend and crosses the CHoCH level
#                 UPWARD  : bearish candle LOW  (or close) <= CHoCH level
#                 DOWNWARD: bullish candle HIGH (or close) >= CHoCH level
#
# The original _boundary_value returned candle.low for UPWARD and _is_hunted
# checked value <= level, which is the CHoCH-hunt logic and is completely
# wrong for BOS hunting (it caused every bullish resume candle to
# "immediately hunt" the UP_BOS, making BOS detection nonsensical).
# ---------------------------------------------------------------------------


def _is_bos_hunted(
    direction: Direction, level: float, candle: Candle, hunt_mode: HuntMode
) -> bool:
    """True when *candle* crosses the BOS *level* in the trend direction.

    UPWARD  BOS (UP_BOS  = max HIGH of pullback range): hunted when HIGH (wick)
            or CLOSE >= level  →  price broke above the swing high.
    DOWNWARD BOS (DOWN_BOS = min LOW  of pullback range): hunted when LOW (wick)
            or CLOSE <= level  →  price broke below the swing low.
    """
    if direction == "UPWARD":
        value = candle.high if hunt_mode == "wick" else candle.close
        return value >= level
    else:
        value = candle.low if hunt_mode == "wick" else candle.close
        return value <= level


def _is_choch_hunted(
    direction: Direction, level: float, candle: Candle, hunt_mode: HuntMode
) -> bool:
    """True when an *opposite* candle crosses the CHoCH support/resistance *level*.

    UPWARD  CHoCH = min LOW  (support).  Bearish candle breaks it: LOW  (wick) or CLOSE <= level.
    DOWNWARD CHoCH = max HIGH (resistance). Bullish candle breaks it: HIGH (wick) or CLOSE >= level.
    """
    if direction == "UPWARD":
        value = candle.low if hunt_mode == "wick" else candle.close
        return value <= level
    else:
        value = candle.high if hunt_mode == "wick" else candle.close
        return value >= level


def _choch_new_level(
    direction: Direction, candle: Candle, hunt_mode: HuntMode
) -> float:
    """The new (updated) CHoCH level when a CHoCH is hunted by *candle*.

    This is the price extreme of the hunting candle in the counter-trend direction:
    UPWARD  → bearish candle LOW  (or close) becomes the cascaded CHoCH.
    DOWNWARD→ bullish candle HIGH (or close) becomes the cascaded CHoCH.
    """
    if direction == "UPWARD":
        return candle.low if hunt_mode == "wick" else candle.close
    else:
        return candle.high if hunt_mode == "wick" else candle.close


# ---------------------------------------------------------------------------
# Range helpers
# ---------------------------------------------------------------------------


def _range_extreme(
    direction: Direction, candles: Sequence[Candle], start: int, end: int
) -> float:
    """Min LOW (UPWARD) or max HIGH (DOWNWARD) over candles[start..end] inclusive."""
    if direction == "UPWARD":
        return min(candles[i].low for i in range(start, end + 1))
    return max(candles[i].high for i in range(start, end + 1))


def _range_extreme_with_index(
    direction: Direction, candles: Sequence[Candle], start: int, end: int
) -> Tuple[float, int]:
    """Like _range_extreme but also returns the candle index of the extreme."""
    if direction == "UPWARD":
        idx = min(range(start, end + 1), key=lambda j: candles[j].low)
        return candles[idx].low, idx
    idx = max(range(start, end + 1), key=lambda j: candles[j].high)
    return candles[idx].high, idx


def _bos_label(direction: Direction, idx: int) -> str:
    return f"UP_BOS_{idx}" if direction == "UPWARD" else f"DOWN_BOS_{idx}"


def _choch_label(direction: Direction, bos_idx: int) -> str:
    return f"UP_CH_{bos_idx}" if direction == "UPWARD" else f"DOWN_CH_{bos_idx}"


# ---------------------------------------------------------------------------
# BUG FIX 2:
# _initial_choch previously used hunt_index as the CHoCH's candle_index in
# CHoCHUpdate, which is wrong for plotting purposes. The CHoCH level is the
# min LOW (or max HIGH) inside the range; the actual candle that holds that
# extreme is what we need to record.  We now return (level, candle_idx).
# ---------------------------------------------------------------------------


def _initial_choch(
    *,
    candles: Sequence[Candle],
    direction: Direction,
    complete_index: int,
    hunt_index: int,
    include_bos_in_choch_range: bool,
    include_hunt_candle_in_choch_range: bool,
) -> Tuple[float, int]:
    """Return (choch_level, choch_candle_index) for the initial CHoCH of a newly confirmed BOS."""
    start = complete_index if include_bos_in_choch_range else complete_index + 1
    end = hunt_index if include_hunt_candle_in_choch_range else hunt_index - 1
    if start > end:
        start = hunt_index
        end = hunt_index
    return _range_extreme_with_index(direction, candles, start, end)


# ---------------------------------------------------------------------------
# Core detection
# ---------------------------------------------------------------------------


def detect_bos_choch(
    candles: Sequence[Candle], config: DetectionConfig | None = None
) -> BOSCHoCHResult:
    if not candles:
        raise ValueError("candles cannot be empty")

    cfg = config or DetectionConfig()
    direction_state = infer_initial_direction(candles, cfg.direction_window)
    result = BOSCHoCHResult(
        direction_state=DirectionState(
            direction_state.direction, direction_state.since_index
        )
    )

    direction = direction_state.direction
    anchor_index = direction_state.since_index

    phase: Literal["WAIT_PULLBACK", "FORMING", "AWAIT_HUNT"] = "WAIT_PULLBACK"
    forming_start_index: int | None = None  # first pullback candle
    forming_level: float | None = None  # accumulating BOS level
    complete_index: int | None = None  # first resume candle

    bos_counter = 0
    active_bos_ids: List[int] = []

    def reset_formation(new_anchor: int) -> None:
        nonlocal phase, forming_start_index, forming_level, complete_index, anchor_index
        phase = "WAIT_PULLBACK"
        forming_start_index = None
        forming_level = None
        complete_index = None
        anchor_index = new_anchor

    def confirm_bos(hunt_index: int) -> None:
        nonlocal bos_counter
        assert (
            forming_start_index is not None
            and forming_level is not None
            and complete_index is not None
        )

        bos = BOSRecord(
            index=bos_counter,
            direction=direction,
            anchor_index=anchor_index,  # BUG FIX 3: store full formation anchor
            start_index=forming_start_index,
            complete_index=complete_index,
            hunt_index=hunt_index,
            level=forming_level,
            state="confirmed",
            hunted=True,
            label=_bos_label(direction, bos_counter),
        )
        bos_counter += 1

        result.bos_records.append(bos)
        active_bos_ids.append(bos.index)

        # BUG FIX 2: use actual candle that holds the CHoCH extreme, not hunt_index
        choch_level, choch_candle_idx = _initial_choch(
            candles=candles,
            direction=direction,
            complete_index=complete_index,
            hunt_index=hunt_index,
            include_bos_in_choch_range=cfg.include_bos_in_choch_range,
            include_hunt_candle_in_choch_range=cfg.include_hunt_candle_in_choch_range,
        )
        result.choch_levels_by_bos[bos.index] = [choch_level]
        result.choch_updates.append(
            CHoCHUpdate(
                bos_index=bos.index,
                candle_index=choch_candle_idx,  # now correctly points to the CHoCH candle
                level=choch_level,
                reason="initial-range",
            )
        )
        result.events.append(
            DetectionEvent(
                candle_index=hunt_index,
                event="BOS_CONFIRMED",
                direction=direction,
                details=f"{bos.label}@{bos.level}",
            )
        )

    i = direction_state.since_index
    while i < len(candles):
        candle = candles[i]

        # ------------------------------------------------------------------
        # CHoCH / direction-reversal check (runs on every opposite candle,
        # regardless of current formation phase).
        # ------------------------------------------------------------------
        reversed_now = False
        if _is_opposite(direction, candle) and active_bos_ids:
            newest_bos_id = active_bos_ids[-1]
            newest_hunted = False

            for bos_id in active_bos_ids:
                levels = result.choch_levels_by_bos.get(bos_id)
                if not levels:
                    continue
                last_level = levels[-1]
                # BUG FIX 1 applied: use _is_choch_hunted (counter-trend check)
                if _is_choch_hunted(direction, last_level, candle, cfg.hunt_mode):
                    new_level = _choch_new_level(direction, candle, cfg.hunt_mode)
                    levels.append(new_level)
                    result.choch_updates.append(
                        CHoCHUpdate(
                            bos_index=bos_id,
                            candle_index=i,
                            level=new_level,
                            reason="continuous",
                        )
                    )
                    result.events.append(
                        DetectionEvent(
                            candle_index=i,
                            event="CHOCH_HUNT",
                            direction=direction,
                            details=f"{_choch_label(direction, bos_id)}@{new_level}",
                        )
                    )
                    if bos_id == newest_bos_id:
                        newest_hunted = True

            if newest_hunted:
                previous_direction = direction
                direction = "DOWNWARD" if direction == "UPWARD" else "UPWARD"
                result.events.append(
                    DetectionEvent(
                        candle_index=i,
                        event="DIRECTION_REVERSED",
                        direction=direction,
                        details=f"{previous_direction}->{direction}",
                    )
                )
                active_bos_ids = []
                result.direction_state = DirectionState(
                    direction=direction, since_index=i
                )
                reset_formation(i)
                reversed_now = True

        if reversed_now:
            i += 1
            continue

        # ------------------------------------------------------------------
        # Phase state machine
        # ------------------------------------------------------------------
        advance_index = True

        # Step 0: wait for the first pullback (opposite) candle.
        if phase == "WAIT_PULLBACK":
            if _is_opposite(direction, candle):
                # Step 1: scan anchor → i (inclusive) for the structural extreme.
                start = anchor_index  # anchor_index <= i always
                level = _extreme(direction, candles[start])
                for j in range(start + 1, i + 1):
                    candidate = _extreme(direction, candles[j])
                    if direction == "UPWARD":
                        level = max(level, candidate)
                    else:
                        level = min(level, candidate)
                forming_start_index = i  # record the pullback trigger candle
                forming_level = level
                complete_index = None
                phase = "FORMING"
            i += 1
            continue

        # Steps 2 / 2-1 / 2-2: accumulate more pullback candles or wait for resume.
        if phase == "FORMING":
            assert forming_level is not None and forming_start_index is not None

            # BUG FIX 4 (dead code removed):
            # The original guarded `if i == forming_start_index` here, but
            # WAIT_PULLBACK always does `i += 1` before switching to FORMING,
            # so i is already forming_start_index+1 when FORMING is first
            # entered.  The guard was unreachable and has been removed.

            if _is_resume(direction, candle):
                # Step 2-1: first resume candle → BOS is "complete".
                complete_index = i
                phase = "AWAIT_HUNT"
                # Step 3: check immediately whether this candle also hunts the BOS.
                # BUG FIX 1 applied: use _is_bos_hunted (trend-direction check)
                if _is_bos_hunted(direction, forming_level, candle, cfg.hunt_mode):
                    confirm_bos(i)
                    reset_formation(i)
                    # Step 5 → Step 0: re-evaluate the *same* candle in WAIT_PULLBACK.
                    advance_index = False
                if advance_index:
                    i += 1
                continue

            if _is_opposite(direction, candle):
                # Step 2-2: another pullback candle → extend the BOS level.
                candidate = _extreme(direction, candle)
                if direction == "UPWARD":
                    forming_level = max(forming_level, candidate)
                else:
                    forming_level = min(forming_level, candidate)

            i += 1
            continue

        # Steps 3 / 4: wait for any candle to hunt the completed BOS.
        assert phase == "AWAIT_HUNT"
        assert forming_level is not None and complete_index is not None

        # BUG FIX 5 (dead code removed):
        # The original skipped `i == complete_index` here, but the FORMING
        # phase always advances i before setting phase = AWAIT_HUNT, so
        # i > complete_index when AWAIT_HUNT is first entered. Guard removed.

        # Step 4: any candle (bull or bear) may hunt the BOS.
        # BUG FIX 1 applied: use _is_bos_hunted (trend-direction check)
        if _is_bos_hunted(direction, forming_level, candle, cfg.hunt_mode):
            confirm_bos(i)
            reset_formation(i)
            # Step 5 → Step 0: allow the same candle to start a new formation.
            advance_index = False

        if advance_index:
            i += 1

    return result


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def build_plotly_figure(candles: Sequence[Candle], result: BOSCHoCHResult):
    """Build a Plotly candlestick chart with short BOS and CHoCH level lines.

    BUG FIX 6:
    The original used add_hline() which draws a full-width horizontal rule
    across the entire chart.  The spec requires "short thin" lines that span
    only the relevant price action (from formation start to BOS hunt candle).
    We now use add_shape() with explicit x-coordinates.

    BUG FIX 7:
    CHoCH annotations were positioned at bos.hunt_index, which is the BOS
    hunt candle, not the CHoCH candle.  We now use the candle_index stored
    in each CHoCHUpdate, which correctly points to the candle that holds the
    CHoCH price extreme.
    """
    try:
        import plotly.graph_objects as go
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "plotly is required for plotting. Install with: pip install plotly"
        ) from exc

    fig = go.Figure(
        data=[
            go.Candlestick(
                x=[c.open_time for c in candles],
                open=[c.open for c in candles],
                high=[c.high for c in candles],
                low=[c.low for c in candles],
                close=[c.close for c in candles],
                name="candles",
            )
        ]
    )

    # --- BOS lines: short horizontal segment from start_index to hunt_index ---
    for bos in result.bos_records:
        color = "#16a34a" if bos.direction == "UPWARD" else "#dc2626"
        x0 = candles[bos.start_index].open_time
        x1 = candles[bos.hunt_index].open_time

        fig.add_shape(
            type="line",
            x0=x0,
            x1=x1,
            y0=bos.level,
            y1=bos.level,
            xref="x",
            yref="y",
            line=dict(color=color, width=1),
        )
        # Label at the left edge of the line (formation start)
        fig.add_annotation(
            x=x0,
            y=bos.level,
            text=bos.label,
            showarrow=False,
            font={"color": color, "size": 10},
            xanchor="right",
            yanchor="bottom",
        )

    # --- CHoCH lines: short dotted line at the last CHoCH level for each BOS ---
    # Build a lookup: bos_id → last CHoCHUpdate (preserves candle_index correctly)
    last_choch_update: Dict[int, CHoCHUpdate] = {}
    for update in result.choch_updates:
        last_choch_update[update.bos_index] = (
            update  # later updates overwrite earlier ones
        )

    for bos_idx, update in last_choch_update.items():
        bos = next((b for b in result.bos_records if b.index == bos_idx), None)
        if bos is None:
            continue
        if not result.choch_levels_by_bos.get(bos_idx):
            continue

        # BUG FIX 7: use update.candle_index (actual CHoCH candle) not bos.hunt_index
        choch_ci = update.candle_index
        # Draw the line ~10 candles past the CHoCH candle so it is clearly visible
        line_end_ci = min(choch_ci + 10, len(candles) - 1)
        x0 = candles[choch_ci].open_time
        x1 = candles[line_end_ci].open_time
        level = update.level

        color = "#2563eb" if bos.direction == "UPWARD" else "#0ea5a4"
        fig.add_shape(
            type="line",
            x0=x0,
            x1=x1,
            y0=level,
            y1=level,
            xref="x",
            yref="y",
            line=dict(color=color, width=1, dash="dot"),
        )
        fig.add_annotation(
            x=x0,
            y=level,
            text=_choch_label(bos.direction, bos_idx),
            showarrow=False,
            font={"color": color, "size": 10},
            xanchor="right",
            yanchor="top",
        )

    fig.update_layout(
        title="BOS / CHoCH Detection",
        xaxis_title="Time",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
    )
    return fig


# ---------------------------------------------------------------------------
# Data acquisition
# ---------------------------------------------------------------------------


def download_candles(
    *,
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    proxies: dict | None = None,
    logger=None,
) -> List[Candle]:
    client = BinanceClient(BinanceClientConfig(proxies=proxies or None), logger=logger)
    step = interval_to_milliseconds(interval) * MAX_BATCH
    cursor = start_ms
    candles: List[Candle] = []
    while cursor < end_ms:
        batch_end = min(end_ms, cursor + step)
        fetched = client.fetch_klines(
            symbol=symbol,
            interval=interval,
            start_ms=cursor,
            end_ms=batch_end,
            limit=MAX_BATCH,
        )
        if not fetched:
            break
        candles.extend(fetched)
        cursor = fetched[-1].open_time_ms + interval_to_milliseconds(interval)
    return candles


def get_candles(
    *,
    source: str,
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    csv_path: str | None = None,
    proxies: dict | None = None,
    logger=None,
) -> List[Candle]:
    source_normalized = source.strip().lower()
    if source_normalized == "binance":
        return download_candles(
            symbol=symbol,
            interval=interval,
            start_ms=start_ms,
            end_ms=end_ms,
            proxies=proxies,
            logger=logger,
        )
    if source_normalized == "csv":
        if not csv_path:
            raise ValueError("csv_path is required when source='csv'")
        return read_candles_from_csv(
            csv_path,
            symbol=symbol,
            interval=interval,
            start_ms=start_ms,
            end_ms=end_ms,
        )
    raise ValueError("source must be either 'binance' or 'csv'")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_datetime(value: str) -> datetime:
    cleaned = value.strip()
    if cleaned.endswith("Z"):
        cleaned = cleaned[:-1] + "+00:00"
    moment = datetime.fromisoformat(cleaned)
    if moment.tzinfo is None:
        return moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run BOS/CHoCH detection on candle data."
    )
    parser.add_argument("--symbol", required=True, help="Symbol, e.g. ETHUSDT")
    parser.add_argument("--timeframe", required=True, help="Timeframe, e.g. 1h")
    parser.add_argument(
        "--start",
        required=True,
        help="Start datetime in ISO format, e.g. 2026-01-01T00:00:00Z",
    )
    parser.add_argument(
        "--end",
        required=True,
        help="End datetime in ISO format, e.g. 2026-01-10T00:00:00Z",
    )
    parser.add_argument("--direction-window", type=int, default=3)
    parser.add_argument("--hunt-mode", choices=["wick", "close"], default="wick")
    parser.add_argument(
        "--include-bos-in-choch-range",
        action="store_true",
        help="Include BOS completion candle in initial CHoCH range",
    )
    parser.add_argument(
        "--exclude-hunt-candle-in-choch-range",
        action="store_true",
        help="Exclude BOS hunt candle from initial CHoCH range",
    )
    parser.add_argument(
        "--source",
        choices=["binance", "csv"],
        default="binance",
        help="Candle source",
    )
    parser.add_argument(
        "--csv-path",
        default=None,
        help="Required when --source csv; path to candle CSV file",
    )
    parser.add_argument("--http-proxy", default=None, help="Optional HTTP proxy URL")
    parser.add_argument("--https-proxy", default=None, help="Optional HTTPS proxy URL")
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Render Plotly chart with BOS and CHoCH lines",
    )
    parser.add_argument(
        "--plot-output",
        default=None,
        help="Optional .html output path for Plotly figure",
    )
    args = parser.parse_args()

    proxies = {}
    if args.http_proxy:
        proxies["http"] = args.http_proxy
    if args.https_proxy:
        proxies["https"] = args.https_proxy

    candles = get_candles(
        source=args.source,
        symbol=args.symbol.strip().upper(),
        interval=args.timeframe.strip(),
        start_ms=to_milliseconds(parse_datetime(args.start)),
        end_ms=to_milliseconds(parse_datetime(args.end)),
        csv_path=args.csv_path,
        proxies=proxies or None,
    )

    result = detect_bos_choch(
        candles,
        DetectionConfig(
            direction_window=args.direction_window,
            hunt_mode=args.hunt_mode,
            include_bos_in_choch_range=args.include_bos_in_choch_range,
            include_hunt_candle_in_choch_range=not args.exclude_hunt_candle_in_choch_range,
        ),
    )

    print(f"Candles: {len(candles)}")
    print(f"Detected BOS records: {len(result.bos_records)}")
    print(
        f"Current direction: {result.direction_state.direction} since index={result.direction_state.since_index}"
    )
    for bos in result.bos_records:
        choch_levels = result.choch_levels_by_bos.get(bos.index, [])
        last_choch = choch_levels[-1] if choch_levels else None
        print(
            f"{bos.label} dir={bos.direction} level={bos.level} "
            f"anchor={bos.anchor_index} start={bos.start_index} "
            f"complete={bos.complete_index} hunt={bos.hunt_index} "
            f"last_choch={last_choch}"
        )

    if args.plot:
        fig = build_plotly_figure(candles, result)
        if args.plot_output:
            fig.write_html(args.plot_output)
            print(f"Plot written to {args.plot_output}")
        else:
            fig.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
