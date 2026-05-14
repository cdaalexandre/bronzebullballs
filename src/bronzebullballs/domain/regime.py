"""Quality filter and market-wide regime gate (Clenow Cap. 5).

Two boolean predicates that gate the entire screening pipeline:

    - :func:`quality_filter` -- per-stock binary filter ("is this name healthy
      enough to be a candidate?"). Implements Clenow's gap and trend criteria.
    - :func:`regime_on` -- portfolio-level kill-switch ("is the broad market
      in a state where momentum makes sense?"). When False, screening returns
      no picks and the CLI prints ``CASH`` mode.

Reference: Clenow, *Stocks on the Move* / *Trading Evolved*, Cap. 5
("Quality and Trend Filters"). Literal port from ``corpus_master_v8.py``
lines 445-460.
"""

from __future__ import annotations

import pandas as pd


def quality_filter(
    close: pd.Series,
    gap_thresh: float = 0.15,
    ma_p: int = 100,
    lookback: int = 90,
) -> bool:
    """Clenow Cap. 5 quality filter: no large gaps AND price above MA-100.

    A stock passes if:

        1. No single-day return exceeds ``gap_thresh`` in absolute value over
           the last ``lookback`` bars (no earnings shocks, no manipulation).
        2. The last close is above the simple moving average of ``ma_p`` bars
           (the stock is in an uptrend on a multi-month horizon).

    Args:
        close: closing prices.
        gap_thresh: maximum absolute single-day return tolerated. Default 0.15
            (15 %).
        ma_p: moving-average period for the trend check. Default 100.
        lookback: window in bars over which the gap check is performed.
            Default 90.

    Returns:
        ``True`` if both conditions are satisfied, else ``False``. Returns
        ``False`` on insufficient data.
    """
    if len(close) < ma_p:
        return False
    recent = close.iloc[-lookback:]
    if (recent.pct_change().abs() > gap_thresh).any():
        return False
    ma = close.rolling(ma_p).mean().iloc[-1]
    return bool(close.iloc[-1] > ma)


def regime_on(spy_close: pd.Series, ma_p: int = 200) -> bool:
    """Clenow Cap. 5 regime gate: SPY above its MA-200.

    The simplest possible market-wide trend filter. When SPY closes below its
    200-day moving average, momentum strategies historically underperform; the
    pipeline switches to CASH mode and emits zero picks.

    Args:
        spy_close: SPY (or any broad benchmark) closing prices.
        ma_p: moving-average period. Default 200.

    Returns:
        ``True`` if the regime is ON (long signals allowed), else ``False``.
    """
    if len(spy_close) < ma_p:
        return False
    return float(spy_close.iloc[-1]) > float(spy_close.rolling(ma_p).mean().iloc[-1])
