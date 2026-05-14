"""Technical indicators (pure functions over price series).

This module is the FIRST product code in `bronzebullballs`. All functions are
pure: no I/O, no network, no clock access. Inputs are pandas Series (close,
high, low) or floats; outputs are pandas Series (rolling indicators) or floats
(window-aggregate indicators). They live in the domain layer because they are
mathematics, not data plumbing.

References (literal port from `corpus_master_v8.py` lines 336-362):
    - RSI (Wilder 1978; Brown Cap. 8 calibration): :func:`rsi`.
    - ATR (Wilder 1978, simple-mean variant; Clenow Cap. 11 sizing): :func:`atr`.
    - Adjusted slope (Clenow Cap. 3 ranking core): :func:`adjusted_slope`.

Two minimal deviations from the monolith, both type-safety driven (Custom
Instructions section 11.4 -- "remove I/O dependencies after a literal copy"):

    - ``stats.linregress`` is unpacked via ``result.slope`` / ``result.rvalue``
      instead of the 5-tuple ``slope, _, r, _, _``. Modern SciPy returns a
      named tuple; attribute access is strict-mypy clean.
    - ``recent.values`` becomes ``recent.to_numpy()``. ``.values`` is soft-
      deprecated by pandas-stubs; ``.to_numpy()`` is the documented API and
      produces an identical ndarray.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """Relative Strength Index (Wilder, n=14 default).

    Args:
        close: Daily close prices.
        n: Smoothing window in trading days.

    Returns:
        RSI series with values in ``[0, 100]``. The first ``n`` rows are NaN
        because ``diff`` produces one NaN at index 0 and the rolling mean
        needs ``n`` valid deltas after that.
    """
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(n).mean()
    loss = (-delta.clip(upper=0)).rolling(n).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def atr(close: pd.Series, high: pd.Series, low: pd.Series, n: int = 20) -> pd.Series:
    """Average True Range (Wilder, simple-mean variant, n=20 default).

    Args:
        close: Daily close prices. Used for the prior-close term in TR.
        high: Daily highs.
        low: Daily lows.
        n: Rolling window in trading days.

    Returns:
        ATR series in price units. The first ``n - 1`` rows are NaN
        (``rolling(n).mean`` needs ``n`` true-range values to emit one).
    """
    tr = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(n).mean()


def adjusted_slope(close: pd.Series, period: int = 90) -> float:
    """Clenow adjusted slope: annualized log-linear slope x R-squared.

    The ranking core of `bronzebullballs` (Clenow Cap. 3). Higher means a
    stronger uptrend with tighter fit; lower means a strong downtrend with
    tight fit; near zero means noise.

    Formula:
        ``(exp(slope * 252) - 1) * 100 * r_squared``

    where ``slope`` is the OLS coefficient of ``log(close)`` against day
    index, ``252`` annualizes from trading days, and the ``r_squared``
    factor penalizes noisy fits per Clenow's recommendation.

    Args:
        close: Daily close prices.
        period: Lookback window in trading days (default 90 = Clenow spec).

    Returns:
        Annualized adjusted slope as a percentage. ``float('nan')`` if the
        series has fewer than ``period`` non-NaN observations.
    """
    s = close.dropna()
    if len(s) < period:
        return float("nan")
    recent = s.iloc[-period:]
    log_p = np.log(recent.to_numpy())
    x = np.arange(len(log_p))
    result = stats.linregress(x, log_p)
    return float((np.exp(result.slope * 252) - 1) * 100 * (result.rvalue**2))
