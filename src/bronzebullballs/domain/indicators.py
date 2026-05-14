"""Technical indicators (pure functions over price series).

This module is the FIRST product code in `bronzebullballs`. All functions are
pure: no I/O, no network, no clock access. Inputs are pandas Series (close,
high, low) or floats; outputs are pandas Series (rolling indicators) or floats
(window-aggregate indicators). They live in the domain layer because they are
mathematics, not data plumbing.

References (literal port from `corpus_master_v8.py`):

    FASE E (lines 336-363):
        - RSI (Wilder 1978; Brown Cap. 8 calibration): :func:`rsi`.
        - ATR (Wilder 1978, simple-mean variant; Clenow Cap. 11 sizing): :func:`atr`.
        - Adjusted slope (Clenow Cap. 3 ranking core): :func:`adjusted_slope`.

    FASE F.1 (lines 365-498, 817-851):
        - EWMAC forecast (Carver Cap. 7 + Tab. 49): :func:`ewmac_forecast`.
        - Annualized volatility (Carver Cap. 9): :func:`vol_ann`.
        - Bollinger z-score (Brown): :func:`bollinger_z`.
        - Hurst exponent (Chan Cap. 2): :func:`hurst_clipped`.
        - Variance ratio (Chan Cap. 2): :func:`variance_ratio`.
        - Half-life of mean reversion (Chan Cap. 2): :func:`half_life`.
        - Past return (helper): :func:`past_return`.
        - Positive reversal (Brown Cap. 8, v7 corrected): :func:`positive_reversal`.
        - Sharpe ratio ex-post (v7 FIX #4): :func:`sharpe_ratio_real`.
        - Stock metrics bundle: :func:`compute_stock`.

Minimal deviations from the monolith, type-safety driven (Custom Instructions
section 11.4 -- "remove I/O dependencies after a literal copy"):

    - ``stats.linregress`` is unpacked via ``result.slope`` / ``result.rvalue``
      instead of the 5-tuple ``slope, _, r, _, _``. Modern SciPy returns a
      named tuple; attribute access is strict-mypy clean.
    - ``recent.values`` becomes ``recent.to_numpy()``. ``.values`` is soft-
      deprecated by pandas-stubs; ``.to_numpy()`` is the documented API and
      produces an identical ndarray.
    - ``OLS(...).params["lag"]`` is cast to ``float`` explicitly; the
      statsmodels return type is loosely typed and strict mypy needs help.
    - In :func:`half_life`, the ``np.log(close / close.shift(1))`` chain is
      wrapped into an explicit ``pd.Series`` constructor. At runtime
      ``np.log`` on a Series returns a Series, but the numpy-stubs declare
      the return as ``ndarray``; the wrapper makes strict mypy agree with
      reality.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant

# ----------------------------------------------------------------------------
# FASE E: RSI / ATR / adjusted slope
# ----------------------------------------------------------------------------


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """Relative Strength Index, Wilder (1978).

    Args:
        close: closing prices, index ignored.
        n: lookback window. Default 14 (Wilder's original).

    Returns:
        RSI series, same index as ``close``. First ``n`` entries are NaN.
        Bounded [0, 100]; 50 is neutral.
    """
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(n).mean()
    loss = (-delta.clip(upper=0)).rolling(n).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def atr(close: pd.Series, high: pd.Series, low: pd.Series, n: int = 20) -> pd.Series:
    """Average True Range (Wilder simple-mean variant).

    Used for ATR-based position sizing (Clenow Cap. 11) and as a raw volatility
    proxy. NOT Wilder's smoothed ATR -- this is a simple rolling mean of TR,
    matching the monolith's implementation for fidelity.

    Args:
        close: closing prices.
        high: session highs.
        low: session lows.
        n: rolling window. Default 20.

    Returns:
        ATR series in price units. First ``n-1`` entries are NaN.
    """
    tr = pd.concat(
        [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(n).mean()


def adjusted_slope(close: pd.Series, period: int = 90) -> float:
    """Clenow Cap. 3 adjusted log-linear slope (annualized times R-squared).

    Args:
        close: closing prices, descending or ascending order, NaNs dropped
            internally.
        period: lookback window in trading days. Default 90 (~4-5 months).

    Returns:
        Adjusted slope as a percentage. NaN if fewer than ``period`` valid
        observations.
    """
    s = close.dropna()
    if len(s) < period:
        return math.nan
    recent = s.iloc[-period:]
    log_p = np.log(recent.to_numpy())
    x = np.arange(len(log_p))
    result = stats.linregress(x, log_p)
    slope = float(result.slope)
    r = float(result.rvalue)
    return (math.exp(slope * 252) - 1) * 100 * (r**2)


# ----------------------------------------------------------------------------
# FASE F.1: trend / vol / mean-reversion indicators
# ----------------------------------------------------------------------------


def ewmac_forecast(
    close: pd.Series,
    fast: int = 16,
    slow: int = 64,
    vol_w: int = 25,
    scalar: float = 3.75,
    cap: float = 20.0,
) -> float:
    """Carver Cap. 7 + Appendix C Tab. 49 EWMAC forecast.

    The default scalar 3.75 corresponds to EWMAC(16, 64). Other pairs (only if
    fast/slow change): EWMAC(2,8)=10.6, (4,16)=7.5, (8,32)=5.3, (16,64)=3.75,
    (32,128)=2.65, (64,256)=1.87.

    Args:
        close: closing prices.
        fast: fast EMA span. Default 16.
        slow: slow EMA span. Default 64.
        vol_w: rolling window for return volatility. Default 25.
        scalar: forecast scalar from Carver Tab. 49. Default 3.75.
        cap: absolute forecast cap. Default +/- 20.

    Returns:
        Last EWMAC value, clipped to ``[-cap, +cap]``. NaN if window
        insufficient.
    """
    if len(close) < slow + vol_w:
        return math.nan
    ef = close.ewm(span=fast, adjust=False).mean()
    es = close.ewm(span=slow, adjust=False).mean()
    raw = ef - es
    rets = close.pct_change()
    sigma = rets.rolling(vol_w).std()
    pv = sigma * close
    fc = (raw / pv * scalar).clip(-cap, cap)
    return float(fc.iloc[-1])


def vol_ann(close: pd.Series, n: int = 25) -> float:
    """Annualized volatility from rolling daily-return std-dev (Carver Cap. 9).

    Args:
        close: closing prices.
        n: rolling window. Default 25 (~1 trading month).

    Returns:
        Last annualized volatility as a percentage. NaN if window insufficient.
    """
    rets = close.pct_change()
    if len(rets) < n:
        return math.nan
    return float(rets.rolling(n).std().iloc[-1] * np.sqrt(252)) * 100


def bollinger_z(close: pd.Series, n: int = 20) -> float:
    """Bollinger z-score: ``(close - SMA_n) / std_n`` at the last bar.

    Args:
        close: closing prices.
        n: rolling window. Default 20.

    Returns:
        Last z-score. Values around 0 are inside the bands; +/- 2 marks the
        classical 2-sigma band edges.
    """
    m = close.rolling(n).mean()
    sd = close.rolling(n).std()
    return float(((close - m) / sd).iloc[-1])


def hurst_clipped(close: pd.Series, max_lag: int = 20) -> float:
    """Clipped Hurst exponent estimator (Chan Cap. 2).

    H ~ 0.5 random walk; H > 0.55 trending; H < 0.45 mean-reverting.
    Output is clipped to ``[0, 1]`` to absorb sampling noise.

    Caveat: this estimator is calibrated for stochastic series (random walks
    with drift). On a deterministic linear ramp the differenced standard
    deviation is effectively zero at every lag, so the regression has no
    signal and the result is meaningless; the test suite documents this.

    Args:
        close: closing prices.
        max_lag: maximum lag for variance-of-differences regression.

    Returns:
        Hurst exponent in ``[0, 1]``. NaN if window insufficient or
        any variance is zero.
    """
    s = close.dropna().to_numpy()
    if len(s) < max_lag * 2:
        return math.nan
    lags = range(2, max_lag)
    tau = [float(np.std(np.subtract(s[lag:], s[:-lag]))) for lag in lags]
    if any(t == 0 for t in tau):
        return math.nan
    poly = np.polyfit(np.log(list(lags)), np.log(tau), 1)
    return max(0.0, min(1.0, float(poly[0] * 2.0)))


def variance_ratio(close: pd.Series, k: int = 5) -> float:
    """Lo-MacKinlay variance ratio (Chan Cap. 2).

    VR > 1 trending; VR < 1 mean-reverting; VR ~ 1 random walk.

    Args:
        close: closing prices.
        k: aggregation lag. Default 5 (one trading week).

    Returns:
        Variance ratio. NaN if window insufficient or 1-day variance is zero.
    """
    s = np.log(close.dropna().to_numpy())
    if len(s) < k * 4:
        return math.nan
    rk = s[k:] - s[:-k]
    r1 = s[1:] - s[:-1]
    v_k, v_1 = float(np.var(rk, ddof=1)), float(np.var(r1, ddof=1))
    if v_1 == 0:
        return math.nan
    return v_k / (k * v_1)


def half_life(close: pd.Series) -> float:
    """Mean-reversion half-life via Ornstein-Uhlenbeck regression (Chan Cap. 2).

    Fits the regression ``delta_log(p_t) = a + b * log(p_{t-1})``. If
    ``b < 0`` the series mean-reverts and the half-life is ``-ln(2) / b``.

    Args:
        close: closing prices.

    Returns:
        Half-life in days, or NaN if the series does not mean-revert
        (``b >= 0``) or if the regression fails / is insufficient.
    """
    # np.log on a Series returns a Series at runtime, but numpy-stubs declare
    # the return as ndarray. Wrap explicitly so strict mypy is satisfied.
    ratio = close / close.shift(1)
    lr = pd.Series(np.log(ratio.to_numpy()), index=ratio.index).dropna()
    if len(lr) < 20:
        return math.nan
    lag = lr.shift(1).dropna()
    delta = lr.diff().dropna()
    aligned = pd.concat([delta, lag], axis=1).dropna()
    if len(aligned) < 10:
        return math.nan
    aligned.columns = ["delta", "lag"]
    try:
        m = OLS(aligned["delta"], add_constant(aligned["lag"])).fit()
        lam = float(m.params["lag"])
        if lam >= 0:
            return math.nan
        return float(-math.log(2) / lam)
    except Exception:
        return math.nan


def past_return(close: pd.Series, days: int) -> float:
    """Simple percentage return over the last ``days`` bars.

    Args:
        close: closing prices.
        days: lookback in bars.

    Returns:
        Return as a percentage. NaN if window insufficient.
    """
    if len(close) < days + 1:
        return math.nan
    return float((close.iloc[-1] / close.iloc[-1 - days] - 1) * 100)


def positive_reversal(close: pd.Series, lookback: int = 120) -> bool:
    """Brown Cap. 8 positive reversal (v7 corrected definition).

    Positive reversal = RSI prints a NEW LOWER low AND price prints a NEW
    HIGHER low at the same swing. Brown's reading: "underlying strength" --
    market holds price up despite weakening momentum.

    The v6 monolith encoded the OPPOSITE (classical bullish divergence,
    a different signal). FASE E preserves the v7 fix.

    Args:
        close: closing prices.
        lookback: window in bars over which to search for swing lows.

    Returns:
        ``True`` if the pattern is found, else ``False``. Returns ``False``
        on insufficient data or any NaN in the window.
    """
    if len(close) < lookback + 20:
        return False
    r = rsi(close, 14)
    window = close.iloc[-lookback:]
    rsi_w = r.iloc[-lookback:]
    if window.isna().any() or rsi_w.isna().any():
        return False
    mins: list[tuple[int, float, float]] = []
    for i in range(5, len(rsi_w) - 5):
        if (
            rsi_w.iloc[i] < rsi_w.iloc[i - 3 : i].min()
            and rsi_w.iloc[i] < rsi_w.iloc[i + 1 : i + 4].min()
            and rsi_w.iloc[i] < 50
        ):
            mins.append((i, float(rsi_w.iloc[i]), float(window.iloc[i])))
    if len(mins) < 2:
        return False
    prev_low, last_low = mins[-2], mins[-1]
    # RSI[last] < RSI[prev] (oscillator low lower)
    # price[last] > price[prev] (price low higher)
    return (last_low[1] < prev_low[1]) and (last_low[2] > prev_low[2])


def sharpe_ratio_real(close: pd.Series, lookback: int = 126, rf_annual: float = 0.0) -> float:
    """Sharpe ratio ex-post over a fixed lookback (v7 FIX #4).

    Formula: ``(mean(r) * 252 - rf) / (std(r) * sqrt(252))``.

    Args:
        close: closing prices.
        lookback: window in bars. Default 126 (~6 months).
        rf_annual: annualized risk-free rate. Default 0.0 (proxy for
            short-term computation).

    Returns:
        Sharpe ratio. NaN if window insufficient or std-dev is zero.
    """
    if len(close) < lookback + 1:
        return math.nan
    r = close.pct_change().dropna().iloc[-lookback:]
    if len(r) < lookback // 2 or r.std() == 0:
        return math.nan
    return float((r.mean() * 252 - rf_annual) / (r.std() * np.sqrt(252)))


def compute_stock(close: pd.Series, high: pd.Series, low: pd.Series) -> dict[str, object]:
    """Bundle all per-stock metrics into a single dict.

    This is the integration entry point used by the screening pipeline (F.4)
    to compute a row of features per ticker. Output dict mirrors the v8 monolith
    keys for backwards compatibility with downstream scoring code.

    Args:
        close: closing prices.
        high: session highs.
        low: session lows.

    Returns:
        Dict with keys: price, slope, ewmac, vol, rsi, z, hurst, vr, hl,
        quality, pos_rev, atr, ret_5d, ret_3d, sharpe_real. Values may be
        NaN/False when the underlying window is insufficient.

    Note:
        ``quality`` and ``pos_rev`` are computed here for convenience but
        their canonical implementations live in :mod:`bronzebullballs.domain.regime`
        and :func:`positive_reversal` respectively. The import is deferred to
        avoid a cycle that the test suite would otherwise flag.
    """
    from bronzebullballs.domain.regime import quality_filter

    return {
        "price": float(close.iloc[-1]),
        "slope": adjusted_slope(close),
        "ewmac": ewmac_forecast(close),
        "vol": vol_ann(close),
        "rsi": float(rsi(close).iloc[-1]),
        "z": bollinger_z(close),
        "hurst": hurst_clipped(close),
        "vr": variance_ratio(close),
        "hl": half_life(close),
        "quality": quality_filter(close),
        "pos_rev": positive_reversal(close),
        "atr": float(atr(close, high, low, 20).iloc[-1]),
        "ret_5d": past_return(close, 5),
        "ret_3d": past_return(close, 3),
        "sharpe_real": sharpe_ratio_real(close),
    }
