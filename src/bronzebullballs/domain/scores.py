"""Cross-sectional scoring of screening candidates.

This module computes the scalar scores that drive ranking in the screening
pipeline (FASE F.4). All functions are pure: they operate on already-computed
metrics dicts and return floats or simple containers.

Five public functions, organized by concern:

    - :func:`compute_v4_score` -- the master scorecard. Combines momentum,
      trend, mean-reversion and quality signals into one float in
      ``[0, 1]``, then applies fundamentals adjustments (earnings, analyst,
      beat rate).
    - :func:`upside_score` -- analyst-target-based upside, with a momentum
      proxy fallback when coverage is missing (v7 FIX #3).
    - :func:`stability_score` -- vol + RSI-centeredness + beta + gap-free.
    - :func:`sharpe_normalized` -- cross-sectional percentile rank of the
      ex-post Sharpe ratio (v7 FIX #4).
    - :func:`earn_penalty_abs` -- additive penalty for upcoming earnings,
      separate from the multiplicative penalty applied inside
      :func:`compute_v4_score`.

Literal port from ``corpus_master_v8.py`` lines 120, 503-644.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

#: Minimum V4 score required to qualify as an AGGRESSIVE-profile pick
#: (Aronson Cap. 6 evidence threshold). Below this, the AGR slot is reported
#: as "[SEM CANDIDATO]".
MIN_SCORE_FOR_AGR_PICK: float = 0.50


def _norm_cross_section(
    v: float,
    all_v: list[float],
    higher_is_better: bool = True,
) -> float:
    """Return the percentile rank of ``v`` against ``all_v``, in ``[0, 1]``.

    Missing or NaN values get a neutral 0.5. If all reference values are
    NaN, also returns 0.5.
    """
    if pd.isna(v):
        return 0.5
    valid = [x for x in all_v if not pd.isna(x)]
    if not valid:
        return 0.5
    rk = sum(1 for x in valid if x <= v) / len(valid)
    return rk if higher_is_better else (1 - rk)


def compute_v4_score(
    metrics: Mapping[str, Any],
    universe_metrics: Mapping[str, Mapping[str, Any]],
    fundamentals: Mapping[str, Any],
) -> tuple[float, dict[str, float], list[str]]:
    """V4 scorecard: V3 base + fundamentals adjustments.

    The V3 base is a fixed-weight linear combination of nine sub-scores:

        - slope (Clenow Cap. 3, normalized cross-sectionally), weight 0.25.
        - EWMAC (Carver Cap. 7, normalized), weight 0.20.
        - RSI healthiness (40 <= RSI <= 65 ideal), weight 0.10.
        - Bollinger z-score (abs(z) < 1 ideal), weight 0.05.
        - Volatility (cross-sectionally low is better), weight 0.10.
        - Quality flag (Clenow Cap. 5), weight 0.10.
        - Hurst (Chan Cap. 2, > 0.55 trending is best), weight 0.05.
        - Variance ratio (Chan Cap. 2, > 1.1 trending is best), weight 0.10.
        - Half-life (Chan Cap. 2, short is best), weight 0.05.

    V4 then applies up to three multiplicative adjustments:

        - Earnings penalty: 30 % / 20 % / 10 % / 5 % / 0 % for earnings in
          < 3d / < 7d / < 14d / < 30d / further.
        - Analyst score boost (factor 1.05) if > 0.75, penalty (factor 0.95)
          if < 0.50.
        - Beat rate boost (factor 1.03) if >= 0.75, penalty (factor 0.97) if
          <= 0.25.

    Args:
        metrics: per-stock indicators dict, as returned by
            :func:`bronzebullballs.domain.indicators.compute_stock`.
        universe_metrics: dict mapping each ticker to its own metrics dict.
            Used for cross-sectional normalization.
        fundamentals: per-stock fundamentals (earn_days, analyst, beat_rate).

    Returns:
        Tuple ``(score, adjustments, flags)``:

            - ``score`` in ``[0, 1]`` -- the final V4 score.
            - ``adjustments`` -- dict mapping adjustment name to delta (e.g.
              ``{"earn_pen": -0.10, "analyst+": 0.05}``).
            - ``flags`` -- human-readable flags for the report layer (e.g.
              ``"EARNINGS in 5d (HIGH RISK)"``).
    """
    # --- V3 base (momentum scorecard) ---
    slope_n = _norm_cross_section(metrics["slope"], [m["slope"] for m in universe_metrics.values()])
    ewmac_n = _norm_cross_section(metrics["ewmac"], [m["ewmac"] for m in universe_metrics.values()])

    rsi_v = float(metrics["rsi"])
    rsi_h = 1.0 if 40 <= rsi_v <= 65 else (0.7 if rsi_v < 40 else 0.5 if rsi_v < 75 else 0.2)

    z = float(metrics["z"])
    bb_s = 1.0 if abs(z) < 1 else (0.6 if abs(z) < 2 else 0.2)

    vol_n = _norm_cross_section(
        metrics["vol"],
        [m["vol"] for m in universe_metrics.values()],
        higher_is_better=False,
    )

    q = 1.0 if metrics["quality"] else 0.0

    h = metrics["hurst"]
    if pd.isna(h):
        hurst_s = 0.5
    else:
        hurst_s = 1.0 if h > 0.55 else (0.5 if h >= 0.45 else 0.3)

    vr = metrics["vr"]
    if pd.isna(vr):
        vr_s = 0.5
    else:
        vr_s = 1.0 if vr > 1.1 else (0.6 if vr > 0.95 else 0.2)

    hl = metrics["hl"]
    if hl is None or pd.isna(hl):
        hl_s = 0.4
    else:
        hl_s = 1.0 if 0 < hl < 10 else (0.7 if hl < 30 else 0.4)

    v3 = (
        0.25 * slope_n
        + 0.20 * ewmac_n
        + 0.10 * rsi_h
        + 0.05 * bb_s
        + 0.10 * vol_n
        + 0.10 * q
        + 0.05 * hurst_s
        + 0.10 * vr_s
        + 0.05 * hl_s
    )

    # --- V4 adjustments ---
    v4 = v3
    adjs: dict[str, float] = {}
    flags: list[str] = []

    # Earnings penalty (multiplicative)
    edays = fundamentals.get("earn_days")
    if edays is not None:
        if edays < 3:
            pen = 0.30
        elif edays < 7:
            pen = 0.20
        elif edays < 14:
            pen = 0.10
        elif edays < 30:
            pen = 0.05
        else:
            pen = 0.0
        if pen > 0:
            v4 *= 1 - pen
            adjs["earn_pen"] = -pen
            if edays < 7:
                flags.append(f"EARNINGS in {edays}d (HIGH RISK)")
            elif edays < 14:
                flags.append(f"earnings in {edays}d")

    # Analyst score
    an = fundamentals.get("analyst")
    if an is not None:
        if an > 0.75:
            v4 = min(1.0, v4 * 1.05)
            adjs["analyst+"] = 0.05
        elif an < 0.50:
            v4 *= 0.95
            adjs["analyst-"] = -0.05

    # Beat rate
    beat = fundamentals.get("beat_rate")
    if beat is not None:
        if beat >= 0.75:
            v4 = min(1.0, v4 * 1.03)
            adjs["beat+"] = 0.03
        elif beat <= 0.25:
            v4 *= 0.97
            adjs["beat-"] = -0.03

    return v4, adjs, flags


def upside_score(
    slope: float,
    ewmac: float,
    rsi_v: float,
    analyst_upside: float | None = None,
) -> float:
    """v7 FIX #3 upside: real analyst target when available, momentum proxy otherwise.

    Math when ``analyst_upside`` is present: linear remap from ``[-0.5, +0.6]``
    to ``[0, 1]``, clipped at both ends. Specifically:

        ``score = clip((analyst_upside + 0.5) / 1.1, 0, 1)``

    Note:
        The v8 monolith docstring claimed the mapping was ``0% -> 0.5``,
        ``+30% -> 0.85``, etc. The actual formula above does NOT match those
        comments: at ``analyst_upside = 0.0`` it returns ``0.4545``, not
        ``0.5``. F.1 is a literal port, so the formula is preserved exactly
        and the documentation is corrected here. Whether the formula should
        be "fixed" to match the original intent is a v9-level design call,
        not a refactor concern; tracked as documented quirk for a future
        rev (see §15 in the project Custom Instructions).

    When ``analyst_upside`` is missing (typical for micro-caps off the analyst
    radar), the function falls back to a momentum-proxy: the mean of
    normalized slope, normalized EWMAC, and RSI headroom.

    Args:
        slope: Clenow adjusted slope (percentage).
        ewmac: Carver EWMAC forecast (-20 to +20 expected range).
        rsi_v: latest RSI value (0-100).
        analyst_upside: ``(target_price - last_price) / last_price`` from
            analysts, or ``None`` if unavailable.

    Returns:
        Score in ``[0, 1]``.
    """
    if analyst_upside is not None and not pd.isna(analyst_upside):
        return max(0.0, min(1.0, (analyst_upside + 0.50) / 1.10))

    if pd.isna(slope):
        slope = 0.0
    if pd.isna(ewmac):
        ewmac = 0.0
    if pd.isna(rsi_v):
        rsi_v = 50.0
    sn = max(0.0, min(1.0, slope / 50))
    en = max(0.0, min(1.0, (ewmac + 20) / 40))
    rsi_space = max(0.0, min(1.0, (75 - rsi_v) / 30))
    return (sn + en + rsi_space) / 3


def stability_score(
    vol: float,
    rsi_v: float,
    beta: float | None,
    no_gap: bool = False,
) -> float:
    """Stability composite: low vol + centered RSI + low beta + gap-free.

    Args:
        vol: annualized volatility, percentage.
        rsi_v: latest RSI.
        beta: market beta. ``None`` is treated as 1.0 (neutral).
        no_gap: True if the stock passed the gap check in
            :func:`bronzebullballs.domain.regime.quality_filter`.

    Returns:
        Score in ``[0, 1]``. Higher = more stable.
    """
    if pd.isna(vol):
        vol = 30.0
    if pd.isna(rsi_v):
        rsi_v = 50.0
    if beta is None or pd.isna(beta):
        beta = 1.0
    vol_s = max(0.0, min(1.0, 1 - vol / 40))
    rsi_c = max(0.0, min(1.0, 1 - abs(rsi_v - 50) / 30))
    beta_s = max(0.0, min(1.0, 1 - beta / 2))
    ng = 1.0 if no_gap else 0.5
    return (vol_s + rsi_c + beta_s + ng) / 4


def sharpe_normalized(sharpe_real: float, all_sharpes: list[float]) -> float:
    """Cross-sectional percentile rank of the ex-post Sharpe ratio (v7 FIX #4).

    The monolith's v6 ``sharpe_normalized`` was a custom blend of slope, EWMAC,
    return and volatility -- a momentum proxy disguised as Sharpe. The v7 fix
    (preserved here) computes the **real** Sharpe ratio in
    :func:`bronzebullballs.domain.indicators.sharpe_ratio_real` and uses this
    function only to convert it to a cross-sectional rank.

    Args:
        sharpe_real: this stock's ex-post Sharpe.
        all_sharpes: universe of Sharpes for rank reference.

    Returns:
        Percentile rank in ``[0, 1]``, or 0.5 when input or reference is
        missing.
    """
    if pd.isna(sharpe_real):
        return 0.5
    valid = [s for s in all_sharpes if not pd.isna(s)]
    if not valid:
        return 0.5
    rk = sum(1 for s in valid if s <= sharpe_real) / len(valid)
    return float(rk)


def earn_penalty_abs(days: int | None) -> float:
    """Additive earnings penalty (independent of the multiplicative one in V4).

    Used by report-layer ranking logic that wants a flat additive haircut
    rather than a multiplicative factor.

    Args:
        days: days until next earnings, or ``None`` if unknown.

    Returns:
        Penalty in ``[0, 0.10]`` to be **subtracted** from another score.
    """
    if days is None:
        return 0.0
    if days < 7:
        return 0.10
    if days < 14:
        return 0.05
    if days < 21:
        return 0.02
    return 0.0
