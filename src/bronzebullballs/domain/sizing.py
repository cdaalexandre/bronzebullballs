"""Position sizing for screening picks (Clenow Cap. 11 + Carver Cap. 9-10).

Two distinct sizing schemes, used in different places by the pipeline:

    - :func:`clenow_risk_parity_size` -- ATR-based "equal risk dollars"
      sizing (Clenow Cap. 11). Each stock contributes the same daily-dollar
      risk; volatile names get fewer shares, calm names get more. Used in
      walk-forward (FASE F.2) for the ATR-risk-parity weighting method.
    - :func:`carver_position_size` -- portfolio volatility-targeting (Carver
      Cap. 9-10). Sizes a single position so that, under the assumption of
      independent moves, the portfolio sits at a target annualized volatility.
      Used in screening (FASE F.4) for the per-profile recommendations.

Literal port from ``corpus_master_v8.py`` lines 108, 858-873, 1460-1493.
"""

from __future__ import annotations

import math

#: Default daily-dollar risk per stock, expressed as a fraction of total
#: capital (Clenow Cap. 11; 10 bps = 0.001). The monolith's ``WF_RISK_FACTOR``.
DEFAULT_RISK_FACTOR: float = 0.001


def clenow_risk_parity_size(
    price: float,
    atr_dollar: float,
    capital: float,
    risk_factor: float = DEFAULT_RISK_FACTOR,
) -> tuple[int, float]:
    """Clenow Cap. 11 ATR risk-parity sizing.

    Logic::

        target_daily_dollar = capital * risk_factor
        n_shares            = target_daily_dollar / ATR_dollar
        notional            = n_shares * price

    Each name in the portfolio receives the same target daily-dollar risk;
    the resulting portfolio is risk-parity in the simple "equal dollar vol"
    sense. The ``int(...)`` truncation can produce zero shares for very
    expensive stocks; we floor at 1 share to keep the position non-trivial,
    matching the monolith's behaviour.

    Args:
        price: current share price.
        atr_dollar: ATR expressed in price units (i.e. dollars per share).
        capital: total portfolio capital.
        risk_factor: target daily-dollar risk as a fraction of capital.
            Default :data:`DEFAULT_RISK_FACTOR` (10 bps).

    Returns:
        Tuple ``(n_shares, notional)``. Returns ``(0, 0.0)`` if ``price`` or
        ``atr_dollar`` is non-positive.
    """
    if atr_dollar <= 0 or price <= 0:
        return 0, 0.0
    target_daily = capital * risk_factor
    n_shares = max(1, int(target_daily / atr_dollar))
    notional = n_shares * price
    return n_shares, notional


def carver_position_size(
    price: float,
    atr_pct: float,
    capital: float,
    vol_target_annual: float = 0.20,
    max_pct_single: float = 0.35,
) -> tuple[int, float, float, float]:
    """Carver Cap. 9-10 volatility-targeting sizing.

    Logic::

        daily_vol_target = vol_target_annual / sqrt(252)    # ~1.26 % for 20 % annual
        cash_at_risk     = capital * daily_vol_target
        atr_dollar       = price * atr_pct
        raw_shares       = cash_at_risk / atr_dollar
        raw_pct          = raw_shares * price / capital
        pct              = min(raw_pct, max_pct_single)
        n_shares         = int(capital * pct / price)

    The position is sized so that its expected daily move (proxied by ATR)
    equals the portfolio's daily volatility budget, then capped at
    ``max_pct_single`` of capital to avoid catastrophic single-name exposure.

    Args:
        price: current share price.
        atr_pct: ATR / price (fractional daily vol approximation).
        capital: total portfolio capital.
        vol_target_annual: target annualized portfolio volatility. Default
            0.20 (conservative for an active long-only book).
        max_pct_single: maximum fraction of capital in a single name.
            Default 0.35 (generous).

    Returns:
        Tuple ``(n_shares, pct_capital, actual_notional, daily_risk)``:

            - ``n_shares`` -- integer share count.
            - ``pct_capital`` -- realized fraction of capital, after the cap.
            - ``actual_notional`` -- ``n_shares * price``.
            - ``daily_risk`` -- ``n_shares * atr_dollar``, expected daily $.
    """
    daily_vol_target = vol_target_annual / math.sqrt(252)
    cash_at_risk = capital * daily_vol_target
    atr_dollar = price * atr_pct
    if atr_dollar <= 0:
        return 0, 0.0, 0.0, 0.0
    raw_shares = cash_at_risk / atr_dollar
    raw_notional = raw_shares * price
    raw_pct = raw_notional / capital
    pct = min(raw_pct, max_pct_single)
    notional = capital * pct
    n_shares = int(notional / price)
    actual_notional = n_shares * price
    daily_risk = n_shares * atr_dollar
    return n_shares, pct, actual_notional, daily_risk
