"""Discrete trade-signal labelling.

The :func:`signal_label` function computes a 5-factor binary scorecard that
maps to one of five labels: ``BUY+``, ``BUY``, ``HOLD``, ``SELL``, ``SELL+``.

This is a deliberate **refactor** from the monolith: the original v8 function
(``corpus_master_v8.py`` lines 289-329) returned an ANSI-colored string
glued to the score. That mixed domain logic (the 5-factor count) with
presentation (terminal colors), violating section 3.1 of the project
spec ("report consumes already-computed data and formats; it does not
compute").

Here the domain returns the plain label and the score; coloring is left to
:mod:`bronzebullballs.report.formatters` (FASE F.4), which will map
``label -> color`` via a small lookup table.

Reference: original logic in ``corpus_master_v8.py`` lines 289-329.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

#: Ordered list of possible labels, lowest n to highest n.
_LABELS_BY_N: dict[int, str] = {
    0: "SELL+",
    1: "SELL+",
    2: "SELL",
    3: "HOLD",
    4: "BUY",
    5: "BUY+",
}


def signal_label(
    metrics: Mapping[str, Any],
    fund: Mapping[str, Any],
    spy_regime_on: bool,
    vix_val: float,
) -> tuple[str, int]:
    """Compute a 5-factor scorecard and the matching discrete label.

    Each of the five conditions below contributes 1 to ``n`` when satisfied:

        1. EWMAC > 0           -- trend up.
        2. Slope > 10 %        -- trending strongly.
        3. RSI < 70            -- not overbought.
        4. Earnings ≥ 14 days  -- no imminent event risk
           (``None`` counts as safe).
        5. Macro favorable     -- regime ON AND VIX < 25.

    Score ``n`` maps to a label:

        - 5     → ``BUY+``
        - 4     → ``BUY``
        - 3     → ``HOLD``
        - 2     → ``SELL``
        - 0-1   → ``SELL+``

    Args:
        metrics: per-stock indicators dict. Reads ``ewmac``, ``slope``,
            ``rsi``. Missing or ``None`` values fail the corresponding check
            (count 0).
        fund: fundamentals dict. Reads ``earn_days``; ``None`` is treated as
            safe (no event scheduled).
        spy_regime_on: result of :func:`bronzebullballs.domain.regime.regime_on`.
        vix_val: latest CBOE VIX close (or any equivalent fear gauge).

    Returns:
        Tuple ``(label, n)`` where ``label`` is one of
        ``{"BUY+", "BUY", "HOLD", "SELL", "SELL+"}`` and ``n`` is the
        underlying integer score in ``[0, 5]``.
    """
    n = 0
    ewmac = metrics.get("ewmac")
    if ewmac is not None and ewmac > 0:
        n += 1
    slope = metrics.get("slope")
    if slope is not None and slope > 10:
        n += 1
    rsi_v = metrics.get("rsi")
    if rsi_v is not None and rsi_v < 70:
        n += 1
    earn = fund.get("earn_days")
    if earn is None or earn >= 14:
        n += 1
    if spy_regime_on and vix_val < 25:
        n += 1

    return _LABELS_BY_N[n], n
