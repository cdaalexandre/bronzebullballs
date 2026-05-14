"""Unit tests for ``bronzebullballs.domain.signals``.

Verifies the 5-factor scorecard and the n -> label mapping. Crucially,
checks that the function returns the plain label name (no ANSI codes),
since FASE F.1 refactored the monolith to separate domain from presentation.
"""

from __future__ import annotations

from bronzebullballs.domain.signals import signal_label


def _make_all_positive() -> tuple[dict, dict, bool, float]:
    """All five factors satisfied -> n == 5 -> BUY+."""
    metrics = {"ewmac": 5.0, "slope": 15.0, "rsi": 60.0}
    fund = {"earn_days": 30}
    return metrics, fund, True, 18.0


class TestSignalLabel:
    def test_all_factors_positive_yields_buy_plus(self) -> None:
        metrics, fund, regime, vix = _make_all_positive()
        label, n = signal_label(metrics, fund, regime, vix)
        assert (label, n) == ("BUY+", 5)

    def test_four_factors_positive_yields_buy(self) -> None:
        # Drop macro: VIX 30 fails the regime+VIX gate.
        metrics, fund, regime, _ = _make_all_positive()
        label, n = signal_label(metrics, fund, regime, vix_val=30.0)
        assert (label, n) == ("BUY", 4)

    def test_three_factors_positive_yields_hold(self) -> None:
        # Drop macro AND earnings risk.
        metrics, _, regime, _ = _make_all_positive()
        label, n = signal_label(metrics, {"earn_days": 5}, regime, vix_val=30.0)
        assert (label, n) == ("HOLD", 3)

    def test_two_factors_positive_yields_sell(self) -> None:
        # Keep slope, RSI; drop EWMAC, earnings, macro.
        metrics = {"ewmac": -1.0, "slope": 15.0, "rsi": 60.0}
        label, n = signal_label(metrics, {"earn_days": 5}, False, vix_val=30.0)
        assert (label, n) == ("SELL", 2)

    def test_zero_factors_yields_sell_plus(self) -> None:
        metrics = {"ewmac": -1.0, "slope": 5.0, "rsi": 80.0}
        label, n = signal_label(metrics, {"earn_days": 5}, False, vix_val=30.0)
        assert (label, n) == ("SELL+", 0)

    def test_missing_earn_days_counts_as_safe(self) -> None:
        # earn_days=None must NOT fail the earnings check.
        metrics = {"ewmac": -1.0, "slope": 5.0, "rsi": 80.0}
        label, n = signal_label(metrics, {}, False, vix_val=30.0)
        assert n == 1  # earnings-safe contributes 1
        assert label == "SELL+"

    def test_no_ansi_codes_in_label(self) -> None:
        # Regression: monolith returned strings like "\x1b[92m BUY  \x1b[0m".
        # The domain refactor strips all that.
        metrics, fund, regime, vix = _make_all_positive()
        label, _ = signal_label(metrics, fund, regime, vix)
        assert "\x1b" not in label
        assert label.strip() == label

    def test_threshold_values_are_strict_inequalities(self) -> None:
        # slope > 10 strict: slope=10 must NOT count.
        # rsi < 70 strict: rsi=70 must NOT count.
        # earnings >= 14: edge-inclusive.
        metrics = {"ewmac": 0.0, "slope": 10.0, "rsi": 70.0}
        fund = {"earn_days": 14}
        _, n = signal_label(metrics, fund, False, vix_val=30.0)
        # EWMAC=0 not >0 (0), slope=10 not >10 (0), rsi=70 not <70 (0),
        # earnings=14 >= 14 (+1), macro fails (0).
        assert n == 1
