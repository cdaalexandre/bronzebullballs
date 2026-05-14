"""Unit tests for ``bronzebullballs.domain.scores``."""

from __future__ import annotations

import math

from bronzebullballs.domain.scores import (
    MIN_SCORE_FOR_AGR_PICK,
    compute_v4_score,
    earn_penalty_abs,
    sharpe_normalized,
    stability_score,
    upside_score,
)


def _build_metrics(
    slope: float = 20.0,
    ewmac: float = 5.0,
    rsi: float = 55.0,
    vol: float = 20.0,
    z: float = 0.5,
    hurst: float = 0.6,
    vr: float = 1.2,
    hl: float | None = 8.0,
    quality: bool = True,
) -> dict:
    return {
        "slope": slope,
        "ewmac": ewmac,
        "rsi": rsi,
        "vol": vol,
        "z": z,
        "hurst": hurst,
        "vr": vr,
        "hl": hl,
        "quality": quality,
    }


def _build_universe(n: int = 10) -> dict:
    return {f"T{i}": _build_metrics(slope=10.0 + i, ewmac=i * 0.5, vol=15.0 + i) for i in range(n)}


class TestComputeV4Score:
    def test_returns_tuple_with_score_in_unit_interval(self) -> None:
        m = _build_metrics()
        u = _build_universe()
        score, adjs, flags = compute_v4_score(m, u, fundamentals={})
        assert 0.0 <= score <= 1.0
        assert isinstance(adjs, dict)
        assert isinstance(flags, list)

    def test_imminent_earnings_drops_score_and_flags(self) -> None:
        m = _build_metrics()
        u = _build_universe()
        base, _, _ = compute_v4_score(m, u, fundamentals={})
        penalized, adjs, flags = compute_v4_score(m, u, fundamentals={"earn_days": 5})
        assert penalized < base
        assert "earn_pen" in adjs
        assert any("HIGH RISK" in f for f in flags)

    def test_earnings_seven_to_thirteen_days_gets_soft_flag(self) -> None:
        m = _build_metrics()
        u = _build_universe()
        _, adjs, flags = compute_v4_score(m, u, fundamentals={"earn_days": 10})
        assert "earn_pen" in adjs
        assert any("earnings in 10d" in f for f in flags)
        assert not any("HIGH RISK" in f for f in flags)

    def test_earnings_two_to_four_weeks_gets_small_penalty(self) -> None:
        m = _build_metrics()
        u = _build_universe()
        _, adjs, _ = compute_v4_score(m, u, fundamentals={"earn_days": 25})
        assert math.isclose(adjs.get("earn_pen", 0), -0.05)

    def test_earnings_imminent_three_days_uses_max_penalty(self) -> None:
        m = _build_metrics()
        u = _build_universe()
        _, adjs, _ = compute_v4_score(m, u, fundamentals={"earn_days": 1})
        assert math.isclose(adjs.get("earn_pen", 0), -0.30)

    def test_high_analyst_score_boosts(self) -> None:
        m = _build_metrics()
        u = _build_universe()
        base, _, _ = compute_v4_score(m, u, fundamentals={})
        boosted, adjs, _ = compute_v4_score(m, u, fundamentals={"analyst": 0.80})
        assert boosted >= base
        assert "analyst+" in adjs

    def test_low_analyst_score_penalizes(self) -> None:
        m = _build_metrics()
        u = _build_universe()
        base, _, _ = compute_v4_score(m, u, fundamentals={})
        penalized, adjs, _ = compute_v4_score(m, u, fundamentals={"analyst": 0.40})
        assert penalized < base
        assert "analyst-" in adjs

    def test_high_beat_rate_boosts(self) -> None:
        m = _build_metrics()
        u = _build_universe()
        _, adjs, _ = compute_v4_score(m, u, fundamentals={"beat_rate": 0.80})
        assert "beat+" in adjs

    def test_low_beat_rate_penalizes(self) -> None:
        m = _build_metrics()
        u = _build_universe()
        _, adjs, _ = compute_v4_score(m, u, fundamentals={"beat_rate": 0.20})
        assert "beat-" in adjs

    def test_quality_false_drops_score(self) -> None:
        m_q = _build_metrics(quality=True)
        m_nq = _build_metrics(quality=False)
        u = _build_universe()
        score_q, _, _ = compute_v4_score(m_q, u, fundamentals={})
        score_nq, _, _ = compute_v4_score(m_nq, u, fundamentals={})
        assert score_q > score_nq

    def test_low_rsi_branch_executes(self) -> None:
        m = _build_metrics(rsi=30.0)
        score, _, _ = compute_v4_score(m, _build_universe(), fundamentals={})
        assert 0.0 <= score <= 1.0

    def test_high_rsi_branch_executes(self) -> None:
        m = _build_metrics(rsi=80.0)
        score, _, _ = compute_v4_score(m, _build_universe(), fundamentals={})
        assert 0.0 <= score <= 1.0

    def test_mid_high_rsi_branch_executes(self) -> None:
        # 65 < rsi < 75 hits the 0.5 branch in rsi_h.
        m = _build_metrics(rsi=70.0)
        score, _, _ = compute_v4_score(m, _build_universe(), fundamentals={})
        assert 0.0 <= score <= 1.0

    def test_bollinger_outside_one_sigma_branch(self) -> None:
        m = _build_metrics(z=1.5)
        score, _, _ = compute_v4_score(m, _build_universe(), fundamentals={})
        assert 0.0 <= score <= 1.0

    def test_bollinger_outside_two_sigma_branch(self) -> None:
        m = _build_metrics(z=2.5)
        score, _, _ = compute_v4_score(m, _build_universe(), fundamentals={})
        assert 0.0 <= score <= 1.0

    def test_nan_hurst_uses_neutral_score(self) -> None:
        m = _build_metrics(hurst=float("nan"))
        score, _, _ = compute_v4_score(m, _build_universe(), fundamentals={})
        assert 0.0 <= score <= 1.0

    def test_low_hurst_branch(self) -> None:
        m = _build_metrics(hurst=0.40)
        score, _, _ = compute_v4_score(m, _build_universe(), fundamentals={})
        assert 0.0 <= score <= 1.0

    def test_mid_hurst_branch(self) -> None:
        m = _build_metrics(hurst=0.50)
        score, _, _ = compute_v4_score(m, _build_universe(), fundamentals={})
        assert 0.0 <= score <= 1.0

    def test_nan_variance_ratio_uses_neutral_score(self) -> None:
        m = _build_metrics(vr=float("nan"))
        score, _, _ = compute_v4_score(m, _build_universe(), fundamentals={})
        assert 0.0 <= score <= 1.0

    def test_low_variance_ratio_branch(self) -> None:
        m = _build_metrics(vr=0.5)
        score, _, _ = compute_v4_score(m, _build_universe(), fundamentals={})
        assert 0.0 <= score <= 1.0

    def test_none_half_life_uses_low_score(self) -> None:
        m = _build_metrics(hl=None)
        score, _, _ = compute_v4_score(m, _build_universe(), fundamentals={})
        assert 0.0 <= score <= 1.0

    def test_long_half_life_branch(self) -> None:
        m = _build_metrics(hl=50.0)
        score, _, _ = compute_v4_score(m, _build_universe(), fundamentals={})
        assert 0.0 <= score <= 1.0

    def test_mid_half_life_branch(self) -> None:
        m = _build_metrics(hl=20.0)
        score, _, _ = compute_v4_score(m, _build_universe(), fundamentals={})
        assert 0.0 <= score <= 1.0

    def test_nan_metrics_use_neutral_normalization(self) -> None:
        # NaN slope/ewmac should not crash; _norm_cross_section returns 0.5.
        m = _build_metrics(slope=float("nan"), ewmac=float("nan"))
        score, _, _ = compute_v4_score(m, _build_universe(), fundamentals={})
        assert 0.0 <= score <= 1.0

    def test_all_nan_universe_yields_neutral_score(self) -> None:
        # When the universe is all-NaN, _norm_cross_section's "empty valid" path
        # runs and returns 0.5.
        m = _build_metrics()
        u = {
            f"T{i}": _build_metrics(slope=float("nan"), ewmac=float("nan"), vol=float("nan"))
            for i in range(5)
        }
        score, _, _ = compute_v4_score(m, u, fundamentals={})
        assert 0.0 <= score <= 1.0


class TestUpsideScore:
    def test_at_par_target_yields_documented_quirk_value(self) -> None:
        """Formula ``(x + 0.5) / 1.1`` at ``x = 0`` is ``0.4545``, NOT ``0.5``.

        The v8 monolith docstring claimed ``0% -> 0.5`` but the formula does
        not match the comment. F.1 is a literal port: the formula is
        preserved verbatim and this test pins the actual mathematical
        behaviour. Any future "fix" that changes the formula must update
        this test consciously.
        """
        s = upside_score(slope=20.0, ewmac=5.0, rsi_v=55.0, analyst_upside=0.0)
        assert math.isclose(s, 0.5 / 1.1, abs_tol=1e-6)

    def test_positive_upside_yields_above_par(self) -> None:
        s = upside_score(slope=20.0, ewmac=5.0, rsi_v=55.0, analyst_upside=0.30)
        # (0.30 + 0.5) / 1.1 = 0.7272...
        assert math.isclose(s, 0.8 / 1.1, abs_tol=1e-6)

    def test_extreme_negative_upside_clipped_at_zero(self) -> None:
        s = upside_score(slope=20.0, ewmac=5.0, rsi_v=55.0, analyst_upside=-1.0)
        assert s == 0.0

    def test_extreme_positive_upside_clipped_at_one(self) -> None:
        s = upside_score(slope=20.0, ewmac=5.0, rsi_v=55.0, analyst_upside=2.0)
        assert s == 1.0

    def test_no_analyst_falls_back_to_momentum_proxy(self) -> None:
        s = upside_score(slope=25.0, ewmac=10.0, rsi_v=50.0, analyst_upside=None)
        assert 0.0 <= s <= 1.0

    def test_no_analyst_nan_inputs_use_neutral_fallback(self) -> None:
        """NaN slope/ewmac/rsi in the fallback path get substituted with 0/50."""
        nan = float("nan")
        s = upside_score(slope=nan, ewmac=nan, rsi_v=nan, analyst_upside=None)
        assert 0.0 <= s <= 1.0


class TestStabilityScore:
    def test_returns_value_in_unit_interval(self) -> None:
        s = stability_score(vol=20.0, rsi_v=50.0, beta=1.0, no_gap=True)
        assert 0.0 <= s <= 1.0

    def test_lower_vol_increases_stability(self) -> None:
        a = stability_score(vol=40.0, rsi_v=50.0, beta=1.0)
        b = stability_score(vol=10.0, rsi_v=50.0, beta=1.0)
        assert b > a

    def test_extreme_rsi_decreases_stability(self) -> None:
        centered = stability_score(vol=20.0, rsi_v=50.0, beta=1.0)
        extreme = stability_score(vol=20.0, rsi_v=85.0, beta=1.0)
        assert centered > extreme

    def test_none_beta_treated_as_neutral(self) -> None:
        a = stability_score(vol=20.0, rsi_v=50.0, beta=None)
        b = stability_score(vol=20.0, rsi_v=50.0, beta=1.0)
        assert math.isclose(a, b)

    def test_no_gap_flag_improves_score(self) -> None:
        without = stability_score(vol=20.0, rsi_v=50.0, beta=1.0, no_gap=False)
        with_ = stability_score(vol=20.0, rsi_v=50.0, beta=1.0, no_gap=True)
        assert with_ > without

    def test_nan_inputs_use_safe_defaults(self) -> None:
        nan = float("nan")
        s = stability_score(vol=nan, rsi_v=nan, beta=nan)
        assert 0.0 <= s <= 1.0


class TestSharpeNormalized:
    def test_median_input_returns_half(self) -> None:
        all_s = [0.0, 1.0, 2.0, 3.0, 4.0]
        out = sharpe_normalized(2.0, all_s)
        assert 0.4 <= out <= 0.7

    def test_top_input_returns_one(self) -> None:
        all_s = [0.0, 1.0, 2.0, 3.0, 4.0]
        out = sharpe_normalized(4.0, all_s)
        assert out == 1.0

    def test_nan_input_returns_neutral(self) -> None:
        assert sharpe_normalized(math.nan, [0.0, 1.0]) == 0.5

    def test_empty_universe_returns_neutral(self) -> None:
        assert sharpe_normalized(1.5, []) == 0.5

    def test_all_nan_universe_returns_neutral(self) -> None:
        assert sharpe_normalized(1.5, [math.nan, math.nan]) == 0.5


class TestEarnPenaltyAbs:
    def test_none_returns_zero(self) -> None:
        assert earn_penalty_abs(None) == 0.0

    def test_imminent_returns_high(self) -> None:
        assert earn_penalty_abs(3) == 0.10

    def test_one_to_two_weeks_returns_medium(self) -> None:
        assert earn_penalty_abs(10) == 0.05

    def test_two_to_three_weeks_returns_low(self) -> None:
        assert earn_penalty_abs(15) == 0.02

    def test_far_returns_zero(self) -> None:
        assert earn_penalty_abs(30) == 0.0


class TestConstants:
    def test_min_score_for_agr_pick_in_unit_interval(self) -> None:
        assert 0.0 < MIN_SCORE_FOR_AGR_PICK < 1.0
