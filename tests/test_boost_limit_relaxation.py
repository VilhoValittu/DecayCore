from types import SimpleNamespace

import numpy as np
import pytest

from decaycore.auto_mode.candidate_generation import _build_auto_mode_candidates, _build_auto_mode_candidates_local
from decaycore.auto_mode.optuna_backend import (
    _auto_optuna_constraint_thresholds,
    _auto_optuna_constraint_vector_from_metrics,
    _auto_optuna_trial_distributions,
)
from decaycore.config.mode_policy import apply_mode_to_cfg
from decaycore.config.models import FilterConfig
from decaycore.dsp.correction_mag import _apply_peak_priority_error_shaping
from decaycore.dsp.gain_policy import apply_cuts_only_guard
from decaycore.dsp.limits import soft_clip_gain
from decaycore.engine_build import _apply_max_boost_safety_cap, _build_config_apply_unsafe_raw


def test_advanced_auto_boost_safety_cap_allows_12_db() -> None:
    cfg = SimpleNamespace(max_boost_db=12.0)

    _apply_max_boost_safety_cap(cfg, max_safe_boost=12.0, unsafe_raw=False)

    assert float(cfg.max_boost_db) == pytest.approx(12.0)
    assert float(cfg.max_boost_db_user) == pytest.approx(12.0)
    assert float(cfg.max_safe_boost_db) == pytest.approx(12.0)


def test_unsafe_raw_bypasses_safe_cap_but_preserves_user_boost_limit() -> None:
    cfg = SimpleNamespace(
        max_boost_db=12.0,
        max_cut_db=15.0,
        max_slope_db_per_oct=24.0,
        max_slope_boost_db_per_oct=24.0,
        max_slope_cut_db_per_oct=24.0,
        reg_strength=30.0,
        low_bass_cut_enable=True,
        low_bass_cut_hz=18.0,
        low_bass_cut_strength=1.0,
        exc_prot=True,
        bass_boost_cap_enable=True,
        bass_boost_post_restore_enable=True,
        acoustic_authority_limits_enable=True,
        bass_smooth_adaptive=True,
        enable_ir_pre_energy_guard=True,
    )

    _build_config_apply_unsafe_raw(
        cfg,
        {"unsafe_raw_dsp": True},
        mode_u="ADVANCED",
        auto_goal="flat",
        max_safe_boost=4.0,
    )

    assert bool(cfg.unsafe_raw_dsp) is True
    assert float(cfg.max_boost_db_user) == pytest.approx(12.0)
    assert float(cfg.max_safe_boost_db) == pytest.approx(4.0)
    assert float(cfg.max_boost_db) == pytest.approx(12.0)
    assert bool(cfg.acoustic_authority_limits_enable) is False


def test_basic_mode_still_clamps_max_boost_to_4_db() -> None:
    cfg = FilterConfig(max_boost_db=12.0)

    apply_mode_to_cfg(cfg, "BASIC", apply_defaults=False)

    assert float(cfg.max_boost_db) == pytest.approx(4.0)


def test_positive_soft_clip_preserves_boost_up_to_cap() -> None:
    gain = np.asarray([6.0, 12.0, 18.0, -18.0], dtype=float)

    out = soft_clip_gain(gain, max_boost_db=12.0, max_cut_db=15.0)

    assert float(out[0]) == pytest.approx(6.0)
    assert float(out[1]) == pytest.approx(12.0)
    assert float(out[2]) > 12.0
    assert float(out[2]) < 18.0
    assert float(out[3]) < -15.0
    assert float(out[3]) > -18.0


def test_peak_priority_error_shaping_preserves_boost_up_to_cap() -> None:
    err = np.asarray([6.0, 12.0, 18.0], dtype=float)
    freq = np.asarray([40.0, 80.0, 160.0], dtype=float)
    cfg = SimpleNamespace(max_boost_db=12.0, peak_priority_enable=True, peak_priority_strength=1.0)

    out = _apply_peak_priority_error_shaping(
        err,
        freq,
        cfg,
        {},
        mask_c=np.ones_like(freq, dtype=bool),
        logger=None,
    )

    assert float(out[0]) == pytest.approx(6.0)
    assert float(out[1]) == pytest.approx(12.0)
    assert float(out[2]) > 12.0
    assert float(out[2]) < 18.0


def test_low_bass_cuts_only_guard_still_blocks_boost() -> None:
    curve = np.asarray([8.0, 12.0, -2.0], dtype=float)
    mask = np.asarray([True, True, True], dtype=bool)
    guard_mask = np.asarray([True, True, False], dtype=bool)

    guarded, meta = apply_cuts_only_guard(curve, mask=mask, guard_mask=guard_mask)

    assert np.allclose(guarded, [0.0, 0.0, -2.0], atol=1e-12, rtol=0.0)
    assert int(meta["boost_clamped_bins"]) == 2


def test_auto_candidates_allow_max_boost_up_to_12_db() -> None:
    candidates = _build_auto_mode_candidates(
        {"auto_goal": "flat", "max_boost": 12.0},
        n_trials=64,
        seed=123,
    )

    boosts = [float(c.get("max_boost", 0.0)) for c in candidates]
    assert all(3.0 <= v <= 12.0 for v in boosts)
    assert any(v > 8.0 for v in boosts)


def test_local_auto_candidates_preserve_boost_above_old_8_db_cap() -> None:
    candidates = _build_auto_mode_candidates_local(
        {"auto_goal": "flat", "max_boost": 11.5},
        {"max_boost": 11.5},
        n_trials=8,
        seed=123,
    )

    boosts = [float(c.get("max_boost", 0.0)) for c in candidates]
    assert all(5.0 <= v <= 12.0 for v in boosts)
    assert float(candidates[0]["max_boost"]) == pytest.approx(11.5)


def test_optuna_search_space_and_constraints_allow_12_db() -> None:
    class _FakeDistributions:
        class FloatDistribution:
            def __init__(self, low, high, step=None):
                self.low = float(low)
                self.high = float(high)
                self.step = step

        class CategoricalDistribution:
            def __init__(self, choices):
                self.choices = list(choices)

    class _FakeOptuna:
        distributions = _FakeDistributions

    dists = _auto_optuna_trial_distributions(
        _FakeOptuna,
        params={"max_boost": 12.0},
        base_data={},
    )
    thresholds = _auto_optuna_constraint_thresholds({}, scope="refine")
    at_limit = _auto_optuna_constraint_vector_from_metrics(
        {"max_net_boost_db": 12.0},
        max_mode_ripple_db=1.0,
        max_events_severity=1.0,
        max_net_boost_db=float(thresholds["max_net_boost_db"]),
    )
    over_limit = _auto_optuna_constraint_vector_from_metrics(
        {"max_net_boost_db": 13.25},
        max_mode_ripple_db=1.0,
        max_events_severity=1.0,
        max_net_boost_db=float(thresholds["max_net_boost_db"]),
    )

    assert dists is not None
    assert float(dists["max_boost"].high) == pytest.approx(12.0)
    assert float(thresholds["max_net_boost_db"]) == pytest.approx(12.0)
    assert at_limit[2] == pytest.approx(0.0)
    assert over_limit[2] == pytest.approx(1.25)
