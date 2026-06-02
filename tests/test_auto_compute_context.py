# DecayCore
# Tests for AutoRunComputeContext and _MEAS_FIXED_CACHE in run_preprocess.

from __future__ import annotations

import numpy as np
import pytest

from decaycore.auto_mode.compute_context import (
    AutoRunComputeContext,
    cached_interp,
    get_cached_target,
    get_cached_metric,
    get_rfft_freq_grid,
)
from decaycore.dsp.dsp_preprocess import (
    _MEAS_FIXED_CACHE,
    clear_meas_fixed_cache,
    clear_preprocess_cache,
    get_meas_fixed_cache_stats,
    run_preprocess,
)


# ---------------------------------------------------------------------------
# AutoRunComputeContext dataclass
# ---------------------------------------------------------------------------

def _make_ctx(**kwargs) -> AutoRunComputeContext:
    defaults = dict(
        fs=48000,
        taps=65536,
        measurement_signature="sig_test",
        filter_key="mixed",
        compat_version="v1",
    )
    defaults.update(kwargs)
    return AutoRunComputeContext(**defaults)


def test_ctx_initial_state():
    ctx = _make_ctx()
    assert ctx.debug_stats == {}
    assert ctx.trial_cache == {}


def test_bump_increments():
    ctx = _make_ctx()
    ctx.bump("hits")
    ctx.bump("hits")
    ctx.bump("misses")
    assert ctx.debug_stats["hits"] == 2
    assert ctx.debug_stats["misses"] == 1


def test_begin_trial_clears_trial_cache():
    ctx = _make_ctx()
    ctx.trial_cache[("some_key",)] = 42
    ctx.begin_trial(1)
    assert ("some_key",) not in ctx.trial_cache
    assert ctx.trial_cache[("trial_number",)] == 1


def test_begin_trial_bumps_counter():
    ctx = _make_ctx()
    ctx.begin_trial(1)
    ctx.begin_trial(2)
    assert ctx.debug_stats.get("trials_started", 0) == 2


# ---------------------------------------------------------------------------
# get_rfft_freq_grid
# ---------------------------------------------------------------------------

def test_freq_grid_returns_correct_values():
    ctx = _make_ctx(fs=48000)
    grid = get_rfft_freq_grid(ctx, 1024)
    expected = np.fft.rfftfreq(1024, 1.0 / 48000)
    np.testing.assert_array_equal(grid, expected)


def test_freq_grid_cache_hit_returns_same_object():
    ctx = _make_ctx(fs=48000)
    g1 = get_rfft_freq_grid(ctx, 1024)
    g2 = get_rfft_freq_grid(ctx, 1024)
    assert g1 is g2
    assert ctx.debug_stats.get("freq_grid_hits", 0) == 1


def test_freq_grid_different_nfft_different_entry():
    ctx = _make_ctx(fs=48000)
    g1 = get_rfft_freq_grid(ctx, 1024)
    g2 = get_rfft_freq_grid(ctx, 2048)
    assert g1 is not g2
    assert len(g1) != len(g2)


def test_freq_grid_different_fs_different_entry():
    ctx48 = _make_ctx(fs=48000)
    ctx96 = _make_ctx(fs=96000)
    g48 = get_rfft_freq_grid(ctx48, 1024)
    g96 = get_rfft_freq_grid(ctx96, 1024)
    assert g48[-1] != g96[-1]


# ---------------------------------------------------------------------------
# cached_interp
# ---------------------------------------------------------------------------

def test_cached_interp_correct_values():
    ctx = _make_ctx()
    x_new = np.linspace(20.0, 20000.0, 100)
    x_old = np.linspace(20.0, 20000.0, 50)
    y_old = np.sin(x_old / 1000.0)
    result = cached_interp(ctx, "test", x_new, x_old, y_old)
    expected = np.interp(x_new, x_old, y_old)
    np.testing.assert_allclose(result, expected, rtol=1e-12)


def test_cached_interp_second_call_returns_same_object():
    ctx = _make_ctx()
    x_new = np.linspace(20.0, 20000.0, 100)
    x_old = np.linspace(20.0, 20000.0, 50)
    y_old = np.sin(x_old / 1000.0)
    r1 = cached_interp(ctx, "test", x_new, x_old, y_old)
    r2 = cached_interp(ctx, "test", x_new, x_old, y_old)
    assert r1 is r2
    assert ctx.debug_stats.get("interp_hits", 0) == 1


def test_cached_interp_different_name_different_entry():
    ctx = _make_ctx()
    x_new = np.linspace(20.0, 20000.0, 100)
    x_old = np.linspace(20.0, 20000.0, 50)
    y_old = np.sin(x_old / 1000.0)
    r1 = cached_interp(ctx, "mag", x_new, x_old, y_old)
    r2 = cached_interp(ctx, "phase", x_new, x_old, y_old)
    assert r1 is not r2


# ---------------------------------------------------------------------------
# get_cached_target
# ---------------------------------------------------------------------------

def test_cached_target_build_fn_called_once():
    ctx = _make_ctx()
    call_count = {"n": 0}

    def build():
        call_count["n"] += 1
        return np.zeros(100)

    key = ("target", "flat", 48000, 65536)
    t1 = get_cached_target(ctx, target_key=key, build_fn=build)
    t2 = get_cached_target(ctx, target_key=key, build_fn=build)
    assert call_count["n"] == 1
    assert t1 is t2


def test_cached_target_different_key_calls_build():
    ctx = _make_ctx()
    call_count = {"n": 0}

    def build():
        call_count["n"] += 1
        return np.zeros(100)

    get_cached_target(ctx, target_key=("t", "a"), build_fn=build)
    get_cached_target(ctx, target_key=("t", "b"), build_fn=build)
    assert call_count["n"] == 2


# ---------------------------------------------------------------------------
# get_cached_metric
# ---------------------------------------------------------------------------

def test_cached_metric_compute_fn_called_once():
    ctx = _make_ctx()
    call_count = {"n": 0}

    def compute():
        call_count["n"] += 1
        return {"value": 42.0}

    key = (48000, 65536, "mixed", "sig_abc", 1)
    m1 = get_cached_metric(ctx, metric_name="ripple", key=key, compute_fn=compute)
    m2 = get_cached_metric(ctx, metric_name="ripple", key=key, compute_fn=compute)
    assert call_count["n"] == 1
    assert m1 is m2


# ---------------------------------------------------------------------------
# _MEAS_FIXED_CACHE in run_preprocess
# ---------------------------------------------------------------------------

def _make_minimal_cfg(fs=48000, taps=4096, enable_afdw=False, fdw_cycles=10.0, comparison_mode=False):
    """Return a minimal duck-typed config object for run_preprocess."""
    class _Cfg:
        num_taps = taps
        plot_smoothing_level = "1/3"
        comparison_mode = False

    cfg = _Cfg()
    cfg.fs = fs
    cfg.num_taps = taps
    cfg.plot_smoothing_level = "1/3"
    cfg.comparison_mode = comparison_mode
    cfg.enable_afdw = enable_afdw
    cfg.fdw_cycles = fdw_cycles
    cfg.comparison_ref_fs = 44100
    cfg.comparison_ref_taps = 4096
    cfg.stereo_link = False
    return cfg


def _make_measurement_arrays(n=200, fs=48000):
    f = np.geomspace(20.0, float(fs) / 2.0, n)
    m = -20.0 * np.log10(f / 1000.0 + 1e-6) + np.random.default_rng(42).normal(0, 1, n)
    p = np.random.default_rng(43).uniform(-180, 180, n)
    return f, m, p


def test_meas_fixed_cache_hit_on_second_call():
    # Use two different fdw_cycles to force _PREPROCESS_CACHE to miss while
    # still hitting _MEAS_FIXED_CACHE (which is keyed only on measurement-fixed
    # parts, not on AFDW params).
    clear_meas_fixed_cache()
    clear_preprocess_cache()
    f, m, p = _make_measurement_arrays()

    cfg1 = _make_minimal_cfg(enable_afdw=True, fdw_cycles=8.0)
    cfg2 = _make_minimal_cfg(enable_afdw=True, fdw_cycles=14.0)

    # First call: _MEAS_FIXED_CACHE miss
    r1 = run_preprocess(f, m, p, cfg1)
    stats_after_first = get_meas_fixed_cache_stats()
    assert stats_after_first.get("misses", 0) >= 1

    # Second call different fdw_cycles: _PREPROCESS_CACHE misses, but
    # _MEAS_FIXED_CACHE should hit.
    r2 = run_preprocess(f, m, p, cfg2)
    stats_after_second = get_meas_fixed_cache_stats()

    assert stats_after_second["hits"] > stats_after_first.get("hits", 0), (
        "Second call with different fdw_cycles should hit _MEAS_FIXED_CACHE"
    )
    # Measurement-fixed fields must be identical
    np.testing.assert_array_equal(r1.m_interp, r2.m_interp)
    np.testing.assert_array_equal(r1.ctx.freq_axis, r2.ctx.freq_axis)
    np.testing.assert_array_equal(r1.conf_mask, r2.conf_mask)


def test_meas_fixed_cache_different_fdw_cycles_still_hits():
    """Cache must hit when only fdw_cycles changes (it's in AFDW, not fixed portion)."""
    clear_meas_fixed_cache()
    clear_preprocess_cache()
    f, m, p = _make_measurement_arrays()

    cfg1 = _make_minimal_cfg(enable_afdw=True, fdw_cycles=8.0)
    cfg2 = _make_minimal_cfg(enable_afdw=True, fdw_cycles=14.0)

    run_preprocess(f, m, p, cfg1)  # miss
    stats_before = get_meas_fixed_cache_stats()

    run_preprocess(f, m, p, cfg2)  # should hit
    stats_after = get_meas_fixed_cache_stats()

    assert stats_after["hits"] > stats_before.get("hits", 0), (
        "Cache should hit when only fdw_cycles changes"
    )


def test_meas_fixed_cache_different_fdw_produces_different_m_anal():
    """AFDW with different cycles must produce different m_anal despite cache hit."""
    clear_meas_fixed_cache()
    clear_preprocess_cache()
    f, m, p = _make_measurement_arrays()

    cfg_low = _make_minimal_cfg(enable_afdw=True, fdw_cycles=5.0)
    cfg_high = _make_minimal_cfg(enable_afdw=True, fdw_cycles=15.0)

    r_low = run_preprocess(f, m, p, cfg_low)
    r_high = run_preprocess(f, m, p, cfg_high)

    # m_anal MUST differ because AFDW uses different cycle counts
    assert not np.allclose(r_low.m_anal, r_high.m_anal), (
        "Different fdw_cycles must produce different m_anal"
    )


def test_meas_fixed_cache_no_afdw_same_as_baseline():
    """Without AFDW, cached result must numerically match non-cached computation."""
    clear_meas_fixed_cache()
    clear_preprocess_cache()
    f, m, p = _make_measurement_arrays()
    cfg1 = _make_minimal_cfg(enable_afdw=False, fdw_cycles=8.0)
    cfg2 = _make_minimal_cfg(enable_afdw=False, fdw_cycles=12.0)

    r1 = run_preprocess(f, m, p, cfg1)  # miss: computes and stores
    r2 = run_preprocess(f, m, p, cfg2)  # hit: reconstructs from cache

    np.testing.assert_array_equal(r1.m_interp, r2.m_interp)
    np.testing.assert_array_equal(r1.p_rad_interp, r2.p_rad_interp)
    np.testing.assert_array_equal(r1.m_anal, r2.m_anal)
    np.testing.assert_array_equal(r1.p_anal_rad, r2.p_anal_rad)
    np.testing.assert_array_equal(r1.conf_mask, r2.conf_mask)
    assert r1.delay_slope == r2.delay_slope
    assert r1.is_psy == r2.is_psy
    assert r1.analysis_mode == r2.analysis_mode


def test_meas_fixed_cache_different_fs_different_entry():
    clear_meas_fixed_cache()
    clear_preprocess_cache()
    f, m, p = _make_measurement_arrays(fs=48000)

    cfg48 = _make_minimal_cfg(fs=48000)
    cfg96 = _make_minimal_cfg(fs=96000)

    run_preprocess(f, m, p, cfg48)
    run_preprocess(f, m, p, cfg96)

    # Both should be misses; cache should have 2 distinct entries
    assert len(_MEAS_FIXED_CACHE) == 2


def test_meas_fixed_cache_different_taps_different_entry():
    clear_meas_fixed_cache()
    clear_preprocess_cache()
    f, m, p = _make_measurement_arrays()

    cfg_small = _make_minimal_cfg(taps=2048)
    cfg_large = _make_minimal_cfg(taps=4096)

    run_preprocess(f, m, p, cfg_small)
    run_preprocess(f, m, p, cfg_large)

    assert len(_MEAS_FIXED_CACHE) == 2


def test_meas_fixed_cache_different_measurement_arrays_different_entry():
    clear_meas_fixed_cache()
    clear_preprocess_cache()
    f1, m1, p1 = _make_measurement_arrays(n=200)
    f2, m2, p2 = _make_measurement_arrays(n=200)
    # f2 is a different array object with different values
    f2 = f2 * 1.0001

    cfg = _make_minimal_cfg()
    run_preprocess(f1, m1, p1, cfg)
    run_preprocess(f2, m2, p2, cfg)

    assert len(_MEAS_FIXED_CACHE) == 2


def test_meas_fixed_cache_clear_works():
    clear_meas_fixed_cache()
    clear_preprocess_cache()
    f, m, p = _make_measurement_arrays()
    cfg = _make_minimal_cfg()
    run_preprocess(f, m, p, cfg)
    assert len(_MEAS_FIXED_CACHE) > 0
    clear_meas_fixed_cache()
    assert len(_MEAS_FIXED_CACHE) == 0


def test_meas_fixed_cache_presolve_mode_separate_entry():
    clear_meas_fixed_cache()
    clear_preprocess_cache()
    f, m, p = _make_measurement_arrays()
    cfg = _make_minimal_cfg()

    run_preprocess(f, m, p, cfg, presolve_mode=False)
    run_preprocess(f, m, p, cfg, presolve_mode=True)

    assert len(_MEAS_FIXED_CACHE) == 2
