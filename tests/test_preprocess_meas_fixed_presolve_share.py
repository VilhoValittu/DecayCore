"""Stage 1 dedup: the meas_fixed core is presolve_mode-independent and shared.

The stereo presolve pass (presolve_mode=True) and the real pipeline pass
(presolve_mode=False) compute the same heavy preprocess core (rFFT axis,
smoothing, confidence, reflections). These tests pin that the core arrays are
bit-identical across the two modes, and that the second call reuses the cached
core instead of recomputing it.
"""

import numpy as np

from decaycore.config.models import FilterConfig
from decaycore.dsp.dsp_preprocess import (
    clear_preprocess_cache,
    get_meas_fixed_cache_stats,
    run_preprocess,
)

_CORE_FIELDS = (
    "m_smooth_std",
    "p_smooth",
    "m_interp",
    "p_rad_raw",
    "p_rad_interp",
    "complex_meas",
    "m_anal",
    "p_anal_rad",
    "complex_anal",
    "conf_mask",
)


def _cfg():
    return FilterConfig(
        fs=48000,
        num_taps=32768,
        filter_type_str="Linear Phase",
        stereo_link=False,
        comparison_mode=False,
    )


def test_meas_fixed_core_identical_across_presolve_mode(lr_measurements):
    (f, m, p), _ = lr_measurements
    cfg = _cfg()

    clear_preprocess_cache()
    pre = run_preprocess(f, m, p, cfg, presolve_mode=True)
    real = run_preprocess(f, m, p, cfg, presolve_mode=False)

    np.testing.assert_array_equal(
        np.asarray(pre.ctx.freq_axis), np.asarray(real.ctx.freq_axis),
        err_msg="core field freq_axis diverged",
    )
    for field in _CORE_FIELDS:
        a = np.asarray(getattr(pre, field))
        b = np.asarray(getattr(real, field))
        assert a.shape == b.shape, field
        np.testing.assert_array_equal(a, b, err_msg=f"core field {field} diverged")

    assert float(pre.delay_slope) == float(real.delay_slope)


def test_real_pass_reuses_cached_presolve_core(lr_measurements):
    (f, m, p), _ = lr_measurements
    cfg = _cfg()

    clear_preprocess_cache()
    before = dict(get_meas_fixed_cache_stats())
    # First call (presolve) populates the core cache.
    run_preprocess(f, m, p, cfg, presolve_mode=True)
    mid = dict(get_meas_fixed_cache_stats())
    # Second call (real pass, same arrays, opposite mode) must hit the core.
    run_preprocess(f, m, p, cfg, presolve_mode=False)
    after = dict(get_meas_fixed_cache_stats())

    assert mid.get("misses", 0) == before.get("misses", 0) + 1
    assert after.get("hits", 0) == mid.get("hits", 0) + 1
    assert after.get("misses", 0) == mid.get("misses", 0)
