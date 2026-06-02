import numpy as np

from decaycore.dsp import smoothing
from decaycore.dsp.smoothing import (
    apply_adaptive_fdw,
    apply_smoothing_std,
    psycho_smooth_safe_gain,
    smooth_gain_fractional_octave,
)


def test_apply_adaptive_fdw_fixed_cycles_uses_fixed_width_result():
    freqs = np.geomspace(20.0, 20000.0, 2048)
    mags = 2.0 * np.sin(np.log(freqs))
    fixed_cycles = 12.0

    out = apply_adaptive_fdw(
        freqs,
        mags,
        np.ones_like(freqs),
        base_cycles=fixed_cycles,
        min_cycles=fixed_cycles,
    )
    ref = smooth_gain_fractional_octave(freqs, mags, fixed_cycles, mult=2.0)

    assert np.allclose(out, ref, atol=1e-12, rtol=1e-12)


def test_apply_adaptive_fdw_adaptive_path_is_finite_and_shape_stable():
    freqs = np.geomspace(20.0, 20000.0, 2048)
    mags = 2.0 * np.sin(np.log(freqs))
    confidence = np.linspace(0.0, 1.0, freqs.size)

    out = apply_adaptive_fdw(
        freqs,
        mags,
        confidence,
        base_cycles=15.0,
        min_cycles=5.0,
    )

    assert out.shape == mags.shape
    assert np.all(np.isfinite(out))


def test_apply_adaptive_fdw_cache_hash_uses_full_magnitude_data():
    smoothing.clear_smoothing_cache()
    freqs = np.geomspace(20.0, 20000.0, 64)
    mags_a = 2.0 * np.sin(np.log(freqs))
    mags_b = mags_a.copy()
    mags_b[1] += 25.0
    confidence = np.linspace(0.0, 1.0, freqs.size)

    out_a = apply_adaptive_fdw(freqs, mags_a, confidence, base_cycles=15.0, min_cycles=5.0)
    out_b = apply_adaptive_fdw(freqs, mags_b, confidence, base_cycles=15.0, min_cycles=5.0)

    assert len(smoothing._AFDW_STACK_CACHE) == 2
    assert not np.allclose(out_a, out_b, atol=1e-9, rtol=1e-9)


def test_apply_adaptive_fdw_stack_cache_uses_lru_eviction_at_64_entries():
    smoothing.clear_smoothing_cache()
    freqs = np.geomspace(20.0, 20000.0, 24)
    confidence = np.linspace(0.0, 1.0, freqs.size)

    for i in range(smoothing._AFDW_STACK_CACHE_MAX):
        apply_adaptive_fdw(
            freqs,
            np.full(freqs.size, float(i), dtype=float),
            confidence,
            base_cycles=15.0,
            min_cycles=5.0,
        )
    first_key = next(iter(smoothing._AFDW_STACK_CACHE))
    second_key = list(smoothing._AFDW_STACK_CACHE.keys())[1]

    apply_adaptive_fdw(
        freqs,
        np.zeros(freqs.size, dtype=float),
        confidence,
        base_cycles=15.0,
        min_cycles=5.0,
    )
    apply_adaptive_fdw(
        freqs,
        np.full(freqs.size, 999.0, dtype=float),
        confidence,
        base_cycles=15.0,
        min_cycles=5.0,
    )

    assert len(smoothing._AFDW_STACK_CACHE) == smoothing._AFDW_STACK_CACHE_MAX
    assert first_key in smoothing._AFDW_STACK_CACHE
    assert second_key not in smoothing._AFDW_STACK_CACHE


def test_clear_smoothing_cache_clears_all_module_caches():
    smoothing._SMOOTHING_PLAN_CACHE["plan"] = {"ok": True}
    smoothing._AFDW_STACK_CACHE["stack"] = {"ok": True}

    smoothing.clear_smoothing_cache()

    assert smoothing._SMOOTHING_PLAN_CACHE == {}
    assert smoothing._AFDW_STACK_CACHE == {}


def test_psycho_smooth_safe_gain_preserves_length_and_finite_values_for_monotonic_freqs():
    freqs = np.geomspace(20.0, 20000.0, 512)
    mags = 3.0 * np.sin(np.log(freqs / 20.0))

    out = psycho_smooth_safe_gain(freqs, mags)

    assert out.shape == mags.shape
    assert np.all(np.isfinite(out))


def test_smooth_gain_fractional_octave_handles_monotonic_frequency_input():
    freqs = np.geomspace(20.0, 20000.0, 512)
    gain = np.linspace(-3.0, 3.0, freqs.size)

    out = smooth_gain_fractional_octave(freqs, gain, 12.0)

    assert out.shape == gain.shape
    assert np.all(np.isfinite(out))


def test_apply_smoothing_std_reuses_cached_plan_and_returns_finite_arrays():
    smoothing.clear_smoothing_cache()
    freqs = np.geomspace(20.0, 20000.0, 512)
    mags = 2.0 * np.sin(np.log(freqs / 20.0))
    phases = np.linspace(-180.0, 180.0, freqs.size)

    mags_a, phases_a = apply_smoothing_std(freqs, mags, phases, octave_fraction=1.0 / 12.0)
    cached_plan = next(iter(smoothing._SMOOTHING_PLAN_CACHE.values()))
    mags_b, phases_b = apply_smoothing_std(freqs, mags, phases, octave_fraction=1.0 / 12.0)

    assert len(smoothing._SMOOTHING_PLAN_CACHE) == 1
    assert next(iter(smoothing._SMOOTHING_PLAN_CACHE.values())) is cached_plan
    assert mags_a.shape == mags.shape
    assert phases_a.shape == phases.shape
    assert np.all(np.isfinite(mags_a))
    assert np.all(np.isfinite(phases_a))
    assert np.allclose(mags_a, mags_b)
    assert np.allclose(phases_a, phases_b)
