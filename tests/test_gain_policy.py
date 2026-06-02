from types import SimpleNamespace

import numpy as np

from decaycore.dsp.gain_policy import (
    GainPolicy,
    apply_cuts_only_guard,
    build_low_frequency_guard_mask,
    clamp_gain_curve,
    resolve_gain_policy,
)


def _cfg_float_allow_zero(cfg, name: str, default: float) -> float:
    try:
        v = getattr(cfg, name, default)
        if v is None:
            return float(default)
        x = float(v)
    except (

        AttributeError,
        TypeError,
        ValueError,
        KeyError,
        IndexError,
        RuntimeError,
        OSError,
        ImportError,
        ModuleNotFoundError,
        NameError,
    ):
        return float(default)
    if not np.isfinite(x):
        return float(default)
    return float(x)


def test_shared_gain_policy_combines_low_bass_and_exc_guards():
    cfg = SimpleNamespace(
        max_boost_db=6.0,
        max_cut_db=15.0,
        low_bass_cut_enable=True,
        low_bass_cut_hz=40.0,
        low_bass_cut_strength=0.75,
        exc_prot=True,
        exc_freq=35.0,
    )
    freq_axis = np.asarray([10.0, 20.0, 35.0, 45.0, 50.0, 60.0], dtype=float)

    policy = resolve_gain_policy(cfg, cfg_float_allow_zero_fn=_cfg_float_allow_zero)
    guard_mask = build_low_frequency_guard_mask(freq_axis, policy)

    assert bool(policy.low_cut_enable) is True
    assert float(policy.low_cut_hz) == 40.0
    assert float(policy.exc_soft_hz) == 35.0 * 1.41
    assert np.array_equal(guard_mask, np.asarray([True, True, True, True, False, False]))


def test_shared_gain_policy_reapplies_cuts_only_floor_and_clamp():
    cfg = SimpleNamespace(
        max_boost_db=3.0,
        max_cut_db=12.0,
        low_bass_cut_enable=True,
        low_bass_cut_hz=40.0,
        low_bass_cut_strength=1.0,
        exc_prot=False,
        exc_freq=0.0,
    )
    freq_axis = np.asarray([20.0, 30.0, 50.0], dtype=float)
    mask = np.asarray([True, True, True], dtype=bool)
    curve = np.asarray([2.0, -1.0, 8.0], dtype=float)
    floor_ref = np.asarray([-2.5, -1.5, np.nan], dtype=float)

    policy = resolve_gain_policy(cfg, cfg_float_allow_zero_fn=_cfg_float_allow_zero)
    low_guard = build_low_frequency_guard_mask(freq_axis, policy, include_exc_soft=False)
    guarded, meta = apply_cuts_only_guard(curve, mask=mask, guard_mask=low_guard, floor_ref=floor_ref)
    clamped = clamp_gain_curve(guarded, policy=policy, mask=mask)

    assert np.allclose(guarded[:2], [-2.5, -1.5], atol=1e-10, rtol=0.0)
    assert int(meta["boost_clamped_bins"]) == 1
    assert int(meta["floor_reapplied_bins"]) == 2
    assert np.allclose(clamped, [-2.5, -1.5, 3.0], atol=1e-10, rtol=0.0)


def test_clamp_gain_curve_supports_local_cut_caps_inside_mask():
    policy = GainPolicy(
        max_cut_db=12.0,
        max_boost_db=6.0,
        low_cut_enable=False,
        low_cut_hz=0.0,
        low_cut_strength=0.0,
        exc_prot=False,
        exc_freq=0.0,
        exc_soft_hz=0.0,
    )
    curve = np.asarray([8.0, -10.0, -10.0, 8.0], dtype=float)
    mask = np.asarray([True, True, False, False], dtype=bool)
    boost_cap = np.asarray([4.0, 4.0, 4.0, 4.0], dtype=float)
    cut_cap = np.asarray([5.0, 5.0, 5.0, 5.0], dtype=float)

    out = clamp_gain_curve(curve, policy=policy, boost_cap_db=boost_cap, cut_cap_db=cut_cap, mask=mask)

    assert np.allclose(out, [4.0, -5.0, -10.0, 6.0], atol=1e-12, rtol=0.0)


def test_clamp_gain_curve_invalid_cut_caps_fall_back_to_global_cut():
    policy = GainPolicy(
        max_cut_db=12.0,
        max_boost_db=6.0,
        low_cut_enable=False,
        low_cut_hz=0.0,
        low_cut_strength=0.0,
        exc_prot=False,
        exc_freq=0.0,
        exc_soft_hz=0.0,
    )
    curve = np.asarray([-14.0, -14.0], dtype=float)
    mask = np.asarray([True, True], dtype=bool)
    invalid_cut_cap = np.asarray([5.0], dtype=float)

    out = clamp_gain_curve(curve, policy=policy, cut_cap_db=invalid_cut_cap, mask=mask)

    assert np.allclose(out, [-12.0, -12.0], atol=1e-12, rtol=0.0)
