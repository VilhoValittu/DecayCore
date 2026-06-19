import numpy as np

from decaycore.config.models import FilterConfig
from decaycore.dsp.decaycore_dsp import generate_filter_pair
from decaycore.dsp.decaycore_dsp_parts.pair import (
    _GUARD_OFFSET_DIFF_DB,
    _GUARD_TILT_ABS_MAX_DB_PER_OCT,
)


def test_generate_filter_pair_smoke_minimal_stereo():
    freq = np.geomspace(20.0, 20000.0, 512).astype(float)
    phase = np.zeros_like(freq)
    left_mag = np.zeros_like(freq)
    right_mag = np.zeros_like(freq)

    cfg = FilterConfig(
        fs=44100,
        num_taps=16384,
        filter_type_str="Linear Phase",
        stereo_link=True,
    )

    l_imp, l_st, r_imp, r_st = generate_filter_pair(freq, left_mag, phase, freq, right_mag, phase, cfg)

    assert isinstance(l_imp, np.ndarray)
    assert isinstance(r_imp, np.ndarray)
    assert l_imp.ndim == 1
    assert r_imp.ndim == 1
    assert l_imp.size > 0
    assert r_imp.size > 0
    assert isinstance(l_st, dict)
    assert isinstance(r_st, dict)
    assert "offset_db" in l_st
    assert "offset_db" in r_st


def test_generate_filter_pair_shared_offset_identical_for_symmetric_lr():
    """Shared strategy must produce the same shared offset for identical L/R."""
    freq = np.geomspace(20.0, 20000.0, 512).astype(float)
    phase = np.zeros_like(freq)
    mag = np.zeros_like(freq)

    cfg = FilterConfig(
        fs=44100,
        num_taps=16384,
        filter_type_str="Linear Phase",
        stereo_link=True,
        stereo_link_strategy="shared",
    )

    _, l_st, _, r_st = generate_filter_pair(freq, mag, phase, freq, mag, phase, cfg)

    assert l_st.get("stereo_link_mode") == "shared"
    assert r_st.get("stereo_link_mode") == "shared"
    off_l = float(l_st["stereo_link_shared_offset_db"])
    off_r = float(r_st["stereo_link_shared_offset_db"])
    assert abs(off_l - off_r) < 1e-9


def test_generate_filter_pair_auto_gain_mode_always_per_channel():
    """stereo_link_auto_gain_mode must always be 'per_channel' regardless of strategy.

    The offset/window are shared in 'shared' strategy, but the final auto gain
    is always computed per-channel so that individual level corrections are preserved.
    """
    freq = np.geomspace(20.0, 20000.0, 512).astype(float)
    phase = np.zeros_like(freq)
    mag = np.zeros_like(freq)

    for strategy in ("shared", "hybrid"):
        cfg = FilterConfig(
            fs=44100,
            num_taps=16384,
            filter_type_str="Linear Phase",
            stereo_link=True,
            stereo_link_strategy=strategy,
        )
        _, l_st, _, r_st = generate_filter_pair(freq, mag, phase, freq, mag, phase, cfg)
        assert l_st.get("stereo_link_auto_gain_mode") == "per_channel", strategy
        assert r_st.get("stereo_link_auto_gain_mode") == "per_channel", strategy


def test_generate_filter_pair_auto_guard_triggers_hybrid_on_large_level_diff():
    """Auto guard must trigger hybrid when L/R level difference exceeds _GUARD_OFFSET_DIFF_DB."""
    freq = np.geomspace(20.0, 20000.0, 512).astype(float)
    phase = np.zeros_like(freq)
    # L channel is uniformly 2x _GUARD_OFFSET_DIFF_DB louder than R.
    mag_l = np.full_like(freq, 2.0 * _GUARD_OFFSET_DIFF_DB)
    mag_r = np.zeros_like(freq)

    cfg = FilterConfig(
        fs=44100,
        num_taps=16384,
        filter_type_str="Linear Phase",
        stereo_link=True,
        stereo_link_strategy="auto",
    )

    _, l_st, _, r_st = generate_filter_pair(freq, mag_l, phase, freq, mag_r, phase, cfg)

    assert l_st.get("stereo_link_guard_triggered") is True, (
        f"guard should have fired; off_diff={l_st.get('stereo_link_guard_off_diff_db')}"
    )
    assert l_st.get("stereo_link_mode") == "hybrid"
    assert r_st.get("stereo_link_mode") == "hybrid"


def test_generate_filter_pair_auto_guard_stays_shared_for_symmetric_lr():
    """Auto guard must not trigger hybrid for symmetric L/R (no tilt, no level diff)."""
    freq = np.geomspace(20.0, 20000.0, 512).astype(float)
    phase = np.zeros_like(freq)
    mag = np.zeros_like(freq)

    cfg = FilterConfig(
        fs=44100,
        num_taps=16384,
        filter_type_str="Linear Phase",
        stereo_link=True,
        stereo_link_strategy="auto",
    )

    _, l_st, _, r_st = generate_filter_pair(freq, mag, phase, freq, mag, phase, cfg)

    assert l_st.get("stereo_link_guard_triggered") is False
    assert l_st.get("stereo_link_mode") == "shared"
    assert r_st.get("stereo_link_mode") == "shared"


def test_guard_tilt_abs_max_threshold_is_2_db_per_oct():
    """_GUARD_TILT_ABS_MAX_DB_PER_OCT must be 2.0 (suitable for normal rooms)."""
    assert _GUARD_TILT_ABS_MAX_DB_PER_OCT == 2.0
