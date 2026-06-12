import numpy as np
import pytest

from decaycore.config.models import FilterConfig
from decaycore.dsp.phase import calculate_minimum_phase
from decaycore.dsp.phase_ir_phase import (
    _PhaseComponents,
    _apply_phase_model,
    _compute_excess_phase,
    _unwrap_phases,
)
from decaycore.dsp.phase_ir_phase_models import unified_correction_gain
from decaycore.dsp.phase_ir_utils import _max_abs_group_delay_ms


class _SilentLogger:
    def info(self, *args, **kwargs):
        pass

    def debug(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass


def _identity_phase_limiter(freq_axis, phase_rad, **kwargs):
    return np.asarray(phase_rad, dtype=float)


def _smooth_lf_allpass_excess(freq_axis: np.ndarray, f0: float = 60.0, scale: float = 0.25) -> np.ndarray:
    """
    Smooth synthetic LF excess phase from a 2nd-order allpass rotation.

    The default scale stays modest for guard-specific tests; separate tests
    exercise multi-rotation excess explicitly.
    """
    w = np.maximum(np.asarray(freq_axis, dtype=float), 1e-9) / float(f0)
    q = 0.7
    ph = -2.0 * np.arctan2(w / q, 1.0 - w**2)
    return float(scale) * ph


def _run_phase_model(
    *,
    excess: np.ndarray,
    freq_axis: np.ndarray,
    n_fft: int,
    fs: int = 48000,
    is_mixed: bool,
    conf_mask: np.ndarray | None = None,
    **cfg_kwargs,
):
    total_mag = np.ones_like(freq_axis, dtype=float)
    min_p = calculate_minimum_phase(total_mag, max_phase_deg=None)
    theo_xo = np.zeros_like(freq_axis, dtype=float)
    p_used = theo_xo + excess
    raw_u, ref_u = _unwrap_phases(p_used, theo_xo)
    excess_u = _compute_excess_phase(raw_u, ref_u)
    field_names = set(FilterConfig.__dataclass_fields__.keys())
    init_kwargs = {k: v for k, v in cfg_kwargs.items() if k in field_names}
    extra_attrs = {k: v for k, v in cfg_kwargs.items() if k not in field_names}
    cfg = FilterConfig(
        fs=fs,
        num_taps=n_fft,
        filter_type_str="Mixed" if is_mixed else "Linear Phase",
        **init_kwargs,
    )
    for k, v in extra_attrs.items():
        setattr(cfg, k, v)
    st: dict = {}
    components = _PhaseComponents(
        raw_u=raw_u,
        ref_u=ref_u,
        excess_u=excess_u,
        min_phase=min_p,
        theo_xo=theo_xo,
        conf_mask=conf_mask if conf_mask is not None else np.ones_like(freq_axis, dtype=float),
        total_mag=total_mag,
        n_fft=n_fft,
        is_mixed=is_mixed,
        mixed_split_hz=180.0,
        mixed_transition_hz=100.0,
        use_bassfirst=False,
        afdw_on=False,
        logger=_SilentLogger(),
        limit_gd_gradient_ms_per_oct_fn=_identity_phase_limiter,
    )
    final_phase = _apply_phase_model(freq_axis, cfg, st, components)
    return final_phase, components, st, excess_u


def test_compute_excess_phase_preserves_smooth_multi_rotation_excess():
    freq_axis = np.linspace(20.0, 400.0, 512)
    # Smooth room/excess rotation that crosses multiple principal phase wraps.
    expected = -4.0 * np.pi * (
        0.5 + 0.5 * np.tanh((np.log2(freq_axis / 80.0)) / 0.45)
    )
    raw = expected + 2.0 * np.pi
    ref = np.zeros_like(raw)

    excess = _compute_excess_phase(raw, ref)

    assert np.all(np.isfinite(excess))
    assert float(np.max(np.abs(excess))) > 2.0 * np.pi
    # The branch-stable principal delta must be unwrapped back to a smooth curve.
    assert float(np.max(np.abs(np.diff(excess)))) < 0.20
    assert np.allclose(excess - excess[0], expected - expected[0], atol=1e-9)


@pytest.mark.parametrize("is_mixed", [False, True])
def test_unified_budget_applies_near_full_lf_correction(is_mixed: bool):
    fs = 48000
    n_fft = 32768
    freq_axis = np.fft.rfftfreq(n_fft, d=1.0 / float(fs))
    excess = _smooth_lf_allpass_excess(freq_axis, f0=60.0)

    _final, components, st, excess_u = _run_phase_model(
        excess=excess,
        freq_axis=freq_axis,
        n_fft=n_fft,
        is_mixed=is_mixed,
        phase_limit=400.0,
        phase_budget_mode="unified",
        # generous guards so the gain curve itself is what is measured
        max_excess_delay_ms=100.0,
        max_excess_delay_cycles=10.0,
        max_pre_ringing_db=0.0,
        phase_authority_enable=False,
    )
    extra_phase = np.asarray(components.extra_phase, dtype=float)
    assert np.all(np.isfinite(extra_phase))

    full_hz = 140.0
    sel = (freq_axis >= 25.0) & (freq_axis <= full_hz) & (np.abs(excess_u) > np.deg2rad(5.0))
    assert np.any(sel)
    eff = np.abs(extra_phase[sel]) / np.abs(excess_u[sel])
    assert float(np.mean(eff)) >= 0.85
    # corrected residual is small below the full-correction band
    residual = excess_u[sel] + extra_phase[sel]
    assert float(np.max(np.abs(residual))) < 0.20 * float(np.max(np.abs(excess_u[sel])))
    assert st["phase_budget_mode"] == "unified"


def test_unified_budget_can_correct_smooth_excess_above_180_degrees():
    fs = 48000
    n_fft = 32768
    freq_axis = np.fft.rfftfreq(n_fft, d=1.0 / float(fs))
    excess = -np.deg2rad(240.0) * np.exp(
        -0.5 * (np.log2(np.maximum(freq_axis, 1e-9) / 55.0) / 0.55) ** 2.0
    )

    _final, components, st, excess_u = _run_phase_model(
        excess=excess,
        freq_axis=freq_axis,
        n_fft=n_fft,
        is_mixed=False,
        phase_limit=400.0,
        phase_budget_mode="unified",
        phase_corr_clamp_lf_deg=720.0,
        phase_corr_clamp_hf_deg=720.0,
        max_excess_delay_ms=1000.0,
        max_excess_delay_cycles=100.0,
        max_pre_ringing_db=0.0,
        phase_authority_enable=False,
    )
    extra_phase = np.asarray(components.extra_phase, dtype=float)
    sel = (freq_axis >= 35.0) & (freq_axis <= 90.0) & (np.abs(excess_u) > np.pi)

    assert np.any(sel)
    assert float(np.max(np.abs(excess_u[sel]))) > np.pi
    assert float(np.max(np.abs(extra_phase[sel]))) > np.pi
    residual = excess_u[sel] + extra_phase[sel]
    assert float(np.max(np.abs(residual))) < 0.20 * float(np.max(np.abs(excess_u[sel])))
    assert float(st["phase_excess_unwrapped_max_abs_deg"]) > 180.0
    assert float(st["phase_extra_post_guard_max_abs_deg"]) > 180.0


def test_unified_gain_monotone_fade_and_zero_above_limit():
    fs = 48000
    n_fft = 16384
    freq_axis = np.fft.rfftfreq(n_fft, d=1.0 / float(fs))
    phase_lim_hz = 400.0
    phase_mask = (freq_axis > 0) & (freq_axis <= phase_lim_hz)
    cfg = FilterConfig(
        fs=fs,
        num_taps=n_fft,
        filter_type_str="Linear Phase",
        phase_limit=phase_lim_hz,
    )
    st: dict = {}
    gain = unified_correction_gain(
        freq_axis,
        cfg,
        st,
        is_mixed=False,
        phase_conf=np.ones_like(freq_axis, dtype=float),
        spike_suppress=np.ones_like(freq_axis, dtype=float),
        phase_mask=phase_mask,
    )
    assert np.all(np.isfinite(gain))
    full_hz = float(st["mixed_phase_full_correction_hz"])
    # non-increasing above the full-correction frequency
    fade_sel = (freq_axis >= full_hz) & (freq_axis <= phase_lim_hz)
    g_fade = gain[fade_sel]
    assert np.all(np.diff(g_fade) <= 1e-9)
    # exactly zero at/above phase_limit
    assert np.all(gain[freq_axis > phase_lim_hz] == 0.0)
    # near strength below full_hz
    lf_sel = (freq_axis >= 20.0) & (freq_axis <= full_hz)
    assert float(np.min(gain[lf_sel])) >= 0.85


@pytest.mark.parametrize("is_mixed", [False, True])
def test_unified_excess_delay_guard_binds(is_mixed: bool):
    fs = 48000
    n_fft = 32768
    freq_axis = np.fft.rfftfreq(n_fft, d=1.0 / float(fs))
    excess = _smooth_lf_allpass_excess(freq_axis, f0=60.0)

    limit_ms = 0.3
    _final, components, st, _excess_u = _run_phase_model(
        excess=excess,
        freq_axis=freq_axis,
        n_fft=n_fft,
        is_mixed=is_mixed,
        phase_limit=400.0,
        phase_budget_mode="unified",
        max_excess_delay_ms=limit_ms,
        max_excess_delay_cycles=0.0,  # absolute-ms behavior
        max_pre_ringing_db=0.0,
        phase_authority_enable=False,
    )
    extra_phase = np.asarray(components.extra_phase, dtype=float)
    assert np.all(np.isfinite(extra_phase))
    scale = float(st["phase_excess_delay_scale"])
    assert scale < 1.0
    # required scale must be above the guard's 0.05 floor for this input,
    # so the post-guard max |GD| lands within the limit
    assert scale > 0.05
    phase_mask = (freq_axis > 0) & (freq_axis <= 400.0)
    post_gd_ms = _max_abs_group_delay_ms(freq_axis, extra_phase, phase_mask)
    assert post_gd_ms <= limit_ms * 1.10 + 1e-6
    assert "phase_excess_delay_before_ms" in st
    assert "phase_excess_delay_limit_min_ms" in st


def test_unified_pre_ringing_guard_linear_mode_bounded():
    fs = 48000
    n_fft = 16384
    freq_axis = np.fft.rfftfreq(n_fft, d=1.0 / float(fs))
    rng = np.random.default_rng(7)
    # spiky excess in the audible band to provoke pre-ringing
    excess = np.zeros_like(freq_axis)
    band = (freq_axis >= 80.0) & (freq_axis <= 380.0)
    excess[band] = rng.uniform(-2.5, 2.5, size=int(np.count_nonzero(band)))

    final_phase, components, st, _excess_u = _run_phase_model(
        excess=excess,
        freq_axis=freq_axis,
        n_fft=n_fft,
        is_mixed=False,
        phase_limit=400.0,
        phase_budget_mode="unified",
        max_pre_ringing_db=-35.0,
        phase_authority_enable=False,
    )
    extra_phase = np.asarray(components.extra_phase, dtype=float)
    assert np.all(np.isfinite(np.asarray(final_phase, dtype=float)))
    assert np.all(np.isfinite(extra_phase))
    assert "phase_pre_ringing_scale" in st
    scale = float(st["phase_pre_ringing_scale"])
    assert 0.0 < scale <= 1.0
    after_db = st.get("phase_pre_ringing_after_db")
    target_db = float(st.get("phase_pre_ringing_target_db", -35.0))
    if after_db is not None and np.isfinite(float(after_db)) and scale < 0.999:
        bass_scale = float(st.get("phase_pre_ringing_scale_bass", 1.0))
        # guard either reaches the target or stops at the bass protection floor
        assert (float(after_db) <= target_db + 3.0) or (bass_scale <= 0.60)


def test_unified_sanity_clamp_caps_excess():
    fs = 48000
    n_fft = 16384
    freq_axis = np.fft.rfftfreq(n_fft, d=1.0 / float(fs))
    # smooth ~120 deg excess bump around 40 Hz; the low configured limit
    # intentionally exercises the sanity clamp.
    excess = np.deg2rad(120.0) * np.exp(
        -0.5 * (np.log2(np.maximum(freq_axis, 1e-9) / 40.0) / 0.7) ** 2.0
    )

    clamp_lf_deg = 30.0
    _final, components, st, _excess_u = _run_phase_model(
        excess=excess,
        freq_axis=freq_axis,
        n_fft=n_fft,
        is_mixed=False,
        phase_limit=400.0,
        phase_budget_mode="unified",
        phase_corr_clamp_lf_deg=clamp_lf_deg,
        phase_corr_clamp_hf_deg=10.0,
        max_excess_delay_ms=1000.0,
        max_excess_delay_cycles=100.0,
        max_pre_ringing_db=0.0,
        phase_authority_enable=False,
    )
    extra_phase = np.asarray(components.extra_phase, dtype=float)
    assert float(np.max(np.abs(np.rad2deg(extra_phase)))) <= clamp_lf_deg + 1e-6
    assert bool(st["phase_corr_clipped"]) is True
    assert int(st["phase_corr_clipped_bins"]) > 0
    assert "(sanity)" in str(st["phase_corr_clamp_msg"])


def test_unified_default_clamp_is_sanity_bound_not_limiter():
    fs = 48000
    n_fft = 16384
    freq_axis = np.fft.rfftfreq(n_fft, d=1.0 / float(fs))
    excess = _smooth_lf_allpass_excess(freq_axis, f0=60.0)

    _final, _components, st, _excess_u = _run_phase_model(
        excess=excess,
        freq_axis=freq_axis,
        n_fft=n_fft,
        is_mixed=False,
        phase_limit=400.0,
        phase_budget_mode="unified",
        max_excess_delay_ms=1000.0,
        max_excess_delay_cycles=100.0,
        max_pre_ringing_db=0.0,
        phase_authority_enable=False,
    )
    # With default clamp limits, this modest smooth excess must pass through
    # unclipped: the clamp is not the primary limiter for normal LF correction.
    assert bool(st["phase_corr_clipped"]) is False


@pytest.mark.parametrize("is_mixed", [False, True])
def test_legacy_mode_reproduces_weak_correction(is_mixed: bool):
    fs = 48000
    n_fft = 32768
    freq_axis = np.fft.rfftfreq(n_fft, d=1.0 / float(fs))
    excess = _smooth_lf_allpass_excess(freq_axis, f0=60.0)

    results = {}
    for mode in ("unified", "legacy"):
        _final, _components, st, _excess_u = _run_phase_model(
            excess=excess,
            freq_axis=freq_axis,
            n_fft=n_fft,
            is_mixed=is_mixed,
            phase_limit=400.0,
            phase_budget_mode=mode,
            max_excess_delay_ms=100.0,
            max_excess_delay_cycles=10.0,
            max_pre_ringing_db=0.0,
            phase_authority_enable=False,
        )
        results[mode] = float(st.get("phase_eff_strength_mean", float("nan")))

    assert np.isfinite(results["unified"]) and np.isfinite(results["legacy"])
    # eff-strength mean spans the whole correction band including the fade
    # region, so the absolute threshold sits below the LF gain (~0.9)
    assert results["unified"] > 0.55
    if is_mixed:
        assert results["legacy"] < 0.50
        assert results["unified"] > 1.30 * results["legacy"]
    else:
        assert results["legacy"] < 0.35
        assert results["unified"] > 1.50 * results["legacy"]
