from types import SimpleNamespace

import numpy as np
import pytest

import decaycore.dsp.dsp_preprocess as preprocess_module
from decaycore.config.models import FilterConfig
from decaycore.dsp.decaycore_dsp import generate_filter
from decaycore.dsp.decaycore_leveling import StereoLinkContext
from decaycore.dsp._pruning import clear_pruning_hook, set_pruning_hook
from decaycore.dsp.filter_pipeline import _run_generate_filter_pre_correction
from decaycore.dsp import correction_baseline
from decaycore.dsp.correction_baseline import apply_null_guard_target
from decaycore.dsp.dsp_ops import apply_hpf_to_mags
from decaycore.dsp.dsp_phase_ir import _apply_hpf_magnitude_to_gain
from decaycore.dsp.dsp_preprocess import clear_preprocess_cache, run_preprocess
from decaycore.dsp.tdc import apply_smart_tdc
from decaycore.auto_mode.scoring_metrics import _auto_score_result


def test_generate_filter_smoke(lr_measurements):
    """
    DSP smoke test: real FR input -> impulse + stats.
    """
    (f, m, p), _ = lr_measurements

    cfg = FilterConfig(
        fs=48000,
        num_taps=32768,
        filter_type_str="Linear Phase",
        stereo_link=False,
    )

    imp, st = generate_filter(f, m, p, cfg)

    assert imp is not None
    assert len(imp) > 0
    assert len(imp) == cfg.num_taps
    assert isinstance(st, dict)
    assert "offset_db" in st
    assert st["acoustic_authority_version"] == 2
    assert st["acoustic_authority_limits_source"] == "stats"


def test_generate_filter_score_only_omits_ui_payload_arrays(lr_measurements):
    (f, m, p), _ = lr_measurements
    cfg = FilterConfig(
        fs=48000,
        num_taps=32768,
        filter_type_str="Linear Phase",
        stereo_link=False,
        comparison_mode=False,
    )

    imp, st = generate_filter(f, m, p, cfg, include_response_arrays=False)

    assert imp is not None
    assert len(st["freq_axis"]) >= 8
    assert len(st["measured_mags"]) == len(st["freq_axis"])
    assert len(st["target_mags"]) == len(st["freq_axis"])
    assert len(st["filter_mags"]) == len(st["freq_axis"])
    assert len(st["authority_voice_risk"]) == len(st["freq_axis"])
    assert len(st["authority_modal_support"]) == len(st["freq_axis"])
    assert len(st["authority_decay_need"]) == len(st["freq_axis"])
    assert len(st["authority_null_risk"]) == len(st["freq_axis"])
    assert "measured_mags_raw" not in st
    assert "fir_only_predicted_filter_mags" not in st
    assert "mag_mask" not in st
    assert "authority_cut" not in st
    assert "authority_boost" not in st
    assert "authority_phase" not in st


def test_generate_filter_score_only_preserves_auto_score_rank_inputs(lr_measurements):
    (f, m, p), _ = lr_measurements
    cfg = FilterConfig(
        fs=48000,
        num_taps=32768,
        filter_type_str="Linear Phase",
        stereo_link=False,
        comparison_mode=False,
    )

    _imp_full, st_full = generate_filter(f, m, p, cfg, include_response_arrays=True)
    _imp_score, st_score = generate_filter(f, m, p, cfg, include_response_arrays=False)

    full_metrics = _auto_score_result(SimpleNamespace(l_st=st_full, r_st=st_full, metrics={}))
    score_metrics = _auto_score_result(SimpleNamespace(l_st=st_score, r_st=st_score, metrics={}))

    assert score_metrics["rank_score"] == pytest.approx(full_metrics["rank_score"], abs=1e-9)
    assert score_metrics["voice_clarity_penalty"] == pytest.approx(
        full_metrics["voice_clarity_penalty"],
        abs=1e-9,
    )
    assert score_metrics["residual_peak_penalty"] == pytest.approx(
        full_metrics["residual_peak_penalty"],
        abs=1e-9,
    )


def test_hpf_blocks_positive_net_boost_below_cutoff():
    freq = np.linspace(10.0, 2000.0, 1024, dtype=float)
    mag = np.zeros_like(freq)
    mag[freq <= 40.0] = -40.0
    phase = np.zeros_like(freq)
    cfg = FilterConfig(
        fs=44100,
        num_taps=65536,
        filter_type_str="Minimum Phase",
        mag_c_min=10.0,
        mag_c_max=200.0,
        max_boost_db=120.0,
        max_cut_db=120.0,
        max_slope_db_per_oct=0.0,
        reg_strength=0.0,
        unsafe_raw_dsp=True,
        hpf_settings={"enabled": True, "freq": 40.0, "order": 2},
        house_freqs=[10.0, 20000.0],
        house_mags=[0.0, 0.0],
    )

    _imp, st = generate_filter(freq, mag, phase, cfg)

    freq_axis = np.asarray(st.get("freq_axis", []), dtype=float)
    filt = np.asarray(st.get("predicted_filter_mags", []), dtype=float)
    hpf = apply_hpf_to_mags(freq_axis, np.zeros_like(freq_axis), 40.0, 2)
    mask = (freq_axis >= 10.0) & (freq_axis <= 40.0)
    assert np.any(mask)
    assert float(np.max(filt[mask] - hpf[mask])) <= 1e-9


def test_low_tap_hpf_is_routed_outside_fir_gain():
    freq_axis = np.asarray([10.0, 20.0, 40.0, 80.0], dtype=float)
    gain_db = np.asarray([1.0, 0.5, 0.0, -0.5], dtype=float)
    logger = SimpleNamespace(info=lambda *_args, **_kwargs: None)
    hpf_settings = {"enabled": True, "freq": 40.0, "order": 2}

    low_tap_cfg = FilterConfig(fs=44100, num_taps=4096, hpf_settings=hpf_settings)
    low_tap_out = _apply_hpf_magnitude_to_gain(
        cfg=low_tap_cfg,
        freq_axis=freq_axis,
        gain_db=gain_db,
        apply_hpf_to_mags=apply_hpf_to_mags,
        logger=logger,
    )
    np.testing.assert_allclose(low_tap_out, gain_db)
    _imp, st = generate_filter(
        np.linspace(10.0, 2000.0, 512, dtype=float),
        np.zeros(512, dtype=float),
        np.zeros(512, dtype=float),
        low_tap_cfg,
    )
    assert st["external_iir_hpf_enabled"] is True
    assert st["external_iir_hpf_freq_hz"] == pytest.approx(40.0)
    assert st["external_iir_hpf_slope_db_oct"] == 12

    high_tap_cfg = FilterConfig(fs=44100, num_taps=65536, hpf_settings=hpf_settings)
    high_tap_out = _apply_hpf_magnitude_to_gain(
        cfg=high_tap_cfg,
        freq_axis=freq_axis,
        gain_db=gain_db,
        apply_hpf_to_mags=apply_hpf_to_mags,
        logger=logger,
    )
    assert not np.allclose(high_tap_out, gain_db)


def test_pre_correction_reraises_optuna_trial_pruned(monkeypatch, caplog):
    trial_pruned_cls = type(
        "TrialPruned",
        (Exception,),
        {"__module__": "optuna.exceptions"},
    )

    prep = SimpleNamespace(
        f_in=np.array([20.0, 1000.0], dtype=float),
        m_in=np.array([0.0, 0.0], dtype=float),
        ctx=SimpleNamespace(
            n_fft=8,
            freq_axis=np.array([20.0, 1000.0], dtype=float),
            gain_db=np.array([0.0, 0.0], dtype=float),
            st={},
            target_mags=np.array([0.0, 0.0], dtype=float),
        ),
        m_interp=np.array([0.0, 0.0], dtype=float),
        p_rad_interp=np.array([0.0, 0.0], dtype=float),
        delay_slope=0.0,
        m_plot_db=None,
        complex_meas=None,
        m_anal=np.array([0.0, 0.0], dtype=float),
        conf_mask=np.array([True, True], dtype=bool),
        reflections=[],
        cmp=None,
        analysis_mode="native",
        is_psy=False,
    )
    corr = SimpleNamespace(
        final_g=np.array([0.5, -0.25], dtype=float),
        over_boost=0.0,
        over_cut=0.0,
    )

    monkeypatch.setattr("decaycore.dsp.filter_pipeline.run_preprocess", lambda *_args, **_kwargs: prep)
    monkeypatch.setattr("decaycore.dsp.filter_pipeline.run_correction_stage", lambda **_kwargs: corr)
    set_pruning_hook(lambda _partial_score: (_ for _ in ()).throw(trial_pruned_cls()))

    cfg = FilterConfig(
        fs=48000,
        num_taps=32768,
        filter_type_str="Linear Phase",
        stereo_link=False,
    )

    try:
        with pytest.raises(trial_pruned_cls):
            _run_generate_filter_pre_correction(
                prep.f_in,
                prep.m_in,
                prep.p_rad_interp,
                cfg,
            )
    finally:
        clear_pruning_hook()

    assert "pruning hook call" not in caplog.text


def test_run_preprocess_preserves_requested_even_fft_size(lr_measurements):
    (f, m, p), _ = lr_measurements

    cfg = FilterConfig(
        fs=48000,
        num_taps=32768,
        filter_type_str="Linear Phase",
        stereo_link=False,
        comparison_mode=False,
    )

    prep = run_preprocess(f, m, p, cfg)

    assert int(prep.ctx.n_fft) == int(cfg.num_taps)
    assert np.array_equal(
        prep.ctx.freq_axis,
        np.fft.rfftfreq(int(cfg.num_taps), d=1.0 / float(cfg.fs)),
    )
    assert prep.p_rad_raw.shape == prep.ctx.freq_axis.shape
    assert np.isfinite(prep.delay_slope)


def test_preprocess_cache_key_hashes_measurement_content_not_sampled_sum():
    freqs = np.geomspace(20.0, 20000.0, 32)
    mags_a = np.zeros_like(freqs)
    mags_b = np.zeros_like(freqs)
    mags_a[0] = 1.0
    mags_a[4] = -1.0
    mags_b[0] = 0.5
    mags_b[4] = -0.5
    phases = np.linspace(0.0, 45.0, freqs.size)
    cfg = FilterConfig(
        fs=48000,
        num_taps=8192,
        filter_type_str="Linear Phase",
        stereo_link=False,
        comparison_mode=False,
    )

    key_a = preprocess_module._preprocess_cache_key(freqs, mags_a, phases, cfg, None, False)
    key_b = preprocess_module._preprocess_cache_key(freqs, mags_b, phases, cfg, None, False)

    assert key_a is not None
    assert key_b is not None
    assert np.asarray(mags_a)[::4].sum() == np.asarray(mags_b)[::4].sum()
    assert key_a != key_b


def test_run_preprocess_cache_hit_returns_independent_mutable_state(monkeypatch):
    freqs = np.geomspace(20.0, 20000.0, 128)
    mags = np.sin(np.linspace(0.0, np.pi, freqs.size)) * 2.0
    phases = np.linspace(0.0, 45.0, freqs.size)
    cfg = FilterConfig(
        fs=48000,
        num_taps=8192,
        filter_type_str="Linear Phase",
        stereo_link=False,
        comparison_mode=False,
    )

    clear_preprocess_cache()
    try:
        result1 = run_preprocess(freqs, mags, phases, cfg)
        clean_gain0 = float(result1.ctx.gain_db[0])
        clean_m_anal0 = float(result1.m_anal[0])
        assert len(preprocess_module._PREPROCESS_CACHE) == 1

        result1.ctx.st["leak_test"] = "bad"
        result1.ctx.gain_db[0] += 99.0
        result1.m_anal[0] += 99.0

        def fail_recompute(*_args, **_kwargs):
            raise AssertionError("preprocess recomputed instead of using cache")

        monkeypatch.setattr(preprocess_module, "analysis_smoothing_lf_to_hf", fail_recompute)

        result2 = run_preprocess(freqs, mags, phases, cfg)

        assert result1 is not result2
        assert result1.ctx is not result2.ctx
        assert "leak_test" not in result2.ctx.st
        assert float(result2.ctx.gain_db[0]) == pytest.approx(clean_gain0)
        assert float(result2.m_anal[0]) == pytest.approx(clean_m_anal0)
        assert len(preprocess_module._PREPROCESS_CACHE) == 1
    finally:
        clear_preprocess_cache()


def test_run_preprocess_records_phase_units_and_radian_suspicion():
    freqs = np.geomspace(20.0, 20000.0, 128)
    mags = np.zeros_like(freqs)
    phase_deg = -360.0 * freqs * 0.002
    cfg = FilterConfig(
        fs=48000,
        num_taps=8192,
        filter_type_str="Linear Phase",
        stereo_link=False,
        comparison_mode=False,
    )

    clear_preprocess_cache()
    prep_deg = run_preprocess(freqs, mags, phase_deg, cfg)
    clear_preprocess_cache()
    prep_rad_like = run_preprocess(freqs, mags, np.linspace(0.0, 3.0, freqs.size), cfg)
    clear_preprocess_cache()

    assert prep_deg.p_rad_raw.shape == prep_deg.ctx.freq_axis.shape
    assert np.isfinite(prep_deg.delay_slope)
    assert prep_rad_like.p_rad_raw.shape == prep_rad_like.ctx.freq_axis.shape
    assert np.isfinite(prep_rad_like.delay_slope)
    assert np.nanmax(np.abs(prep_rad_like.p_rad_raw)) < np.nanmax(np.abs(prep_deg.p_rad_raw))


def test_run_preprocess_presolve_skips_comparison_and_plot_smoothing(lr_measurements):
    (f, m, p), _ = lr_measurements

    cfg = FilterConfig(
        fs=48000,
        num_taps=32768,
        filter_type_str="Linear Phase",
        stereo_link=False,
        comparison_mode=True,
    )
    cfg.plot_smoothing_level = "psy"

    prep = run_preprocess(f, m, p, cfg, presolve_mode=True)

    assert prep.cmp is None
    assert prep.analysis_mode == "native"
    assert prep.m_plot_db is None


def test_run_preprocess_uses_updated_smart_scan_analysis_smoothing(monkeypatch, lr_measurements):
    (f, m, p), _ = lr_measurements

    cfg = FilterConfig(
        fs=48000,
        num_taps=32768,
        filter_type_str="Linear Phase",
        stereo_link=False,
        comparison_mode=False,
    )

    calls: list[dict[str, float]] = []

    def fake_analysis_smoothing(freqs, mags, *, low_bw, high_bw, f_lo, f_hi):
        calls.append(
            {
                "low_bw": float(low_bw),
                "high_bw": float(high_bw),
                "f_lo": float(f_lo),
                "f_hi": float(f_hi),
            }
        )
        return np.asarray(mags, dtype=float)

    monkeypatch.setattr(preprocess_module, "analysis_smoothing_lf_to_hf", fake_analysis_smoothing)

    clear_preprocess_cache()
    try:
        prep = run_preprocess(f, m, p, cfg)
    finally:
        clear_preprocess_cache()

    assert prep.m_anal.shape == prep.ctx.freq_axis.shape
    assert len(calls) == 1
    assert calls[0]["low_bw"] == pytest.approx(1.0 / 3.0)
    assert calls[0]["high_bw"] == pytest.approx(1.0)
    assert calls[0]["f_lo"] == pytest.approx(230.0)
    assert calls[0]["f_hi"] == pytest.approx(400.0)


def test_apply_null_guard_target_matches_reference_convolution():
    freq_axis = np.linspace(20.0, 20000.0, 4096, dtype=float)
    target = np.zeros_like(freq_axis)
    measured = np.zeros_like(freq_axis)
    null_band = (freq_axis >= 1800.0) & (freq_axis <= 2200.0)
    measured[null_band] = -12.0

    out = apply_null_guard_target(
        freq_axis,
        target,
        measured,
        mag_c_min=20.0,
        mag_c_max=20000.0,
        enable=True,
        depth_db=12.0,
        max_blend=0.85,
        max_total_relax_db=12.0,
        smooth_oct=0.18,
    )

    f = np.asarray(freq_axis, dtype=float)
    t = np.asarray(target, dtype=float)
    m = np.asarray(measured, dtype=float)
    band = (f >= 20.0) & (f <= 20000.0)
    lf = np.log2(np.clip(f, 1e-3, None))
    dlf = np.diff(lf[band])
    dlf = dlf[np.isfinite(dlf) & (dlf > 0)]
    step = float(np.median(dlf))
    sigma_bins = max(1.0, float(0.18) / max(step, 1e-6))
    half = int(np.ceil(3.0 * sigma_bins))
    x = np.arange(-half, half + 1, dtype=float)
    k = np.exp(-0.5 * (x / sigma_bins) ** 2)
    k /= np.sum(k)
    env = m.copy()
    env[band] = np.convolve(env[band], k, mode="same")
    dip = np.where(band, env - m, 0.0)
    w = np.clip((dip - 12.0) / 12.0, 0.0, 1.0) ** 1.4
    w2 = np.zeros_like(w)
    w2[band] = np.clip(np.convolve(w[band], k, mode="same"), 0.0, 1.0) * 0.85
    ref = np.maximum(((1.0 - w2) * t) + (w2 * m), t - 12.0)

    assert np.allclose(out, ref, atol=1e-9, rtol=1e-9)


def test_apply_null_guard_target_respects_guard_max_hz_and_leaves_treble_unchanged():
    freq_axis = np.linspace(20.0, 20000.0, 4096, dtype=float)
    target = np.zeros_like(freq_axis)
    measured = np.zeros_like(freq_axis)
    null_band = (freq_axis >= 1800.0) & (freq_axis <= 2200.0)
    measured[null_band] = -12.0

    out = apply_null_guard_target(
        freq_axis,
        target,
        measured,
        mag_c_min=20.0,
        mag_c_max=20000.0,
        guard_max_hz=300.0,
        enable=True,
        depth_db=12.0,
        max_blend=0.85,
        max_total_relax_db=12.0,
        smooth_oct=0.18,
    )

    high = freq_axis >= 1000.0
    assert np.allclose(out[high], target[high], atol=1e-9, rtol=1e-9)


def test_apply_null_guard_target_does_not_treat_lf_rolloff_as_local_null():
    freq_axis = np.geomspace(10.0, 20000.0, 4096, dtype=float)
    target = np.interp(
        freq_axis,
        [10.0, 20.0, 40.0, 100.0, 1000.0, 20000.0],
        [10.0, 10.0, 9.0, 4.8, 0.0, -6.0],
    )
    measured = target.copy()
    low = freq_axis < 22.0
    measured[low] -= 72.0 * np.log2(22.0 / np.maximum(freq_axis[low], 1e-9))

    out = apply_null_guard_target(
        freq_axis,
        target,
        measured,
        mag_c_min=10.0,
        mag_c_max=200.0,
        guard_max_hz=200.0,
        enable=True,
        depth_db=12.0,
        max_blend=0.85,
        max_total_relax_db=12.0,
        smooth_oct=0.18,
    )

    low_band = (freq_axis >= 10.0) & (freq_axis <= 31.5)
    np.testing.assert_allclose(out[low_band], target[low_band], atol=1e-9)

    local_null_measured = target - 36.0 * np.exp(
        -0.5 * (np.log2(freq_axis / 60.0) / 0.06) ** 2
    )
    local_null_out = apply_null_guard_target(
        freq_axis,
        target,
        local_null_measured,
        mag_c_min=10.0,
        mag_c_max=200.0,
        guard_max_hz=200.0,
        enable=True,
        depth_db=12.0,
        max_blend=0.85,
        max_total_relax_db=12.0,
        smooth_oct=0.18,
    )

    assert float(np.interp(60.0, freq_axis, local_null_out - target)) < -3.0


def test_prepare_correction_baseline_skips_second_leveling_pass_after_uniform_target_shift(monkeypatch):
    freq_axis = np.linspace(20.0, 20000.0, 2048, dtype=float)
    measured = np.zeros_like(freq_axis)
    gain_db = np.zeros_like(freq_axis)
    cfg = SimpleNamespace(
        fs=44100,
        num_taps=65536,
        house_freqs=None,
        house_mags=None,
        hpf_settings=None,
        enable_tdc=False,
        enable_null_guard=False,
        lvl_min=500.0,
        lvl_max=2000.0,
        comparison_mode=False,
        lvl_force_offset_db=None,
    )

    correction_baseline._clear_rt60_cache()
    rt60_key = correction_baseline._rt60_cache_key(freq_axis, measured, cfg.fs, cfg.num_taps)
    correction_baseline._RT60_CACHE[rt60_key] = (0.3, {}, 0.3)

    calls = {"count": 0}

    def fake_compute_leveling(_cfg, _freq, _meas, _target, *, stereo_link_ctx=None):
        calls["count"] += 1
        assert stereo_link_ctx is None
        return (5.0, 1.5, 5.0, 0.0, "SmartScanMedian", 500.0, 2000.0)

    monkeypatch.setattr(correction_baseline, "compute_leveling", fake_compute_leveling)

    baseline = correction_baseline._prepare_correction_baseline(
        cfg=cfg,
        freq_axis=freq_axis,
        f_in=freq_axis,
        m_in=measured,
        reflections=[],
        st={},
        m_anal=measured,
        m_plot_db=None,
        is_psy=False,
        cmp=None,
        analysis_mode="native",
        gain_db=gain_db,
        logger=None,
        interpolate_response=lambda *_args, **_kwargs: np.zeros_like(freq_axis),
        _cfg_float_allow_zero=lambda *_args, **_kwargs: 0.0,
        stereo_link_ctx=None,
        presolve_mode=False,
    )

    correction_baseline._clear_rt60_cache()

    assert calls["count"] == 1
    assert baseline.target_shift_db == 5.0
    assert baseline.calc_offset_db == -3.5
    assert baseline.target_level_db_window == 5.0
    assert np.allclose(baseline.target_mags, 5.0)


def test_prepare_correction_baseline_keeps_forced_offset_when_shared_target_shift_is_applied(monkeypatch):
    freq_axis = np.linspace(20.0, 20000.0, 2048, dtype=float)
    measured = np.zeros_like(freq_axis)
    gain_db = np.zeros_like(freq_axis)
    cfg = SimpleNamespace(
        fs=44100,
        num_taps=65536,
        house_freqs=None,
        house_mags=None,
        hpf_settings=None,
        enable_tdc=False,
        enable_null_guard=False,
        lvl_min=500.0,
        lvl_max=2000.0,
        comparison_mode=False,
        lvl_force_offset_db=None,
    )
    stereo_ctx = StereoLinkContext(
        forced_window_hz=(500.0, 2000.0),
        forced_offset_db=1.5,
        shared_target_level_db=5.0,
        shared_target_shift_db=3.0,
    )

    correction_baseline._clear_rt60_cache()
    rt60_key = correction_baseline._rt60_cache_key(freq_axis, measured, cfg.fs, cfg.num_taps)
    correction_baseline._RT60_CACHE[rt60_key] = (0.3, {}, 0.3)

    calls = {"count": 0}

    def fake_compute_leveling(_cfg, _freq, _meas, _target, *, stereo_link_ctx=None):
        calls["count"] += 1
        assert stereo_link_ctx is stereo_ctx
        return (5.0, 1.5, 5.0, 0.0, "ForcedOffset", 500.0, 2000.0)

    monkeypatch.setattr(correction_baseline, "compute_leveling", fake_compute_leveling)

    baseline = correction_baseline._prepare_correction_baseline(
        cfg=cfg,
        freq_axis=freq_axis,
        f_in=freq_axis,
        m_in=measured,
        reflections=[],
        st={},
        m_anal=measured,
        m_plot_db=None,
        is_psy=False,
        cmp=None,
        analysis_mode="native",
        gain_db=gain_db,
        logger=None,
        interpolate_response=lambda *_args, **_kwargs: np.zeros_like(freq_axis),
        _cfg_float_allow_zero=lambda *_args, **_kwargs: 0.0,
        stereo_link_ctx=stereo_ctx,
        presolve_mode=False,
    )

    correction_baseline._clear_rt60_cache()

    assert calls["count"] == 1
    assert baseline.target_shift_db == 3.0
    assert baseline.calc_offset_db == 1.5
    assert baseline.target_level_db_window == 3.0
    assert np.allclose(baseline.target_mags, 3.0)


def test_generate_filter_smoke_trans_width_none_does_not_crash(lr_measurements):
    (f, m, p), _ = lr_measurements

    cfg = FilterConfig(
        fs=48000,
        num_taps=32768,
        filter_type_str="Linear Phase",
        stereo_link=False,
    )
    cfg.trans_width = None

    imp, st = generate_filter(f, m, p, cfg)

    assert imp is not None
    assert len(imp) > 0
    assert isinstance(st, dict)
    assert "filter_mags" in st


@pytest.mark.parametrize("filter_type", ["Linear Phase", "Asymmetric", "Mixed Phase"])
def test_generate_filter_records_ir_alignment_telemetry(filter_type, lr_measurements):
    (f, m, p), _ = lr_measurements

    cfg = FilterConfig(
        fs=48000,
        num_taps=16384,
        filter_type_str=filter_type,
        stereo_link=False,
        comparison_mode=False,
    )
    cfg.ir_anchor_mode = "peak"
    cfg.ir_window_ms_left = 12.0
    cfg.enable_ir_pre_energy_guard = True

    imp, st = generate_filter(f, m, p, cfg)

    assert imp.shape == (cfg.num_taps,)
    assert np.all(np.isfinite(imp))
    assert st["ir_anchor_mode"] == "peak"
    assert "ir_pre_energy_guard_enabled" in st
    assert "ir_realized_level_match_scale" in st
    assert "ir_realized_level_match_reason" in st
    assert "pre_energy_metric_valid" in st
    if "Mixed" in filter_type:
        assert st["mixed_forced_peak_ms"] == pytest.approx(90.0)
        assert "mixed_forced_shift_samples" in st
        assert "post_align_shift_samples" in st


def _sample_tdc_inputs():
    freq_axis = np.linspace(0.0, 500.0, 2001, dtype=float)
    target_mags = np.zeros_like(freq_axis, dtype=float)
    reflections = [{"freq": 50.0, "gd_error": 500.0}]
    rt60_info = 0.4
    return freq_axis, target_mags, reflections, rt60_info


def test_tdc_strength_zero_disables_reduction():
    freq_axis, target_mags, reflections, rt60_info = _sample_tdc_inputs()

    out_enabled = apply_smart_tdc(
        freq_axis,
        target_mags,
        reflections,
        rt60_info,
        base_strength=1.0,
        max_total_reduction_db=9.0,
        max_slope_db_per_oct=0.0,
    )
    out_zero = apply_smart_tdc(
        freq_axis,
        target_mags,
        reflections,
        rt60_info,
        base_strength=0.0,
        max_total_reduction_db=9.0,
        max_slope_db_per_oct=0.0,
    )

    assert float(np.max(target_mags - out_enabled)) > 0.1
    assert np.allclose(out_zero, target_mags, atol=1e-12)


def test_tdc_max_reduction_zero_disables_reduction():
    freq_axis, target_mags, reflections, rt60_info = _sample_tdc_inputs()

    out = apply_smart_tdc(
        freq_axis,
        target_mags,
        reflections,
        rt60_info,
        base_strength=0.8,
        max_total_reduction_db=0.0,
        max_slope_db_per_oct=0.0,
    )

    assert np.allclose(out, target_mags, atol=1e-12)


def test_tdc_soft_knee_handles_moderate_low_frequency_resonance():
    freq_axis = np.linspace(0.0, 500.0, 2001, dtype=float)
    target_mags = np.zeros_like(freq_axis, dtype=float)
    telemetry = {}

    out = apply_smart_tdc(
        freq_axis,
        target_mags,
        [{"freq": 50.0, "gd_error": 80.0, "type": "Resonance"}],
        0.4,
        base_strength=0.5,
        max_total_reduction_db=9.0,
        max_slope_db_per_oct=0.0,
        telemetry=telemetry,
    )

    reduction = target_mags - out
    assert float(np.max(reduction)) > 0.5
    assert bool(telemetry.get("tdc_applied")) is True
    assert int(telemetry.get("tdc_events_used", 0)) == 1
    assert float(telemetry.get("tdc_peak_reduction_db", 0.0)) > 0.5
    assert "tdc_strongest_event_severity" in telemetry
    assert 45.0 <= float(telemetry.get("tdc_peak_reduction_hz", 0.0)) <= 55.0
    assert float(telemetry.get("tdc_reduction_band_high_hz", 0.0)) > float(
        telemetry.get("tdc_reduction_band_low_hz", 0.0)
    )


def test_tdc_modal_events_absent_preserves_existing_reduction():
    freq_axis, target_mags, reflections, rt60_info = _sample_tdc_inputs()

    out_without_modal = apply_smart_tdc(
        freq_axis,
        target_mags,
        reflections,
        rt60_info,
        base_strength=0.7,
        max_total_reduction_db=9.0,
        max_slope_db_per_oct=0.0,
    )
    out_empty_modal = apply_smart_tdc(
        freq_axis,
        target_mags,
        reflections,
        rt60_info,
        base_strength=0.7,
        max_total_reduction_db=9.0,
        max_slope_db_per_oct=0.0,
        modal_events=[],
    )

    assert np.allclose(out_empty_modal, out_without_modal, atol=1e-12)


def test_tdc_strong_modal_decay_evidence_adds_bounded_support():
    freq_axis = np.linspace(0.0, 500.0, 2001, dtype=float)
    target_mags = np.zeros_like(freq_axis, dtype=float)
    reflections = [{"freq": 95.0, "gd_error": 120.0, "type": "Resonance"}]
    modal_events = [
        {
            "freq_hz": 96.0,
            "severity": 0.9,
            "confidence": 0.9,
            "gd_excess_ms": 95.0,
            "decay_severity": 0.8,
            "safe_cut_db": 5.0,
            "safe_width_oct": 0.14,
            "voice_clarity_risk": 0.2,
            "kind": "room_mode",
        }
    ]
    base_telemetry = {}
    modal_telemetry = {}

    base = apply_smart_tdc(
        freq_axis,
        target_mags,
        reflections,
        0.4,
        base_strength=0.6,
        max_total_reduction_db=9.0,
        max_slope_db_per_oct=0.0,
        telemetry=base_telemetry,
    )
    supported = apply_smart_tdc(
        freq_axis,
        target_mags,
        reflections,
        0.4,
        base_strength=0.6,
        max_total_reduction_db=9.0,
        max_slope_db_per_oct=0.0,
        modal_events=modal_events,
        telemetry=modal_telemetry,
    )

    assert float(np.max(target_mags - supported)) > float(np.max(target_mags - base))
    assert float(np.max(target_mags - supported)) <= 5.5 + 1e-9
    assert int(modal_telemetry.get("tdc_modal_events_seen", 0)) == 1
    assert int(modal_telemetry.get("tdc_modal_events_used", 0)) == 1
    assert float(modal_telemetry.get("tdc_strongest_event_modal_support", 0.0)) > 0.5


def test_tdc_uncertain_narrow_modal_event_reduces_enthusiasm():
    freq_axis = np.linspace(0.0, 500.0, 2001, dtype=float)
    target_mags = np.zeros_like(freq_axis, dtype=float)
    reflections = [{"freq": 70.0, "gd_error": 120.0, "type": "Resonance"}]
    modal_events = [
        {
            "freq_hz": 70.0,
            "severity": 0.8,
            "confidence": 0.25,
            "gd_excess_ms": 5.0,
            "decay_severity": 0.05,
            "safe_cut_db": 4.0,
            "safe_width_oct": 1.0 / 72.0,
            "kind": "uncertain",
        }
    ]

    base = apply_smart_tdc(
        freq_axis,
        target_mags,
        reflections,
        0.4,
        base_strength=0.6,
        max_total_reduction_db=9.0,
        max_slope_db_per_oct=0.0,
    )
    cautious = apply_smart_tdc(
        freq_axis,
        target_mags,
        reflections,
        0.4,
        base_strength=0.6,
        max_total_reduction_db=9.0,
        max_slope_db_per_oct=0.0,
        modal_events=modal_events,
    )

    assert float(np.max(target_mags - cautious)) < float(np.max(target_mags - base))


def test_tdc_voice_band_modal_event_is_prioritized_but_capped():
    freq_axis = np.linspace(0.0, 500.0, 2001, dtype=float)
    target_mags = np.zeros_like(freq_axis, dtype=float)
    telemetry = {}

    out = apply_smart_tdc(
        freq_axis,
        target_mags,
        [{"freq": 120.0, "gd_error": 100.0, "type": "Resonance"}],
        0.4,
        base_strength=0.7,
        max_total_reduction_db=9.0,
        max_slope_db_per_oct=0.0,
        modal_events=[
            {
                "freq_hz": 120.0,
                "severity": 0.7,
                "confidence": 0.8,
                "gd_excess_ms": 55.0,
                "decay_severity": 0.5,
                "safe_cut_db": 3.0,
                "safe_width_oct": 0.12,
                "voice_clarity_risk": 0.9,
                "kind": "room_mode",
            }
        ],
        telemetry=telemetry,
    )

    assert bool(telemetry.get("tdc_applied")) is True
    assert int(telemetry.get("tdc_modal_voice_events_used", 0)) == 1
    assert float(np.max(target_mags - out)) <= 3.3 + 1e-9


def test_tdc_skips_high_frequency_reflection_nodes():
    freq_axis = np.linspace(0.0, 1000.0, 4001, dtype=float)
    target_mags = np.zeros_like(freq_axis, dtype=float)
    telemetry = {}

    out = apply_smart_tdc(
        freq_axis,
        target_mags,
        [{"freq": 350.0, "gd_error": 500.0, "type": "Reflection"}],
        0.4,
        base_strength=1.0,
        max_total_reduction_db=9.0,
        max_slope_db_per_oct=0.0,
        max_tdc_hz=250.0,
        telemetry=telemetry,
    )

    assert np.allclose(out, target_mags, atol=1e-12)
    assert bool(telemetry.get("tdc_applied")) is False
    assert int(telemetry.get("tdc_events_seen", 0)) == 1
    assert int(telemetry.get("tdc_events_skipped_high", 0)) == 1


def test_tdc_wide_overreach_records_bounded_reduction_metrics():
    freq_axis = np.linspace(0.0, 500.0, 2001, dtype=float)
    target_mags = np.zeros_like(freq_axis, dtype=float)
    telemetry = {}

    out = apply_smart_tdc(
        freq_axis,
        target_mags,
        [
            {"freq": 45.0, "gd_error": 900.0, "type": "Resonance"},
            {"freq": 62.0, "gd_error": 820.0, "type": "Resonance"},
            {"freq": 88.0, "gd_error": 760.0, "type": "Resonance"},
        ],
        {},
        base_strength=1.0,
        max_total_reduction_db=12.0,
        max_slope_db_per_oct=0.0,
        telemetry=telemetry,
    )

    reduction = target_mags - out
    assert bool(telemetry.get("tdc_applied")) is True
    assert int(telemetry.get("tdc_events_used", 0)) == 3
    assert float(telemetry["tdc_peak_reduction_db"]) <= 12.0 + 1e-9
    assert float(telemetry["tdc_reduction_band_high_hz"]) > float(telemetry["tdc_reduction_band_low_hz"])
    assert float(telemetry["tdc_reduction_area_db_hz"]) > 0.0
    assert np.all(np.isfinite(reduction))
