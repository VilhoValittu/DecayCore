"""Ground-truth-testit DSP-estimaattoreille synteettisilla signaaleilla."""

import numpy as np
import scipy.signal

from decaycore.dsp.decaycore_analysis import (
    Rt60WindowPolicy,
    analyze_acoustic_confidence,
    calculate_rt60,
)
from decaycore.dsp.dsp_ops import _limit_gd_gradient_ms_per_oct
from decaycore.dsp.hybrid_iir import HybridIIRPolicy, design_hybrid_iir, peaking_eq_response
from decaycore.dsp.modal_analysis import RoomModeEvent
from decaycore.dsp.mode_q import estimate_peak_q
from decaycore.dsp.phase import calculate_minimum_phase
from decaycore.dsp.phase_ir_phase_gradient import gd_grad_metrics


def _synthetic_decay(fs: int, rt60_s: float, dur_s: float, seed: int, noise_floor: float = 1e-4):
    rng = np.random.default_rng(seed)
    t = np.arange(int(dur_s * fs)) / fs
    envelope = 10.0 ** (-3.0 * t / rt60_s)
    impulse = rng.standard_normal(t.size) * envelope
    impulse += rng.standard_normal(t.size) * noise_floor
    return impulse


def test_rt60_recovers_known_decay_time():
    fs = 48000
    rt60_true = 0.6
    impulse = _synthetic_decay(fs, rt60_true, dur_s=1.5, seed=42)

    rt60 = calculate_rt60(impulse, fs)

    assert rt60 > 0.0
    assert abs(rt60 - rt60_true) <= 0.15 * rt60_true


def test_rt60_robust_to_late_burst():
    fs = 48000
    rt60_true = 0.6
    impulse = _synthetic_decay(fs, rt60_true, dur_s=1.5, seed=7)
    rng = np.random.default_rng(123)
    i0 = int(0.45 * fs)
    impulse[i0:i0 + 480] += rng.standard_normal(480) * 0.2

    rt60 = calculate_rt60(impulse, fs)

    assert rt60 > 0.0
    assert abs(rt60 - rt60_true) <= 0.20 * rt60_true


def test_rt60_short_decay():
    fs = 48000
    rt60_true = 0.25
    impulse = _synthetic_decay(fs, rt60_true, dur_s=0.8, seed=99)

    rt60 = calculate_rt60(impulse, fs)

    assert rt60 > 0.0
    assert abs(rt60 - rt60_true) <= 0.20 * rt60_true


def test_estimate_peak_q_gaussian_peak():
    freqs = np.linspace(20.0, 320.0, 2401)
    f0 = 100.0
    sigma_hz = 5.0
    fwhm_hz = 2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma_hz
    values = 6.0 * np.exp(-0.5 * ((freqs - f0) / sigma_hz) ** 2)
    peaks, _ = scipy.signal.find_peaks(values, prominence=1.0)
    assert peaks.size == 1

    q_values, bw_hz = estimate_peak_q(freqs, values, peaks, min_bw_ratio=0.0)

    q_expected = f0 / fwhm_hz
    assert abs(bw_hz[0] - fwhm_hz) <= 0.05 * fwhm_hz
    assert abs(q_values[0] - q_expected) <= 0.05 * q_expected


def test_estimate_peak_q_bandwidth_floor_caps_single_bin_spike():
    freqs = np.linspace(20.0, 320.0, 601)
    values = np.zeros_like(freqs)
    spike_idx = 200
    values[spike_idx] = 1.0
    peaks, _ = scipy.signal.find_peaks(values, height=0.5)
    assert peaks.size == 1

    q_values, bw_hz = estimate_peak_q(freqs, values, peaks, min_bw_ratio=0.02)

    f_peak = freqs[spike_idx]
    assert bw_hz[0] >= 0.02 * f_peak - 1e-9
    assert q_values[0] <= 50.0 + 1e-9


def test_estimate_peak_q_empty_input():
    q_values, bw_hz = estimate_peak_q(np.asarray([]), np.asarray([]), np.asarray([], dtype=int))
    assert q_values.size == 0
    assert bw_hz.size == 0


def _legacy_min_phase_hilbert(mags_lin_fft: np.ndarray) -> np.ndarray:
    ln_mag = np.log(np.maximum(np.abs(mags_lin_fft), 1e-10))
    full_ln_mag = np.concatenate((ln_mag, ln_mag[-2:0:-1]))
    analytic = scipy.signal.hilbert(full_ln_mag)
    return np.unwrap(-np.imag(analytic)[: len(mags_lin_fft)])


def test_minimum_phase_matches_analytic_reference():
    # Minimivaiheinen FIR H(z) = (1 - a z^-1)^2, jonka vaihe tunnetaan
    # suljetussa muodossa. Nolla lahella yksikkoympyraa tekee kepstrista
    # hitaasti vaimenevan, jolloin aliasoituminen nakyy ilman
    # ylinaytteistysta.
    n_fft = 1024
    a = 0.998
    w = 2.0 * np.pi * np.arange(n_fft // 2 + 1) / n_fft
    factor = 1.0 - a * np.exp(-1j * w)
    mags = np.abs(factor) ** 2
    phase_true = 2.0 * np.arctan2(a * np.sin(w), 1.0 - a * np.cos(w))

    phase_new = calculate_minimum_phase(mags, max_phase_deg=None)
    phase_legacy = _legacy_min_phase_hilbert(mags)

    interior = slice(8, len(mags) - 8)
    err_new = float(np.max(np.abs(phase_new[interior] - phase_true[interior])))
    err_legacy = float(np.max(np.abs(phase_legacy[interior] - phase_true[interior])))

    assert err_new < err_legacy
    assert err_new < 0.02


def test_minimum_phase_flat_magnitude_has_zero_phase():
    mags = np.ones(513)
    phase = calculate_minimum_phase(mags, max_phase_deg=None)
    assert float(np.max(np.abs(phase))) < 1e-9


def _complex_from_gd_profile(freq: np.ndarray, gd_ms: np.ndarray) -> np.ndarray:
    gd_s = np.asarray(gd_ms, dtype=float) / 1000.0
    seg = 0.5 * (gd_s[1:] + gd_s[:-1]) * np.diff(freq)
    phase = -2.0 * np.pi * np.concatenate([[0.0], np.cumsum(seg)])
    return np.exp(1j * phase)


def test_confidence_local_threshold_flags_lf_anomaly_despite_noisy_hf():
    # HF-kohinan ei pida nostaa kynnysta niin, etta selva LF-anomalia
    # luokittuu luotettavaksi - eika puhtaan LF:n konfidenssin pida karsia.
    fs = 48000
    freq = np.fft.rfftfreq(65536, 1.0 / fs)
    rng = np.random.default_rng(11)

    gd_ms = np.zeros_like(freq)
    anomaly = (freq >= 62.0) & (freq <= 72.0)
    gd_ms[anomaly] += rng.uniform(-12.0, 12.0, int(np.count_nonzero(anomaly)))
    hf = freq > 2000.0
    gd_ms[hf] += rng.uniform(-20.0, 20.0, int(np.count_nonzero(hf)))

    conf, _nodes, _gd = analyze_acoustic_confidence(freq, _complex_from_gd_profile(freq, gd_ms), fs)

    anomaly_core = (freq >= 63.0) & (freq <= 71.0)
    clean_lf = (freq >= 120.0) & (freq <= 1000.0)
    assert float(np.mean(conf[anomaly_core])) < 0.5
    assert float(np.mean(conf[clean_lf])) > 0.85
    assert float(np.mean(conf[hf])) < 0.7
    assert np.all(conf >= 0.0) and np.all(conf <= 1.0)


def _room_mode_event(**overrides):
    data = {
        "freq_hz": 41.5,
        "peak_db": 8.0,
        "width_hz": 6.0,
        "width_oct": 0.12,
        "q_estimate": 6.9,
        "area_db_oct": 0.8,
        "gd_excess_ms": 35.0,
        "confidence": 0.9,
        "severity": 0.8,
        "correction_priority": 0.85,
        "cut_priority": 0.9,
        "safe_cut_db": 4.5,
        "voice_clarity_risk": 0.1,
        "kind": "room_mode",
    }
    data.update(overrides)
    return RoomModeEvent(**data)


def test_hybrid_iir_mode_fit_refines_freq_and_q():
    fs = 48000
    freq = np.linspace(20.0, 200.0, 1024)
    # Todellinen moodi: +8 dB @ 60 Hz, Q 6. RBJ-peakingin boost on cutin
    # kaanteisvaste, joten mitattu ylimaara saadaan peilaamalla cut.
    cut = peaking_eq_response(freq, fs, 60.0, 6.0, -8.0)
    mag = -20.0 * np.log10(np.maximum(np.abs(cut), 1e-12))

    event = _room_mode_event(freq_hz=64.0, q_estimate=3.2, width_oct=0.2, width_hz=12.0)
    policy = HybridIIRPolicy(enabled=True)

    result = design_hybrid_iir([event], freq, fs, policy, measured_mag_db=mag)
    assert len(result.biquads) == 1
    biquad = result.biquads[0]
    assert biquad.fit_refined is True
    assert abs(biquad.freq_hz - 60.0) <= 1.0
    assert abs(biquad.q - 6.0) <= 1.5
    # Leikkaussyvyys ei muutu sovituksessa.
    assert biquad.gain_db == -4.5


def test_hybrid_iir_without_measurement_keeps_event_parameters():
    fs = 48000
    freq = np.linspace(20.0, 200.0, 1024)
    event = _room_mode_event(freq_hz=64.0, q_estimate=3.2)

    result = design_hybrid_iir([event], freq, fs, HybridIIRPolicy(enabled=True))
    assert len(result.biquads) == 1
    biquad = result.biquads[0]
    assert biquad.fit_refined is False
    assert biquad.freq_hz == 64.0


def test_hybrid_iir_mode_fit_falls_back_when_no_local_peak():
    # Tasainen vaste: sovitus ei voi parantaa jaannosta -> alkuperaiset arvot.
    fs = 48000
    freq = np.linspace(20.0, 200.0, 1024)
    mag = np.zeros_like(freq)
    event = _room_mode_event(freq_hz=64.0, q_estimate=5.0)

    result = design_hybrid_iir([event], freq, fs, HybridIIRPolicy(enabled=True), measured_mag_db=mag)
    assert len(result.biquads) == 1
    assert result.biquads[0].freq_hz == 64.0


def test_rt60_explicit_default_policy_matches_implicit():
    fs = 48000
    impulse = _synthetic_decay(fs, 0.5, dur_s=1.2, seed=3)

    assert calculate_rt60(impulse, fs) == calculate_rt60(impulse, fs, Rt60WindowPolicy())


def test_rt60_custom_window_policy_recovers_clean_decay():
    # Matalat ikkunat puhtaalla vasteella: EDC on lineaarinen, joten
    # mukautetut ikkunarajat tuottavat saman vaimenemisajan - todentaa,
    # etta politiikan rajat oikeasti ohjaavat sovitusta.
    fs = 48000
    rt60_true = 0.5
    impulse = _synthetic_decay(fs, rt60_true, dur_s=1.2, seed=21, noise_floor=1e-5)

    shallow = Rt60WindowPolicy(t20_lo_db=-3.0, t20_hi_db=-15.0, t30_lo_db=-3.0, t30_hi_db=-18.0)
    rt60 = calculate_rt60(impulse, fs, shallow)

    assert rt60 > 0.0
    assert abs(rt60 - rt60_true) <= 0.15 * rt60_true


def test_rt60_policy_min_fit_points_gate():
    fs = 48000
    impulse = _synthetic_decay(fs, 0.5, dur_s=1.2, seed=21)

    impossible = Rt60WindowPolicy(min_fit_points=10**9)
    assert calculate_rt60(impulse, fs, impossible) == 0.0


def _phase_from_gd_log2f(freq: np.ndarray, gd_ms: np.ndarray) -> np.ndarray:
    gd_s = np.asarray(gd_ms, dtype=float) / 1000.0
    seg = 0.5 * (gd_s[1:] + gd_s[:-1]) * np.diff(freq)
    return -2.0 * np.pi * np.concatenate([[0.0], np.cumsum(seg)])


def test_gd_curvature_limiter_smooths_sharp_kink_without_worsening_gradient():
    # GD-aaltoilu, jonka gradientti pysyy rajan alla mutta kaarevuus ylittaa
    # automaattisen 4x-rajan selvasti: rajoittimen pitaa tasoittaa kaarevuus
    # eika gradientti saa pahentua (sama invariantti kuin gradienttirajalla).
    freq = np.geomspace(20.0, 250.0, 512)
    period_oct = 0.5
    amp_ms = 1.2
    gd_ms = amp_ms * np.sin(2.0 * np.pi * np.log2(freq / 20.0) / period_oct)
    phase = _phase_from_gd_log2f(freq, gd_ms)
    mask = np.ones_like(freq, dtype=bool)

    before = gd_grad_metrics(freq, phase, mask=mask)
    limited = _limit_gd_gradient_ms_per_oct(
        freq,
        phase,
        mask=mask,
        max_grad_ms_per_oct=30.0,
        f_min=20.0,
        f_max=250.0,
        grad_smooth_sigma=0.8,
        soft_limit=True,
    )
    after = gd_grad_metrics(freq, limited, mask=mask)

    assert float(before["max_ms_per_oct2"]) > 80.0
    assert float(after["max_ms_per_oct2"]) <= 0.7 * float(before["max_ms_per_oct2"])
    assert float(after["max_ms_per_oct"]) <= float(before["max_ms_per_oct"]) + 1e-9


def test_gd_curvature_limit_disabled_restores_legacy_behavior():
    # max_curv_ms_per_oct2 <= 0 kytkee kaarevuusrajan pois: matalagradienttinen
    # aaltoilu ei kayttaydy eri tavalla kuin pelkalla gradienttirajalla.
    freq = np.geomspace(20.0, 250.0, 512)
    gd_ms = 1.2 * np.sin(2.0 * np.pi * np.log2(freq / 20.0) / 0.5)
    phase = _phase_from_gd_log2f(freq, gd_ms)
    mask = np.ones_like(freq, dtype=bool)

    limited_off = _limit_gd_gradient_ms_per_oct(
        freq, phase, mask=mask, max_grad_ms_per_oct=30.0,
        f_min=20.0, f_max=250.0, grad_smooth_sigma=0.8, soft_limit=True,
        max_curv_ms_per_oct2=0.0,
    )
    off_metrics = gd_grad_metrics(freq, limited_off, mask=mask)
    on_metrics = gd_grad_metrics(
        freq,
        _limit_gd_gradient_ms_per_oct(
            freq, phase, mask=mask, max_grad_ms_per_oct=30.0,
            f_min=20.0, f_max=250.0, grad_smooth_sigma=0.8, soft_limit=True,
        ),
        mask=mask,
    )
    # Pois kytkettyna kaarevuus jaa korkeaksi, paalle kytkettyna laskee.
    assert float(off_metrics["max_ms_per_oct2"]) > float(on_metrics["max_ms_per_oct2"])
