import math

import numpy as np

from decaycore.auto_mode.search.penalty_voice import compute_voice_clarity_penalty


def _freq():
    return np.geomspace(10.0, 20000.0, 2048)


def _gaussian(freq, center_hz, gain_db, width_oct=1.0 / 10.0):
    x = (np.log2(freq) - math.log2(center_hz)) / width_oct
    return gain_db * np.exp(-0.5 * x * x)


def test_flat_response_has_tiny_voice_penalty():
    freq = _freq()
    out = compute_voice_clarity_penalty(
        freq,
        np.zeros_like(freq),
        target_mag_db=np.zeros_like(freq),
        authority_voice_risk=np.zeros_like(freq),
    )

    assert out["voice_clarity_penalty"] <= 0.05


def test_voice_band_excess_gets_penalty():
    freq = _freq()
    corrected = _gaussian(freq, 110.0, 6.0)
    out = compute_voice_clarity_penalty(
        freq,
        corrected,
        target_mag_db=np.zeros_like(freq),
        authority_voice_risk=np.ones_like(freq),
    )

    assert out["voice_clarity_penalty"] > 0.5
    assert out["voice_band_peak_excess_db"] > 4.0


def test_deep_bass_lift_is_not_voice_penalized():
    freq = _freq()
    corrected = _gaussian(freq, 35.0, 8.0)
    out = compute_voice_clarity_penalty(
        freq,
        corrected,
        target_mag_db=np.zeros_like(freq),
        authority_voice_risk=np.zeros_like(freq),
    )

    assert out["voice_clarity_penalty"] <= 0.1


def test_voice_band_null_dip_remains_low_with_null_risk():
    freq = _freq()
    corrected = _gaussian(freq, 120.0, -8.0)
    null_risk = _gaussian(freq, 120.0, 1.0, width_oct=1.0 / 8.0)
    out = compute_voice_clarity_penalty(
        freq,
        corrected,
        target_mag_db=np.zeros_like(freq),
        authority_null_risk=null_risk,
    )

    assert out["voice_clarity_penalty"] <= 0.25


def test_lr_mismatch_adds_modest_penalty():
    freq = _freq()
    left = np.zeros_like(freq)
    right = _gaussian(freq, 120.0, 6.0)
    corrected = 0.5 * (left + right)
    base = compute_voice_clarity_penalty(
        freq,
        corrected,
        target_mag_db=np.zeros_like(freq),
    )
    with_lr = compute_voice_clarity_penalty(
        freq,
        corrected,
        target_mag_db=np.zeros_like(freq),
        left_corrected_mag_db=left,
        right_corrected_mag_db=right,
    )

    assert with_lr["voice_band_lr_mismatch_db"] > 4.0
    assert with_lr["voice_clarity_penalty"] > base["voice_clarity_penalty"]


def test_missing_optional_inputs_do_not_crash():
    freq = _freq()
    corrected = _gaussian(freq, 110.0, 2.0)
    out = compute_voice_clarity_penalty(freq, corrected)

    assert set(out) == {
        "voice_clarity_penalty",
        "voice_band_rms_error_db",
        "voice_band_peak_excess_db",
        "voice_band_energy_excess_db",
        "voice_band_lr_mismatch_db",
        "voice_band_gd_peak_ms",
        "voice_band_risk_mean",
        "voice_band_risk_peak",
    }
    assert all(np.isfinite(float(value)) for value in out.values())
