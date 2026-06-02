import numpy as np
import pytest

from decaycore.dsp.target_synthesis import (
    _SYNTH_BASE_MAGS,
    _SYNTH_FREQS,
    synthesize_target_from_measurements,
)

_FLAT_FREQS = np.logspace(np.log10(20), np.log10(20000), 512)


def _flat_mags(db=0.0):
    return np.full_like(_FLAT_FREQS, float(db))


def _harman6_mags():
    return np.interp(_FLAT_FREQS, _SYNTH_FREQS[1:], _SYNTH_BASE_MAGS[1:])


def _bass_peak_mags(peak_db=10.0, peak_hz=80.0, width_oct=1.0):
    """Harman6-shaped measurement plus a bass peak."""
    base = _harman6_mags()
    sigma = width_oct * np.log(2) / (2 * np.sqrt(2 * np.log(2)))
    bump = peak_db * np.exp(-0.5 * ((np.log2(_FLAT_FREQS / peak_hz)) / sigma) ** 2)
    return base + bump


def test_bass_heavy_room_no_double_compensation():
    """Bass-shelf and tilt must not both independently penalise bass.

    Synthesise from a measurement that has Harman6 mid/HF but a 10 dB bass
    peak at 80 Hz.  The adaptive target should lower the bass shelf somewhat,
    but the tilt stage (now mid-range only) should add negligible additional
    bass reduction.
    """
    m = _bass_peak_mags(peak_db=10.0, peak_hz=80.0)
    result = synthesize_target_from_measurements(
        _FLAT_FREQS, m, _FLAT_FREQS, m,
        bass_comp_frac=0.50,
        tilt_comp_frac=0.30,
        hf_comp_frac=0.25,
    )
    assert result is not None
    out_f, out_m = result

    # Reference: synthesise with tilt_comp_frac=0 (bass shelf only)
    result_shelf_only = synthesize_target_from_measurements(
        _FLAT_FREQS, m, _FLAT_FREQS, m,
        bass_comp_frac=0.50,
        tilt_comp_frac=0.0,
        hf_comp_frac=0.25,
    )
    assert result_shelf_only is not None
    _, ref_m = result_shelf_only

    # At 80 Hz the tilt stage must add at most 1.0 dB of extra bass reduction
    # (previously the full-spectrum slope caused ~2–4 dB of additional dip)
    idx_80 = int(np.argmin(np.abs(out_f - 80.0)))
    extra_penalty_db = ref_m[idx_80] - out_m[idx_80]
    assert extra_penalty_db < 1.0, (
        f"Tilt added {extra_penalty_db:.2f} dB extra bass penalty at 80 Hz — "
        "double-compensation still present"
    )


def test_bright_room_hf_target_raised():
    """A room brighter than Harman6 above 5 kHz should yield a higher HF target.

    Previously the HF compensation was one-directional (only lowered target).
    """
    base = _harman6_mags()
    # Add +3 dB/oct HF rise above 5 kHz
    hf_boost = np.where(
        _FLAT_FREQS > 5000.0,
        3.0 * np.log2(_FLAT_FREQS / 5000.0),
        0.0,
    )
    m = base + hf_boost

    result_bright = synthesize_target_from_measurements(
        _FLAT_FREQS, m, _FLAT_FREQS, m,
        hf_comp_frac=0.25,
        bass_comp_frac=0.0,
        tilt_comp_frac=0.0,
    )
    result_neutral = synthesize_target_from_measurements(
        _FLAT_FREQS, _harman6_mags(), _FLAT_FREQS, _harman6_mags(),
        hf_comp_frac=0.25,
        bass_comp_frac=0.0,
        tilt_comp_frac=0.0,
    )
    assert result_bright is not None and result_neutral is not None
    _, m_bright = result_bright
    _, m_neutral = result_neutral

    idx_16k = int(np.argmin(np.abs(np.asarray(result_bright[0]) - 16000.0)))
    assert m_bright[idx_16k] > m_neutral[idx_16k], (
        "Bright room should yield higher HF target than neutral room "
        "(bidirectional HF compensation not working)"
    )


def test_rt60_long_bass_reduces_adaptive_bass_target():
    m = _bass_peak_mags(peak_db=10.0, peak_hz=80.0)
    short_bass = {
        "measured_rt60_bands_l": {31.5: 0.22, 63.0: 0.24, 100.0: 0.25, 500.0: 0.42, 1000.0: 0.40, 2000.0: 0.38},
        "measured_rt60_bands_r": {31.5: 0.23, 63.0: 0.25, 100.0: 0.26, 500.0: 0.41, 1000.0: 0.39, 2000.0: 0.37},
    }
    long_bass = {
        "measured_rt60_bands_l": {31.5: 0.78, 63.0: 0.82, 100.0: 0.80, 500.0: 0.42, 1000.0: 0.40, 2000.0: 0.38},
        "measured_rt60_bands_r": {31.5: 0.80, 63.0: 0.84, 100.0: 0.82, 500.0: 0.41, 1000.0: 0.39, 2000.0: 0.37},
    }

    result_short = synthesize_target_from_measurements(
        _FLAT_FREQS, m, _FLAT_FREQS, m,
        measurements=short_bass,
    )
    result_long = synthesize_target_from_measurements(
        _FLAT_FREQS, m, _FLAT_FREQS, m,
        measurements=long_bass,
    )

    assert result_short is not None and result_long is not None
    out_f, short_m = result_short
    _, long_m = result_long
    idx_80 = int(np.argmin(np.abs(out_f - 80.0)))
    assert long_m[idx_80] < short_m[idx_80]


@pytest.mark.parametrize(
    "measurements",
    [
        None,
        {},
        {"measured_rt60_bands_l": {"bad": "value"}, "measured_rt60_bands_r": {63.0: 0.0}},
    ],
)
def test_rt60_metadata_missing_or_invalid_is_neutral(measurements):
    m = _bass_peak_mags(peak_db=10.0, peak_hz=80.0)
    baseline = synthesize_target_from_measurements(_FLAT_FREQS, m, _FLAT_FREQS, m)
    result = synthesize_target_from_measurements(
        _FLAT_FREQS, m, _FLAT_FREQS, m,
        measurements=measurements,
    )

    assert baseline is not None and result is not None
    np.testing.assert_allclose(result[0], baseline[0])
    np.testing.assert_allclose(result[1], baseline[1])


def test_rt60_metadata_delta_is_limited_per_anchor():
    m = _bass_peak_mags(peak_db=14.0, peak_hz=80.0)
    extreme_decay = {
        "measured_rt60_bands_l": {31.5: 1.8, 63.0: 1.9, 100.0: 1.7, 500.0: 0.22, 1000.0: 0.20, 2000.0: 0.18, 4000.0: 1.2, 8000.0: 1.1},
        "measured_rt60_bands_r": {31.5: 1.7, 63.0: 1.8, 100.0: 1.6, 500.0: 0.23, 1000.0: 0.21, 2000.0: 0.19, 4000.0: 1.1, 8000.0: 1.0},
    }

    baseline = synthesize_target_from_measurements(_FLAT_FREQS, m, _FLAT_FREQS, m)
    adjusted = synthesize_target_from_measurements(
        _FLAT_FREQS, m, _FLAT_FREQS, m,
        measurements=extreme_decay,
    )

    assert baseline is not None and adjusted is not None
    delta = np.asarray(adjusted[1], dtype=float) - np.asarray(baseline[1], dtype=float)
    assert float(np.max(np.abs(delta))) <= 2.0 + 1e-9
