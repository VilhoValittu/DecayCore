import numpy as np

from decaycore.auto_mode.api import (
    _estimate_auto_hpf_from_response,
    _estimate_auto_mag_c_min_hz,
    _resolve_auto_hpf_application,
)


def _butter_hpf_mag_db(freqs: np.ndarray, fc_hz: float, order: int) -> np.ndarray:
    f = np.asarray(freqs, dtype=float)
    with np.errstate(divide="ignore"):
        return -10.0 * np.log10(1.0 + np.power(float(fc_hz) / np.maximum(f, 1e-9), 2 * int(order)))


def _butter_f6_hz(fc_hz: float, order: int) -> float:
    c6 = float((10.0**0.6) - 1.0)
    return float(fc_hz) / float(np.power(c6, 1.0 / (2.0 * int(order))))


def test_auto_f6_estimator_recovers_butterworth_like_rolloff():
    f = np.geomspace(10.0, 20000.0, 4096).astype(float)
    fc_true = 36.0
    order_true = 4
    m = _butter_hpf_mag_db(f, fc_true, order_true)

    f6 = _estimate_auto_mag_c_min_hz(f, m, f, m, default_hz=25.0)

    assert abs(float(f6) - _butter_f6_hz(fc_true, order_true)) <= 5.0


def test_auto_f6_estimator_rejects_isolated_infrabass_spike():
    f = np.geomspace(10.0, 20000.0, 4096).astype(float)
    fc_true = 42.0
    order_true = 4
    spike = 18.0 * np.exp(-0.5 * (np.log2(f / 17.0) / 0.035) ** 2.0)
    m = _butter_hpf_mag_db(f, fc_true, order_true) + spike

    f6 = _estimate_auto_mag_c_min_hz(f, m, f, m, default_hz=25.0)

    assert float(f6) >= _butter_f6_hz(fc_true, order_true) - 8.0


def test_auto_f6_estimator_stays_stable_with_local_bass_null():
    f = np.geomspace(10.0, 20000.0, 4096).astype(float)
    fc_true = 38.0
    order_true = 4
    local_null = -9.0 * np.exp(-0.5 * (np.log2(f / 34.0) / 0.045) ** 2.0)
    m = _butter_hpf_mag_db(f, fc_true, order_true) + local_null

    f6 = _estimate_auto_mag_c_min_hz(f, m, f, m, default_hz=25.0)

    assert 28.0 <= float(f6) <= 48.0


def test_auto_f6_estimator_uses_higher_channel_when_stereo_disagrees():
    f = np.geomspace(10.0, 20000.0, 4096).astype(float)
    left = _butter_hpf_mag_db(f, 25.0, 4)
    right = _butter_hpf_mag_db(f, 52.0, 4)

    f6 = _estimate_auto_mag_c_min_hz(f, left, f, right, default_hz=25.0)

    assert float(f6) >= _butter_f6_hz(52.0, 4) - 4.0


def test_auto_f6_estimator_keeps_full_range_response_at_minimum():
    f = np.geomspace(10.0, 20000.0, 2048).astype(float)
    m = np.zeros_like(f, dtype=float)

    f6 = _estimate_auto_mag_c_min_hz(f, m, f, m, default_hz=25.0)

    assert 10.0 <= float(f6) <= 12.0


def test_auto_hpf_estimator_recovers_butterworth_like_rolloff():
    rng = np.random.default_rng(1234)
    f = np.geomspace(10.0, 20000.0, 4096).astype(float)

    fc_true = 38.0
    order_true = 4  # 24 dB/oct
    hpf = _butter_hpf_mag_db(f, fc_true, order_true)

    # Add mild room-like structure while keeping LF roll-off dominant.
    room = 0.35 * np.sin(np.log(f) * 1.9) * np.exp(-0.5 * (np.log2(np.maximum(f, 1.0) / 120.0) / 1.8) ** 2)
    n_l = rng.normal(0.0, 0.06, size=f.size)
    n_r = rng.normal(0.0, 0.06, size=f.size)
    m_l = hpf + room + n_l
    m_r = _butter_hpf_mag_db(f, fc_true * 1.03, order_true) + room * 0.92 + n_r

    res = _estimate_auto_hpf_from_response(
        f,
        m_l,
        f,
        m_r,
        default_freq_hz=20.0,
        default_slope_db_oct=24,
    )

    assert isinstance(res, dict)
    assert str(res.get("method")) == "response_fit"
    assert bool(res.get("enabled", False))
    assert abs(float(res.get("freq", 0.0)) - fc_true) <= 6.0
    assert int(res.get("slope_db_oct", 0)) == 24
    assert float(res.get("confidence", 0.0)) >= 0.70


def test_auto_hpf_estimator_stays_off_for_flat_response():
    f = np.geomspace(10.0, 20000.0, 2048).astype(float)
    m = np.zeros_like(f, dtype=float)

    res = _estimate_auto_hpf_from_response(
        f,
        m,
        f,
        m,
        default_freq_hz=23.0,
        default_slope_db_oct=24,
    )

    assert isinstance(res, dict)
    assert not bool(res.get("enabled", True))
    assert str(res.get("method")) == "fallback_default"
    assert abs(float(res.get("freq", 0.0)) - 23.0) <= 0.1
    assert int(res.get("slope_db_oct", 0)) == 24
    assert float(res.get("confidence", 1.0)) <= 0.1


def test_auto_hpf_response_fit_becomes_search_seed_even_when_user_hpf_is_off():
    auto_hpf = {
        "enabled": True,
        "freq": 18.2,
        "slope_db_oct": 24,
        "confidence": 0.81,
        "method": "response_fit",
    }

    resolved = _resolve_auto_hpf_application(auto_hpf, user_hpf_enabled=False)

    assert bool(resolved.get("enabled", False))
    assert bool(resolved.get("applied", False))
    assert str(resolved.get("decision")) == "apply_seed"


def test_auto_hpf_falls_back_to_user_seed_when_fit_is_not_enabled():
    auto_hpf = {
        "enabled": False,
        "freq": 22.0,
        "slope_db_oct": 18,
        "confidence": 0.52,
        "method": "response_fit",
    }

    resolved = _resolve_auto_hpf_application(auto_hpf, user_hpf_enabled=True)

    assert not bool(resolved.get("applied", True))
    assert str(resolved.get("decision")) == "keep_user_seed"
