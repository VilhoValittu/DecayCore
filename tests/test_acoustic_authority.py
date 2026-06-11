import math

import numpy as np
import pytest

from decaycore.dsp.acoustic_authority import (
    AcousticAuthority,
    acoustic_authority_to_stats,
    build_acoustic_authority_map,
)
from decaycore.dsp.modal_analysis import RoomModeEvent


def _bump(freq_axis, center_hz, gain_db, width_oct):
    return gain_db * np.exp(-0.5 * (np.log2(freq_axis / float(center_hz)) / float(width_oct)) ** 2)


def _nearest_idx(freq_axis, freq_hz):
    return int(np.argmin(np.abs(np.asarray(freq_axis, dtype=float) - float(freq_hz))))


def _authority_arrays(authority: AcousticAuthority):
    return [
        authority.cut_authority,
        authority.boost_authority,
        authority.phase_authority,
        authority.modal_support,
        authority.decay_need,
        authority.null_risk,
        authority.reflection_risk,
        authority.repeatability,
        authority.minphase_likelihood,
        authority.voice_risk,
    ]


def _assert_jsonish(value):
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        assert math.isfinite(value)
        return
    if isinstance(value, list):
        for item in value:
            _assert_jsonish(item)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            assert isinstance(key, str)
            _assert_jsonish(item)
        return
    raise AssertionError(f"non JSON-friendly value: {type(value)!r}")


def test_acoustic_authority_shape_finite_and_clipped():
    freq = np.geomspace(10.0, 20000.0, 1024)
    measured = np.zeros_like(freq)
    confidence = np.ones_like(freq)

    authority = build_acoustic_authority_map(freq, measured, confidence_mask=confidence)

    assert authority.freq_axis.shape == freq.shape
    assert np.all(np.isfinite(authority.freq_axis))
    for arr in _authority_arrays(authority):
        assert arr.shape == freq.shape
        assert np.all(np.isfinite(arr))
        assert float(np.min(arr)) >= 0.0
        assert float(np.max(arr)) <= 1.0


def test_acoustic_authority_deep_null_blocks_boost():
    freq = np.geomspace(10.0, 20000.0, 1024)
    measured = -_bump(freq, 80.0, 18.0, 0.035)
    idx = _nearest_idx(freq, 80.0)

    authority = build_acoustic_authority_map(
        freq,
        measured,
        confidence_mask=np.ones_like(freq),
        modal_events=[],
    )

    assert float(authority.null_risk[idx]) > 0.5
    assert float(authority.boost_authority[idx]) < 0.35
    assert float(np.mean(authority.cut_authority)) > 0.55


def test_acoustic_authority_modal_peak_allows_cut():
    freq = np.geomspace(10.0, 20000.0, 1024)
    measured = _bump(freq, 45.0, 8.0, 0.12)
    event = RoomModeEvent(
        freq_hz=45.0,
        peak_db=8.0,
        width_hz=12.0,
        width_oct=0.25,
        q_estimate=4.0,
        area_db_oct=1.2,
        gd_excess_ms=35.0,
        decay_severity=0.8,
        confidence=0.9,
        severity=0.85,
        correction_priority=0.85,
        cut_priority=0.9,
        safe_cut_db=5.0,
        safe_width_oct=0.28,
        kind="room_mode",
    )
    idx = _nearest_idx(freq, 45.0)

    authority = build_acoustic_authority_map(
        freq,
        measured,
        confidence_mask=np.full_like(freq, 0.9),
        modal_events=[event],
    )

    assert float(authority.modal_support[idx]) > 0.6
    assert float(authority.decay_need[idx]) > 0.5
    assert float(authority.cut_authority[idx]) > float(authority.boost_authority[idx])
    assert float(authority.cut_authority[idx]) > 0.55


def test_acoustic_authority_decay_need_increases_modal_cut_authority():
    freq = np.geomspace(10.0, 20000.0, 1024)
    measured = _bump(freq, 55.0, 7.0, 0.12)
    idx = _nearest_idx(freq, 55.0)

    base_event = dict(
        freq_hz=55.0,
        peak_db=7.0,
        width_hz=12.0,
        width_oct=0.22,
        q_estimate=4.0,
        area_db_oct=1.0,
        gd_excess_ms=10.0,
        confidence=0.9,
        severity=0.75,
        correction_priority=0.75,
        cut_priority=0.75,
        safe_cut_db=4.0,
        safe_width_oct=0.24,
        kind="room_mode",
    )
    low_decay = RoomModeEvent(decay_severity=0.05, **base_event)
    high_decay = RoomModeEvent(decay_severity=0.9, **base_event)

    low = build_acoustic_authority_map(
        freq,
        measured,
        confidence_mask=np.full_like(freq, 0.9),
        modal_events=[low_decay],
    )
    high = build_acoustic_authority_map(
        freq,
        measured,
        confidence_mask=np.full_like(freq, 0.9),
        modal_events=[high_decay],
    )

    assert float(high.decay_need[idx]) > float(low.decay_need[idx])
    assert float(high.cut_authority[idx]) > float(low.cut_authority[idx])
    assert float(high.boost_authority[idx]) == pytest.approx(float(low.boost_authority[idx]), abs=0.08)


def test_acoustic_authority_reflection_node_reduces_phase_authority():
    freq = np.geomspace(10.0, 20000.0, 1024)
    measured = np.zeros_like(freq)
    reflection_idx = _nearest_idx(freq, 700.0)
    clean_idx = _nearest_idx(freq, 520.0)

    authority = build_acoustic_authority_map(
        freq,
        measured,
        confidence_mask=np.ones_like(freq),
        reflection_nodes=[{"freq": 700.0, "gd_error": 18.0, "type": "Reflection"}],
        phase_limit_hz=1000.0,
        modal_events=[],
    )

    assert float(authority.reflection_risk[reflection_idx]) > 0.4
    assert float(authority.phase_authority[reflection_idx]) < float(authority.phase_authority[clean_idx])


def test_acoustic_authority_missing_optional_inputs_do_not_crash():
    freq = np.geomspace(10.0, 20000.0, 1024)
    measured = np.zeros_like(freq)

    authority = build_acoustic_authority_map(freq, measured)

    assert authority.freq_axis.shape == freq.shape
    for arr in _authority_arrays(authority):
        assert arr.shape == freq.shape
        assert np.all(np.isfinite(arr))


def test_acoustic_authority_stats_are_json_friendly():
    freq = np.geomspace(10.0, 20000.0, 128)
    authority = build_acoustic_authority_map(freq, np.zeros_like(freq), modal_events=[])

    stats = acoustic_authority_to_stats(authority, include_arrays=True)

    assert stats["acoustic_authority_version"] == 2
    assert len(stats["authority_cut"]) == freq.size
    assert len(stats["authority_decay_need"]) == freq.size
    assert "authority_boost_mean_20_300" in stats
    _assert_jsonish(stats)
