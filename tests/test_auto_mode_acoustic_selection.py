from types import SimpleNamespace

import numpy as np
import pytest

from decaycore.auto_mode.runtime_context import (
    _auto_get_top_modes_hz,
    _auto_get_worst_mode_hz,
    _auto_mode_band,
)
from decaycore.auto_mode.materialize import (
    AutoModeMaterializeContext,
    build_materialize_helpers,
)
from decaycore.auto_mode.scoring_metrics import (
    _auto_score_result,
    _auto_bass_boost_metrics_from_stats,
    _auto_channel_overfit_metrics_from_stats,
    _auto_correction_sharpness_metrics_from_stats,
    _auto_dip_fill_risk_metrics_from_stats,
    _auto_focus_ripple_from_stats,
    _auto_residual_peak_metrics_from_stats,
    _auto_target_tracking_metrics_from_stats,
    _auto_target_tracking_penalty,
    compute_broad_residual_peak_metrics,
)
from decaycore.auto_mode.scoring_ranking import _auto_is_better_refine, _auto_phase2_hard_gate_pool
from decaycore.auto_mode.winner_polish import (
    apply_hpf_winner_polish,
    apply_low_bass_cut_winner_polish,
    apply_mag_c_min_winner_polish,
    apply_residual_peak_winner_polish,
)
from decaycore.auto_mode.search.residual_analysis import _run_bounds_from_mask


def test_focus_ripple_tracks_corrected_response_error_not_filter_realization_delta():
    freq_axis = [20.0, 25.0, 31.5, 40.0, 50.0, 63.0, 80.0, 100.0, 125.0, 160.0]
    measured = [3.0] * len(freq_axis)
    target = [0.0] * len(freq_axis)
    realized_filter = [-3.0] * len(freq_axis)
    predicted_filter = [-1.0] * len(freq_axis)

    ripple = _auto_focus_ripple_from_stats(
        {
            "freq_axis": freq_axis,
            "measured_mags": measured,
            "target_mags": target,
            "realized_filter_mags": realized_filter,
            "predicted_filter_mags": predicted_filter,
            "confidence_mask": [1.0] * len(freq_axis),
        },
        focus_lo_hz=20.0,
        focus_hi_hz=160.0,
    )

    assert ripple is not None
    assert float(ripple) == pytest.approx(0.0, abs=1e-9)


def test_residual_peak_metrics_detect_local_peak_without_penalizing_tilt():
    freq_axis = np.geomspace(20.0, 250.0, 160)
    broad_tilt = 0.55 * np.log2(freq_axis / 20.0)
    local_peak = 6.0 * np.exp(-0.5 * (np.log2(freq_axis / 63.0) / 0.14) ** 2)
    err = broad_tilt + local_peak

    metrics = _auto_residual_peak_metrics_from_stats(
        {
            "freq_axis": freq_axis,
            "measured_mags": err,
            "target_mags": np.zeros_like(freq_axis),
            "realized_filter_mags": np.zeros_like(freq_axis),
            "confidence_mask": np.ones_like(freq_axis),
        },
        lo_hz=20.0,
        hi_hz=250.0,
    )

    assert int(metrics["residual_peak_count"]) >= 1
    assert float(metrics["worst_residual_peak_db"]) > 2.0
    assert float(metrics["worst_residual_peak_hz"]) == pytest.approx(63.0, rel=0.08)


def test_residual_peak_run_bounds_cover_full_threshold_run():
    mask = np.array([False, True, True, True, False, True, False], dtype=bool)

    left, right = _run_bounds_from_mask(mask)

    assert left.tolist() == [0, 1, 1, 1, 4, 5, 6]
    assert right.tolist() == [0, 3, 3, 3, 4, 5, 6]


def _synthetic_residual_peak_metrics(*, peak_hz: float, peak_db: float, width_oct: float = 0.13):
    freq_axis = np.geomspace(20.0, 250.0, 220)
    err = peak_db * np.exp(-0.5 * (np.log2(freq_axis / float(peak_hz)) / float(width_oct)) ** 2)
    return _auto_residual_peak_metrics_from_stats(
        {
            "freq_axis": freq_axis,
            "measured_mags": np.zeros_like(freq_axis),
            "target_mags": np.zeros_like(freq_axis),
            "predicted_filter_mags": err,
            "realized_filter_mags": err,
            "confidence_mask": np.ones_like(freq_axis),
        },
        lo_hz=20.0,
        hi_hz=250.0,
    )


def test_residual_peak_metrics_detect_40hz_extreme_peak_and_gate_metric():
    metrics = _synthetic_residual_peak_metrics(peak_hz=40.0, peak_db=8.0)

    assert int(metrics["residual_peak_count"]) >= 1
    assert float(metrics["worst_residual_peak_hz"]) == pytest.approx(40.0, rel=0.06)
    assert float(metrics["worst_residual_peak_raw_db"]) > 6.0
    assert _auto_phase2_hard_gate_pool(
        [
            {"id": "bad", "metrics": {"rank_score": 99.0, "worst_residual_peak_raw_db": 8.0}},
            {"id": "ok1", "metrics": {"rank_score": 90.0, "worst_residual_peak_raw_db": 2.0}},
            {"id": "ok2", "metrics": {"rank_score": 89.0, "worst_residual_peak_raw_db": 2.1}},
            {"id": "ok3", "metrics": {"rank_score": 88.0, "worst_residual_peak_raw_db": 2.2}},
            {"id": "ok4", "metrics": {"rank_score": 87.0, "worst_residual_peak_raw_db": 2.3}},
            {"id": "ok5", "metrics": {"rank_score": 86.0, "worst_residual_peak_raw_db": 2.4}},
        ],
        min_keep=3,
        abs_max_peak_db=6.0,
    )[0][0]["id"] != "bad"


def test_residual_peak_metrics_detect_70hz_boundary_peak_and_penalty():
    metrics = _synthetic_residual_peak_metrics(peak_hz=70.0, peak_db=6.0)

    assert float(metrics["worst_residual_peak_hz"]) == pytest.approx(70.0, rel=0.06)
    assert float(metrics["worst_residual_peak_raw_db"]) >= 5.7
    scored = _auto_score_result(
        SimpleNamespace(
            l_st={
                "freq_axis": np.geomspace(20.0, 250.0, 220),
                "measured_mags": np.zeros(220),
                "target_mags": np.zeros(220),
                "predicted_filter_mags": 6.0
                * np.exp(-0.5 * (np.log2(np.geomspace(20.0, 250.0, 220) / 70.0) / 0.13) ** 2),
                "confidence_mask": np.ones(220),
            },
            r_st={},
            metrics={},
        ),
        base_data={"mag_c_min": 20.0, "mag_c_max": 250.0},
    )
    assert float(scored["residual_peak_penalty"]) > 0.0
    assert float(scored["worst_residual_peak_raw_db"]) >= 5.7


def test_residual_peak_penalty_cap_is_configurable():
    freq_axis = np.geomspace(20.0, 250.0, 220)
    st = {
        "freq_axis": freq_axis,
        "measured_mags": np.zeros(220),
        "target_mags": np.zeros(220),
        "predicted_filter_mags": 20.0
        * np.exp(-0.5 * (np.log2(freq_axis / 70.0) / 0.13) ** 2),
        "confidence_mask": np.ones(220),
    }

    capped = _auto_score_result(
        SimpleNamespace(l_st=st, r_st={}, metrics={}),
        base_data={
            "mag_c_min": 20.0,
            "mag_c_max": 250.0,
            "auto_mode_residual_peak_penalty_cap": 1.0,
        },
    )
    uncapped = _auto_score_result(
        SimpleNamespace(l_st=st, r_st={}, metrics={}),
        base_data={
            "mag_c_min": 20.0,
            "mag_c_max": 250.0,
            "auto_mode_residual_peak_penalty_cap": 20.0,
        },
    )

    assert float(capped["residual_peak_penalty"]) == pytest.approx(1.0)
    assert float(capped["residual_peak_penalty_cap"]) == pytest.approx(1.0)
    assert float(uncapped["residual_peak_penalty"]) > float(capped["residual_peak_penalty"])


def test_residual_peak_metrics_keep_110hz_narrow_modal_peak_visible():
    metrics = _synthetic_residual_peak_metrics(peak_hz=110.0, peak_db=9.0, width_oct=0.025)

    assert int(metrics["residual_peak_count"]) == 0
    assert float(metrics["worst_residual_peak_raw_db"]) == pytest.approx(0.0, abs=1e-9)
    assert float(metrics["residual_peak_severity"]) == pytest.approx(0.0, abs=1e-9)


def test_broad_residual_peak_scorer_penalizes_wide_90hz_peak():
    freq_axis = np.geomspace(20.0, 250.0, 320)
    err = 8.0 * np.exp(-0.5 * (np.log2(freq_axis / 90.0) / 0.11) ** 2)

    metrics = compute_broad_residual_peak_metrics(
        {
            "freq_axis": freq_axis,
            "measured_mags": err,
            "target_mags": np.zeros_like(freq_axis),
            "realized_filter_mags": np.zeros_like(freq_axis),
            "confidence_mask": np.ones_like(freq_axis),
        },
        lo_hz=20.0,
        hi_hz=250.0,
        threshold_db=4.0,
        hard_gate_db=6.0,
    )

    assert int(metrics["residual_peak_count"]) >= 1
    assert float(metrics["worst_residual_peak_hz"]) == pytest.approx(90.0, rel=0.08)
    assert float(metrics["worst_residual_peak_raw_db"]) > 6.0
    assert float(metrics["worst_residual_peak_width_oct"]) >= 1.0 / 8.0
    assert float(metrics["residual_peak_area_db_oct"]) > 0.0


def test_narrow_spike_is_ignored_by_broad_peak_scorer():
    metrics = _synthetic_residual_peak_metrics(peak_hz=90.0, peak_db=8.0, width_oct=0.018)

    assert int(metrics["residual_peak_count"]) == 0
    assert float(metrics["worst_residual_peak_raw_db"]) == pytest.approx(0.0, abs=1e-9)


def test_smooth_cut_candidate_beats_broad_peak_left_behind():
    freq_axis = np.geomspace(20.0, 250.0, 320)
    peak = 8.0 * np.exp(-0.5 * (np.log2(freq_axis / 90.0) / 0.11) ** 2)
    left_peak = {
        "freq_axis": freq_axis,
        "measured_mags": peak,
        "target_mags": np.zeros_like(freq_axis),
        "realized_filter_mags": np.zeros_like(freq_axis),
        "confidence_mask": np.ones_like(freq_axis),
    }
    smooth_cut = dict(left_peak)
    smooth_cut["realized_filter_mags"] = -7.0 * np.exp(-0.5 * (np.log2(freq_axis / 90.0) / 0.14) ** 2)

    bad = _auto_score_result(SimpleNamespace(l_st=left_peak, r_st={}, metrics={}), base_data={"mag_c_min": 20.0, "mag_c_max": 250.0})
    good = _auto_score_result(SimpleNamespace(l_st=smooth_cut, r_st={}, metrics={}), base_data={"mag_c_min": 20.0, "mag_c_max": 250.0})

    assert float(bad["residual_peak_penalty"]) > float(good["residual_peak_penalty"])
    assert float(good["rank_score"]) > float(bad["rank_score"])


def test_deep_narrow_dip_boost_has_dip_fill_risk():
    freq_axis = np.geomspace(20.0, 250.0, 320)
    dip = -12.0 * np.exp(-0.5 * (np.log2(freq_axis / 95.0) / 0.025) ** 2)
    boost = 10.0 * np.exp(-0.5 * (np.log2(freq_axis / 95.0) / 0.025) ** 2)

    metrics = _auto_dip_fill_risk_metrics_from_stats(
        {
            "freq_axis": freq_axis,
            "measured_mags": dip,
            "target_mags": np.zeros_like(freq_axis),
            "predicted_filter_mags": boost,
        }
    )

    assert float(metrics["dip_fill_risk_penalty"]) > 0.0
    assert int(metrics["dip_fill_deep_narrow_count"]) >= 1


def test_rew_style_correction_curve_has_high_sharpness_penalty():
    freq_axis = np.geomspace(20.0, 250.0, 420)
    x = np.log2(freq_axis / 20.0)
    corr = 4.0 * np.sin(32.0 * x) + 2.5 * np.sin(55.0 * x)

    metrics = _auto_correction_sharpness_metrics_from_stats(
        {"freq_axis": freq_axis, "predicted_filter_mags": corr}
    )

    assert float(metrics["correction_sharpness_penalty"]) > 1.0
    assert int(metrics["direction_change_count"]) > 3


def test_smooth_modal_cut_has_low_sharpness_penalty():
    freq_axis = np.geomspace(20.0, 250.0, 420)
    corr = -6.0 * np.exp(-0.5 * (np.log2(freq_axis / 82.0) / 0.18) ** 2)

    metrics = _auto_correction_sharpness_metrics_from_stats(
        {"freq_axis": freq_axis, "predicted_filter_mags": corr}
    )

    assert float(metrics["correction_sharpness_penalty"]) < 1.0
    assert int(metrics["narrow_notch_count"]) == 0


def test_stereo_narrow_divergence_increases_channel_overfit_penalty():
    freq_axis = np.geomspace(20.0, 250.0, 420)
    left = -7.0 * np.exp(-0.5 * (np.log2(freq_axis / 120.0) / 0.025) ** 2)
    right = np.zeros_like(freq_axis)

    metrics = _auto_channel_overfit_metrics_from_stats(
        {"freq_axis": freq_axis, "predicted_filter_mags": left},
        {"freq_axis": freq_axis, "predicted_filter_mags": right},
    )

    assert int(metrics["narrow_lr_divergence_count"]) >= 1
    assert float(metrics["channel_overfit_penalty"]) > 0.0


def test_residual_peak_metrics_ignore_broad_harmless_tilt():
    freq_axis = np.geomspace(20.0, 250.0, 220)
    err = 0.80 * np.log2(freq_axis / 20.0)

    metrics = _auto_residual_peak_metrics_from_stats(
        {
            "freq_axis": freq_axis,
            "measured_mags": err,
            "target_mags": np.zeros_like(freq_axis),
            "predicted_filter_mags": np.zeros_like(freq_axis),
            "confidence_mask": np.ones_like(freq_axis),
        },
        lo_hz=20.0,
        hi_hz=250.0,
    )

    assert int(metrics["residual_peak_count"]) == 0
    assert float(metrics["worst_residual_peak_raw_db"]) == pytest.approx(0.0, abs=1e-9)
    assert float(metrics["residual_peak_severity"]) == pytest.approx(0.0, abs=1e-9)


def test_target_tracking_penalty_catches_broad_100_500_error():
    freq_axis = np.geomspace(20.0, 500.0, 96)
    err_bad = np.where(freq_axis >= 100.0, 5.0, 0.25)
    st_bad = {
        "freq_axis": freq_axis,
        "measured_mags": err_bad,
        "target_mags": np.zeros_like(freq_axis),
        "filter_mags": np.zeros_like(freq_axis),
        "confidence_mask": np.ones_like(freq_axis),
    }
    st_good = {
        "freq_axis": freq_axis,
        "measured_mags": np.zeros_like(freq_axis),
        "target_mags": np.zeros_like(freq_axis),
        "filter_mags": np.zeros_like(freq_axis),
        "confidence_mask": np.ones_like(freq_axis),
    }

    metrics = _auto_target_tracking_metrics_from_stats(st_bad)
    penalty = _auto_target_tracking_penalty(metrics)

    assert float(metrics["target_tracking_rms_100_500_db"]) > 4.0
    assert float(metrics["target_tracking_max_100_500_db"]) == pytest.approx(5.0, abs=1e-9)
    assert float(penalty) > 8.0

    result_metrics = _auto_score_result(
        SimpleNamespace(l_st=st_bad, r_st=st_good, metrics={}),
        base_data={"mag_c_max": 101.6},
    )

    assert float(result_metrics["target_tracking_rms_100_500_db"]) > 4.0
    assert float(result_metrics["target_tracking_penalty"]) > 8.0
    assert float(result_metrics["rank_score_components"]["target_tracking_penalty"]) > 8.0


def test_bass_boost_metric_ignores_boost_below_low_bass_guard():
    freq_axis = np.geomspace(10.0, 200.0, 80)
    filter_mags = np.zeros_like(freq_axis)
    filter_mags[freq_axis < 30.0] = 8.0
    filter_mags[(freq_axis >= 40.0) & (freq_axis <= 120.0)] = 2.0

    metrics = _auto_bass_boost_metrics_from_stats(
        {
            "freq_axis": freq_axis,
            "filter_mags": filter_mags,
            "low_bass_cut_hz": 30.0,
            "exc_prot": False,
        }
    )

    assert float(metrics["bass_boost_20_200_db"]) == pytest.approx(2.0, abs=0.05)
    assert float(metrics["bass_boost_peak_20_200_db"]) == pytest.approx(2.0, abs=0.05)


def test_bass_boost_metric_counts_sparse_positive_boost_above_guard():
    freq_axis = np.geomspace(20.0, 200.0, 80)
    filter_mags = np.zeros_like(freq_axis)
    filter_mags[(freq_axis >= 95.0) & (freq_axis <= 130.0)] = 1.8

    metrics = _auto_bass_boost_metrics_from_stats(
        {
            "freq_axis": freq_axis,
            "filter_mags": filter_mags,
            "low_bass_cut_hz": 36.0,
            "exc_prot": True,
            "exc_freq": 23.8,
        }
    )

    assert float(metrics["bass_boost_20_200_db"]) == pytest.approx(1.8, abs=0.05)
    assert float(metrics["bass_boost_peak_20_200_db"]) == pytest.approx(1.8, abs=0.05)


def test_phase2_hard_gate_drops_high_residual_peak_bad_actors():
    pool = []
    for idx in range(10):
        peak = 6.2 if idx in (0, 1) else 1.0 + idx * 0.03
        pool.append(
            {
                "id": idx,
                "metrics": {
                    "rank_score": 100.0 - idx,
                    "events_severity": 0.1,
                    "focus_ripple_db": 0.05,
                    "worst_residual_peak_db": peak,
                },
            }
        )

    kept, _ev_thr, _rp_thr, pk_thr = _auto_phase2_hard_gate_pool(
        pool,
        min_keep=4,
        keep_event_fraction=0.70,
        keep_ripple_fraction=0.75,
        keep_peak_fraction=0.70,
        abs_max_peak_db=5.5,
        fallback_to_rank=True,
    )

    kept_ids = {int(item["id"]) for item in kept}
    assert 0 not in kept_ids
    assert 1 not in kept_ids
    assert len(kept) >= 4
    assert float(pk_thr) < 5.5


def test_mode_detection_includes_sub_bass_resonance_with_moderate_gd():
    result = SimpleNamespace(
        l_st={
            "reflections": [
                {"type": "resonance", "freq": 28.0, "gd_error": 140.0},
            ]
        },
        r_st={},
    )

    top_modes = _auto_get_top_modes_hz(result, top_n=1)

    assert top_modes == pytest.approx([28.0], abs=1e-9)
    assert _auto_get_worst_mode_hz(result) == pytest.approx(28.0, abs=1e-9)


def test_mode_band_is_not_clamped_by_bass_first_mode_limit():
    band = _auto_mode_band(
        240.0,
        base_data={
            "bass_first_ai": True,
            "bass_first_mode_max_hz": 180.0,
        },
    )

    assert band is not None
    assert float(band[0]) < 200.0
    assert float(band[1]) > 220.0


def test_mag_c_min_winner_polish_can_improve_upward():
    def fake_materialize_preset_result(
        preset,
        *,
        include_response_arrays,
        summarize,
        base_data_override,
    ):
        mag_c_min = float(dict(preset or {}).get("mag_c_min", 25.0))
        rank_score = 80.0 - abs(mag_c_min - 26.0)
        metrics = {
            "rank_score": float(rank_score),
            "avg_score": 90.0,
        }
        return object(), metrics, dict(preset or {})

    def fake_cache_ready_preset(preset, *, best_metrics=None):
        _ = best_metrics
        return dict(preset or {})

    def fake_auto_is_better_refine(new_metrics, best_metrics, goal, *, return_reason=False):
        _ = goal
        better = float(dict(new_metrics or {}).get("rank_score", 0.0)) > float(
            dict(best_metrics or {}).get("rank_score", 0.0)
        )
        return (better, "rank") if return_reason else better

    best_preset, best_metrics, improved, meta = apply_mag_c_min_winner_polish(
        best_preset={"mag_c_min": 25.0},
        best_metrics={"rank_score": 79.0, "avg_score": 90.0},
        base_data_ref={"mag_c_min": 25.0},
        phase_label="test",
        goal="balanced",
        enabled=True,
        step_hz=1.0,
        max_down_hz=2.0,
        max_up_hz=4.0,
        status_cb=None,
        materialize_preset_result=fake_materialize_preset_result,
        cache_ready_preset=fake_cache_ready_preset,
        auto_is_better_refine=fake_auto_is_better_refine,
    )

    assert bool(improved) is True
    assert float(best_preset.get("mag_c_min", 0.0)) == pytest.approx(26.0, abs=1e-9)
    assert float(best_metrics.get("rank_score", 0.0)) == pytest.approx(80.0, abs=1e-9)
    assert 26.0 in [float(v) for v in list(meta.get("tested_mag_c_min_hz", []) or [])]


def test_mag_c_min_winner_polish_prefers_auto_mag_c_min_seed_center():
    def fake_materialize_preset_result(
        preset,
        *,
        include_response_arrays,
        summarize,
        base_data_override,
    ):
        _ = include_response_arrays, summarize, base_data_override
        mag_c_min = float(dict(preset or {}).get("mag_c_min", 25.0))
        rank_score = 80.0 - abs(mag_c_min - 20.0)
        metrics = {
            "rank_score": float(rank_score),
            "avg_score": 90.0,
        }
        return object(), metrics, dict(preset or {})

    def fake_cache_ready_preset(preset, *, best_metrics=None):
        _ = best_metrics
        return dict(preset or {})

    def fake_auto_is_better_refine(new_metrics, best_metrics, goal, *, return_reason=False):
        _ = goal
        better = float(dict(new_metrics or {}).get("rank_score", 0.0)) > float(
            dict(best_metrics or {}).get("rank_score", 0.0)
        )
        return (better, "rank") if return_reason else better

    best_preset, best_metrics, improved, meta = apply_mag_c_min_winner_polish(
        best_preset={"mag_c_min": 30.0},
        best_metrics={"rank_score": 79.0, "avg_score": 90.0},
        base_data_ref={"mag_c_min": 30.0, "_auto_mag_c_min_hz": 18.0},
        phase_label="test",
        goal="balanced",
        enabled=True,
        step_hz=2.0,
        max_down_hz=4.0,
        max_up_hz=4.0,
        status_cb=None,
        materialize_preset_result=fake_materialize_preset_result,
        cache_ready_preset=fake_cache_ready_preset,
        auto_is_better_refine=fake_auto_is_better_refine,
    )

    assert bool(improved) is True
    assert float(meta.get("start_mag_c_min_hz", 0.0)) == pytest.approx(18.0, abs=1e-9)
    assert [float(v) for v in list(meta.get("tested_mag_c_min_hz", []) or [])] == [16.0, 20.0, 22.0]
    assert 20.0 in [float(v) for v in list(meta.get("accepted_mag_c_min_hz", []) or [])]
    assert float(best_preset.get("mag_c_min", 0.0)) == pytest.approx(20.0, abs=1e-9)
    assert float(best_metrics.get("rank_score", 0.0)) == pytest.approx(80.0, abs=1e-9)


def test_mag_c_min_winner_polish_reuses_matching_candidate_metrics():
    calls = {"materialize": 0}

    def fake_materialize_preset_result(
        preset,
        *,
        include_response_arrays,
        summarize,
        base_data_override,
    ):
        _ = preset, include_response_arrays, summarize, base_data_override
        calls["materialize"] += 1
        raise AssertionError("materialize_preset_result should not run for reused exact candidate")

    def fake_cache_ready_preset(preset, *, best_metrics=None):
        _ = best_metrics
        return dict(preset or {})

    def fake_auto_is_better_refine(new_metrics, best_metrics, goal, *, return_reason=False):
        _ = goal
        better = float(dict(new_metrics or {}).get("rank_score", 0.0)) > float(
            dict(best_metrics or {}).get("rank_score", 0.0)
        )
        return (better, "rank") if return_reason else better

    best_preset, best_metrics, improved, meta = apply_mag_c_min_winner_polish(
        best_preset={"mag_c_min": 25.0, "_auto_exc_freq_hz": 31.5},
        best_metrics={
            "rank_score": 79.0,
            "avg_score": 90.0,
            "target_tracking_rms_20_200_db": 2.0,
            "target_tracking_rms_100_500_db": 1.5,
        },
        base_data_ref={"mag_c_min": 25.0},
        phase_label="test",
        goal="balanced",
        enabled=True,
        step_hz=1.0,
        max_down_hz=0.0,
        max_up_hz=1.0,
        status_cb=None,
        materialize_preset_result=fake_materialize_preset_result,
        cache_ready_preset=fake_cache_ready_preset,
        auto_is_better_refine=fake_auto_is_better_refine,
        candidate_items=[
            {
                "preset": {"mag_c_min": 26.0, "_auto_exc_freq_hz": 31.5},
                "metrics": {
                    "rank_score": 80.0,
                    "avg_score": 90.0,
                    "target_tracking_rms_20_200_db": 2.0,
                    "target_tracking_rms_100_500_db": 1.5,
                },
                "phase": "phase 2/2 local center#1",
            }
        ],
    )

    assert bool(improved) is True
    assert float(best_preset.get("mag_c_min", 0.0)) == pytest.approx(26.0, abs=1e-9)
    assert float(best_metrics.get("rank_score", 0.0)) == pytest.approx(80.0, abs=1e-9)
    assert int(calls["materialize"]) == 0
    assert [float(v) for v in list(meta.get("tested_mag_c_min_hz", []) or [])] == [26.0]


def test_mag_c_min_winner_polish_rejects_score_only_gain_when_tracking_worsens():
    materialize_calls = []

    def _metrics_for(mag_c_min: float) -> dict:
        if mag_c_min >= 40.0:
            return {
                "rank_score": 49.38,
                "rank_score_refine": 49.38,
                "avg_score": 74.1,
                "target_tracking_rms_20_200_db": 4.3,
                "target_tracking_rms_100_500_db": 3.5,
            }
        return {
            "rank_score": 49.34,
            "rank_score_refine": 49.34,
            "avg_score": 74.0,
            "target_tracking_rms_20_200_db": 3.2,
            "target_tracking_rms_100_500_db": 2.8,
        }

    def fake_materialize_preset_result(
        preset,
        *,
        include_response_arrays,
        summarize,
        base_data_override,
    ):
        _ = summarize, base_data_override
        materialize_calls.append(bool(include_response_arrays))
        mag_c_min = float(dict(preset or {}).get("mag_c_min", 18.7))
        return object(), _metrics_for(mag_c_min), dict(preset or {})

    def fake_cache_ready_preset(preset, *, best_metrics=None):
        _ = best_metrics
        return dict(preset or {})

    best_preset, best_metrics, improved, meta = apply_mag_c_min_winner_polish(
        best_preset={"mag_c_min": 18.7},
        best_metrics={"rank_score": 49.34, "rank_score_refine": 49.34, "avg_score": 74.0},
        base_data_ref={"mag_c_min": 18.7},
        phase_label="test",
        goal="balanced",
        enabled=True,
        step_hz=24.0,
        max_down_hz=0.0,
        max_up_hz=24.0,
        status_cb=None,
        materialize_preset_result=fake_materialize_preset_result,
        cache_ready_preset=fake_cache_ready_preset,
        auto_is_better_refine=_auto_is_better_refine,
    )

    assert bool(improved) is False
    assert float(best_preset.get("mag_c_min", 0.0)) == pytest.approx(18.7, abs=1e-9)
    assert float(best_metrics.get("rank_score", 0.0)) == pytest.approx(49.34, abs=1e-9)
    assert bool(meta.get("applied", True)) is False
    assert materialize_calls and all(bool(v) is True for v in materialize_calls)


def test_low_bass_cut_winner_polish_can_improve_upward():
    def fake_materialize_preset_result(
        preset,
        *,
        include_response_arrays,
        summarize,
        base_data_override,
    ):
        _ = include_response_arrays, summarize, base_data_override
        low_bass_cut_hz = float(dict(preset or {}).get("low_bass_cut_hz", 34.0))
        rank_score = 80.0 - abs(low_bass_cut_hz - 36.0)
        metrics = {
            "rank_score": float(rank_score),
            "avg_score": 90.0,
        }
        return object(), metrics, dict(preset or {})

    def fake_cache_ready_preset(preset, *, best_metrics=None):
        _ = best_metrics
        return dict(preset or {})

    def fake_auto_is_better_refine(new_metrics, best_metrics, goal, *, return_reason=False):
        _ = goal
        better = float(dict(new_metrics or {}).get("rank_score", 0.0)) > float(
            dict(best_metrics or {}).get("rank_score", 0.0)
        )
        return (better, "rank") if return_reason else better

    best_preset, best_metrics, improved, meta = apply_low_bass_cut_winner_polish(
        best_preset={"mag_c_min": 25.0, "low_bass_cut_hz": 34.0},
        best_metrics={"rank_score": 78.0, "avg_score": 90.0},
        base_data_ref={"mag_c_min": 25.0, "low_bass_cut_hz": 34.0, "low_bass_cut_enable": True},
        phase_label="test",
        goal="balanced",
        enabled=True,
        step_hz=2.0,
        max_delta_hz=4.0,
        status_cb=None,
        materialize_preset_result=fake_materialize_preset_result,
        cache_ready_preset=fake_cache_ready_preset,
        auto_is_better_refine=fake_auto_is_better_refine,
    )

    assert bool(improved) is True
    assert float(best_preset.get("low_bass_cut_hz", 0.0)) == pytest.approx(36.0, abs=1e-9)
    assert float(best_metrics.get("rank_score", 0.0)) == pytest.approx(80.0, abs=1e-9)
    assert 36.0 in [float(v) for v in list(meta.get("tested_low_bass_cut_hz", []) or [])]


def test_low_bass_cut_winner_polish_can_improve_downward():
    def fake_materialize_preset_result(
        preset,
        *,
        include_response_arrays,
        summarize,
        base_data_override,
    ):
        _ = include_response_arrays, summarize, base_data_override
        low_bass_cut_hz = float(dict(preset or {}).get("low_bass_cut_hz", 36.0))
        rank_score = 80.0 - abs(low_bass_cut_hz - 32.0)
        metrics = {
            "rank_score": float(rank_score),
            "avg_score": 90.0,
        }
        return object(), metrics, dict(preset or {})

    def fake_cache_ready_preset(preset, *, best_metrics=None):
        _ = best_metrics
        return dict(preset or {})

    def fake_auto_is_better_refine(new_metrics, best_metrics, goal, *, return_reason=False):
        _ = goal
        better = float(dict(new_metrics or {}).get("rank_score", 0.0)) > float(
            dict(best_metrics or {}).get("rank_score", 0.0)
        )
        return (better, "rank") if return_reason else better

    best_preset, best_metrics, improved, meta = apply_low_bass_cut_winner_polish(
        best_preset={"mag_c_min": 25.0, "low_bass_cut_hz": 36.0},
        best_metrics={"rank_score": 76.0, "avg_score": 90.0},
        base_data_ref={"mag_c_min": 25.0, "low_bass_cut_hz": 36.0, "low_bass_cut_enable": True},
        phase_label="test",
        goal="balanced",
        enabled=True,
        step_hz=2.0,
        max_delta_hz=6.0,
        status_cb=None,
        materialize_preset_result=fake_materialize_preset_result,
        cache_ready_preset=fake_cache_ready_preset,
        auto_is_better_refine=fake_auto_is_better_refine,
    )

    assert bool(improved) is True
    assert float(best_preset.get("low_bass_cut_hz", 0.0)) == pytest.approx(32.0, abs=1e-9)
    assert float(best_metrics.get("rank_score", 0.0)) == pytest.approx(80.0, abs=1e-9)
    assert 32.0 in [float(v) for v in list(meta.get("tested_low_bass_cut_hz", []) or [])]


def test_low_bass_cut_winner_polish_can_drop_below_mag_c_min():
    def fake_materialize_preset_result(
        preset,
        *,
        include_response_arrays,
        summarize,
        base_data_override,
    ):
        _ = include_response_arrays, summarize, base_data_override
        low_bass_cut_hz = float(dict(preset or {}).get("low_bass_cut_hz", 40.0))
        rank_score = 80.0 - abs(low_bass_cut_hz - 34.0)
        metrics = {
            "rank_score": float(rank_score),
            "avg_score": 90.0,
        }
        return object(), metrics, dict(preset or {})

    def fake_cache_ready_preset(preset, *, best_metrics=None):
        _ = best_metrics
        return dict(preset or {})

    def fake_auto_is_better_refine(new_metrics, best_metrics, goal, *, return_reason=False):
        _ = goal
        better = float(dict(new_metrics or {}).get("rank_score", 0.0)) > float(
            dict(best_metrics or {}).get("rank_score", 0.0)
        )
        return (better, "rank") if return_reason else better

    best_preset, best_metrics, improved, meta = apply_low_bass_cut_winner_polish(
        best_preset={"mag_c_min": 45.0, "low_bass_cut_hz": 40.0},
        best_metrics={"rank_score": 74.0, "avg_score": 90.0},
        base_data_ref={"mag_c_min": 45.0, "low_bass_cut_hz": 40.0, "low_bass_cut_enable": True},
        phase_label="test",
        goal="balanced",
        enabled=True,
        step_hz=2.0,
        max_delta_hz=8.0,
        status_cb=None,
        materialize_preset_result=fake_materialize_preset_result,
        cache_ready_preset=fake_cache_ready_preset,
        auto_is_better_refine=fake_auto_is_better_refine,
    )

    assert bool(improved) is True
    assert float(best_preset.get("low_bass_cut_hz", 0.0)) == pytest.approx(34.0, abs=1e-9)
    assert float(best_preset.get("low_bass_cut_hz", 0.0)) < float(best_preset.get("mag_c_min", 0.0))
    assert float(best_metrics.get("rank_score", 0.0)) == pytest.approx(80.0, abs=1e-9)
    assert 34.0 in [float(v) for v in list(meta.get("tested_low_bass_cut_hz", []) or [])]


def test_low_bass_cut_winner_polish_reuses_matching_candidate_metrics():
    materialized_low_cuts = []

    def fake_materialize_preset_result(
        preset,
        *,
        include_response_arrays,
        summarize,
        base_data_override,
    ):
        _ = include_response_arrays, summarize, base_data_override
        low_bass_cut_hz = float(dict(preset or {}).get("low_bass_cut_hz", 0.0))
        materialized_low_cuts.append(float(low_bass_cut_hz))
        return object(), {"rank_score": 70.0, "avg_score": 90.0}, dict(preset or {})

    def fake_cache_ready_preset(preset, *, best_metrics=None):
        _ = best_metrics
        return dict(preset or {})

    def fake_auto_is_better_refine(new_metrics, best_metrics, goal, *, return_reason=False):
        _ = goal
        better = float(dict(new_metrics or {}).get("rank_score", 0.0)) > float(
            dict(best_metrics or {}).get("rank_score", 0.0)
        )
        return (better, "rank") if return_reason else better

    best_preset, best_metrics, improved, meta = apply_low_bass_cut_winner_polish(
        best_preset={"mag_c_min": 33.0, "low_bass_cut_hz": 34.0, "_auto_exc_freq_hz": 31.5},
        best_metrics={"rank_score": 78.0, "avg_score": 90.0},
        base_data_ref={"mag_c_min": 33.0, "low_bass_cut_hz": 34.0, "low_bass_cut_enable": True},
        phase_label="test",
        goal="balanced",
        enabled=True,
        step_hz=2.0,
        max_delta_hz=2.0,
        status_cb=None,
        materialize_preset_result=fake_materialize_preset_result,
        cache_ready_preset=fake_cache_ready_preset,
        auto_is_better_refine=fake_auto_is_better_refine,
        candidate_items=[
            {
                "preset": {"mag_c_min": 33.0, "low_bass_cut_hz": 36.0, "_auto_exc_freq_hz": 31.5},
                "metrics": {"rank_score": 80.0, "avg_score": 90.0},
                "phase": "phase 2/2 local center#1",
            }
        ],
    )

    assert bool(improved) is True
    assert float(best_preset.get("low_bass_cut_hz", 0.0)) == pytest.approx(36.0, abs=1e-9)
    assert float(best_metrics.get("rank_score", 0.0)) == pytest.approx(80.0, abs=1e-9)
    assert materialized_low_cuts == [32.0]
    assert [float(v) for v in list(meta.get("tested_low_bass_cut_hz", []) or [])] == [32.0, 36.0]


def test_hpf_winner_polish_can_turn_hpf_off():
    def fake_materialize_preset_result(
        preset,
        *,
        include_response_arrays,
        summarize,
        base_data_override,
    ):
        _ = include_response_arrays, summarize, base_data_override
        override = dict(dict(preset or {}).get("_auto_hpf_runtime_override", {}) or {})
        enabled = bool(override.get("enabled", dict(preset or {}).get("hpf_enable", True)))
        metrics = {
            "rank_score": 83.0 if not enabled else 80.0,
            "avg_score": 90.0,
        }
        return object(), metrics, dict(preset or {})

    def fake_cache_ready_preset(preset, *, best_metrics=None):
        _ = best_metrics
        return dict(preset or {})

    def fake_auto_is_better_refine(new_metrics, best_metrics, goal, *, return_reason=False):
        _ = goal
        better = float(dict(new_metrics or {}).get("rank_score", 0.0)) > float(
            dict(best_metrics or {}).get("rank_score", 0.0)
        )
        return (better, "rank") if return_reason else better

    best_preset, best_metrics, improved, meta = apply_hpf_winner_polish(
        best_preset={"hpf_enable": True, "hpf_freq": 27.5, "hpf_slope": 24},
        best_metrics={"rank_score": 80.0, "avg_score": 90.0},
        base_data_ref={"hpf_freq": 27.5, "hpf_slope": 24},
        phase_label="test",
        goal="balanced",
        enabled=True,
        status_cb=None,
        materialize_preset_result=fake_materialize_preset_result,
        cache_ready_preset=fake_cache_ready_preset,
        auto_is_better_refine=fake_auto_is_better_refine,
    )

    assert bool(improved) is True
    assert bool(best_preset.get("hpf_enable", True)) is False
    assert bool(dict(best_preset.get("_auto_hpf_runtime_override", {}) or {}).get("enabled", True)) is False
    assert float(best_metrics.get("rank_score", 0.0)) == pytest.approx(83.0, abs=1e-9)
    assert bool(dict(meta.get("accepted_candidates", [{}])[0]).get("enabled", True)) is False


def test_hpf_winner_polish_can_prefer_nearby_freq():
    def fake_materialize_preset_result(
        preset,
        *,
        include_response_arrays,
        summarize,
        base_data_override,
    ):
        _ = include_response_arrays, summarize, base_data_override
        freq_hz = float(dict(preset or {}).get("hpf_freq", 27.5) or 27.5)
        metrics = {
            "rank_score": 84.0 if freq_hz == pytest.approx(24.5, abs=1e-9) else 80.0,
            "avg_score": 90.0,
        }
        return object(), metrics, dict(preset or {})

    def fake_cache_ready_preset(preset, *, best_metrics=None):
        _ = best_metrics
        return dict(preset or {})

    def fake_auto_is_better_refine(new_metrics, best_metrics, goal, *, return_reason=False):
        _ = goal
        better = float(dict(new_metrics or {}).get("rank_score", 0.0)) > float(
            dict(best_metrics or {}).get("rank_score", 0.0)
        )
        return (better, "rank") if return_reason else better

    best_preset, best_metrics, improved, meta = apply_hpf_winner_polish(
        best_preset={"hpf_enable": True, "hpf_freq": 27.5, "hpf_slope": 24},
        best_metrics={"rank_score": 80.0, "avg_score": 90.0},
        base_data_ref={"hpf_freq": 27.5, "hpf_slope": 24},
        phase_label="test",
        goal="balanced",
        enabled=True,
        status_cb=None,
        materialize_preset_result=fake_materialize_preset_result,
        cache_ready_preset=fake_cache_ready_preset,
        auto_is_better_refine=fake_auto_is_better_refine,
    )

    assert bool(improved) is True
    assert float(best_preset.get("hpf_freq", 0.0)) == pytest.approx(24.5, abs=1e-9)
    assert float(best_metrics.get("rank_score", 0.0)) == pytest.approx(84.0, abs=1e-9)
    assert "HPF 24.5 Hz/24 dB/oct" in [str(dict(item).get("label", "")) for item in list(meta.get("tested_candidates", []) or [])]


def test_hpf_winner_polish_can_prefer_looser_slope():
    def fake_materialize_preset_result(
        preset,
        *,
        include_response_arrays,
        summarize,
        base_data_override,
    ):
        _ = include_response_arrays, summarize, base_data_override
        slope = int(dict(preset or {}).get("hpf_slope", 24) or 24)
        metrics = {
            "rank_score": 84.0 if slope == 18 else 80.0,
            "avg_score": 90.0,
        }
        return object(), metrics, dict(preset or {})

    def fake_cache_ready_preset(preset, *, best_metrics=None):
        _ = best_metrics
        return dict(preset or {})

    def fake_auto_is_better_refine(new_metrics, best_metrics, goal, *, return_reason=False):
        _ = goal
        better = float(dict(new_metrics or {}).get("rank_score", 0.0)) > float(
            dict(best_metrics or {}).get("rank_score", 0.0)
        )
        return (better, "rank") if return_reason else better

    best_preset, best_metrics, improved, meta = apply_hpf_winner_polish(
        best_preset={"hpf_enable": True, "hpf_freq": 27.5, "hpf_slope": 24},
        best_metrics={"rank_score": 80.0, "avg_score": 90.0},
        base_data_ref={"hpf_freq": 27.5, "hpf_slope": 24},
        phase_label="test",
        goal="balanced",
        enabled=True,
        status_cb=None,
        materialize_preset_result=fake_materialize_preset_result,
        cache_ready_preset=fake_cache_ready_preset,
        auto_is_better_refine=fake_auto_is_better_refine,
    )

    assert bool(improved) is True
    assert int(best_preset.get("hpf_slope", 0)) == 18
    assert float(best_metrics.get("rank_score", 0.0)) == pytest.approx(84.0, abs=1e-9)
    assert int(meta.get("final_slope_db_oct", 0)) == 18


def test_hpf_winner_polish_can_prefer_steeper_slope():
    def fake_materialize_preset_result(
        preset,
        *,
        include_response_arrays,
        summarize,
        base_data_override,
    ):
        _ = include_response_arrays, summarize, base_data_override
        slope = int(dict(preset or {}).get("hpf_slope", 24) or 24)
        metrics = {
            "rank_score": 84.0 if slope == 30 else 80.0,
            "avg_score": 90.0,
        }
        return object(), metrics, dict(preset or {})

    def fake_cache_ready_preset(preset, *, best_metrics=None):
        _ = best_metrics
        return dict(preset or {})

    def fake_auto_is_better_refine(new_metrics, best_metrics, goal, *, return_reason=False):
        _ = goal
        better = float(dict(new_metrics or {}).get("rank_score", 0.0)) > float(
            dict(best_metrics or {}).get("rank_score", 0.0)
        )
        return (better, "rank") if return_reason else better

    best_preset, best_metrics, improved, meta = apply_hpf_winner_polish(
        best_preset={"hpf_enable": True, "hpf_freq": 27.5, "hpf_slope": 24},
        best_metrics={"rank_score": 80.0, "avg_score": 90.0},
        base_data_ref={"hpf_freq": 27.5, "hpf_slope": 24},
        phase_label="test",
        goal="balanced",
        enabled=True,
        status_cb=None,
        materialize_preset_result=fake_materialize_preset_result,
        cache_ready_preset=fake_cache_ready_preset,
        auto_is_better_refine=fake_auto_is_better_refine,
    )

    assert bool(improved) is True
    assert int(best_preset.get("hpf_slope", 0)) == 30
    assert float(best_metrics.get("rank_score", 0.0)) == pytest.approx(84.0, abs=1e-9)
    assert int(meta.get("final_slope_db_oct", 0)) == 30


def test_hpf_winner_polish_rejects_near_tie_rank_gain():
    def fake_materialize_preset_result(
        preset,
        *,
        include_response_arrays,
        summarize,
        base_data_override,
    ):
        _ = include_response_arrays, summarize, base_data_override
        override = dict(dict(preset or {}).get("_auto_hpf_runtime_override", {}) or {})
        enabled = bool(override.get("enabled", dict(preset or {}).get("hpf_enable", True)))
        metrics = {
            "rank_score": 80.005 if not enabled else 80.0,
            "avg_score": 90.0,
        }
        return object(), metrics, dict(preset or {})

    def fake_cache_ready_preset(preset, *, best_metrics=None):
        _ = best_metrics
        return dict(preset or {})

    def fake_auto_is_better_refine(new_metrics, best_metrics, goal, *, return_reason=False):
        _ = goal
        better = float(dict(new_metrics or {}).get("rank_score", 0.0)) > float(
            dict(best_metrics or {}).get("rank_score", 0.0)
        )
        return (better, "rank") if return_reason else better

    best_preset, best_metrics, improved, meta = apply_hpf_winner_polish(
        best_preset={"hpf_enable": True, "hpf_freq": 27.5, "hpf_slope": 24},
        best_metrics={"rank_score": 80.0, "avg_score": 90.0},
        base_data_ref={"hpf_freq": 27.5, "hpf_slope": 24},
        phase_label="test",
        goal="balanced",
        enabled=True,
        status_cb=None,
        materialize_preset_result=fake_materialize_preset_result,
        cache_ready_preset=fake_cache_ready_preset,
        auto_is_better_refine=fake_auto_is_better_refine,
    )

    assert bool(improved) is False
    assert bool(best_preset.get("hpf_enable", False)) is True
    assert float(best_metrics.get("rank_score", 0.0)) == pytest.approx(80.0, abs=1e-9)
    assert bool(meta.get("applied", True)) is False


def test_residual_peak_winner_polish_accepts_clear_peak_improvement_with_small_rank_drop():
    def fake_materialize_preset_result(
        preset,
        *,
        include_response_arrays,
        summarize,
        base_data_override,
    ):
        _ = include_response_arrays, summarize, base_data_override
        slope = float(dict(preset or {}).get("max_slope_cut_db_per_oct", 0.0) or 0.0)
        if abs(float(slope) - 6.0) <= 1e-9:
            metrics = {
                "rank_score": 89.90,
                "avg_score": 94.60,
                "max_net_boost_db": 3.0,
                "worst_residual_peak_db": 3.05,
                "top3_residual_peak_mean_db": 2.7,
            }
        else:
            metrics = {
                "rank_score": 89.0,
                "avg_score": 94.0,
                "max_net_boost_db": 3.0,
                "worst_residual_peak_db": 3.2,
                "top3_residual_peak_mean_db": 2.8,
            }
        return object(), metrics, dict(preset or {})

    def fake_cache_ready_preset(preset, *, best_metrics=None):
        _ = best_metrics
        return dict(preset or {})

    best_preset, best_metrics, improved, meta = apply_residual_peak_winner_polish(
        best_preset={"max_slope_cut_db_per_oct": 0.0},
        best_metrics={
            "rank_score": 90.0,
            "avg_score": 95.0,
            "max_net_boost_db": 3.0,
            "worst_residual_peak_db": 4.0,
            "top3_residual_peak_mean_db": 3.2,
        },
        base_data_ref={},
        phase_label="test",
        goal="balanced",
        enabled=True,
        max_variants=2,
        min_improvement_db=0.75,
        status_cb=None,
        materialize_preset_result=fake_materialize_preset_result,
        cache_ready_preset=fake_cache_ready_preset,
    )

    assert bool(improved) is True
    assert float(best_preset["max_slope_cut_db_per_oct"]) == pytest.approx(6.0)
    assert float(best_metrics["rank_score"]) == pytest.approx(89.90)
    assert float(best_metrics["worst_residual_peak_db"]) == pytest.approx(3.05)
    assert bool(meta["applied"]) is True
    assert float(meta["worst_peak_before_db"]) == pytest.approx(4.0)
    assert float(meta["worst_peak_after_db"]) == pytest.approx(3.05)


def test_materialize_score_only_cache_reuses_exact_preset():
    calls = {
        "build_config": 0,
        "run_pipeline": 0,
        "auto_score_result": 0,
        "summarize_run": 0,
    }

    def fake_build_config(*args, **kwargs):
        _ = args, kwargs
        calls["build_config"] += 1
        return SimpleNamespace()

    def fake_run_pipeline(cfg, measurements, *, include_response_arrays):
        _ = cfg, measurements, include_response_arrays
        calls["run_pipeline"] += 1
        return SimpleNamespace(metrics={}, l_st={}, r_st={})

    def fake_auto_score_result(result, *, auto_exc_freq_hz, base_data):
        _ = result, auto_exc_freq_hz, base_data
        calls["auto_score_result"] += 1
        return {"rank_score": 80.0, "avg_score": 90.0}

    def fake_summarize_run(result):
        _ = result
        calls["summarize_run"] += 1
        return "summary"

    ctx = AutoModeMaterializeContext(
        cfg=SimpleNamespace(exc_min_hz=20.0, exc_max_hz=80.0),
        cache_base_data={"phase_limit": 400.0, "mag_c_min": 25.0},
        measurements={},
        fs_v=44100,
        taps_v=65536,
        xos=[],
        hpf=None,
        hc_f=None,
        hc_m=None,
        pin_obj=None,
        filter_key="linear",
        max_safe_boost=6.0,
        goal="balanced",
        status_cb=None,
        exact_cached_metrics_getter=None,
        auto_score_result_fn=fake_auto_score_result,
        auto_optuna_jsonable_fn=lambda value: value,
        auto_rank_key_fn=lambda metrics: float(dict(metrics or {}).get("rank_score", 0.0)),
        auto_is_better_refine_fn=lambda *args, **kwargs: False,
        build_config_fn=fake_build_config,
        run_pipeline_fn=fake_run_pipeline,
        summarize_run_fn=fake_summarize_run,
        preset_transient_keys=(),
        residual_tiebreak_enabled=False,
        residual_top_k=3,
        residual_rank_eps=0.35,
    )
    _cache_ready_preset, materialize_preset_result, _preset_signature, _maybe_apply = build_materialize_helpers(ctx)
    _ = _cache_ready_preset, _preset_signature, _maybe_apply

    preset = {"mag_c_min": 25.0, "_auto_exc_freq_hz": 31.5}
    result_a, metrics_a, data_a = materialize_preset_result(
        preset,
        include_response_arrays=False,
        summarize=False,
    )
    result_b, metrics_b, data_b = materialize_preset_result(
        dict(preset),
        include_response_arrays=False,
        summarize=False,
    )

    assert result_a is not None
    assert result_b is None
    metrics_a_stable = {k: v for k, v in dict(metrics_a).items() if k != "cache_info"}
    metrics_b_stable = {k: v for k, v in dict(metrics_b).items() if k != "cache_info"}
    assert metrics_b_stable == metrics_a_stable
    assert metrics_b["cache_info"]["score_only_cache"] == "hit"
    assert data_b == data_a
    assert calls == {
        "build_config": 1,
        "run_pipeline": 1,
        "auto_score_result": 1,
        "summarize_run": 0,
    }


def test_materialize_score_only_cache_ignores_auto_metadata_keys():
    calls = {"run_pipeline": 0}

    def fake_build_config(*args, **kwargs):
        _ = args, kwargs
        return SimpleNamespace()

    def fake_run_pipeline(cfg, measurements, *, include_response_arrays):
        _ = cfg, measurements, include_response_arrays
        calls["run_pipeline"] += 1
        return SimpleNamespace(metrics={}, l_st={}, r_st={})

    ctx = AutoModeMaterializeContext(
        cfg=SimpleNamespace(exc_min_hz=20.0, exc_max_hz=80.0),
        cache_base_data={"phase_limit": 400.0, "mag_c_min": 25.0},
        measurements={},
        fs_v=44100,
        taps_v=65536,
        xos=[],
        hpf=None,
        hc_f=None,
        hc_m=None,
        pin_obj=None,
        filter_key="linear",
        max_safe_boost=6.0,
        goal="balanced",
        status_cb=None,
        exact_cached_metrics_getter=None,
        auto_score_result_fn=lambda *args, **kwargs: {"rank_score": 80.0},
        auto_optuna_jsonable_fn=lambda value: value,
        auto_rank_key_fn=lambda metrics: float(dict(metrics or {}).get("rank_score", 0.0)),
        auto_is_better_refine_fn=lambda *args, **kwargs: False,
        build_config_fn=fake_build_config,
        run_pipeline_fn=fake_run_pipeline,
        summarize_run_fn=lambda result: "summary",
        preset_transient_keys=(),
        residual_tiebreak_enabled=False,
        residual_top_k=3,
        residual_rank_eps=0.35,
    )
    _, materialize_preset_result, _, _ = build_materialize_helpers(ctx)

    preset = {"mag_c_min": 25.0, "_auto_exc_freq_hz": 31.5}
    materialize_preset_result(
        {**preset, "_auto_trial_index": 1, "selection_method": "first"},
        include_response_arrays=False,
        summarize=False,
    )
    materialize_preset_result(
        {**preset, "_auto_trial_index": 2, "selection_method": "second"},
        include_response_arrays=False,
        summarize=False,
    )

    assert calls["run_pipeline"] == 1


def test_materialize_score_only_cache_does_not_replace_full_materialization():
    calls = {
        "build_config": 0,
        "run_pipeline": 0,
        "auto_score_result": 0,
        "summarize_run": 0,
    }

    def fake_build_config(*args, **kwargs):
        _ = args, kwargs
        calls["build_config"] += 1
        return SimpleNamespace()

    def fake_run_pipeline(cfg, measurements, *, include_response_arrays):
        _ = cfg, measurements
        calls["run_pipeline"] += 1
        return SimpleNamespace(metrics={"include_response_arrays": bool(include_response_arrays)}, l_st={}, r_st={})

    def fake_auto_score_result(result, *, auto_exc_freq_hz, base_data):
        _ = result, auto_exc_freq_hz, base_data
        calls["auto_score_result"] += 1
        return {"rank_score": 80.0, "avg_score": 90.0}

    def fake_summarize_run(result):
        _ = result
        calls["summarize_run"] += 1
        return "summary"

    ctx = AutoModeMaterializeContext(
        cfg=SimpleNamespace(exc_min_hz=20.0, exc_max_hz=80.0),
        cache_base_data={"phase_limit": 400.0, "mag_c_min": 25.0},
        measurements={},
        fs_v=44100,
        taps_v=65536,
        xos=[],
        hpf=None,
        hc_f=None,
        hc_m=None,
        pin_obj=None,
        filter_key="linear",
        max_safe_boost=6.0,
        goal="balanced",
        status_cb=None,
        exact_cached_metrics_getter=None,
        auto_score_result_fn=fake_auto_score_result,
        auto_optuna_jsonable_fn=lambda value: value,
        auto_rank_key_fn=lambda metrics: float(dict(metrics or {}).get("rank_score", 0.0)),
        auto_is_better_refine_fn=lambda *args, **kwargs: False,
        build_config_fn=fake_build_config,
        run_pipeline_fn=fake_run_pipeline,
        summarize_run_fn=fake_summarize_run,
        preset_transient_keys=(),
        residual_tiebreak_enabled=False,
        residual_top_k=3,
        residual_rank_eps=0.35,
    )
    _cache_ready_preset, materialize_preset_result, _preset_signature, _maybe_apply = build_materialize_helpers(ctx)
    _ = _cache_ready_preset, _preset_signature, _maybe_apply

    preset = {"mag_c_min": 25.0, "_auto_exc_freq_hz": 31.5}
    materialize_preset_result(
        preset,
        include_response_arrays=False,
        summarize=False,
    )
    result_full, metrics_full, data_full = materialize_preset_result(
        dict(preset),
        include_response_arrays=True,
        summarize=True,
    )

    assert result_full.metrics.get("summary") == "summary"
    assert metrics_full == {"rank_score": 80.0, "avg_score": 90.0}
    assert float(data_full.get("mag_c_min", 0.0)) == pytest.approx(25.0, abs=1e-9)
    assert calls == {
        "build_config": 2,
        "run_pipeline": 2,
        "auto_score_result": 2,
        "summarize_run": 1,
    }
