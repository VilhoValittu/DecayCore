from decaycore.auto_mode.cache_signature import _auto_measurement_signature, _auto_signature
from decaycore.auto_mode.rank_score import build_rank_score_breakdown, compute_rank_score_components
from decaycore.auto_mode.scoring_metrics import _auto_dsp_quality_penalty, _auto_tdc_decay_tradeoff


def test_tdc_overreach_debug_does_not_crash_when_tdc_applied():
    penalty, _bass_prering, dbg = _auto_dsp_quality_penalty(
        {
            "tdc_applied": True,
            "tdc_peak_reduction_db": 1.5,
            "tdc_reduction_band_low_hz": 42.0,
            "tdc_reduction_band_high_hz": 62.0,
            "tdc_reduction_area_db_hz": 20.0,
            "tdc_events_used": 1,
        }
    )

    assert penalty >= 0.0
    assert dbg["tdc_overreach_penalty"] == 0.0
    assert dbg["tdc"]["tdc_applied"] is True


def test_moderate_tdc_metrics_do_not_add_overreach_penalty():
    penalty, _bass_prering, dbg = _auto_dsp_quality_penalty(
        {
            "tdc_applied": True,
            "tdc_peak_reduction_db": 2.0,
            "tdc_reduction_band_low_hz": 42.0,
            "tdc_reduction_band_high_hz": 62.0,
            "tdc_reduction_area_db_hz": 30.0,
            "tdc_events_used": 1,
            "tdc_guard_scale": 0.5,
            "tdc_guard_reason": "rt60_uncertain",
        }
    )

    assert penalty == 0.0
    assert dbg["tdc_overreach_penalty"] == 0.0
    assert "tdc_guard_scale" not in dbg["tdc"]


def test_phase_ir_risk_telemetry_is_ignored_by_dsp_quality_penalty():
    penalty, _bass_prering, dbg = _auto_dsp_quality_penalty(
        {
            "phase_ir_acoustic_risk_score": 0.75,
            "ir_realized_level_match_risk_score": 0.5,
        }
    )

    assert penalty == 0.0
    assert "phase_ir_acoustic_risk_score" not in dbg


def test_low_tdc_wins_when_decay_is_clean():
    clean_meta = {"enable_tdc": True, "tdc_strength": 25.0, "measured_rt60_bands_l": {40.0: 0.31, 80.0: 0.32}}
    low_pen, low_dbg = _auto_tdc_decay_tradeoff(clean_meta, {}, {})
    high_pen, high_dbg = _auto_tdc_decay_tradeoff({**clean_meta, "tdc_strength": 70.0}, {}, {})

    assert low_pen < high_pen
    assert low_dbg["tdc_decision"] == "reduced_by_optimizer"
    assert low_dbg["tdc_action_hint"] == "keep_tdc"
    assert high_dbg["tdc_decision"] == "possible_overdamping_risk"
    assert high_dbg["tdc_action_hint"] == "decrease_tdc"
    assert high_dbg["tdc_overdamping_penalty"] > high_dbg["tdc_weak_penalty"]


def test_medium_tdc_wins_when_modal_decay_exists():
    modal_meta = {"enable_tdc": True, "measured_rt60_bands_l": {42.0: 0.78, 80.0: 0.70, 250.0: 0.45}}
    low_pen, _ = _auto_tdc_decay_tradeoff({**modal_meta, "tdc_strength": 20.0}, {}, {})
    mid_pen, mid_dbg = _auto_tdc_decay_tradeoff({**modal_meta, "tdc_strength": 55.0}, {}, {})
    high_pen, _ = _auto_tdc_decay_tradeoff({**modal_meta, "tdc_strength": 75.0}, {}, {})

    assert mid_pen < low_pen
    assert mid_pen < high_pen
    assert mid_dbg["tdc_decision"] == "tdc_helped"
    assert mid_dbg["tdc_action_hint"] == "keep_tdc"
    assert mid_dbg["strongest_modal_decay_hz"] == 42.0


def test_high_tdc_is_penalized_when_bass_is_clean():
    clean_meta = {"enable_tdc": True, "tdc_strength": 75.0, "measured_rt60_bands_l": {50.0: 0.30, 100.0: 0.34}}
    penalty, dbg = _auto_tdc_decay_tradeoff(
        clean_meta,
        {"tdc_peak_reduction_db": 7.5, "tdc_reduction_area_db_hz": 900.0},
        {},
    )

    assert penalty > 1.0
    assert dbg["tdc_decision"] == "possible_overdamping_risk"
    assert dbg["tdc_action_hint"] == "decrease_tdc"
    assert dbg["tdc_extreme_overreach"] is False


def test_high_modal_decay_need_penalizes_too_low_tdc_strength():
    modal_meta = {"enable_tdc": True, "measured_rt60_bands_l": {40.0: 0.90, 80.0: 0.82, 250.0: 0.55}}
    weak_penalty, weak_dbg = _auto_tdc_decay_tradeoff({**modal_meta, "tdc_strength": 15.0}, {}, {})
    keep_penalty, keep_dbg = _auto_tdc_decay_tradeoff({**modal_meta, "tdc_strength": 60.0}, {}, {})

    assert weak_penalty > keep_penalty
    assert weak_dbg["tdc_weak_penalty"] > 0.0
    assert weak_dbg["tdc_action_hint"] == "increase_tdc"
    assert keep_dbg["tdc_action_hint"] == "keep_tdc"


def test_low_modal_decay_need_penalizes_too_high_tdc_strength():
    clean_meta = {"enable_tdc": True, "measured_rt60_bands_l": {40.0: 0.30, 80.0: 0.32, 250.0: 0.34}}
    keep_penalty, keep_dbg = _auto_tdc_decay_tradeoff({**clean_meta, "tdc_strength": 25.0}, {}, {})
    high_penalty, high_dbg = _auto_tdc_decay_tradeoff({**clean_meta, "tdc_strength": 80.0}, {}, {})

    assert high_penalty > keep_penalty
    assert high_dbg["tdc_overdamping_penalty"] > 0.0
    assert high_dbg["tdc_action_hint"] == "decrease_tdc"
    assert keep_dbg["tdc_action_hint"] == "keep_tdc"


def test_tdc_decay_penalty_cap_is_configurable():
    meta = {
        "enable_tdc": True,
        "tdc_strength": 100.0,
        "auto_mode_tdc_decay_penalty_cap": 1.25,
        "auto_mode_tdc_decay_penalty_weight": 4.0,
        "measured_rt60_bands_l": {40.0: 0.28, 80.0: 0.30},
    }
    penalty, dbg = _auto_tdc_decay_tradeoff(
        meta,
        {"tdc_peak_reduction_db": 14.0, "tdc_reduction_area_db_hz": 2000.0},
        {},
    )

    assert penalty == 1.25
    assert dbg["tdc_decay_penalty_total"] == 1.25


def test_extreme_tdc_overreach_is_strong_penalty_not_hard_gate():
    meta = {"enable_tdc": True, "tdc_strength": 80.0, "measured_rt60_bands_l": {40.0: 0.32, 80.0: 0.34}}
    penalty, dbg = _auto_tdc_decay_tradeoff(
        meta,
        {"tdc_peak_reduction_db": 12.0, "tdc_reduction_area_db_hz": 400.0},
        {},
    )

    assert penalty > 4.0
    assert dbg["tdc_extreme_overreach"] is True
    assert dbg["tdc_overreach_penalty"] > 4.0


def test_decay_penalty_affects_rank_score():
    clean = compute_rank_score_components(avg_score=90.0, decay_penalty=0.0)
    penalized = compute_rank_score_components(avg_score=90.0, decay_penalty=2.0)

    assert penalized["rank_score"] == clean["rank_score"] - 2.0
    assert penalized["decay_penalty"] == 2.0


def test_rank_score_penalties_reduce_and_bonuses_increase_score():
    base = compute_rank_score_components(avg_score=80.0)
    penalized = compute_rank_score_components(
        avg_score=80.0,
        boost_penalty=1.0,
        mode_penalty=2.0,
        residual_peak_penalty=3.0,
    )
    boosted = compute_rank_score_components(
        avg_score=80.0,
        phase_benefit_bonus=1.5,
        bass_preference_bonus=0.5,
        worst_channel_relief_bonus=1.0,
    )

    assert penalized["rank_score"] == base["rank_score"] - 6.0
    assert boosted["rank_score"] == base["rank_score"] + 3.0


def test_voice_clarity_penalty_affects_rank_score_and_breakdown():
    clean = compute_rank_score_components(avg_score=90.0, voice_clarity_penalty=0.0)
    penalized = compute_rank_score_components(avg_score=90.0, voice_clarity_penalty=1.25)
    breakdown = build_rank_score_breakdown(penalized)

    assert penalized["rank_score"] == clean["rank_score"] - 1.25
    assert penalized["voice_clarity_penalty"] == 1.25
    assert breakdown["voice_clarity_component"] == 1.25
    assert breakdown["components"]["voice_clarity_penalty"]["value"] == 1.25


def test_cache_signature_changes_when_decay_metadata_changes():
    base = {"f_l": [20.0, 40.0], "m_l": [0.0, 1.0], "f_r": [20.0, 40.0], "m_r": [0.0, 1.0]}
    sig_a = _auto_measurement_signature({**base, "measured_rt60_bands_l": {40.0: 0.35}})
    sig_b = _auto_measurement_signature({**base, "measured_rt60_bands_l": {40.0: 0.85}})

    assert sig_a != sig_b


def test_auto_signature_changes_when_tdc_decay_scoring_config_changes():
    base = {"filter_type": "mixed", "tdc_strength": 45.0}
    measurements = {"f_l": [20.0, 100.0], "m_l": [0.0, 0.0]}
    sig_a = _auto_signature(
        base_data=base,
        measurements=measurements,
        fs_v=44100,
        taps_v=65536,
        xos=[],
        hpf=None,
        hc_mode=None,
        include_hc_mode=False,
    )
    sig_b = _auto_signature(
        base_data={**base, "auto_mode_tdc_rt60_target_low_s": 0.55},
        measurements=measurements,
        fs_v=44100,
        taps_v=65536,
        xos=[],
        hpf=None,
        hc_mode=None,
        include_hc_mode=False,
    )

    assert sig_a != sig_b


def test_tdc_decay_tradeoff_no_crash_when_metadata_missing():
    penalty, dbg = _auto_tdc_decay_tradeoff({"enable_tdc": True, "tdc_strength": 50.0}, {}, {})

    assert penalty == 0.0
    assert dbg["tdc_decision"] == "neutral_no_decay_metadata"
    assert dbg["tdc_action_hint"] == "neutral_no_metadata"
    assert dbg["tdc_decay_penalty_total"] == 0.0
