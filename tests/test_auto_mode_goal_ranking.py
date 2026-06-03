from decaycore.io.decaycore_automatic_mode import (
    AutoModeConfig,
    _build_auto_mode_candidates,
    _build_auto_mode_candidates_optuna,
    _build_auto_mode_candidates_local,
    _auto_run_optuna_eval_loop,
    _auto_exc_penalty_bins_from_dbg,
    _auto_exc_zero_penalty_freq_hz_from_stats,
    _auto_goal,
    _auto_hybrid_mixed_freq_penalty,
    _auto_optuna_sampler_kwargs,
    _auto_optimizer_backend,
    _auto_rank_key,
    _auto_rank_key_goal,
    _auto_reject,
    _auto_trial_workers,
)
from decaycore.auto_mode.shared import (
    AUTO_MODE_GOAL_FLAT,
    AUTO_MODE_MAG_C_MAX_MIN_HZ,
    AUTO_MODE_RANK_SCORE_BIAS,
    AUTO_MODE_RANK_SCORE_GAIN,
    _auto_goal_norm,
    _auto_safe_float,
    _auto_sample_mag_low_pair,
)
import logging
from decaycore.io.auto_mode.candidate_generation import (
    _suggest_auto_mode_candidate_optuna,
    _seed_auto_mode_candidate_optuna_params,
    _seed_auto_mode_candidate_local_optuna_params,
    _seed_auto_mode_candidate_micro_optuna_params,
)
from decaycore.io.auto_mode.orchestrator_refine import (
    _CacheRefineContext,
    _CacheRefineProgress,
    _CacheRefineRound,
    _CacheRefineSeed,
    _load_exact_cache_seed,
    _run_optuna_cache_refine_round,
)
from decaycore.auto_mode._refine_cache_rounds import _run_cache_refine_rounds
from decaycore.auto_mode.optuna_backend import (
    _OPTUNA_CROSS_STUDY_BEST_PARAMS,
    _OPTUNA_KNOWN_RECORDS,
    _OPTUNA_KNOWN_SIGNATURES,
    _OPTUNA_KNOWN_SIGNATURES_PRIMED,
    _auto_optuna_cross_study_best_params,
    _auto_optuna_cached_study_records,
    _auto_optuna_bass_preference_bonus,
    _auto_optuna_build_completed_trial,
    _auto_optuna_effective_scope,
    _auto_optuna_objective_value,
    _auto_optuna_scope_for_filter,
    _auto_optuna_study_records,
)
from decaycore.auto_mode.optuna_backend_params import _auto_optuna_sanitize_enqueued_params
from decaycore.auto_mode.candidate_base import _derive_adaptive_freq_bounds
from decaycore.auto_mode.materialize import AutoModeMaterializeContext, build_materialize_helpers
from decaycore.auto_mode.rank_score import compute_rank_score_components
from decaycore.auto_mode.search.rank_combiner import _collect_rank_cached_focus_ripple
from decaycore.auto_mode.scoring_ranking import (
    _auto_hard_gate_reasons,
    _auto_is_better_refine,
    _auto_select_best_scored,
    filter_hard_failed_candidates,
    maybe_override_hard_failed_winner,
    select_best_safe_candidate,
)
import pytest
from types import SimpleNamespace


def test_auto_goal_defaults_to_balanced():
    assert _auto_goal({}) == "balanced"
    assert _auto_goal({"auto_goal": "unknown"}) == "balanced"
    assert _auto_goal({"auto_goal": "hybrid"}) == "low-ripple"


def test_auto_goal_prefer_bass_aliases_keep_flat_compatibility():
    assert _auto_goal({"auto_goal": "flat"}) == AUTO_MODE_GOAL_FLAT
    assert _auto_goal({"auto_goal": "prefer bass"}) == AUTO_MODE_GOAL_FLAT
    assert _auto_goal({"auto_goal": "prefer-bass"}) == AUTO_MODE_GOAL_FLAT
    assert _auto_goal({"auto_goal": "prefer_bass"}) == AUTO_MODE_GOAL_FLAT
    assert _auto_goal({"auto_goal": "bass"}) == AUTO_MODE_GOAL_FLAT
    assert _auto_goal_norm("flat") == AUTO_MODE_GOAL_FLAT


def test_auto_mode_config_from_base_data_reads_overrides():
    cfg = AutoModeConfig.from_base_data(
        {
            "auto_mode_trials": 77,
            "auto_mode_phase1_plateau_rounds": 4,
            "auto_mode_local_refine_enabled": False,
            "auto_mode_local_refine_top_k": 3,
            "auto_mode_local_refine_trials_per_top": 9,
            "auto_mode_optuna": True,
            "auto_mode_optuna_min_trials": 40,
            "auto_mode_optuna_startup_trials": 11,
            "auto_mode_optuna_multivariate": False,
            "auto_mode_optuna_group": True,
            "auto_mode_optuna_constant_liar": False,
            "auto_mode_optuna_persistent_study": False,
            "auto_mode_optuna_avoid_duplicates": False,
            "auto_mode_residual_peak_penalty_cap": 12.5,
            "auto_mode_max_avg_score_loss_for_safety_override": 0.07,
        }
    )
    assert cfg.trials == 77
    assert cfg.phase1_plateau_rounds == 4
    assert cfg.local_refine_enabled is False
    assert cfg.local_refine_top_k == 3
    assert cfg.local_refine_trials_per_top == 9
    assert cfg.optuna_pilot_enabled is True
    assert cfg.optuna_pilot_min_trials == 40
    assert cfg.optuna_pilot_startup_trials == 11
    assert cfg.optuna_multivariate is False
    assert cfg.optuna_group is True
    assert cfg.optuna_constant_liar is False
    assert cfg.optuna_persistent_study is False
    assert cfg.optuna_avoid_duplicates is False
    assert cfg.residual_peak_penalty_cap == 12.5
    assert cfg.max_avg_score_loss_for_safety_override == 0.07


def test_auto_mode_config_defaults_use_wider_phase1_startup_and_three_anchors():
    cfg = AutoModeConfig.from_base_data({})

    assert cfg.optuna_startup_phase1 == 6
    assert cfg.local_refine_top_k == 3
    assert cfg.local_refine_trials_per_top == 8


def test_rank_score_calibration_no_longer_saturates_near_95_avg():
    comp = compute_rank_score_components(
        avg_score=95.092,
        gain=AUTO_MODE_RANK_SCORE_GAIN,
        bias=AUTO_MODE_RANK_SCORE_BIAS,
    )

    assert float(comp["rank_score"]) < 100.0
    assert abs(float(comp["rank_score"]) - 93.3374) < 1e-6


def test_target_tracking_penalty_can_override_higher_legacy_avg_score():
    legacy_higher_avg_bad_tracking = compute_rank_score_components(
        avg_score=95.0,
        target_tracking_penalty=10.0,
        gain=AUTO_MODE_RANK_SCORE_GAIN,
        bias=AUTO_MODE_RANK_SCORE_BIAS,
    )
    lower_avg_good_tracking = compute_rank_score_components(
        avg_score=91.0,
        target_tracking_penalty=0.0,
        gain=AUTO_MODE_RANK_SCORE_GAIN,
        bias=AUTO_MODE_RANK_SCORE_BIAS,
    )

    assert float(legacy_higher_avg_bad_tracking["rank_score"]) < float(
        lower_avg_good_tracking["rank_score"]
    )


def test_bass_preference_bonus_can_prefer_bassier_equal_candidate():
    lean = compute_rank_score_components(
        avg_score=90.0,
        bass_preference_bonus=0.0,
        gain=AUTO_MODE_RANK_SCORE_GAIN,
        bias=AUTO_MODE_RANK_SCORE_BIAS,
    )
    bassier = compute_rank_score_components(
        avg_score=90.0,
        bass_preference_bonus=1.0,
        gain=AUTO_MODE_RANK_SCORE_GAIN,
        bias=AUTO_MODE_RANK_SCORE_BIAS,
    )

    assert float(bassier["rank_score"]) > float(lean["rank_score"])


def test_rank_score_components_include_structured_breakdown():
    comp = compute_rank_score_components(
        avg_score=90.0,
        boost_penalty=1.0,
        residual_peak_penalty=2.0,
        correction_sharpness_penalty=0.5,
        dip_fill_risk_penalty=0.75,
        channel_overfit_penalty=0.25,
        phase_risk_penalty=0.5,
        phase_benefit_bonus=0.25,
    )

    from decaycore.auto_mode.rank_score import build_rank_score_breakdown

    breakdown = build_rank_score_breakdown(comp, hard_gates=["residual_peak_hard_gate"])
    assert breakdown["total"] == comp["rank_score"]
    assert breakdown["boost_cut_component"] == 1.0
    assert breakdown["residual_peak_component"] == 2.0
    assert breakdown["anti_overfit_component"] == 1.5
    assert breakdown["hard_gates"] == ["residual_peak_hard_gate"]


def test_anti_overfit_penalties_reduce_rank_score():
    base = compute_rank_score_components(avg_score=90.0)
    penalized = compute_rank_score_components(
        avg_score=90.0,
        correction_sharpness_penalty=1.0,
        dip_fill_risk_penalty=2.0,
        channel_overfit_penalty=3.0,
    )

    assert float(penalized["rank_score_raw"]) == pytest.approx(float(base["rank_score_raw"]) - 6.0)
    assert float(penalized["rank_score"]) < float(base["rank_score"])


def test_score_only_materialize_cache_returns_same_metrics_without_recompute():
    calls = {"build": 0, "run": 0, "score": 0}

    def _build_config(final_data, **_kwargs):
        calls["build"] += 1
        return dict(final_data or {})

    def _run_pipeline(cfg, _measurements, *, include_response_arrays):
        calls["run"] += 1
        return SimpleNamespace(metrics={"cfg_id": cfg.get("id"), "include": include_response_arrays})

    def _score_result(result, **_kwargs):
        calls["score"] += 1
        return {
            "rank_score": 90.0,
            "avg_score": 88.0,
            "result_cfg_id": result.metrics.get("cfg_id"),
        }

    ctx = AutoModeMaterializeContext(
        cfg=SimpleNamespace(exc_min_hz=20.0, exc_max_hz=80.0),
        cache_base_data={"filter_type": "mixed", "hc_mode": "Harman8"},
        measurements={"f_l": [20.0, 100.0], "m_l": [0.0, 0.0]},
        fs_v=48000,
        taps_v=65536,
        xos=[],
        hpf=None,
        hc_f=[20.0, 100.0],
        hc_m=[0.0, 0.0],
        pin_obj=None,
        filter_key="mixed",
        max_safe_boost=8.0,
        goal="balanced",
        status_cb=None,
        exact_cached_metrics_getter=None,
        auto_score_result_fn=_score_result,
        auto_optuna_jsonable_fn=lambda value: value,
        auto_rank_key_fn=_auto_rank_key,
        auto_is_better_refine_fn=_auto_is_better_refine,
        build_config_fn=_build_config,
        run_pipeline_fn=_run_pipeline,
        summarize_run_fn=lambda result: "summary",
        preset_transient_keys=(),
        residual_tiebreak_enabled=False,
        residual_top_k=1,
        residual_rank_eps=0.0,
    )
    _ready, materialize, _sig, _residual = build_materialize_helpers(ctx)

    _result1, metrics1, _data1 = materialize({"id": "same"}, include_response_arrays=False, summarize=False)
    _result2, metrics2, _data2 = materialize({"id": "same"}, include_response_arrays=False, summarize=False)

    assert calls == {"build": 1, "run": 1, "score": 1}
    assert metrics1["rank_score"] == metrics2["rank_score"]
    assert metrics2["cache_info"]["score_only_cache"] == "hit"


def test_auto_rank_key_prefers_real_bass_boost_over_zero_boost_tie():
    zero_boost = {
        "rank_score": 86.0,
        "avg_score": 80.0,
        "bass_boost_20_200_db": 0.0,
        "post_filter_boost_peak_db": 0.0,
        "max_net_boost_db": -0.1,
        "events_total": 30,
        "events_severity": 1.0,
    }
    bassier = {
        "rank_score": 86.0,
        "avg_score": 80.0,
        "bass_boost_20_200_db": 0.0,
        "post_filter_boost_peak_db": 1.6,
        "max_net_boost_db": -0.2,
        "events_total": 30,
        "events_severity": 1.0,
    }

    assert _auto_rank_key(bassier) < _auto_rank_key(zero_boost)


def test_auto_select_best_scored_skips_nonfinite_and_hard_gated_candidates():
    winner = _auto_select_best_scored(
        [
            {
                "metrics": {
                    "rank_score": float("nan"),
                    "avg_score": 99.0,
                },
                "preset": {"id": "nan"},
            },
            {
                "metrics": {
                    "rank_score": 99.0,
                    "avg_score": 99.0,
                    "worst_residual_peak_db": 6.2,
                    "residual_peak_hard_gate_db": 6.0,
                },
                "preset": {"id": "hard"},
            },
            {
                "metrics": {
                    "rank_score": 80.0,
                    "avg_score": 80.0,
                    "worst_residual_peak_db": 2.0,
                    "residual_peak_hard_gate_db": 6.0,
                },
                "preset": {"id": "safe"},
            },
        ]
    )

    assert dict(winner or {}).get("preset", {}).get("id") == "safe"


def test_auto_is_better_refine_rejects_residual_peak_hard_gated_candidate():
    best = {
        "rank_score": 80.0,
        "avg_score": 80.0,
        "worst_residual_peak_db": 2.0,
        "residual_peak_hard_gate_db": 6.0,
    }
    unsafe = {
        "rank_score": 99.0,
        "avg_score": 99.0,
        "worst_residual_peak_db": 6.2,
        "residual_peak_hard_gate_db": 6.0,
    }

    better, reason = _auto_is_better_refine(unsafe, best, return_reason=True)

    assert better is False
    assert "residual_peak_hard_gate" in reason


def test_auto_select_best_scored_prefers_safe_candidate_over_better_hard_gated_avg_score():
    winner = _auto_select_best_scored(
        [
            {
                "metrics": {
                    "rank_score": 95.0,
                    "avg_score": 99.0,
                    "worst_residual_peak_db": 7.0,
                    "residual_peak_hard_gate_db": 6.0,
                },
                "preset": {"id": "unsafe"},
            },
            {
                "metrics": {
                    "rank_score": 78.0,
                    "avg_score": 82.0,
                    "worst_residual_peak_db": 2.0,
                    "residual_peak_hard_gate_db": 6.0,
                },
                "preset": {"id": "safe"},
            },
        ]
    )

    assert dict(winner or {}).get("preset", {}).get("id") == "safe"


def test_residual_peak_gate_is_strictly_greater_than_threshold():
    safe_at_gate = {
        "rank_score": 90.0,
        "avg_score": 90.0,
        "worst_residual_peak_db": 6.0,
        "residual_peak_hard_gate_db": 6.0,
    }

    assert "residual_peak_hard_gate" not in _auto_hard_gate_reasons(safe_at_gate)


def test_residual_peak_hard_gate_prefers_explicit_raw_db_source():
    raw_failed = {
        "rank_score": 90.0,
        "avg_score": 90.0,
        "worst_residual_peak_db": 4.0,
        "worst_residual_peak_raw_db": 6.2,
        "residual_peak_hard_gate_db": 6.0,
        "residual_peak_gate_value_db": 6.2,
        "residual_peak_gate_source": "raw_db",
    }
    severity_failed = {
        "rank_score": 90.0,
        "avg_score": 90.0,
        "worst_residual_peak_db": 6.2,
        "worst_residual_peak_raw_db": float("nan"),
        "residual_peak_hard_gate_db": 6.0,
        "residual_peak_gate_value_db": 6.2,
        "residual_peak_gate_source": "severity",
    }

    assert _auto_hard_gate_reasons(raw_failed) == ["residual_peak_hard_gate"]
    assert _auto_hard_gate_reasons(severity_failed) == ["residual_peak_severity_gate"]


def test_prefer_bass_residual_peak_hard_gate_allows_supported_boost():
    supported = {
        "rank_score": 90.0,
        "avg_score": 90.0,
        "worst_residual_peak_raw_db": 10.0,
        "residual_peak_hard_gate_db": 6.0,
        "residual_peak_gate_value_db": 10.0,
        "residual_peak_gate_source": "raw_db",
        "bass_boost_20_200_db": 5.0,
        "max_net_boost_db": 14.0,
        "hard_gate_failures": ["residual_peak_hard_gate"],
    }
    severe = dict(supported)
    severe["worst_residual_peak_raw_db"] = 13.0
    severe["residual_peak_gate_value_db"] = 13.0

    assert "residual_peak_hard_gate" not in _auto_hard_gate_reasons(supported, goal="prefer bass")
    assert "residual_peak_hard_gate" in _auto_hard_gate_reasons(severe, goal="prefer bass")


def test_prefer_bass_finalize_does_not_persist_supported_residual_hard_gate():
    from decaycore.auto_mode.search.score_result_finalize import finalize_score_result_metrics

    metrics = {
        "rank_score": 90.0,
        "avg_score": 90.0,
        "residual_peak_hard_gate_db": 6.0,
        "residual_peak_gate_value_db": 10.0,
        "residual_peak_gate_source": "raw_db",
        "bass_boost_20_200_db": 5.0,
        "max_net_boost_db": 14.0,
    }

    out = finalize_score_result_metrics(
        metrics,
        base_data={"auto_goal": "prefer bass"},
        worst_residual_peak_raw_db=10.0,
        worst_residual_peak_db=10.0,
        stereo_policy_gate_failed=False,
        rank_score=90.0,
        rank_components={},
    )

    assert out["hard_gate_failures"] == []
    assert out["residual_peak_hard_gate_effective_db"] == 11.0


def test_residual_peak_hard_gate_does_not_fail_low_confidence_or_out_of_band_modal_fallback():
    low_confidence = {
        "rank_score": 90.0,
        "avg_score": 90.0,
        "residual_peak_hard_gate_db": 6.0,
        "modal_residual_fallback_used": False,
        "modal_residual_fallback_peak_db": 8.0,
    }
    out_of_band = {
        **low_confidence,
        "modal_residual_fallback_used": False,
        "modal_residual_fallback_hz": 500.0,
    }

    assert _auto_hard_gate_reasons(low_confidence) == []
    assert _auto_hard_gate_reasons(out_of_band) == []


def test_select_best_safe_candidate_falls_back_when_all_hard_fail(caplog):
    caplog.set_level(logging.WARNING, logger="DecayCore")
    winner = select_best_safe_candidate(
        [
            {
                "metrics": {
                    "rank_score": 70.0,
                    "avg_score": 70.0,
                    "worst_residual_peak_db": 7.0,
                    "residual_peak_hard_gate_db": 6.0,
                },
                "preset": {"id": "least-bad"},
            },
            {
                "metrics": {
                    "rank_score": 60.0,
                    "avg_score": 60.0,
                    "worst_residual_peak_db": 8.0,
                    "residual_peak_hard_gate_db": 6.0,
                },
                "preset": {"id": "worse"},
            },
        ]
    )

    assert dict(winner or {}).get("preset", {}).get("id") == "least-bad"
    assert "fallback selected least-bad hard-failed candidate" in caplog.text


def test_filter_hard_failed_candidates_reports_residual_peak_diagnostics():
    safe, diagnostics = filter_hard_failed_candidates(
        [
            {
                "metrics": {
                    "rank_score": 99.0,
                    "avg_score": 99.0,
                    "worst_residual_peak_db": 7.1,
                    "residual_peak_hard_gate_db": 6.0,
                },
                "preset": {"id": "unsafe"},
            },
            {
                "metrics": {
                    "rank_score": 80.0,
                    "avg_score": 80.0,
                    "worst_residual_peak_db": 2.0,
                    "residual_peak_hard_gate_db": 6.0,
                },
                "preset": {"id": "safe"},
            },
        ]
    )

    assert [dict(item.get("preset", {}) or {}).get("id") for item in safe] == ["safe"]
    failed = [d for d in diagnostics if d["hard_gate_failed"]]
    assert failed
    assert failed[0]["residual_peak_db"] == 7.1
    assert failed[0]["residual_peak_hard_gate_db"] == 6.0
    assert "residual_peak_hard_gate" in failed[0]["hard_gate_reasons"]


def test_bass_integration_infeasible_candidate_cannot_win_over_safe_candidate():
    winner = select_best_safe_candidate(
        [
            {
                "metrics": {
                    "rank_score": 99.0,
                    "avg_score": 99.0,
                    "bass_integration_enable": True,
                    "bass_feasibility_class": "infeasible",
                    "bass_feasibility_reason": "Right channel remains limiting.",
                },
                "preset": {"id": "infeasible"},
            },
            {
                "metrics": {
                    "rank_score": 80.0,
                    "avg_score": 80.0,
                    "bass_integration_enable": True,
                    "bass_feasibility_class": "good",
                },
                "preset": {"id": "safe"},
            },
        ]
    )

    assert dict(winner or {}).get("preset", {}).get("id") == "safe"


def test_bass_integration_marginal_is_penalty_only_not_hard_gate():
    metrics = {
        "rank_score": 88.0,
        "avg_score": 88.0,
        "bass_integration_enable": True,
        "bass_feasibility_class": "marginal",
        "bass_feasibility_reason": "Limited shared sub compromise.",
    }

    assert "bass_integration_infeasible_hard_gate" not in _auto_hard_gate_reasons(metrics)
    assert not filter_hard_failed_candidates([{"metrics": metrics, "preset": {"id": "marginal"}}])[1][0][
        "hard_gate_failed"
    ]


def test_direct_dac_bass_integration_infeasible_is_penalty_only_not_hard_gate():
    metrics = {
        "rank_score": 88.0,
        "avg_score": 88.0,
        "bass_integration_enable": True,
        "bass_integration_mode": "direct_dac",
        "bass_direct_dac_export_model": "camilladsp_yaml_compatible",
        "bass_feasibility_class": "infeasible",
        "bass_feasibility_reason": "Right channel remains limiting.",
    }

    assert "bass_integration_infeasible_hard_gate" not in _auto_hard_gate_reasons(metrics)
    assert not filter_hard_failed_candidates([{"metrics": metrics, "preset": {"id": "direct-dac"}}])[1][0][
        "hard_gate_failed"
    ]


def test_bass_integration_all_infeasible_fallback_selects_least_bad(caplog):
    caplog.set_level(logging.WARNING)

    winner = select_best_safe_candidate(
        [
            {
                "metrics": {
                    "rank_score": 70.0,
                    "avg_score": 70.0,
                    "bass_integration_enable": True,
                    "bass_feasibility_class": "infeasible",
                },
                "preset": {"id": "least-bad"},
            },
            {
                "metrics": {
                    "rank_score": 60.0,
                    "avg_score": 60.0,
                    "bass_integration_enable": True,
                    "bass_feasibility_class": "infeasible",
                },
                "preset": {"id": "worse"},
            },
        ]
    )

    assert dict(winner or {}).get("preset", {}).get("id") == "least-bad"
    assert "fallback selected least-bad hard-failed candidate" in caplog.text


def test_maybe_override_hard_failed_winner_accepts_small_avg_loss():
    current = {
        "metrics": {
            "rank_score": 95.0,
            "avg_score": 90.0,
            "worst_residual_peak_db": 7.0,
            "residual_peak_hard_gate_db": 6.0,
        },
        "preset": {"id": "unsafe"},
    }
    safe = {
        "metrics": {
            "rank_score": 90.0,
            "avg_score": 89.95,
            "worst_residual_peak_db": 2.0,
            "residual_peak_hard_gate_db": 6.0,
        },
        "preset": {"id": "safe"},
    }

    winner, meta = maybe_override_hard_failed_winner(current, [current, safe], SimpleNamespace(max_avg_score_loss_for_safety_override=0.10))

    assert dict(winner or {}).get("preset", {}).get("id") == "safe"
    assert meta["applied"] is True
    assert meta["average_score_loss"] == pytest.approx(0.05)


def test_maybe_override_hard_failed_winner_respects_avg_loss_guard():
    current = {
        "metrics": {
            "rank_score": 95.0,
            "avg_score": 90.0,
            "worst_residual_peak_db": 7.0,
            "residual_peak_hard_gate_db": 6.0,
        },
        "preset": {"id": "unsafe"},
    }
    safe = {
        "metrics": {
            "rank_score": 90.0,
            "avg_score": 89.0,
            "worst_residual_peak_db": 2.0,
            "residual_peak_hard_gate_db": 6.0,
        },
        "preset": {"id": "safe"},
    }

    winner, meta = maybe_override_hard_failed_winner(current, [current, safe], SimpleNamespace(max_avg_score_loss_for_safety_override=0.10))

    assert dict(winner or {}).get("preset", {}).get("id") == "unsafe"
    assert meta["applied"] is False
    assert meta["reason"] == "avg_score_loss_guard"


def test_auto_hard_gate_reasons_reports_existing_guards():
    metrics = {
        "rank_score": 90.0,
        "max_net_boost_db": 12.5,
        "worst_residual_peak_db": 6.1,
        "residual_peak_hard_gate_db": 6.0,
        "stereo_policy_gate_failed": True,
    }
    st = {"pre_energy_metric_suspect": False, "ir_pre_post_ratio": 0.08}

    reasons = _auto_hard_gate_reasons(metrics, st, st, goal="prefer bass")
    assert "excessive_net_boost" in reasons
    assert "residual_peak_hard_gate" in reasons
    assert "stereo_policy_gate_failed" in reasons
    assert "unsafe_prepost_l" in reasons


def test_compute_rank_score_components_hard_gate_failures_round_trip_in_breakdown():
    components = compute_rank_score_components(avg_score=90.0, residual_peak_penalty=2.0)
    metrics = {
        **components,
        "worst_residual_peak_db": 6.0,
        "residual_peak_hard_gate_db": 6.0,
        "hard_gate_failures": ["residual_peak_hard_gate", "bass_integration_infeasible_hard_gate"],
    }

    from decaycore.auto_mode.rank_score import attach_official_rank_score

    enriched = attach_official_rank_score(metrics)

    assert enriched["rank_score_breakdown"]["hard_gates"] == [
        "residual_peak_hard_gate",
        "bass_integration_infeasible_hard_gate",
    ]


def test_auto_rank_key_prefers_better_target_tracking_before_avg_tie():
    better_tracking = {
        "rank_score": 86.0,
        "avg_score": 79.0,
        "target_tracking_rms_20_200_db": 2.0,
        "target_tracking_rms_100_500_db": 1.5,
    }
    worse_tracking = {
        "rank_score": 86.0,
        "avg_score": 82.0,
        "target_tracking_rms_20_200_db": 4.0,
        "target_tracking_rms_100_500_db": 3.5,
    }

    assert _auto_rank_key(better_tracking) < _auto_rank_key(worse_tracking)


def test_refine_rejects_small_rank_gain_when_target_tracking_worsens():
    best = {
        "rank_score": 49.34,
        "target_tracking_rms_20_200_db": 3.2,
        "target_tracking_rms_100_500_db": 2.8,
    }
    worse_tracking = {
        "rank_score": 49.38,
        "target_tracking_rms_20_200_db": 4.3,
        "target_tracking_rms_100_500_db": 3.5,
    }

    better, reason = _auto_is_better_refine(worse_tracking, best, "balanced", return_reason=True)

    assert better is False
    assert reason == "target_tracking"


def test_target_curve_selection_prefers_next_bassier_target_on_close_rank():
    winner = _auto_select_best_scored(
        [
            {
                "_auto_select_kind": "target_curve",
                "_target_rank_tie_eps": 0.05,
                "hc_mode": "Harman8",
                "avg_rank_score": 34.098,
                "fit_rms_db": 4.66,
                "boost_penalty": 0.0,
                "best_metrics": {"rank_score": 54.625, "avg_score": 71.0},
            },
            {
                "_auto_select_kind": "target_curve",
                "_target_rank_tie_eps": 0.05,
                "hc_mode": "Harman10",
                "avg_rank_score": 34.338,
                "fit_rms_db": 4.80,
                "boost_penalty": 0.0,
                "best_metrics": {"rank_score": 53.704, "avg_score": 71.2},
            },
            {
                "_auto_select_kind": "target_curve",
                "_target_rank_tie_eps": 0.05,
                "hc_mode": "Harman12",
                "avg_rank_score": 35.841,
                "fit_rms_db": 5.20,
                "boost_penalty": 0.0,
                "best_metrics": {"rank_score": 50.032, "avg_score": 70.8},
            },
        ]
    )

    assert str(dict(winner or {}).get("hc_mode")) == "Harman10"
    assert str(dict(winner or {}).get("_auto_selection_method")) == "top3x10_trials_bass_forward_close_rank"


def test_target_curve_selection_keeps_rank_winner_when_bassier_target_too_far():
    winner = _auto_select_best_scored(
        [
            {
                "_auto_select_kind": "target_curve",
                "hc_mode": "Harman8",
                "best_metrics": {"rank_score": 54.625},
            },
            {
                "_auto_select_kind": "target_curve",
                "hc_mode": "Harman10",
                "best_metrics": {"rank_score": 52.90},
            },
        ]
    )

    assert str(dict(winner or {}).get("hc_mode")) == "Harman8"


def test_auto_optimizer_backend_selection(monkeypatch):
    monkeypatch.delenv("CAMILLAFIR_AUTO_MODE_OPTIMIZER", raising=False)
    assert _auto_optimizer_backend({"auto_mode_optimizer": "optuna"}) == "optuna"
    assert _auto_optimizer_backend({"auto_mode_optuna": True}) == "optuna"
    assert _auto_optimizer_backend({"auto_mode_optimizer": "builtin", "auto_mode_optuna": True}) == "builtin"

    monkeypatch.setenv("CAMILLAFIR_AUTO_MODE_OPTIMIZER", "builtin")
    assert _auto_optimizer_backend({"auto_mode_optimizer": "optuna", "auto_mode_optuna": True}) == "builtin"


def test_auto_safe_float_none_uses_default_without_error_log(caplog):
    with caplog.at_level(logging.ERROR, logger="DecayCore"):
        assert _auto_safe_float(None, 12.5) == 12.5

    assert "safe float parse" not in caplog.text


def test_auto_safe_float_logs_unexpected_parse_failure(caplog):
    class _ExplodingFloat:
        def __float__(self):
            raise RuntimeError("boom")

    with caplog.at_level(logging.ERROR, logger="DecayCore"):
        assert _auto_safe_float(_ExplodingFloat(), 7.0) == 7.0

    assert "safe float parse" in caplog.text


def test_auto_optuna_sampler_kwargs_defaults_and_parallel_behavior():
    assert _auto_optuna_sampler_kwargs({}, workers=1) == {
        "multivariate": True,
        "group": False,
        "constant_liar": False,
        "warn_independent_sampling": False,
    }
    assert _auto_optuna_sampler_kwargs({}, workers=4) == {
        "multivariate": True,
        "group": False,
        "constant_liar": True,
        "warn_independent_sampling": False,
    }


def test_auto_optuna_sampler_kwargs_group_requires_multivariate():
    assert _auto_optuna_sampler_kwargs(
        {
            "auto_mode_optuna_multivariate": False,
            "auto_mode_optuna_group": True,
            "auto_mode_optuna_constant_liar": True,
        },
        workers=8,
    ) == {
        "multivariate": False,
        "group": False,
        "constant_liar": True,
        "warn_independent_sampling": False,
    }


def test_auto_optuna_objective_value_prefers_bassier_candidate_on_equal_rank():
    lean = {
        "rank_score": 90.0,
        "lf_boost_max_db": 0.2,
        "realized_rms_20_200_db": 0.45,
        "bass_integration_penalty": 0.0,
        "bass_feasibility_penalty": 0.0,
        "max_net_boost_db": 4.0,
    }
    bassier = {
        "rank_score": 90.0,
        "lf_boost_max_db": 2.6,
        "realized_rms_20_200_db": 0.10,
        "bass_integration_penalty": 0.0,
        "bass_feasibility_penalty": 0.0,
        "max_net_boost_db": 4.4,
    }

    assert _auto_optuna_objective_value(bassier) > _auto_optuna_objective_value(lean)


def test_auto_optuna_bass_preference_uses_final_bass_boost_metric():
    lean = {
        "rank_score": 90.0,
        "bass_boost_20_200_db": 0.0,
        "target_tracking_rms_20_200_db": 0.20,
        "bass_integration_penalty": 0.0,
        "bass_feasibility_penalty": 0.0,
        "max_net_boost_db": -0.1,
    }
    bassier = {
        "rank_score": 90.0,
        "bass_boost_20_200_db": 2.2,
        "target_tracking_rms_20_200_db": 0.20,
        "bass_integration_penalty": 0.0,
        "bass_feasibility_penalty": 0.0,
        "max_net_boost_db": -0.2,
    }

    assert _auto_optuna_bass_preference_bonus(bassier) > _auto_optuna_bass_preference_bonus(lean)
    assert _auto_optuna_objective_value(bassier) > _auto_optuna_objective_value(lean)


def test_auto_optuna_bass_preference_uses_target_tracking_when_available():
    good_tracking = {
        "lf_boost_max_db": 2.6,
        "target_tracking_rms_20_200_db": 0.10,
        "realized_rms_20_200_db": 0.45,
        "bass_integration_penalty": 0.0,
        "bass_feasibility_penalty": 0.0,
        "max_net_boost_db": 4.4,
    }
    poor_tracking = {
        "lf_boost_max_db": 2.6,
        "target_tracking_rms_20_200_db": 0.90,
        "realized_rms_20_200_db": 0.10,
        "bass_integration_penalty": 0.0,
        "bass_feasibility_penalty": 0.0,
        "max_net_boost_db": 4.4,
    }

    assert _auto_optuna_bass_preference_bonus(good_tracking) > _auto_optuna_bass_preference_bonus(poor_tracking)


def test_auto_optuna_bass_preference_bonus_scales_down_with_bass_risk_penalties():
    controlled = {
        "lf_boost_max_db": 3.0,
        "realized_rms_20_200_db": 0.08,
        "bass_integration_penalty": 0.0,
        "bass_feasibility_penalty": 0.0,
        "max_net_boost_db": 4.8,
    }
    risky = {
        "lf_boost_max_db": 3.0,
        "realized_rms_20_200_db": 0.08,
        "bass_integration_penalty": 4.0,
        "bass_feasibility_penalty": 6.0,
        "max_net_boost_db": 7.0,
    }

    assert _auto_optuna_bass_preference_bonus(controlled) > _auto_optuna_bass_preference_bonus(risky)


def test_auto_optuna_cross_study_seeds_use_summary_best_trial_without_loading_studies(monkeypatch):
    _OPTUNA_CROSS_STUDY_BEST_PARAMS.clear()
    loaded = {"n": 0}

    summaries = [
        SimpleNamespace(
            study_name="camillafir-phase1-scope-aaa",
            best_trial=SimpleNamespace(value=92.0, params={"mag_c_min": 28.0}),
        ),
        SimpleNamespace(
            study_name="camillafir-phase1-scope-bbb",
            best_trial=SimpleNamespace(value=91.0, params={"mag_c_min": 32.0}),
        ),
    ]
    fake_optuna = SimpleNamespace(
        get_all_study_summaries=lambda storage=None: summaries,
        load_study=lambda **kwargs: loaded.__setitem__("n", loaded["n"] + 1),
    )

    monkeypatch.setattr(
        "decaycore.auto_mode.optuna_backend_params._auto_optuna_create_storage",
        lambda *args, **kwargs: object(),
    )

    params = _auto_optuna_cross_study_best_params(
        fake_optuna,
        base_data={},
        scope="phase1-scope",
        current_study_name="camillafir-phase1-scope-current",
        top_n=1,
    )

    assert params == [{"mag_c_min": 28.0}]
    assert loaded["n"] == 0


def test_auto_optuna_known_records_cache_avoids_second_study_scan():
    _OPTUNA_KNOWN_SIGNATURES.clear()
    _OPTUNA_KNOWN_SIGNATURES_PRIMED.clear()
    _OPTUNA_KNOWN_RECORDS.clear()

    study = SimpleNamespace(
        get_trials=lambda deepcopy=False: [
            SimpleNamespace(params={"mag_c_min": 25.0}, value=88.0, user_attrs={})
        ]
    )

    first = _auto_optuna_study_records(study)
    _OPTUNA_KNOWN_RECORDS["study"] = dict(first)
    _OPTUNA_KNOWN_SIGNATURES_PRIMED.add("study")

    assert first
    assert _auto_optuna_cached_study_records("study") == first


def test_seed_auto_mode_candidate_optuna_params_clamps_adaptive_output_tilt_non_negative():
    params = _seed_auto_mode_candidate_optuna_params(
        {
            "auto_target_mode": "adaptive",
            "output_tilt_db_per_oct": -1.25,
        },
        None,
    )

    assert float(params.get("output_tilt_db_per_oct", 0.0)) == 0.0


def test_seed_auto_mode_candidate_optuna_params_clamps_to_adaptive_low_bass_bounds():
    params = _seed_auto_mode_candidate_optuna_params(
        {
            "harmonic_freq_hz_l": [20.0, 80.0],
            "harmonic_freq_hz_r": [20.0, 90.0],
        },
        {"low_bass_cut_hz": 29.7},
    )

    assert float(params["low_bass_cut_hz"]) == 24.0


def test_auto_rank_key_goal_balanced_matches_legacy():
    metrics = {
        "rank_score": 81.2,
        "avg_score": 86.5,
        "max_net_boost_db": 4.1,
        "events_severity": 0.7,
        "events_total": 2,
        "lr_delta_score": 0.4,
        "dsp_penalty_raw": 3.0,
        "exc_penalty_raw": 1.2,
    }
    assert _auto_rank_key_goal(metrics, goal="balanced") == _auto_rank_key(metrics)
    assert _auto_rank_key_goal(metrics, goal="not-a-goal") == _auto_rank_key(metrics)


def test_auto_rank_key_goal_acoustic_prefers_avg_score_first():
    high_rank_low_avg = {
        "rank_score": 88.0,
        "avg_score": 76.0,
        "lr_delta_score": 0.6,
        "dsp_penalty_raw": 1.2,
        "events_severity": 0.2,
        "max_net_boost_db": 3.0,
        "exc_penalty_raw": 0.7,
    }
    lower_rank_higher_avg = {
        "rank_score": 82.0,
        "avg_score": 79.0,
        "lr_delta_score": 0.7,
        "dsp_penalty_raw": 1.2,
        "events_severity": 0.2,
        "max_net_boost_db": 3.0,
        "exc_penalty_raw": 0.7,
    }

    assert _auto_rank_key_goal(high_rank_low_avg, goal="balanced") < _auto_rank_key_goal(
        lower_rank_higher_avg, goal="balanced"
    )
    assert _auto_rank_key_goal(lower_rank_higher_avg, goal="acoustic") < _auto_rank_key_goal(
        high_rank_low_avg, goal="acoustic"
    )


def test_auto_rank_key_goal_prefer_bass_prefers_safe_bassier_tie():
    lean = {
        "rank_score": 86.0,
        "avg_score": 80.0,
        "phase_risk_penalty": 0.0,
        "phase_net_score": 0.0,
        "target_tracking_rms_20_200_db": 1.0,
        "target_tracking_rms_100_500_db": 1.0,
        "events_severity": 0.0,
        "bass_boost_20_200_db": 0.8,
        "max_net_boost_db": 2.0,
    }
    bassier = dict(lean)
    bassier["bass_boost_20_200_db"] = 4.2
    bassier["max_net_boost_db"] = 4.8

    assert _auto_rank_key_goal(bassier, goal="prefer bass") < _auto_rank_key_goal(
        lean, goal="prefer bass"
    )


def test_auto_rank_key_goal_flat_prefers_safe_bass_boost_over_small_avg_loss():
    lean = {
        "rank_score": 88.0,
        "avg_score": 82.0,
        "phase_risk_penalty": 0.0,
        "phase_net_score": 0.0,
        "target_tracking_rms_20_200_db": 0.7,
        "target_tracking_rms_100_500_db": 0.7,
        "events_severity": 0.0,
        "events_total": 0,
        "bass_boost_20_200_db": 0.4,
        "max_net_boost_db": 1.0,
        "exc_penalty_raw": 0.0,
        "bass_feasibility_penalty": 0.0,
    }
    boosted = dict(lean)
    boosted["rank_score"] = 87.0
    boosted["avg_score"] = 77.5
    boosted["bass_boost_20_200_db"] = 3.0
    boosted["max_net_boost_db"] = 3.4

    assert _auto_rank_key_goal(boosted, goal="flat") < _auto_rank_key_goal(lean, goal="flat")


def test_auto_rank_key_goal_flat_prefers_bass_before_small_tracking_loss():
    lean = {
        "rank_score": 89.0,
        "avg_score": 82.0,
        "phase_risk_penalty": 0.0,
        "phase_net_score": 0.0,
        "target_tracking_rms_20_200_db": 0.6,
        "target_tracking_rms_100_500_db": 0.6,
        "events_severity": 0.0,
        "events_total": 0,
        "bass_boost_20_200_db": 0.3,
        "max_net_boost_db": 1.0,
        "exc_penalty_raw": 0.0,
        "bass_feasibility_penalty": 0.0,
    }
    boosted = dict(lean)
    boosted["rank_score"] = 86.0
    boosted["avg_score"] = 78.8
    boosted["target_tracking_rms_20_200_db"] = 1.1
    boosted["target_tracking_rms_100_500_db"] = 1.1
    boosted["bass_boost_20_200_db"] = 4.0
    boosted["max_net_boost_db"] = 5.4

    assert _auto_rank_key_goal(boosted, goal="flat") < _auto_rank_key_goal(lean, goal="flat")


def test_auto_select_best_scored_uses_flat_bass_preference_goal():
    lean = {
        "rank_score": 88.0,
        "avg_score": 82.0,
        "phase_risk_penalty": 0.0,
        "phase_net_score": 0.0,
        "target_tracking_rms_20_200_db": 0.7,
        "target_tracking_rms_100_500_db": 0.7,
        "events_severity": 0.0,
        "events_total": 0,
        "bass_boost_20_200_db": 0.4,
        "max_net_boost_db": 1.0,
        "exc_penalty_raw": 0.0,
        "bass_feasibility_penalty": 0.0,
    }
    boosted = dict(lean)
    boosted["rank_score"] = 87.0
    boosted["avg_score"] = 77.5
    boosted["bass_boost_20_200_db"] = 3.0
    boosted["max_net_boost_db"] = 3.4

    winner = _auto_select_best_scored(
        [
            {"name": "lean", "metrics": lean},
            {"name": "boosted", "metrics": boosted},
        ],
        goal="flat",
    )

    assert winner["name"] == "boosted"


def test_auto_rank_key_goal_prefer_bass_prefers_boost_before_events():
    clean = {
        "rank_score": 86.0,
        "avg_score": 80.0,
        "phase_risk_penalty": 0.0,
        "phase_net_score": 0.0,
        "target_tracking_rms_20_200_db": 1.0,
        "target_tracking_rms_100_500_db": 1.0,
        "events_severity": 0.0,
        "bass_boost_20_200_db": 0.8,
        "max_net_boost_db": 2.0,
    }
    bassier_with_events = dict(clean)
    bassier_with_events["events_severity"] = 0.6
    bassier_with_events["bass_boost_20_200_db"] = 4.2
    bassier_with_events["max_net_boost_db"] = 4.8

    assert _auto_rank_key_goal(bassier_with_events, goal="prefer bass") < _auto_rank_key_goal(
        clean, goal="prefer bass"
    )


def test_auto_reject_is_hard_guard_for_acoustic_only():
    metrics = {"max_net_boost_db": 8.5}
    st = {"pre_energy_metric_suspect": False, "ir_pre_post_ratio": 0.02}
    assert _auto_reject(metrics, st, st, goal="acoustic")
    assert not _auto_reject(metrics, st, st, goal="balanced")


def test_auto_reject_applies_hard_guards_to_prefer_bass_alias():
    ok = {"pre_energy_metric_suspect": False, "ir_pre_post_ratio": 0.02}
    bad_pre = {"pre_energy_metric_suspect": False, "ir_pre_post_ratio": 0.08}
    bad_gd = {"pre_energy_metric_suspect": False, "gd_grad_limiter_after_max_ms_per_oct": 48.0}

    assert not _auto_reject({"rank_score": 90.0, "max_net_boost_db": 8.5}, ok, ok, goal="prefer bass")
    assert _auto_reject({"rank_score": 90.0, "max_net_boost_db": 12.5}, ok, ok, goal="prefer bass")
    assert not _auto_reject(
        {"rank_score": 90.0, "max_net_boost_db": 14.0, "bass_boost_20_200_db": 5.0},
        ok,
        ok,
        goal="prefer bass",
    )
    assert _auto_reject(
        {"rank_score": 90.0, "max_net_boost_db": 19.0, "bass_boost_20_200_db": 7.0},
        ok,
        ok,
        goal="prefer bass",
    )
    assert _auto_reject({"rank_score": 90.0, "max_net_boost_db": 4.5}, bad_pre, ok, goal="prefer bass")
    assert _auto_reject({"rank_score": 90.0, "max_net_boost_db": 4.5}, bad_gd, ok, goal="prefer bass")


def test_auto_rank_key_goal_hybrid_keeps_rank_score_primary():
    better_rank_worse_avg = {
        "rank_score": 85.0,
        "avg_score": 74.0,
        "lr_delta_score": 0.6,
        "dsp_penalty_raw": 1.2,
        "events_severity": 0.2,
        "mixed_freq_penalty": 0.0,
        "max_net_boost_db": 3.0,
        "exc_penalty_raw": 0.7,
    }
    tied_rank_better_avg = {
        "rank_score": 85.0,
        "avg_score": 76.0,
        "lr_delta_score": 0.6,
        "dsp_penalty_raw": 1.2,
        "events_severity": 0.2,
        "mixed_freq_penalty": 0.0,
        "max_net_boost_db": 3.0,
        "exc_penalty_raw": 0.7,
    }
    lower_rank_best_avg = {
        "rank_score": 84.0,
        "avg_score": 90.0,
        "lr_delta_score": 0.6,
        "dsp_penalty_raw": 1.2,
        "events_severity": 0.2,
        "mixed_freq_penalty": 0.0,
        "max_net_boost_db": 3.0,
        "exc_penalty_raw": 0.7,
    }

    assert _auto_rank_key_goal(tied_rank_better_avg, goal="hybrid") < _auto_rank_key_goal(
        better_rank_worse_avg, goal="hybrid"
    )
    assert _auto_rank_key_goal(better_rank_worse_avg, goal="hybrid") < _auto_rank_key_goal(
        lower_rank_best_avg, goal="hybrid"
    )

    low_mixed_pen = dict(tied_rank_better_avg)
    high_mixed_pen = dict(tied_rank_better_avg)
    low_mixed_pen["mixed_freq_penalty"] = 0.0
    high_mixed_pen["mixed_freq_penalty"] = 1.0
    assert _auto_rank_key_goal(low_mixed_pen, goal="hybrid") < _auto_rank_key_goal(
        high_mixed_pen, goal="hybrid"
    )


def test_hybrid_mixed_freq_penalty_is_soft_tiebreak():
    base = {"filter_type": "Mixed", "bass_first_ai": True}
    assert _auto_hybrid_mixed_freq_penalty({"mixed_freq": 95.0}, base_data=base, goal="hybrid") == 0.0
    assert _auto_hybrid_mixed_freq_penalty({"mixed_freq": 150.0}, base_data=base, goal="hybrid") > 0.0
    assert _auto_hybrid_mixed_freq_penalty({"mixed_freq": 150.0}, base_data=base, goal="balanced") == 0.0


def test_auto_api_and_shared_bounds_stay_in_sync():
    from decaycore.auto_mode import api as auto_api
    from decaycore.auto_mode import shared as auto_shared

    assert float(auto_api.AUTO_MODE_MAG_C_MIN_MIN_HZ) == float(auto_shared.AUTO_MODE_MAG_C_MIN_MIN_HZ) == 15.0
    assert float(auto_api.AUTO_MODE_MAG_C_MIN_MAX_HZ) == float(auto_shared.AUTO_MODE_MAG_C_MIN_MAX_HZ) == 70.0
    assert float(auto_api.AUTO_MODE_LOW_BASS_MIN_HZ) == float(auto_shared.AUTO_MODE_LOW_BASS_MIN_HZ) == 18.0
    assert float(auto_api.AUTO_MODE_LOW_BASS_MAX_HZ) == float(auto_shared.AUTO_MODE_LOW_BASS_MAX_HZ) == 55.0
    assert float(auto_api.AUTO_MODE_PHASE_LIMIT_MIN_HZ) == float(auto_shared.AUTO_MODE_PHASE_LIMIT_MIN_HZ) == 100.0
    assert float(auto_api.AUTO_MODE_PHASE_LIMIT_MAX_HZ) == float(auto_shared.AUTO_MODE_PHASE_LIMIT_MAX_HZ) == 500.0
    assert float(auto_api.AUTO_MODE_PHASE_LIMIT_DEFAULT_HZ) == float(auto_shared.AUTO_MODE_PHASE_LIMIT_DEFAULT_HZ) == 380.0


def test_auto_mode_candidates_do_not_force_low_bass_cut_on():
    candidates = _build_auto_mode_candidates(
        {
            "filter_type": "Asymmetric",
            "low_bass_cut_enable": False,
            "low_bass_cut_hz": 40.0,
            "mag_c_min": 25.0,
        },
        n_trials=8,
        seed=123,
    )

    assert candidates
    assert all(bool(dict(c).get("low_bass_cut_enable", False)) is False for c in candidates)
    assert any("low_bass_cut_hz" in dict(c) for c in candidates[1:])


def test_suggest_auto_mode_candidate_optuna_clamps_tdc_soft_protections():
    class _FakeTrial:
        def suggest_float(self, name, low, high, step=None, log=False):
            _ = step, log
            return {
                "fdw_cycles": 5.0,
                "tdc_strength": 0.0,
                "tdc_max_reduction_db": 0.0,
                "reg_strength": 15.0,
                "max_boost": 3.0,
                "mag_c_min": 15.0,
                "mag_c_max": 170.0,
                "trans_width": 70.0,
                "bass_first_mode_max_hz": 120.0,
                "conf_pull_max_hz": 80.0,
                "low_bass_cut_hz": 18.0,
                "mixed_freq": 80.0,
                "phase_limit_range_u": 1.0,
                "phase_limit": 100.0,
                "phase_limit_u": 0.0,
                "output_tilt_db_per_oct": 0.0,
            }.get(str(name), float(low))

        def suggest_categorical(self, name, choices):
            values = {
                "enable_afdw": False,
                "bass_first_ai": False,
                "tdc_slope_db_per_oct": 3.0,
                "max_slope_db_per_oct": 24.0,
                "max_slope_boost_db_per_oct": 0.0,
                "max_slope_cut_db_per_oct": 0.0,
            }
            return values.get(str(name), list(choices)[0])

    cand = _suggest_auto_mode_candidate_optuna(
        {
            "auto_goal": "balanced",
            "enable_tdc": True,
            "enable_afdw": True,
            "bass_first_ai": True,
            "filter_type": "Linear Phase",
        },
        _FakeTrial(),
    )

    assert float(cand.get("tdc_strength", -1.0)) == 5.0
    assert float(cand.get("tdc_max_reduction_db", -1.0)) == 1.0
    assert bool(cand.get("enable_afdw", False)) is True
    assert bool(cand.get("bass_first_ai", False)) is True
    assert float(cand.get("max_slope_db_per_oct", 0.0)) == 24.0
    assert float(cand.get("bass_first_mode_max_hz", 0.0)) == 120.0
    assert float(cand.get("conf_pull_max_hz", 0.0)) == 80.0
    assert float(cand.get("phase_limit", 0.0)) == 180.0


def test_phase1_candidate_ranges_are_conservative_for_balanced_goal():
    candidates = _build_auto_mode_candidates(
        {
            "auto_goal": "balanced",
            "filter_type": "Linear Phase",
            "enable_afdw": True,
            "bass_first_ai": True,
        },
        n_trials=64,
        seed=12345,
    )

    trial_candidates = [dict(c) for c in candidates[1:]]
    assert trial_candidates
    assert all(float(c["max_boost"]) <= 8.0 for c in trial_candidates)
    assert all(float(c["tdc_max_reduction_db"]) <= 18.0 for c in trial_candidates)
    assert all(bool(c["enable_afdw"]) is True for c in trial_candidates)
    assert all(bool(c["bass_first_ai"]) is True for c in trial_candidates)


def test_phase1_optuna_ranges_cap_boost_and_tdc_for_balanced_goal():
    class _FakeTrial:
        number = 1

        def __init__(self):
            self.float_calls = {}

        def suggest_float(self, name, low, high, step=None, log=False):
            self.float_calls[str(name)] = {
                "low": float(low),
                "high": float(high),
                "step": step,
                "log": bool(log),
            }
            if str(name) == "phase_limit_u":
                return 1.0
            if str(name) == "phase_limit_range_u":
                return 1.0
            return float(high)

        def suggest_categorical(self, name, choices):
            values = {
                "enable_afdw": False,
                "bass_first_ai": False,
                "tdc_slope_db_per_oct": 36.0,
                "max_slope_db_per_oct": 24.0,
                "max_slope_boost_db_per_oct": 36.0,
                "max_slope_cut_db_per_oct": 36.0,
            }
            return values.get(str(name), list(choices)[-1])

    trial = _FakeTrial()
    cand = _suggest_auto_mode_candidate_optuna(
        {
            "auto_goal": "balanced",
            "filter_type": "Linear Phase",
            "enable_afdw": True,
            "bass_first_ai": True,
        },
        trial,
    )

    assert float(cand["max_boost"]) <= 8.0
    assert float(cand["tdc_max_reduction_db"]) <= 18.0
    assert bool(cand["enable_afdw"]) is True
    assert bool(cand["bass_first_ai"]) is True
    assert float(cand["phase_limit"]) == 420.0
    assert trial.float_calls["max_boost"]["high"] == 8.0
    assert trial.float_calls["tdc_max_reduction_db"]["high"] == 18.0
    assert trial.float_calls["phase_limit_u"]["low"] == 0.0
    assert trial.float_calls["phase_limit_u"]["high"] == 1.0
    assert trial.float_calls["phase_limit_range_u"]["low"] == 0.0
    assert trial.float_calls["phase_limit_range_u"]["high"] == 1.0


def test_phase1_flat_goal_keeps_wider_boost_range():
    class _FakeTrial:
        number = 1

        def suggest_float(self, name, low, high, step=None, log=False):
            _ = step, log
            if str(name) == "phase_limit_range_u":
                return 1.0
            if str(name) == "phase_limit_u":
                return 0.5
            return float(high)

        def suggest_categorical(self, name, choices):
            values = {
                "enable_afdw": False,
                "bass_first_ai": False,
                "tdc_slope_db_per_oct": 36.0,
                "max_slope_db_per_oct": 24.0,
                "max_slope_boost_db_per_oct": 36.0,
                "max_slope_cut_db_per_oct": 36.0,
            }
            return values.get(str(name), list(choices)[-1])

    cand = _suggest_auto_mode_candidate_optuna(
        {
            "auto_goal": "prefer bass",
            "filter_type": "Linear Phase",
            "enable_afdw": True,
            "bass_first_ai": True,
        },
        _FakeTrial(),
    )

    assert float(cand["max_boost"]) == 12.0
    assert bool(cand["enable_afdw"]) is False
    assert bool(cand["bass_first_ai"]) is False


def test_suggest_auto_mode_candidate_optuna_uses_filter_specific_space():
    class _FakeTrial:
        number = 1

        def suggest_float(self, name, low, high, step=None, log=False):
            _ = step, log
            values = {
                "fdw_cycles": 10.0,
                "tdc_strength": 50.0,
                "tdc_max_reduction_db": 9.0,
                "reg_strength": 30.0,
                "max_boost": 4.0,
                "mag_c_min": 24.0,
                "mag_c_max": 220.0,
                "trans_width": 100.0,
                "bass_first_mode_max_hz": 180.0,
                "conf_pull_max_hz": 180.0,
                "low_bass_cut_hz": 40.0,
                "mixed_freq": 180.0,
                "phase_limit_range_u": 1.0,
                "phase_limit": 320.0,
                "phase_limit_u": 0.5,
                "output_tilt_db_per_oct": 0.0,
            }
            return float(values.get(str(name), low))

        def suggest_categorical(self, name, choices):
            values = {
                "enable_afdw": True,
                "bass_first_ai": True,
                "tdc_slope_db_per_oct": 6.0,
                "max_slope_db_per_oct": 12.0,
                "max_slope_boost_db_per_oct": 0.0,
                "max_slope_cut_db_per_oct": 0.0,
            }
            return values.get(str(name), list(choices)[0])

    mixed = _suggest_auto_mode_candidate_optuna({"filter_type": "Mixed"}, _FakeTrial())
    linear = _suggest_auto_mode_candidate_optuna({"filter_type": "Linear Phase"}, _FakeTrial())
    asym = _suggest_auto_mode_candidate_optuna({"filter_type": "Asymmetric"}, _FakeTrial())
    minimum = _suggest_auto_mode_candidate_optuna({"filter_type": "Minimum Phase"}, _FakeTrial())

    assert "mixed_freq" in mixed
    assert "phase_limit" not in mixed
    assert "phase_limit" in linear
    assert "mixed_freq" not in linear
    assert "phase_limit" in asym
    assert "mixed_freq" not in asym
    assert "mixed_freq" not in minimum
    assert "phase_limit" not in minimum


def test_auto_optuna_effective_scope_is_filter_specific():
    asym_scope = _auto_optuna_effective_scope(
        {"filter_type": "Asymmetric", "auto_mode_optuna_constraints": False},
        "phase1",
        phase_kind="phase1",
    )
    mixed_scope = _auto_optuna_effective_scope(
        {"filter_type": "Mixed", "auto_mode_optuna_constraints": False},
        "phase1",
        phase_kind="phase1",
    )
    linear_scope = _auto_optuna_effective_scope(
        {"filter_type": "Linear Phase", "auto_mode_optuna_constraints": False},
        "phase1",
        phase_kind="phase1",
    )

    assert asym_scope == "phase1-filter-asym"
    assert mixed_scope == "phase1-filter-mixed"
    assert linear_scope == "phase1-filter-linear"
    assert len({asym_scope, mixed_scope, linear_scope}) == 3
    assert _auto_optuna_scope_for_filter({"filter_type": "Mixed"}, mixed_scope) == mixed_scope


def test_build_auto_mode_candidates_varies_mag_and_low_cut():
    cands = _build_auto_mode_candidates(
        {
            "auto_goal": "balanced",
            "filter_type": "Asymmetric",
            "mag_c_min": 24.0,
            "low_bass_cut_hz": 32.0,
        },
        n_trials=24,
        seed=4321,
    )
    mags = {
        round(float(c.get("mag_c_min")), 1)
        for c in cands
        if isinstance(c, dict) and ("mag_c_min" in c)
    }
    lows = {
        round(float(c.get("low_bass_cut_hz")), 1)
        for c in cands
        if isinstance(c, dict) and ("low_bass_cut_hz" in c)
    }
    assert len(mags) > 1
    assert len(lows) > 1
    assert all(15.0 <= v <= 70.0 for v in mags)
    assert all(18.0 <= v <= 55.0 for v in lows)


def test_build_auto_mode_candidates_optuna_optional_backend():
    cands = _build_auto_mode_candidates_optuna(
        {
            "auto_goal": "balanced",
            "filter_type": "Mixed",
            "mag_c_min": 24.0,
            "low_bass_cut_hz": 32.0,
        },
        n_trials=8,
        seed=1234,
    )
    if cands is None:
        assert cands is None
        return
    assert isinstance(cands, list)
    assert len(cands) == 8


def test_auto_optuna_completed_trial_accepts_full_seed_param_space():
    import optuna

    base_data = {"filter_type": "Mixed", "auto_target_mode": "adaptive"}
    params = _seed_auto_mode_candidate_optuna_params(
        base_data,
        {
            "fdw_cycles": 10.0,
            "max_boost": 4.5,
            "max_slope_boost_db_per_oct": 6.0,
            "max_slope_cut_db_per_oct": 6.0,
            "mixed_freq": 80.0,
            "output_tilt_db_per_oct": 0.5,
            "synth_tilt_frac": 0.31,
        },
    )

    trial = _auto_optuna_build_completed_trial(
        optuna,
        params=params,
        value=1.0,
        user_attrs={},
        base_data=base_data,
    )

    assert trial is not None
    assert float(trial.params["fdw_cycles"]) == 10.0
    assert float(trial.params["max_boost"]) == 4.5
    assert float(trial.params["max_slope_boost_db_per_oct"]) == 6.0
    assert float(trial.params["max_slope_cut_db_per_oct"]) == 6.0
    assert float(trial.params["output_tilt_db_per_oct"]) == 0.5
    assert float(trial.params["synth_tilt_frac"]) == 0.31


def test_auto_run_optuna_eval_loop_feeds_seed_trials_into_study():  # noqa: C901 - optuna integration test intentionally covers many control paths
    class _FakeTrial:
        def __init__(self, fixed=None):
            self.fixed = dict(fixed or {})

        def suggest_float(self, name, low, high):
            if name in self.fixed:
                return float(self.fixed[name])
            return float(low)

        def suggest_categorical(self, name, choices):
            if name in self.fixed:
                return self.fixed[name]
            return list(choices)[0]

    class _FakeStudy:
        def __init__(self):
            self.enqueued = []
            self.told = []

        def enqueue_trial(self, params):
            self.enqueued.append(dict(params or {}))

        def ask(self):
            if self.enqueued:
                return _FakeTrial(self.enqueued.pop(0))
            return _FakeTrial()

        def tell(self, trial, value=None, state=None):
            self.told.append({"trial": trial, "value": value, "state": state})

    class _FakeTrialState:
        FAIL = "fail"

    class _FakeOptuna:
        class samplers:
            class TPESampler:
                def __init__(self, seed=None, n_startup_trials=None, **kwargs):
                    self.seed = seed
                    self.n_startup_trials = n_startup_trials
                    self.kwargs = dict(kwargs or {})

        class trial:
            TrialState = _FakeTrialState

        @staticmethod
        def create_study(direction=None, sampler=None):
            return _FakeStudy()

    seen = []

    def _build_preset(trial):
        return {
            "max_boost": float(trial.suggest_float("max_boost", 3.0, 12.0)),
            "reg_strength": float(trial.suggest_float("reg_strength", 15.0, 45.0)),
        }

    def _eval_one(idx, preset):
        seen.append((int(idx), dict(preset or {})))
        return {
            "idx": int(idx),
            "ok": True,
            "metrics": {"rank_score": float(80.0 + idx)},
        }

    _auto_run_optuna_eval_loop(
        optuna_mod=_FakeOptuna,
        n_total=2,
        seed=123,
        startup_trials=2,
        base_data={},
        seed_presets=[{"max_boost": 6.7, "reg_strength": 22.5}],
        build_preset=_build_preset,
        eval_one=_eval_one,
        consume_one=lambda idx, out: False,
        objective_value=lambda out: float(dict(out.get("metrics", {}) or {}).get("rank_score", 0.0)),
        workers=1,
        seed_to_params=lambda preset: {
            "max_boost": float(preset.get("max_boost", 4.0)),
            "reg_strength": float(preset.get("reg_strength", 30.0)),
        },
    )

    assert seen
    assert seen[0][1]["max_boost"] == 6.7
    assert seen[0][1]["reg_strength"] == 22.5


def test_auto_run_optuna_eval_loop_uses_persistent_study_when_available():  # noqa: C901 - optuna integration test intentionally covers many control paths
    class _FakeTrial:
        def suggest_float(self, name, low, high):
            return float(low)

        def suggest_categorical(self, name, choices):
            return list(choices)[0]

        def set_user_attr(self, name, value):
            self.user_attrs = getattr(self, "user_attrs", {})
            self.user_attrs[name] = value

    class _FakeStudy:
        def __init__(self):
            self.told = []

        def ask(self):
            return _FakeTrial()

        def tell(self, trial, value=None, state=None):
            self.told.append({"trial": trial, "value": value, "state": state})

    calls = []

    class _FakeTrialState:
        FAIL = "fail"

    class _FakeOptuna:
        class samplers:
            class TPESampler:
                def __init__(self, seed=None, n_startup_trials=None, **kwargs):
                    self.seed = seed
                    self.n_startup_trials = n_startup_trials
                    self.kwargs = dict(kwargs or {})

        class storages:
            class JournalFileStorage:
                def __init__(self, path, lock_obj=None):
                    self.path = path
                    self.lock_obj = lock_obj

            class JournalFileOpenLock:
                def __init__(self, path):
                    self.path = path

            class JournalStorage:
                def __init__(self, backend):
                    self.backend = backend

        class trial:
            TrialState = _FakeTrialState

        @staticmethod
        def create_study(**kwargs):
            calls.append(dict(kwargs or {}))
            return _FakeStudy()

    _auto_run_optuna_eval_loop(
        optuna_mod=_FakeOptuna,
        n_total=1,
        seed=123,
        startup_trials=1,
        base_data={"auto_mode_optuna_persistent_study": True},
        seed_presets=[],
        build_preset=lambda trial: {"max_boost": float(trial.suggest_float("max_boost", 3.0, 12.0))},
        eval_one=lambda idx, preset: {"idx": int(idx), "ok": True, "metrics": {"rank_score": 90.0}},
        consume_one=lambda idx, out: False,
        objective_value=lambda out: float(dict(out.get("metrics", {}) or {}).get("rank_score", 0.0)),
        workers=1,
        study_name="camillafir-test-study",
    )

    assert calls
    assert calls[0]["study_name"] == "camillafir-test-study"
    assert calls[0]["load_if_exists"] is True
    assert "storage" in calls[0]


def test_auto_run_optuna_eval_loop_keeps_per_run_startup_for_existing_persistent_study():  # noqa: C901 - optuna integration test intentionally covers many control paths
    class _ExistingTrial:
        user_attrs = {}
        params = {}
        value = 90.0
        state = "complete"

    class _FakeTrial:
        def __init__(self, number):
            self.number = int(number)
            self.params = {}
            self.user_attrs = {}

        def suggest_float(self, name, low, high):
            self.params[str(name)] = float(low)
            return float(low)

        def set_user_attr(self, name, value):
            self.user_attrs[str(name)] = value

    class _FakeStudy:
        def __init__(self):
            self.existing = [_ExistingTrial() for _ in range(8)]
            self.asked = []
            self.told = []

        def get_trials(self, deepcopy=False):
            return [*self.existing, *self.told]

        def ask(self):
            trial = _FakeTrial(len(self.existing) + len(self.asked))
            self.asked.append(trial)
            return trial

        def tell(self, trial, value=None, state=None):
            trial.value = value
            trial.state = "complete" if state is None else state
            self.told.append(trial)

    class _FakeMedianPruner:
        def __init__(self, **kwargs):
            self.kwargs = dict(kwargs or {})

        def prune(self, study, trial):
            return True

    calls = []

    class _FakeTrialState:
        FAIL = "fail"
        PRUNED = "pruned"

    class _FakeOptuna:
        class samplers:
            class TPESampler:
                def __init__(self, seed=None, n_startup_trials=None, **kwargs):
                    self.seed = seed
                    self.n_startup_trials = n_startup_trials
                    self.kwargs = dict(kwargs or {})

        class pruners:
            MedianPruner = _FakeMedianPruner

        class storages:
            class JournalFileStorage:
                def __init__(self, path, lock_obj=None):
                    self.path = path
                    self.lock_obj = lock_obj

            class JournalFileOpenLock:
                def __init__(self, path):
                    self.path = path

            class JournalStorage:
                def __init__(self, backend):
                    self.backend = backend

        class trial:
            TrialState = _FakeTrialState

        @staticmethod
        def create_study(**kwargs):
            calls.append(dict(kwargs or {}))
            return _FakeStudy()

    _auto_run_optuna_eval_loop(
        optuna_mod=_FakeOptuna,
        n_total=8,
        seed=123,
        base_data={
            "auto_mode_optuna_persistent_study": True,
            "auto_mode_optuna_pruning_enabled": True,
            "auto_mode_optuna_pruning_n_startup": 8,
        },
        seed_presets=[],
        build_preset=lambda trial: {"max_boost": float(trial.suggest_float("max_boost", 3.0, 12.0))},
        eval_one=lambda idx, preset: {"idx": int(idx), "ok": True, "metrics": {"rank_score": 80.0}},
        consume_one=lambda idx, out: False,
        objective_value=lambda out: float(dict(out.get("metrics", {}) or {}).get("rank_score", 0.0)),
        workers=1,
        study_name="decaycore-existing-micro",
        phase_kind="micro",
    )

    assert calls
    pruner = calls[0]["pruner"]
    assert pruner.prune(None, SimpleNamespace(number=11)) is False
    assert pruner.prune(None, SimpleNamespace(number=12)) is True


def test_auto_run_optuna_eval_loop_skips_duplicate_trials_from_existing_study():  # noqa: C901 - optuna integration test intentionally covers many control paths
    class _FrozenTrial:
        def __init__(self, params, value):
            self.params = dict(params or {})
            self.value = float(value)
            self.state = "complete"
            self.user_attrs = {}

    class _FakeTrial:
        def __init__(self, fixed=None):
            self.fixed = dict(fixed or {})
            self.params = {}
            self.user_attrs = {}

        def suggest_float(self, name, low, high):
            value = float(self.fixed.get(name, low))
            self.params[str(name)] = float(value)
            return float(value)

        def suggest_categorical(self, name, choices):
            value = self.fixed.get(name, list(choices)[0])
            self.params[str(name)] = value
            return value

        def set_user_attr(self, name, value):
            self.user_attrs[str(name)] = value

    class _FakeStudy:
        def __init__(self):
            self.trials = [_FrozenTrial({"max_boost": 4.0}, 91.0)]
            self._ask_queue = [_FakeTrial({"max_boost": 4.0}), _FakeTrial({"max_boost": 6.0})]
            self.told = []

        def ask(self):
            return self._ask_queue.pop(0)

        def tell(self, trial, value=None, state=None):
            self.told.append({"trial": trial, "value": value, "state": state})

    class _FakeTrialState:
        FAIL = "fail"

    study = _FakeStudy()
    seen = []

    class _FakeOptuna:
        class samplers:
            class TPESampler:
                def __init__(self, seed=None, n_startup_trials=None, **kwargs):
                    self.seed = seed
                    self.n_startup_trials = n_startup_trials
                    self.kwargs = dict(kwargs or {})

        class trial:
            TrialState = _FakeTrialState

        @staticmethod
        def create_study(direction=None, sampler=None):
            return study

    _auto_run_optuna_eval_loop(
        optuna_mod=_FakeOptuna,
        n_total=1,
        seed=123,
        startup_trials=1,
        base_data={"auto_mode_optuna_persistent_study": False, "auto_mode_optuna_avoid_duplicates": True},
        seed_presets=[],
        build_preset=lambda trial: {"max_boost": float(trial.suggest_float("max_boost", 3.0, 12.0))},
        eval_one=lambda idx, preset: seen.append(dict(preset or {})) or {
            "idx": int(idx),
            "ok": True,
            "metrics": {"rank_score": 82.0},
        },
        consume_one=lambda idx, out: False,
        objective_value=lambda out: float(dict(out.get("metrics", {}) or {}).get("rank_score", 0.0)),
        workers=1,
        seed_to_params=lambda preset: {"max_boost": float(preset.get("max_boost", 0.0))},
        study_name="camillafir-test-dup",
    )

    assert seen == [{"max_boost": 6.0}]
    assert len(study.told) == 2
    assert float(study.told[0]["value"]) == 91.0
    assert float(study.told[1]["value"]) == 82.0


def test_auto_run_optuna_eval_loop_consumes_pruned_trials_in_parallel():  # noqa: C901 - optuna integration test intentionally covers many control paths
    class _TrialPruned(Exception):
        pass

    class _FakeTrial:
        def __init__(self):
            self.params = {}
            self.user_attrs = {}

        def suggest_float(self, name, low, high):
            value = float(low)
            self.params[str(name)] = float(value)
            return float(value)

        def set_user_attr(self, name, value):
            self.user_attrs[str(name)] = value

    class _FakeStudy:
        def __init__(self):
            self.told = []

        def ask(self):
            return _FakeTrial()

        def tell(self, trial, value=None, state=None):
            self.told.append({"trial": trial, "value": value, "state": state})

    class _FakeTrialState:
        FAIL = "fail"
        PRUNED = "pruned"

    study = _FakeStudy()
    consumed = []

    class _FakeOptuna:
        TrialPruned = _TrialPruned

        class samplers:
            class TPESampler:
                def __init__(self, seed=None, n_startup_trials=None, **kwargs):
                    self.seed = seed
                    self.n_startup_trials = n_startup_trials
                    self.kwargs = dict(kwargs or {})

        class trial:
            TrialState = _FakeTrialState

        @staticmethod
        def create_study(direction=None, sampler=None):
            return study

    def _eval_one(idx, preset):
        if int(idx) == 1:
            raise _TrialPruned()
        return {
            "idx": int(idx),
            "ok": True,
            "metrics": {"rank_score": float(80.0 + idx)},
        }

    _auto_run_optuna_eval_loop(
        optuna_mod=_FakeOptuna,
        n_total=2,
        seed=123,
        startup_trials=1,
        base_data={"auto_mode_optuna_avoid_duplicates": False},
        seed_presets=[],
        build_preset=lambda trial: {"max_boost": float(trial.suggest_float("max_boost", 3.0, 12.0))},
        eval_one=_eval_one,
        consume_one=lambda idx, out: consumed.append((int(idx), dict(out or {}))) or False,
        objective_value=lambda out: float(dict(out.get("metrics", {}) or {}).get("rank_score", 0.0)),
        workers=2,
    )

    assert consumed == [
        (1, {"idx": 1, "ok": False, "error": "optuna trial pruned", "pruned": True}),
        (2, {"idx": 2, "ok": True, "metrics": {"rank_score": 82.0}}),
    ]
    assert study.told[0]["state"] == _FakeTrialState.PRUNED
    assert float(study.told[1]["value"]) == 82.0


def test_cache_refine_pruned_trial_logs_info_instead_of_warning(caplog):
    def _auto_run_optuna_eval_loop(**kwargs):
        consume_one = kwargs["consume_one"]
        consume_one(
            1,
            {
                "idx": 1,
                "ok": False,
                "error": "optuna trial pruned",
                "pruned": True,
            },
        )
        return {}

    runtime = SimpleNamespace(
        cache_refine_max_rounds=1,
        auto_run_optuna_eval_loop=_auto_run_optuna_eval_loop,
        auto_trial_workers=lambda _base_data, _n_trials: 1,
        auto_optuna_study_name=lambda **_kwargs: "camillafir-test-cache-refine",
        auto_optuna_effective_scope=lambda *_args, **_kwargs: "phase3-micro-test",
    )
    context = _CacheRefineContext(
        params={
            "cache_base_data": {},
            "status_cb": None,
            "cfg": None,
            "goal": "balanced",
            "filter_key": "linear",
            "optuna_mod": object(),
            "seed": 123,
            "optuna_search_sig": "sig",
            "_cache_ready_preset": lambda preset, best_metrics=None: dict(preset or {}),
            "_materialize_preset_result": lambda preset, **kwargs: (None, {}, {}),
        }
    )
    progress = _CacheRefineProgress(
        cache_target_name="Harman10",
        micro_trials=1,
    )
    round_state = _CacheRefineRound(
        round_idx=1,
        round_start_metrics={},
        round_start_preset={},
        raw_scope="phase3-micro-test",
    )

    with caplog.at_level(logging.INFO, logger="DecayCore"):
        _run_optuna_cache_refine_round(
            context=context,
            progress=progress,
            round_state=round_state,
            runtime=runtime,
        )

    assert "Automatic mode cache refine round 1 trial 1/1 failed" not in caplog.text
    assert "Automatic mode cache refine round 1 trial 1/1 pruned: optuna trial pruned" in caplog.text


def test_target_preselect_seed_loads_cache_refine_seed_without_cache_entry():
    materialize_calls = []
    seed_preset = {
        "mixed_freq": 92.0,
        "tdc_strength": 55.0,
    }
    seed_metrics = {
        "rank_score": 88.0,
        "avg_score": 84.0,
    }
    runtime = SimpleNamespace(
        cache_refine_max_rounds=2,
        cache_refine_micro_trials=3,
    )
    context = _CacheRefineContext(
        params={
            "cache_base_data": {"hc_mode": "Harman10"},
            "measurements": {},
            "fs_v": 44100,
            "taps_v": 65536,
            "xos": [],
            "hpf": None,
            "status_cb": None,
            "cfg": SimpleNamespace(cache_enabled=False, optuna_persistent_study=False),
            "goal": "balanced",
            "filter_key": "mixed",
            "compat_version": "test-version",
            "optimizer_backend": "builtin",
            "optuna_mod": None,
            "optuna_search_sig": "sig",
            "seed_preset": dict(seed_preset),
            "seed_metrics": dict(seed_metrics),
            "seed_source": "target_preselect",
            "_cache_ready_preset": lambda preset, best_metrics=None: {"ready": True, **dict(preset or {})},
            "_materialize_preset_result": lambda preset, **kwargs: materialize_calls.append(dict(preset or {})),
            "runtime": runtime,
        }
    )

    seed = _load_exact_cache_seed(context=context)

    assert seed is not None
    assert seed.seed_source == "target_preselect"
    assert seed.cache_target_name == "Harman10"
    assert seed.best_preset["ready"] is True
    assert seed.best_preset["mixed_freq"] == 92.0
    assert seed.best_metrics == seed_metrics
    assert materialize_calls == []


def test_invalid_target_preselect_seed_falls_back_without_cache_entry():
    runtime = SimpleNamespace(
        cache_refine_max_rounds=2,
        cache_refine_micro_trials=3,
    )
    context = _CacheRefineContext(
        params={
            "cache_base_data": {"hc_mode": "Harman10"},
            "measurements": {},
            "fs_v": 44100,
            "taps_v": 65536,
            "xos": [],
            "hpf": None,
            "status_cb": None,
            "cfg": SimpleNamespace(cache_enabled=False, optuna_persistent_study=False),
            "goal": "balanced",
            "filter_key": "mixed",
            "compat_version": "test-version",
            "optimizer_backend": "builtin",
            "optuna_mod": None,
            "optuna_search_sig": "sig",
            "seed_preset": {"mixed_freq": 92.0},
            "seed_metrics": {},
            "seed_source": "target_preselect",
            "_cache_ready_preset": lambda preset, best_metrics=None: {},
            "_materialize_preset_result": lambda preset, **kwargs: (None, {}, {}),
            "runtime": runtime,
        }
    )

    assert _load_exact_cache_seed(context=context) is None


def test_cache_refine_runs_all_planned_phase4_rounds_without_early_stop():
    materialized = []

    def _build_micro_candidates(_base_data, center, *, n_trials, shrink):
        return [
            {**dict(center or {}), "candidate": idx}
            for idx in range(1, int(n_trials) + 1)
        ]

    runtime = SimpleNamespace(
        cache_refine_max_rounds=3,
        cache_refine_micro_trials=2,
        cache_refine_min_rank_improvement=99.0,
        auto_optuna_module_ready=lambda _optuna_mod: False,
        auto_optuna_scope_with_context=lambda scope, **_kwargs: scope,
        build_auto_mode_candidates_micro=_build_micro_candidates,
        auto_optuna_telemetry_rollup=lambda _tels: {},
        auto_optuna_telemetry_text=lambda _tel: "",
    )
    context = _CacheRefineContext(
        params={
            "cache_base_data": {},
            "measurements": {},
            "fs_v": 44100,
            "taps_v": 65536,
            "xos": [],
            "hpf": None,
            "status_cb": None,
            "cfg": SimpleNamespace(cache_enabled=False, optuna_persistent_study=False),
            "goal": "balanced",
            "filter_key": "mixed",
            "compat_version": "test-version",
            "optimizer_backend": "builtin",
            "optuna_mod": None,
            "seed": 123,
            "optuna_search_sig": "sig",
            "_cache_ready_preset": lambda preset, best_metrics=None: dict(preset or {}),
            "_materialize_preset_result": lambda preset, **_kwargs: (
                materialized.append(dict(preset or {})),
                {"rank_score": 80.0, "avg_score": 80.0},
                {},
            ),
            "runtime": runtime,
        }
    )
    seed_state = _CacheRefineSeed(
        cache_target_name="Harman10",
        best_preset={"candidate": 0},
        best_metrics={"rank_score": 80.0, "avg_score": 80.0},
        seed_source="exact_cache",
    )

    outcome = _run_cache_refine_rounds(context=context, seed_state=seed_state)

    assert len(materialized) == 2
    assert outcome.result["executed_micro_trials_total"] == 2
    assert outcome.result["stop_reason"] == "no_improvement"


def test_build_auto_mode_candidates_local_center_first_and_clamped():
    base = {
        "filter_type": "Mixed",
        "enable_tdc": True,
        "enable_afdw": True,
        "bass_first_ai": True,
        "mag_c_min": 26.0,
        "low_bass_cut_hz": 34.0,
    }
    center = {
        "mixed_freq": 92.0,
        "fdw_cycles": 10.0,
        "tdc_strength": 58.0,
        "tdc_max_reduction_db": 12.0,
        "tdc_slope_db_per_oct": 6.0,
        "reg_strength": 28.0,
        "max_boost": 4.0,
        "mag_c_max": 230.0,
        "trans_width": 105.0,
        "bass_first_mode_max_hz": 185.0,
        "conf_pull_max_hz": 180.0,
    }
    cands = _build_auto_mode_candidates_local(base, center, n_trials=9, seed=12345, shrink=0.35)
    assert len(cands) == 9
    first = dict(cands[0])
    assert float(first.get("mixed_freq", 0.0)) == 92.0
    assert float(first.get("mag_c_min", 0.0)) == 26.0
    assert float(first.get("low_bass_cut_hz", 0.0)) == 34.0

    for c in cands:
        assert 80.0 <= float(c.get("mixed_freq", 0.0)) <= 320.0
        assert 8.0 <= float(c.get("fdw_cycles", 0.0)) <= 16.0
        assert 20.0 <= float(c.get("tdc_strength", 0.0)) <= 75.0
        assert 0.0 <= float(c.get("tdc_max_reduction_db", 0.0)) <= 18.0
        assert 15.0 <= float(c.get("reg_strength", 0.0)) <= 45.0
        assert 3.0 <= float(c.get("max_boost", 0.0)) <= 12.0
        assert 15.0 <= float(c.get("mag_c_min", 0.0)) <= 70.0
        assert 170.0 <= float(c.get("mag_c_max", 0.0)) <= 300.0
        assert 70.0 <= float(c.get("trans_width", 0.0)) <= 150.0
        assert 120.0 <= float(c.get("bass_first_mode_max_hz", 0.0)) <= 220.0
        assert 80.0 <= float(c.get("conf_pull_max_hz", 0.0)) <= 220.0
        assert 18.0 <= float(c.get("low_bass_cut_hz", 0.0)) <= 55.0


def test_build_auto_mode_candidates_local_varies_mag_and_low_cut():
    base = {
        "filter_type": "Mixed",
        "enable_tdc": True,
        "enable_afdw": True,
        "bass_first_ai": True,
        "mag_c_min": 26.0,
        "low_bass_cut_hz": 34.0,
    }
    center = {
        "mixed_freq": 92.0,
        "fdw_cycles": 10.0,
        "tdc_strength": 58.0,
        "tdc_max_reduction_db": 12.0,
        "tdc_slope_db_per_oct": 6.0,
        "reg_strength": 28.0,
        "max_boost": 4.0,
        "mag_c_max": 230.0,
        "trans_width": 105.0,
        "bass_first_mode_max_hz": 185.0,
    }
    cands = _build_auto_mode_candidates_local(base, center, n_trials=16, seed=12345, shrink=0.60)
    mags = {round(float(c.get("mag_c_min", 0.0)), 1) for c in cands}
    lows = {round(float(c.get("low_bass_cut_hz", 0.0)), 1) for c in cands}
    assert len(mags) > 1
    assert len(lows) > 1


def test_auto_sample_mag_low_pair_allows_low_cut_below_mag_min():
    mag_c_min, low_bass_cut = _auto_sample_mag_low_pair(
        None,
        mag_center=45.0,
        low_center=22.0,
        mag_sigma=0.0,
        low_sigma=0.0,
    )

    assert float(mag_c_min) == 45.0
    assert float(low_bass_cut) == 22.0
    assert float(low_bass_cut) < float(mag_c_min)


def test_optuna_candidate_mag_c_max_floor_is_170_hz():
    class _LowTrial:
        def __init__(self):
            self.ranges = {}

        def suggest_float(self, name, low, high, **kwargs):
            _ = kwargs
            self.ranges[str(name)] = (float(low), float(high))
            return float(low)

        def suggest_categorical(self, name, choices):
            _ = name
            return list(choices)[0]

    trial = _LowTrial()
    candidate = _suggest_auto_mode_candidate_optuna(
        {
            "filter_type": "Linear",
            "enable_tdc": True,
            "enable_afdw": True,
            "bass_first_ai": True,
        },
        trial,
    )

    assert trial.ranges["mag_c_max"][0] == float(AUTO_MODE_MAG_C_MAX_MIN_HZ)
    assert float(candidate["mag_c_max"]) >= float(AUTO_MODE_MAG_C_MAX_MIN_HZ)


def test_optuna_seed_params_clamp_mag_c_max_to_adaptive_bounds():
    base = {
        "filter_type": "Linear",
        "harmonic_freq_hz_l": [20.0, 40.0],
        "harmonic_freq_hz_r": [20.0, 40.0],
        "measured_rt60_l": 0.32,
        "measured_rt60_r": 0.32,
    }
    expected_hi = float(_derive_adaptive_freq_bounds(base)["mag_c_max_hi"])

    params = _seed_auto_mode_candidate_optuna_params(
        base,
        {"mag_c_max": 300.0},
    )

    assert float(params["mag_c_max"]) == pytest.approx(expected_hi, abs=0.05)
    assert float(params["mag_c_max"]) < 300.0


def test_optuna_sanitize_enqueued_params_clamp_mag_c_max_to_adaptive_bounds():
    base = {
        "filter_type": "Linear",
        "harmonic_freq_hz_l": [20.0, 40.0],
        "harmonic_freq_hz_r": [20.0, 40.0],
        "measured_rt60_l": 0.32,
        "measured_rt60_r": 0.32,
    }
    expected_hi = float(_derive_adaptive_freq_bounds(base)["mag_c_max_hi"])

    params = _auto_optuna_sanitize_enqueued_params(
        {"mag_c_max": 300.0},
        base_data=base,
    )

    assert float(params["mag_c_max"]) == pytest.approx(expected_hi, abs=0.05)
    assert float(params["mag_c_max"]) < 300.0


def test_rank_combiner_focus_ripple_cache_is_initialized():
    assert _collect_rank_cached_focus_ripple({}, 20.0, 80.0) is None


def test_auto_select_best_scored_rejects_too_low_auto_mag_c_max():
    winner = _auto_select_best_scored(
        [
            {
                "preset": {"mag_c_max": 100.0},
                "metrics": {
                    "rank_score": 99.0,
                    "avg_score": 99.0,
                    "events_total": 0,
                },
            },
            {
                "preset": {"mag_c_max": 170.0},
                "metrics": {
                    "rank_score": 90.0,
                    "avg_score": 90.0,
                    "events_total": 0,
                },
            },
        ]
    )

    assert float(dict(winner or {}).get("preset", {}).get("mag_c_max", 0.0)) == 170.0


def test_optuna_local_and_micro_seed_params_use_static_unit_space():
    base = {
        "filter_type": "Linear",
        "mag_c_min": 26.0,
        "low_bass_cut_hz": 34.0,
        "phase_limit": 360.0,
    }
    center = {
        "fdw_cycles": 10.0,
        "tdc_strength": 58.0,
        "tdc_max_reduction_db": 12.0,
        "tdc_slope_db_per_oct": 6.0,
        "reg_strength": 28.0,
        "max_slope_db_per_oct": 12.0,
        "max_boost": 4.0,
        "mag_c_max": 230.0,
        "trans_width": 105.0,
        "bass_first_mode_max_hz": 185.0,
        "phase_limit": 350.0,
    }
    preset = {
        "fdw_cycles": 11.0,
        "tdc_strength": 60.0,
        "tdc_max_reduction_db": 14.0,
        "tdc_slope_db_per_oct": 8.0,
        "reg_strength": 31.0,
        "max_slope_db_per_oct": 14.0,
        "max_boost": 4.5,
        "mag_c_min": 27.5,
        "low_bass_cut_hz": 36.0,
        "mag_c_max": 240.0,
        "trans_width": 110.0,
        "bass_first_mode_max_hz": 190.0,
        "phase_limit": 340.0,
    }

    local_params = _seed_auto_mode_candidate_local_optuna_params(
        base,
        center,
        preset,
        shrink=0.6,
        optimize_mag_low=True,
    )
    micro_params = _seed_auto_mode_candidate_micro_optuna_params(
        base,
        center,
        preset,
        shrink=0.8,
    )

    assert local_params
    assert micro_params
    assert all(str(key).endswith("_u") for key in local_params.keys())
    assert all(str(key).endswith("_u") for key in micro_params.keys())
    assert "tdc_strength" not in local_params
    assert "tdc_strength" not in micro_params
    assert all(0.0 <= float(value) <= 1.0 for value in local_params.values())
    assert all(0.0 <= float(value) <= 1.0 for value in micro_params.values())


def test_auto_trial_workers_respects_auto_and_caps(monkeypatch):
    monkeypatch.setenv("CAMILLAFIR_AUTO_MODE_WORKERS", "")
    monkeypatch.setattr("decaycore.io.decaycore_automatic_mode.os.cpu_count", lambda: 8)
    assert _auto_trial_workers({"auto_mode_workers": 0}, 32) == 8
    assert _auto_trial_workers({"auto_mode_workers": 3}, 32) == 3
    assert _auto_trial_workers({"auto_mode_workers": 99}, 32) == 8


def test_auto_trial_workers_env_override_and_min_trials(monkeypatch):
    monkeypatch.setenv("CAMILLAFIR_AUTO_MODE_WORKERS", "2")
    monkeypatch.setattr("decaycore.io.decaycore_automatic_mode.os.cpu_count", lambda: 16)
    assert _auto_trial_workers({"auto_mode_workers": 8}, 32) == 2
    assert _auto_trial_workers({"auto_mode_workers": 8}, 4) == 1


def test_auto_exc_penalty_bins_from_dbg_prefers_pen_bins_field():
    assert _auto_exc_penalty_bins_from_dbg({"pen_bins": 1.25, "exc_bins": 99}) == 1.25
    assert abs(_auto_exc_penalty_bins_from_dbg({"exc_bins": 12}) - 1.2) < 1e-9


def test_auto_exc_zero_penalty_freq_from_stats_clips_to_limits():
    assert _auto_exc_zero_penalty_freq_hz_from_stats({"boost_candidate_min_hz": 10.0}) == 20.0
    assert _auto_exc_zero_penalty_freq_hz_from_stats({"boost_candidate_min_hz": 120.0}) == 80.0
    assert _auto_exc_zero_penalty_freq_hz_from_stats({"boost_candidate_min_hz": 42.5}) == 42.5


def test_auto_score_result_waives_exc_bins_using_zero_penalty_floor():
    st = {
        "exc_prot": True,
        "exc_freq": 24.0,
        "boost_candidate_bins_excprot": 6,
        "boost_candidate_min_hz": 18.2,
        "lf_boost_max_db": 0.0,
        "net_boost_peak_db": 0.0,
        "avg_confidence": 90.0,
        "freq_axis": [20.0, 30.0, 40.0],
        "measured_mags": [0.0, 0.0, 0.0],
        "target_mags": [0.0, 0.0, 0.0],
        "filter_mags": [0.0, 0.0, 0.0],
    }
    from decaycore.io.decaycore_automatic_mode import _auto_score_result

    out = _auto_score_result(
        SimpleNamespace(l_st=dict(st), r_st=dict(st)),
        auto_exc_freq_hz=24.0,
        base_data={},
    )
    assert out["auto_exc_zero_penalty_hz"] == 20.0
    assert out["exc_penalty_raw_total"] > 0.0
    assert out["exc_penalty_bins_raw"] > 0.0
    assert out["exc_penalty_bins_waived"] is True
    assert out["exc_penalty_raw"] == 0.0


def test_auto_score_result_waives_exc_bins_at_zero_penalty_floor():
    st = {
        "exc_prot": True,
        "exc_freq": 20.0,
        "boost_candidate_bins_excprot": 2,
        "boost_candidate_min_hz": 20.0,
        "lf_boost_max_db": 0.0,
        "net_boost_peak_db": 0.0,
        "avg_confidence": 90.0,
        "freq_axis": [20.0, 30.0, 40.0],
        "measured_mags": [0.0, 0.0, 0.0],
        "target_mags": [0.0, 0.0, 0.0],
        "filter_mags": [0.0, 0.0, 0.0],
    }
    from decaycore.io.decaycore_automatic_mode import _auto_score_result

    out = _auto_score_result(
        SimpleNamespace(l_st=dict(st), r_st=dict(st)),
        auto_exc_freq_hz=20.0,
        base_data={},
    )
    assert out["auto_exc_zero_penalty_hz"] == 20.0
    assert out["exc_penalty_bins_raw"] > 0.0
    assert out["exc_penalty_bins_waived"] is True
    assert out["exc_penalty_raw"] == 0.0


def test_auto_score_result_zeroes_exc_penalty_for_optuna_backend():
    st = {
        "exc_prot": True,
        "exc_freq": 24.0,
        "boost_candidate_bins_excprot": 6,
        "boost_candidate_min_hz": 18.2,
        "lf_boost_max_db": 0.0,
        "net_boost_peak_db": 0.0,
        "avg_confidence": 90.0,
        "freq_axis": [20.0, 30.0, 40.0],
        "measured_mags": [0.0, 0.0, 0.0],
        "target_mags": [0.0, 0.0, 0.0],
        "filter_mags": [0.0, 0.0, 0.0],
    }
    from decaycore.io.decaycore_automatic_mode import _auto_score_result

    out = _auto_score_result(
        SimpleNamespace(l_st=dict(st), r_st=dict(st)),
        auto_exc_freq_hz=24.0,
        base_data={"auto_mode_optuna": True},
    )

    assert out["auto_exc_zero_penalty_hz"] == 20.0
    assert out["exc_penalty"] == 0.0
    assert out["exc_penalty_raw_total"] == 0.0
    assert out["exc_penalty_bins_raw"] == 0.0
    assert out["exc_penalty_raw"] == 0.0
    assert out["rank_score_components"]["exc_penalty"] == 0.0


def test_auto_score_result_prefers_useful_phase_over_hf_phase_risk(monkeypatch):
    monkeypatch.setattr(
        "decaycore.auto_mode.scoring_metrics.calc_ai_summary_from_stats",
        lambda st: {"score": 84.0},
    )

    common = {
        "avg_confidence": 85.0,
        "freq_axis": [20.0, 40.0, 80.0, 120.0, 200.0, 320.0, 400.0, 500.0],
        "measured_mags": [0.0] * 8,
        "target_mags": [0.0] * 8,
        "filter_mags": [0.0] * 8,
        "ripple_rms": 0.03,
        "net_boost_peak_db": 0.0,
        "lf_boost_max_db": 0.0,
        "ir_pre_ringing_db": -48.0,
        "gd_grad_limiter_after_max_ms_per_oct": 8.0,
        "gd_abs_max_20_500_ms": 14.0,
    }
    better_phase = dict(
        common,
        phase_useful_lf_score=0.78,
        phase_useful_xo_score=0.52,
        phase_useful_audible_score=0.66,
        phase_risk_hf_score=0.05,
        phase_risk_spiky_score=0.04,
        phase_risk_clamp_score=0.03,
        phase_confidence_mean=0.80,
        phase_confidence_lf_mean=0.90,
        phase_confidence_xo_mean=0.72,
        phase_guard_scale_total=1.0,
    )
    worse_phase = dict(
        common,
        phase_useful_lf_score=0.30,
        phase_useful_xo_score=0.12,
        phase_useful_audible_score=0.22,
        phase_risk_hf_score=0.58,
        phase_risk_spiky_score=0.44,
        phase_risk_clamp_score=0.18,
        phase_confidence_mean=0.42,
        phase_confidence_lf_mean=0.55,
        phase_confidence_xo_mean=0.28,
        phase_guard_scale_total=0.72,
        ir_pre_ringing_db=-36.0,
        gd_grad_limiter_after_max_ms_per_oct=18.0,
        gd_abs_max_20_500_ms=30.0,
    )

    from decaycore.io.decaycore_automatic_mode import _auto_score_result

    better = _auto_score_result(
        SimpleNamespace(l_st=dict(better_phase), r_st=dict(better_phase)),
        auto_exc_freq_hz=None,
        base_data={"filter_type": "Linear Phase", "phase_limit": 420.0},
    )
    worse = _auto_score_result(
        SimpleNamespace(l_st=dict(worse_phase), r_st=dict(worse_phase)),
        auto_exc_freq_hz=None,
        base_data={"filter_type": "Linear Phase", "phase_limit": 420.0},
    )

    assert better["phase_benefit_bonus"] > worse["phase_benefit_bonus"]
    assert better["phase_risk_penalty"] < worse["phase_risk_penalty"]
    assert better["phase_net_score"] > worse["phase_net_score"]
    assert better["rank_score_components"]["rank_score_raw"] > worse["rank_score_components"]["rank_score_raw"]


def test_auto_phase_limit_prior_penalty_is_disabled():
    from decaycore.auto_mode.shared import _auto_phase_limit_prior_penalty

    assert _auto_phase_limit_prior_penalty(100.0, filter_key="linear") == 0.0
    assert _auto_phase_limit_prior_penalty(500.0, filter_key="asym") == 0.0


def test_auto_score_result_prefers_higher_fit_over_small_extra_boost(monkeypatch):
    monkeypatch.setattr(
        "decaycore.auto_mode.scoring_metrics.calc_ai_summary_from_stats",
        lambda st: {"score": float(st.get("score", 0.0))},
    )

    def _stats(score: float, net_boost_peak_db: float) -> dict:
        return {
            "score": float(score),
            "avg_confidence": 85.0,
            "freq_axis": [20.0, 40.0, 80.0, 120.0, 200.0, 320.0, 400.0, 500.0],
            "measured_mags": [0.0] * 8,
            "target_mags": [0.0] * 8,
            "filter_mags": [0.0] * 8,
            "ripple_rms": 0.03,
            "net_boost_peak_db": float(net_boost_peak_db),
            "lf_boost_max_db": float(net_boost_peak_db),
            "ir_pre_ringing_db": -48.0,
            "gd_grad_limiter_after_max_ms_per_oct": 8.0,
            "gd_abs_max_20_500_ms": 14.0,
        }

    from decaycore.auto_mode.scoring_metrics import _auto_score_result

    better_fit = _auto_score_result(
        SimpleNamespace(l_st=_stats(86.0, 6.3), r_st=_stats(86.0, 6.3)),
        auto_exc_freq_hz=None,
        base_data={"filter_type": "Asymmetric"},
    )
    lower_fit = _auto_score_result(
        SimpleNamespace(l_st=_stats(84.7, 5.6), r_st=_stats(84.7, 5.6)),
        auto_exc_freq_hz=None,
        base_data={"filter_type": "Asymmetric"},
    )

    assert better_fit["boost_penalty"] > lower_fit["boost_penalty"]
    assert better_fit["avg_score"] > lower_fit["avg_score"]
    assert better_fit["rank_score_components"]["rank_score_raw"] > lower_fit["rank_score_components"]["rank_score_raw"]


def test_auto_score_result_prefer_bass_uses_light_boost_penalty(monkeypatch):
    monkeypatch.setattr(
        "decaycore.auto_mode.scoring_metrics.calc_ai_summary_from_stats",
        lambda st: {"score": 84.0},
    )

    def _stats(net_boost_peak_db: float) -> dict:
        return {
            "avg_confidence": 85.0,
            "freq_axis": [20.0, 40.0, 80.0, 120.0, 200.0, 320.0, 400.0, 500.0],
            "measured_mags": [0.0] * 8,
            "target_mags": [0.0] * 8,
            "filter_mags": [0.0] * 8,
            "ripple_rms": 0.03,
            "net_boost_peak_db": float(net_boost_peak_db),
            "lf_boost_max_db": float(net_boost_peak_db),
            "ir_pre_ringing_db": -48.0,
            "gd_grad_limiter_after_max_ms_per_oct": 8.0,
            "gd_abs_max_20_500_ms": 14.0,
        }

    from decaycore.auto_mode.scoring_metrics import _auto_score_result

    prefer_bass_5 = _auto_score_result(
        SimpleNamespace(l_st=_stats(5.0), r_st=_stats(5.0)),
        auto_exc_freq_hz=None,
        base_data={"auto_goal": "prefer bass"},
    )
    prefer_bass_7 = _auto_score_result(
        SimpleNamespace(l_st=_stats(7.0), r_st=_stats(7.0)),
        auto_exc_freq_hz=None,
        base_data={"auto_goal": "prefer bass"},
    )
    balanced_7 = _auto_score_result(
        SimpleNamespace(l_st=_stats(7.0), r_st=_stats(7.0)),
        auto_exc_freq_hz=None,
        base_data={"auto_goal": "balanced"},
    )

    assert prefer_bass_5["boost_penalty"] == 0.0
    assert 0.0 < prefer_bass_7["boost_penalty"] < balanced_7["boost_penalty"]


def test_auto_score_result_prefer_bass_penalizes_small_bass_under_target(monkeypatch):
    monkeypatch.setattr(
        "decaycore.auto_mode.scoring_metrics.calc_ai_summary_from_stats",
        lambda st: {"score": 84.0},
    )

    def _stats(filter_gain_db: float) -> dict:
        return {
            "avg_confidence": 85.0,
            "freq_axis": [20.0, 40.0, 80.0, 120.0, 160.0, 200.0, 320.0, 500.0],
            "measured_mags": [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.0, 0.0],
            "target_mags": [0.0] * 8,
            "filter_mags": [filter_gain_db] * 6 + [0.0, 0.0],
            "realized_filter_mags": [filter_gain_db] * 6 + [0.0, 0.0],
            "confidence_mask": [1.0] * 8,
            "ripple_rms": 0.03,
            "net_boost_peak_db": max(0.0, filter_gain_db),
            "lf_boost_max_db": max(0.0, filter_gain_db),
            "ir_pre_ringing_db": -48.0,
            "gd_grad_limiter_after_max_ms_per_oct": 8.0,
            "gd_abs_max_20_500_ms": 14.0,
        }

    from decaycore.auto_mode.scoring_metrics import _auto_score_result

    no_boost = _auto_score_result(
        SimpleNamespace(l_st=_stats(0.0), r_st=_stats(0.0)),
        auto_exc_freq_hz=None,
        base_data={"auto_goal": "prefer bass"},
    )
    boosted = _auto_score_result(
        SimpleNamespace(l_st=_stats(1.0), r_st=_stats(1.0)),
        auto_exc_freq_hz=None,
        base_data={"auto_goal": "prefer bass"},
    )
    balanced_no_boost = _auto_score_result(
        SimpleNamespace(l_st=_stats(0.0), r_st=_stats(0.0)),
        auto_exc_freq_hz=None,
        base_data={"auto_goal": "balanced"},
    )

    assert no_boost["bass_under_target_penalty"] > 0.0
    assert boosted["bass_under_target_penalty"] == 0.0
    assert no_boost["target_tracking_penalty"] > balanced_no_boost["target_tracking_penalty"]
    assert boosted["rank_score_components"]["rank_score_raw"] > no_boost["rank_score_components"]["rank_score_raw"]


def test_build_auto_mode_candidates_prefer_bass_biases_max_boost_upward():
    candidates = _build_auto_mode_candidates(
        {"auto_goal": "prefer bass", "max_boost": 4.0},
        n_trials=8,
        seed=123,
    )

    assert candidates
    assert all(float(c.get("max_boost", 0.0)) >= 5.0 for c in candidates)
    assert float(candidates[0]["max_boost"]) >= 6.0


def test_auto_score_result_adds_separate_bass_feasibility_downgrade(monkeypatch):
    monkeypatch.setattr(
        "decaycore.auto_mode.scoring_metrics.calc_ai_summary_from_stats",
        lambda st: {"score": 84.0},
    )

    base_stats = {
        "avg_confidence": 85.0,
        "freq_axis": [20.0, 40.0, 80.0, 120.0, 200.0, 320.0, 400.0, 500.0],
        "measured_mags": [0.0] * 8,
        "target_mags": [0.0] * 8,
        "filter_mags": [0.0] * 8,
        "ripple_rms": 0.03,
        "net_boost_peak_db": 0.0,
        "lf_boost_max_db": 0.0,
        "ir_pre_ringing_db": -48.0,
        "gd_grad_limiter_after_max_ms_per_oct": 8.0,
        "gd_abs_max_20_500_ms": 14.0,
    }

    better = {
        "bass_cancellation_risk": 0.10,
        "bass_overlap_ripple": 6.0,
        "bass_sub_dominance": 6.0,
        "bass_null_severity": 0.2,
        "bass_xo_gd_rms_mismatch_ms": 8.0,
        "bass_overlap_ripple_delta_db": 1.0,
        "bass_sub_dominance_delta_db": 1.0,
        "bass_xo_gd_mismatch_delta_ms": 2.0,
        "bass_dominant_channel": "balanced",
        "bass_feasibility_class": "good",
        "bass_feasibility_reason": "Shared mono-sub integration meets balance and crossover guard targets.",
    }
    worse = {
        "bass_cancellation_risk": 0.10,
        "bass_overlap_ripple": 6.0,
        "bass_sub_dominance": 6.0,
        "bass_null_severity": 0.2,
        "bass_xo_gd_rms_mismatch_ms": 8.0,
        "bass_overlap_ripple_delta_db": 9.0,
        "bass_sub_dominance_delta_db": 11.0,
        "bass_xo_gd_mismatch_delta_ms": 24.0,
        "bass_dominant_channel": "right",
        "bass_feasibility_class": "infeasible",
        "bass_feasibility_reason": "Right channel remains limiting.",
    }

    from decaycore.auto_mode.scoring_metrics import _auto_score_result

    better_metrics = _auto_score_result(
        SimpleNamespace(l_st=dict(base_stats), r_st=dict(base_stats), metrics=dict(better)),
        auto_exc_freq_hz=None,
        base_data={"bass_integration_enable": True, "bass_integration_profile": "safe"},
    )
    worse_metrics = _auto_score_result(
        SimpleNamespace(l_st=dict(base_stats), r_st=dict(base_stats), metrics=dict(worse)),
        auto_exc_freq_hz=None,
        base_data={"bass_integration_enable": True, "bass_integration_profile": "safe"},
    )

    assert better_metrics["bass_feasibility_penalty"] == 0.0
    assert worse_metrics["bass_feasibility_penalty"] > 3.0
    assert worse_metrics["bass_integration_hard_gate_failed"] is True
    assert worse_metrics["bass_integration_hard_gate_reason"] == "Right channel remains limiting."
    assert "bass_integration_infeasible_hard_gate" in worse_metrics["hard_gate_failures"]
    assert worse_metrics["bass_integration_penalty"] > better_metrics["bass_integration_penalty"]
    assert worse_metrics["rank_score_components"]["bass_feasibility_penalty"] == worse_metrics["bass_feasibility_penalty"]
    assert worse_metrics["rank_score"] < better_metrics["rank_score"]


def test_auto_score_result_does_not_hard_gate_bass_feasibility_when_disabled(monkeypatch):
    monkeypatch.setattr(
        "decaycore.auto_mode.scoring_metrics.calc_ai_summary_from_stats",
        lambda st: {"score": 84.0},
    )

    base_stats = {
        "avg_confidence": 85.0,
        "freq_axis": [20.0, 40.0, 80.0, 120.0, 200.0, 320.0, 400.0, 500.0],
        "measured_mags": [0.0] * 8,
        "target_mags": [0.0] * 8,
        "filter_mags": [0.0] * 8,
        "ripple_rms": 0.03,
        "net_boost_peak_db": 0.0,
        "lf_boost_max_db": 0.0,
        "ir_pre_ringing_db": -48.0,
        "gd_grad_limiter_after_max_ms_per_oct": 8.0,
        "gd_abs_max_20_500_ms": 14.0,
    }
    result_metrics = {
        "bass_cancellation_risk": 0.10,
        "bass_overlap_ripple": 6.0,
        "bass_sub_dominance": 6.0,
        "bass_feasibility_class": "infeasible",
        "bass_feasibility_reason": "Ignored because Bass Integration is disabled.",
    }

    from decaycore.auto_mode.scoring_metrics import _auto_score_result

    metrics = _auto_score_result(
        SimpleNamespace(l_st=dict(base_stats), r_st=dict(base_stats), metrics=dict(result_metrics)),
        auto_exc_freq_hz=None,
        base_data={"bass_integration_enable": False, "bass_integration_profile": "safe"},
    )

    assert metrics["bass_integration_hard_gate_failed"] is False
    assert "bass_integration_infeasible_hard_gate" not in metrics["hard_gate_failures"]


def test_auto_score_result_keeps_overlap_extension_penalty_light_when_core_band_improves(monkeypatch):
    monkeypatch.setattr(
        "decaycore.auto_mode.scoring_metrics.calc_ai_summary_from_stats",
        lambda st: {"score": 84.0},
    )

    base_stats = {
        "avg_confidence": 85.0,
        "freq_axis": [20.0, 40.0, 80.0, 120.0, 200.0, 320.0, 400.0, 500.0],
        "measured_mags": [0.0] * 8,
        "target_mags": [0.0] * 8,
        "filter_mags": [0.0] * 8,
        "ripple_rms": 0.03,
        "net_boost_peak_db": 0.0,
        "lf_boost_max_db": 0.0,
        "ir_pre_ringing_db": -48.0,
        "gd_grad_limiter_after_max_ms_per_oct": 8.0,
        "gd_abs_max_20_500_ms": 14.0,
    }

    overlap_candidate = {
        "bass_cancellation_risk": 0.08,
        "bass_overlap_ripple": 3.8,
        "bass_sub_dominance": 3.5,
        "bass_null_severity": 0.2,
        "bass_xo_gd_rms_mismatch_ms": 4.0,
        "bass_overlap_extension_active": True,
        "bass_overlap_extension_flatness_db": 6.5,
        "bass_overlap_extension_cancellation_risk": 0.18,
        "bass_overlap_extension_peak_excess_db": 3.5,
        "bass_overlap_extension_sub_dominance_db": 10.0,
        "bass_overlap_ripple_delta_db": 1.0,
        "bass_sub_dominance_delta_db": 1.0,
        "bass_xo_gd_mismatch_delta_ms": 2.0,
        "bass_dominant_channel": "balanced",
        "bass_feasibility_class": "good",
        "bass_feasibility_reason": "Shared mono-sub integration meets balance and crossover guard targets.",
    }
    no_overlap_worse = {
        "bass_cancellation_risk": 0.12,
        "bass_overlap_ripple": 6.8,
        "bass_sub_dominance": 6.0,
        "bass_null_severity": 0.2,
        "bass_xo_gd_rms_mismatch_ms": 5.0,
        "bass_overlap_extension_active": False,
        "bass_overlap_extension_flatness_db": float("nan"),
        "bass_overlap_extension_cancellation_risk": float("nan"),
        "bass_overlap_extension_peak_excess_db": float("nan"),
        "bass_overlap_extension_sub_dominance_db": float("nan"),
        "bass_overlap_ripple_delta_db": 1.0,
        "bass_sub_dominance_delta_db": 1.0,
        "bass_xo_gd_mismatch_delta_ms": 2.0,
        "bass_dominant_channel": "balanced",
        "bass_feasibility_class": "good",
        "bass_feasibility_reason": "Shared mono-sub integration meets balance and crossover guard targets.",
    }

    from decaycore.auto_mode.scoring_metrics import _auto_score_result

    overlap_metrics = _auto_score_result(
        SimpleNamespace(l_st=dict(base_stats), r_st=dict(base_stats), metrics=dict(overlap_candidate)),
        auto_exc_freq_hz=None,
        base_data={"bass_integration_enable": True, "bass_integration_profile": "safe"},
    )
    no_overlap_metrics = _auto_score_result(
        SimpleNamespace(l_st=dict(base_stats), r_st=dict(base_stats), metrics=dict(no_overlap_worse)),
        auto_exc_freq_hz=None,
        base_data={"bass_integration_enable": True, "bass_integration_profile": "safe"},
    )

    assert overlap_metrics["bass_overlap_extension_active"] is True
    assert overlap_metrics["bass_integration_penalty"] < no_overlap_metrics["bass_integration_penalty"]
    assert (no_overlap_metrics["bass_integration_penalty"] - overlap_metrics["bass_integration_penalty"]) > 0.5


def test_finalize_search_result_keeps_rank_best_winner_when_pareto_differs(monkeypatch):
    from decaycore.auto_mode.orchestrator_finalize import finalize_search_result

    rank_best_metrics = {
        "rank_score": 91.8,
        "rank_score_official": 91.8,
        "avg_score": 84.3,
        "max_net_boost_db": 1.8,
        "events_total": 0,
        "rank_score_components": {"rank_score": 91.8},
    }
    pareto_metrics = {
        "rank_score": 88.5,
        "rank_score_official": 88.5,
        "avg_score": 84.35,
        "max_net_boost_db": 1.5,
        "events_total": 0,
        "ir_pre_post_energy_ratio_max": 0.09,
        "mode_ripple_db": 0.78,
        "rank_score_components": {"rank_score": 88.5},
    }
    rank_best_item = {"preset": {"preset_id": "rank-best"}, "metrics": dict(rank_best_metrics)}
    pareto_item = {"preset": {"preset_id": "pareto"}, "metrics": dict(pareto_metrics)}
    search_state = SimpleNamespace(
        phase2_pool=[dict(rank_best_item), dict(pareto_item)],
        scored=[dict(rank_best_item), dict(pareto_item)],
        best_metrics=dict(rank_best_metrics),
        best_preset={"preset_id": "rank-best"},
        best_result=None,
        winner_explanation={"phase_label": "phase 1"},
    )
    statuses = []

    def _noop_polish(**kwargs):
        return kwargs["best_preset"], kwargs["best_metrics"], False, {}

    def _noop_stereo(**kwargs):
        return kwargs["best_preset"], kwargs["best_metrics"], False, {}

    def _select_best(items, *, goal=None):
        pool = list(items or [])
        if pool and any("_auto_select_kind" in dict(it or {}) for it in pool):
            return next(it for it in pool if dict(it.get("preset", {}) or {}).get("preset_id") == "pareto")
        return max(pool, key=lambda it: float(dict(it.get("metrics", {}) or {}).get("rank_score", float("-inf"))))

    monkeypatch.setattr("decaycore.auto_mode.orchestrator_finalize_run._auto_phase2_pareto_front", lambda items: list(items))
    monkeypatch.setattr("decaycore.auto_mode.orchestrator_finalize_run._auto_select_best_scored", _select_best)
    monkeypatch.setattr("decaycore.auto_mode.orchestrator_finalize_polish.apply_phase_limit_winner_polish", _noop_polish)
    monkeypatch.setattr("decaycore.auto_mode.orchestrator_finalize_polish.apply_mag_c_min_winner_polish", _noop_polish)
    monkeypatch.setattr("decaycore.auto_mode.orchestrator_finalize_polish.apply_low_bass_cut_winner_polish", _noop_polish)
    monkeypatch.setattr("decaycore.auto_mode.orchestrator_finalize_polish.apply_hpf_winner_polish", _noop_polish)
    monkeypatch.setattr("decaycore.auto_mode.orchestrator_finalize_polish.apply_excess_phase_strength_winner_polish", _noop_polish)
    monkeypatch.setattr("decaycore.auto_mode.orchestrator_finalize_polish.apply_stereo_policy_refine", _noop_stereo)

    result = finalize_search_result(
        search_base_data={"filter_type": "Asymmetric"},
        cache_base_data={"filter_type": "Asymmetric"},
        measurements={},
        fs_v=44100,
        taps_v=4096,
        xos=[],
        hpf=None,
        hc_f=None,
        hc_m=None,
        pin_obj=None,
        cfg=SimpleNamespace(
            phase2_pareto_rank_window=5.0,
            phase2_pareto_pool_max=8,
            phase2_hard_gate_enabled=False,
            phase2_hard_gate_min_keep=8,
            phase2_hard_gate_keep_event_fraction=0.7,
            phase2_hard_gate_keep_ripple_fraction=0.75,
            phase2_hard_gate_fallback_to_rank=True,
            phase2_pareto_pool_min=2,
            phase2_pareto_acoustic_drop=0.35,
            cache_enabled=False,
        ),
        goal="balanced",
        rank_basis="rank_score",
        filter_key="asym",
        compat_version="test",
        optimizer_backend="builtin",
        status_cb=statuses.append,
        optuna_mod=None,
        optuna_search_sig="sig",
        seed=123,
        search_state=search_state,
        winner_target_name="Adaptive",
        phase1_ok=2,
        phase2_ok=2,
        phase1_tried=2,
        phase2_tried=2,
        phase1_plateau_hit=False,
        phase2_plateau_hit=False,
        phase1_optuna_tel={},
        phase2_local_optuna_tels=[],
        phase3_micro_optuna_tel={},
        phase2_rollup_tel={},
        _cache_ready_preset=lambda preset, best_metrics=None: dict(preset or {}),
        _materialize_preset_result=lambda preset, **kwargs: (
            SimpleNamespace(),
            dict(rank_best_metrics if dict(preset or {}).get("preset_id") == "rank-best" else pareto_metrics),
            dict(preset or {}),
        ),
        _maybe_apply_residual_tiebreak=lambda **kwargs: (
            kwargs["best_preset"],
            kwargs["best_metrics"],
            False,
        ),
        runtime=SimpleNamespace(
            phase_limit_winner_polish_enabled=False,
            phase_limit_winner_polish_offsets_hz=(),
            mag_c_min_winner_polish_enabled=False,
            mag_c_min_winner_polish_step_hz=2.0,
            mag_c_min_winner_polish_max_down_hz=40.0,
            mag_c_min_winner_polish_max_up_hz=4.0,
            hpf_winner_polish_enabled=False,
            excess_phase_strength_winner_polish_enabled=False,
            excess_phase_strength_winner_polish_step=0.05,
            excess_phase_strength_winner_polish_max_delta=0.15,
        ),
    )

    assert dict(result.get("best_preset", {}) or {}).get("preset_id") == "rank-best"
    assert dict(result.get("best_applied_preset", {}) or {}).get("preset_id") == "rank-best"
    assert float(result.get("best_metrics", {}).get("rank_score", 0.0)) == 91.8
    assert any("phase 2 pareto comparison" in str(msg) for msg in statuses)
    assert not any("phase 2 pareto selected winner" in str(msg) for msg in statuses)


def test_finalize_search_result_applies_residual_peak_safety_override(monkeypatch):
    from decaycore.auto_mode.orchestrator_finalize import finalize_search_result

    unsafe_metrics = {
        "rank_score": 95.0,
        "rank_score_official": 95.0,
        "avg_score": 90.0,
        "worst_residual_peak_db": 7.0,
        "residual_peak_hard_gate_db": 6.0,
        "hard_gate_failed": True,
        "hard_gate_failures": ["residual_peak_hard_gate"],
        "rank_score_components": {"rank_score": 95.0},
    }
    safe_metrics = {
        "rank_score": 90.0,
        "rank_score_official": 90.0,
        "avg_score": 89.95,
        "worst_residual_peak_db": 2.0,
        "residual_peak_hard_gate_db": 6.0,
        "hard_gate_failed": False,
        "hard_gate_failures": [],
        "rank_score_components": {"rank_score": 90.0},
    }
    unsafe_item = {"preset": {"preset_id": "unsafe"}, "metrics": dict(unsafe_metrics)}
    safe_item = {"preset": {"preset_id": "safe"}, "metrics": dict(safe_metrics)}
    search_state = SimpleNamespace(
        phase2_pool=[dict(unsafe_item), dict(safe_item)],
        scored=[dict(unsafe_item), dict(safe_item)],
        best_metrics=dict(unsafe_metrics),
        best_preset={"preset_id": "unsafe"},
        best_result=SimpleNamespace(),
        winner_explanation={},
    )

    def _noop_polish(**kwargs):
        return kwargs["best_preset"], kwargs["best_metrics"], False, {}

    def _noop_stereo(**kwargs):
        return kwargs["best_preset"], kwargs["best_metrics"], False, {}

    monkeypatch.setattr("decaycore.auto_mode.orchestrator_finalize_polish.apply_phase_limit_winner_polish", _noop_polish)
    monkeypatch.setattr("decaycore.auto_mode.orchestrator_finalize_polish.apply_mag_c_min_winner_polish", _noop_polish)
    monkeypatch.setattr("decaycore.auto_mode.orchestrator_finalize_polish.apply_low_bass_cut_winner_polish", _noop_polish)
    monkeypatch.setattr("decaycore.auto_mode.orchestrator_finalize_polish.apply_hpf_winner_polish", _noop_polish)
    monkeypatch.setattr("decaycore.auto_mode.orchestrator_finalize_polish.apply_excess_phase_strength_winner_polish", _noop_polish)
    monkeypatch.setattr("decaycore.auto_mode.orchestrator_finalize_polish.apply_residual_peak_winner_polish", _noop_polish)
    monkeypatch.setattr("decaycore.auto_mode.orchestrator_finalize_polish.apply_stereo_policy_refine", _noop_stereo)

    def _materialize(preset, **kwargs):
        preset_id = dict(preset or {}).get("preset_id")
        metrics = dict(safe_metrics if preset_id == "safe" else unsafe_metrics)
        return SimpleNamespace(), metrics, dict(preset or {})

    result = finalize_search_result(
        search_base_data={"filter_type": "Asymmetric"},
        cache_base_data={"filter_type": "Asymmetric"},
        measurements={},
        fs_v=44100,
        taps_v=4096,
        xos=[],
        hpf=None,
        hc_f=None,
        hc_m=None,
        pin_obj=None,
        cfg=SimpleNamespace(
            phase2_pareto_rank_window=5.0,
            phase2_pareto_pool_max=8,
            phase2_hard_gate_enabled=False,
            phase2_hard_gate_min_keep=8,
            phase2_hard_gate_keep_event_fraction=0.7,
            phase2_hard_gate_keep_ripple_fraction=0.75,
            phase2_hard_gate_fallback_to_rank=True,
            phase2_pareto_pool_min=99,
            phase2_pareto_acoustic_drop=0.35,
            max_avg_score_loss_for_safety_override=0.10,
            cache_enabled=False,
        ),
        goal="balanced",
        rank_basis="rank_score",
        filter_key="asym",
        compat_version="test",
        optimizer_backend="builtin",
        status_cb=lambda _msg: None,
        optuna_mod=None,
        optuna_search_sig="sig",
        seed=123,
        search_state=search_state,
        winner_target_name="Adaptive",
        phase1_ok=2,
        phase2_ok=2,
        phase1_tried=2,
        phase2_tried=2,
        phase1_plateau_hit=False,
        phase2_plateau_hit=False,
        phase1_optuna_tel={},
        phase2_local_optuna_tels=[],
        phase3_micro_optuna_tel={},
        phase2_rollup_tel={},
        _cache_ready_preset=lambda preset, best_metrics=None: dict(preset or {}),
        _materialize_preset_result=_materialize,
        _maybe_apply_residual_tiebreak=lambda **kwargs: (
            kwargs["best_preset"],
            kwargs["best_metrics"],
            False,
        ),
        runtime=SimpleNamespace(
            phase_limit_winner_polish_enabled=False,
            phase_limit_winner_polish_offsets_hz=(),
            mag_c_min_winner_polish_enabled=False,
            mag_c_min_winner_polish_step_hz=2.0,
            mag_c_min_winner_polish_max_down_hz=40.0,
            mag_c_min_winner_polish_max_up_hz=4.0,
            hpf_winner_polish_enabled=False,
            excess_phase_strength_winner_polish_enabled=False,
            excess_phase_strength_winner_polish_step=0.05,
            excess_phase_strength_winner_polish_max_delta=0.15,
        ),
    )

    assert dict(result.get("best_preset", {}) or {}).get("preset_id") == "safe"
    assert result.get("best_metrics", {}).get("hard_gate_failed") is False
    assert result.get("residual_peak_safety_override", {}).get("applied") is True
    debug = dict(result.get("auto_mode_debug", {}) or {})
    assert "cache_schema_version" in debug
    assert "cache_stats" in debug
    assert "winning_score_breakdown" in debug
    assert debug.get("residual_peak_safety_override", {}).get("applied") is True
    winner = dict(result.get("winner", {}) or {})
    assert winner.get("rank_score_components") == {"rank_score": 90.0}
