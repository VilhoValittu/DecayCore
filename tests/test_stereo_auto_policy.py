from types import SimpleNamespace

from decaycore.config.decaycore_config import load_config
from decaycore.config.decaycore_pipeline import build_filter_config
from decaycore.config.models import (
    FilterConfig,
    ResolvedChannelAutoPolicy,
    StereoAutoPolicyConfig,
    StereoResolvedAutoPolicies,
)
from decaycore.dsp.decaycore_dsp import _maybe_per_channel_cfg
from decaycore.dsp.quality_metrics import worst_channel_relief_db
from decaycore.engine_build import build_config
from decaycore.auto_mode.scoring_metrics import _auto_score_result
from decaycore.auto_mode.stereo_policy_refine import apply_stereo_policy_refine
from decaycore.auto_mode.materialize import AutoModeMaterializeContext, build_materialize_helpers
from decaycore.auto_mode.orchestrator_finalize import finalize_search_result, _resolve_winner_auto_exc_hz
from decaycore.ui.export_summary_text import _append_auto_stereo_policy_summary


def test_build_filter_config_defaults_stereo_auto_policy_off():
    data = load_config()
    data["enable_channel_specific_auto_policy"] = False

    cfg = build_filter_config(
        FilterConfig_cls=FilterConfig,
        fs_v=44100,
        taps_v=65536,
        data=data,
        xos=[],
        hpf=None,
        hc_f=None,
        hc_m=None,
        pin=None,
    )

    assert bool(cfg.stereo_auto_policy.enable_channel_specific_auto_policy) is False
    assert float(cfg.stereo_auto_policy.channel_specific_policy_max_hz) == 220.0


def test_resolve_winner_auto_exc_hz_prefers_current_seed_for_builtin():
    resolved = _resolve_winner_auto_exc_hz(
        optimizer_backend="builtin",
        materialized_best_preset={"_auto_exc_freq_hz": 80.0, "exc_freq": 80.0},
        search_base_data={"_auto_exc_seed_freq_hz": 31.5, "_auto_exc_freq_hz": 31.5, "exc_freq": 31.5},
        best_metrics={"auto_exc_zero_penalty_hz": 80.0},
    )

    assert resolved == 31.5


def test_cache_ready_preset_prefers_current_excursion_seed():
    _cache_ready_preset, _, _, _ = build_materialize_helpers(
        AutoModeMaterializeContext(
            cfg=SimpleNamespace(exc_min_hz=20.0, exc_max_hz=80.0),
            cache_base_data={"_auto_exc_seed_freq_hz": 31.5, "_auto_exc_freq_hz": 31.5, "exc_freq": 31.5},
            measurements={},
            fs_v=44100,
            taps_v=2048,
            xos=[],
            hpf=None,
            hc_f=None,
            hc_m=None,
            pin_obj=None,
            filter_key="minimum",
            max_safe_boost=8.0,
            goal="balanced",
            status_cb=None,
            exact_cached_metrics_getter=lambda: None,
            auto_score_result_fn=lambda *args, **kwargs: {},
            auto_optuna_jsonable_fn=lambda value: value,
            auto_rank_key_fn=lambda metrics: 0.0,
            auto_is_better_refine_fn=lambda *args, **kwargs: False,
            build_config_fn=lambda *args, **kwargs: SimpleNamespace(),
            run_pipeline_fn=lambda *args, **kwargs: SimpleNamespace(metrics={}),
            summarize_run_fn=lambda result: "",
            preset_transient_keys=(),
            residual_tiebreak_enabled=False,
            residual_top_k=0,
            residual_rank_eps=0.0,
        )
    )

    ready = _cache_ready_preset(
        {"preset_id": "winner", "_auto_exc_freq_hz": 80.0, "exc_freq": 80.0},
        best_metrics={"auto_exc_zero_penalty_hz": 80.0},
    )

    assert ready["_auto_exc_freq_hz"] == 31.5
    assert ready["best_auto_exc_freq_hz"] == 31.5
    assert ready["exc_freq"] == 31.5


def test_build_filter_config_reads_stereo_auto_policy_controls():
    data = load_config()
    data.update(
        {
            "mode": "ADVANCED",
            "camillafir_automatic_mode": False,
            "enable_channel_specific_auto_policy": True,
            "channel_specific_policy_max_hz": 185.0,
        }
    )

    cfg = build_filter_config(
        FilterConfig_cls=FilterConfig,
        fs_v=44100,
        taps_v=65536,
        data=data,
        xos=[],
        hpf=None,
        hc_f=None,
        hc_m=None,
        pin=None,
    )

    assert bool(cfg.stereo_auto_policy.enable_channel_specific_auto_policy) is True
    assert float(cfg.stereo_auto_policy.channel_specific_policy_max_hz) == 185.0


def test_build_config_parses_resolved_stereo_auto_policy_overlay():
    data = load_config()
    data.update(
        {
            "mode": "AUTO",
            "camillafir_automatic_mode": True,
            "_stereo_resolved_auto_policies": {
                "split_hz": 220.0,
                "shared": {"conf_pull_floor": 0.05, "tdc_strength": 50.0},
                "left": {"conf_pull_floor": 0.20},
                "right": {"tdc_strength": 62.0},
            },
        }
    )

    cfg = build_config(data)

    assert cfg.stereo_resolved_auto_policies is not None
    assert float(cfg.stereo_resolved_auto_policies.left.conf_pull_floor) == 0.20
    assert float(cfg.stereo_resolved_auto_policies.right.tdc_strength) == 62.0


def test_maybe_per_channel_cfg_applies_stereo_policy_overlay():
    cfg = FilterConfig(
        conf_pull_floor=0.05,
        tdc_strength=50.0,
        tdc_max_reduction_db=9.0,
        bass_first_mode_max_hz=180.0,
        low_bass_cut_strength=0.0,
        stereo_resolved_auto_policies=StereoResolvedAutoPolicies(
            split_hz=220.0,
            shared=ResolvedChannelAutoPolicy(
                conf_pull_floor=0.05,
                tdc_strength=50.0,
                tdc_max_reduction_db=9.0,
                bass_first_mode_max_hz=180.0,
                low_bass_cut_strength=0.0,
            ),
            left=ResolvedChannelAutoPolicy(
                conf_pull_floor=0.20,
                tdc_strength=62.0,
                tdc_max_reduction_db=10.5,
                bass_first_mode_max_hz=205.0,
                low_bass_cut_strength=0.25,
            ),
            right=ResolvedChannelAutoPolicy(),
        ),
    )

    left_cfg = _maybe_per_channel_cfg(cfg, "l")
    right_cfg = _maybe_per_channel_cfg(cfg, "r")

    assert left_cfg is not cfg
    assert float(left_cfg.conf_pull_floor) == 0.20
    assert float(left_cfg.tdc_strength) == 62.0
    assert float(left_cfg.tdc_max_reduction_db) == 10.5
    assert float(left_cfg.bass_first_mode_max_hz) == 205.0
    assert float(left_cfg.low_bass_cut_strength) == 0.25
    assert float(right_cfg.conf_pull_floor) == 0.05
    assert float(cfg.conf_pull_floor) == 0.05


def test_auto_stereo_policy_summary_reports_off_state():
    summary = _append_auto_stereo_policy_summary(
        "",
        {
            "enable_channel_specific_auto_policy": False,
        },
        {},
    )

    assert "AUTO Stereo Policy: OFF" in summary


def test_stereo_auto_policy_default_min_relief_is_0_015_db():
    cfg = StereoAutoPolicyConfig.from_dict({})

    assert float(cfg.min_worst_channel_improvement_db) == 0.015
    assert int(cfg.channel_specific_refine_trials) == 32
    assert float(cfg.stereo_coherence_weight) == 0.6
    assert float(cfg.phantom_center_stability_weight) == 0.6
    assert float(cfg.policy_divergence_weight) == 0.25
    assert float(cfg.asymmetry_budget_weight) == 0.5
    assert float(cfg.max_lr_predicted_delta_db_below_split) == 4.0
    assert float(cfg.max_lr_predicted_delta_db_above_split) == 1.5
    assert float(cfg.max_policy_divergence_score) == 1.4
    assert float(cfg.shared_preference_bias) == 0.0


def test_worst_channel_relief_tracks_band_boost_reduction():
    freq_axis = [20.0, 30.0, 40.0, 60.0, 80.0, 120.0, 160.0, 220.0]

    def _stats(boost_db: float) -> dict:
        return {
            "freq_axis": list(freq_axis),
            "measured_mags": [-8.0] * len(freq_axis),
            "target_mags": [0.0] * len(freq_axis),
            "filter_mags": [float(boost_db)] * len(freq_axis),
            "predicted_filter_mags": [float(boost_db)] * len(freq_axis),
            "confidence_mask": [1.0] * len(freq_axis),
        }

    relief = worst_channel_relief_db(
        _stats(6.0),
        _stats(1.0),
        _stats(3.0),
        _stats(1.0),
        lo_hz=20.0,
        hi_hz=220.0,
    )

    assert float(relief) == 3.0


def test_auto_stereo_policy_summary_reports_applied_state():
    summary = _append_auto_stereo_policy_summary(
        "",
        {
            "enable_channel_specific_auto_policy": True,
            "channel_specific_policy_max_hz": 220.0,
        },
        {
            "stereo_policy_refine": {
                "state": "applied",
                "gate_passed": True,
                "split_hz": 220.0,
                "worse_side": "left",
                "shared_rank": 89.1,
                "refined_rank": 89.8,
                "worst_channel_relief_db": 0.18,
                "stereo_coherence_penalty": 0.12,
                "phantom_center_stability_penalty": 0.04,
                "policy_divergence_penalty": 0.08,
                "resolved": {
                    "shared": {
                        "conf_pull_floor": 0.05,
                        "tdc_strength": 50.0,
                    },
                    "left": {
                        "conf_pull_floor": 0.18,
                        "tdc_strength": 62.0,
                    },
                    "right": {},
                },
            }
        },
    )

    assert "AUTO Stereo Policy: shared target + per-channel LF restraint refinement" in summary
    assert "Refined stereo-safety gate: PASSED" in summary
    assert "Protected / limiting side: LEFT" in summary
    assert "Policy deltas:" in summary
    assert "Minimum required worst-channel relief: +0.050 dB" not in summary


def test_auto_stereo_policy_summary_hides_internal_shared_fallback_code():
    summary = _append_auto_stereo_policy_summary(
        "",
        {
            "enable_channel_specific_auto_policy": True,
            "channel_specific_policy_max_hz": 220.0,
        },
        {},
    )

    assert "Refined stereo-safety gate: FALLBACK" in summary
    assert "Fallback reason: shared_fallback" not in summary
    assert "Fallback reason: shared winner kept; stereo refine metadata unavailable" in summary


def test_auto_stereo_policy_summary_reports_relief_threshold_on_fallback():
    summary = _append_auto_stereo_policy_summary(
        "",
        {
            "enable_channel_specific_auto_policy": True,
            "channel_specific_policy_max_hz": 220.0,
        },
        {
            "stereo_policy_refine": {
                "state": "shared_fallback",
                "gate_passed": False,
                "gate_reason": "worst_channel_gain_too_small",
                "shared_rank": 92.214,
                "refined_rank": 92.214,
                "worst_channel_relief_db": 0.021,
                "min_worst_channel_improvement_db": 0.05,
            }
        },
    )

    assert "Fallback reason: shared winner kept; worst-channel LF improvement stayed below the minimum threshold" in summary
    assert "Worst-channel relief: +0.021 dB" in summary
    assert "Minimum required worst-channel relief: +0.050 dB" in summary


def test_auto_score_result_handles_active_stereo_policy(monkeypatch):
    monkeypatch.setattr(
        "decaycore.auto_mode.scoring_metrics.calc_ai_summary_from_stats",
        lambda st: {"score": 84.0},
    )

    base_stats = {
        "avg_confidence": 85.0,
        "freq_axis": [20.0, 30.0, 40.0, 60.0, 80.0, 120.0, 160.0, 220.0, 320.0, 500.0],
        "measured_mags": [0.0] * 10,
        "target_mags": [0.0] * 10,
        "filter_mags": [0.0] * 10,
        "confidence_mask": [1.0] * 10,
        "ripple_rms": 0.03,
        "net_boost_peak_db": 0.0,
        "lf_boost_max_db": 0.0,
        "ir_pre_ringing_db": -48.0,
        "gd_grad_limiter_after_max_ms_per_oct": 8.0,
        "gd_abs_max_20_500_ms": 14.0,
    }

    metrics = _auto_score_result(
        SimpleNamespace(
            l_st=dict(base_stats),
            r_st=dict(base_stats),
            metrics={},
        ),
        auto_exc_freq_hz=None,
        base_data={
            "enable_channel_specific_auto_policy": True,
            "channel_specific_policy_max_hz": 220.0,
            "mag_c_max": 3000.0,
            "_stereo_resolved_auto_policies": {
                "split_hz": 220.0,
                "shared": {"conf_pull_floor": 0.05, "tdc_strength": 50.0},
                "left": {"conf_pull_floor": 0.20},
                "right": {"tdc_strength": 62.0},
            },
        },
    )

    assert metrics["stereo_policy_active"] is True
    assert metrics["stereo_policy_split_hz"] == 220.0
    assert metrics["policy_divergence_score"] >= 0.0


def test_auto_score_result_stereo_gate_uses_delta_vs_shared_baseline(monkeypatch):
    monkeypatch.setattr(
        "decaycore.auto_mode.scoring_metrics.calc_ai_summary_from_stats",
        lambda st: {"score": 84.0},
    )

    left_stats = {
        "avg_confidence": 85.0,
        "freq_axis": [20.0, 30.0, 40.0, 60.0, 80.0, 120.0, 160.0, 220.0, 260.0, 320.0, 400.0, 500.0, 700.0, 900.0, 1200.0, 1800.0, 2400.0, 3000.0],
        "measured_mags": [0.0] * 18,
        "target_mags": [0.0] * 18,
        "filter_mags": [0.0] * 18,
        "confidence_mask": [1.0] * 18,
        "ripple_rms": 0.03,
        "net_boost_peak_db": 0.0,
        "lf_boost_max_db": 0.0,
        "ir_pre_ringing_db": -48.0,
        "gd_grad_limiter_after_max_ms_per_oct": 8.0,
        "gd_abs_max_20_500_ms": 14.0,
    }
    right_stats = dict(left_stats)
    right_stats["measured_mags"] = [2.0] * 18

    metrics = _auto_score_result(
        SimpleNamespace(
            l_st=dict(left_stats),
            r_st=dict(right_stats),
            metrics={},
        ),
        auto_exc_freq_hz=None,
        base_data={
            "enable_channel_specific_auto_policy": True,
            "channel_specific_policy_max_hz": 220.0,
            "mag_c_max": 3000.0,
            "_stereo_shared_l_st": dict(left_stats),
            "_stereo_shared_r_st": dict(right_stats),
            "_stereo_resolved_auto_policies": {
                "split_hz": 220.0,
                "shared": {"conf_pull_floor": 0.05, "tdc_strength": 50.0},
                "left": {"conf_pull_floor": 0.20},
                "right": {},
            },
        },
    )

    assert metrics["stereo_policy_active"] is True
    assert metrics["stereo_lr_mismatch_above_split_db"] > 1.25
    assert metrics["stereo_lr_mismatch_above_split_delta_vs_shared_db"] == 0.0
    assert metrics["stereo_policy_gate_failed"] is False


def test_stereo_policy_refine_skips_trivial_relief_candidate_and_can_apply_real_one(monkeypatch):
    shared_result = SimpleNamespace(
        l_st={"side": "left"},
        r_st={"side": "right"},
    )
    candidate1_result = SimpleNamespace(
        l_st={"relief": 0.01},
        r_st={"relief": 0.01},
    )
    candidate2_result = SimpleNamespace(
        l_st={"relief": 0.10},
        r_st={"relief": 0.10},
    )
    call_index = {"value": 0}

    monkeypatch.setattr(
        "decaycore.auto_mode.stereo_policy_refine.band_lr_mismatch_rms_from_stats",
        lambda *args, **kwargs: 0.0,
    )
    monkeypatch.setattr(
        "decaycore.auto_mode.stereo_policy_refine._seed_side_metrics",
        lambda stats, split_hz: {"side": str(dict(stats or {}).get("side", ""))},
    )
    monkeypatch.setattr(
        "decaycore.auto_mode.stereo_policy_refine._seed_side_score",
        lambda metrics: 1.0 if str(metrics.get("side", "")) == "left" else 0.0,
    )
    monkeypatch.setattr(
        "decaycore.auto_mode.stereo_policy_refine.worst_channel_relief_db",
        lambda shared_l, shared_r, cand_l, cand_r, **kwargs: float(dict(cand_l or {}).get("relief", 0.0)),
    )

    def _materialize(preset, **kwargs):
        call_index["value"] += 1
        idx = int(call_index["value"])
        if idx == 1:
            return shared_result, {"rank_score": 92.214, "rank_score_official": 92.214}, {}
        if idx == 2:
            return candidate1_result, {"rank_score": 92.300, "rank_score_official": 92.300}, {}
        return candidate2_result, {"rank_score": 92.290, "rank_score_official": 92.290}, {}

    preset, metrics, improved, meta = apply_stereo_policy_refine(
        best_preset={},
        best_metrics={"rank_score": 92.214, "rank_score_official": 92.214},
        base_data_ref={
            "enable_channel_specific_auto_policy": True,
            "channel_specific_policy_max_hz": 220.0,
            "channel_specific_refine_trials": 2,
            "min_worst_channel_improvement_db": 0.05,
            "conf_pull_floor": 0.05,
            "tdc_strength": 50.0,
            "tdc_max_reduction_db": 9.0,
            "bass_first_mode_max_hz": 180.0,
            "low_bass_cut_strength": 0.0,
        },
        goal="balanced",
        phase_label="test stereo policy refine",
        materialize_preset_result=_materialize,
        auto_is_better_refine=lambda new, best, goal: float(new.get("rank_score", 0.0)) > float(best.get("rank_score", 0.0)),
    )

    assert improved is True
    assert meta["state"] == "applied"
    assert float(meta["worst_channel_relief_db"]) == 0.10
    assert float(meta["min_worst_channel_improvement_db"]) == 0.05
    assert float(metrics["rank_score"]) == 92.290


def test_finalize_search_result_preserves_stereo_refine_rank_context(monkeypatch):
    from decaycore.auto_mode.search_state import _AutoModeSearchState

    monkeypatch.setattr(
        "decaycore.auto_mode.orchestrator_finalize_run.coerce_orchestrator_runtime",
        lambda runtime: SimpleNamespace(
            phase_limit_winner_polish_enabled=False,
            phase_limit_winner_polish_offsets_hz=(),
            mag_c_min_winner_polish_enabled=False,
            mag_c_min_winner_polish_step_hz=2.0,
            mag_c_min_winner_polish_max_down_hz=0.0,
            excess_phase_strength_winner_polish_enabled=False,
            excess_phase_strength_winner_polish_step=0.05,
            excess_phase_strength_winner_polish_max_delta=0.15,
        ),
    )
    monkeypatch.setattr(
        "decaycore.auto_mode.orchestrator_finalize_polish.apply_phase_limit_winner_polish",
        lambda **kwargs: (kwargs["best_preset"], kwargs["best_metrics"], False, {}),
    )
    monkeypatch.setattr(
        "decaycore.auto_mode.orchestrator_finalize_polish.apply_mag_c_min_winner_polish",
        lambda **kwargs: (kwargs["best_preset"], kwargs["best_metrics"], False, {}),
    )
    monkeypatch.setattr(
        "decaycore.auto_mode.orchestrator_finalize_polish.apply_hpf_winner_polish",
        lambda **kwargs: (kwargs["best_preset"], kwargs["best_metrics"], False, {}),
    )
    monkeypatch.setattr(
        "decaycore.auto_mode.orchestrator_finalize_polish.apply_excess_phase_strength_winner_polish",
        lambda **kwargs: (kwargs["best_preset"], kwargs["best_metrics"], False, {}),
    )
    monkeypatch.setattr(
        "decaycore.auto_mode.orchestrator_finalize_polish._save_cached_best",
        lambda **kwargs: None,
    )

    refined_meta = {
        "state": "applied",
        "gate_passed": True,
        "shared_rank": 92.214,
        "refined_rank": 93.275,
        "resolved": {"left": {"conf_pull_floor": 0.25}},
        "_shared_l_st": {"shared": "left"},
        "_shared_r_st": {"shared": "right"},
    }
    monkeypatch.setattr(
        "decaycore.auto_mode.orchestrator_finalize_polish.apply_stereo_policy_refine",
        lambda **kwargs: (
            {"_stereo_resolved_auto_policies": {"left": {"conf_pull_floor": 0.25}}},
            {"rank_score": 93.275, "rank_score_official": 93.275, "rank_score_components": {"rank_score": 93.275}},
            True,
            dict(refined_meta),
        ),
    )

    def _materialize(preset, *, include_response_arrays, summarize, base_data_override):
        assert dict(base_data_override or {}).get("_stereo_shared_l_st") == {"shared": "left"}
        assert dict(base_data_override or {}).get("_stereo_shared_r_st") == {"shared": "right"}
        return (
            SimpleNamespace(),
            {"rank_score": 93.275, "rank_score_official": 93.275, "rank_score_components": {"rank_score": 93.275}},
            dict(preset or {}),
        )

    search_state = _AutoModeSearchState(
        best_result=SimpleNamespace(),
        best_metrics={"rank_score": 92.214, "rank_score_official": 92.214, "rank_score_components": {"rank_score": 92.214}},
        best_preset={},
        winner_explanation={},
        scored=[],
        phase2_pool=[],
    )

    out = finalize_search_result(
        search_base_data={},
        cache_base_data={},
        measurements={},
        fs_v=44100,
        taps_v=65536,
        xos=[],
        hpf=None,
        hc_f=None,
        hc_m=None,
        pin_obj=None,
        cfg=SimpleNamespace(cache_enabled=False),
        goal="balanced",
        rank_basis="rank_score",
        filter_key="Asymmetric",
        compat_version="am7",
        optimizer_backend="optuna",
        status_cb=None,
        optuna_mod=None,
        optuna_search_sig="sig",
        seed=1,
        search_state=search_state,
        winner_target_name="Adaptive",
        phase1_ok=1,
        phase2_ok=0,
        phase1_tried=1,
        phase2_tried=0,
        phase1_plateau_hit=False,
        phase2_plateau_hit=False,
        phase1_optuna_tel={},
        phase2_local_optuna_tels=[],
        phase3_micro_optuna_tel={},
        phase2_rollup_tel={},
        _cache_ready_preset=lambda preset, **kwargs: dict(preset or {}),
        _materialize_preset_result=_materialize,
        _maybe_apply_residual_tiebreak=lambda **kwargs: (kwargs["best_preset"], kwargs["best_metrics"], False),
        cache_refine_result=None,
        runtime=SimpleNamespace(),
    )

    assert float(out["best_metrics"]["rank_score_official"]) == 93.275
    assert out["stereo_policy_refine"]["state"] == "applied"
    assert "_shared_l_st" not in out["stereo_policy_refine"]


def test_finalize_search_result_keeps_optuna_excursion_seed(monkeypatch):
    from decaycore.auto_mode.search_state import _AutoModeSearchState

    monkeypatch.setattr(
        "decaycore.auto_mode.orchestrator_finalize_run.coerce_orchestrator_runtime",
        lambda runtime: SimpleNamespace(
            phase_limit_winner_polish_enabled=False,
            phase_limit_winner_polish_offsets_hz=(),
            mag_c_min_winner_polish_enabled=False,
            mag_c_min_winner_polish_step_hz=2.0,
            mag_c_min_winner_polish_max_down_hz=0.0,
            hpf_winner_polish_enabled=False,
            excess_phase_strength_winner_polish_enabled=False,
            excess_phase_strength_winner_polish_step=0.05,
            excess_phase_strength_winner_polish_max_delta=0.15,
        ),
    )
    monkeypatch.setattr(
        "decaycore.auto_mode.orchestrator_finalize_polish.apply_stereo_policy_refine",
        lambda **kwargs: (kwargs["best_preset"], kwargs["best_metrics"], False, {}),
    )
    monkeypatch.setattr(
        "decaycore.auto_mode.orchestrator_finalize_polish.apply_phase_limit_winner_polish",
        lambda **kwargs: (kwargs["best_preset"], kwargs["best_metrics"], False, {}),
    )
    monkeypatch.setattr(
        "decaycore.auto_mode.orchestrator_finalize_polish.apply_mag_c_min_winner_polish",
        lambda **kwargs: (kwargs["best_preset"], kwargs["best_metrics"], False, {}),
    )
    monkeypatch.setattr(
        "decaycore.auto_mode.orchestrator_finalize_polish.apply_low_bass_cut_winner_polish",
        lambda **kwargs: (kwargs["best_preset"], kwargs["best_metrics"], False, {}),
    )
    monkeypatch.setattr(
        "decaycore.auto_mode.orchestrator_finalize_polish.apply_hpf_winner_polish",
        lambda **kwargs: (kwargs["best_preset"], kwargs["best_metrics"], False, {}),
    )
    monkeypatch.setattr(
        "decaycore.auto_mode.orchestrator_finalize_polish.apply_excess_phase_strength_winner_polish",
        lambda **kwargs: (kwargs["best_preset"], kwargs["best_metrics"], False, {}),
    )
    monkeypatch.setattr(
        "decaycore.auto_mode.orchestrator_finalize_polish._save_cached_best",
        lambda **kwargs: None,
    )

    search_state = _AutoModeSearchState(
        best_result=SimpleNamespace(),
        best_metrics={
            "rank_score": 92.214,
            "rank_score_official": 92.214,
            "rank_score_components": {"rank_score": 92.214},
            "auto_exc_zero_penalty_hz": 44.0,
        },
        best_preset={},
        winner_explanation={},
        scored=[],
        phase2_pool=[],
    )

    out = finalize_search_result(
        search_base_data={"_auto_exc_freq_hz": 31.5, "exc_freq": 31.5},
        cache_base_data={},
        measurements={},
        fs_v=44100,
        taps_v=65536,
        xos=[],
        hpf=None,
        hc_f=None,
        hc_m=None,
        pin_obj=None,
        cfg=SimpleNamespace(cache_enabled=False),
        goal="balanced",
        rank_basis="rank_score",
        filter_key="Asymmetric",
        compat_version="am7",
        optimizer_backend="optuna",
        status_cb=None,
        optuna_mod=None,
        optuna_search_sig="sig",
        seed=1,
        search_state=search_state,
        winner_target_name="Adaptive",
        phase1_ok=1,
        phase2_ok=0,
        phase1_tried=1,
        phase2_tried=0,
        phase1_plateau_hit=False,
        phase2_plateau_hit=False,
        phase1_optuna_tel={},
        phase2_local_optuna_tels=[],
        phase3_micro_optuna_tel={},
        phase2_rollup_tel={},
        _cache_ready_preset=lambda preset, **kwargs: dict(preset or {}),
        _materialize_preset_result=lambda preset, **kwargs: (
            SimpleNamespace(),
            {
                "rank_score": 92.214,
                "rank_score_official": 92.214,
                "rank_score_components": {"rank_score": 92.214},
                "auto_exc_zero_penalty_hz": 44.0,
            },
            dict(preset or {}),
        ),
        _maybe_apply_residual_tiebreak=lambda **kwargs: (kwargs["best_preset"], kwargs["best_metrics"], False),
        cache_refine_result=None,
        runtime=SimpleNamespace(),
    )

    assert out is not None
    assert out["best_auto_exc_freq_hz"] == 31.5
