from decaycore.workflow.auto_flow import (
    _AUTO_PROGRESS_FINALIZE,
    _build_auto_finalize_status,
    _estimate_auto_progress_from_status,
    _get_auto_status_callback,
    _run_auto_mode_search_if_needed,
    _set_auto_progress,
)
from decaycore.ui import ui_state
from decaycore.auto_mode.search_v2 import finalize_adapter
import decaycore.auto_mode.scoring_ranking as scoring_ranking
from decaycore.auto_mode.orchestrator_finalize import _build_phase2_pareto_status
from decaycore.auto_mode.refine_eval import RefineEvalContext, _consume_phase_result
from decaycore.auto_mode.search_state import _AutoModePhaseState, _AutoModeSearchState
from decaycore.auto_mode.shared import _auto_metric_text
from types import SimpleNamespace


def test_auto_progress_bridge_failure_does_not_raise():
    class _FailingBridge:
        def set_progress(self, _value):
            raise RuntimeError("ui disconnected")

    support = SimpleNamespace(ui_bridge=_FailingBridge())
    ctx = {}

    _set_auto_progress(ctx, support=support, value=0.5)

    assert ctx["_auto_progress_state"]["value"] == 0.5


def test_auto_status_callback_failure_does_not_raise_and_keeps_progress():
    class _Bridge:
        def __init__(self):
            self.values = []

        def set_progress(self, value):
            self.values.append(float(value))

    def _failing_status(_msg):
        raise RuntimeError("ui disconnected")

    support = SimpleNamespace(ui_bridge=_Bridge())
    callbacks = SimpleNamespace(status=_failing_status)
    ctx = {}
    status_cb = _get_auto_status_callback(ctx, callbacks=callbacks, support=support)

    status_cb("DecayCore automatic mode: Phase 4 finalize / Pareto / winner polish")

    assert ctx["_auto_progress_state"]["value"] == _AUTO_PROGRESS_FINALIZE
    assert support.ui_bridge.values == [_AUTO_PROGRESS_FINALIZE]


def test_auto_progress_recognizes_phase4_finalize_status():
    assert (
        _estimate_auto_progress_from_status(
            "DecayCore automatic mode: Phase 4 finalize / Pareto / winner polish"
        )
        == _AUTO_PROGRESS_FINALIZE
    )


def test_auto_progress_advances_across_auto_search_phases():
    init_progress = _estimate_auto_progress_from_status(
        "DecayCore automatic mode: init (goal balanced, basis preset_objective_score, filter mixed, taps 65536)"
    )
    target_progress = _estimate_auto_progress_from_status(
        "DecayCore automatic mode: target search init (top-3, 10 trials/curve, fs 44100 Hz, taps 65536, -6 dB point 16.0 Hz, goal balanced)"
    )
    search_progress = _estimate_auto_progress_from_status(
        "DecayCore automatic mode: phase search init (phase1 100 + phase2 24 + micro 6 trials @ 44100 Hz, -6 dB point 16.0 Hz, goal balanced, basis preset_objective_score, target Adaptive)"
    )
    phase1_progress = _estimate_auto_progress_from_status(
        "DecayCore automatic mode [Adaptive] (-6 dB 16.0 Hz, low-cut 18.0 Hz, exc seed 24.0 Hz, hpf off): phase 1/2 50/100 (rank 90.000, ok 48/50)"
    )
    phase2_progress = _estimate_auto_progress_from_status(
        "DecayCore automatic mode [Adaptive] (-6 dB 16.0 Hz, low-cut 18.0 Hz, exc seed 24.0 Hz, hpf off): phase 2/2 local center#2 6/12 (rank 92.000, ok 6/6)"
    )
    pareto_progress = _estimate_auto_progress_from_status(
        "DecayCore automatic mode: phase 2 pareto comparison (rank_best 91.831 -> pareto 88.510, avg 84.335 -> 84.352, prepost 0.2200 -> 0.0900, mode_ripple 1.303 dB -> 0.780 dB)"
    )
    micro_progress = _estimate_auto_progress_from_status(
        "DecayCore automatic mode [Adaptive] (-6 dB 16.0 Hz, low-cut 18.0 Hz, exc seed 24.0 Hz, hpf off): micro refine 3/6 (rank 92.000, ok 3/3)"
    )
    finalize_progress = _estimate_auto_progress_from_status(
        "DecayCore automatic mode: finalize (winner rank 93.500/100, avg 88.300, boost 1.20 dB, events 0, via phase 2 pareto)"
    )

    assert init_progress is not None
    assert target_progress is not None
    assert search_progress is not None
    assert phase1_progress is None  # phase 1/2 handler removed in refactor; progress not tracked
    assert phase2_progress is not None
    assert pareto_progress is not None
    assert micro_progress is not None
    assert finalize_progress is not None
    assert init_progress < target_progress < search_progress < phase2_progress < micro_progress < pareto_progress < finalize_progress


def test_auto_progress_keeps_legacy_status_aliases():
    target_progress = _estimate_auto_progress_from_status(
        "DecayCore automatic mode: target preselect init (top-3, 10 trials/curve, fs 44100 Hz, taps 65536, -6 dB point 16.0 Hz, goal balanced)"
    )
    search_progress = _estimate_auto_progress_from_status(
        "DecayCore automatic mode: preset search init (phase1 100 + refine 24 trials @ 44100 Hz, -6 dB point 16.0 Hz, goal balanced, basis preset_objective_score, target Adaptive)"
    )
    micro_progress = _estimate_auto_progress_from_status(
        "DecayCore automatic mode [Adaptive]: phase 3/3 micro 3/6 (rank 92.000, ok 3/3)"
    )

    assert target_progress is not None
    assert search_progress is not None
    assert micro_progress is not None
    assert target_progress < search_progress < micro_progress


def test_auto_status_details_keep_canonical_auto_phase_rows():
    ui_state.reset_auto_status_details()
    ui_state.update_status(
        "DecayCore automatic mode: phase search init "
        "(phase1 100 + phase2 24 + micro 6 trials @ 44100 Hz, -6 dB point 16.0 Hz, "
        "goal balanced, basis preset_objective_score, target Harman12)"
    )
    ui_state.update_status(
        "DecayCore automatic mode: micro refine 6 trials around current best"
    )
    ui_state.update_status(
        "DecayCore automatic mode: micro refine summary "
        "current_best_rank=83.500, avg_score=87.600, optuna run=6, ok=6"
    )
    ui_state.update_status(
        "DecayCore automatic mode: finalize "
        "(winner rank 83.500/100, avg 87.600, boost 0.00 dB, events 30)"
    )

    body = ui_state.get_status_snapshot()["auto_status_detail_body"]
    assert "phase" in body.lower() or "vaihe" in body.lower()
    assert "micro" in body.lower()
    assert "Done" in body or "Valmis" in body


def test_auto_status_details_keep_full_run_history():
    ui_state.reset_auto_status_details()

    first = (
        "DecayCore automatic mode [Harman12]: phase 1/2 1/120 "
        "(best rank=10.000)"
    )
    last = (
        "DecayCore automatic mode [Harman12]: phase 1/2 120/120 "
        "(best rank=90.000)"
    )
    ui_state.update_status(first)
    for idx in range(2, 120):
        ui_state.update_status(
            "DecayCore automatic mode [Harman12]: phase 1/2 "
            f"{idx}/120 (best rank={10.0 + idx:.3f})"
        )
    ui_state.update_status(last)

    snap = ui_state.get_status_snapshot()
    details = snap["auto_status_details"]
    body = snap["auto_status_detail_body"]

    assert len(details) == 120
    assert "1/120" in body
    assert "120/120" in body


def test_auto_status_details_humanize_phase3_skip_notice():
    ui_state.reset_auto_status_details()
    ui_state.update_status("DecayCore automatic mode: phase 3 skipped")

    snap = ui_state.get_status_snapshot()
    details = snap["auto_status_details"]
    body = snap["auto_status_detail_body"]

    assert len(details) == 1
    assert "Phase 3 skipped" in body


def test_phase2_pareto_front_vectorized_matches_scalar_fallback(monkeypatch):
    vectors = [
        (1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0),
        (0.9, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0),
        (1.0, 0.8, 1.0, 1.0, 1.0, -1.0, 1.0),
        (1.2, 1.2, 1.2, 1.2, 1.2, -0.8, 1.2),
    ]
    pool = [{"id": idx, "metrics": {"idx": idx}} for idx in range(len(vectors))]

    def fake_vector(metrics):
        return vectors[int(metrics["idx"])]

    monkeypatch.setattr(scoring_ranking, "_auto_phase2_pareto_vector", fake_vector)

    expected = []
    for i, cand in enumerate(pool):
        if not any(
            i != j and scoring_ranking._pareto_dominates(vectors[j], vectors[i])
            for j in range(len(pool))
        ):
            expected.append(dict(cand))

    assert scoring_ranking._auto_phase2_pareto_front(pool) == expected


def test_auto_progress_tracks_target_trials_inside_shortlist():
    first_trial = _estimate_auto_progress_from_status(
        "DecayCore automatic mode: target trials (target 1/3 Harman6, trial 1/10, rank 88.100)"
    )
    late_same_target = _estimate_auto_progress_from_status(
        "DecayCore automatic mode: target trials best improved (target 1/3 Harman6, trial 8/10, -6 dB point 16.0 Hz, goal balanced, rank 89.400, avg 84.200, fit 0.320, pre 0.410)"
    )
    next_target = _estimate_auto_progress_from_status(
        "DecayCore automatic mode: target trials (target 2/3 Studio, trial 1/10, rank 89.600)"
    )

    assert first_trial is not None
    assert late_same_target is not None
    assert next_target is not None
    assert first_trial < late_same_target < next_target


def test_refine_status_uses_official_rank_score():
    messages = []
    search_state = _AutoModeSearchState()
    phase_state = _AutoModePhaseState()
    ctx = RefineEvalContext(
        search_base_data={},
        measurements={},
        fs_v=44100,
        taps_v=65536,
        xos=[],
        hpf=None,
        hc_f=None,
        hc_m=None,
        pin_obj=None,
        cfg=SimpleNamespace(),
        goal="flat",
        filter_key="linear",
        optimizer_backend="builtin",
        optuna_mod=None,
        seed=1,
        optuna_search_sig="sig",
        status_cb=messages.append,
        status_prefix="DecayCore automatic mode [Harman12]",
        winner_target_name="Harman12",
        search_state=search_state,
        runtime=SimpleNamespace(),
    )

    _consume_phase_result(
        ctx,
        phase_state=phase_state,
        out={
            "ok": True,
            "metrics": {
                "rank_score": 34.810,
                "rank_score_official": 91.517,
                "avg_score": 75.367,
                "mode_ripple_db": 2.545,
                "max_net_boost_db": -0.14,
            },
            "preset": {"phase_limit": 400.0},
            "result": object(),
        },
        idx=1,
        n_total=100,
        phase_label="phase 1/2",
        plateau_after_no_improve=0,
        use_refine_tiebreak=False,
    )

    assert messages
    assert "best improved trial 1/100" in messages[-1]
    assert "rank 91.517" in messages[-1]
    assert "rank 34.810" not in messages[-1]


def test_refine_status_explains_hard_gate_display_cap():
    messages = []
    search_state = _AutoModeSearchState()
    phase_state = _AutoModePhaseState()
    ctx = RefineEvalContext(
        search_base_data={},
        measurements={},
        fs_v=44100,
        taps_v=65536,
        xos=[],
        hpf=None,
        hc_f=None,
        hc_m=None,
        pin_obj=None,
        cfg=SimpleNamespace(),
        goal="balanced",
        filter_key="asym",
        optimizer_backend="builtin",
        optuna_mod=None,
        seed=1,
        optuna_search_sig="sig",
        status_cb=messages.append,
        status_prefix="DecayCore automatic mode [Harman10]",
        winner_target_name="Harman10",
        search_state=search_state,
        runtime=SimpleNamespace(),
    )

    _consume_phase_result(
        ctx,
        phase_state=phase_state,
        out={
            "ok": True,
            "metrics": {
                "rank_score": 69.732,
                "avg_score": 70.6,
                "hard_gate_failed": True,
                "hard_gate_failures": ["residual_peak_hard_gate"],
                "mode_ripple_db": 2.5,
                "max_net_boost_db": 0.0,
            },
            "preset": {"phase_limit": 236.0},
            "result": object(),
        },
        idx=1,
        n_total=10,
        phase_label="phase 1/2",
        plateau_after_no_improve=0,
        use_refine_tiebreak=False,
    )

    assert messages
    assert "rank 59.000 (capped by residual_peak_hard_gate; raw 69.732)" in messages[-1]


def test_plateau_metric_text_prefers_official_rank_score():
    assert (
        _auto_metric_text(
            {"rank_score": 47.278, "rank_score_official": 77.783},
            "balanced",
        )
        == "rank=77.783"
    )


def test_target_finalize_detail_does_not_show_selection_pool_score():
    detail = ui_state._humanize_auto_status_detail(
        "DecayCore automatic mode: target finalize "
        "(winner Harman10, method top3x10_trials, rank 85.300, avg 79.200, "
        "pre 1.234, fit 4.200 dB)"
    )

    assert detail == "Room curve selected: Harman10 — fit 4.2\u00a0dB"
    assert "85.3" not in detail
    assert "score" not in detail.lower()


def test_refine_status_reports_decision_reason_when_candidate_display_rank_is_higher():
    messages = []
    search_state = _AutoModeSearchState()
    phase_state = _AutoModePhaseState()
    ctx = RefineEvalContext(
        search_base_data={},
        measurements={},
        fs_v=44100,
        taps_v=65536,
        xos=[],
        hpf=None,
        hc_f=None,
        hc_m=None,
        pin_obj=None,
        cfg=SimpleNamespace(),
        goal="balanced",
        filter_key="linear",
        optimizer_backend="builtin",
        optuna_mod=None,
        seed=1,
        optuna_search_sig="sig",
        status_cb=messages.append,
        status_prefix="DecayCore automatic mode [Harman12]",
        winner_target_name="Harman12",
        search_state=search_state,
        runtime=SimpleNamespace(),
    )

    _consume_phase_result(
        ctx,
        phase_state=phase_state,
        out={
            "ok": True,
            "metrics": {
                "rank_score": 45.75,
                "rank_score_refine": 45.10,
                "rank_score_official": 77.783,
                "avg_score": 77.2,
                "mode_ripple_db": 2.3,
            },
            "trial_preset": {"phase_limit": 130.0},
        },
        idx=1,
        n_total=8,
        phase_label="phase 2/2 local center#1",
        plateau_after_no_improve=0,
        use_refine_tiebreak=True,
    )
    _consume_phase_result(
        ctx,
        phase_state=phase_state,
        out={
            "ok": True,
            "metrics": {
                "rank_score": 45.80,
                "rank_score_refine": 45.00,
                "rank_score_official": 77.870,
                "avg_score": 77.2,
                "mode_ripple_db": 3.2,
            },
            "trial_preset": {"phase_limit": 131.0},
        },
        idx=2,
        n_total=8,
        phase_label="phase 2/2 local center#1",
        plateau_after_no_improve=0,
        use_refine_tiebreak=True,
    )

    assert "candidate rank 77.870" in messages[-1]
    assert "decision mode_ripple" in messages[-1]
    assert "refine rank 45.000" in messages[-1]


def _finalize_adapter_context(status_cb):
    return SimpleNamespace(
        profiler=None,
        status_cb=status_cb,
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
        cfg=SimpleNamespace(),
        goal="balanced",
        rank_basis="preset_objective_score",
        filter_key="mixed",
        compat_version="test",
        optimizer_backend="builtin",
        optuna_mod=None,
        optuna_search_sig="sig",
        seed=1,
        search_state=SimpleNamespace(),
        winner_target_name="Harman6",
        cache_ready_preset=lambda *args, **kwargs: {},
        materialize_preset_result=lambda *args, **kwargs: None,
        maybe_apply_residual_tiebreak=lambda *args, **kwargs: None,
        runtime=SimpleNamespace(),
        canonical_signature="sig",
    )


def test_finalize_from_refine_stats_continues_after_phase4_status_failure(monkeypatch):
    calls = []

    def _failing_status(_msg):
        raise RuntimeError("ui disconnected")

    def _fake_finalize_search_result(**kwargs):
        calls.append(dict(kwargs))
        return {"auto_mode_debug": {}}

    monkeypatch.setattr(
        finalize_adapter.orchestrator_finalize,
        "finalize_search_result",
        _fake_finalize_search_result,
    )

    result = finalize_adapter.finalize_from_refine_stats(
        _finalize_adapter_context(_failing_status),
        {"phase3_micro_optuna_tel": {}},
    )

    assert result == {"auto_mode_debug": {}}
    assert len(calls) == 1


def test_finalize_from_cache_refine_continues_after_phase4_status_failure(monkeypatch):
    calls = []

    def _failing_status(_msg):
        raise RuntimeError("ui disconnected")

    def _fake_finalize_search_result(**kwargs):
        calls.append(dict(kwargs))
        return {"auto_mode_debug": {}}

    monkeypatch.setattr(
        finalize_adapter.orchestrator_finalize,
        "finalize_search_result",
        _fake_finalize_search_result,
    )

    result = finalize_adapter.finalize_from_cache_refine(
        _finalize_adapter_context(_failing_status),
        {"ok": True},
    )

    assert result == {"auto_mode_debug": {}}
    assert len(calls) == 1
    assert calls[0]["cache_refine_result"] == {"ok": True}


def test_auto_finalize_status_calls_out_special_winner_stage():
    text = _build_auto_finalize_status(
        {
            "rank_score": 88.510,
            "rank_score_official": 88.510,
            "avg_score": 84.352,
            "max_net_boost_db": -0.01,
            "bass_boost_20_200_db": 1.42,
            "events_total": 30,
        },
        winner_explanation={"phase_label": "phase 2 pareto"},
    )

    assert "winner rank 88.510/100" in text
    assert "boost 1.42 dB" in text
    assert "via phase 2 pareto" in text


def test_phase2_pareto_status_explains_rank_tradeoff():
    text = _build_phase2_pareto_status(
        rank_best_metrics={
            "rank_score": 91.831,
            "rank_score_official": 96.123,
            "avg_score": 84.335,
            "ir_pre_post_energy_ratio_max": 0.2200,
            "mode_ripple_db": 1.303,
        },
        winner_metrics={
            "rank_score": 88.510,
            "rank_score_official": 94.456,
            "avg_score": 84.352,
            "ir_pre_post_energy_ratio_max": 0.0900,
            "mode_ripple_db": 0.780,
        },
    )

    assert "phase 2 pareto comparison" in text
    assert "selected winner" not in text
    assert "rank_best 96.123 -> pareto 94.456" in text
    assert "rank_best 91.831 -> pareto 88.510" not in text
    assert "prepost 0.2200 -> 0.0900" in text


def test_auto_flow_keeps_stereo_policy_refine_meta(monkeypatch):
    class _DummyUiBridge:
        def __init__(self):
            self.progress = []

        def set_progress(self, value):
            self.progress.append(float(value))

    class _DummySupport:
        def __init__(self):
            self.version = "test-version"
            self.ui_bridge = _DummyUiBridge()

    class _DummyCallbacks:
        def __init__(self):
            self.status_messages = []
            self.selected_bar = None

        def status(self, msg):
            self.status_messages.append(str(msg))

        def set_auto_selected_bar(self, text):
            self.selected_bar = str(text)

    stereo_meta = {
        "state": "shared_fallback",
        "gate_passed": False,
        "gate_reason": "asymmetry_too_small",
        "split_hz": 220.0,
    }

    monkeypatch.setattr(
        "decaycore.workflow.auto_flow._run_auto_mode_search",
        lambda **kwargs: {
            "best_preset": {"filter_type": "Asymmetric", "hc_mode": "Adaptive", "phase_limit": 240.0},
            "best_applied_preset": {"filter_type": "Asymmetric", "hc_mode": "Adaptive", "phase_limit": 240.0},
            "best_metrics": {
                "rank_score": 91.0,
                "rank_score_official": 91.0,
                "rank_score_components": {"rank_score": 91.0},
                "avg_score": 84.0,
                "dsp_penalty": 0.5,
                "exc_penalty": 0.1,
                "max_net_boost_db": 1.0,
                "events_total": 0,
                "events_severity": 0.0,
            },
            "winner": {
                "rank_score_official": 91.0,
                "rank_score_components": {"rank_score": 91.0},
            },
            "stereo_policy_refine": dict(stereo_meta),
            "auto_goal": "balanced",
            "selection_basis": "rank_score",
            "optimizer_backend": "builtin",
            "trials_total": 4,
            "trials_ok": 4,
            "trials_phase1_total": 4,
            "trials_phase1_ok": 4,
            "trials_phase2_total": 0,
            "trials_phase2_ok": 0,
            "search_fs": 44100,
            "search_taps": 2048,
            "top": [],
            "winner_explanation": {"phase_label": "phase 1"},
        },
    )

    ctx = {
        "auto_mode_enabled": True,
        "data": {
            "filter_type": "Asymmetric",
            "hc_mode": "Adaptive",
            "camillafir_automatic_mode": True,
            "enable_channel_specific_auto_policy": True,
            "channel_specific_policy_max_hz": 220.0,
        },
        "measurements": {},
        "target_rates": [44100],
        "xos": [],
        "hpf": None,
        "hc_f": None,
        "hc_m": None,
        "taps_base": 2048,
        "auto_goal": "balanced",
        "auto_basis": "rank_score",
    }

    callbacks = _DummyCallbacks()
    support = _DummySupport()

    _run_auto_mode_search_if_needed(ctx, callbacks=callbacks, support=support)

    auto_meta = dict(ctx["data"].get("_auto_mode_meta", {}) or {})
    assert auto_meta.get("stereo_policy_refine") == stereo_meta
