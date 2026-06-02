from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from decaycore.application.run_contracts import ResolvedRunConfig
from decaycore.application.run_request import RunRequest
from decaycore.ui.ng_bridge import ProcessRunCallbacks
from decaycore.workflow.auto_flow import _run_auto_mode_search_if_needed
from decaycore.workflow.pipeline_flow import _run_pipeline
from decaycore.workflow.run_prepare import _prepare_ui_and_measurements


class _UiBridge:
    def __init__(self) -> None:
        self.progress: list[float] = []

    def ensure_progress_bar(self) -> None:
        return None

    def set_progress(self, value: float) -> None:
        self.progress.append(float(value))

    def toast_health_gate_result(self, hr, mode) -> bool:
        return False


class _Support:
    def __init__(self) -> None:
        self.version = "test-version"
        self.max_safe_boost = 8.0
        self.force_single_plot_fs_hz = 0
        self.pick_target_curve_label = lambda data: "Flat"
        self.slugify_filename_token = lambda text, default="target": "flat"
        self.auto_target_mode_norm = lambda value: str(value or "auto")
        self.auto_target_selection_method_text = lambda value: str(value or "")
        self.has_uploaded_target_file = lambda data: False
        self.ui_bridge = _UiBridge()


def _callbacks() -> ProcessRunCallbacks:
    return ProcessRunCallbacks(
        status=lambda msg: None,
        set_auto_selected_bar=lambda msg="": None,
    )


def _measurement_tuple():
    f = np.asarray([20.0, 100.0], dtype=float)
    m = np.asarray([0.0, -3.0], dtype=float)
    p = np.asarray([0.0, 0.0], dtype=float)
    return f, m, p, f, m, p


def test_prepare_keeps_raw_request_separate_from_resolved_data(monkeypatch) -> None:
    raw = {
        "mode": "AUTO",
        "bass_integration_enable": False,
        "ir_export_tukey_alpha": "bad",
    }
    request = RunRequest(raw_ui_data=raw)

    monkeypatch.setattr("decaycore.workflow.run_prepare.compute_health", lambda data, mode: object())
    monkeypatch.setattr("decaycore.workflow.run_prepare.save_config", lambda data: None)
    monkeypatch.setattr(
        "decaycore.workflow.run_prepare.load_measurements_lr",
        lambda data, logger=None: _measurement_tuple(),
    )
    monkeypatch.setattr(
        "decaycore.workflow.run_prepare.load_bass_integration_measurements",
        lambda data, logger=None: (_ for _ in ()).throw(AssertionError("unexpected bass integration load")),
    )
    monkeypatch.setattr(
        "decaycore.workflow.run_prepare.load_raw_irs_lr",
        lambda data, logger=None, **kwargs: (None, 0, None, 0),
    )
    monkeypatch.setattr(
        "decaycore.workflow.run_prepare.load_raw_ir_sub",
        lambda data, logger=None: (_ for _ in ()).throw(AssertionError("sub IR should stay inactive")),
    )

    ctx = _prepare_ui_and_measurements(
        request=request,
        callbacks=_callbacks(),
        support=_Support(),
    )

    assert ctx is not None
    assert request.raw_ui_data == raw
    assert ctx["source_ui_data"] == raw
    assert ctx["resolved_data"] is ctx["data"]
    assert ctx["resolved_data"]["ir_export_tukey_alpha"] == 0.25
    assert ctx["source_ui_data"]["ir_export_tukey_alpha"] == "bad"


def test_auto_mode_winner_updates_resolved_data_not_source_ui_data(monkeypatch) -> None:
    source = {"mode": "AUTO", "filter_type": "Mixed", "mixed_freq": 120.0, "hc_mode": "Flat"}
    resolved = dict(source)
    measurements = {"ui_data": resolved}
    ctx = {
        "source_ui_data": source,
        "resolved_data": resolved,
        "data": resolved,
        "resolved_config": ResolvedRunConfig(
            source_ui_data=source,
            resolved_data=resolved,
            measurements=measurements,
            xos=[],
            hpf=None,
            target_rates=[44100],
            dash_fs=44100,
        ),
        "measurements": measurements,
        "target_rates": [44100],
        "xos": [],
        "hpf": None,
        "hc_f": None,
        "hc_m": None,
        "taps_base": 65536,
        "auto_goal": "balanced",
        "auto_basis": "preset_objective_score",
        "auto_mode_enabled": True,
    }
    auto_res = {
        "best_preset": {"mixed_freq": 180.0},
        "best_applied_preset": {"mixed_freq": 180.0, "phase_limit": 320.0},
        "best_metrics": {"avg_score": 1.0, "rank_score": 1.0},
        "winner": {"rank_score_official": 1.0, "rank_score_components": {}},
        "optimizer_backend": "builtin",
        "auto_goal": "balanced",
        "selection_basis": "rank_score",
    }
    prior_update_seen: dict[str, object] = {}

    monkeypatch.setattr("decaycore.workflow.auto_flow._run_auto_mode_search", lambda **kwargs: auto_res)
    monkeypatch.setattr(
        "decaycore.auto_mode.filter_priors.update_auto_mode_filter_priors_from_winner",
        lambda *args, **kwargs: prior_update_seen.update({"args": args, "kwargs": kwargs}),
    )

    _run_auto_mode_search_if_needed(ctx, callbacks=_callbacks(), support=_Support())

    assert source == {"mode": "AUTO", "filter_type": "Mixed", "mixed_freq": 120.0, "hc_mode": "Flat"}
    assert ctx["source_ui_data"] == source
    assert ctx["resolved_data"]["mixed_freq"] == 180.0
    assert ctx["resolved_data"]["phase_limit"] == 320.0
    assert ctx["auto_applied_preset"] == {"mixed_freq": 180.0, "phase_limit": 320.0}
    assert ctx["measurements"]["ui_data"] is ctx["resolved_data"]
    assert prior_update_seen["kwargs"]["measurements"] is ctx["measurements"]


def test_auto_flat_search_enables_unsafe_raw_dsp_in_base_data(monkeypatch) -> None:
    source = {"mode": "AUTO", "auto_goal": "flat", "filter_type": "Mixed", "mixed_freq": 120.0}
    measurements = {"ui_data": source}
    ctx = {
        "source_ui_data": source,
        "resolved_data": dict(source),
        "data": dict(source),
        "measurements": measurements,
        "target_rates": [44100],
        "xos": [],
        "hpf": {},
        "hc_f": None,
        "hc_m": None,
        "taps_base": 65536,
        "auto_goal": "flat",
        "auto_basis": "preset_objective_score",
        "auto_mode_enabled": True,
    }
    seen: dict[str, object] = {}
    auto_res = {
        "best_preset": {},
        "best_applied_preset": {},
        "best_metrics": {"avg_score": 1.0, "rank_score": 1.0},
        "winner": {"rank_score_official": 1.0, "rank_score_components": {}},
        "optimizer_backend": "builtin",
        "auto_goal": "flat",
        "selection_basis": "rank_score",
    }

    def _fake_auto_search(**kwargs):
        seen["base_data"] = dict(kwargs["base_data"])
        return auto_res

    monkeypatch.setattr("decaycore.workflow.auto_flow._run_auto_mode_search", _fake_auto_search)

    _run_auto_mode_search_if_needed(ctx, callbacks=_callbacks(), support=_Support())

    assert bool(seen["base_data"]["unsafe_raw_dsp"]) is True
    assert bool(ctx["resolved_data"]["unsafe_raw_dsp"]) is True


def test_pipeline_builds_filter_config_from_resolved_data(monkeypatch) -> None:
    source = {"mode": "BASIC", "filter_type": "Mixed", "mixed_freq": 100.0}
    resolved = {"mode": "BASIC", "filter_type": "Mixed", "mixed_freq": 220.0}
    measurements = {"ui_data": source}
    seen: dict[str, object] = {}

    def _fake_build_config(data, **kwargs):
        seen["data"] = data
        return SimpleNamespace()

    def _fake_run_pipeline(cfg, dsp_measurements):
        seen["measurements"] = dsp_measurements
        return SimpleNamespace(
            metrics={},
            l_st={},
            r_st={},
            l_ir=None,
            r_ir=None,
            sub_ir=None,
            sub_st=None,
            measurements={},
        )

    monkeypatch.setattr("decaycore.workflow.pipeline_flow.build_config", _fake_build_config)
    monkeypatch.setattr("decaycore.workflow.pipeline_flow.run_pipeline", _fake_run_pipeline)
    monkeypatch.setattr("decaycore.workflow.pipeline_flow.summarize_run", lambda result: {})

    ctx = {
        "source_ui_data": source,
        "resolved_data": resolved,
        "data": source,
        "measurements": measurements,
        "target_rates": [44100],
        "xos": [],
        "hpf": None,
        "hc_f": None,
        "hc_m": None,
        "taps_base": 65536,
        "dash_fs": 44100,
        "results_by_fs": [],
        "perf_stats": {"dsp_s": 0.0},
        "per_fs_stats": {},
        "auto_mode_enabled": False,
    }

    assert _run_pipeline(ctx, callbacks=_callbacks(), support=_Support()) is True
    assert seen["data"] is resolved
    assert seen["measurements"]["ui_data"] is resolved
    assert measurements["ui_data"] is source
