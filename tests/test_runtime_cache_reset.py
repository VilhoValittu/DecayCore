import sys
import types

import numpy as np

from decaycore import decaycore as app_module
from decaycore.auto_mode import cache_signature, filter_priors
from decaycore.dsp import (
    bass_integration,
    decaycore_analysis,
    decaycore_leveling,
    correction_baseline,
    smoothing,
)
from decaycore.workflow.runtime_cache import reset_runtime_caches


def test_reset_runtime_caches_clears_process_local_caches(monkeypatch):
    correction_baseline._RT60_CACHE["rt60"] = (0.3, {}, 0.3)
    decaycore_leveling._LEVELING_CACHE["leveling"] = ((0.0, 0.0, 0.0, 0.0, "", 0.0, 0.0), {})
    decaycore_leveling._LEVEL_WINDOW_CACHE["window"] = (500.0, 2000.0)
    smoothing._SMOOTHING_PLAN_CACHE["plan"] = {"ok": True}
    smoothing._AFDW_STACK_CACHE["stack"] = {"ok": True}
    decaycore_analysis._SOS_CACHE[("band",)] = np.asarray([1.0], dtype=float)
    bass_integration._BUTTER_RESPONSE_CACHE["butter"] = np.asarray([1.0 + 0.0j], dtype=np.complex128)
    cache_signature._SYNTH_TARGET_CACHE["target"] = ("hc_f", "hc_m")
    monkeypatch.setattr(filter_priors, "_PRIORS_CACHE", {"filters": {"linear": {}}})
    filter_priors._PRIOR_LOOKUP_CACHE[("linear", "measurement")] = {"seed_preset": {}}

    reset_runtime_caches()

    assert correction_baseline._RT60_CACHE == {}
    assert decaycore_leveling._LEVELING_CACHE == {}
    assert decaycore_leveling._LEVEL_WINDOW_CACHE == {}
    assert smoothing._SMOOTHING_PLAN_CACHE == {}
    assert smoothing._AFDW_STACK_CACHE == {}
    assert decaycore_analysis._SOS_CACHE == {}
    assert bass_integration._BUTTER_RESPONSE_CACHE == {}
    assert cache_signature._SYNTH_TARGET_CACHE == {}
    assert filter_priors._PRIORS_CACHE is None
    assert filter_priors._PRIOR_LOOKUP_CACHE == {}


def test_process_run_resets_runtime_caches_before_building_request(monkeypatch):
    order: list[str] = []
    fake_request = object()
    fake_ui_bridge = object()

    class _FakeNgPinProxy:
        pass

    class _FakeProcessRunSupport:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    def _build_run_request_from_pin(*_args, **_kwargs):
        order.append("build_request")
        return fake_request

    def _run_process_flow(*, request, support):
        order.append("run")
        assert request is fake_request
        assert isinstance(support, _FakeProcessRunSupport)
        assert support.kwargs["ui_bridge"] is fake_ui_bridge
        return "ok"

    monkeypatch.setattr(app_module, "reset_runtime_caches", lambda: order.append("reset"))
    monkeypatch.setitem(
        sys.modules,
        "decaycore.application.request_builder",
        types.SimpleNamespace(build_run_request_from_pin=_build_run_request_from_pin),
    )
    monkeypatch.setitem(
        sys.modules,
        "decaycore.auto_mode.api",
        types.SimpleNamespace(AUTO_MODE_COMPAT_VERSION="am-test"),
    )
    monkeypatch.setitem(
        sys.modules,
        "decaycore.ui.ng_controls",
        types.SimpleNamespace(NgPinProxy=_FakeNgPinProxy),
    )
    monkeypatch.setitem(
        sys.modules,
        "decaycore.workflow.process_run_flow",
        types.SimpleNamespace(
            ProcessRunSupport=_FakeProcessRunSupport,
            run_process_flow=_run_process_flow,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "decaycore.workflow.process_support",
        types.SimpleNamespace(
            auto_target_mode_norm=lambda value: str(value or "auto"),
            auto_target_selection_method_text=lambda value: str(value or ""),
            pick_target_curve_label=lambda _data: "Target",
            slugify_filename_token=lambda *_args, **_kwargs: "target",
            has_uploaded_target_file=lambda _data: False,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "decaycore.ui.ng_bridge",
        types.SimpleNamespace(build_default_ui_bridge=lambda: fake_ui_bridge),
    )

    result = app_module.process_run()

    assert result == "ok"
    assert order == ["reset", "build_request", "run"]
