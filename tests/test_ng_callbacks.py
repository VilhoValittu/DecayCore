from __future__ import annotations

from decaycore.ui import ng_controls as ctrl
from decaycore.ui.ng_callbacks import (
    _register_ir_window_callbacks,
    _register_bass_integration_callbacks,
    _register_mode_callbacks,
    _register_target_callbacks,
    _sync_bass_integration_visibility,
)


def _t(key: str) -> str:
    return key


class _DummyEvent:
    def __init__(self, value):
        self.value = value


class _ReactiveControl:
    def __init__(self, value):
        self.value = value
        self._callbacks = []

    def on_value_change(self, callback):
        self._callbacks.append(callback)

    def set_value(self, value):
        self.value = value
        for callback in list(self._callbacks):
            callback(_DummyEvent(value))


class _ToggleControl:
    def __init__(self, value):
        self.value = value
        self.enabled = True

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False


class _DummyContainer:
    def __init__(self):
        self.visible = None

    def set_visibility(self, visible):
        self.visible = bool(visible)


def test_mode_change_disables_bass_integration_when_leaving_auto(monkeypatch) -> None:
    ctrl.reset()
    ctrl.register("mode", _ReactiveControl("AUTO"))
    ctrl.register("bass_integration_enable", _ReactiveControl(True))
    ctrl.register("camillafir_automatic_mode", ctrl._ValueHolder(True))

    seen_modes: list[str] = []
    monkeypatch.setattr("decaycore.ui.ng_callbacks._sync_bass_integration_visibility", lambda: None)
    monkeypatch.setattr("decaycore.ui.ng_callbacks._update_target_preview", lambda: None)
    monkeypatch.setattr(
        "decaycore.ui.ng_mode_controls.on_mode_change",
        lambda *, mode, t: seen_modes.append(mode),
    )

    _register_mode_callbacks(t=_t)
    ctrl.set_value("mode", "BASIC")

    assert ctrl.value("mode") == "BASIC"
    assert ctrl.value("bass_integration_enable") is False
    assert ctrl.value("camillafir_automatic_mode") is False
    assert seen_modes == ["BASIC"]


def test_mode_change_restores_last_manual_target_when_leaving_auto_adaptive(monkeypatch) -> None:
    ctrl.reset()
    ctrl.register("mode", _ReactiveControl("AUTO"))
    ctrl.register("bass_integration_enable", _ReactiveControl(False))
    ctrl.register("camillafir_automatic_mode", ctrl._ValueHolder(True))
    ctrl.register("hc_mode", _ReactiveControl("Adaptive"))
    ctrl.register("auto_target_mode", _ReactiveControl("adaptive"))
    ctrl.register("_manual_hc_mode", ctrl._ValueHolder("Harman6"))

    seen_modes: list[str] = []
    monkeypatch.setattr("decaycore.ui.ng_callbacks._sync_bass_integration_visibility", lambda: None)
    monkeypatch.setattr("decaycore.ui.ng_callbacks._update_target_preview", lambda: None)
    monkeypatch.setattr(
        "decaycore.ui.ng_mode_controls.on_mode_change",
        lambda *, mode, t: seen_modes.append(mode),
    )

    _register_mode_callbacks(t=_t)
    ctrl.set_value("mode", "BASIC")

    assert ctrl.value("hc_mode") == "Harman6"
    assert ctrl.value("camillafir_automatic_mode") is False
    assert seen_modes == ["BASIC"]


def test_enabling_bass_integration_forces_auto_mode(monkeypatch) -> None:
    ctrl.reset()
    ctrl.register("mode", _ReactiveControl("BASIC"))
    ctrl.register("bass_integration_enable", _ReactiveControl(False))
    ctrl.register("camillafir_automatic_mode", ctrl._ValueHolder(False))

    monkeypatch.setattr("decaycore.ui.ng_callbacks._sync_bass_integration_visibility", lambda: None)
    monkeypatch.setattr("decaycore.ui.ng_callbacks._update_target_preview", lambda: None)

    _register_bass_integration_callbacks(t=_t)
    ctrl.set_value("bass_integration_enable", True)

    assert ctrl.value("mode") == "AUTO"
    assert ctrl.value("camillafir_automatic_mode") is True


def test_auto_target_mode_selected_restores_last_manual_target_from_adaptive(monkeypatch) -> None:
    ctrl.reset()
    ctrl.register("mode", _ReactiveControl("AUTO"))
    ctrl.register("hc_mode", _ReactiveControl("Adaptive"))
    ctrl.register("auto_goal", _ReactiveControl("balanced"))
    ctrl.register("auto_target_mode", _ReactiveControl("adaptive"))
    ctrl.register("hc_custom_file", ctrl._ValueHolder(None))
    ctrl.register("_manual_hc_mode", ctrl._ValueHolder("Harman8"))

    monkeypatch.setattr("decaycore.ui.ng_callbacks._update_target_preview", lambda: None)
    monkeypatch.setattr("decaycore.ui.ng_mode_controls.update_target_curve_controls_ui", lambda: None)

    _register_target_callbacks(t=_t)
    ctrl.set_value("auto_target_mode", "selected")

    assert ctrl.value("hc_mode") == "Harman8"


def test_bass_integration_scopes_stay_hidden_outside_auto_mode() -> None:
    ctrl.reset()
    ctrl.register("mode", _ReactiveControl("BASIC"))
    ctrl.register("bass_integration_enable", _ReactiveControl(True))
    ctrl.register("bass_integration_mode", _ReactiveControl("direct_dac"))
    ctrl.register("sub_crossover_manual_override", _ReactiveControl(True))
    ctrl.register("sub_crossover_hz", _ToggleControl(80.0))
    ctrl.register("sub_crossover_slope", _ToggleControl(24))
    ctrl.register("bass_integration_allpass_auto_enable", _ToggleControl(False))

    legacy_scope = _DummyContainer()
    direct_scope = _DummyContainer()
    xo_scope = _DummyContainer()
    basic_direct_scope = _DummyContainer()
    ctrl.register_container("files_legacy_topology_scope", legacy_scope)
    ctrl.register_container("files_direct_dac_topology_scope", direct_scope)
    ctrl.register_container("bass_integration_xo_info_scope", xo_scope)
    ctrl.register_container("bass_integration_direct_scope", basic_direct_scope)

    _sync_bass_integration_visibility()

    assert legacy_scope.visible is True
    assert direct_scope.visible is False
    assert xo_scope.visible is False
    assert basic_direct_scope.visible is False
    assert ctrl.get("sub_crossover_hz").enabled is False
    assert ctrl.get("sub_crossover_slope").enabled is False
    assert ctrl.get("bass_integration_allpass_auto_enable").enabled is False


def test_filter_type_callback_updates_xo_tab_state() -> None:
    ctrl.reset()
    ctrl.register("filter_type", _ReactiveControl("Asymmetric"))
    ctrl.register("tab_xo", _ToggleControl(None))
    ctrl.register("xo1_f", _ToggleControl(500.0))
    ctrl.register("xo1_s", _ToggleControl(12))
    xo_scope = _DummyContainer()
    ctrl.register_container("xo_tab_content_scope", xo_scope)

    _register_ir_window_callbacks(t=_t)
    ctrl.set_value("filter_type", "Mixed")

    assert ctrl.get("tab_xo").enabled is False
    assert ctrl.get("xo1_f").enabled is False
    assert ctrl.get("xo1_s").enabled is False
    assert xo_scope.visible is False
