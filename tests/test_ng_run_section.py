import sys
import types

from decaycore.resources.i8n.decaycore_i18n import t
from decaycore.ui import ng_bridge, ng_run_section, ui_state


class _DummyButton:
    def __init__(self) -> None:
        self.disable_calls = 0
        self.enable_calls = 0

    def disable(self) -> None:
        self.disable_calls += 1

    def enable(self) -> None:
        self.enable_calls += 1


class _DummyContainer:
    def __init__(self) -> None:
        self.clear_calls = 0

    def clear(self) -> None:
        self.clear_calls += 1


class _DummyProgress:
    def __init__(self) -> None:
        self.value = None
        self.text_color = None
        self.visible = None

    def set_value(self, value) -> None:
        self.value = value

    def set_text_color(self, color) -> None:
        self.text_color = color

    def set_visibility(self, visible: bool) -> None:
        self.visible = bool(visible)


class _ImmediateThread:
    def __init__(self, *args, target=None, daemon=None, **kwargs) -> None:
        self._target = target
        self.daemon = daemon

    def start(self) -> None:
        if callable(self._target):
            self._target()


def test_default_progress_bridge_queues_latest_value_without_touching_element():
    progress = _DummyProgress()
    ng_bridge.consume_pending_progress()
    ng_bridge.set_progress_element_getter(lambda: progress)

    try:
        bridge = ng_bridge.build_default_ui_bridge()
        bridge.set_progress(0.42)

        assert progress.value is None
        assert ng_bridge.consume_pending_progress() == 0.42
        assert ng_bridge.consume_pending_progress() is None

        bridge.set_progress(0.25)
        bridge.set_progress(0.75)

        assert progress.value is None
        assert ng_bridge.consume_pending_progress() == 0.75
    finally:
        ng_bridge.consume_pending_progress()
        ng_bridge.set_progress_element_getter(None)


def test_render_results_bridge_queues_latest_payload_once():
    ng_bridge.consume_pending_render_results()

    bridge = ng_bridge.build_default_ui_bridge()
    bridge.render_results("old", value=1)
    bridge.render_results("new", value=2)

    pending = ng_bridge.consume_pending_render_results()

    assert pending == (("new",), {"value": 2})
    assert ng_bridge.consume_pending_render_results() is None


def test_handle_start_clears_previous_results_and_status(monkeypatch):
    monkeypatch.setitem(sys.modules, "nicegui", types.SimpleNamespace(ui=object()))
    monkeypatch.setattr(ng_run_section.threading, "Thread", _ImmediateThread)
    ng_run_section._consume_pending_start_button_enable()

    container = _DummyContainer()
    progress = _DummyProgress()
    button = _DummyButton()
    run_clock = {"started_at": None, "active": False, "elapsed_s": None}
    run_calls: list[str] = []

    monkeypatch.setattr(ng_run_section, "_results_container_ref", container)
    monkeypatch.setattr(ng_run_section, "_progress_ref", progress)

    ui_state.set_last_run_info({"score": 97.5, "match": 88.0, "conf": 77.0})
    ui_state.update_status("DecayCore automatic mode: target shortlist (old)")
    ui_state.update_status("Done previously")
    ui_state.update_status_notices(summary_text="old summary", info_text="old info")
    ui_state.update_auto_selected_bar("old auto")

    def _on_start_click() -> None:
        run_calls.append("started")

    ng_run_section._handle_start(_on_start_click, button, run_clock)

    snap = ui_state.get_status_snapshot()

    assert container.clear_calls == 1
    assert ui_state.get_last_run_info() == {}
    assert ui_state.get_status_base_message() == t("stat_reading")
    assert snap["status_summary_text"] == ""
    assert snap["status_info_text"] == ""
    assert snap["auto_selected_bar_text"] == ""
    assert snap["auto_status_detail_body"] == ""
    assert progress.value == 0.0
    assert progress.text_color == "primary"
    assert progress.visible is True
    assert button.disable_calls == 1
    assert button.enable_calls == 0
    assert run_calls == ["started"]
    assert run_clock["started_at"] is not None
    assert run_clock["active"] is False
    assert run_clock["elapsed_s"] is not None

    pending_button = ng_run_section._consume_pending_start_button_enable()
    assert pending_button is button
    pending_button.enable()
    assert button.enable_calls == 1


def test_build_measurement_status_line_shows_loaded_sub_slots_for_bass_integration():
    values = {
        "file_l_main": {"name": "left.wav"},
        "file_r_main": {"name": "right.wav"},
        "local_path_l_sub": r"C:\measurements\sub1.wav",
        "local_path_r_sub": r"C:\measurements\sub2.wav",
    }
    labels = {
        "info_panel_meas": "Meas",
        "info_panel_sub1": "Sub1",
        "info_panel_sub2": "Sub2",
    }

    text, severity = ng_run_section._build_measurement_status_line(
        bass_integration_enabled=True,
        value_getter=values.get,
        tr=labels.__getitem__,
    )

    assert text == "Meas  L ✓  R ✓  Sub1 ✓  Sub2 ✓"
    assert severity == "ok"


def test_auto_details_autoscroll_only_when_running_and_body_changes():
    assert (
        ng_run_section._should_autoscroll_auto_details(
            previous_body="a",
            current_body="a\nb",
            run_active=True,
        )
        is True
    )
    assert (
        ng_run_section._should_autoscroll_auto_details(
            previous_body="a",
            current_body="a\nb",
            run_active=False,
        )
        is False
    )
    assert (
        ng_run_section._should_autoscroll_auto_details(
            previous_body="a",
            current_body="a",
            run_active=True,
        )
        is False
    )


def test_auto_details_autoscroll_js_uses_previous_scroll_height():
    js = ng_run_section._auto_details_autoscroll_js(123)

    assert "document.getElementById('c123')" in js
    assert "previousScrollHeight - sc.scrollTop - sc.clientHeight" in js
    assert "previousRemaining < 80" in js
    assert "cfAutoDetailsScrollHeight" in js
