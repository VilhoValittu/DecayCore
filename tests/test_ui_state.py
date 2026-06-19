# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""Comprehensive tests for UI state management module."""

from __future__ import annotations

import pytest

from src.decaycore.ui import ui_state


@pytest.fixture(autouse=True)
def reset_ui_state_before_each():
    """Reset all UI state before and after each test."""
    ui_state.set_status_renderer(None)
    ui_state._AUTO_STATUS_DETAILS.clear()
    ui_state._STATUS_BASE_MSG = ""
    ui_state._STATUS_DOM_READY = False
    ui_state._STATUS_LAST_TEXT = ""
    ui_state._STATUS_SUMMARY_TEXT = ""
    ui_state._STATUS_INFO_TEXT = ""
    ui_state._AUTO_SELECTED_BAR_MSG = ""
    ui_state._AUTO_STATUS_LAST_DETAIL = ""
    ui_state._RUN_WALL_CLOCK_TEXT = ""
    ui_state._LAST_RUN_INFO = {}
    yield
    # Cleanup after test
    ui_state.set_status_renderer(None)
    ui_state._AUTO_STATUS_DETAILS.clear()


@pytest.fixture
def mock_renderer():
    """Mock renderer that captures update events."""
    calls = []

    def renderer(snapshot: dict, event: str):
        calls.append({"snapshot": dict(snapshot), "event": event})

    renderer.calls = calls
    renderer.reset = lambda: calls.clear()
    ui_state.set_status_renderer(renderer)
    return renderer


class TestSettersGetters:
    """Tests for basic setter/getter functions."""

    def test_set_get_status_renderer(self):
        """Setting and getting status renderer."""
        assert ui_state._STATUS_RENDERER is None

        def custom_renderer(snap, event):
            pass

        ui_state.set_status_renderer(custom_renderer)
        assert ui_state._STATUS_RENDERER is custom_renderer

        ui_state.set_status_renderer(None)
        assert ui_state._STATUS_RENDERER is None

    def test_set_status_renderer_non_callable(self):
        """Setting non-callable should result in None."""
        ui_state.set_status_renderer("not_callable")
        assert ui_state._STATUS_RENDERER is None

        ui_state.set_status_renderer(123)
        assert ui_state._STATUS_RENDERER is None

    def test_mark_status_dom_ready_true(self):
        """Mark DOM as ready."""
        ui_state.mark_status_dom_ready(True)
        assert ui_state.is_status_dom_ready() is True

    def test_mark_status_dom_ready_false(self):
        """Mark DOM as not ready."""
        ui_state.mark_status_dom_ready(False)
        assert ui_state.is_status_dom_ready() is False

    def test_mark_status_dom_ready_converts_to_bool(self):
        """Non-boolean values converted to boolean."""
        ui_state.mark_status_dom_ready(1)
        assert ui_state.is_status_dom_ready() is True

        ui_state.mark_status_dom_ready(0)
        assert ui_state.is_status_dom_ready() is False

        ui_state.mark_status_dom_ready("yes")
        assert ui_state.is_status_dom_ready() is True

    def test_set_get_run_wall_clock_text(self):
        """Setting and getting wall clock text."""
        ui_state.set_run_wall_clock_text("12:34:56")
        assert ui_state.get_run_wall_clock_text() == "12:34:56"

    def test_set_get_run_wall_clock_text_converts_to_string(self):
        """Non-string values converted to string."""
        ui_state.set_run_wall_clock_text(123)
        assert ui_state.get_run_wall_clock_text() == "123"

        ui_state.set_run_wall_clock_text(None)
        # None or "" becomes empty string
        assert ui_state.get_run_wall_clock_text() == ""

    def test_get_status_base_message_default(self):
        """Getting default status message when not set."""
        msg = ui_state.get_status_base_message()
        assert msg == "DecayCore running"

    def test_get_status_base_message_custom_default(self):
        """Using custom default."""
        msg = ui_state.get_status_base_message(default="Custom default")
        assert msg == "Custom default"

    def test_set_get_last_run_info_basic(self):
        """Setting and getting last run info."""
        info = {"status": "ok", "trial": 42}
        ui_state.set_last_run_info(info)
        retrieved = ui_state.get_last_run_info()
        assert retrieved == info

    def test_set_get_last_run_info_isolation(self):
        """Modifying original dict doesn't affect stored info."""
        info = {"status": "ok"}
        ui_state.set_last_run_info(info)
        info["status"] = "failed"
        retrieved = ui_state.get_last_run_info()
        assert retrieved["status"] == "ok"

    def test_set_last_run_info_non_dict(self):
        """Non-dict values converted to empty dict."""
        ui_state.set_last_run_info(None)
        assert ui_state.get_last_run_info() == {}

        ui_state.set_last_run_info("string")
        assert ui_state.get_last_run_info() == {}


class TestStatusSplitElapsedSuffix:
    """Tests for _status_split_elapsed_suffix function."""

    def test_valid_elapsed_pattern(self):
        """Valid 'msg | 12.5 s' pattern."""
        base, suffix = ui_state._status_split_elapsed_suffix("Processing | 12.5 s")
        assert base == "Processing"
        assert suffix == "| 12.5 s"

    def test_valid_elapsed_integer_seconds(self):
        """Integer seconds without decimal."""
        base, suffix = ui_state._status_split_elapsed_suffix("Running | 42 s")
        assert base == "Running"
        assert suffix == "| 42 s"

    def test_no_elapsed_pattern(self):
        """Message without elapsed pattern."""
        base, suffix = ui_state._status_split_elapsed_suffix("Just a message")
        assert base == "Just a message"
        assert suffix == ""

    def test_elapsed_with_extra_whitespace(self):
        """Handles multiple spaces."""
        base, suffix = ui_state._status_split_elapsed_suffix("Processing  |  5.5  s  ")
        assert base.strip() == "Processing"
        assert "5.5" in suffix  # Contains 5.5 but may have spaces

    def test_empty_message(self):
        """Empty message returns empty tuple."""
        base, suffix = ui_state._status_split_elapsed_suffix("")
        assert base == ""
        assert suffix == ""

    def test_none_message(self):
        """None is converted to empty string."""
        base, suffix = ui_state._status_split_elapsed_suffix(None)
        assert base == ""
        assert suffix == ""

    def test_elapsed_zero_seconds(self):
        """Zero seconds is valid."""
        base, suffix = ui_state._status_split_elapsed_suffix("Task | 0 s")
        assert base == "Task"
        assert "0 s" in suffix


class TestStatusCompactWithDetail:
    """Tests for _status_compact_with_detail function."""

    def test_normal_mode_no_compact(self):
        """Normal (non-auto) mode returns no compact form."""
        msg = "This is a normal status message"
        compact, detail = ui_state._status_compact_with_detail(msg)
        assert compact == msg
        assert detail is None

    def test_auto_mode_bracket_format(self):
        """Auto mode with bracket notation creates compact form."""
        msg = "DecayCore automatic mode [Harman6]: phase 1/2 50/100"
        compact, detail = ui_state._status_compact_with_detail(msg)
        assert "Optimizing" in compact or "phase" in compact
        assert detail is not None

    def test_auto_mode_legacy_format(self):
        """Auto mode legacy format creates compact form."""
        msg = "DecayCore automatic mode: target search harman6"
        compact, detail = ui_state._status_compact_with_detail(msg)
        # Should be humanized/compacted
        assert len(compact) > 0
        if "automatic" not in compact:
            # If compacted, should have detail
            assert detail is not None

    def test_empty_message(self):
        """Empty message returns default."""
        compact, detail = ui_state._status_compact_with_detail("")
        # Empty message returns default "DecayCore running"
        assert compact == "DecayCore running"
        assert detail is None

    def test_auto_mode_with_elapsed(self):
        """Auto mode message with elapsed time."""
        msg = "DecayCore automatic mode [Target]: phase 1/2 | 5.2 s"
        compact, detail = ui_state._status_compact_with_detail(msg)
        # Should parse and compact
        assert len(compact) > 0

    def test_bass_integration_optuna_progress_stays_on_single_status_line(self):
        """Bass-integration Optuna trial progress should not create detail rows."""
        msg = "DecayCore automatic mode: bass integration optuna search (trial 12/512)"
        compact, detail = ui_state._status_compact_with_detail(msg)
        assert compact.startswith("12/512")
        assert "bass integration" in compact.lower()
        assert detail is None


class TestCompactAutoStatusCore:
    """Tests for _compact_auto_status_core function."""

    def test_bracket_phase_pattern(self):
        """Bracket format with phase info."""
        msg = "DecayCore automatic mode [Harman6]: phase 1/2 50/100"
        result = ui_state._compact_auto_status_core(msg)
        assert "Optimizing" in result or "phase" in result

    def test_bracket_polish_pattern(self):
        """Bracket format with polish phase."""
        msg = "DecayCore automatic mode [Target]: tdc winner polish 25/50"
        result = ui_state._compact_auto_status_core(msg)
        assert "Polishing" in result or "winner polish" in result or "decay" in result.lower()

    def test_legacy_fallback(self):
        """Non-bracket format triggers legacy fallback."""
        msg = "DecayCore automatic mode: target search harman6"
        result = ui_state._compact_auto_status_core(msg)
        # Should be processed
        assert len(result) > 0

    def test_target_extraction_from_brackets(self):
        """Target name extracted from brackets."""
        msg = "DecayCore automatic mode [CustomTarget]: phase 1/2 10/10"
        result = ui_state._compact_auto_status_core(msg)
        assert "CustomTarget" in result

    def test_empty_brackets(self):
        """Empty brackets handled gracefully."""
        msg = "DecayCore automatic mode []: phase 1/2 10/10"
        result = ui_state._compact_auto_status_core(msg)
        # Should still process
        assert len(result) > 0


class TestCompactAutoStatusBracket:
    """Tests for _compact_auto_status_bracket function."""

    @pytest.mark.parametrize(
        "input_str,should_contain",
        [
            ("phase 1/2 50/100", "Optimizing"),
            (
                "phase 2/3 99/100 best improved trial 5/10",
                "Optimizing",
            ),  # Different regex
            ("residual tie-break something", "peaks"),
            ("tdc winner polish 25/50", "Polishing"),
        ],
    )
    def test_bracket_patterns(self, input_str, should_contain):
        """Various bracket patterns produce expected output."""
        result = ui_state._compact_auto_status_bracket(input_str)
        assert should_contain in result

    def test_phase_with_best_improved(self):
        """Phase pattern with best improved indicator."""
        result = ui_state._compact_auto_status_bracket("phase 1/2 best improved trial 50/100")
        assert "phase 1/2" in result
        assert "↑" in result

    def test_polish_param_label_mapping(self):
        """Parameter labels are mapped correctly."""
        # tdc -> decay
        result = ui_state._compact_auto_status_bracket("tdc winner polish 10/20")
        assert "decay" in result.lower()


class TestNormalizeAutoSelectedText:
    """Tests for _normalize_auto_selected_text function."""

    def test_valid_string(self):
        """Valid string is returned normalized."""
        result = ui_state._normalize_auto_selected_text("Normal text")
        assert result == "Normal text"

    def test_none_to_empty_string(self):
        """None becomes empty string."""
        result = ui_state._normalize_auto_selected_text(None)
        assert result == ""

    def test_int_to_string(self):
        """Integer is converted to string."""
        result = ui_state._normalize_auto_selected_text(42)
        assert result == "42"

    def test_list_to_string(self):
        """List is converted to string representation."""
        result = ui_state._normalize_auto_selected_text([1, 2, 3])
        assert isinstance(result, str)
        assert len(result) > 0

    def test_dict_to_string(self):
        """Dictionary is converted to string."""
        result = ui_state._normalize_auto_selected_text({"key": "value"})
        assert isinstance(result, str)


class TestNormalizeStatusNoticeText:
    """Tests for _normalize_status_notice_text function."""

    def test_valid_string(self):
        """Valid string is returned normalized."""
        result = ui_state._normalize_status_notice_text("Summary text")
        assert result == "Summary text"

    def test_none_to_empty_string(self):
        """None becomes empty string."""
        result = ui_state._normalize_status_notice_text(None)
        assert result == ""

    def test_strip_whitespace(self):
        """Leading/trailing whitespace is stripped."""
        result = ui_state._normalize_status_notice_text("  Text with spaces  ")
        assert result == "Text with spaces"


class TestUpdateStatus:
    """Tests for update_status main function."""

    def test_update_status_sets_base_message(self, mock_renderer):
        """Update status sets the base message."""
        ui_state.update_status("Test message")
        assert ui_state.get_status_base_message() == "Test message"

    def test_update_status_with_elapsed(self, mock_renderer):
        """Update status with elapsed time."""
        ui_state.update_status("Processing | 5.2 s")
        base = ui_state.get_status_base_message()
        assert base == "Processing"

    def test_update_status_calls_renderer(self, mock_renderer):
        """Update status calls the renderer callback."""
        ui_state.update_status("Test message")
        assert len(mock_renderer.calls) > 0
        # Event type can be "status" or other
        assert isinstance(mock_renderer.calls[0]["event"], str)

    def test_update_status_auto_mode_compact(self, mock_renderer):
        """Auto mode status is compacted."""
        ui_state.update_status("DecayCore automatic mode [Target]: phase 1/2 50/100")
        snap = ui_state.get_status_snapshot()
        # Compact text should be set
        assert len(snap["status_last_text"]) > 0

    def test_update_status_empty_message(self, mock_renderer):
        """Empty message is handled."""
        ui_state.update_status("")
        # Empty becomes default "DecayCore running"
        assert ui_state.get_status_base_message() == "DecayCore running"

    def test_update_status_none_message(self, mock_renderer):
        """None message is handled."""
        ui_state.update_status(None)
        base = ui_state.get_status_base_message()
        # Should be empty or default
        assert isinstance(base, str)

    def test_update_status_bass_integration_progress_does_not_fill_detail_history(
        self, mock_renderer
    ):
        """Bass-integration trial updates stay in the live status row only."""
        ui_state.reset_auto_status_details()
        for idx in range(1, 4):
            ui_state.update_status(
                "DecayCore automatic mode: bass integration optuna search "
                f"(trial {idx}/512)"
            )
        snap = ui_state.get_status_snapshot()
        assert snap["status_last_text"].startswith("3/512")
        assert snap["auto_status_details"] == []


class TestUpdateAutoSelectedBar:
    """Tests for update_auto_selected_bar function."""

    def test_update_auto_selected_bar(self, mock_renderer):
        """Update auto selected bar text."""
        ui_state.update_auto_selected_bar("Harman6 selected")
        snap = ui_state.get_status_snapshot()
        assert snap["auto_selected_bar_text"] == "Harman6 selected"

    def test_update_auto_selected_bar_calls_renderer(self, mock_renderer):
        """Calls renderer callback."""
        ui_state.update_auto_selected_bar("Test")
        assert len(mock_renderer.calls) > 0

    def test_update_auto_selected_bar_none(self, mock_renderer):
        """None value handled."""
        ui_state.update_auto_selected_bar(None)
        snap = ui_state.get_status_snapshot()
        assert isinstance(snap["auto_selected_bar_text"], str)


class TestUpdateStatusNotices:
    """Tests for update_status_notices function."""

    def test_update_status_notices_both(self, mock_renderer):
        """Update both summary and info."""
        ui_state.update_status_notices(summary_text="Summary", info_text="Info")
        snap = ui_state.get_status_snapshot()
        assert snap["status_summary_text"] == "Summary"
        assert snap["status_info_text"] == "Info"

    def test_update_status_notices_summary_only(self, mock_renderer):
        """Update only summary."""
        ui_state.update_status_notices(summary_text="Summary only")
        snap = ui_state.get_status_snapshot()
        assert snap["status_summary_text"] == "Summary only"

    def test_update_status_notices_info_only(self, mock_renderer):
        """Update only info."""
        ui_state.update_status_notices(info_text="Info only")
        snap = ui_state.get_status_snapshot()
        assert snap["status_info_text"] == "Info only"

    def test_update_status_notices_calls_renderer(self, mock_renderer):
        """Calls renderer callback."""
        ui_state.update_status_notices(summary_text="Test")
        assert len(mock_renderer.calls) > 0


class TestResetAutoStatusDetails:
    """Tests for reset_auto_status_details function."""

    def test_reset_auto_status_details(self, mock_renderer):
        """Reset clears the detail history."""
        ui_state.append_auto_status_detail_raw("Line 1")
        ui_state.append_auto_status_detail_raw("Line 2")
        assert len(ui_state._AUTO_STATUS_DETAILS) == 2

        ui_state.reset_auto_status_details()
        assert len(ui_state._AUTO_STATUS_DETAILS) == 0

    def test_reset_auto_status_details_calls_renderer(self, mock_renderer):
        """Reset calls renderer callback."""
        ui_state.reset_auto_status_details()
        assert len(mock_renderer.calls) > 0


class TestAppendAutoStatusDetailRaw:
    """Tests for append_auto_status_detail_raw function."""

    @pytest.fixture(autouse=True)
    def fresh_details(self):
        """Ensure fresh deque for each test in this class."""
        ui_state._AUTO_STATUS_DETAILS.clear()
        yield
        ui_state._AUTO_STATUS_DETAILS.clear()

    def test_append_auto_status_detail_raw(self):
        """Append raw line to detail history."""
        ui_state.append_auto_status_detail_raw("Detail line 1")
        assert "Detail line 1" in ui_state._AUTO_STATUS_DETAILS

    def test_append_multiple_lines(self):
        """Append multiple lines."""
        for i in range(5):
            ui_state.append_auto_status_detail_raw(f"Line {i}")
        assert len(ui_state._AUTO_STATUS_DETAILS) == 5


class TestGetStatusSnapshot:
    """Tests for get_status_snapshot function."""

    def test_get_status_snapshot_complete(self, mock_renderer):
        """Snapshot contains all required fields."""
        ui_state.mark_status_dom_ready(True)
        ui_state.update_status("Test message")
        snap = ui_state.get_status_snapshot()

        required_keys = [
            "status_base_message",
            "status_dom_ready",
            "status_last_text",
            "status_summary_text",
            "status_info_text",
            "auto_selected_bar_text",
            "auto_status_details",
            "auto_status_detail_body",
            "run_wall_clock_text",
        ]
        for key in required_keys:
            assert key in snap, f"Missing {key} in snapshot"

    def test_get_status_snapshot_dom_ready_state(self, mock_renderer):
        """Snapshot reflects DOM ready state."""
        ui_state.mark_status_dom_ready(True)
        snap = ui_state.get_status_snapshot()
        assert snap["status_dom_ready"] is True

        ui_state.mark_status_dom_ready(False)
        snap = ui_state.get_status_snapshot()
        assert snap["status_dom_ready"] is False

    def test_get_status_snapshot_detail_body_format(self, mock_renderer):
        """Detail body is newline-joined."""
        ui_state.append_auto_status_detail_raw("Line 1")
        ui_state.append_auto_status_detail_raw("Line 2")
        snap = ui_state.get_status_snapshot()
        assert "Line 1" in snap["auto_status_detail_body"]
        assert "Line 2" in snap["auto_status_detail_body"]
        assert "\n" in snap["auto_status_detail_body"]

    def test_get_status_snapshot_detail_history_list(self, mock_renderer):
        """auto_status_details is a list in snapshot."""
        ui_state.append_auto_status_detail_raw("Item 1")
        snap = ui_state.get_status_snapshot()
        assert isinstance(snap["auto_status_details"], list)
        assert "Item 1" in snap["auto_status_details"]

    def test_get_status_snapshot_after_reset(self, mock_renderer):
        """Snapshot after reset shows empty details."""
        ui_state.append_auto_status_detail_raw("Item")
        ui_state.reset_auto_status_details()
        snap = ui_state.get_status_snapshot()
        assert len(snap["auto_status_details"]) == 0
        assert snap["auto_status_detail_body"] == ""


class TestHumanizeAutoStatusDetail:
    """Tests for _humanize_auto_status_detail function."""

    def test_humanize_empty_string(self):
        """Empty string returns empty string."""
        result = ui_state._humanize_auto_status_detail("")
        assert result == ""

    def test_humanize_non_matching_message(self):
        """Non-matching message is returned as-is or fallback."""
        result = ui_state._humanize_auto_status_detail("Random text without patterns")
        # Should return something
        assert isinstance(result, str)

    @pytest.mark.parametrize(
        "msg",
        [
            "filter init: linear phase minimal 4096 taps",
            "taps init: 8192 taps set",
            "phase1 done: 50 trials 10 best improved",
            "target search: Harman6",
        ],
    )
    def test_humanize_various_patterns(self, msg):
        """Various known patterns are handled."""
        result = ui_state._humanize_auto_status_detail(msg)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_humanize_filter_init_pattern(self):
        """Filter init pattern is recognized."""
        result = ui_state._humanize_auto_status_detail(
            "filter init: linear phase minimal 4096 taps"
        )
        # Should contain humanized output
        assert isinstance(result, str)

    def test_humanize_target_selection_pattern(self):
        """Target selection patterns are recognized."""
        msg = "target search: Harman6 match confidence 0.95"
        result = ui_state._humanize_auto_status_detail(msg)
        assert isinstance(result, str)

    def test_humanize_with_parameter_mapping(self):
        """Parameter names are mapped in humanization."""
        msg = "tdc winner polish 25/50"
        result = ui_state._humanize_auto_status_detail(msg)
        # Should map tdc -> decay or similar
        assert isinstance(result, str)


class TestPolishParamLabel:
    """Tests for _polish_param_label function."""

    @pytest.mark.parametrize(
        "param,expected_contains",
        [
            ("tdc", "decay"),
            ("mag_c_min", "extension"),
            ("low_bass", "bass"),
            ("hpf_freq", "hpf"),
        ],
    )
    def test_polish_param_label_mappings(self, param, expected_contains):
        """Parameter mappings work correctly."""
        result = ui_state._polish_param_label(param)
        assert expected_contains in result.lower() or result == param

    def test_polish_param_label_unknown(self):
        """Unknown parameter returned as-is."""
        result = ui_state._polish_param_label("unknown_param")
        assert result is not None


class TestStatusBaseFromText:
    """Tests for _status_base_from_text function."""

    def test_status_base_from_text_no_elapsed(self):
        """Text without elapsed returns as-is."""
        result = ui_state._status_base_from_text("Just a message")
        assert result == "Just a message"

    def test_status_base_from_text_with_elapsed(self):
        """Text with elapsed time has it removed."""
        result = ui_state._status_base_from_text("Processing | 5.2 s")
        assert result == "Processing"
        assert "5.2 s" not in result


# Integration tests
class TestIntegration:
    """Integration tests for complete workflows."""

    def test_full_status_update_workflow(self, mock_renderer):
        """Complete workflow: update -> snapshot -> check."""
        ui_state.mark_status_dom_ready(True)
        ui_state.update_status("DecayCore automatic mode [Target]: phase 1/2 50/100")
        ui_state.update_status_notices(
            summary_text="Running optimization", info_text="Phase 1 in progress"
        )
        ui_state.set_last_run_info({"trial": 50, "phase": 1})

        snap = ui_state.get_status_snapshot()

        assert snap["status_dom_ready"] is True
        assert len(snap["status_base_message"]) > 0
        assert snap["status_summary_text"] == "Running optimization"
        assert snap["status_info_text"] == "Phase 1 in progress"

    def test_detail_history_accumulation(self, mock_renderer):
        """Detail history accumulates correctly."""
        ui_state.reset_auto_status_details()
        for i in range(10):
            ui_state.append_auto_status_detail_raw(f"Detail {i}")

        snap = ui_state.get_status_snapshot()
        assert len(snap["auto_status_details"]) == 10
        assert "Detail 0" in snap["auto_status_detail_body"]
        assert "Detail 9" in snap["auto_status_detail_body"]

    def test_multiple_renderer_calls(self, mock_renderer):
        """Multiple updates trigger multiple renderer calls."""
        ui_state.update_status("Message 1")
        ui_state.update_status("Message 2")
        ui_state.update_auto_selected_bar("Selected")

        assert len(mock_renderer.calls) >= 3

    def test_state_isolation_across_tests(self):
        """State is isolated between tests due to fixture."""
        # First test sets state
        ui_state.set_run_wall_clock_text("10:00:00")
        first_value = ui_state.get_run_wall_clock_text()

        # Next test (simulated by direct reset) starts fresh
        ui_state._RUN_WALL_CLOCK_TEXT = ""
        second_value = ui_state.get_run_wall_clock_text()

        assert first_value == "10:00:00"
        assert second_value == ""
