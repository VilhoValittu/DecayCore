from __future__ import annotations

from decaycore.ui.measurement_session_dialog import _format_session_progress_percent


def test_format_session_progress_percent_returns_percent_text() -> None:
    assert _format_session_progress_percent(0, 0) == "0%"
    assert _format_session_progress_percent(0, 10) == "0%"
    assert _format_session_progress_percent(1, 4) == "25%"
    assert _format_session_progress_percent(2, 3) == "67%"
    assert _format_session_progress_percent(10, 10) == "100%"


def test_format_session_progress_percent_clamps_bounds() -> None:
    assert _format_session_progress_percent(-1, 4) == "0%"
    assert _format_session_progress_percent(6, 4) == "100%"
