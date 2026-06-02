from decaycore.ui.ng_tab_basic import (
    _auto_target_mode_options,
    _normalize_auto_target_mode_value,
    _normalize_filter_type_value,
)


def test_normalize_filter_type_value_accepts_legacy_mixed_label():
    assert _normalize_filter_type_value("Mixed") == "Mixed"
    assert _normalize_filter_type_value("Mixed Phase") == "Mixed"


def test_normalize_filter_type_value_accepts_legacy_asymmetric_variants():
    assert _normalize_filter_type_value("Asymmetric") == "Asymmetric"
    assert _normalize_filter_type_value("Asymmetric (low-latency)") == "Asymmetric"


def test_normalize_auto_target_mode_value_accepts_adaptive_choice():
    assert _normalize_auto_target_mode_value("auto") == "auto"
    assert _normalize_auto_target_mode_value("selected") == "selected"
    assert _normalize_auto_target_mode_value("adaptive") == "adaptive"
    assert _normalize_auto_target_mode_value("Adaptive target") == "adaptive"
    assert _normalize_auto_target_mode_value("manual") == "selected"
    assert _normalize_auto_target_mode_value("invalid") == "auto"


def test_prefer_bass_forces_selected_auto_target_mode():
    assert _normalize_auto_target_mode_value("auto", auto_goal="flat") == "selected"
    assert _normalize_auto_target_mode_value("auto", auto_goal="prefer bass") == "selected"


def test_prefer_bass_auto_target_options_only_include_selected():
    labels = {
        "auto_target_mode_auto": "Auto: search best built-in",
        "auto_target_mode_adaptive": "Adaptive: derive target from room acoustics",
        "auto_target_mode_selected": "Use selected target curve from Target page",
    }

    assert _auto_target_mode_options(t=labels.__getitem__, auto_goal="flat") == {
        "selected": "Use selected target curve from Target page",
    }
    assert _auto_target_mode_options(t=labels.__getitem__, auto_goal="balanced") == {
        "auto": "Auto: search best built-in",
        "adaptive": "Adaptive: derive target from room acoustics",
        "selected": "Use selected target curve from Target page",
    }
    assert _auto_target_mode_options(t=labels.__getitem__, auto_goal="room-safe") == {
        "auto": "Auto: search best built-in",
        "adaptive": "Adaptive: derive target from room acoustics",
        "selected": "Use selected target curve from Target page",
    }
    assert _auto_target_mode_options(t=labels.__getitem__, auto_goal="subwoofers") == {
        "auto": "Auto: search best built-in",
        "adaptive": "Adaptive: derive target from room acoustics",
        "selected": "Use selected target curve from Target page",
    }
