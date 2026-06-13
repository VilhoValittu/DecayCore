import numpy as np

from decaycore.ui import ng_tab_target


def test_target_decay_hint_payload_returns_unavailable_without_metadata(monkeypatch):
    monkeypatch.setattr(ng_tab_target.ctrl, "value", lambda name, default=None: default)

    result = ng_tab_target._build_target_decay_hint_payload()

    assert result["status"] == "unavailable"
    assert result["has_data"] is False


def test_target_decay_hint_payload_uses_generated_measurement_metadata(monkeypatch):
    generated = {
        "measured_rt60": 0.52,
        "measured_rt60_bands": {63.0: 0.55, 125.0: 0.46, 250.0: 0.42},
        "analysis_freq_hz": np.array([20.0, 40.0, 80.0, 160.0, 320.0], dtype=float),
        "analysis_magnitude_db": np.array([-3.0, -2.0, -1.0, -1.5, -2.0], dtype=float),
        "harmonic_freq_hz": np.array([20.0, 40.0, 80.0, 160.0, 320.0], dtype=float),
        "harmonic_magnitudes_db": {
            2: np.array([-36.0, -33.0, -31.0, -38.0, -45.0], dtype=float),
        },
    }
    values = {
        "mode": "ADVANCED",
        "bass_integration_enable": False,
        "generated_measurement_l": generated,
        "generated_measurement_r": None,
        "local_path_l": "",
        "local_path_r": "",
    }
    monkeypatch.setattr(ng_tab_target.ctrl, "value", lambda name, default=None: values.get(name, default))

    result = ng_tab_target._build_target_decay_hint_payload()

    assert result["status"] in {"caution", "strong"}
    assert result["has_data"] is True


def test_target_decay_hint_payload_uses_main_sidecars_in_bass_integration_mode(monkeypatch):
    values = {
        "mode": "BASIC",
        "bass_integration_enable": True,
        "generated_measurement_l": None,
        "generated_measurement_r": None,
        "local_path_l_main": "left_main.wav",
        "local_path_r_main": "right_main.wav",
    }
    monkeypatch.setattr(ng_tab_target.ctrl, "value", lambda name, default=None: values.get(name, default))
    monkeypatch.setattr(
        "decaycore.io.measurements_loader._try_load_rt60_sidecar",
        lambda path: (0.0, {63.0: 0.80, 125.0: 0.72}) if "left" in path else (None, None),
    )
    monkeypatch.setattr(
        "decaycore.io.measurements_loader._try_load_harmonic_sidecar",
        lambda path: (None, None),
    )

    result = ng_tab_target._build_target_decay_hint_payload()

    assert result["status"] == "strong"
    assert result["has_data"] is True
