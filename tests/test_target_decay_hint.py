import numpy as np

from decaycore.ui import ng_tab_target


def test_target_preview_metadata_payload_returns_empty_without_metadata(monkeypatch):
    monkeypatch.setattr(ng_tab_target.ctrl, "value", lambda name, default=None: default)

    result = ng_tab_target._build_target_preview_metadata_payload()

    assert result["has_any_metadata"] is False
    assert result["has_rt60"] is False
    assert result["has_harmonics"] is False
    assert result["has_harmonic_risk"] is False


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
    monkeypatch.setattr(
        "decaycore.io.generated_measurement_source.generated_source_matches_upload",
        lambda generated, upload: True,
    )

    result = ng_tab_target._build_target_decay_hint_payload()

    assert result["status"] in {"caution", "strong"}
    assert result["has_data"] is True


def test_target_preview_metadata_payload_uses_generated_measurement_metadata(monkeypatch):
    generated = {
        "measured_rt60": 0.52,
        "measured_rt60_bands": {63.0: 0.55, 125.0: 0.46, 250.0: 0.42},
        "analysis_freq_hz": np.array([20.0, 40.0, 80.0, 160.0, 320.0], dtype=float),
        "analysis_magnitude_db": np.array([-3.0, -2.0, -1.0, -1.5, -2.0], dtype=float),
        "harmonic_freq_hz": np.array([20.0, 40.0, 80.0, 160.0, 320.0], dtype=float),
        "harmonic_magnitudes_db": {
            2: np.array([-36.0, -33.0, -31.0, -38.0, -45.0], dtype=float),
            3: np.array([-42.0, -40.0, -37.0, -41.0, -48.0], dtype=float),
        },
    }
    values = {
        "mode": "ADVANCED",
        "bass_integration_enable": False,
        "generated_measurement_l": generated,
        "generated_measurement_r": None,
        "local_path_l": "",
        "local_path_r": "",
        "file_l": {"filename": "left.wav"},
    }
    monkeypatch.setattr(ng_tab_target.ctrl, "value", lambda name, default=None: values.get(name, default))
    monkeypatch.setattr(
        "decaycore.io.generated_measurement_source.generated_source_matches_upload",
        lambda generated, upload: True,
    )

    result = ng_tab_target._build_target_preview_metadata_payload()

    assert result["has_any_metadata"] is True
    assert result["has_rt60"] is True
    assert result["has_harmonics"] is True
    assert result["has_harmonic_risk"] is True
    assert result["channels"]["L"]["source_kind"] == "generated"
    assert result["channels"]["L"]["rt60_bands"] == {63.0: 0.55, 125.0: 0.46, 250.0: 0.42}
    assert result["channels"]["L"]["harmonic_risk_curve"] is not None
    assert result["channels"]["R"]["source_kind"] == "none"


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


def test_target_preview_metadata_payload_uses_main_sidecars_in_bass_integration_mode(monkeypatch):
    values = {
        "mode": "BASIC",
        "bass_integration_enable": True,
        "generated_measurement_l": {
            "measured_rt60": 0.10,
            "measured_rt60_bands": {63.0: 0.10},
        },
        "generated_measurement_r": None,
        "local_path_l_main": "left_main.wav",
        "local_path_r_main": "right_main.wav",
    }
    seen_rt60_paths = []
    seen_harm_paths = []

    def _fake_try_load_rt60_sidecar(path: str):
        seen_rt60_paths.append(path)
        if "left" in path:
            return 0.0, {63.0: 0.80, 125.0: 0.72}
        return None, None

    def _fake_try_load_harmonic_sidecar(path: str):
        seen_harm_paths.append(path)
        if "left" in path:
            return (
                np.array([20.0, 40.0, 80.0, 160.0], dtype=float),
                {2: np.array([-35.0, -32.0, -33.0, -40.0], dtype=float)},
            )
        return None, None

    monkeypatch.setattr(ng_tab_target.ctrl, "value", lambda name, default=None: values.get(name, default))
    monkeypatch.setattr(
        "decaycore.io.measurements_loader._try_load_rt60_sidecar",
        _fake_try_load_rt60_sidecar,
    )
    monkeypatch.setattr(
        "decaycore.io.measurements_loader._try_load_harmonic_sidecar",
        _fake_try_load_harmonic_sidecar,
    )

    result = ng_tab_target._build_target_preview_metadata_payload()

    assert seen_rt60_paths == ["left_main.wav", "right_main.wav"]
    assert seen_harm_paths == ["left_main.wav", "right_main.wav"]
    assert result["channels"]["L"]["source_kind"] == "sidecar"
    assert result["channels"]["L"]["rt60_bands"] == {63.0: 0.80, 125.0: 0.72}
    assert result["channels"]["L"]["harmonic_magnitudes_db"] is not None
    assert result["channels"]["R"]["source_kind"] == "sidecar"


def test_target_preview_metadata_payload_uses_measurement_curve_as_risk_fundamental(monkeypatch):
    values = {
        "mode": "BASIC",
        "bass_integration_enable": False,
        "generated_measurement_l": None,
        "generated_measurement_r": None,
        "file_l": None,
        "local_path_l": "left.wav",
        "local_path_r": "",
        "ir_window_left": 85.0,
        "ir_window_right": 500.0,
    }

    monkeypatch.setattr(ng_tab_target.ctrl, "value", lambda name, default=None: values.get(name, default))
    monkeypatch.setattr(
        "decaycore.io.measurements_loader._try_load_rt60_sidecar",
        lambda path: (None, None),
    )
    monkeypatch.setattr(
        "decaycore.io.measurements_loader._try_load_harmonic_sidecar",
        lambda path: (
            np.array([20.0, 40.0, 80.0, 160.0], dtype=float),
            {2: np.array([-50.0, -48.0, -45.0, -43.0], dtype=float)},
        ) if "left" in path else (None, None),
    )
    monkeypatch.setattr(
        "decaycore.ui.target_preview_cache.load_path_measurement_curve",
        lambda path, **kwargs: (
            np.array([20.0, 40.0, 80.0, 160.0], dtype=float),
            np.array([-20.0, -20.0, -20.0, -20.0], dtype=float),
        ) if "left" in path else (None, None),
    )

    result = ng_tab_target._build_target_preview_metadata_payload()

    risk_curve = np.asarray(result["channels"]["L"]["harmonic_risk_curve"], dtype=float)
    assert np.any(risk_curve > 0.0)


def test_target_preview_metadata_plot_builders_cover_rt60_harmonics_and_risk():
    metadata_payload = {
        "channels": {
            "L": {
                "rt60_bands": {63.0: 0.60, 125.0: 0.48},
                "harmonic_freq_hz": np.array([20.0, 40.0, 80.0, 160.0], dtype=float),
                "harmonic_magnitudes_db": {
                    2: np.array([-35.0, -32.0, -33.0, -40.0], dtype=float),
                },
                "harmonic_risk_freq_hz": np.array([20.0, 40.0, 80.0, 160.0], dtype=float),
                "harmonic_risk_curve": np.array([0.2, 0.4, 0.5, 0.1], dtype=float),
            },
            "R": {
                "rt60_bands": {63.0: 0.72, 125.0: 0.58},
                "harmonic_freq_hz": np.array([20.0, 40.0, 80.0, 160.0], dtype=float),
                "harmonic_magnitudes_db": {
                    2: np.array([-38.0, -34.0, -35.0, -42.0], dtype=float),
                },
                "harmonic_risk_freq_hz": np.array([20.0, 40.0, 80.0, 160.0], dtype=float),
                "harmonic_risk_curve": np.array([0.1, 0.3, 0.4, 0.2], dtype=float),
            },
        }
    }

    rt60_fig = ng_tab_target._build_target_preview_rt60_fig(metadata_payload)
    harmonics_fig = ng_tab_target._build_target_preview_harmonics_fig(metadata_payload)
    risk_fig = ng_tab_target._build_target_preview_harmonic_risk_fig(metadata_payload)

    assert rt60_fig is not None
    assert [trace["name"] for trace in rt60_fig["data"]] == ["RT60 L", "RT60 R"]
    assert harmonics_fig is not None
    assert {"H2 L", "H2 R"} <= {trace["name"] for trace in harmonics_fig["data"]}
    assert risk_fig is not None
    assert {"Risk L", "Risk R"} == {trace["name"] for trace in risk_fig["data"]}
    assert risk_fig["layout"]["xaxis"]["range"] == [np.log10(20.0), np.log10(800.0)]
    assert abs(float(risk_fig["layout"]["yaxis"]["range"][1]) - 0.6) < 1e-9


def test_target_preview_harmonic_risk_fig_returns_none_for_flat_zero_risk():
    metadata_payload = {
        "channels": {
            "L": {
                "harmonic_risk_freq_hz": np.array([20.0, 40.0, 80.0, 160.0], dtype=float),
                "harmonic_risk_curve": np.zeros(4, dtype=float),
            },
            "R": {
                "harmonic_risk_freq_hz": np.array([20.0, 40.0, 80.0, 160.0], dtype=float),
                "harmonic_risk_curve": np.zeros(4, dtype=float),
            },
        }
    }

    risk_fig = ng_tab_target._build_target_preview_harmonic_risk_fig(metadata_payload)

    assert risk_fig is None
