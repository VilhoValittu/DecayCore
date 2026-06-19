import numpy as np

from decaycore.io import measurements_txt
from decaycore.ui import ng_tab_target
from decaycore.ui.plot_prediction import generate_prediction_plot


def test_target_preview_uses_light_theme(monkeypatch):
    values = {
        "hc_mode": "Flat",
        "lvl_min": 500.0,
        "lvl_max": 2000.0,
        "mag_c_min": 20.0,
        "mag_c_max": 200.0,
        "auto_goal": "balanced",
        "mode": "ADVANCED",
        "lvl_mode": "Manual",
        "lvl_manual_db": 1.5,
        "manual_target_tilt_db_per_oct": 0.0,
        "ir_window_left": 85.0,
        "ir_window_right": 500.0,
        "filter_smooth": 0,
    }

    monkeypatch.setattr(
        ng_tab_target.ctrl,
        "value",
        lambda name, default=None: values.get(name, default),
    )

    fig_dict, _drag_points, _tilt_points = ng_tab_target._build_target_preview_fig()

    assert fig_dict is not None
    layout = fig_dict["layout"]
    assert layout["paper_bgcolor"] == "#ffffff"
    assert layout["plot_bgcolor"] == "#ffffff"
    assert layout["font"]["color"] == "#1f2937"
    assert fig_dict["data"][1]["line"]["color"] == "rgba(15,23,42,0.35)"


def test_target_preview_speaker_avg_is_red(monkeypatch):
    values = {
        "hc_mode": "Flat",
        "lvl_min": 500.0,
        "lvl_max": 2000.0,
        "mag_c_min": 20.0,
        "mag_c_max": 200.0,
        "auto_goal": "balanced",
        "mode": "ADVANCED",
        "lvl_mode": "Auto",
        "ir_window_left": 85.0,
        "ir_window_right": 500.0,
        "filter_smooth": 0,
        "local_path_l": "left.txt",
        "local_path_r": "right.txt",
    }

    monkeypatch.setattr(
        ng_tab_target.ctrl,
        "value",
        lambda name, default=None: values.get(name, default),
    )

    def _fake_parse_txt_path(path, logger=None):
        freqs = np.array([20.0, 40.0, 80.0, 160.0, 320.0, 640.0, 1280.0, 2560.0, 5120.0, 10240.0], dtype=float)
        if "left" in str(path):
            mags = np.array([0.0, 0.8, 1.1, 0.7, 0.2, -0.3, -0.8, -1.1, -1.5, -1.8], dtype=float)
        else:
            mags = np.array([0.5, 0.9, 0.6, 0.1, -0.4, -0.8, -1.2, -1.6, -1.9, -2.1], dtype=float)
        return freqs, mags, None

    monkeypatch.setattr(measurements_txt, "parse_measurements_from_path", _fake_parse_txt_path)

    fig_dict, _drag_points, _tilt_points = ng_tab_target._build_target_preview_fig()

    assert fig_dict is not None
    avg_trace = next(trace for trace in fig_dict["data"] if trace.get("name") == "Speaker avg")
    assert avg_trace["line"]["color"] == "#dc2626"


def test_prediction_plot_uses_light_theme():
    freqs = np.array([20.0, 100.0, 1000.0, 10000.0, 20000.0], dtype=float)
    mags = np.zeros_like(freqs)
    phases = np.zeros_like(freqs)
    filt_ir = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

    _html, fig = generate_prediction_plot(
        freqs,
        mags,
        phases,
        filt_ir,
        48000,
        "Test",
        create_full_html=False,
        return_fig=True,
    )

    assert fig is not None
    assert fig.layout.paper_bgcolor == "#0e1219"
    assert fig.layout.plot_bgcolor == "#0e1219"
    assert fig.layout.font.color == "#c8cdd5"


def test_prediction_plot_uses_direct_dac_sum_traces_when_available():
    freqs = np.array([20.0, 40.0, 80.0, 160.0, 320.0, 640.0, 1280.0, 2560.0], dtype=float)
    mags = np.zeros_like(freqs)
    phases = np.zeros_like(freqs)
    filt_ir = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

    _html, fig = generate_prediction_plot(
        freqs,
        mags,
        phases,
        filt_ir,
        48000,
        "Test",
        target_stats={
            "freq_axis": freqs.tolist(),
            "measured_mags": mags.tolist(),
            "target_mags": np.zeros_like(freqs).tolist(),
            "direct_dac_sum_measured_mags": [6.0, 6.0, 6.0, 3.0, 0.0, 0.0, 0.0, 0.0],
            "direct_dac_sum_predicted_mags": [10.0, 10.0, 10.0, 7.0, 0.0, 0.0, 0.0, 0.0],
            "direct_dac_sum_predicted_mags_comp": [9.0, 9.0, 9.0, 6.0, 0.0, 0.0, 0.0, 0.0],
            "eff_target_db": 75.0,
            "offset_db": 0.0,
        },
        create_full_html=False,
        return_fig=True,
    )

    assert fig is not None
    measured_trace = next(trace for trace in fig.data if trace.name == "Measured")
    predicted_trace = next(trace for trace in fig.data if trace.name == "Predicted (exported)")
    assert float(np.asarray(measured_trace.y, dtype=float)[0]) > 80.0
    assert float(np.asarray(predicted_trace.y, dtype=float)[0]) > 84.0


def test_prediction_plot_aligns_low_level_raw_measurement_when_stats_curve_missing():
    freqs = np.array([20.0, 40.0, 80.0, 160.0, 320.0, 640.0, 1280.0, 2560.0, 5120.0, 10240.0], dtype=float)
    mags = np.array([-42.0, -41.0, -40.0, -39.0, -38.0, -38.0, -39.0, -40.0, -42.0, -45.0], dtype=float)
    phases = np.zeros_like(freqs)
    filt_ir = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

    _html, fig = generate_prediction_plot(
        freqs,
        mags,
        phases,
        filt_ir,
        48000,
        "Test",
        target_stats={
            "freq_axis": freqs.tolist(),
            "target_mags": np.zeros_like(freqs).tolist(),
            "eff_target_db": 75.0,
            "offset_db": 0.0,
            "smart_scan_range": [320.0, 2560.0],
        },
        create_full_html=False,
        return_fig=True,
    )

    assert fig is not None
    measured_trace = next(trace for trace in fig.data if trace.name == "Measured")
    measured_y = np.asarray(measured_trace.y, dtype=float)
    assert float(np.nanmedian(measured_y)) > 70.0
    assert fig.layout.yaxis.autorange is True


def test_prediction_plot_normalizes_negative_native_measurement_reference_to_zero():
    freqs = np.array([20.0, 40.0, 80.0, 160.0, 320.0, 640.0, 1280.0, 2560.0], dtype=float)
    mags = np.array([-3.0, -2.0, -1.5, -1.0, -0.5, -0.5, -0.8, -1.0], dtype=float)
    phases = np.zeros_like(freqs)
    filt_ir = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

    _html, fig = generate_prediction_plot(
        freqs,
        mags,
        phases,
        filt_ir,
        48000,
        "Test",
        target_stats={
            "freq_axis": freqs.tolist(),
            "measured_mags": mags.tolist(),
            "direct_dac_sum_measured_mags": [-109.0, -108.0, -107.0, -106.0, -105.5, -105.2, -105.1, -105.0],
            "target_mags": np.zeros_like(freqs).tolist(),
            "direct_dac_sum_predicted_mags": [-108.0, -107.0, -106.0, -105.0, -104.8, -104.7, -104.7, -104.7],
            "eff_target_db": -105.0,
            "target_level_db_window": -105.0,
            "offset_db": 0.0,
            "smart_scan_range": [160.0, 1280.0],
        },
        create_full_html=False,
        return_fig=True,
    )

    assert fig is not None
    measured_trace = next(trace for trace in fig.data if trace.name == "Measured")
    target_trace = next(trace for trace in fig.data if trace.name == "Target")
    predicted_trace = next(trace for trace in fig.data if trace.name == "Predicted (exported)")

    measured_y = np.asarray(measured_trace.y, dtype=float)
    target_y = np.asarray(target_trace.y, dtype=float)
    predicted_y = np.asarray(predicted_trace.y, dtype=float)

    assert abs(float(np.nanmedian(target_y))) < 1.0
    assert abs(float(np.nanmedian(measured_y))) < 2.0
    assert abs(float(np.nanmedian(predicted_y))) < 2.0


def test_prediction_plot_keeps_negative_absolute_target_from_double_shift():
    freqs = np.array([20.0, 40.0, 80.0, 160.0, 320.0, 640.0, 1280.0, 2560.0], dtype=float)
    mags = np.array([-57.0, -56.0, -55.0, -54.5, -54.0, -54.0, -54.5, -55.0], dtype=float)
    phases = np.zeros_like(freqs)
    filt_ir = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    target_abs = np.array([-55.0, -55.0, -54.5, -54.0, -54.0, -54.5, -55.0, -55.5], dtype=float)

    _html, fig = generate_prediction_plot(
        freqs,
        mags,
        phases,
        filt_ir,
        48000,
        "Test",
        target_stats={
            "freq_axis": freqs.tolist(),
            "measured_mags": mags.tolist(),
            "target_mags": target_abs.tolist(),
            "eff_target_db": -54.5,
            "target_level_db_window": -54.5,
            "offset_db": 0.0,
            "smart_scan_range": [160.0, 1280.0],
        },
        create_full_html=False,
        return_fig=True,
    )

    assert fig is not None
    measured_trace = next(trace for trace in fig.data if trace.name == "Measured")
    target_trace = next(trace for trace in fig.data if trace.name == "Target")

    measured_y = np.asarray(measured_trace.y, dtype=float)
    target_y = np.asarray(target_trace.y, dtype=float)

    assert abs(float(np.nanmedian(target_y))) < 1.0
    assert abs(float(np.nanmedian(measured_y))) < 2.0


def test_prediction_plot_exposes_selected_and_effective_targets_when_available():
    freqs = np.array([20.0, 40.0, 80.0, 160.0, 320.0, 640.0, 1280.0, 2560.0], dtype=float)
    mags = np.array([-57.0, -56.0, -55.0, -54.5, -54.0, -54.0, -54.5, -55.0], dtype=float)
    phases = np.zeros_like(freqs)
    filt_ir = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

    _html, fig = generate_prediction_plot(
        freqs,
        mags,
        phases,
        filt_ir,
        48000,
        "Test",
        target_stats={
            "freq_axis": freqs.tolist(),
            "measured_mags": mags.tolist(),
            "target_mags": [-54.0, -53.5, -53.0, -52.5, -52.5, -53.0, -53.5, -54.0],
            "selected_target_mags": [0.0, 0.0, -0.5, -1.0, -1.0, -1.5, -2.0, -2.5],
            "eff_target_db": -53.0,
            "target_level_db_window": -53.0,
            "offset_db": 0.0,
            "smart_scan_range": [160.0, 1280.0],
        },
        create_full_html=False,
        return_fig=True,
    )

    assert fig is not None
    trace_names = [trace.name for trace in fig.data]
    assert "Target" in trace_names
    assert "Effective target" in trace_names

    target_trace = next(trace for trace in fig.data if trace.name == "Target")
    effective_trace = next(trace for trace in fig.data if trace.name == "Effective target")
    np.testing.assert_allclose(np.asarray(target_trace.y, dtype=float), np.array([0.0, 0.0, -0.5, -1.0, -1.0, -1.5, -2.0, -2.5], dtype=float))
    assert effective_trace.line.dash == "dot"


def test_prediction_plot_keeps_magnitude_legend_focused_on_core_curves():
    freqs = np.array([20.0, 40.0, 80.0, 160.0, 320.0, 640.0, 1280.0, 2560.0], dtype=float)
    mags = np.zeros_like(freqs)
    phases = np.zeros_like(freqs)
    filt_ir = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

    _html, fig = generate_prediction_plot(
        freqs,
        mags,
        phases,
        filt_ir,
        48000,
        "Test",
        target_stats={
            "freq_axis": freqs.tolist(),
            "measured_mags": mags.tolist(),
            "target_mags": np.zeros_like(freqs).tolist(),
            "confidence_mask": np.ones_like(freqs).tolist(),
            "eff_target_db": 75.0,
            "offset_db": 0.0,
            "smart_scan_range": [320.0, 2560.0],
            "mag_c_min": 20.0,
            "mag_c_max": 200.0,
            "auto_global_gain_db": -1.5,
            "auto_headroom_db": 0.0,
        },
        create_full_html=False,
        return_fig=True,
    )

    assert fig is not None
    trace_names = [trace.name for trace in fig.data]
    assert "Confidence" in trace_names
    assert any(name and name.startswith("Error (M") for name in trace_names)
    assert not any(name and name.startswith("Level reference") for name in trace_names)

    fig_dict = fig.to_dict()
    legend_names = [trace["name"] for trace in fig_dict["data"] if trace.get("showlegend", True)]
    assert legend_names[:4] == ["Measured", "Target", "Predicted (exported)", "Confidence"]
    assert len(legend_names) == 5
    assert legend_names[-1].startswith("Error (M")
    assert "Auto gain:" in fig_dict["layout"]["title"]["text"]
    assert fig_dict["layout"]["margin"]["b"] >= 90
