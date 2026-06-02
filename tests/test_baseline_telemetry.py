import numpy as np

from decaycore.config.models import FilterConfig
from decaycore.dsp._measurement_ctx_local import clear_measurement_ctx, set_measurement_ctx
from decaycore.dsp.correction_baseline import _prepare_correction_baseline
from decaycore.dsp.correction_types import BaselineComparisonTelemetry, BaselineNativeTelemetry, MeasurementSideContext
from decaycore.dsp.dsp_telemetry import apply_baseline_telemetry_to_stats


def test_apply_baseline_telemetry_to_stats_populates_legacy_keys():
    st = {}
    cmp = {}
    native = BaselineNativeTelemetry(
        analysis_mode="native",
        freq_axis=np.array([20.0, 40.0, 80.0], dtype=float),
        measured_mags=np.array([1.0, 2.0, 3.0], dtype=float),
        target_mags=np.array([0.0, 0.5, 1.0], dtype=float),
        target_env_lo=np.array([-1.0, -0.5, 0.0], dtype=float),
        target_env_hi=np.array([1.0, 1.5, 2.0], dtype=float),
        target_env_pivot_hz=63.0,
        target_shift_db=2.5,
        eff_target_db=74.0,
        target_level_db_window=73.5,
        meas_level_db_window=75.0,
        offset_db=1.0,
        offset_method="Median",
        smart_scan_range=(200.0, 3000.0),
    )
    comparison = BaselineComparisonTelemetry(
        analysis_mode="comparison",
        target_mags=np.array([0.0, 0.5, 1.0], dtype=float),
        measured_mags=np.array([1.0, 2.0, 3.0], dtype=float),
        filter_mags=np.array([0.2, 0.1, 0.0], dtype=float),
        eff_target_db=74.0,
        offset_db=1.25,
        smart_scan_range=(180.0, 2800.0),
        meas_level_db_window=75.0,
        target_level_db_window=73.5,
        offset_method="Median",
        target_shift_db=2.5,
    )

    apply_baseline_telemetry_to_stats(
        st=st,
        cmp=cmp,
        native=native,
        comparison=comparison,
    )

    assert st["analysis_mode"] == "native"
    assert st["freq_axis"] == [20.0, 40.0, 80.0]
    assert st["target_mags"] == [0.0, 0.5, 1.0]
    assert st["target_env_pivot_hz"] == 63.0
    assert st["smart_scan_range"] == [200.0, 3000.0]
    assert cmp["analysis_mode"] == "comparison"
    assert cmp["cmp_target_mags"] == [0.0, 0.5, 1.0]
    assert cmp["cmp_filter_mags"] == [0.2, 0.1, 0.0]
    assert cmp["cmp_smart_scan_range"] == [180.0, 2800.0]


def test_apply_baseline_telemetry_skips_optional_target_envelope_when_missing():
    st = {}
    apply_baseline_telemetry_to_stats(
        st=st,
        cmp=None,
        native=BaselineNativeTelemetry(
            analysis_mode="native",
            freq_axis=np.array([20.0, 40.0], dtype=float),
            measured_mags=np.array([1.0, 2.0], dtype=float),
            target_mags=np.array([0.0, 0.5], dtype=float),
            target_env_lo=None,
            target_env_hi=None,
            target_env_pivot_hz=None,
            target_shift_db=0.0,
            eff_target_db=74.0,
            target_level_db_window=74.0,
            meas_level_db_window=74.0,
            offset_db=0.0,
            offset_method="Median",
            smart_scan_range=(200.0, 2000.0),
        ),
        comparison=None,
    )

    assert "target_env_lo" not in st
    assert "target_env_hi" not in st
    assert "target_env_pivot_hz" not in st


def test_prepare_correction_baseline_sets_schroeder_from_measured_rt60():
    freq_axis = np.geomspace(20.0, 20000.0, 128)
    cfg = FilterConfig(fs=48000, num_taps=4096, room_volume_m3=40.0)
    st = {}

    set_measurement_ctx(MeasurementSideContext(measured_rt60=0.36))
    try:
        _prepare_correction_baseline(
            cfg=cfg,
            freq_axis=freq_axis,
            f_in=freq_axis,
            m_in=np.zeros_like(freq_axis),
            reflections=[],
            st=st,
            m_anal=np.zeros_like(freq_axis),
            m_plot_db=np.zeros_like(freq_axis),
            is_psy=False,
            cmp={},
            analysis_mode="native",
            gain_db=np.zeros_like(freq_axis),
            logger=None,
            interpolate_response=np.interp,
            _cfg_float_allow_zero=lambda _cfg, _name, default: default,
            presolve_mode=True,
        )
    finally:
        clear_measurement_ctx()

    assert np.isclose(st["schroeder_hz_estimate"], 2000.0 * np.sqrt(0.36 / 40.0))
