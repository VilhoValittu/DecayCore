from __future__ import annotations

import numpy as np

from decaycore.config.models import FilterConfig
from decaycore.engine_run import run_pipeline


def test_run_pipeline_prefers_saved_measurement_rt60_over_txt_baseline(monkeypatch) -> None:
    freqs = np.asarray([20.0, 40.0, 80.0, 160.0], dtype=float)

    def _fake_generate_filter(freqs_in, mags_in, phases_in, cfg, *, stereo_link_ctx=None):
        _ = mags_in, phases_in, cfg, stereo_link_ctx
        impulse = np.zeros(32, dtype=float)
        impulse[0] = 1.0
        return impulse, {
            "freq_axis": np.asarray(freqs_in, dtype=float),
            "filter_mags": np.zeros_like(np.asarray(freqs_in, dtype=float)),
            "analysis_mode": "native",
            "delay_samples": 0.0,
            "rt60_val": 0.18,
            "rt60_band_avg": 0.2,
            "rt60_bands": {63.0: 0.18},
        }

    monkeypatch.setattr("decaycore.engine_run.dsp.generate_filter", _fake_generate_filter)

    result = run_pipeline(
        FilterConfig(fs=48000, num_taps=4096),
        {
            "f_l": freqs,
            "m_l": np.zeros_like(freqs),
            "p_l": np.zeros_like(freqs),
            "f_r": freqs,
            "m_r": np.zeros_like(freqs),
            "p_r": np.zeros_like(freqs),
            "ui_data": {"comparison_mode": False},
            "is_wav_source": False,
            "measured_rt60_l": 0.41,
            "measured_rt60_bands_l": {125.0: 0.35, 250.0: 0.45},
            "measured_rt60_r": 0.53,
            "measured_rt60_bands_r": {250.0: 0.5, 500.0: 0.6},
        },
        include_response_arrays=False,
    )

    assert result.l_st["rt60_val"] == 0.41
    assert result.l_st["rt60_reliability"] == 1.0
    assert result.l_st["rt60_source"] == "measured"
    assert np.isclose(result.l_st["schroeder_hz_estimate"], 2000.0 * np.sqrt(0.41 / 40.0))
    assert result.l_st["rt60_bands"] == {125.0: 0.35, 250.0: 0.45}
    assert result.l_st["rt60_band_avg"] == 0.4

    assert result.r_st["rt60_val"] == 0.53
    assert result.r_st["rt60_reliability"] == 1.0
    assert result.r_st["rt60_source"] == "measured"
    assert np.isclose(result.r_st["schroeder_hz_estimate"], 2000.0 * np.sqrt(0.53 / 40.0))
    assert result.r_st["rt60_bands"] == {250.0: 0.5, 500.0: 0.6}
    assert result.r_st["rt60_band_avg"] == 0.55
