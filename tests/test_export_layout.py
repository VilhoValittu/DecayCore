import io
import zipfile

import numpy as np
import scipy.io.wavfile

from decaycore.config.results import FilterResult
from decaycore.core.runner_parts.headless_export_bundle import _build_headless_export_zip
from decaycore.ui.export_bundle import build_export_zip


def _make_result(fs: int = 48000, *, with_sub: bool = False) -> FilterResult:
    l_ir = np.asarray([0.0, 0.25, -0.25, 0.0], dtype=np.float32)
    r_ir = np.asarray([0.0, -0.5, 0.5, 0.0], dtype=np.float32)
    sub_ir = np.asarray([0.0, 0.4, -0.1, 0.0], dtype=np.float32) if with_sub else None
    freq = np.asarray([20.0, 100.0, 1000.0], dtype=float)
    zeros = np.zeros_like(freq)
    st = {
        "offset_method": "Auto",
        "smart_scan_range": [20.0, 200.0],
        "offset_db": 0.0,
        "eff_target_db": 0.0,
    }
    measurements = {
        "f_l": freq,
        "m_l": zeros,
        "p_l": zeros,
        "f_r": freq,
        "m_r": zeros,
        "p_r": zeros,
    }
    return FilterResult(
        fs=fs,
        taps=4,
        l_ir=l_ir,
        r_ir=r_ir,
        l_mag=zeros,
        r_mag=zeros,
        l_phase=zeros,
        r_phase=zeros,
        freq_axis=freq,
        l_st=dict(st),
        r_st=dict(st),
        measurements=measurements,
        sub_ir=sub_ir,
    )


def _base_data(layout: str) -> dict:
    return {
        "layout": layout,
        "target_curve_tag": "target",
        "multi_rate_opt": False,
        "program_version": "v.0.0.0",
        "filter_type": "Asymmetric",
        "mixed_freq": 200.0,
        "camillafir_automatic_mode": False,
    }


def test_build_export_zip_keeps_dual_mono_layout():
    result = _make_result()
    zip_buffer, _, _ = build_export_zip(
        data=_base_data("Mono"),
        results=[result],
        ft_short="Asymmetric",
        file_ts="1200_290326",
        irw_tag="auto",
        write_dashboards=False,
    )

    with zipfile.ZipFile(zip_buffer) as zf:
        names = set(zf.namelist())
        assert "L_Asymmetric_48000Hz_target_1200_290326_auto.wav" in names
        assert "R_Asymmetric_48000Hz_target_1200_290326_auto.wav" in names
        assert "bypass/Bypass_L_Asymmetric_48000Hz_target_1200_290326_auto.wav" in names
        assert "bypass/Bypass_R_Asymmetric_48000Hz_target_1200_290326_auto.wav" in names
        assert "Stereo_Asymmetric_48000Hz_target_1200_290326_auto.wav" not in names

        fs_l, bypass_l = scipy.io.wavfile.read(
            io.BytesIO(zf.read("bypass/Bypass_L_Asymmetric_48000Hz_target_1200_290326_auto.wav"))
        )
        fs_r, bypass_r = scipy.io.wavfile.read(
            io.BytesIO(zf.read("bypass/Bypass_R_Asymmetric_48000Hz_target_1200_290326_auto.wav"))
        )
        assert fs_l == 48000
        assert fs_r == 48000
        assert np.allclose(bypass_l, [0.0, 1.0, 0.0, 0.0])
        assert np.allclose(bypass_r, [0.0, 1.0, 0.0, 0.0])

        yaml_name = next(name for name in names if name.endswith(".yml"))
        yaml_text = zf.read(yaml_name).decode("utf-8")
        assert "filename: ../coeffs/L_Asymmetric_$samplerate$Hz_target_1200_290326_auto.wav" in yaml_text
        assert "filename: ../coeffs/R_Asymmetric_$samplerate$Hz_target_1200_290326_auto.wav" in yaml_text
        assert yaml_text.count("channel: 0") >= 2

        cfg_text = zf.read("Config_Asymmetric_48000Hz_auto.cfg").decode("utf-8")
        assert "L_Asymmetric_48000Hz_target_1200_290326_auto.wav" in cfg_text
        assert "R_Asymmetric_48000Hz_target_1200_290326_auto.wav" in cfg_text

        bypass_cfg_text = zf.read("bypass/Bypass_Config_Asymmetric_48000Hz_auto.cfg").decode("utf-8")
        assert "Bypass_L_Asymmetric_48000Hz_target_1200_290326_auto.wav" in bypass_cfg_text
        assert "Bypass_R_Asymmetric_48000Hz_target_1200_290326_auto.wav" in bypass_cfg_text
        assert "bypass/Bypass_L_Asymmetric_48000Hz_target_1200_290326_auto.wav" not in bypass_cfg_text


def test_build_export_zip_writes_hybrid_iir_before_conv():
    result = _make_result()
    result.l_st["hybrid_iir_biquads"] = [
        {"type": "Peaking", "freq": 41.5, "q": 6.4, "gain": -4.0, "confidence": 0.9, "safe_cut_db": 4.0}
    ]
    result.r_st["hybrid_iir_biquads"] = [
        {"type": "Peaking", "freq": 83.0, "q": 5.2, "gain": -3.0, "confidence": 0.8, "safe_cut_db": 3.0}
    ]
    zip_buffer, _, _ = build_export_zip(
        data=_base_data("Mono"),
        results=[result],
        ft_short="Asymmetric",
        file_ts="1200_290326",
        irw_tag="auto",
        write_dashboards=False,
    )

    with zipfile.ZipFile(zip_buffer) as zf:
        yaml_name = next(name for name in zf.namelist() if name.endswith(".yml"))
        yaml_text = zf.read(yaml_name).decode("utf-8")
        assert "l_hybrid_iir_1:" in yaml_text
        assert "r_hybrid_iir_1:" in yaml_text
        assert "type: Peaking" in yaml_text
        assert "names: [mastergain, l_hybrid_iir_1, ir_left]" in yaml_text
        assert "names: [mastergain, r_hybrid_iir_1, ir_right]" in yaml_text
        iir_name = "IIR_Asymmetric_48000Hz_auto.txt"
        assert iir_name in zf.namelist()
        iir_text = zf.read(iir_name).decode("utf-8")
        assert "Hybrid FIR-IIR modal cuts" in iir_text
        assert "l_hybrid_iir_1: Peaking, freq=41.500 Hz" in iir_text
        assert "r_hybrid_iir_1: Peaking, freq=83.000 Hz" in iir_text


def test_build_export_zip_writes_low_tap_main_hpf_as_iir():
    result = _make_result()
    data = _base_data("Mono")
    data.update(
        {
            "hpf_enable": True,
            "hpf_freq": 32.0,
            "hpf_slope": 24,
        }
    )
    zip_buffer, _, _ = build_export_zip(
        data=data,
        results=[result],
        ft_short="Asymmetric",
        file_ts="1200_290326",
        irw_tag="auto",
        write_dashboards=False,
    )

    with zipfile.ZipFile(zip_buffer) as zf:
        yaml_name = next(name for name in zf.namelist() if name.endswith(".yml"))
        yaml_text = zf.read(yaml_name).decode("utf-8")
        assert "channels: 2" in yaml_text
        assert "main_hpf:" in yaml_text
        assert "freq: 32.000" in yaml_text
        assert "type: LinkwitzRileyHighpass" in yaml_text
        assert "names: [mastergain, main_hpf, ir_left]" in yaml_text
        assert "names: [mastergain, main_hpf, ir_right]" in yaml_text
        iir_name = "IIR_Asymmetric_48000Hz_auto.txt"
        assert iir_name in zf.namelist()
        iir_text = zf.read(iir_name).decode("utf-8")
        assert "main_hpf: BiquadCombo LinkwitzRileyHighpass" in iir_text
        assert "freq=32.000 Hz" in iir_text
        assert "slope=24 dB/oct" in iir_text


def test_build_export_zip_writes_single_stereo_wav_when_requested():
    result = _make_result()
    zip_buffer, _, _ = build_export_zip(
        data=_base_data("Stereo"),
        results=[result],
        ft_short="Asymmetric",
        file_ts="1200_290326",
        irw_tag="auto",
        write_dashboards=False,
    )

    with zipfile.ZipFile(zip_buffer) as zf:
        names = set(zf.namelist())
        stereo_name = "Stereo_Asymmetric_48000Hz_target_1200_290326_auto.wav"
        bypass_name = "bypass/Bypass_Stereo_Asymmetric_48000Hz_target_1200_290326_auto.wav"
        assert stereo_name in names
        assert bypass_name in names
        assert "L_Asymmetric_48000Hz_target_1200_290326_auto.wav" not in names
        assert "R_Asymmetric_48000Hz_target_1200_290326_auto.wav" not in names

        fs, stereo_data = scipy.io.wavfile.read(io.BytesIO(zf.read(stereo_name)))
        assert fs == 48000
        assert stereo_data.shape == (4, 2)
        assert np.allclose(stereo_data[:, 0], result.l_ir)
        assert np.allclose(stereo_data[:, 1], result.r_ir)

        fs_bypass, bypass_data = scipy.io.wavfile.read(io.BytesIO(zf.read(bypass_name)))
        assert fs_bypass == 48000
        assert bypass_data.shape == (4, 2)
        assert np.allclose(bypass_data[:, 0], [0.0, 1.0, 0.0, 0.0])
        assert np.allclose(bypass_data[:, 1], [0.0, 1.0, 0.0, 0.0])

        yaml_name = next(name for name in names if name.endswith(".yml"))
        yaml_text = zf.read(yaml_name).decode("utf-8")
        assert yaml_text.count("filename: ../coeffs/Stereo_Asymmetric_$samplerate$Hz_target_1200_290326_auto.wav") == 2
        assert "channel: 0" in yaml_text
        assert "channel: 1" in yaml_text

        cfg_text = zf.read("Config_Asymmetric_48000Hz_auto.cfg").decode("utf-8")
        assert cfg_text.count("Stereo_Asymmetric_48000Hz_target_1200_290326_auto.wav") == 2
        assert "\n0\n0.0\n0.0\nStereo_Asymmetric_48000Hz_target_1200_290326_auto.wav\n1\n1.0\n1.0" in cfg_text

        bypass_cfg_text = zf.read("bypass/Bypass_Config_Asymmetric_48000Hz_auto.cfg").decode("utf-8")
        assert bypass_cfg_text.count("Bypass_Stereo_Asymmetric_48000Hz_target_1200_290326_auto.wav") == 2
        assert "bypass/Bypass_Stereo_Asymmetric_48000Hz_target_1200_290326_auto.wav" not in bypass_cfg_text


def test_build_export_zip_accepts_stable_layout_key():
    result = _make_result()
    zip_buffer, _, _ = build_export_zip(
        data=_base_data("stereo"),
        results=[result],
        ft_short="Asymmetric",
        file_ts="1200_290326",
        irw_tag="auto",
        write_dashboards=False,
    )

    with zipfile.ZipFile(zip_buffer) as zf:
        names = set(zf.namelist())
        assert "Stereo_Asymmetric_48000Hz_target_1200_290326_auto.wav" in names


def test_headless_export_zip_writes_bypass_hlc_companion():
    result = _make_result()
    zip_buffer, _, _ = _build_headless_export_zip(
        data=_base_data("Mono"),
        results=[result],
        ft_short="Asymmetric",
        file_ts="1200_290326",
        irw_tag="auto",
        write_dashboards=False,
    )

    with zipfile.ZipFile(zip_buffer) as zf:
        names = set(zf.namelist())
        assert "bypass/Bypass_L_Asymmetric_48000Hz_target_1200_290326_auto.wav" in names
        assert "bypass/Bypass_R_Asymmetric_48000Hz_target_1200_290326_auto.wav" in names

        bypass_cfg_text = zf.read("bypass/Bypass_Config_Asymmetric_48000Hz_auto.cfg").decode("utf-8")
        assert "Bypass_L_Asymmetric_48000Hz_target_1200_290326_auto.wav" in bypass_cfg_text
        assert "Bypass_R_Asymmetric_48000Hz_target_1200_290326_auto.wav" in bypass_cfg_text


def test_build_export_zip_direct_dac_yaml_includes_sub_pipeline_and_allpass():
    result = _make_result(with_sub=True)
    data = _base_data("Mono")
    data.update(
        {
            "bass_integration_enable": True,
            "bass_integration_mode": "direct_dac",
            "bass_integration_allpass_auto_applied": True,
            "bass_integration_allpass_freq_hz": 77.5,
            "bass_integration_allpass_q": 0.9,
        }
    )

    zip_buffer, _, _ = build_export_zip(
        data=data,
        results=[result],
        ft_short="Asymmetric",
        file_ts="1200_290326",
        irw_tag="auto",
        write_dashboards=False,
    )

    with zipfile.ZipFile(zip_buffer) as zf:
        names = set(zf.namelist())
        assert "Sub_Asymmetric_48000Hz_1200_290326_auto.wav" in names
        assert "bypass/Bypass_Sub_Asymmetric_48000Hz_1200_290326_auto.wav" in names

        fs_sub, bypass_sub = scipy.io.wavfile.read(
            io.BytesIO(zf.read("bypass/Bypass_Sub_Asymmetric_48000Hz_1200_290326_auto.wav"))
        )
        assert fs_sub == 48000
        assert np.allclose(bypass_sub, [0.0, 1.0, 0.0, 0.0])

        yaml_name = next(name for name in names if name.endswith(".yml"))
        yaml_text = zf.read(yaml_name).decode("utf-8")
        assert "format: S32_LE" in yaml_text
        assert "channels: 3" in yaml_text
        assert "out: 3" in yaml_text
        assert "filename: ../coeffs/Sub_Asymmetric_$samplerate$Hz_1200_290326_auto.wav" in yaml_text
        assert "type: Biquad" in yaml_text
        assert "type: Allpass" in yaml_text
        assert "freq: 77.500" in yaml_text
        assert "q: 0.900000" in yaml_text
        assert "channels: [2]" in yaml_text
        assert "main_hpf:" in yaml_text
        assert "sub_hpf:" in yaml_text
        assert "sub_lpf:" in yaml_text
        assert "names: [mastergain, sub_hpf, sub_lpf, sub_allpass, ir_sub]" in yaml_text


def test_build_export_zip_summary_includes_bass_dsp_settings_section():
    result = _make_result(with_sub=True)
    data = _base_data("Mono")
    data.update(
        {
            "bass_integration_enable": True,
            "_bass_integration_meta": {
                "avr_crossover_hz": 54.1,
                "direct_dac_sub_lpf_hz": 86.6,
                "alignment": {
                    "delay_ms": 11.989,
                    "main_l_delay_ms": 0.0,
                    "main_r_delay_ms": 0.0,
                    "polarity_invert": True,
                    "gain_trim_db": -8.785,
                },
                "recommended_allpass": {
                    "enabled": False,
                },
                "inputs": {},
            },
        }
    )

    zip_buffer, _, _ = build_export_zip(
        data=data,
        results=[result],
        ft_short="Asymmetric",
        file_ts="1200_290326",
        irw_tag="auto",
        write_dashboards=False,
    )

    with zipfile.ZipFile(zip_buffer) as zf:
        summary_name = next(name for name in zf.namelist() if name.startswith("Summary_") and name.endswith(".txt"))
        summary_text = zf.read(summary_name).decode("utf-8")
        assert "=== DSP SETTINGS TO ENTER IN YOUR DSP ===" in summary_text
        assert "=== BASS INTEGRATION DIAGNOSTICS ===" in summary_text
        assert "Use these exact crossover, delay, polarity, gain, and allpass values in your DSP." not in summary_text
        assert "Main HPF: 54.1 Hz / 12 dB/oct" in summary_text
        assert "Sub LPF: 86.6 Hz / 12 dB/oct" in summary_text
        assert "Sub delay: 11.989 ms" in summary_text
        assert "Main delay L/R: L 0.000 ms, R 0.000 ms" in summary_text
        assert "Sub polarity: INVERTED" in summary_text
        assert "Sub gain trim: -8.785 dB" in summary_text
        assert "Bass allpass: OFF" in summary_text


def test_build_export_zip_direct_dac_yaml_uses_linkwitz_riley_for_24db_crossovers():
    result = _make_result(with_sub=True)
    data = _base_data("Mono")
    data.update(
        {
            "bass_integration_enable": True,
            "bass_integration_mode": "direct_dac",
            "sub_crossover_hz": 80.0,
            "direct_dac_sub_lpf_hz": 85.0,
            "sub_hpf_freq": 20.0,
            "sub_crossover_slope": 24,
            "sub_hpf_slope": 24,
        }
    )

    zip_buffer, _, _ = build_export_zip(
        data=data,
        results=[result],
        ft_short="Asymmetric",
        file_ts="1200_290326",
        irw_tag="auto",
        write_dashboards=False,
    )

    with zipfile.ZipFile(zip_buffer) as zf:
        yaml_name = next(name for name in zf.namelist() if name.endswith(".yml"))
        yaml_text = zf.read(yaml_name).decode("utf-8")
        assert yaml_text.count("type: BiquadCombo") == 3
        assert "type: LinkwitzRileyHighpass" in yaml_text
        assert "type: LinkwitzRileyLowpass" in yaml_text
        assert "order: 4" in yaml_text
        assert "freq: 85.000" in yaml_text


def test_build_export_zip_direct_dac_yaml_keeps_biquad_for_12db_crossovers():
    result = _make_result(with_sub=True)
    data = _base_data("Mono")
    data.update(
        {
            "bass_integration_enable": True,
            "bass_integration_mode": "direct_dac",
            "sub_crossover_slope": 12,
            "sub_hpf_slope": 12,
        }
    )

    zip_buffer, _, _ = build_export_zip(
        data=data,
        results=[result],
        ft_short="Asymmetric",
        file_ts="1200_290326",
        irw_tag="auto",
        write_dashboards=False,
    )

    with zipfile.ZipFile(zip_buffer) as zf:
        yaml_name = next(name for name in zf.namelist() if name.endswith(".yml"))
        yaml_text = zf.read(yaml_name).decode("utf-8")
        assert "type: BiquadCombo" not in yaml_text
        assert "type: Highpass" in yaml_text
        assert "type: Lowpass" in yaml_text
        assert "q: 0.707107" in yaml_text


def test_build_export_zip_direct_dac_yaml_includes_sub_alignment_filters():
    result = _make_result(with_sub=True)
    data = _base_data("Mono")
    data.update(
        {
            "bass_integration_enable": True,
            "bass_integration_mode": "direct_dac",
            "bass_integration_sub_delay_ms": 3.25,
            "bass_integration_sub_polarity_invert": True,
            "bass_integration_sub_gain_trim_db": -2.5,
        }
    )

    zip_buffer, _, _ = build_export_zip(
        data=data,
        results=[result],
        ft_short="Asymmetric",
        file_ts="1200_290326",
        irw_tag="auto",
        write_dashboards=False,
    )

    with zipfile.ZipFile(zip_buffer) as zf:
        yaml_name = next(name for name in zf.namelist() if name.endswith(".yml"))
        yaml_text = zf.read(yaml_name).decode("utf-8")
        assert "type: Delay" in yaml_text
        assert "delay: 3.250" in yaml_text
        assert "scale: dB" in yaml_text
        assert "gain: -2.500" in yaml_text
        assert "inverted: true" in yaml_text
        assert "names: [mastergain, main_hpf, ir_left]" in yaml_text
        assert "names: [mastergain, main_hpf, ir_right]" in yaml_text
        assert "names: [mastergain, sub_gain, sub_delay, sub_hpf, sub_lpf, ir_sub]" in yaml_text


def test_build_export_zip_direct_dac_yaml_uses_main_delay_for_negative_sub_delay():
    result = _make_result(with_sub=True)
    data = _base_data("Mono")
    data.update(
        {
            "bass_integration_enable": True,
            "bass_integration_mode": "direct_dac",
            "bass_integration_sub_delay_ms": -2.5,
        }
    )

    zip_buffer, _, _ = build_export_zip(
        data=data,
        results=[result],
        ft_short="Asymmetric",
        file_ts="1200_290326",
        irw_tag="auto",
        write_dashboards=False,
    )

    with zipfile.ZipFile(zip_buffer) as zf:
        yaml_name = next(name for name in zf.namelist() if name.endswith(".yml"))
        yaml_text = zf.read(yaml_name).decode("utf-8")
        assert "main_delay:" in yaml_text
        assert "delay: 2.500" in yaml_text
        assert "names: [mastergain, main_delay, main_hpf, ir_left]" in yaml_text
        assert "names: [mastergain, main_delay, main_hpf, ir_right]" in yaml_text
        assert "names: [mastergain, sub_hpf, sub_lpf, ir_sub]" in yaml_text
        assert "sub_delay:" not in yaml_text


def test_build_export_zip_multi_rate_direct_dac_yaml_uses_sub_pipeline_tokenized_name():
    result_48 = _make_result(fs=48000, with_sub=True)
    result_96 = _make_result(fs=96000, with_sub=True)
    data = _base_data("Mono")
    data.update(
        {
            "multi_rate_opt": True,
            "fs": 48000,
            "bass_integration_enable": True,
            "bass_integration_mode": "direct_dac",
        }
    )

    zip_buffer, _, _ = build_export_zip(
        data=data,
        results=[result_48, result_96],
        ft_short="Asymmetric",
        file_ts="1200_290326",
        irw_tag="auto",
        write_dashboards=False,
    )

    with zipfile.ZipFile(zip_buffer) as zf:
        names = set(zf.namelist())
        yaml_names = [name for name in names if name.endswith(".yml")]
        assert len(yaml_names) == 1
        assert "Sub_Asymmetric_48000Hz_1200_290326_auto.wav" in names
        assert "Sub_Asymmetric_96000Hz_1200_290326_auto.wav" in names

        yaml_text = zf.read(yaml_names[0]).decode("utf-8")
        assert "channels: 3" in yaml_text
        assert "filename: ../coeffs/Sub_Asymmetric_$samplerate$Hz_1200_290326_auto.wav" in yaml_text
        assert "names: [mastergain, sub_hpf, sub_lpf, ir_sub]" in yaml_text


def test_choose_target_rates_adds_ultra_high_only_when_requested():
    from decaycore.config.decaycore_pipeline import choose_target_rates

    assert choose_target_rates({"multi_rate_opt": True}) == [44100, 48000, 88200, 96000, 176400, 192000]
    assert choose_target_rates({"multi_rate_opt": True, "multi_rate_ultra_high_opt": True}) == [
        44100,
        48000,
        88200,
        96000,
        176400,
        192000,
        352800,
        384000,
    ]
