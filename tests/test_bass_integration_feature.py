from __future__ import annotations

import io

import numpy as np
import pytest
import scipy.io.wavfile

from decaycore.application.health_service import compute_health
from decaycore.auto_mode.cache_signature import _auto_measurement_signature
from decaycore.config.decaycore_pipeline import build_filter_config, build_xos_hpf, collect_ui_data
from decaycore.config.models import FilterConfig
from decaycore.engine_run import run_pipeline
from decaycore.io.measurement_bundle import BassIntegrationBundle, TransferData
from decaycore.io.measurements_loader import load_bass_integration_measurements
from decaycore.io.measurements_wav import parse_coherent_transfer_from_wav_bytes
from decaycore.dsp.bass_integration import prepare_dual_sub_peak_aligned_average
from decaycore.dsp.bass_integration._recommend_alignment import _best_metric_grid_row
from decaycore.engine_run_parts.subwoofer_target import build_subwoofer_target_with_lpf
from decaycore.ui.export_summary_text import (
    _append_bass_integration_allpass_auto_summary,
    _append_bass_integration_summary,
    _append_main_speaker_xo_hpf_summary,
)
from decaycore.workflow.auto_flow import (
    _build_auto_selected_text,
    _resolve_auto_hpf_seed_source,
)


def _impulse_wav_bytes(*, fs_hz: int = 48000, length: int = 4096, delay_samples: int = 0) -> bytes:
    sig = np.zeros(int(length), dtype=np.float32)
    sig[int(delay_samples)] = 1.0
    buf = io.BytesIO()
    scipy.io.wavfile.write(buf, int(fs_hz), sig)
    return buf.getvalue()


def test_bass_metric_grid_best_row_keeps_first_strict_winner():
    rows = [
        (("first",), 1.0, {"id": "first"}),
        (("tie",), 1.0, {"id": "tie"}),
        (("winner",), 1.5, {"id": "winner"}),
    ]

    candidate, score, metrics = _best_metric_grid_row(rows, current_score=float("nan"))

    assert candidate == ("winner",)
    assert score == pytest.approx(1.5)
    assert metrics == {"id": "winner"}
    assert _best_metric_grid_row(rows, current_score=1.5)[0] is None


def _upload_dict(name: str, content: bytes) -> dict:
    return {"filename": name, "content": bytes(content)}


def _flat_transfer(freqs_hz: np.ndarray, complex_gain: complex, *, label: str) -> TransferData:
    spec = np.full(np.asarray(freqs_hz, dtype=float).shape, complex_gain, dtype=np.complex128)
    return TransferData(
        freqs_hz=np.asarray(freqs_hz, dtype=float),
        complex_spec=spec,
        mag_db=20.0 * np.log10(np.maximum(np.abs(spec), 1e-12)),
        phase_deg=np.rad2deg(np.unwrap(np.angle(spec))),
        sample_rate=48000,
        label=label,
    )


def _delayed_transfer(freqs_hz: np.ndarray, delay_s: float, *, label: str) -> TransferData:
    freqs = np.asarray(freqs_hz, dtype=float)
    spec = np.exp(-1j * 2.0 * np.pi * freqs * float(delay_s)).astype(np.complex128)
    return TransferData(
        freqs_hz=freqs,
        complex_spec=spec,
        mag_db=np.zeros_like(freqs, dtype=float),
        phase_deg=np.rad2deg(np.unwrap(np.angle(spec))),
        sample_rate=48000,
        label=label,
    )


def test_subwoofer_target_lpf_preserves_bass_and_rolls_off_above_200hz() -> None:
    house_freqs = [20.0, 80.0, 200.0, 1000.0]
    house_mags = [6.0, 3.0, 0.0, 0.0]

    sub_target = build_subwoofer_target_with_lpf(house_freqs, house_mags)

    assert sub_target.applied is True
    out_f = np.asarray(sub_target.house_freqs, dtype=float)
    out_m = np.asarray(sub_target.house_mags, dtype=float)
    base_80 = np.interp(80.0, house_freqs, house_mags)
    base_200 = np.interp(200.0, house_freqs, house_mags)
    base_400 = np.interp(400.0, house_freqs, house_mags)
    assert np.all(np.isfinite(out_f))
    assert np.all(np.isfinite(out_m))
    assert np.interp(80.0, out_f, out_m) == pytest.approx(base_80)
    assert np.interp(200.0, out_f, out_m) == pytest.approx(base_200)
    assert np.interp(400.0, out_f, out_m) == pytest.approx(base_400 - 80.0)
    assert house_freqs == [20.0, 80.0, 200.0, 1000.0]
    assert house_mags == [6.0, 3.0, 0.0, 0.0]


def test_subwoofer_target_lpf_handles_sparse_target_data() -> None:
    sub_target = build_subwoofer_target_with_lpf([20.0, 20000.0], [0.0, 0.0])

    out_f = np.asarray(sub_target.house_freqs, dtype=float)
    out_m = np.asarray(sub_target.house_mags, dtype=float)
    assert sub_target.applied is True
    assert out_f.size >= 32
    assert np.all(np.isfinite(out_f))
    assert np.all(np.isfinite(out_m))
    assert np.interp(100.0, out_f, out_m) == pytest.approx(0.0)
    assert np.interp(400.0, out_f, out_m) == pytest.approx(-80.0)


def _make_asymmetric_shared_sub_bundle(
    *,
    right_main_gain: float = 0.25,
    sub_gain: float = 1.0,
    n_points: int = 160,
) -> BassIntegrationBundle:
    freqs = np.logspace(np.log10(10.0), np.log10(200.0), int(n_points))
    l_main = _flat_transfer(freqs, 1.0 + 0.0j, label="l_main")
    r_main = _flat_transfer(freqs, complex(float(right_main_gain), 0.0), label="r_main")
    l_sub = _flat_transfer(freqs, complex(float(sub_gain), 0.0), label="l_sub")
    r_sub = _flat_transfer(freqs, complex(float(sub_gain), 0.0), label="r_sub")

    def _total(main: TransferData, sub: TransferData, *, label: str) -> TransferData:
        spec = np.asarray(main.complex_spec, dtype=np.complex128) + np.asarray(sub.complex_spec, dtype=np.complex128)
        return TransferData(
            freqs_hz=np.asarray(freqs, dtype=float),
            complex_spec=spec,
            mag_db=20.0 * np.log10(np.maximum(np.abs(spec), 1e-12)),
            phase_deg=np.rad2deg(np.unwrap(np.angle(spec))),
            sample_rate=48000,
            label=label,
        )

    return BassIntegrationBundle(
        l_main=l_main,
        r_main=r_main,
        l_sub=l_sub,
        r_sub=r_sub,
        l_total=_total(l_main, l_sub, label="l_total"),
        r_total=_total(r_main, r_sub, label="r_total"),
        avr_crossover_hz=80.0,
        profile="safe",
        diagnostics={},
    )


def _make_flat_overlap_bundle(*, n_points: int = 160) -> BassIntegrationBundle:
    freqs = np.logspace(np.log10(10.0), np.log10(200.0), int(n_points))
    main = _flat_transfer(freqs, 1.0 + 0.0j, label="main")
    sub = _flat_transfer(freqs, 1.0 + 0.0j, label="sub")
    total = _flat_transfer(freqs, 2.0 + 0.0j, label="total")
    return BassIntegrationBundle(
        l_main=main,
        r_main=main,
        l_sub=sub,
        r_sub=sub,
        l_total=total,
        r_total=total,
        avr_crossover_hz=80.0,
        profile="safe",
        diagnostics={},
    )


def test_parse_coherent_transfer_from_wav_bytes_preserves_delay_phase() -> None:
    fs_hz = 48000
    delayed_samples = 24
    ref = parse_coherent_transfer_from_wav_bytes(
        _impulse_wav_bytes(fs_hz=fs_hz, delay_samples=0),
        label="ref",
    )
    delayed = parse_coherent_transfer_from_wav_bytes(
        _impulse_wav_bytes(fs_hz=fs_hz, delay_samples=delayed_samples),
        label="delayed",
    )

    assert ref is not None
    assert delayed is not None

    idx = int(np.argmin(np.abs(ref.freqs_hz - 1000.0)))
    ratio = delayed.complex_spec[idx] / ref.complex_spec[idx]
    expected_phase = -2.0 * np.pi * ref.freqs_hz[idx] * float(delayed_samples) / float(fs_hz)
    phase_err = np.angle(np.exp(1j * (np.angle(ratio) - expected_phase)))

    assert abs(float(phase_err)) < 0.05


def test_load_bass_integration_measurements_normalizes_legacy_mode_to_direct_main_sources() -> None:
    data = {
        "bass_integration_enable": True,
        "bass_integration_mode": "avr_lfe_main_decomposed",
        "bass_integration_profile": "normal",
        "avr_crossover_hz": 80.0,
        "file_l_main": _upload_dict("l_main.wav", _impulse_wav_bytes(delay_samples=0)),
        "file_r_main": _upload_dict("r_main.wav", _impulse_wav_bytes(delay_samples=0)),
        "file_l_sub": _upload_dict("l_sub.wav", _impulse_wav_bytes(delay_samples=0)),
        "file_r_sub": _upload_dict("r_sub.wav", _impulse_wav_bytes(delay_samples=0)),
    }

    bundle, f_l, m_l, p_l, f_r, m_r, p_r = load_bass_integration_measurements(data)

    assert bundle is not None
    assert f_l is not None and m_l is not None and p_l is not None
    assert f_r is not None and m_r is not None and p_r is not None
    assert int(bundle.l_main.sample_rate) == 48000
    assert bundle.profile == "normal"
    assert "cancellation_risk" in bundle.diagnostics

    idx = int(np.argmin(np.abs(np.asarray(f_l, dtype=float) - 1000.0)))
    l_main_db = float(bundle.l_main.mag_db[idx])
    l_returned_db = float(np.asarray(m_l, dtype=float)[idx])
    r_main_db = float(bundle.r_main.mag_db[idx])
    r_returned_db = float(np.asarray(m_r, dtype=float)[idx])

    np.testing.assert_allclose(
        np.asarray(m_l, dtype=float),
        np.asarray(bundle.l_main.mag_db, dtype=float),
        atol=1e-9,
    )
    np.testing.assert_allclose(
        np.asarray(m_r, dtype=float),
        np.asarray(bundle.r_main.mag_db, dtype=float),
        atol=1e-9,
    )
    assert abs(l_returned_db - l_main_db) < 1e-9
    assert abs(r_returned_db - r_main_db) < 1e-9


def test_load_bass_integration_measurements_uses_shared_anchor_for_late_rew_ir() -> None:
    base_delay = 48_000
    data = {
        "bass_integration_enable": True,
        "bass_integration_mode": "avr_lfe_main_decomposed",
        "bass_integration_profile": "safe",
        "avr_crossover_hz": 120.0,
        "file_l_main": _upload_dict("l_main.wav", _impulse_wav_bytes(length=64_000, delay_samples=base_delay)),
        "file_r_main": _upload_dict("r_main.wav", _impulse_wav_bytes(length=64_000, delay_samples=base_delay)),
        "file_l_sub": _upload_dict("l_sub.wav", _impulse_wav_bytes(length=64_000, delay_samples=base_delay)),
        "file_r_sub": _upload_dict("r_sub.wav", _impulse_wav_bytes(length=64_000, delay_samples=base_delay)),
    }

    bundle, f_l, m_l, _p_l, _f_r, _m_r, _p_r = load_bass_integration_measurements(data)

    assert bundle is not None
    idx = int(np.argmin(np.abs(np.asarray(f_l, dtype=float) - 1000.0)))
    assert float(bundle.l_main.mag_db[idx]) > -1.0
    assert float(bundle.l_sub.mag_db[idx]) > -1.0
    np.testing.assert_allclose(
        np.asarray(m_l, dtype=float),
        np.asarray(bundle.l_main.mag_db, dtype=float),
        atol=1e-9,
    )
    assert float(np.asarray(m_l, dtype=float)[idx]) > -1.0


def test_load_bass_integration_measurements_accepts_single_direct_dac_sub() -> None:
    data = {
        "bass_integration_enable": True,
        "bass_integration_mode": "direct_dac",
        "bass_integration_profile": "safe",
        "file_l_main": _upload_dict("l_main.wav", _impulse_wav_bytes(delay_samples=0)),
        "file_r_main": _upload_dict("r_main.wav", _impulse_wav_bytes(delay_samples=0)),
        "file_l_sub": _upload_dict("sub_1.wav", _impulse_wav_bytes(delay_samples=0)),
    }

    bundle, f_l, m_l, _p_l, f_r, m_r, _p_r = load_bass_integration_measurements(data)

    assert bundle is not None
    assert f_l is not None and m_l is not None
    assert f_r is not None and m_r is not None
    assert float(np.max(np.abs(bundle.r_sub.complex_spec))) == 0.0

    # In direct_dac mode the returned m_l/m_r come from the main-only measurement,
    # not the combined total, so the level is ~0 dB for a unit impulse (not +6 dB).
    idx = int(np.argmin(np.abs(np.asarray(f_l, dtype=float) - 1000.0)))
    assert float(np.asarray(m_l, dtype=float)[idx]) > -1.0
    assert float(np.asarray(m_r, dtype=float)[idx]) > -1.0


def test_prepare_dual_sub_peak_aligned_average_preserves_complex_phase() -> None:
    freqs = np.asarray([20.0, 40.0, 80.0, 160.0], dtype=float)
    sub1 = _flat_transfer(freqs, 1.0 + 0.0j, label="sub1")
    sub2 = _flat_transfer(freqs, 0.0 + 1.0j, label="sub2")

    combined, diagnostics = prepare_dual_sub_peak_aligned_average(
        sub1,
        sub2,
        sub1_peak_samples=0,
        sub2_peak_samples=0,
    )

    np.testing.assert_allclose(combined.complex_spec, np.full(freqs.shape, 0.5 + 0.5j), atol=1e-12)
    assert diagnostics["dual_sub_combined_method"] == "peak_aligned_complex_vector_average"


def test_load_bass_integration_measurements_direct_dac_peak_aligns_dual_subs() -> None:
    data = {
        "bass_integration_enable": True,
        "bass_integration_mode": "direct_dac",
        "bass_integration_profile": "safe",
        "bass_integration_sub_combine_mode": "sum",
        "file_l_main": _upload_dict("l_main.wav", _impulse_wav_bytes(delay_samples=0)),
        "file_r_main": _upload_dict("r_main.wav", _impulse_wav_bytes(delay_samples=0)),
        "file_l_sub": _upload_dict("sub_1.wav", _impulse_wav_bytes(delay_samples=0)),
        "file_r_sub": _upload_dict("sub_2.wav", _impulse_wav_bytes(delay_samples=24)),
    }

    bundle, f_l, m_l, _p_l, f_r, m_r, _p_r = load_bass_integration_measurements(data)

    assert bundle is not None
    assert f_l is not None and m_l is not None
    assert f_r is not None and m_r is not None
    diag = dict(bundle.diagnostics)
    assert diag["dual_sub_preprocessing_applied"] is True
    assert diag["dual_sub_sub1_peak_samples"] == 0
    assert diag["dual_sub_sub2_peak_samples"] == 24
    assert diag["dual_sub_relative_delay_samples"] == -24
    assert diag["dual_sub_original_sub_combine_mode"] == "sum"
    assert diag["sub_slots_present"] == ["l_sub"]
    assert diag["sub_combine_mode"] == "dual_sub_peak_aligned_average"
    assert float(np.max(np.abs(bundle.r_sub.complex_spec))) == 0.0
    idx = int(np.argmin(np.abs(np.asarray(bundle.l_sub.freqs_hz, dtype=float) - 1000.0)))
    assert float(bundle.l_sub.mag_db[idx]) > -0.1


def test_load_bass_integration_measurements_direct_dac_single_sub_has_no_dual_sub_preprocessing() -> None:
    data = {
        "bass_integration_enable": True,
        "bass_integration_mode": "direct_dac",
        "bass_integration_profile": "safe",
        "file_l_main": _upload_dict("l_main.wav", _impulse_wav_bytes(delay_samples=0)),
        "file_r_main": _upload_dict("r_main.wav", _impulse_wav_bytes(delay_samples=0)),
        "file_l_sub": _upload_dict("sub_1.wav", _impulse_wav_bytes(delay_samples=0)),
    }

    bundle, *_rest = load_bass_integration_measurements(data)

    assert bundle is not None
    assert "dual_sub_preprocessing_applied" not in bundle.diagnostics
    assert bundle.diagnostics["sub_slots_present"] == ["l_sub"]


def test_load_bass_integration_measurements_legacy_mode_uses_direct_dual_sub_preprocessing() -> None:
    data = {
        "bass_integration_enable": True,
        "bass_integration_mode": "avr_lfe_main_decomposed",
        "bass_integration_profile": "safe",
        "bass_integration_sub_combine_mode": "sum",
        "file_l_main": _upload_dict("l_main.wav", _impulse_wav_bytes(delay_samples=0)),
        "file_r_main": _upload_dict("r_main.wav", _impulse_wav_bytes(delay_samples=0)),
        "file_l_sub": _upload_dict("sub_1.wav", _impulse_wav_bytes(delay_samples=0)),
        "file_r_sub": _upload_dict("sub_2.wav", _impulse_wav_bytes(delay_samples=24)),
    }

    bundle, *_rest = load_bass_integration_measurements(data)

    assert bundle is not None
    assert bundle.diagnostics["dual_sub_preprocessing_applied"] is True
    assert bundle.diagnostics["sub_slots_present"] == ["l_sub"]
    assert bundle.diagnostics["sub_combine_mode"] == "dual_sub_peak_aligned_average"
    assert bundle.diagnostics["dual_sub_original_sub_combine_mode"] == "sum"
    assert float(np.max(np.abs(bundle.r_sub.complex_spec))) == 0.0


def test_auto_measurement_signature_includes_dual_sub_preprocessing_metadata() -> None:
    freqs = np.asarray([20.0, 40.0, 80.0, 160.0], dtype=float)
    main = _flat_transfer(freqs, 1.0 + 0.0j, label="main")
    sub = _flat_transfer(freqs, 1.0 + 0.0j, label="sub")
    silent = _flat_transfer(freqs, 0.0 + 0.0j, label="silent")
    base = {
        "f_l": freqs,
        "m_l": np.zeros_like(freqs),
        "f_r": freqs,
        "m_r": np.zeros_like(freqs),
        "bass_integration_enabled": True,
        "avr_crossover_hz": 80.0,
        "bass_integration_profile": "safe",
        "bass_integration_mode": "direct_dac",
    }
    bundle_a = BassIntegrationBundle(
        l_main=main,
        r_main=main,
        l_sub=sub,
        r_sub=silent,
        l_total=main,
        r_total=main,
        avr_crossover_hz=80.0,
        profile="safe",
        diagnostics={
            "dual_sub_preprocessing_applied": True,
            "dual_sub_preprocessing_version": 1,
            "dual_sub_combined_method": "peak_aligned_complex_vector_average",
            "dual_sub_relative_delay_samples": -24,
            "dual_sub_original_sub_combine_mode": "sum",
            "sub_combine_mode": "dual_sub_peak_aligned_average",
        },
    )
    bundle_b = BassIntegrationBundle(
        l_main=main,
        r_main=main,
        l_sub=sub,
        r_sub=silent,
        l_total=main,
        r_total=main,
        avr_crossover_hz=80.0,
        profile="safe",
        diagnostics={**bundle_a.diagnostics, "dual_sub_relative_delay_samples": -12},
    )

    assert _auto_measurement_signature({**base, "bass_integration_bundle": bundle_a}) != _auto_measurement_signature(
        {**base, "bass_integration_bundle": bundle_b}
    )


def test_collect_ui_data_preserves_direct_dac_and_forces_main_hpf_from_sub_xo() -> None:
    data = collect_ui_data(
        {
            "mode": "AUTO",
            "bass_integration_enable": True,
            "bass_integration_mode": "direct_dac",
            "bass_integration_allpass_auto_enable": True,
            "sub_crossover_hz": 92.0,
            "sub_crossover_slope": 24,
            "sub_crossover_manual_override": True,
            "sub_hpf_freq": 19.0,
            "sub_hpf_slope": 12,
        }
    )
    xos, hpf = build_xos_hpf(data)
    cfg = build_filter_config(
        FilterConfig_cls=FilterConfig,
        fs_v=48000,
        taps_v=65536,
        data=data,
        xos=xos,
        hpf=hpf,
        hc_f=None,
        hc_m=None,
    )

    assert data["bass_integration_mode"] == "direct_dac"
    assert data["bass_integration_allpass_auto_enable"] is False
    assert data["sub_crossover_manual_override"] is True
    assert cfg.sub_integration_enable is True
    assert cfg.sub_generate_ir is True
    assert cfg.hpf_settings is None
    assert float(cfg.sub_crossover_hz) == 92.0
    assert float(cfg.direct_dac_sub_lpf_hz) == 92.0
    assert int(cfg.sub_crossover_order) == 4
    assert float(cfg.sub_hpf_freq) == 19.0
    assert int(cfg.sub_hpf_order) == 2


def test_build_filter_config_keeps_direct_dac_sub_lpf_separate_from_main_hpf() -> None:
    data = collect_ui_data(
        {
            "mode": "AUTO",
            "bass_integration_enable": True,
            "bass_integration_mode": "direct_dac",
            "sub_crossover_hz": 73.0,
            "direct_dac_sub_lpf_hz": 118.0,
            "sub_crossover_slope": 24,
            "sub_hpf_freq": 19.0,
            "sub_hpf_slope": 12,
        }
    )
    xos, hpf = build_xos_hpf(data)
    cfg = build_filter_config(
        FilterConfig_cls=FilterConfig,
        fs_v=48000,
        taps_v=65536,
        data=data,
        xos=xos,
        hpf=hpf,
        hc_f=None,
        hc_m=None,
    )

    assert cfg.hpf_settings is None
    assert float(cfg.sub_crossover_hz) == 73.0
    assert float(cfg.direct_dac_sub_lpf_hz) == 118.0


def test_build_xos_hpf_ignores_main_speaker_xo_for_unsupported_filter_types() -> None:
    xos, hpf = build_xos_hpf(
        {
            "filter_type": "mixed phase",
            "xo1_f": 500.0,
            "xo1_s": 12,
            "hpf_enable": True,
            "hpf_freq": 20.0,
            "hpf_slope": 24,
        }
    )

    assert xos == []
    assert hpf == {"enabled": True, "freq": 20.0, "order": 4}


def test_main_speaker_xo_summary_reports_off_for_unsupported_filter_types() -> None:
    summary = _append_main_speaker_xo_hpf_summary(
        "",
        {
            "filter_type": "minimum phase",
            "xo1_f": 500.0,
            "xo1_s": 12,
            "hpf_enable": True,
            "hpf_freq": 20.0,
            "hpf_slope": 24,
        },
    )

    assert "Crossovers: OFF" in summary
    assert "500.0 Hz / 12 dB/oct" not in summary
    assert "HPF: ON (20.0 Hz / 24 dB/oct)" in summary


def test_direct_dac_summary_reports_main_hpf_from_sub_crossover() -> None:
    summary = _append_main_speaker_xo_hpf_summary(
        "",
        {
            "bass_integration_enable": True,
            "bass_integration_mode": "direct_dac",
            "sub_crossover_hz": 40.0,
            "sub_crossover_slope": 24,
            "hpf_enable": True,
            "hpf_freq": 17.9,
            "hpf_slope": 12,
        },
    )

    assert "HPF: ON (40.0 Hz / 24 dB/oct)" in summary
    assert "17.9 Hz / 12 dB/oct" not in summary


def test_direct_dac_summary_labels_xo_as_main_sub_crossover() -> None:
    summary = _append_bass_integration_summary(
        "",
        {
            "bass_integration_enable": True,
            "bass_integration_mode": "direct_dac",
            "sub_crossover_hz": 70.0,
            "direct_dac_sub_lpf_hz": 118.0,
            "_bass_integration_meta": {
                "mode": "direct_dac",
                "avr_crossover_hz": 70.0,
                "direct_dac_sub_lpf_hz": 118.0,
                "sub_target_lpf_hz": 200.0,
                "sub_target_lpf_slope_db_per_oct": 96.0,
                "recommended_crossover_hz": 80.0,
                "recommended_sub_lpf_hz": 118.0,
                "profile": "safe",
                "inputs": {},
            },
        },
    )

    assert "Main HPF: 70.0 Hz" in summary
    assert "Sub LPF: 118.0 Hz" in summary
    assert "Sub target LPF: 200 Hz, 96 dB/oct" in summary
    assert "Recommended Main HPF: 80 Hz" in summary
    assert "Recommended Sub LPF: 118 Hz" in summary
    assert "Direct-DAC XO is the crossover between the main-speaker HPF and the subwoofer LPF." in summary
    assert "AVR crossover:" not in summary
    assert "AUTO may use overlap by setting Sub FIR LPF above Main HPF." in summary


def test_direct_dac_summary_reports_dual_sub_preprocessing() -> None:
    summary = _append_bass_integration_summary(
        "",
        {
            "bass_integration_enable": True,
            "bass_integration_mode": "direct_dac",
            "sub_crossover_hz": 70.0,
            "_bass_integration_meta": {
                "mode": "direct_dac",
                "avr_crossover_hz": 70.0,
                "profile": "safe",
                "sub_combine_mode": "dual_sub_peak_aligned_average",
                "inputs": {},
                "diagnostics": {
                    "dual_sub_preprocessing_applied": True,
                    "dual_sub_sub1_peak_ms": 0.0,
                    "dual_sub_sub2_peak_ms": 0.5,
                    "dual_sub_relative_delay_ms": -0.5,
                },
            },
        },
    )

    assert "Bass Integration dual-sub preprocessing:" in summary
    assert "- SUB1 peak: 0.000 ms" in summary
    assert "- SUB2 peak: 0.500 ms" in summary
    assert "- Applied relative delay: -0.500 ms" in summary
    assert "- Combined sub response: vector average of aligned SUB1 + SUB2" in summary


def test_bass_integration_summary_uses_direct_dac_mode_for_legacy_mode_input() -> None:
    summary = _append_bass_integration_summary(
        "",
        {
            "bass_integration_enable": True,
            "bass_integration_mode": "avr_lfe_main_decomposed",
            "avr_crossover_hz": 80.0,
            "_bass_integration_meta": {
                "mode": "avr_lfe_main_decomposed",
                "avr_crossover_hz": 80.0,
                "direct_dac_sub_lpf_hz": 100.0,
                "profile": "safe",
                "inputs": {},
            },
        },
    )

    assert "Mode: Direct DAC / CamillaDSP sub output" in summary
    assert "Main HPF: 80.0 Hz" in summary
    assert "Sub LPF: 100.0 Hz" in summary
    assert "LFE+Main" not in summary
    assert "AVR crossover" not in summary


def test_compute_health_blocks_bass_integration_without_all_four_wav_files() -> None:
    hr = compute_health(
        {
            "mode": "AUTO",
            "bass_integration_enable": True,
            "avr_crossover_hz": 80.0,
            "file_l_main": _upload_dict("l_main.wav", _impulse_wav_bytes(delay_samples=0)),
        },
        mode="AUTO",
    )

    assert hr.blocked
    assert any(issue.level == "crit" for issue in hr.issues)
    assert any("Bass Integration" in issue.title for issue in hr.issues)


def test_compute_health_accepts_single_direct_dac_sub() -> None:
    hr = compute_health(
        {
            "mode": "AUTO",
            "bass_integration_enable": True,
            "bass_integration_mode": "direct_dac",
            "avr_crossover_hz": 80.0,
            "file_l_main": _upload_dict("l_main.wav", _impulse_wav_bytes(delay_samples=0)),
            "file_r_main": _upload_dict("r_main.wav", _impulse_wav_bytes(delay_samples=0)),
            "file_l_sub": _upload_dict("sub_1.wav", _impulse_wav_bytes(delay_samples=0)),
        },
        mode="AUTO",
    )

    assert not any(issue.level == "crit" for issue in hr.issues)


def test_auto_measurement_signature_changes_when_component_split_changes() -> None:
    data_a = {
        "bass_integration_enable": True,
        "file_l_main": _upload_dict("l_main.wav", _impulse_wav_bytes(delay_samples=0)),
        "file_r_main": _upload_dict("r_main.wav", _impulse_wav_bytes(delay_samples=0)),
        "file_l_sub": _upload_dict("l_sub.wav", _impulse_wav_bytes(delay_samples=24)),
        "file_r_sub": _upload_dict("r_sub.wav", _impulse_wav_bytes(delay_samples=24)),
    }
    data_b = {
        "bass_integration_enable": True,
        "file_l_main": _upload_dict("l_main.wav", _impulse_wav_bytes(delay_samples=0)),
        "file_r_main": _upload_dict("r_main.wav", _impulse_wav_bytes(delay_samples=0)),
        "file_l_sub": _upload_dict("l_sub.wav", _impulse_wav_bytes(delay_samples=12)),
        "file_r_sub": _upload_dict("r_sub.wav", _impulse_wav_bytes(delay_samples=12)),
    }

    bundle_a, f_l_a, m_l_a, p_l_a, f_r_a, m_r_a, p_r_a = load_bass_integration_measurements(data_a)
    bundle_b, f_l_b, m_l_b, p_l_b, f_r_b, m_r_b, p_r_b = load_bass_integration_measurements(data_b)

    assert bundle_a is not None
    assert bundle_b is not None

    sig_a = _auto_measurement_signature(
        {
            "f_l": f_l_a,
            "m_l": m_l_a,
            "p_l": p_l_a,
            "f_r": f_r_a,
            "m_r": m_r_a,
            "p_r": p_r_a,
            "bass_integration_enabled": True,
            "bass_integration_bundle": bundle_a,
            "avr_crossover_hz": 80.0,
            "bass_integration_profile": "safe",
            "bass_integration_mode": "direct_dac",
        }
    )
    sig_b = _auto_measurement_signature(
        {
            "f_l": f_l_b,
            "m_l": m_l_b,
            "p_l": p_l_b,
            "f_r": f_r_b,
            "m_r": m_r_b,
            "p_r": p_r_b,
            "bass_integration_enabled": True,
            "bass_integration_bundle": bundle_b,
            "avr_crossover_hz": 80.0,
            "bass_integration_profile": "safe",
            "bass_integration_mode": "direct_dac",
        }
    )

    assert sig_a != sig_b


def test_run_pipeline_generates_direct_dac_sub_ir_from_sub_bundle(monkeypatch) -> None:
    freqs = np.asarray([20.0, 40.0, 80.0, 160.0], dtype=float)
    l_main = _flat_transfer(freqs, 1.0 + 0.0j, label="l_main")
    r_main = _flat_transfer(freqs, 1.0 + 0.0j, label="r_main")
    l_sub = _flat_transfer(freqs, 2.0 + 0.0j, label="sub_1")
    r_sub = _flat_transfer(freqs, 3.0 + 0.0j, label="sub_2")
    bundle = BassIntegrationBundle(
        l_main=l_main,
        r_main=r_main,
        l_sub=l_sub,
        r_sub=r_sub,
        l_total=_flat_transfer(freqs, 6.0 + 0.0j, label="l_total"),
        r_total=_flat_transfer(freqs, 6.0 + 0.0j, label="r_total"),
        avr_crossover_hz=80.0,
        profile="safe",
        diagnostics={},
    )

    seen_mags: list[np.ndarray] = []

    def _fake_generate_filter(freqs_in, mags_in, phases_in, cfg, *, stereo_link_ctx=None):
        seen_mags.append(np.asarray(mags_in, dtype=float))
        imp = np.zeros(32, dtype=float)
        imp[0] = 1.0
        return imp, {
            "freq_axis": np.asarray(freqs_in, dtype=float),
            "filter_mags": np.zeros_like(np.asarray(freqs_in, dtype=float)),
            "analysis_mode": "native",
            "delay_samples": 0.0,
        }

    monkeypatch.setattr("decaycore.engine_run.dsp.generate_filter", _fake_generate_filter)

    cfg = FilterConfig(
        fs=48000,
        num_taps=4096,
        bass_integration_enable=True,
        bass_integration_mode="direct_dac",
        sub_integration_enable=True,
        sub_generate_ir=True,
        sub_crossover_hz=80.0,
        sub_crossover_order=4,
        sub_hpf_freq=20.0,
        sub_hpf_order=2,
    )
    result = run_pipeline(
        cfg,
        {
            "f_l": freqs,
            "m_l": np.zeros_like(freqs),
            "p_l": np.zeros_like(freqs),
            "f_r": freqs,
            "m_r": np.zeros_like(freqs),
            "p_r": np.zeros_like(freqs),
            "bass_integration_bundle": bundle,
            "bass_integration_enabled": True,
            "bass_integration_mode": "direct_dac",
            "ui_data": {"comparison_mode": False},
        },
    )

    assert result.sub_ir is not None
    assert len(seen_mags) == 3
    np.testing.assert_allclose(
        seen_mags[-1],
        20.0 * np.log10(np.full(freqs.shape, 2.5, dtype=float)),
        atol=1e-6,
    )
    assert np.all(np.isfinite(np.asarray(result.l_st["direct_dac_sum_measured_mags"], dtype=float)))
    assert np.all(np.isfinite(np.asarray(result.l_st["direct_dac_sum_predicted_mags"], dtype=float)))
    assert np.all(np.isfinite(np.asarray(result.r_st["direct_dac_sum_predicted_mags"], dtype=float)))


def test_run_pipeline_uses_direct_dac_sub_lpf_for_sub_pass(monkeypatch) -> None:
    freqs = np.asarray([20.0, 40.0, 80.0, 160.0], dtype=float)
    seen_cfgs: list[dict] = []

    def _fake_generate_filter(freqs_in, mags_in, phases_in, cfg, *, stereo_link_ctx=None):
        seen_cfgs.append(
            {
                "lpf_settings": dict(getattr(cfg, "lpf_settings", {}) or {}),
                "hpf_settings": dict(getattr(cfg, "hpf_settings", {}) or {}),
                "mag_c_max": float(getattr(cfg, "mag_c_max", 0.0) or 0.0),
                "house_freqs": list(getattr(cfg, "house_freqs", []) or []),
                "house_mags": list(getattr(cfg, "house_mags", []) or []),
            }
        )
        imp = np.zeros(32, dtype=float)
        imp[0] = 1.0
        return imp, {
            "freq_axis": np.asarray(freqs_in, dtype=float),
            "filter_mags": np.zeros_like(np.asarray(freqs_in, dtype=float)),
            "analysis_mode": "native",
            "delay_samples": 0.0,
        }

    monkeypatch.setattr("decaycore.engine_run.dsp.generate_filter", _fake_generate_filter)

    cfg = FilterConfig(
        fs=48000,
        num_taps=4096,
        bass_integration_enable=True,
        bass_integration_mode="direct_dac",
        sub_integration_enable=True,
        sub_generate_ir=True,
        sub_crossover_hz=73.0,
        direct_dac_sub_lpf_hz=118.0,
        sub_crossover_order=4,
        sub_hpf_freq=20.0,
        sub_hpf_order=2,
        house_freqs=[20.0, 80.0, 200.0, 1000.0],
        house_mags=[6.0, 3.0, 0.0, 0.0],
    )
    result = run_pipeline(
        cfg,
        {
            "f_l": freqs,
            "m_l": np.zeros_like(freqs),
            "p_l": np.zeros_like(freqs),
            "f_r": freqs,
            "m_r": np.zeros_like(freqs),
            "p_r": np.zeros_like(freqs),
            "ui_data": {"comparison_mode": False},
        },
    )

    assert result.sub_ir is not None
    assert len(seen_cfgs) == 3
    assert seen_cfgs[-1]["lpf_settings"] == {}
    assert seen_cfgs[-1]["hpf_settings"] == {}
    assert abs(float(seen_cfgs[-1]["mag_c_max"]) - 200.0) < 1e-6
    assert seen_cfgs[0]["house_freqs"] == [20.0, 80.0, 200.0, 1000.0]
    assert seen_cfgs[1]["house_freqs"] == [20.0, 80.0, 200.0, 1000.0]
    assert np.interp(80.0, seen_cfgs[-1]["house_freqs"], seen_cfgs[-1]["house_mags"]) == pytest.approx(3.0)
    assert np.interp(400.0, seen_cfgs[-1]["house_freqs"], seen_cfgs[-1]["house_mags"]) == pytest.approx(-80.0)
    assert result.sub_st["sub_target_policy"] == "main_target_plus_steep_200hz_lpf"
    assert result.metrics["bass_integration_sub_target_lpf_hz"] == pytest.approx(200.0)


def test_run_pipeline_applies_direct_dac_alignment_trim_to_sub_bundle(monkeypatch) -> None:
    freqs = np.asarray([20.0, 40.0, 80.0, 160.0], dtype=float)
    l_main = _flat_transfer(freqs, 1.0 + 0.0j, label="l_main")
    r_main = _flat_transfer(freqs, 1.0 + 0.0j, label="r_main")
    l_sub = _flat_transfer(freqs, 2.0 + 0.0j, label="sub_1")
    r_sub = _flat_transfer(freqs, 3.0 + 0.0j, label="sub_2")
    bundle = BassIntegrationBundle(
        l_main=l_main,
        r_main=r_main,
        l_sub=l_sub,
        r_sub=r_sub,
        l_total=_flat_transfer(freqs, 6.0 + 0.0j, label="l_total"),
        r_total=_flat_transfer(freqs, 6.0 + 0.0j, label="r_total"),
        avr_crossover_hz=80.0,
        profile="safe",
        diagnostics={},
    )

    seen_mags: list[np.ndarray] = []

    def _fake_generate_filter(freqs_in, mags_in, phases_in, cfg, *, stereo_link_ctx=None):
        seen_mags.append(np.asarray(mags_in, dtype=float))
        imp = np.zeros(32, dtype=float)
        imp[0] = 1.0
        return imp, {
            "freq_axis": np.asarray(freqs_in, dtype=float),
            "filter_mags": np.zeros_like(np.asarray(freqs_in, dtype=float)),
            "analysis_mode": "native",
            "delay_samples": 0.0,
        }

    monkeypatch.setattr("decaycore.engine_run.dsp.generate_filter", _fake_generate_filter)

    cfg = FilterConfig(
        fs=48000,
        num_taps=4096,
        bass_integration_enable=True,
        bass_integration_mode="direct_dac",
        sub_integration_enable=True,
        sub_generate_ir=True,
        sub_crossover_hz=80.0,
        sub_crossover_order=4,
        sub_hpf_freq=20.0,
        sub_hpf_order=2,
    )
    gain_trim_db = 6.0
    result = run_pipeline(
        cfg,
        {
            "f_l": freqs,
            "m_l": np.zeros_like(freqs),
            "p_l": np.zeros_like(freqs),
            "f_r": freqs,
            "m_r": np.zeros_like(freqs),
            "p_r": np.zeros_like(freqs),
            "bass_integration_bundle": bundle,
            "bass_integration_enabled": True,
            "bass_integration_mode": "direct_dac",
            "bass_integration_sub_gain_trim_db": gain_trim_db,
            "ui_data": {"comparison_mode": False, "bass_integration_sub_gain_trim_db": gain_trim_db},
        },
    )

    assert result.sub_ir is not None
    gain_factor = float(np.power(10.0, gain_trim_db / 20.0))
    expected_sub_mag = 20.0 * np.log10(np.full(freqs.shape, 2.5 * gain_factor, dtype=float))
    np.testing.assert_allclose(seen_mags[-1], expected_sub_mag, atol=1e-6)
    assert np.all(np.isfinite(np.asarray(result.l_st["direct_dac_sum_measured_mags"], dtype=float)))


def test_run_pipeline_aligns_direct_dac_sub_ir_to_main_ir_delay(monkeypatch) -> None:
    freqs = np.asarray([20.0, 40.0, 80.0, 160.0], dtype=float)
    calls = {"n": 0}

    def _fake_generate_filter(freqs_in, mags_in, phases_in, cfg, *, stereo_link_ctx=None):
        calls["n"] += 1
        delay_samples = 10.0 if calls["n"] <= 2 else 2.0
        peak_idx = int(round(delay_samples))
        imp = np.zeros(32, dtype=float)
        imp[peak_idx] = 1.0
        return imp, {
            "freq_axis": np.asarray(freqs_in, dtype=float),
            "filter_mags": np.zeros_like(np.asarray(freqs_in, dtype=float)),
            "analysis_mode": "native",
            "delay_samples": delay_samples,
        }

    monkeypatch.setattr("decaycore.engine_run.dsp.generate_filter", _fake_generate_filter)

    cfg = FilterConfig(
        fs=48000,
        num_taps=4096,
        bass_integration_enable=True,
        bass_integration_mode="direct_dac",
        sub_integration_enable=True,
        sub_generate_ir=True,
        sub_crossover_hz=80.0,
        sub_crossover_order=4,
        sub_hpf_freq=20.0,
        sub_hpf_order=2,
    )
    result = run_pipeline(
        cfg,
        {
            "f_l": freqs,
            "m_l": np.zeros_like(freqs),
            "p_l": np.zeros_like(freqs),
            "f_r": freqs,
            "m_r": np.zeros_like(freqs),
            "p_r": np.zeros_like(freqs),
            "ui_data": {"comparison_mode": False},
        },
        include_response_arrays=False,
    )

    assert result.sub_ir is not None
    assert int(np.argmax(np.abs(result.l_ir))) == 10
    assert int(np.argmax(np.abs(result.r_ir))) == 10
    assert int(np.argmax(np.abs(result.sub_ir))) == 2


def _rolloff_transfer(
    freqs_hz: np.ndarray,
    *,
    f3db: float,
    order: int = 4,
    label: str,
) -> TransferData:
    """TransferData with a Butterworth-like low-pass rolloff for the main speaker."""
    freqs = np.asarray(freqs_hz, dtype=float)
    # Simple Butterworth magnitude: 1 / sqrt(1 + (f/f3db)^(2*order))
    # The speaker is flat in the passband and rolls off below f3db.
    # We model the speaker as a HIGH-pass (passes high freqs, rolls off low).
    hp_mag = (freqs / np.maximum(f3db, 1e-9)) ** order / np.sqrt(
        1.0 + (freqs / np.maximum(f3db, 1e-9)) ** (2 * order)
    )
    hp_mag = np.clip(hp_mag, 1e-12, None)
    spec = hp_mag.astype(np.complex128)  # zero phase, real-positive
    mag_db = 20.0 * np.log10(hp_mag)
    return TransferData(
        freqs_hz=freqs,
        complex_spec=spec,
        mag_db=mag_db,
        phase_deg=np.zeros_like(freqs),
        sample_rate=48000,
        label=label,
    )


def test_recommend_direct_dac_crossover_scores_filtered_acoustic_sum() -> None:
    from decaycore.dsp.bass_integration import recommend_direct_dac_crossover

    freqs = np.logspace(np.log10(10.0), np.log10(20000.0), 1200)
    main = _flat_transfer(freqs, 1.0 + 0.0j, label="main")
    # 6.25 ms delay makes the 80 Hz trial sum acoustically poor after HPF/LPF
    # branch filtering, while 40 Hz remains smooth.
    sub = _delayed_transfer(freqs, 0.00625, label="sub")
    sub_zero = _flat_transfer(freqs, 0.0 + 0.0j, label="sub_r_zero")

    total_spec = main.complex_spec + sub.complex_spec
    total = TransferData(
        freqs_hz=freqs,
        complex_spec=total_spec,
        mag_db=20.0 * np.log10(np.maximum(np.abs(total_spec), 1e-12)),
        phase_deg=np.rad2deg(np.unwrap(np.angle(total_spec))),
        sample_rate=48000,
        label="total",
    )
    bundle = BassIntegrationBundle(
        l_main=main,
        r_main=main,
        l_sub=sub,
        r_sub=sub_zero,
        l_total=total,
        r_total=total,
        avr_crossover_hz=80.0,
        profile="safe",
        diagnostics={},
    )

    result = recommend_direct_dac_crossover(
        bundle,
        candidates=(40.0, 80.0, 120.0),
        profile="safe",
        main_hpf_order=4,
        sub_lpf_order=4,
        sub_hpf_hz=20.0,
        sub_hpf_order=2,
    )

    assert float(result["recommended_hz"]) == 80.0
    assert np.isfinite(float(result["scores"][80.0]["score"]))


def test_compute_bass_integration_diagnostics_keeps_core_guard_band_separate_from_overlap_extension() -> None:
    from decaycore.dsp.bass_integration import compute_bass_integration_diagnostics

    bundle = _make_flat_overlap_bundle()
    diag = compute_bass_integration_diagnostics(
        bundle,
        80.0,
        "safe",
        sub_combine_mode="average",
        sub_lpf_hz=120.0,
    )

    assert abs(float(diag["guard_hi_hz"]) - 112.0) < 1e-6
    assert bool(diag["overlap_extension_active"]) is True
    assert abs(float(diag["overlap_extension_lo_hz"]) - 80.0) < 1e-6
    assert abs(float(diag["overlap_extension_hi_hz"]) - 120.0) < 1e-6
    assert np.isfinite(float(diag["overlap_extension_flatness_db"]))
    assert np.isfinite(float(diag["overlap_extension_cancellation_risk"]))


def test_recommend_direct_dac_crossover_can_select_overlap_sub_lpf(monkeypatch) -> None:
    import decaycore.dsp.bass_integration as bass_integration

    bundle = _make_flat_overlap_bundle(n_points=64)

    def _fake_metrics(_bundle, fc_hz, _profile, *, sub_lpf_hz=None, **_kwargs):
        fc = float(fc_hz)
        sub_lpf = float(sub_lpf_hz or fc)
        distance = abs(fc - 80.0) + 0.15 * abs(sub_lpf - 112.0)
        return {
            "objective": -distance,
            "bass_cancellation_risk": 0.05 + 0.01 * abs(sub_lpf - 112.0) / 32.0,
            "bass_overlap_ripple": 2.0 + 0.05 * distance,
            "bass_sub_dominance": 1.0,
            "bass_null_severity": 0.1,
            "bass_overlap_extension_active": bool(sub_lpf > fc + 1e-6),
            "bass_overlap_extension_flatness_db": 3.0 + 0.05 * abs(sub_lpf - 112.0),
            "bass_overlap_extension_cancellation_risk": 0.08 + 0.01 * abs(sub_lpf - 112.0) / 32.0,
            "bass_overlap_extension_peak_excess_db": 1.5 + 0.05 * abs(sub_lpf - 112.0),
            "bass_overlap_extension_sub_dominance_db": 6.0,
            "bass_xo_gd_rms_mismatch_ms": 2.0,
            "bass_xo_gd_max_mismatch_ms": 3.0,
            "bass_predicted_sum_flatness_db": 2.0,
            "bass_predicted_sum_dip_depth_db": 1.0,
            "bass_predicted_sum_peak_excess_db": 1.0,
            "bass_overlap_ripple_delta_db": 0.5,
            "bass_sub_dominance_delta_db": 0.5,
            "bass_xo_gd_mismatch_delta_ms": 1.0,
            "bass_dominant_channel": "balanced",
            "bass_feasibility_class": "good",
            "bass_feasibility_reason": "Shared mono-sub integration meets balance and crossover guard targets.",
            "bass_metric_channel_mode": "worst_case",
        }

    monkeypatch.setattr(bass_integration, "compute_final_bass_integration_metrics", _fake_metrics)
    monkeypatch.setattr(bass_integration, "_main_guard_band_drop_db", lambda _main, _fc_hz: 0.0)

    result = bass_integration.recommend_direct_dac_crossover(
        bundle,
        candidates=(80.0,),
        profile="safe",
    )

    assert float(result["recommended_hz"]) == 80.0
    assert float(result["recommended_sub_lpf_hz"]) == 112.0
    assert bool(result["scores"][80.0]["overlap_extension_active"]) is True
    assert float(result["scores"][80.0]["sub_lpf_hz"]) == 112.0


def test_compute_final_bass_integration_metrics_flags_shared_sub_asymmetry_as_infeasible() -> None:
    from decaycore.dsp.bass_integration import compute_final_bass_integration_metrics

    bundle = _make_asymmetric_shared_sub_bundle()
    metrics = compute_final_bass_integration_metrics(
        bundle,
        80.0,
        "safe",
        mode="direct_dac",
        sub_combine_mode="sum",
    )

    assert metrics["bass_feasibility_class"] == "infeasible"
    assert metrics["bass_dominant_channel"] == "right"
    assert float(metrics["bass_sub_dominance_delta_db"]) > 10.0
    assert float(metrics["bass_sub_combined_level_delta_db_20_120"]) > 5.5
    assert "Right channel remains limiting" in str(metrics["bass_feasibility_reason"])
    assert "sum combine adds" in str(metrics["bass_feasibility_reason"])
    assert metrics["diagnostics"]["feasibility_class"] == "infeasible"


def test_compute_final_bass_integration_metrics_preserves_explicit_sub_combine_mode_reporting() -> None:
    from decaycore.dsp.bass_integration import compute_final_bass_integration_metrics

    bundle = _make_asymmetric_shared_sub_bundle()
    average = compute_final_bass_integration_metrics(
        bundle,
        80.0,
        "safe",
        mode="direct_dac",
        sub_combine_mode="average",
    )
    summed = compute_final_bass_integration_metrics(
        bundle,
        80.0,
        "safe",
        mode="direct_dac",
        sub_combine_mode="sum",
    )

    assert abs(float(average["bass_sub_combined_level_delta_db_20_120"])) < 0.1
    assert float(summed["bass_sub_combined_level_delta_db_20_120"]) > 5.5
    assert average["bass_feasibility_class"] == "infeasible"
    assert summed["bass_feasibility_class"] == "infeasible"
    assert "sum combine adds" not in str(average["bass_feasibility_reason"])
    assert "sum combine adds" in str(summed["bass_feasibility_reason"])


def test_recommend_direct_dac_alignment_can_still_report_infeasible_shared_sub_solution(monkeypatch) -> None:
    import decaycore.dsp.bass_integration as bass_integration

    bundle = _make_asymmetric_shared_sub_bundle(n_points=80)
    monkeypatch.setattr(bass_integration, "DIRECT_DAC_ALIGNMENT_DELAY_CANDIDATES_MS", (-2.0, 0.0, 2.0))
    monkeypatch.setattr(bass_integration, "DIRECT_DAC_ALIGNMENT_GAIN_CANDIDATES_DB", (-3.0, 0.0, 3.0))

    result = bass_integration.recommend_direct_dac_alignment(
        bundle,
        fc_hz=80.0,
        profile="safe",
        main_hpf_order=4,
        sub_lpf_order=4,
        sub_hpf_hz=20.0,
        sub_hpf_order=2,
        sub_combine_mode="sum",
    )

    assert result["optimized"]["feasibility_class"] == "infeasible"
    assert result["optimized"]["dominant_channel"] == "right"
    assert "Right channel remains limiting" in str(result["optimized"]["feasibility_reason"])


def test_direct_dac_good_sounding_high_diagnostic_metrics_stay_soft_limited() -> None:
    from decaycore.dsp.bass_integration._evaluate_direct_dac import _reject_reasons_for_channel
    from decaycore.dsp.bass_integration._gd_feasibility import _classify_bass_integration_feasibility

    channel_metrics = {
        "cancellation_risk": 0.099,
        "predicted_sum_dip_depth_db": 3.0,
        "null_depth_db": 6.1,
        "null_width_hz": 2.0,
        "xo_gd_rms_mismatch_ms": 33.5,
        "xo_gd_max_mismatch_ms": 150.7,
        "overlap_ripple_db": 20.5,
        "sub_dominance_db": 17.0,
    }

    assert _reject_reasons_for_channel(channel_metrics, "right") == ()

    klass, reason = _classify_bass_integration_feasibility(
        overlap_ripple_worst=20.5,
        sub_dominance_worst=17.0,
        xo_gd_rms_worst=33.5,
        overlap_ripple_delta=3.3,
        sub_dominance_delta=10.7,
        xo_gd_delta=14.4,
        dominant_channel="right",
        sub_combine_mode="dual_sub_peak_aligned_average",
        sub_level_delta_db_20_120=0.0,
    )

    assert klass == "marginal"
    assert "Right channel remains limiting" in reason
    assert "infeasible" not in reason.lower()


def test_recommend_direct_dac_crossover_uses_decimal_grid_by_default(monkeypatch) -> None:
    import decaycore.dsp.bass_integration as bass_integration

    freqs = np.asarray([20.0, 40.0, 80.0, 160.0], dtype=float)
    main = _flat_transfer(freqs, 1.0 + 0.0j, label="main")
    sub = _flat_transfer(freqs, 1.0 + 0.0j, label="sub")
    silent = _flat_transfer(freqs, 0.0 + 0.0j, label="silent")
    bundle = BassIntegrationBundle(
        l_main=main,
        r_main=main,
        l_sub=sub,
        r_sub=silent,
        l_total=main,
        r_total=main,
        avr_crossover_hz=80.0,
        profile="safe",
        diagnostics={},
    )

    def _fake_metrics(_bundle, fc_hz, _profile, **_kwargs):
        delta = abs(float(fc_hz) - 82.5)
        return {
            "objective": -delta,
            "bass_cancellation_risk": delta / 100.0,
            "bass_overlap_ripple": delta,
            "bass_sub_dominance": 0.0,
            "bass_null_severity": 0.0,
            "bass_xo_gd_rms_mismatch_ms": delta / 10.0,
            "bass_xo_gd_max_mismatch_ms": delta / 10.0,
            "bass_overlap_ripple_delta_db": delta / 5.0,
            "bass_sub_dominance_delta_db": delta / 4.0,
            "bass_xo_gd_mismatch_delta_ms": delta / 3.0,
            "bass_dominant_channel": "balanced" if delta < 0.1 else "right",
            "bass_feasibility_class": "good" if delta < 0.1 else "marginal",
            "bass_feasibility_reason": "Balanced shared mono-sub candidate." if delta < 0.1 else "Right channel remains limiting.",
            "bass_predicted_sum_flatness_db": delta,
            "bass_predicted_sum_dip_depth_db": delta,
            "bass_predicted_sum_peak_excess_db": 0.0,
            "bass_metric_channel_mode": "worst_case",
        }

    monkeypatch.setattr(
        bass_integration,
        "compute_final_bass_integration_metrics",
        _fake_metrics,
    )
    monkeypatch.setattr(
        bass_integration,
        "_main_guard_band_drop_db",
        lambda _main, _fc_hz: 0.0,
    )

    result = bass_integration.recommend_direct_dac_crossover(bundle, profile="safe")

    assert float(result["recommended_hz"]) == 82.5
    assert 82.5 in result["scores"]
    assert float(result["scores"][82.5]["score"]) > float(result["scores"][80.0]["score"])
    assert result["feasibility_class"] == "good"
    assert result["dominant_channel"] == "balanced"


def test_direct_dac_auto_hpf_seed_source_uses_sub_bundle_data() -> None:
    freqs = np.asarray([20.0, 40.0, 80.0, 160.0], dtype=float)
    l_main = _flat_transfer(freqs, 1.0 + 0.0j, label="l_main")
    r_main = _flat_transfer(freqs, 1.0 + 0.0j, label="r_main")
    l_sub = _flat_transfer(freqs, 2.0 + 0.0j, label="l_sub")
    r_sub = _flat_transfer(freqs, 3.0 + 0.0j, label="r_sub")
    bundle = BassIntegrationBundle(
        l_main=l_main,
        r_main=r_main,
        l_sub=l_sub,
        r_sub=r_sub,
        l_total=_flat_transfer(freqs, 6.0 + 0.0j, label="l_total"),
        r_total=_flat_transfer(freqs, 6.0 + 0.0j, label="r_total"),
        avr_crossover_hz=80.0,
        profile="safe",
        diagnostics={},
    )

    out = _resolve_auto_hpf_seed_source(
        {"bass_integration_bundle": bundle},
        {"bass_integration_enable": True, "bass_integration_mode": "direct_dac"},
        np.asarray([20.0, 40.0], dtype=float),
        np.asarray([0.0, 0.0], dtype=float),
        np.asarray([20.0, 40.0], dtype=float),
        np.asarray([0.0, 0.0], dtype=float),
    )

    f_l_src, m_l_src, f_r_src, m_r_src, freq_key, slope_key, label, user_hpf_enabled = out
    np.testing.assert_allclose(f_l_src, l_sub.freqs_hz)
    np.testing.assert_allclose(m_l_src, l_sub.mag_db)
    np.testing.assert_allclose(f_r_src, r_sub.freqs_hz)
    np.testing.assert_allclose(m_r_src, r_sub.mag_db)
    assert freq_key == "sub_hpf_freq"
    assert slope_key == "sub_hpf_slope"
    assert label == "Sub HPF"
    assert user_hpf_enabled is True


def test_direct_dac_summary_formats_decimal_recommendation() -> None:
    summary = _append_bass_integration_summary(
        "",
        {
            "bass_integration_enable": True,
            "bass_integration_mode": "direct_dac",
            "sub_crossover_hz": 82.5,
            "direct_dac_sub_lpf_hz": 96.5,
            "_bass_integration_meta": {
                "mode": "direct_dac",
                "avr_crossover_hz": 82.5,
                "direct_dac_sub_lpf_hz": 96.5,
                "recommended_crossover_hz": 82.5,
                "recommended_sub_lpf_hz": 96.5,
                "profile": "safe",
                "inputs": {},
            },
        },
    )

    assert "Main HPF: 82.5 Hz" in summary
    assert "Sub LPF: 96.5 Hz" in summary
    assert "Recommended Main HPF: 82.5 Hz" in summary
    assert "Recommended Sub LPF: 96.5 Hz" in summary


def test_bass_integration_summary_reports_system_limited_feasibility_details() -> None:
    summary = _append_bass_integration_summary(
        "",
        {
            "bass_integration_enable": True,
            "bass_integration_mode": "direct_dac",
            "sub_crossover_hz": 80.0,
            "_auto_mode_meta": {
                "best_metrics": {
                    "bass_feasibility_class": "infeasible",
                    "bass_feasibility_reason": "Right channel remains limiting. sub dominance 12.0 dB.",
                    "bass_dominant_channel": "right",
                    "bass_overlap_ripple": 4.0,
                    "bass_overlap_ripple_delta_db": 1.5,
                    "bass_sub_dominance": 12.0,
                    "bass_sub_dominance_delta_db": 12.0,
                    "bass_xo_gd_rms_mismatch_ms": 5.0,
                    "bass_xo_gd_mismatch_delta_ms": 19.5,
                    "bass_cancellation_risk": 0.2,
                    "bass_null_severity": 0.3,
                }
            },
            "_bass_integration_meta": {
                "mode": "direct_dac",
                "avr_crossover_hz": 80.0,
                "recommended_crossover_hz": 80.0,
                "profile": "safe",
                "inputs": {},
                "diagnostics": {
                    "sub_combined_level_delta_db_20_120": 6.02,
                    "sub_combined_level_delta_db_30_90": 6.02,
                },
            },
        },
    )

    assert "Feasibility: INFEASIBLE (SYSTEM-LIMITED)" in summary
    assert "Dominant / limiting channel: RIGHT" in summary
    assert "Feasibility reason: Right channel remains limiting." in summary
    assert "Overlap ripple delta L/R: 1.500 dB" in summary
    assert "Sub dominance delta L/R: 12.000 dB" in summary
    assert "GD mismatch delta L/R: 19.500 ms" in summary


def test_direct_dac_allpass_summary_block_reports_recommendation_state() -> None:
    summary = _append_bass_integration_allpass_auto_summary(
        "",
        {
            "bass_integration_enable": True,
            "bass_integration_mode": "direct_dac",
            "_bass_integration_meta": {
                "recommended_allpass": {
                    "enabled": True,
                    "freq_hz": 77.5,
                    "q": 0.9,
                    "improvement_score": 0.321,
                    "reason": "Applied shared mono-sub allpass.",
                },
                "allpass_baseline_metrics": {
                    "cancellation_risk": 0.42,
                    "overlap_ripple_db": 4.8,
                    "xo_gd_mismatch_ms": 1.2,
                },
                "allpass_optimized_metrics": {
                    "cancellation_risk": 0.18,
                    "overlap_ripple_db": 3.2,
                    "xo_gd_mismatch_ms": 0.6,
                },
            },
        },
    )

    assert "=== BASS INTEGRATION ALLPASS AUTO ===" in summary
    assert "State: ON" in summary
    assert "Freq: 77.5 Hz" in summary
    assert "Q: 0.900" in summary
    assert "Improvement score: 0.321" in summary
    assert "Applied in the exported CamillaDSP Direct DAC sub pipeline when State is ON." in summary


def test_build_auto_selected_text_reports_sub_hpf_in_direct_dac() -> None:
    txt = _build_auto_selected_text(
        {
            "bass_integration_enable": True,
            "bass_integration_mode": "direct_dac",
            "sub_hpf_freq": 18.5,
            "sub_hpf_slope": 24,
            "hpf_enable": False,
            "hpf_freq": 44.0,
            "hpf_slope": 12,
            "mag_c_min": 28.0,
            "filter_type": "Asymmetric",
            "phase_limit": 500.0,
            "target_curve_name": "Harman8",
            "best_metrics": {},
        }
    )

    assert "Sub HPF 18.5 Hz/24 dB/oct" in txt
    assert "HPF 44.0 Hz/12 dB/oct" not in txt


def test_build_auto_selected_text_uses_final_hpf_state_not_seed_meta() -> None:
    txt = _build_auto_selected_text(
        {
            "hpf_enable": False,
            "hpf_freq": 27.5,
            "hpf_slope": 24,
            "_auto_hpf_meta": {"applied": True, "enabled": True, "freq": 27.5, "slope_db_oct": 24},
            "mag_c_min": 24.0,
            "filter_type": "Asymmetric",
            "phase_limit": 500.0,
            "target_curve_name": "Harman8",
            "best_metrics": {},
        }
    )

    assert "HPF off" in txt
    assert "HPF 27.5 Hz/24 dB/oct" not in txt


def test_direct_dac_score_prefers_in_phase_over_cancellation() -> None:
    from decaycore.dsp.bass_integration import score_direct_dac_bass_integration

    freqs = np.linspace(20.0, 220.0, 401)
    main = np.ones_like(freqs, dtype=np.complex128)
    good_sub = np.ones_like(freqs, dtype=np.complex128)
    bad_sub = -np.ones_like(freqs, dtype=np.complex128)

    good = score_direct_dac_bass_integration(main, good_sub, freqs, 80.0, 0.0, 0.0, False)
    bad = score_direct_dac_bass_integration(main, bad_sub, freqs, 80.0, 0.0, 0.0, False)

    assert good["score"] < bad["score"]
    assert good["cancellation_risk"] == "low"


def test_direct_dac_optimizer_uses_inverted_polarity_when_sub_is_opposite() -> None:
    from decaycore.dsp.bass_integration import run_direct_dac_bass_integration

    freqs = np.linspace(20.0, 220.0, 401)
    main = np.ones_like(freqs, dtype=np.complex128)
    sub = -np.ones_like(freqs, dtype=np.complex128)

    result = run_direct_dac_bass_integration(main, sub, freqs)

    assert result.sub_polarity_invert is True
    assert 50.0 <= result.main_hpf_hz <= 120.0
    assert result.sub_lpf_hz == result.main_hpf_hz + 20.0
    assert result.sub_hpf_hz == 20.0


def test_direct_dac_score_prefers_normal_polarity_for_true_tie() -> None:
    from decaycore.dsp.bass_integration import score_direct_dac_bass_integration

    freqs = np.linspace(20.0, 220.0, 401)
    main = np.ones_like(freqs, dtype=np.complex128)
    sub = np.ones_like(freqs, dtype=np.complex128)

    normal = score_direct_dac_bass_integration(main, sub, freqs, 80.0, 0.0, 0.0, False)
    inverted = score_direct_dac_bass_integration(main, sub, freqs, 80.0, 0.0, 0.0, True)

    assert normal["score"] < inverted["score"]


def test_direct_dac_score_handles_sparse_finite_data() -> None:
    from decaycore.dsp.bass_integration import score_direct_dac_bass_integration

    freqs = np.asarray([50.0, 65.0, 80.0, 95.0, 120.0], dtype=float)
    main = np.ones_like(freqs, dtype=np.complex128)
    sub = np.exp(1j * np.linspace(0.0, 0.2, freqs.size))

    score = score_direct_dac_bass_integration(main, sub, freqs, 80.0, 0.0, 0.0, False)

    assert np.isfinite(float(score["score"]))
    assert np.isfinite(float(score["phase_error_deg"]))
