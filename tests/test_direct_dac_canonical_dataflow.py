from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from decaycore.dsp.bass_integration import (
    DirectDacCandidate,
    direct_dac_candidate_from_data,
    evaluate_direct_dac_candidate,
    write_direct_dac_candidate_to_data,
)
from decaycore.engine_run_parts.direct_dac_bass_integration import apply_direct_dac_bass_integration_result
from decaycore.io.measurement_bundle import BassIntegrationBundle, TransferData
from decaycore.ui.export_outputs import _direct_dac_yaml_export_settings


def _transfer(freqs_hz: np.ndarray, spec: np.ndarray | complex, *, label: str) -> TransferData:
    freqs = np.asarray(freqs_hz, dtype=float)
    if np.isscalar(spec):
        complex_spec = np.full(freqs.shape, spec, dtype=np.complex128)
    else:
        complex_spec = np.asarray(spec, dtype=np.complex128)
    return TransferData(
        freqs_hz=freqs,
        complex_spec=complex_spec,
        mag_db=20.0 * np.log10(np.maximum(np.abs(complex_spec), 1e-12)),
        phase_deg=np.rad2deg(np.unwrap(np.angle(complex_spec))),
        sample_rate=48000,
        label=label,
    )


def _bundle(*, r_main_gain: complex = 1.0 + 0.0j, r_sub_gain: complex = 1.0 + 0.0j) -> BassIntegrationBundle:
    freqs = np.linspace(20.0, 220.0, 401)
    l_main = _transfer(freqs, 1.0 + 0.0j, label="l_main")
    r_main = _transfer(freqs, r_main_gain, label="r_main")
    l_sub = _transfer(freqs, 1.0 + 0.0j, label="l_sub")
    r_sub = _transfer(freqs, r_sub_gain, label="r_sub")
    return BassIntegrationBundle(
        l_main=l_main,
        r_main=r_main,
        l_sub=l_sub,
        r_sub=r_sub,
        l_total=_transfer(freqs, 2.0 + 0.0j, label="l_total"),
        r_total=_transfer(freqs, 0.0 + 0.0j, label="r_total"),
        avr_crossover_hz=80.0,
        profile="safe",
        diagnostics={},
    )


def test_direct_dac_export_settings_preserve_optimizer_overlap() -> None:
    settings = _direct_dac_yaml_export_settings(
        {
            "bass_integration_mode": "direct_dac",
            "sub_crossover_hz": 80.0,
            "direct_dac_sub_lpf_hz": 88.0,
        },
        include_sub=True,
    )

    assert settings["sub_lpf_hz"] == 88.0


def test_direct_dac_candidate_roundtrips_to_export_settings() -> None:
    data = {"bass_integration_mode": "direct_dac"}
    candidate = DirectDacCandidate(
        main_hpf_hz=70.0,
        main_hpf_order=4,
        sub_hpf_hz=18.0,
        sub_hpf_order=2,
        sub_lpf_hz=91.0,
        sub_lpf_order=4,
        sub_delay_ms=7.5,
        sub_array_delay_ms=7.5,
        sub1_delay_ms=0.0,
        sub2_delay_ms=-0.5,
        main_l_delay_ms=0.0,
        main_r_delay_ms=0.0,
        sub_gain_trim_db=-2.0,
        sub_polarity_invert=True,
        source="test",
    )

    write_direct_dac_candidate_to_data(data, candidate, None, reason="test")
    settings = _direct_dac_yaml_export_settings(data, include_sub=True)

    assert settings["main_hpf_hz"] == candidate.main_hpf_hz
    assert settings["sub_lpf_hz"] == candidate.sub_lpf_hz
    assert settings["sub_delay_ms"] == candidate.sub_delay_ms
    assert settings["sub_gain_trim_db"] == candidate.sub_gain_trim_db
    assert settings["sub_polarity_invert"] == candidate.sub_polarity_invert
    roundtrip = direct_dac_candidate_from_data(data)
    assert roundtrip.sub_array_delay_ms == candidate.sub_array_delay_ms
    assert roundtrip.sub1_delay_ms == candidate.sub1_delay_ms
    assert roundtrip.sub2_delay_ms == candidate.sub2_delay_ms
    assert roundtrip.main_l_delay_ms == candidate.main_l_delay_ms
    assert roundtrip.main_r_delay_ms == candidate.main_r_delay_ms


def test_direct_dac_evaluator_uses_worst_case_channel_not_lr_average() -> None:
    candidate = DirectDacCandidate(
        main_hpf_hz=80.0,
        main_hpf_order=2,
        sub_hpf_hz=20.0,
        sub_hpf_order=2,
        sub_lpf_hz=80.0,
        sub_lpf_order=2,
        source="test",
    )

    metrics = evaluate_direct_dac_candidate(_bundle(r_main_gain=-1.0 + 0.0j, r_sub_gain=1.0 + 0.0j), candidate, profile="safe")

    assert metrics.dominant_channel == "left"
    assert metrics.left.score > metrics.right.score
    assert metrics.score > ((metrics.left.score + metrics.right.score) / 2.0)
    assert metrics.feasible is True


def test_direct_dac_candidate_reject_notes_do_not_force_auto_mode_hard_gate() -> None:
    from decaycore.dsp.bass_integration import compute_final_bass_integration_metrics
    from decaycore.auto_mode.search.ranking_gates import _auto_hard_gate_reasons

    metrics = compute_final_bass_integration_metrics(
        _bundle(r_main_gain=-1.0 + 0.0j, r_sub_gain=1.0 + 0.0j),
        80.0,
        "safe",
        mode="direct_dac",
        main_hpf_order=2,
        sub_lpf_order=2,
        sub_hpf_hz=20.0,
        sub_hpf_order=2,
        sub_gain_trim_db=13.0,
    )

    assert metrics["bass_metric_channel_mode"] == "worst_case"
    assert "sub_gain_trim_above_safe_range" in metrics["bass_direct_dac_reject_reasons"]
    hard_gate_metrics = {
        **metrics,
        "rank_score": 80.0,
        "avg_score": 80.0,
        "bass_integration_enable": True,
    }
    assert "bass_integration_infeasible_hard_gate" not in _auto_hard_gate_reasons(hard_gate_metrics)


def test_apply_direct_dac_result_verifies_without_reoptimizing_or_overwriting() -> None:
    data = {
        "bass_integration_enable": True,
        "bass_integration_mode": "direct_dac",
        "sub_crossover_hz": 70.0,
        "direct_dac_sub_lpf_hz": 88.0,
        "sub_hpf_freq": 19.0,
        "bass_integration_sub_delay_ms": 7.5,
        "bass_integration_sub_gain_trim_db": -2.0,
        "bass_integration_sub_polarity_invert": True,
        "sub_crossover_slope": 24,
        "sub_hpf_slope": 12,
    }
    cfg = SimpleNamespace(
        fs=48000,
        bass_integration_enable=True,
        bass_integration_mode="direct_dac",
        bass_integration_profile="safe",
    )
    result = apply_direct_dac_bass_integration_result(
        cfg=cfg,
        measurements={"bass_integration_bundle": _bundle(r_sub_gain=1.0 + 0.0j)},
        data=data,
        l_imp=np.asarray([1.0, 0.0]),
        r_imp=np.asarray([1.0, 0.0]),
        sub_ir=np.asarray([1.0, 0.0]),
        l_st={},
        r_st={},
        sub_st={},
    )
    candidate = direct_dac_candidate_from_data(data, cfg)

    assert result["main_hpf_hz"] == 70.0
    assert candidate.main_hpf_hz == 70.0
    assert candidate.sub_lpf_hz == 88.0
    assert candidate.sub_delay_ms == 7.5
    assert candidate.sub_gain_trim_db == -2.0
    assert candidate.sub_polarity_invert is True
    assert isinstance(result["cancellation_risk"], float)
    assert 0.0 <= result["cancellation_risk"] <= 1.0
    assert result["feasibility_class"] in {"good", "marginal", "infeasible"}
    assert data["_bass_integration_meta"]["export_model"] == "camilladsp_yaml_compatible"
    assert data["_bass_integration_meta"]["direct_dac_candidate"]["main_hpf_hz"] == 70.0


def test_direct_dac_candidate_prefers_current_data_over_stale_raw_candidate_alignment() -> None:
    cfg = SimpleNamespace(
        bass_integration_main_l_delay_ms=0.0,
        bass_integration_main_r_delay_ms=0.0,
        bass_integration_sub_delay_ms=0.0,
    )
    data = {
        "bass_integration_mode": "direct_dac",
        "bass_integration_sub_delay_ms": -2.5,
        "bass_integration_sub_array_delay_ms": -2.5,
        "bass_integration_main_l_delay_ms": 2.5,
        "bass_integration_main_r_delay_ms": 2.5,
        "_bass_integration_meta": {
            "direct_dac_candidate": {
                "sub_delay_ms": 0.0,
                "sub_array_delay_ms": 0.0,
                "main_l_delay_ms": 0.0,
                "main_r_delay_ms": 0.0,
            }
        },
    }

    candidate = direct_dac_candidate_from_data(data, cfg)

    assert candidate.sub_delay_ms == -2.5
    assert candidate.sub_array_delay_ms == -2.5
    assert candidate.main_l_delay_ms == 2.5
    assert candidate.main_r_delay_ms == 2.5
