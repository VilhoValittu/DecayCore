from __future__ import annotations

import math
import logging

import numpy as np
import pytest

import decaycore.dsp.bass_integration as bass_integration
from decaycore.dsp.bass_integration import (
    _apply_allpass_to_transfer,
    _apply_branch_filters,
    _build_direct_dac_trial_bundle,
    _get_filtered_branches,
    build_combined_sub_transfer,
    recommend_direct_dac_alignment,
    recommend_direct_dac_allpass,
    recommend_direct_dac_prepare_optuna,
)
from decaycore.io.measurement_bundle import BassIntegrationBundle, TransferData


def _transfer(
    freqs_hz: np.ndarray,
    *,
    delay_s: float = 0.0,
    gain: float = 1.0,
    label: str,
) -> TransferData:
    freqs = np.asarray(freqs_hz, dtype=float)
    spec = float(gain) * np.exp(-1j * 2.0 * np.pi * freqs * float(delay_s))
    return TransferData(
        freqs_hz=freqs,
        complex_spec=np.asarray(spec, dtype=np.complex128),
        mag_db=np.asarray(20.0 * np.log10(np.maximum(np.abs(spec), 1e-12)), dtype=float),
        phase_deg=np.asarray(np.rad2deg(np.unwrap(np.angle(spec))), dtype=float),
        sample_rate=48_000,
        label=label,
    )


def _bundle(
    *,
    main_delay_s: float = 0.0,
    sub_delay_s: float = 0.0,
) -> BassIntegrationBundle:
    freqs = np.geomspace(10.0, 320.0, 1024)
    l_main = _transfer(freqs, delay_s=main_delay_s, gain=1.0, label="l_main")
    r_main = _transfer(freqs, delay_s=main_delay_s, gain=1.0, label="r_main")
    l_sub = _transfer(freqs, delay_s=sub_delay_s, gain=0.5, label="l_sub")
    r_sub = _transfer(freqs, delay_s=sub_delay_s, gain=0.5, label="r_sub")
    l_total = _transfer(freqs, delay_s=0.0, gain=1.0, label="l_total")
    r_total = _transfer(freqs, delay_s=0.0, gain=1.0, label="r_total")
    return BassIntegrationBundle(
        l_main=l_main,
        r_main=r_main,
        l_sub=l_sub,
        r_sub=r_sub,
        l_total=l_total,
        r_total=r_total,
        avr_crossover_hz=80.0,
        profile="safe",
        diagnostics={},
    )


def test_direct_dac_trial_bundle_identity_when_allpass_not_provided() -> None:
    bundle = _bundle()

    baseline = _build_direct_dac_trial_bundle(
        bundle,
        fc_hz=80.0,
        main_hpf_order=4,
        sub_lpf_order=4,
        sub_hpf_hz=20.0,
        sub_hpf_order=2,
    )
    explicit_none = _build_direct_dac_trial_bundle(
        bundle,
        fc_hz=80.0,
        main_hpf_order=4,
        sub_lpf_order=4,
        sub_hpf_hz=20.0,
        sub_hpf_order=2,
        sub_allpass_freq_hz=None,
        sub_allpass_q=None,
    )

    np.testing.assert_allclose(baseline.l_sub.complex_spec, explicit_none.l_sub.complex_spec, atol=1e-12)
    np.testing.assert_allclose(baseline.r_sub.complex_spec, explicit_none.r_sub.complex_spec, atol=1e-12)
    np.testing.assert_allclose(baseline.l_total.complex_spec, explicit_none.l_total.complex_spec, atol=1e-12)
    np.testing.assert_allclose(baseline.r_total.complex_spec, explicit_none.r_total.complex_spec, atol=1e-12)


def test_direct_dac_trial_bundle_applies_shared_allpass_to_combined_sub_branch() -> None:
    bundle = _bundle()
    trial = _build_direct_dac_trial_bundle(
        bundle,
        fc_hz=80.0,
        main_hpf_order=4,
        sub_lpf_order=4,
        sub_hpf_hz=20.0,
        sub_hpf_order=2,
        sub_allpass_freq_hz=76.0,
        sub_allpass_q=0.9,
    )
    expected_main = _apply_branch_filters(
        bundle.l_main,
        hpf_hz=80.0,
        hpf_order=4,
        label="expected_main",
    )
    expected_sub_l = _apply_branch_filters(
        bundle.l_sub,
        hpf_hz=20.0,
        hpf_order=2,
        lpf_hz=80.0,
        lpf_order=4,
        label="expected_l_sub",
    )
    expected_sub_r = _apply_branch_filters(
        bundle.r_sub,
        hpf_hz=20.0,
        hpf_order=2,
        lpf_hz=80.0,
        lpf_order=4,
        label="expected_r_sub",
    )
    combined_sub, _diag = build_combined_sub_transfer(
        expected_main,
        expected_sub_l,
        expected_sub_r,
        mode="average",
        label="expected_combined_sub",
    )
    expected_sub = _apply_allpass_to_transfer(
        combined_sub,
        freq_hz=76.0,
        q=0.9,
        label="expected_combined_sub_ap",
    )
    np.testing.assert_allclose(
        trial.l_total.complex_spec,
        expected_main.complex_spec + expected_sub.complex_spec,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        trial.r_total.complex_spec,
        expected_main.complex_spec + expected_sub.complex_spec,
        atol=1e-12,
    )


def test_filtered_branch_cache_is_bundle_scoped(monkeypatch) -> None:
    bundle_a = _bundle()
    bundle_b = _bundle()
    seen_calls: list[str] = []

    def _fake_apply_branch_filters(transfer, **_kwargs):
        seen_calls.append(str(transfer.label))
        return transfer

    monkeypatch.setattr(bass_integration, "_apply_branch_filters", _fake_apply_branch_filters)

    params = {
        "fc": 80.0,
        "xo_order": 4,
        "sub_hp_hz": 20.0,
        "sub_hp_order": 2,
        "sub_lpf": 80.0,
        "lpf_order": 4,
    }

    first = _get_filtered_branches(bundle_a, **params)
    second = _get_filtered_branches(bundle_a, **params)
    third = _get_filtered_branches(bundle_b, **params)

    assert len(seen_calls) == 8
    assert first == second
    assert hasattr(bundle_a, "_camillafir_filtered_branch_cache")
    assert hasattr(bundle_b, "_camillafir_filtered_branch_cache")
    assert object.__getattribute__(bundle_a, "_camillafir_filtered_branch_cache") is not object.__getattribute__(
        bundle_b, "_camillafir_filtered_branch_cache"
    )
    assert third == (bundle_b.l_main, bundle_b.r_main, bundle_b.l_sub, bundle_b.r_sub)


def test_recommend_direct_dac_allpass_handles_nominal_case() -> None:
    bundle = _bundle(main_delay_s=0.0, sub_delay_s=0.0)

    result = recommend_direct_dac_allpass(
        bundle,
        fc_hz=80.0,
        profile="safe",
        main_hpf_order=4,
        sub_lpf_order=4,
        sub_hpf_hz=20.0,
        sub_hpf_order=2,
    )

    assert "baseline" in result and isinstance(result["baseline"], dict)
    assert "optimized" in result and isinstance(result["optimized"], dict)
    assert math.isfinite(float(result["improvement_score"]))
    if result["enabled"]:
        assert float(result["improvement_score"]) > 0.0
        assert math.isfinite(float(result["freq_hz"]))
        assert math.isfinite(float(result["q"]))
    else:
        assert result["freq_hz"] == 0.0


def test_recommend_direct_dac_allpass_improves_synthetic_misaligned_case() -> None:
    bundle = _bundle(main_delay_s=0.0, sub_delay_s=-0.0030)

    result = recommend_direct_dac_allpass(
        bundle,
        fc_hz=80.0,
        profile="safe",
        main_hpf_order=4,
        sub_lpf_order=4,
        sub_hpf_hz=20.0,
        sub_hpf_order=2,
    )

    assert result["enabled"] is True
    baseline = dict(result["baseline"])
    optimized = dict(result["optimized"])
    assert (
        float(optimized.get("cancellation_risk", float("inf"))) < float(baseline.get("cancellation_risk", float("inf")))
        or float(optimized.get("overlap_ripple_db", float("inf"))) < float(baseline.get("overlap_ripple_db", float("inf")))
        or float(optimized.get("xo_gd_mismatch_ms", float("inf"))) < float(baseline.get("xo_gd_mismatch_ms", float("inf")))
    )


def test_recommend_direct_dac_allpass_returns_structured_metrics() -> None:
    bundle = _bundle(main_delay_s=0.0, sub_delay_s=-0.0030)

    result = recommend_direct_dac_allpass(
        bundle,
        fc_hz=80.0,
        profile="safe",
        main_hpf_order=4,
        sub_lpf_order=4,
        sub_hpf_hz=20.0,
        sub_hpf_order=2,
    )

    assert isinstance(result["enabled"], bool)
    assert math.isfinite(float(result["improvement_score"]))
    assert isinstance(result["baseline"], dict)
    assert isinstance(result["optimized"], dict)
    if result["enabled"]:
        assert math.isfinite(float(result["freq_hz"]))
        assert math.isfinite(float(result["q"]))


def test_build_combined_sub_transfer_aligned_sum_uses_wide_default_lag(monkeypatch) -> None:
    freqs = np.geomspace(10.0, 320.0, 1024)
    ref = _transfer(freqs, delay_s=0.0, gain=1.0, label="ref")
    delayed = _transfer(freqs, delay_s=0.035, gain=1.0, label="delayed")
    seen: dict[str, float] = {}

    def _fake_xcorr(_main_spec, _sub_spec, _freqs_hz, _fs, max_lag_ms):
        seen["max_lag_ms"] = float(max_lag_ms)
        return 35.0, 0.95

    monkeypatch.setattr("decaycore.dsp.bass_integration._xcorr_lag_from_spectra", _fake_xcorr)

    combined, diagnostics = build_combined_sub_transfer(
        ref,
        ref,
        delayed,
        mode="aligned_sum",
        label="combined",
    )

    assert seen["max_lag_ms"] == 50.0
    assert diagnostics["whether_alignment_applied"] is True
    assert diagnostics["alignment_offset_ms"] == 35.0
    np.testing.assert_allclose(combined.complex_spec, 2.0 * ref.complex_spec, atol=1e-12)


def test_recommend_direct_dac_alignment_respects_documented_delay_bounds(monkeypatch) -> None:
    bundle = _bundle()

    def _fake_metrics(_bundle, _fc_hz, _profile, **kwargs):
        delay_ms = float(kwargs.get("sub_delay_ms", 0.0) or 0.0)
        gain_db = float(kwargs.get("sub_gain_trim_db", 0.0) or 0.0)
        polarity = bool(kwargs.get("sub_polarity_invert", False))
        objective = 10.0 - abs(delay_ms - 35.0) * 0.1 - abs(gain_db + 12.0) * 0.2 - (0.5 if polarity else 0.0)
        return {"objective": objective}

    monkeypatch.setattr("decaycore.dsp.bass_integration.compute_final_bass_integration_metrics", _fake_metrics)

    result = recommend_direct_dac_alignment(
        bundle,
        fc_hz=80.0,
        profile="safe",
        main_hpf_order=4,
        sub_lpf_order=4,
        sub_hpf_hz=20.0,
        sub_hpf_order=2,
    )

    assert result["applied"] is True
    assert float(result["sub_delay_ms"]) == 15.0
    assert float(result["sub_gain_trim_db"]) == -12.0


def test_direct_dac_prepare_allpass_failure_keeps_stable_result_shape(monkeypatch, caplog) -> None:
    from decaycore.dsp.bass_integration import _recommend_prepare_dac as prepare_dac

    bundle = _bundle()

    def _fake_metrics(_bundle, _fc_hz, _profile, **_kwargs):
        return {
            "objective": 0.0,
            "bass_cancellation_risk": 0.1,
            "bass_overlap_ripple": 2.0,
            "bass_sub_dominance": 1.0,
            "bass_null_severity": 0.0,
            "bass_xo_gd_rms_mismatch_ms": 1.0,
        }

    def _raise_allpass(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(bass_integration, "compute_final_bass_integration_metrics", _fake_metrics)
    monkeypatch.setattr(prepare_dac, "recommend_direct_dac_allpass", _raise_allpass)

    with caplog.at_level(logging.DEBUG, logger="DecayCore.dsp"):
        result = prepare_dac._recommend_direct_dac_prepare_builtin_core(
            bundle,
            profile="safe",
            main_hpf_order=4,
            sub_lpf_order=4,
            sub_hpf_hz=20.0,
            sub_hpf_order=2,
            allpass_auto_enable=True,
        )

    assert result["backend"] == "builtin"
    assert result["allpass_enabled"] is False
    assert result["allpass_freq_hz"] == 0.0
    assert result["allpass_q"] == 0.707
    assert result["allpass_reason"] == "Allpass post-pass failed."
    assert "Direct-DAC allpass post-pass failed" in caplog.text


def test_direct_dac_optuna_vectorized_seed_path_skips_legacy_seed_scans(monkeypatch) -> None:
    pytest.importorskip("optuna")
    from decaycore.dsp.bass_integration import _recommend_prepare_optuna as prepare_optuna
    from decaycore.dsp.bass_integration.direct_dac import DirectDacBassIntegrationResult

    bundle = _bundle()
    seen = {"fast_seed_calls": 0}

    def _fake_metrics(_bundle, _fc_hz, _profile, **_kwargs):
        return {
            "objective": 0.5,
            "bass_cancellation_risk": 0.1,
            "bass_overlap_ripple": 2.0,
            "bass_sub_dominance": 1.0,
            "bass_null_severity": 0.0,
            "bass_xo_gd_rms_mismatch_ms": 1.0,
            "bass_direct_dac_candidate_score": 0.0,
            "bass_direct_dac_reject_reasons": [],
            "bass_direct_dac_worst_channel": "balanced",
        }

    def _fake_fast_seed(*_args, **_kwargs):
        seen["fast_seed_calls"] += 1
        return DirectDacBassIntegrationResult(
            enabled=True,
            main_hpf_hz=95.0,
            sub_hpf_hz=20.0,
            sub_lpf_hz=118.75,
            sub_overlap_hz=23.75,
            sub_delay_ms=3.0,
            sub_gain_db=-1.5,
            sub_polarity_invert=False,
            phase_error_deg=0.0,
            gd_mismatch_ms=0.0,
            cancellation_risk="low",
            cancellation_score=0.0,
            magnitude_ripple_db=0.0,
            score=0.0,
            rejected=False,
            reject_reason="",
        )

    def _raise_legacy(*_args, **_kwargs):
        raise AssertionError("legacy seed scan should not run")

    monkeypatch.setattr(bass_integration, "compute_final_bass_integration_metrics", _fake_metrics)
    monkeypatch.setattr(bass_integration, "_main_guard_band_drop_db", lambda _main, _fc_hz: 0.0)
    monkeypatch.setattr(prepare_optuna, "run_direct_dac_bass_integration", _fake_fast_seed)
    monkeypatch.setattr(prepare_optuna, "recommend_direct_dac_alignment", _raise_legacy)
    monkeypatch.setattr(prepare_optuna, "recommend_direct_dac_crossover", _raise_legacy)

    result = recommend_direct_dac_prepare_optuna(
        bundle,
        profile="safe",
        main_hpf_order=4,
        sub_lpf_order=4,
        sub_hpf_hz=20.0,
        sub_hpf_order=2,
        trials=1,
        startup_trials=1,
        local_trials=1,
    )

    assert seen["fast_seed_calls"] == 2
    assert result["backend"] == "optuna"
    assert "applied" in result
    assert "recommended_hz" in result
    assert "candidate_score" in result


def test_direct_dac_optuna_seed_failure_logs_and_preserves_result_shape(monkeypatch, caplog) -> None:
    pytest.importorskip("optuna")
    from decaycore.dsp.bass_integration import _recommend_prepare_optuna as prepare_optuna

    bundle = _bundle()

    def _fake_metrics(_bundle, _fc_hz, _profile, **_kwargs):
        return {
            "objective": 0.0,
            "bass_cancellation_risk": 0.1,
            "bass_overlap_ripple": 2.0,
            "bass_sub_dominance": 1.0,
            "bass_null_severity": 0.0,
            "bass_xo_gd_rms_mismatch_ms": 1.0,
            "bass_direct_dac_candidate_score": 0.0,
            "bass_direct_dac_reject_reasons": [],
            "bass_direct_dac_worst_channel": "balanced",
        }

    def _raise_alignment(*_args, **_kwargs):
        raise RuntimeError("seed failed")

    monkeypatch.setattr(bass_integration, "compute_final_bass_integration_metrics", _fake_metrics)
    monkeypatch.setattr(bass_integration, "_main_guard_band_drop_db", lambda _main, _fc_hz: 0.0)
    monkeypatch.setattr(prepare_optuna, "_direct_dac_fast_seed_candidates", _raise_alignment)
    monkeypatch.setattr(prepare_optuna, "recommend_direct_dac_alignment", _raise_alignment)
    monkeypatch.setattr(
        prepare_optuna,
        "recommend_direct_dac_crossover",
        lambda *_args, **_kwargs: {"recommended_hz": 80.0, "recommended_sub_lpf_hz": 80.0},
    )

    with caplog.at_level(logging.DEBUG, logger="DecayCore.dsp"):
        result = recommend_direct_dac_prepare_optuna(
            bundle,
            profile="safe",
            main_hpf_order=4,
            sub_lpf_order=4,
            sub_hpf_hz=20.0,
            sub_hpf_order=2,
            trials=1,
            startup_trials=1,
            local_trials=1,
        )

    assert result["backend"] == "optuna"
    assert "applied" in result
    assert "recommended_hz" in result
    assert "candidate_score" in result
    assert result["reject_reasons"] == []
    assert "Direct-DAC Optuna vectorized seed build failed" in caplog.text
    assert "Direct-DAC Optuna alignment seed failed" in caplog.text


def test_direct_dac_prepare_builtin_core_remains_package_importable() -> None:
    from decaycore.dsp.bass_integration import _recommend_direct_dac_prepare_builtin_core

    assert callable(_recommend_direct_dac_prepare_builtin_core)
