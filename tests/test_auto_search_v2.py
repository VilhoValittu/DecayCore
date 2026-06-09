import logging
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest

from decaycore.auto_mode import refine_eval
from decaycore.auto_mode import _refine_search_core as refine_search_core
from decaycore.auto_mode.cache_signature import (
    _auto_cache_get_entry,
    _auto_cache_put_best,
    _auto_cache_put_last_used_best,
    _auto_compat_version,
    _auto_signature,
    _auto_cache_stats_snapshot,
    _auto_measurement_signature,
    clear_auto_mode_runtime_caches,
    get_or_build_synth_target,
)
from decaycore.auto_mode.optuna_backend_constraints import (
    _auto_optuna_constraint_thresholds,
    _auto_optuna_constraint_vector_from_metrics,
)
from decaycore.auto_mode.search_state import _AutoModeSearchState
from decaycore.auto_mode._refine_types import _SearchPhase1State
from decaycore.auto_mode.search_v2.input_model import build_auto_search_input
from decaycore.auto_mode.search_v2.context import build_execution_context
from decaycore.auto_mode.search_v2.plan import AutoSearchPlan, AutoSearchPlanDecision
from decaycore.auto_mode.search_v2.planner import determine_auto_search_plan
from decaycore.auto_mode.search_v2.runner import run_auto_search_v2
from decaycore.auto_mode.search_v2.seeds import _apply_seed_payload
from decaycore.auto_mode.search_v2.signature import (
    compute_auto_search_signature,
    compute_auto_search_signature_object,
)
from decaycore.auto_mode.optuna_backend_constraints import _auto_optuna_effective_scope
from decaycore.auto_mode.optuna_backend_storage import _auto_optuna_storage_filename


def _measurements():
    return {
        "f_l": np.array([20.0, 40.0, 80.0, 160.0]),
        "m_l": np.array([1.0, -2.0, 0.5, 0.0]),
    }


def _base_data():
    return {
        "filter_type": "mixed",
        "auto_goal": "balanced",
        "auto_target_mode": "selected",
        "hc_mode": "Harman6",
        "mag_c_min": 20.0,
        "mag_c_max": 250.0,
        "mixed_freq": 180.0,
        "tdc_strength": 50.0,
        "auto_mode_cache_enabled": True,
    }


def test_auto_measurement_signature_full_hash_detects_unsampled_middle_change():
    n = 2401
    freqs = np.linspace(20.0, 22000.0, n, dtype=float)
    mags = np.linspace(-3.0, 3.0, n, dtype=float)
    sampled = set(np.linspace(0, n - 1, 1200).astype(int).tolist())
    change_idx = next(idx for idx in range(1, n - 1) if idx not in sampled)

    changed = mags.copy()
    changed[change_idx] += 0.25

    sig1 = _auto_measurement_signature({"f_l": freqs, "m_l": mags})
    sig2 = _auto_measurement_signature({"f_l": freqs, "m_l": changed})

    assert sig2 != sig1


def test_auto_measurement_signature_detects_sub_rounding_change():
    freqs = np.array([20.0, 40.0, 80.0, 160.0], dtype=float)
    mags = np.array([1.0, -2.0, 0.5, 0.0], dtype=float)
    changed = mags.copy()
    changed[2] += 0.00001

    sig1 = _auto_measurement_signature({"f_l": freqs, "m_l": mags})
    sig2 = _auto_measurement_signature({"f_l": freqs, "m_l": changed})

    assert sig2 != sig1


def test_auto_measurement_signature_detects_phase_change():
    freqs = np.array([20.0, 40.0, 80.0, 160.0])
    mags = np.zeros_like(freqs)
    phase_a = np.array([0.0, 0.1, 0.2, 0.3])
    phase_b = phase_a.copy()
    phase_b[2] += 0.001

    sig1 = _auto_measurement_signature({"f_l": freqs, "m_l": mags, "p_l": phase_a})
    sig2 = _auto_measurement_signature({"f_l": freqs, "m_l": mags, "p_l": phase_b})

    assert sig2 != sig1


def test_auto_search_signature_object_reports_decision_inputs():
    ctx = {"fs_v": 48000, "taps_v": 65536, "xos": [], "hpf": None}
    signature = compute_auto_search_signature_object(build_auto_search_input(_base_data(), _measurements(), ctx))

    assert signature.signature
    assert signature.measurement_identity
    assert signature.payload["measurement_metadata_identity"]
    assert signature.frequency_grid_identity
    assert "measurement identity" in signature.diff_reason
    assert "measurement metadata identity" in signature.diff_reason
    assert "DSP/scorer policy versions" in signature.diff_reason


def test_optuna_storage_filename_includes_filter_and_measurement():
    name = _auto_optuna_storage_filename(
        compat_version="am20",
        filter_key="asym",
        measurement_identity="abcdef1234567890",
        journal_kind="filter",
    )
    assert "asymmetric" in name
    assert "abcdef1234567890"[:16] in name
    assert "am20" in name


def test_target_optuna_storage_filename_includes_filter_and_measurement():
    name = _auto_optuna_storage_filename(
        compat_version="am20",
        filter_key="asym",
        measurement_identity="abcdef1234567890",
        journal_kind="target",
    )
    assert "target" in name
    assert "asymmetric" in name
    assert "abcdef1234567890"[:16] in name


def test_auto_search_v2_mixed_optuna_scope_and_storage_stay_mixed():
    scope = _auto_optuna_effective_scope(
        {"filter_type": "Mixed", "auto_mode_optuna_constraints": False},
        "phase1",
        phase_kind="phase1",
    )
    name = _auto_optuna_storage_filename(
        compat_version="am20",
        filter_key="mixed",
        measurement_identity="abcdef1234567890",
        journal_kind="filter",
    )

    assert scope == "phase1-filter-mixed"
    assert "mixed" in name
    assert "asymmetric" not in name


def test_auto_search_v2_execution_context_preserves_mixed_filter_key():
    context = build_execution_context(
        base_data={"filter_type": "Mixed", "auto_mode_cache_enabled": False},
        measurements=_measurements(),
        fs_v=48000,
        taps_v=65536,
        xos=[],
        hpf=None,
        hc_f=None,
        hc_m=None,
        pin_obj=None,
        status_cb=None,
        n_trials=1,
    )

    assert context.filter_key == "mixed"
    assert context.search_base_data["_optuna_filter_key"] == "mixed"
    assert context.cache_base_data["_optuna_filter_key"] == "mixed"
    assert context.search_base_data["_optuna_journal_kind"] == "filter"


def test_auto_search_v2_signature_stable_after_winner_applied():
    measurements = _measurements()
    base = _base_data()
    ctx = {"fs_v": 48000, "taps_v": 65536, "xos": [], "hpf": None}
    sig1 = compute_auto_search_signature(build_auto_search_input(base, measurements, ctx))

    rerun = {
        **base,
        "_auto_exc_freq_hz": 42.0,
        "_auto_low_bass_cut_hz": 18.0,
        "_auto_target_seed_preset": {"mixed_freq": 211.0},
        "_auto_mode_meta": {"best_preset": {"mixed_freq": 211.0}},
        "_auto_target_curve_meta": {"selected_hc_mode": "Harman6"},
        "auto_applied_preset": {"mixed_freq": 211.0},
        "best_preset": {"mixed_freq": 211.0},
        "winner": {"rank_score": 1.0},
    }
    sig2 = compute_auto_search_signature(build_auto_search_input(rerun, measurements, ctx))

    assert sig2 == sig1


def test_auto_search_v2_auto_applied_values_do_not_change_signature():
    measurements = _measurements()
    base = _base_data()
    ctx = {"fs_v": 48000, "taps_v": 65536, "xos": [], "hpf": None}
    sig1 = compute_auto_search_signature(build_auto_search_input(base, measurements, ctx))
    applied = {
        "mag_c_min": 21.3,
        "low_bass_cut_hz": 41.4,
        "phase_limit": 110.7,
        "tdc_strength": 37.6,
    }
    rerun = {
        **base,
        **applied,
        "auto_applied_preset": dict(applied),
        "_auto_search_identity_base_data": build_auto_search_input(base, measurements, ctx).base_data(),
    }
    sig2 = compute_auto_search_signature(build_auto_search_input(rerun, measurements, ctx))
    user_changed = {**rerun, "phase_limit": 125.0}
    sig3 = compute_auto_search_signature(build_auto_search_input(user_changed, measurements, ctx))

    assert sig2 == sig1
    assert sig3 != sig1


def test_auto_search_v2_seed_payload_restores_measurement_mag_c_min():
    search_base_data = {
        "mag_c_min": 31.0,
        "_auto_mag_c_min_hz": 16.0,
        "exc_freq": 42.0,
    }
    cache_base_data = {"exc_freq": 55.0}
    context = SimpleNamespace(prior_seed_preset={})

    applied = _apply_seed_payload(
        search_base_data=search_base_data,
        cache_base_data=cache_base_data,
        seed_preset={"mag_c_min": 44.0, "exc_freq": 60.0},
        seed_metrics={"rank_score": 1.0},
        context=context,
        success_log="seed payload test",
    )

    assert applied is True
    assert float(search_base_data["mag_c_min"]) == pytest.approx(16.0, abs=1e-9)
    assert float(search_base_data["_auto_mag_c_min_hz"]) == pytest.approx(16.0, abs=1e-9)
    assert float(search_base_data["exc_freq"]) == pytest.approx(55.0, abs=1e-9)
    assert context.prior_seed_preset == {"mag_c_min": 44.0, "exc_freq": 60.0}


def test_auto_search_v2_real_user_input_changes_signature():
    measurements = _measurements()
    ctx = {"fs_v": 48000, "taps_v": 65536, "xos": [], "hpf": None}
    sig1 = compute_auto_search_signature(build_auto_search_input(_base_data(), measurements, ctx))
    changed = {**_base_data(), "mag_c_max": 180.0}
    sig2 = compute_auto_search_signature(build_auto_search_input(changed, measurements, ctx))

    assert sig2 != sig1


def test_auto_search_v2_signature_normalizes_filter_alias():
    measurements = _measurements()
    ctx = {"fs_v": 48000, "taps_v": 65536, "xos": [], "hpf": None}
    sig1 = compute_auto_search_signature(
        build_auto_search_input({**_base_data(), "filter_type": "Asymmetric"}, measurements, ctx)
    )
    sig2 = compute_auto_search_signature(
        build_auto_search_input({**_base_data(), "filter_type": "asym"}, measurements, ctx)
    )
    sig3 = compute_auto_search_signature(
        build_auto_search_input({**_base_data(), "filter_type": "Mixed"}, measurements, ctx)
    )

    assert sig2 == sig1
    assert sig3 != sig1


def test_auto_search_v2_exact_cache_plans_phase4_only(monkeypatch, tmp_path):
    cache_path = tmp_path / "auto_cache.json"
    from decaycore.auto_mode import cache_signature

    monkeypatch.setattr(cache_signature, "_auto_cache_path", lambda *args, **kwargs: str(cache_path))
    clear_auto_mode_runtime_caches()
    measurements = _measurements()
    base = _base_data()
    ctx = {"fs_v": 48000, "taps_v": 65536, "xos": [], "hpf": None}
    search_input = build_auto_search_input(base, measurements, ctx)
    sig = compute_auto_search_signature(search_input)
    _auto_cache_put_best(
        sig,
        best_preset={"preset_id": "cached", "mixed_freq": 180.0},
        best_metrics={"rank_score": 1.0},
        measurement_sig=search_input.measurement_identity,
        goal="balanced",
        filter_key="mixed",
        compat_version="am15",
    )

    decision = determine_auto_search_plan(
        search_input,
        base,
        options={
            "signature": sig,
            "filter_key": "mixed",
            "goal": "balanced",
            "compat_version": "am15",
        },
    )

    assert decision.plan == AutoSearchPlan.CACHE_MICRO_REFINE
    assert decision.skipped_phases == ("target_search", "phase1", "phase2", "phase3")
    assert decision.enabled_phases == ("phase4",)

    clear_auto_mode_runtime_caches()


def test_auto_search_v2_second_run_with_auto_applied_values_hits_phase4(monkeypatch, tmp_path):
    cache_path = tmp_path / "auto_cache.json"
    from decaycore.auto_mode import cache_signature

    monkeypatch.setattr(cache_signature, "_auto_cache_path", lambda *args, **kwargs: str(cache_path))
    clear_auto_mode_runtime_caches()
    measurements = _measurements()
    base = _base_data()
    ctx = {"fs_v": 48000, "taps_v": 65536, "xos": [], "hpf": None}
    first_input = build_auto_search_input(base, measurements, ctx)
    sig = compute_auto_search_signature(first_input)
    applied = {
        "mag_c_min": 21.3,
        "low_bass_cut_hz": 41.4,
        "phase_limit": 110.7,
        "tdc_strength": 37.6,
    }
    _auto_cache_put_best(
        sig,
        best_preset={"preset_id": "cached", **applied},
        best_metrics={"rank_score": 1.0},
        measurement_sig=first_input.measurement_identity,
        goal="balanced",
        filter_key="mixed",
        compat_version="am15",
    )
    rerun = {
        **base,
        **applied,
        "auto_applied_preset": dict(applied),
        "_auto_search_identity_base_data": first_input.base_data(),
    }
    rerun_input = build_auto_search_input(rerun, measurements, ctx)

    decision = determine_auto_search_plan(
        rerun_input,
        rerun,
        options={
            "signature": compute_auto_search_signature(rerun_input),
            "filter_key": "mixed",
            "goal": "balanced",
            "compat_version": "am15",
        },
    )

    assert decision.signature == sig
    assert decision.plan == AutoSearchPlan.CACHE_MICRO_REFINE
    assert decision.enabled_phases == ("phase4",)
    assert decision.skipped_phases == ("target_search", "phase1", "phase2", "phase3")
    clear_auto_mode_runtime_caches()


def test_auto_search_v2_save_cache_uses_canonical_signature(monkeypatch, tmp_path):
    cache_path = tmp_path / "auto_cache.json"
    from decaycore.auto_mode import cache_signature, orchestrator_finalize

    monkeypatch.setattr(cache_signature, "_auto_cache_path", lambda *args, **kwargs: str(cache_path))
    clear_auto_mode_runtime_caches()
    measurements = _measurements()
    base = _base_data()
    drift = {**base, "phase_limit": 111.0, "tdc_strength": 38.0}
    ctx = {"fs_v": 48000, "taps_v": 65536, "xos": [], "hpf": None}
    canonical_sig = compute_auto_search_signature(build_auto_search_input(base, measurements, ctx))
    drift_sig = _auto_signature(
        base_data=drift,
        measurements=measurements,
        fs_v=48000,
        taps_v=65536,
        xos=[],
        hpf=None,
        hc_mode=drift["hc_mode"],
        include_hc_mode=True,
    )

    orchestrator_finalize._save_cached_best(
        cache_base_data=drift,
        measurements=measurements,
        fs_v=48000,
        taps_v=65536,
        xos=[],
        hpf=None,
        cfg=SimpleNamespace(cache_enabled=True),
        goal="balanced",
        filter_key="mixed",
        compat_version="am15",
        best_preset={"preset_id": "winner", "phase_limit": 111.0},
        best_metrics={"rank_score": 1.0},
        best_hc_mode=drift["hc_mode"],
        canonical_signature=canonical_sig,
    )

    canonical_entry = _auto_cache_get_entry(canonical_sig, filter_key="mixed", compat_version="am15")
    drift_entry = _auto_cache_get_entry(drift_sig, filter_key="mixed", compat_version="am15")
    assert canonical_entry["first_run_complete"] is True
    assert set(canonical_entry["completed_stages"]) >= {"target_search", "phase1", "phase2", "phase3"}
    assert drift_entry["first_run_complete"] is False
    clear_auto_mode_runtime_caches()


def test_auto_search_v2_old_exact_cache_seeds_without_phase4(monkeypatch, tmp_path):
    cache_path = tmp_path / "auto_cache.json"
    from decaycore.auto_mode import cache_signature

    monkeypatch.setattr(cache_signature, "_auto_cache_path", lambda *args, **kwargs: str(cache_path))
    clear_auto_mode_runtime_caches()
    measurements = _measurements()
    base = _base_data()
    ctx = {"fs_v": 48000, "taps_v": 65536, "xos": [], "hpf": None}
    search_input = build_auto_search_input(base, measurements, ctx)
    sig = compute_auto_search_signature(search_input)
    _auto_cache_put_best(
        sig,
        best_preset={"preset_id": "old-cache", "mixed_freq": 180.0},
        best_metrics={"rank_score": 1.0},
        measurement_sig=search_input.measurement_identity,
        goal="balanced",
        filter_key="mixed",
        compat_version="am15",
    )
    cache = cache_signature._auto_cache_load(compat_version="am15")
    entry = cache["by_filter"]["mixed"]["items"][sig]
    entry.pop("first_run_complete", None)
    entry.pop("completed_stages", None)
    cache_signature._auto_cache_save(cache, compat_version="am15")
    clear_auto_mode_runtime_caches()

    decision = determine_auto_search_plan(
        search_input,
        base,
        options={
            "signature": sig,
            "filter_key": "mixed",
            "goal": "balanced",
            "compat_version": "am15",
        },
    )

    assert decision.plan == AutoSearchPlan.FIRST_RUN_FULL_SEARCH
    assert decision.enabled_phases == ("target_search", "phase1", "phase2", "phase3")
    assert decision.skipped_phases == ("phase4",)
    assert decision.seed_source == "old_exact_cache"
    assert dict(decision.seed_preset or {}).get("preset_id") == "old-cache"
    clear_auto_mode_runtime_caches()


def test_auto_search_v2_partial_exact_cache_seeds_without_phase4(monkeypatch, tmp_path):
    cache_path = tmp_path / "auto_cache.json"
    from decaycore.auto_mode import cache_signature

    monkeypatch.setattr(cache_signature, "_auto_cache_path", lambda *args, **kwargs: str(cache_path))
    clear_auto_mode_runtime_caches()
    measurements = _measurements()
    base = _base_data()
    ctx = {"fs_v": 48000, "taps_v": 65536, "xos": [], "hpf": None}
    search_input = build_auto_search_input(base, measurements, ctx)
    sig = compute_auto_search_signature(search_input)
    _auto_cache_put_best(
        sig,
        best_preset={"preset_id": "partial-cache", "mixed_freq": 180.0},
        best_metrics={"rank_score": 1.0},
        measurement_sig=search_input.measurement_identity,
        goal="balanced",
        filter_key="mixed",
        compat_version="am15",
    )
    cache = cache_signature._auto_cache_load(compat_version="am15")
    cache["by_filter"]["mixed"]["items"][sig]["completed_stages"] = ["target_search", "phase1", "phase2"]
    cache_signature._auto_cache_save(cache, compat_version="am15")
    clear_auto_mode_runtime_caches()

    decision = determine_auto_search_plan(
        search_input,
        base,
        options={
            "signature": sig,
            "filter_key": "mixed",
            "goal": "balanced",
            "compat_version": "am15",
        },
    )

    assert decision.plan == AutoSearchPlan.FIRST_RUN_FULL_SEARCH
    assert decision.seed_source == "old_exact_cache"
    assert "exact cache skipped: cache completed stages incomplete" in decision.fallback_reasons
    clear_auto_mode_runtime_caches()


def test_auto_search_v2_target_seed_allows_exact_cache_phase4(monkeypatch, tmp_path):
    cache_path = tmp_path / "auto_cache.json"
    from decaycore.auto_mode import cache_signature

    monkeypatch.setattr(cache_signature, "_auto_cache_path", lambda *args, **kwargs: str(cache_path))
    clear_auto_mode_runtime_caches()
    measurements = _measurements()
    base = {
        **_base_data(),
        "auto_target_mode": "selected",
        "_auto_target_seed_preset": {"preset_id": "target-preselect", "mixed_freq": 180.0},
        "_auto_target_seed_metrics": {"rank_score": 2.0},
    }
    ctx = {"fs_v": 48000, "taps_v": 65536, "xos": [], "hpf": None}
    search_input = build_auto_search_input(base, measurements, ctx)
    sig = compute_auto_search_signature(search_input)
    _auto_cache_put_best(
        sig,
        best_preset={"preset_id": "cached", "mixed_freq": 180.0},
        best_metrics={"rank_score": 1.0},
        measurement_sig=search_input.measurement_identity,
        goal="balanced",
        filter_key="mixed",
        compat_version="am15",
    )

    decision = determine_auto_search_plan(
        search_input,
        base,
        options={
            "signature": sig,
            "filter_key": "mixed",
            "goal": "balanced",
            "compat_version": "am15",
        },
    )

    assert decision.plan == AutoSearchPlan.CACHE_MICRO_REFINE
    assert decision.skipped_phases == ("target_search", "phase1", "phase2", "phase3")
    assert decision.enabled_phases == ("phase4",)
    assert decision.seed_source == "exact_cache"

    clear_auto_mode_runtime_caches()


def test_auto_search_v2_cached_target_seed_allows_exact_cache_phase4(monkeypatch, tmp_path):
    cache_path = tmp_path / "auto_cache.json"
    from decaycore.auto_mode import cache_signature

    monkeypatch.setattr(cache_signature, "_auto_cache_path", lambda *args, **kwargs: str(cache_path))
    clear_auto_mode_runtime_caches()
    measurements = _measurements()
    base = {
        **_base_data(),
        "_auto_target_seed_preset": {"preset_id": "target-cache-seed", "mixed_freq": 181.0},
        "_auto_target_seed_metrics": {"rank_score": 2.0},
        "_auto_target_seed_source": "cache_signature_hit",
    }
    ctx = {"fs_v": 48000, "taps_v": 65536, "xos": [], "hpf": None}
    search_input = build_auto_search_input(base, measurements, ctx)
    sig = compute_auto_search_signature(search_input)
    _auto_cache_put_best(
        sig,
        best_preset={"preset_id": "cached-winner", "mixed_freq": 180.0},
        best_metrics={"rank_score": 1.0},
        measurement_sig=search_input.measurement_identity,
        goal="balanced",
        filter_key="mixed",
        compat_version="am15",
    )

    decision = determine_auto_search_plan(
        search_input,
        base,
        options={
            "signature": sig,
            "filter_key": "mixed",
            "goal": "balanced",
            "compat_version": "am15",
        },
    )

    assert decision.plan == AutoSearchPlan.CACHE_MICRO_REFINE
    assert decision.skipped_phases == ("target_search", "phase1", "phase2", "phase3")
    assert decision.enabled_phases == ("phase4",)
    assert decision.seed_source == "exact_cache"
    assert dict(decision.seed_preset or {}).get("preset_id") == "cached-winner"

    clear_auto_mode_runtime_caches()


def test_auto_search_v2_cached_target_does_not_run_phase4_without_exact_cache(monkeypatch, tmp_path):
    cache_path = tmp_path / "auto_cache.json"
    from decaycore.auto_mode import cache_signature

    monkeypatch.setattr(cache_signature, "_auto_cache_path", lambda *args, **kwargs: str(cache_path))
    clear_auto_mode_runtime_caches()
    base = {
        **_base_data(),
        "_auto_target_seed_preset": {"preset_id": "target-cache-seed", "mixed_freq": 181.0},
        "_auto_target_seed_metrics": {"rank_score": 2.0},
        "_auto_target_seed_source": "cache_measurement_hit",
    }
    ctx = {"fs_v": 48000, "taps_v": 65536, "xos": [], "hpf": None}
    search_input = build_auto_search_input(base, _measurements(), ctx)
    decision = determine_auto_search_plan(
        search_input,
        base,
        options={
            "signature": compute_auto_search_signature(search_input),
            "filter_key": "mixed",
            "goal": "balanced",
            "compat_version": "am15",
        },
    )

    assert decision.plan == AutoSearchPlan.FIRST_RUN_FULL_SEARCH
    assert decision.skipped_phases == ("phase4",)
    assert decision.enabled_phases == ("target_search", "phase1", "phase2", "phase3")
    assert decision.seed_source == "cache_measurement_target_seed"
    assert dict(decision.seed_preset or {}).get("preset_id") == "target-cache-seed"
    assert dict(decision.cache_record or {}).get("best_metrics") == {"rank_score": 2.0}

    clear_auto_mode_runtime_caches()


def test_auto_cache_save_failure_logs_debug_reason(monkeypatch, tmp_path, caplog):
    from decaycore.auto_mode import cache_signature

    monkeypatch.setattr(cache_signature, "_auto_cache_path", lambda *args, **kwargs: str(tmp_path))
    clear_auto_mode_runtime_caches()

    with caplog.at_level(logging.DEBUG, logger="DecayCore"):
        _auto_cache_put_best(
            "sig",
            best_preset={"preset_id": "candidate"},
            best_metrics={"rank_score": 1.0},
            measurement_sig="measurement",
            goal="balanced",
            filter_key="mixed",
            compat_version="am15",
        )

    stats = _auto_cache_stats_snapshot()
    assert stats["saves"] == 0
    assert stats["save_failures"] == 1
    assert stats["last_save_error_path"] == str(tmp_path)
    assert "IsADirectoryError" in stats["last_save_error"]
    assert "Automatic mode cache save failed" in caplog.text
    assert str(tmp_path) in caplog.text
    assert "IsADirectoryError" in caplog.text
    clear_auto_mode_runtime_caches()


def test_auto_search_v2_full_search_when_no_cache(monkeypatch, tmp_path):
    cache_path = tmp_path / "auto_cache.json"
    from decaycore.auto_mode import cache_signature

    monkeypatch.setattr(cache_signature, "_auto_cache_path", lambda *args, **kwargs: str(cache_path))
    clear_auto_mode_runtime_caches()
    base = {**_base_data(), "auto_target_mode": "off"}
    ctx = {"fs_v": 48000, "taps_v": 65536, "xos": [], "hpf": None}
    search_input = build_auto_search_input(base, _measurements(), ctx)
    decision = determine_auto_search_plan(
        search_input,
        base,
        options={
            "signature": compute_auto_search_signature(search_input),
            "filter_key": "mixed",
            "goal": "balanced",
            "compat_version": "am15",
        },
    )

    assert decision.plan == AutoSearchPlan.FIRST_RUN_FULL_SEARCH


def test_auto_search_v2_selected_target_plans_phase1_optimization(monkeypatch, tmp_path):
    cache_path = tmp_path / "auto_cache.json"
    from decaycore.auto_mode import cache_signature

    monkeypatch.setattr(cache_signature, "_auto_cache_path", lambda *args, **kwargs: str(cache_path))
    clear_auto_mode_runtime_caches()
    base = {**_base_data(), "auto_target_mode": "selected", "hc_mode": "Harman6"}
    ctx = {"fs_v": 48000, "taps_v": 65536, "xos": [], "hpf": None}
    search_input = build_auto_search_input(base, _measurements(), ctx)
    decision = determine_auto_search_plan(
        search_input,
        base,
        options={
            "signature": compute_auto_search_signature(search_input),
            "filter_key": "mixed",
            "goal": "balanced",
            "compat_version": "am15",
        },
    )

    assert decision.plan == AutoSearchPlan.MANUAL_PRESET_REFINE
    assert "phase1" in decision.enabled_phases
    assert "phase1" not in decision.skipped_phases
    clear_auto_mode_runtime_caches()


def test_auto_search_plan_contains_cache_decision_report_on_miss(monkeypatch, tmp_path):
    cache_path = tmp_path / "auto_cache.json"
    from decaycore.auto_mode import cache_signature

    monkeypatch.setattr(cache_signature, "_auto_cache_path", lambda *args, **kwargs: str(cache_path))
    clear_auto_mode_runtime_caches()
    base = {**_base_data(), "auto_target_mode": "off"}
    ctx = {"fs_v": 48000, "taps_v": 65536, "xos": [], "hpf": None}
    search_input = build_auto_search_input(base, _measurements(), ctx)
    decision = determine_auto_search_plan(
        search_input,
        base,
        options={
            "signature": compute_auto_search_signature(search_input),
            "filter_key": "mixed",
            "goal": "balanced",
            "compat_version": "am15",
        },
    )

    report = decision.cache_decision_report
    assert report["selected_plan"] == AutoSearchPlan.FULL_SEARCH.value
    assert report["signature"] == decision.signature
    assert report["exact_cache"]["status"] in {"miss", "invalid"}
    assert report["measurement_identity"] == search_input.measurement_identity
    clear_auto_mode_runtime_caches()


def test_synth_target_cache_thread_safe_single_shape():
    clear_auto_mode_runtime_caches()
    measurements = _measurements()

    def _synth_fn(f_l, m_l, f_r, m_r, **kwargs):
        return np.asarray(f_l, dtype=float), np.asarray(m_l, dtype=float) + float(kwargs["tilt_comp_frac"])

    def _run_once():
        return get_or_build_synth_target(
            measurements,
            tilt_comp_frac=0.3,
            bass_comp_frac=0.5,
            bass_comp_ref_db=8.0,
            hf_comp_frac=0.5,
            smooth_oct=1.0 / 3.0,
            synth_fn=_synth_fn,
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _idx: _run_once(), range(16)))

    first_f, first_m = results[0]
    assert all(np.array_equal(first_f, item[0]) for item in results)
    assert all(np.array_equal(first_m, item[1]) for item in results)
    clear_auto_mode_runtime_caches()


def test_synth_target_cache_keys_include_rt60_bands_and_forward_measurements():
    clear_auto_mode_runtime_caches()
    freqs = np.array([20.0, 40.0, 80.0, 160.0], dtype=float)
    mags = np.array([1.0, -2.0, 0.5, 0.0], dtype=float)
    calls = {"count": 0}

    def _synth_fn(f_l, m_l, f_r, m_r, **kwargs):
        calls["count"] += 1
        measurements = kwargs["measurements"]
        rt60 = float(measurements["measured_rt60_bands_l"][63.0])
        return np.asarray(f_l, dtype=float), np.asarray(m_l, dtype=float) + rt60

    base = {"f_l": freqs, "m_l": mags, "f_r": freqs, "m_r": mags}
    dry = {**base, "measured_rt60_bands_l": {63.0: 0.30, 100.0: 0.32}}
    live = {**base, "measured_rt60_bands_l": {63.0: 0.90, 100.0: 0.92}}

    first = get_or_build_synth_target(
        dry,
        tilt_comp_frac=0.3,
        bass_comp_frac=0.5,
        bass_comp_ref_db=8.0,
        hf_comp_frac=0.5,
        smooth_oct=1.0 / 3.0,
        synth_fn=_synth_fn,
    )
    second = get_or_build_synth_target(
        dry,
        tilt_comp_frac=0.3,
        bass_comp_frac=0.5,
        bass_comp_ref_db=8.0,
        hf_comp_frac=0.5,
        smooth_oct=1.0 / 3.0,
        synth_fn=_synth_fn,
    )
    changed = get_or_build_synth_target(
        live,
        tilt_comp_frac=0.3,
        bass_comp_frac=0.5,
        bass_comp_ref_db=8.0,
        hf_comp_frac=0.5,
        smooth_oct=1.0 / 3.0,
        synth_fn=_synth_fn,
    )

    assert calls["count"] == 2
    assert second is first
    assert first is not None and changed is not None
    assert not np.array_equal(first[1], changed[1])
    clear_auto_mode_runtime_caches()


def test_synth_target_cache_key_includes_algo_version():
    from decaycore.auto_mode import cache_synth_target

    key = cache_synth_target._synth_target_cache_key(
        "measurement-sig",
        tilt_comp_frac=0.3,
        bass_comp_frac=0.5,
        bass_comp_ref_db=8.0,
        hf_comp_frac=0.5,
        smooth_oct=1.0 / 3.0,
    )

    assert key[0] == cache_synth_target._SYNTH_TARGET_ALGO_V


def test_auto_search_v2_fallback_reasons_attached_on_cache_miss(monkeypatch, tmp_path):
    cache_path = tmp_path / "auto_cache.json"
    from decaycore.auto_mode import cache_signature
    from decaycore.auto_mode.search_v2 import runner as search_runner

    monkeypatch.setattr(cache_signature, "_auto_cache_path", lambda *args, **kwargs: str(cache_path))
    clear_auto_mode_runtime_caches()
    base = {**_base_data(), "auto_target_mode": "off"}
    captured = {}

    def _execute_probe(**kwargs):
        captured["decision"] = kwargs["decision"]
        return {"auto_mode_debug": {}}

    monkeypatch.setattr(search_runner, "execute_auto_search_plan", _execute_probe)

    result = run_auto_search_v2(
        base_data=base,
        measurements=_measurements(),
        fs_v=48000,
        taps_v=65536,
        xos=[],
        hpf=None,
        hc_f=None,
        hc_m=None,
        pin_obj=None,
        status_cb=None,
        n_trials=4,
    )

    assert captured["decision"].plan == AutoSearchPlan.FULL_SEARCH
    debug = result["auto_mode_debug"]
    assert any("exact cache skipped: missing cache record" == item for item in debug["fallback_reasons"])
    assert debug["auto_search_v2"]["fallback_reasons"] == debug["fallback_reasons"]
    clear_auto_mode_runtime_caches()


def test_auto_search_v2_last_best_measurement_mismatch_is_reported(monkeypatch, tmp_path):
    cache_path = tmp_path / "auto_cache.json"
    from decaycore.auto_mode import cache_signature

    monkeypatch.setattr(cache_signature, "_auto_cache_path", lambda *args, **kwargs: str(cache_path))
    clear_auto_mode_runtime_caches()
    base = {**_base_data(), "auto_target_mode": "off"}
    ctx = {"fs_v": 48000, "taps_v": 65536, "xos": [], "hpf": None}
    search_input = build_auto_search_input(base, _measurements(), ctx)
    _auto_cache_put_last_used_best(
        best_preset={"preset_id": "stale", "mixed_freq": 177.0},
        best_metrics={"rank_score": 1.0},
        measurement_sig="different-measurement",
        goal="balanced",
        filter_key="mixed",
        compat_version="am15",
    )

    decision = determine_auto_search_plan(
        search_input,
        base,
        options={
            "signature": compute_auto_search_signature(search_input),
            "filter_key": "mixed",
            "goal": "balanced",
            "compat_version": "am15",
        },
    )

    assert decision.plan == AutoSearchPlan.FIRST_RUN_FULL_SEARCH
    assert any("last-best skipped: cache measurement mismatch" == item for item in decision.fallback_reasons)
    clear_auto_mode_runtime_caches()


def test_auto_search_v2_last_best_seed_runs_full_phase_chain(monkeypatch, tmp_path):
    cache_path = tmp_path / "auto_cache.json"
    from decaycore.auto_mode import cache_signature

    monkeypatch.setattr(cache_signature, "_auto_cache_path", lambda *args, **kwargs: str(cache_path))
    clear_auto_mode_runtime_caches()
    base = {**_base_data(), "auto_target_mode": "off"}
    ctx = {"fs_v": 48000, "taps_v": 65536, "xos": [], "hpf": None}
    search_input = build_auto_search_input(base, _measurements(), ctx)
    _auto_cache_put_last_used_best(
        best_preset={"preset_id": "last-best", "mixed_freq": 177.0},
        best_metrics={"rank_score": 1.0},
        measurement_sig=search_input.measurement_identity,
        goal="balanced",
        filter_key="mixed",
        compat_version="am15",
    )

    decision = determine_auto_search_plan(
        search_input,
        base,
        options={
            "signature": compute_auto_search_signature(search_input),
            "filter_key": "mixed",
            "goal": "balanced",
            "compat_version": "am15",
        },
    )

    assert decision.plan == AutoSearchPlan.FIRST_RUN_FULL_SEARCH
    assert decision.skipped_phases == ("phase4",)
    assert decision.enabled_phases == ("target_search", "phase1", "phase2", "phase3")
    assert decision.seed_source == "last_best"
    assert dict(decision.seed_preset or {}).get("preset_id") == "last-best"
    clear_auto_mode_runtime_caches()


def test_auto_search_v2_asym_last_best_does_not_shortcut_mixed(monkeypatch, tmp_path):
    cache_path = tmp_path / "auto_cache.json"
    from decaycore.auto_mode import cache_signature

    monkeypatch.setattr(cache_signature, "_auto_cache_path", lambda *args, **kwargs: str(cache_path))
    clear_auto_mode_runtime_caches()
    base = {**_base_data(), "filter_type": "Mixed", "auto_target_mode": "off"}
    ctx = {"fs_v": 48000, "taps_v": 65536, "xos": [], "hpf": None}
    search_input = build_auto_search_input(base, _measurements(), ctx)
    _auto_cache_put_last_used_best(
        best_preset={"preset_id": "asym-last-best", "mixed_freq": 177.0},
        best_metrics={"rank_score": 1.0},
        measurement_sig=search_input.measurement_identity,
        goal="balanced",
        filter_key="asym",
        compat_version="am15",
    )

    decision = determine_auto_search_plan(
        search_input,
        base,
        options={
            "signature": compute_auto_search_signature(search_input),
            "filter_key": "mixed",
            "goal": "balanced",
            "compat_version": "am15",
        },
    )

    assert decision.plan == AutoSearchPlan.FIRST_RUN_FULL_SEARCH
    assert decision.seed_source is None
    assert dict(decision.seed_preset or {}) == {}
    clear_auto_mode_runtime_caches()


def test_auto_search_v2_cache_disabled_is_reported():
    base = {**_base_data(), "auto_target_mode": "off"}
    ctx = {"fs_v": 48000, "taps_v": 65536, "xos": [], "hpf": None}
    search_input = build_auto_search_input(base, _measurements(), ctx)
    decision = determine_auto_search_plan(
        search_input,
        base,
        options={
            "signature": compute_auto_search_signature(search_input),
            "filter_key": "mixed",
            "goal": "balanced",
            "compat_version": "am15",
            "cache_enabled": False,
        },
    )

    assert decision.plan == AutoSearchPlan.FULL_SEARCH
    assert decision.fallback_reasons == ("cache skipped: disabled",)


def test_auto_search_v2_cache_read_failure_is_reported(monkeypatch):
    from decaycore.auto_mode.search_v2 import cache as search_cache

    def _raise_cache(*args, **kwargs):
        raise RuntimeError("cache unavailable")

    monkeypatch.setattr(search_cache.auto_api, "_auto_cache_get_entry", _raise_cache)
    base = {**_base_data(), "auto_target_mode": "off"}
    ctx = {"fs_v": 48000, "taps_v": 65536, "xos": [], "hpf": None}
    search_input = build_auto_search_input(base, _measurements(), ctx)
    decision = determine_auto_search_plan(
        search_input,
        base,
        options={
            "signature": compute_auto_search_signature(search_input),
            "filter_key": "mixed",
            "goal": "balanced",
            "compat_version": "am15",
        },
    )

    assert decision.plan == AutoSearchPlan.FALLBACK_FULL_SEARCH
    assert any(
        "exact cache skipped: exact cache read failed: RuntimeError" == item
        for item in decision.fallback_reasons
    )


def test_auto_search_v2_exact_cache_plan_emits_phase4_only(monkeypatch, tmp_path):
    cache_path = tmp_path / "auto_cache.json"
    from decaycore.auto_mode import cache_signature
    from decaycore.auto_mode.search_v2 import runner as search_runner

    monkeypatch.setattr(cache_signature, "_auto_cache_path", lambda *args, **kwargs: str(cache_path))
    clear_auto_mode_runtime_caches()
    base = _base_data()
    compat_version = _auto_compat_version(base)
    measurements = _measurements()
    ctx = {"fs_v": 48000, "taps_v": 65536, "xos": [], "hpf": None}
    search_input = build_auto_search_input(base, measurements, ctx)
    sig = compute_auto_search_signature(search_input)
    _auto_cache_put_best(
        sig,
        best_preset={"preset_id": "cached", "mixed_freq": 180.0},
        best_metrics={"rank_score": 1.0},
        measurement_sig=search_input.measurement_identity,
        goal="balanced",
        filter_key="mixed",
        compat_version=compat_version,
    )

    seen = {}
    statuses = []

    def _execute_probe(**kwargs):
        decision = kwargs["decision"]
        seen["plan"] = decision.plan
        statuses.extend(decision.enabled_phases)
        return {"auto_mode_debug": {}}

    monkeypatch.setattr(search_runner, "execute_auto_search_plan", _execute_probe)

    run_auto_search_v2(
        base_data=base,
        measurements=measurements,
        fs_v=48000,
        taps_v=65536,
        xos=[],
        hpf=None,
        hc_f=None,
        hc_m=None,
        pin_obj=None,
        status_cb=statuses.append,
        n_trials=4,
    )

    assert seen["plan"] == AutoSearchPlan.CACHE_MICRO_REFINE
    assert statuses == ["phase4"]
    clear_auto_mode_runtime_caches()


def test_auto_search_v2_cached_target_plan_emits_first_run_stages(monkeypatch, tmp_path):
    cache_path = tmp_path / "auto_cache.json"
    from decaycore.auto_mode import cache_signature
    from decaycore.auto_mode.search_v2 import runner as search_runner

    monkeypatch.setattr(cache_signature, "_auto_cache_path", lambda *args, **kwargs: str(cache_path))
    clear_auto_mode_runtime_caches()
    base = {
        **_base_data(),
        "_auto_target_seed_preset": {"preset_id": "target-cache-seed", "mixed_freq": 181.0},
        "_auto_target_seed_metrics": {"rank_score": 2.0},
        "_auto_target_seed_source": "cache_signature_hit",
    }
    seen = {}
    statuses = []

    def _execute_probe(**kwargs):
        decision = kwargs["decision"]
        seen["plan"] = decision.plan
        seen["seed_source"] = decision.seed_source
        statuses.extend(list(decision.enabled_phases))
        return {"auto_mode_debug": {}}

    monkeypatch.setattr(search_runner, "execute_auto_search_plan", _execute_probe)

    run_auto_search_v2(
        base_data=base,
        measurements=_measurements(),
        fs_v=48000,
        taps_v=65536,
        xos=[],
        hpf=None,
        hc_f=None,
        hc_m=None,
        pin_obj=None,
        status_cb=statuses.append,
        n_trials=4,
    )

    assert seen["plan"] == AutoSearchPlan.FIRST_RUN_FULL_SEARCH
    assert seen["seed_source"] == "cache_signature_target_seed"
    assert statuses == ["target_search", "phase1", "phase2", "phase3"]
    clear_auto_mode_runtime_caches()


def test_auto_search_v2_executor_cache_plan_runs_micro_refine_only(monkeypatch):
    from decaycore.auto_mode.search_v2 import executor

    calls = []
    messages = []
    context = SimpleNamespace(search_base_data={})
    decision = AutoSearchPlanDecision(
        plan=AutoSearchPlan.CACHE_MICRO_REFINE,
        reason="test",
        signature="sig",
        seed_preset={"preset_id": "cached"},
        cache_record={"winner_metrics": {"rank_score": 1.0}},
        skipped_phases=("phase1", "phase2"),
        enabled_phases=("micro_refine",),
        fallback_reasons=(),
    )

    context.status_cb = messages.append
    context.cache_base_data = {}
    context.measurements = {}
    context.fs_v = 48000
    context.taps_v = 65536
    context.xos = []
    context.hpf = None
    context.hc_f = None
    context.hc_m = None
    context.pin_obj = None
    context.cfg = SimpleNamespace()
    context.goal = "balanced"
    context.filter_key = "mixed"
    context.compat_version = "am15"
    context.optimizer_backend = "builtin"
    context.optuna_mod = None
    context.seed = 1
    context.optuna_search_sig = "sig"
    context.cache_ready_preset = {}
    context.materialize_preset_result = lambda *args, **kwargs: None
    context.runtime = SimpleNamespace()
    context.winner_target_name = "Harman8"
    context.prior_seed_preset = {}
    context.use_optuna_trials = False
    context.candidates = []
    context.status_prefix = "DecayCore automatic mode [Harman8]"
    context.search_state = SimpleNamespace()

    monkeypatch.setattr(executor, "build_execution_context", lambda **kwargs: context)
    monkeypatch.setattr(
        "decaycore.auto_mode.orchestrator_refine.run_exact_cache_micro_refine",
        lambda **kwargs: calls.append("micro") or {"best_preset": {}, "best_metrics": {}},
    )
    monkeypatch.setattr(
        executor,
        "finalize_from_cache_refine",
        lambda ctx, result: calls.append("finalize_cache") or {"auto_mode_debug": {}},
    )
    monkeypatch.setattr(
        executor,
        "run_refine_stages",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("phase search must not run")),
    )

    result = executor.execute_auto_search_plan(
        decision=decision,
        base_data={},
        measurements={},
        fs_v=48000,
        taps_v=65536,
        xos=[],
        hpf=None,
        hc_f=None,
        hc_m=None,
        pin_obj=None,
        status_cb=None,
        n_trials=4,
    )

    assert result == {"auto_mode_debug": {}}
    assert calls == ["micro", "finalize_cache"]
    assert messages == ["DecayCore automatic mode: phase 3 skipped"]


def test_auto_search_v2_executor_legacy_reuse_valid_result_runs_finalize_only(monkeypatch):
    from decaycore.auto_mode.search_v2 import executor

    calls = []
    context = SimpleNamespace(search_base_data={}, cache_base_data={})
    decision = AutoSearchPlanDecision(
        plan=AutoSearchPlan.REUSE_VALID_RESULT,
        reason="exact canonical cache hit; running Phase 4 only",
        signature="sig",
        seed_preset={"preset_id": "cached"},
        cache_record={
            "winner_preset": {"preset_id": "cached"},
            "winner_metrics": {"rank_score": 1.0},
        },
        skipped_phases=("phase1", "phase2", "phase3"),
        enabled_phases=("phase4",),
        fallback_reasons=(),
    )

    monkeypatch.setattr(executor, "build_execution_context", lambda **kwargs: context)
    monkeypatch.setattr(
        executor,
        "run_micro_refine_from_seed",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("phase3 must not run")),
    )
    monkeypatch.setattr(
        executor,
        "run_refine_stages",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("phase search must not run")),
    )
    monkeypatch.setattr(
        executor,
        "finalize_from_cache_refine",
        lambda ctx, result: calls.append(dict(result)) or {"auto_mode_debug": {}},
    )

    result = executor.execute_auto_search_plan(
        decision=decision,
        base_data={},
        measurements={},
        fs_v=48000,
        taps_v=65536,
        xos=[],
        hpf=None,
        hc_f=None,
        hc_m=None,
        pin_obj=None,
        status_cb=None,
        n_trials=4,
    )

    assert result == {"auto_mode_debug": {}}
    assert calls[0]["best_preset"]["preset_id"] == "cached"
    assert calls[0]["stop_reason"] == "reuse_valid_result"


def test_auto_search_v2_executor_cache_failure_falls_back_to_full_search(monkeypatch):
    from decaycore.auto_mode.search_v2 import executor

    calls = []
    context = SimpleNamespace(search_base_data={})
    decision = AutoSearchPlanDecision(
        plan=AutoSearchPlan.LAST_BEST_MICRO_REFINE,
        reason="test",
        signature="sig",
        seed_preset={"preset_id": "last-best"},
        cache_record={"winner_metrics": {"rank_score": 1.0}},
        skipped_phases=("phase1", "phase2"),
        enabled_phases=("micro_refine",),
        fallback_reasons=(),
    )

    monkeypatch.setattr(executor, "build_execution_context", lambda **kwargs: context)
    monkeypatch.setattr(executor, "run_micro_refine_from_seed", lambda ctx, dec: calls.append("micro") or None)
    monkeypatch.setattr(
        executor,
        "run_refine_stages",
        lambda ctx, *, skip_phase1=False: calls.append(("refine", skip_phase1)) or {"phase1_ok": 1},
    )
    monkeypatch.setattr(
        executor,
        "finalize_from_refine_stats",
        lambda ctx, stats: calls.append("finalize_search") or {"auto_mode_debug": {}},
    )

    result = executor.execute_auto_search_plan(
        decision=decision,
        base_data={},
        measurements={},
        fs_v=48000,
        taps_v=65536,
        xos=[],
        hpf=None,
        hc_f=None,
        hc_m=None,
        pin_obj=None,
        status_cb=None,
        n_trials=4,
    )

    assert result == {"auto_mode_debug": {}}
    assert calls == ["micro", ("refine", False), "finalize_search"]
    assert context.search_base_data["_auto_search_fallback_reasons"] == [
        "last_best_micro_refine unavailable: falling back to full search"
    ]


def test_auto_search_v2_executor_passes_last_best_seed_to_full_search(monkeypatch):
    from decaycore.auto_mode.search_v2 import executor

    captured = {}
    context = SimpleNamespace(search_base_data={})
    decision = AutoSearchPlanDecision(
        plan=AutoSearchPlan.PRESELECTED_TARGET_REFINE,
        reason="last-best seed; full search",
        signature="sig",
        seed_preset={"preset_id": "last-best", "mixed_freq": 177.0},
        cache_record={"winner_metrics": {"rank_score": 1.0}},
        skipped_phases=(),
        enabled_phases=("phase1", "phase2", "phase3", "phase4"),
        fallback_reasons=(),
        seed_source="last_best",
    )

    def _build_context(**kwargs):
        captured["base_data"] = dict(kwargs["base_data"])
        return context

    monkeypatch.setattr(executor, "build_execution_context", _build_context)
    monkeypatch.setattr(
        executor,
        "run_refine_stages",
        lambda ctx, *, skip_phase1=False: {"phase1_ok": 1},
    )
    monkeypatch.setattr(
        executor,
        "finalize_from_refine_stats",
        lambda ctx, stats: {"auto_mode_debug": {}},
    )

    result = executor.execute_auto_search_plan(
        decision=decision,
        base_data={"filter_type": "Mixed"},
        measurements={},
        fs_v=48000,
        taps_v=65536,
        xos=[],
        hpf=None,
        hc_f=None,
        hc_m=None,
        pin_obj=None,
        status_cb=None,
        n_trials=4,
    )

    assert result == {"auto_mode_debug": {}}
    assert captured["base_data"]["_auto_target_seed_preset"]["preset_id"] == "last-best"
    assert captured["base_data"]["_auto_target_seed_source"] == "last_best"
    assert captured["base_data"]["mixed_freq"] == 177.0


def test_auto_search_v2_executor_cache_finalize_none_falls_back_to_full_search(monkeypatch):
    from decaycore.auto_mode.search_v2 import executor

    calls = []
    context = SimpleNamespace(search_base_data={})
    decision = AutoSearchPlanDecision(
        plan=AutoSearchPlan.CACHE_MICRO_REFINE,
        reason="test",
        signature="sig",
        seed_preset={"preset_id": "cached"},
        cache_record={"winner_metrics": {"rank_score": 1.0}},
        skipped_phases=("phase1", "phase2"),
        enabled_phases=("micro_refine",),
        fallback_reasons=(),
    )

    monkeypatch.setattr(executor, "build_execution_context", lambda **kwargs: context)
    monkeypatch.setattr(
        executor,
        "run_micro_refine_from_seed",
        lambda ctx, dec: calls.append("micro") or {"best_preset": {"preset_id": "cached"}},
    )
    monkeypatch.setattr(
        executor,
        "finalize_from_cache_refine",
        lambda ctx, result: calls.append("finalize_cache") or None,
    )
    monkeypatch.setattr(
        executor,
        "run_refine_stages",
        lambda ctx, *, skip_phase1=False: calls.append(("refine", skip_phase1)) or {"phase1_ok": 1},
    )
    monkeypatch.setattr(
        executor,
        "finalize_from_refine_stats",
        lambda ctx, stats: calls.append("finalize_search") or {"auto_mode_debug": {}},
    )

    result = executor.execute_auto_search_plan(
        decision=decision,
        base_data={},
        measurements={},
        fs_v=48000,
        taps_v=65536,
        xos=[],
        hpf=None,
        hc_f=None,
        hc_m=None,
        pin_obj=None,
        status_cb=None,
        n_trials=4,
    )

    assert result == {"auto_mode_debug": {}}
    assert calls == ["micro", "finalize_cache", ("refine", False), "finalize_search"]
    assert context.search_base_data["_auto_search_fallback_reasons"] == [
        "cache micro-refine finalize returned no result; falling back to full search"
    ]


def test_auto_search_v2_executor_asym_cache_finalize_none_falls_back_to_full_search(monkeypatch):
    from decaycore.auto_mode.search_v2 import executor

    calls = []
    context = SimpleNamespace(search_base_data={"filter_type": "Asymmetric"})
    decision = AutoSearchPlanDecision(
        plan=AutoSearchPlan.CACHE_MICRO_REFINE,
        reason="exact canonical cache hit",
        signature="sig",
        seed_preset={"preset_id": "cached-asym", "phase_limit": 320.0},
        cache_record={"winner_metrics": {"rank_score": 1.0}},
        skipped_phases=("phase1", "phase2"),
        enabled_phases=("micro_refine",),
        fallback_reasons=(),
    )

    monkeypatch.setattr(executor, "build_execution_context", lambda **kwargs: context)
    monkeypatch.setattr(
        executor,
        "run_micro_refine_from_seed",
        lambda ctx, dec: calls.append("micro") or {"best_preset": {"preset_id": "cached-asym"}},
    )
    monkeypatch.setattr(
        executor,
        "finalize_from_cache_refine",
        lambda ctx, result: calls.append("finalize_cache") or None,
    )
    monkeypatch.setattr(
        executor,
        "run_refine_stages",
        lambda ctx, *, skip_phase1=False: calls.append(("refine", skip_phase1)) or {"phase1_ok": 1},
    )
    monkeypatch.setattr(
        executor,
        "finalize_from_refine_stats",
        lambda ctx, stats: calls.append("finalize_search") or {"best_preset": {"preset_id": "full-search"}},
    )

    result = executor.execute_auto_search_plan(
        decision=decision,
        base_data={"filter_type": "Asymmetric"},
        measurements={},
        fs_v=48000,
        taps_v=65536,
        xos=[],
        hpf=None,
        hc_f=None,
        hc_m=None,
        pin_obj=None,
        status_cb=None,
        n_trials=4,
    )

    assert result == {"best_preset": {"preset_id": "full-search"}}
    assert calls == ["micro", "finalize_cache", ("refine", False), "finalize_search"]
    assert context.search_base_data["filter_type"] == "Asymmetric"
    assert context.search_base_data["_auto_search_fallback_reasons"] == [
        "cache micro-refine finalize returned no result; falling back to full search"
    ]


def test_auto_search_v2_cache_finalize_skips_repeated_winner_polish(monkeypatch):
    from decaycore.auto_mode import orchestrator_finalize

    def _raise_polish(**kwargs):
        raise AssertionError("cache fast finalize should not replay winner polish")

    monkeypatch.setattr(orchestrator_finalize, "apply_phase_limit_winner_polish", _raise_polish)
    monkeypatch.setattr(orchestrator_finalize, "apply_mag_c_min_winner_polish", _raise_polish)
    monkeypatch.setattr(orchestrator_finalize, "apply_low_bass_cut_winner_polish", _raise_polish)
    monkeypatch.setattr(orchestrator_finalize, "apply_hpf_winner_polish", _raise_polish)
    monkeypatch.setattr(orchestrator_finalize, "apply_excess_phase_strength_winner_polish", _raise_polish)
    monkeypatch.setattr(orchestrator_finalize, "apply_residual_peak_winner_polish", _raise_polish)
    monkeypatch.setattr(orchestrator_finalize, "apply_tdc_strength_winner_polish", _raise_polish)
    monkeypatch.setattr(orchestrator_finalize, "apply_stereo_policy_refine", _raise_polish)

    def _materialize(preset, **kwargs):
        metrics = {
            "rank_score": 82.9,
            "rank_score_official": 82.9,
            "avg_score": 77.3,
            "rank_score_components": {"rank_score": 82.9},
        }
        return SimpleNamespace(), metrics, dict(preset or {})

    result = orchestrator_finalize.finalize_search_result(
        search_base_data={"filter_type": "Asymmetric"},
        cache_base_data={"filter_type": "Asymmetric", "hc_mode": "Harman10"},
        measurements={},
        fs_v=48000,
        taps_v=65536,
        xos=[],
        hpf=None,
        hc_f=None,
        hc_m=None,
        pin_obj=None,
        cfg=SimpleNamespace(cache_enabled=False),
        goal="balanced",
        rank_basis="rank_score",
        filter_key="asym",
        compat_version="test",
        optimizer_backend="builtin",
        status_cb=None,
        optuna_mod=None,
        optuna_search_sig="sig",
        seed=123,
        search_state=None,
        winner_target_name="Harman10",
        phase1_ok=0,
        phase2_ok=0,
        phase1_tried=0,
        phase2_tried=0,
        phase1_plateau_hit=False,
        phase2_plateau_hit=False,
        phase1_optuna_tel={},
        phase2_local_optuna_tels=[],
        phase3_micro_optuna_tel={},
        phase2_rollup_tel={},
        _cache_ready_preset=lambda preset, best_metrics=None: dict(preset or {}),
        _materialize_preset_result=_materialize,
        _maybe_apply_residual_tiebreak=lambda **kwargs: (
            kwargs["best_preset"],
            kwargs["best_metrics"],
            False,
        ),
        cache_refine_result={
            "cache_target_name": "Harman10",
            "seed_source": "exact_cache",
            "best_preset": {"preset_id": "cached-asym", "phase_limit": 100.0},
            "best_metrics": {"rank_score": 82.8, "avg_score": 77.2},
            "executed_micro_trials_total": 60,
        },
        runtime=SimpleNamespace(cache_winner_polish_enabled=False),
    )

    assert dict(result.get("best_preset", {}) or {}).get("preset_id") == "cached-asym"
    assert result.get("phase_limit_winner_polish", {}).get("reason") == "cache_fast_finalize_skips_winner_polish"
    assert result.get("trials_total") == 60


def test_auto_search_v2_executor_manual_refine_keeps_phase1(monkeypatch):
    from decaycore.auto_mode.search_v2 import executor

    calls = []
    context = SimpleNamespace(search_base_data={})
    decision = AutoSearchPlanDecision(
        plan=AutoSearchPlan.MANUAL_PRESET_REFINE,
        reason="test",
        signature="sig",
        seed_preset=None,
        cache_record=None,
        skipped_phases=(),
        enabled_phases=("phase1", "phase2", "micro_refine"),
        fallback_reasons=(),
    )

    monkeypatch.setattr(executor, "build_execution_context", lambda **kwargs: context)
    monkeypatch.setattr(
        executor,
        "run_refine_stages",
        lambda ctx, *, skip_phase1=False: calls.append(("refine", skip_phase1)) or {"phase1_ok": 0},
    )
    monkeypatch.setattr(
        executor,
        "finalize_from_refine_stats",
        lambda ctx, stats: calls.append("finalize_search") or {"auto_mode_debug": {}},
    )

    executor.execute_auto_search_plan(
        decision=decision,
        base_data={},
        measurements={},
        fs_v=48000,
        taps_v=65536,
        xos=[],
        hpf=None,
        hc_f=None,
        hc_m=None,
        pin_obj=None,
        status_cb=None,
        n_trials=4,
    )

    assert calls == [("refine", False), "finalize_search"]


def test_preselected_target_refine_plan_does_not_claim_target_preselection(monkeypatch, tmp_path):
    cache_path = tmp_path / "auto_cache.json"
    from decaycore.auto_mode import cache_signature

    monkeypatch.setattr(cache_signature, "_auto_cache_path", lambda *args, **kwargs: str(cache_path))
    clear_auto_mode_runtime_caches()
    base = {**_base_data(), "auto_target_mode": "auto"}
    ctx = {"fs_v": 48000, "taps_v": 65536, "xos": [], "hpf": None}
    search_input = build_auto_search_input(base, _measurements(), ctx)
    decision = determine_auto_search_plan(
        search_input,
        base,
        options={
            "signature": compute_auto_search_signature(search_input),
            "filter_key": "mixed",
            "goal": "balanced",
            "compat_version": "am15",
        },
    )

    assert decision.plan == AutoSearchPlan.FIRST_RUN_FULL_SEARCH
    assert "target_search" in decision.enabled_phases
    assert "target_preselect" not in decision.enabled_phases
    assert "target_preselect" not in decision.skipped_phases
    clear_auto_mode_runtime_caches()


def test_run_candidate_phase_builtin_evaluates_seed_presets(monkeypatch):
    evaluated = []

    def _eval(ctx, *, preset, use_refine_tiebreak, focus_lo_hz, focus_hi_hz):
        evaluated.append(dict(preset))
        return {
            "ok": True,
            "metrics": {"rank_score": float(len(evaluated)), "avg_score": 1.0},
            "trial_preset": dict(preset),
        }

    monkeypatch.setattr(refine_eval, "evaluate_search_candidate", _eval)
    ctx = SimpleNamespace(
        optimizer_backend="builtin",
        runtime=SimpleNamespace(auto_optuna_module_ready=lambda mod: False),
        optuna_mod=None,
        search_base_data={},
        search_state=_AutoModeSearchState(),
        cfg=SimpleNamespace(),
        goal="balanced",
        status_cb=None,
        status_prefix="auto",
        winner_target_name=None,
    )

    refine_eval.run_candidate_phase(
        ctx,
        [{"preset_id": "generated", "mixed_freq": 180.0}],
        phase_label="phase 1/2",
        seed_presets=[{"preset_id": "seed", "mixed_freq": 120.0}],
    )

    assert [item["preset_id"] for item in evaluated] == ["seed", "generated"]


def test_run_candidate_phase_optuna_receives_seed_presets():
    captured = {}

    def _capture_optuna_loop(**kwargs):
        captured["seed_presets"] = list(kwargs["seed_presets"])
        return {}

    runtime = SimpleNamespace(
        auto_optuna_module_ready=lambda mod: True,
        auto_optuna_effective_scope=lambda base, scope, phase_kind=None: scope,
        auto_optuna_study_name=lambda study_sig, scope: "study",
        auto_run_optuna_eval_loop=_capture_optuna_loop,
        auto_optuna_needs_zero_feasible_rescue=lambda **kwargs: False,
    )
    ctx = SimpleNamespace(
        optimizer_backend="optuna",
        runtime=runtime,
        optuna_mod=object(),
        cfg=SimpleNamespace(),
        seed=1,
        search_base_data={},
        optuna_search_sig="sig",
        search_state=_AutoModeSearchState(),
    )

    refine_eval.run_candidate_phase(
        ctx,
        [],
        phase_label="phase 1/2",
        n_total_override=1,
        seed_presets=[{"preset_id": "seed"}],
        optuna_builder=lambda trial: {},
    )

    assert captured["seed_presets"] == [{"preset_id": "seed"}]


def test_refine_optuna_dynamic_constraint_uses_gate_metric() -> None:
    metrics = {
        "mode_ripple_db": 2.5,
        "focus_ripple_db": 4.3,
        "target_tracking_rms_20_200_db": 4.1,
        "max_net_boost_db": 0.0,
    }
    search_state = SimpleNamespace(best_metrics=dict(metrics))

    base_data = refine_search_core._build_refine_optuna_base_data({}, search_state)

    assert base_data is not None
    assert base_data["auto_mode_optuna_constraints_max_mode_ripple_db"] == pytest.approx(4.73)
    assert base_data["_auto_mode_refine_constraint_gate_metric_db"] == pytest.approx(4.3)

    thr = _auto_optuna_constraint_thresholds(base_data, scope="phase3-micro-c1")
    violations = _auto_optuna_constraint_vector_from_metrics(
        metrics,
        max_mode_ripple_db=thr["max_mode_ripple_db"],
        max_events_severity=thr["max_events_severity"],
        max_net_boost_db=thr["max_net_boost_db"],
        use_events=False,
    )
    assert violations == (0.0, 0.0, 0.0)


def test_phase1_optuna_plateau_has_exploration_floor(monkeypatch) -> None:
    captured = {}

    monkeypatch.setattr(
        refine_search_core,
        "_build_auto_mode_candidates",
        lambda *args, **kwargs: [{"preset_id": "seed"}],
    )

    def _fake_run_candidate_phase(*args, **kwargs):
        captured["plateau_after_no_improve"] = kwargs["plateau_after_no_improve"]
        captured["plateau_min_trials"] = kwargs["plateau_min_trials"]
        return {"ok": 0, "tried": 0, "plateau_hit": False, "optuna_telemetry": {}}

    monkeypatch.setattr(refine_search_core, "run_candidate_phase", _fake_run_candidate_phase)

    ctx = SimpleNamespace(
        search_state=_AutoModeSearchState(),
        goal="balanced",
        filter_key="linear",
        status_cb=None,
    )
    cfg = SimpleNamespace(
        phase1_plateau_rounds=20,
        optuna_startup_phase1=32,
        local_refine_top_k=1,
    )
    runtime = SimpleNamespace(suggest_auto_mode_candidate_optuna=lambda base, trial: {})

    refine_search_core._run_search_refine_phase1_core(
        search_base_data={},
        candidates=[],
        prior_seed_preset=None,
        use_optuna_trials=True,
        cfg=cfg,
        filter_key="linear",
        seed=1,
        n_trials_eff=100,
        runtime=runtime,
        status_cb=None,
        ctx=ctx,
    )

    assert captured["plateau_after_no_improve"] == 20
    assert captured["plateau_min_trials"] == 50


def test_phase1_top_anchors_skip_hard_gated_candidates(monkeypatch) -> None:
    monkeypatch.setattr(
        refine_search_core,
        "_build_auto_mode_candidates",
        lambda *args, **kwargs: [{"preset_id": "seed"}],
    )

    def _fake_run_candidate_phase(ctx, *args, **kwargs):
        _ = args, kwargs
        ctx.search_state.scored.extend(
            [
                {
                    "preset": {"preset_id": "hard", "phase_limit": 220.0},
                    "metrics": {
                        "phase": "phase 1/2",
                        "rank_score": 90.0,
                        "worst_residual_peak_raw_db": 8.0,
                        "residual_peak_hard_gate_db": 6.0,
                    },
                },
                {
                    "preset": {"preset_id": "safe-a", "phase_limit": 260.0},
                    "metrics": {
                        "phase": "phase 1/2",
                        "rank_score": 82.0,
                        "avg_score": 81.0,
                        "residual_peak_hard_gate_db": 6.0,
                    },
                },
                {
                    "preset": {"preset_id": "safe-b", "phase_limit": 300.0},
                    "metrics": {
                        "phase": "phase 1/2",
                        "rank_score": 81.0,
                        "avg_score": 80.0,
                        "residual_peak_hard_gate_db": 6.0,
                    },
                },
                {
                    "preset": {"preset_id": "safe-c", "phase_limit": 340.0},
                    "metrics": {
                        "phase": "phase 1/2",
                        "rank_score": 80.0,
                        "avg_score": 79.0,
                        "residual_peak_hard_gate_db": 6.0,
                    },
                },
            ]
        )
        return {"ok": 4, "tried": 4, "plateau_hit": False, "optuna_telemetry": {}}

    monkeypatch.setattr(refine_search_core, "run_candidate_phase", _fake_run_candidate_phase)

    ctx = SimpleNamespace(
        search_state=_AutoModeSearchState(),
        goal="balanced",
        filter_key="linear",
        status_cb=None,
    )
    cfg = SimpleNamespace(
        phase1_plateau_rounds=20,
        optuna_startup_phase1=32,
        local_refine_top_k=3,
    )
    runtime = SimpleNamespace(
        suggest_auto_mode_candidate_optuna=lambda base, trial: {},
        auto_optuna_telemetry_text=lambda tel: "",
    )

    phase1 = refine_search_core._run_search_refine_phase1_core(
        search_base_data={},
        candidates=[],
        prior_seed_preset=None,
        use_optuna_trials=False,
        cfg=cfg,
        filter_key="linear",
        seed=1,
        n_trials_eff=100,
        runtime=runtime,
        status_cb=None,
        ctx=ctx,
    )

    assert [x["preset"]["preset_id"] for x in phase1.phase1_top] == [
        "safe-a",
        "safe-b",
        "safe-c",
    ]


def test_local_refine_optuna_budget_includes_seed_startup_and_model_trials(monkeypatch) -> None:
    captured = {}
    ctx = SimpleNamespace(goal="balanced")
    phase1 = _SearchPhase1State(
        ctx=ctx,
        phase1_plateau_hit=False,
        phase1_top=[
            {
                "preset": {"phase_limit": 130.0, "tdc_strength": 50.0},
                "metrics": {"rank_score": 70.0, "mode_ripple_db": 2.0},
            }
        ],
    )
    search_state = SimpleNamespace(
        best_metrics={"rank_score": 70.0, "mode_ripple_db": 2.0},
        best_preset={"phase_limit": 130.0},
    )
    cfg = SimpleNamespace(
        local_refine_enabled=True,
        local_refine_shrink=0.35,
        local_refine_trials_per_top=2,
        optuna_startup_local=4,
        seed_revalidate_rank_gap=3.0,
    )
    runtime = SimpleNamespace(
        auto_optuna_scope_with_context=lambda scope, **kwargs: scope,
        auto_optuna_telemetry_text=lambda tel: "",
        auto_optuna_fallback_summary_text=lambda tel: "",
        auto_optuna_fmt_value=lambda value, ndigits=3: (
            "n/a" if value is None else f"{float(value):.{ndigits}f}"
        ),
    )

    monkeypatch.setattr(
        refine_search_core,
        "_build_auto_mode_candidates_local",
        lambda *args, **kwargs: [{"preset_id": "local-seed"}],
    )

    def _fake_run_candidate_phase(*args, **kwargs):
        captured["n_total_override"] = kwargs["n_total_override"]
        captured["seed_presets"] = list(kwargs["seed_presets"])
        return {"ok": 0, "tried": 0, "plateau_hit": False, "optuna_telemetry": {}}

    monkeypatch.setattr(refine_search_core, "run_candidate_phase", _fake_run_candidate_phase)

    refine_search_core._run_search_refine_phase2_local_core(
        search_base_data={},
        cfg=cfg,
        goal="balanced",
        filter_key="linear",
        seed=1,
        winner_target_name="Harman12",
        use_optuna_trials=True,
        status_cb=None,
        search_state=search_state,
        runtime=runtime,
        phase1=phase1,
    )

    assert captured["seed_presets"] == [{"preset_id": "local-seed"}]
    assert captured["n_total_override"] == 8
