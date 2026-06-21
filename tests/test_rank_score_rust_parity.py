import importlib
import importlib.metadata

import pytest

from decaycore.auto_mode import rank_score


RANK_SCORE_INPUTS = {
    "avg_score": 82.375,
    "phase_benefit_bonus": 1.125,
    "boost_penalty": 0.75,
    "event_penalty": 1.5,
    "lr_delta_penalty": 0.375,
    "dsp_penalty": 2.25,
    "bass_prering_penalty": 0.5625,
    "exc_penalty": 0.625,
    "bass_integration_penalty": 1.875,
    "bass_feasibility_penalty": 0.5,
    "bass_preference_bonus": 0.875,
    "mode_penalty": 1.25,
    "decay_penalty": 0.9375,
    "residual_peak_penalty": 3.125,
    "correction_sharpness_penalty": 0.3125,
    "dip_fill_risk_penalty": 0.4375,
    "channel_overfit_penalty": 0.5625,
    "target_tracking_penalty": 2.625,
    "voice_clarity_penalty": 0.34375,
    "phase_risk_penalty": 0.8125,
    "phase_limit_penalty": 0.6875,
    "thd_boost_penalty": 0.1875,
    "stereo_coherence_penalty": 0.9375,
    "phantom_center_stability_penalty": 0.5625,
    "policy_divergence_penalty": 0.3125,
    "asymmetry_budget_overflow_penalty": 0.25,
    "worst_channel_relief_bonus": 0.4375,
    "shared_preference_bias": 0.875,
    "shared_preference_penalty": 0.625,
    "rt60_policy_penalty": 0.375,
    "harmonic_local_boost_penalty": 0.28125,
    "gain": 1.125,
    "bias": -0.75,
    "score_min": 5.0,
    "score_max": 97.0,
    "context": "preset_objective_score",
}

SCALAR_COMPONENT_KEYS = (
    "rank_score",
    "rank_score_raw",
    "avg_score",
    "phase_benefit_bonus",
    "boost_penalty",
    "event_penalty",
    "lr_delta_penalty",
    "dsp_penalty",
    "bass_prering_penalty",
    "exc_penalty",
    "bass_integration_penalty",
    "bass_feasibility_penalty",
    "bass_preference_bonus",
    "mode_penalty",
    "decay_penalty",
    "residual_peak_penalty",
    "correction_sharpness_penalty",
    "dip_fill_risk_penalty",
    "channel_overfit_penalty",
    "target_tracking_penalty",
    "voice_clarity_penalty",
    "phase_risk_penalty",
    "phase_limit_penalty",
    "thd_boost_penalty",
    "stereo_coherence_penalty",
    "phantom_center_stability_penalty",
    "policy_divergence_penalty",
    "asymmetry_budget_overflow_penalty",
    "worst_channel_relief_bonus",
    "shared_preference_bias",
    "shared_preference_penalty",
    "rt60_policy_penalty",
    "harmonic_local_boost_penalty",
    "rank_score_gain",
    "rank_score_bias",
)

BREAKDOWN_SCALAR_KEYS = (
    "total",
    "raw",
    "magnitude_error_component",
    "residual_peak_component",
    "anti_overfit_component",
    "phase_gd_component",
    "tdc_decay_component",
    "boost_cut_component",
    "excess_correction_component",
    "bass_component",
    "stereo_policy_component",
    "rt60_policy_component",
    "harmonic_local_component",
)


@pytest.mark.requires_rust
def test_rank_score_python_fallback_matches_rust_extension(monkeypatch):
    rust_scoring = pytest.importorskip("decaycore_scoring")

    monkeypatch.setattr(rank_score, "_RUST_SCORING_AVAILABLE", False)
    monkeypatch.setattr(rank_score, "_rust_scoring", None)
    python_components = rank_score.compute_rank_score_components(**RANK_SCORE_INPUTS)

    rust_components = dict(rust_scoring.compute_rank_score_components(**RANK_SCORE_INPUTS))

    for key in SCALAR_COMPONENT_KEYS:
        assert rust_components[key] == pytest.approx(python_components[key], abs=1e-9), key

    assert rust_components["context"] == python_components["context"]

    python_breakdown = rank_score.build_rank_score_breakdown(python_components)
    rust_breakdown = rank_score.build_rank_score_breakdown(rust_components)

    for key in BREAKDOWN_SCALAR_KEYS:
        assert rust_breakdown[key] == pytest.approx(python_breakdown[key], abs=1e-9), key

    for key in rank_score.rank_score_component_metadata():
        if key in python_components and key in rust_breakdown["components"]:
            assert rust_breakdown["components"][key]["value"] == pytest.approx(
                python_breakdown["components"][key]["value"],
                abs=1e-9,
            ), key


@pytest.mark.requires_rust
def test_rank_score_dispatch_uses_installed_rust_extension():
    rust_scoring = pytest.importorskip("decaycore_scoring")

    importlib.reload(rank_score)
    assert rank_score._RUST_SCORING_AVAILABLE is True

    dispatched = rank_score.compute_rank_score_components(**RANK_SCORE_INPUTS)
    direct_rust = dict(rust_scoring.compute_rank_score_components(**RANK_SCORE_INPUTS))

    assert dispatched["rank_score"] == pytest.approx(direct_rust["rank_score"], abs=1e-9)
    assert dispatched["rank_score_raw"] == pytest.approx(direct_rust["rank_score_raw"], abs=1e-9)


@pytest.mark.requires_rust
def test_rank_score_extension_version_metadata_matches_distribution():
    rust_scoring = pytest.importorskip("decaycore_scoring")

    assert rust_scoring.__version__ == "0.2.0"
    assert importlib.metadata.version("decaycore-scoring") == rust_scoring.__version__
