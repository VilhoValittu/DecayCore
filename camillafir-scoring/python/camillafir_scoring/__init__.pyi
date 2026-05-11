from typing import Any

__version__: str

def compute_rank_score_components(
    avg_score: float,
    *,
    phase_benefit_bonus: float = ...,
    boost_penalty: float = ...,
    event_penalty: float = ...,
    lr_delta_penalty: float = ...,
    dsp_penalty: float = ...,
    exc_penalty: float = ...,
    bass_integration_penalty: float = ...,
    bass_feasibility_penalty: float = ...,
    bass_preference_bonus: float = ...,
    mode_penalty: float = ...,
    decay_penalty: float = ...,
    residual_peak_penalty: float = ...,
    correction_sharpness_penalty: float = ...,
    dip_fill_risk_penalty: float = ...,
    channel_overfit_penalty: float = ...,
    target_tracking_penalty: float = ...,
    voice_clarity_penalty: float = ...,
    phase_risk_penalty: float = ...,
    phase_limit_penalty: float = ...,
    thd_boost_penalty: float = ...,
    stereo_coherence_penalty: float = ...,
    phantom_center_stability_penalty: float = ...,
    policy_divergence_penalty: float = ...,
    asymmetry_budget_overflow_penalty: float = ...,
    worst_channel_relief_bonus: float = ...,
    shared_preference_bias: float = ...,
    rt60_policy_penalty: float = ...,
    harmonic_local_boost_penalty: float = ...,
    shared_preference_penalty: float = ...,
    gain: float = ...,
    bias: float = ...,
    score_min: float = ...,
    score_max: float = ...,
    context: str | None = ...,
) -> dict[str, Any]: ...

def calibrated_auto_quality(
    rank_score_0_100: float,
    metrics: dict[str, Any] | None = ...,
) -> float: ...
