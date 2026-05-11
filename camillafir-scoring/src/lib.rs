// DecayCore scoring extension — Rust/PyO3 0.22
// Mirrors rank_score.py: compute_rank_score_components() and calibrated_auto_quality()

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

#[inline]
fn safe_f64(v: f64) -> f64 {
    if v.is_finite() { v } else { 0.0 }
}

#[inline]
fn clamp(v: f64, lo: f64, hi: f64) -> f64 {
    v.max(lo).min(hi)
}

// ---------------------------------------------------------------------------
// compute_rank_score_components
// ---------------------------------------------------------------------------
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (
    avg_score,
    phase_benefit_bonus = 0.0,
    boost_penalty = 0.0,
    event_penalty = 0.0,
    lr_delta_penalty = 0.0,
    dsp_penalty = 0.0,
    exc_penalty = 0.0,
    bass_integration_penalty = 0.0,
    bass_feasibility_penalty = 0.0,
    bass_preference_bonus = 0.0,
    mode_penalty = 0.0,
    decay_penalty = 0.0,
    residual_peak_penalty = 0.0,
    correction_sharpness_penalty = 0.0,
    dip_fill_risk_penalty = 0.0,
    channel_overfit_penalty = 0.0,
    target_tracking_penalty = 0.0,
    voice_clarity_penalty = 0.0,
    phase_risk_penalty = 0.0,
    phase_limit_penalty = 0.0,
    thd_boost_penalty = 0.0,
    stereo_coherence_penalty = 0.0,
    phantom_center_stability_penalty = 0.0,
    policy_divergence_penalty = 0.0,
    asymmetry_budget_overflow_penalty = 0.0,
    worst_channel_relief_bonus = 0.0,
    shared_preference_bias = 0.0,
    rt60_policy_penalty = 0.0,
    harmonic_local_boost_penalty = 0.0,
    shared_preference_penalty = 0.0,
    gain = 1.0,
    bias = 0.0,
    score_min = 0.0,
    score_max = 100.0,
    context = None
))]
fn compute_rank_score_components(
    py: Python<'_>,
    avg_score: f64,
    phase_benefit_bonus: f64,
    boost_penalty: f64,
    event_penalty: f64,
    lr_delta_penalty: f64,
    dsp_penalty: f64,
    exc_penalty: f64,
    bass_integration_penalty: f64,
    bass_feasibility_penalty: f64,
    bass_preference_bonus: f64,
    mode_penalty: f64,
    decay_penalty: f64,
    residual_peak_penalty: f64,
    correction_sharpness_penalty: f64,
    dip_fill_risk_penalty: f64,
    channel_overfit_penalty: f64,
    target_tracking_penalty: f64,
    voice_clarity_penalty: f64,
    phase_risk_penalty: f64,
    phase_limit_penalty: f64,
    thd_boost_penalty: f64,
    stereo_coherence_penalty: f64,
    phantom_center_stability_penalty: f64,
    policy_divergence_penalty: f64,
    asymmetry_budget_overflow_penalty: f64,
    worst_channel_relief_bonus: f64,
    shared_preference_bias: f64,
    rt60_policy_penalty: f64,
    harmonic_local_boost_penalty: f64,
    shared_preference_penalty: f64,
    gain: f64,
    bias: f64,
    score_min: f64,
    score_max: f64,
    context: Option<&str>,
) -> PyResult<PyObject> {
    let avg        = safe_f64(avg_score);
    let phase_bon  = safe_f64(phase_benefit_bonus).max(0.0);
    let boost      = safe_f64(boost_penalty).max(0.0);
    let event      = safe_f64(event_penalty).max(0.0);
    let lr_pen     = safe_f64(lr_delta_penalty).max(0.0);
    let dsp_pen    = safe_f64(dsp_penalty).max(0.0);
    let exc_pen    = safe_f64(exc_penalty).max(0.0);
    let bass_pen   = safe_f64(bass_integration_penalty).max(0.0);
    let bass_feas  = safe_f64(bass_feasibility_penalty).max(0.0);
    let bass_bon   = safe_f64(bass_preference_bonus).max(0.0);
    let mode_pen   = safe_f64(mode_penalty).max(0.0);
    let decay_pen  = safe_f64(decay_penalty).max(0.0);
    let rp_pen     = safe_f64(residual_peak_penalty).max(0.0);
    let sharp_pen  = safe_f64(correction_sharpness_penalty).max(0.0);
    let dip_pen    = safe_f64(dip_fill_risk_penalty).max(0.0);
    let ch_of_pen  = safe_f64(channel_overfit_penalty).max(0.0);
    let tt_pen     = safe_f64(target_tracking_penalty).max(0.0);
    let voice_pen  = safe_f64(voice_clarity_penalty).max(0.0);
    let phase_risk = safe_f64(phase_risk_penalty).max(0.0);
    let phase_pen  = safe_f64(phase_limit_penalty).max(0.0);
    let thd_pen    = safe_f64(thd_boost_penalty).max(0.0);
    let stereo_pen = safe_f64(stereo_coherence_penalty).max(0.0);
    let center_pen = safe_f64(phantom_center_stability_penalty).max(0.0);
    let policy_pen = safe_f64(policy_divergence_penalty).max(0.0);
    let asym_pen   = safe_f64(asymmetry_budget_overflow_penalty).max(0.0);
    let worst_bon  = safe_f64(worst_channel_relief_bonus).max(0.0);
    let shared_pen = safe_f64(shared_preference_bias).max(0.0)
                        .max(safe_f64(shared_preference_penalty).max(0.0));
    let rt60_pen   = safe_f64(rt60_policy_penalty).max(0.0);
    let harm_pen   = safe_f64(harmonic_local_boost_penalty).max(0.0);

    let g  = if gain.is_finite() { gain } else { 1.0 };
    let b  = if bias.is_finite() { bias } else { 0.0 };
    let lo = if score_min.is_finite() { score_min } else { 0.0 };
    let hi = if score_max.is_finite() { score_max } else { 100.0 };
    let (lo, hi) = if hi < lo { (hi, lo) } else { (lo, hi) };

    let rank_raw =
          avg
        + phase_bon
        - boost
        - event
        - lr_pen
        - dsp_pen
        - exc_pen
        - bass_pen
        - bass_feas
        + bass_bon
        - mode_pen
        - decay_pen
        - rp_pen
        - sharp_pen
        - dip_pen
        - ch_of_pen
        - tt_pen
        - voice_pen
        - phase_risk
        - phase_pen
        - thd_pen
        - stereo_pen
        - center_pen
        - policy_pen
        - asym_pen
        + worst_bon
        - shared_pen
        - rt60_pen
        - harm_pen;

    let rank_score = clamp(g * rank_raw + b, lo, hi);

    let official_ctx = "preset_objective_score";
    let score_kind   = context.unwrap_or(official_ctx).trim();
    let score_kind   = if score_kind.is_empty() { official_ctx } else { score_kind };
    let score_label  = if score_kind == official_ctx { "Best rank score" } else { "Run ranking score" };

    let d = PyDict::new_bound(py);
    d.set_item("rank_score",                         rank_score)?;
    d.set_item("avg_score",                          avg)?;
    d.set_item("phase_benefit_bonus",                phase_bon)?;
    d.set_item("boost_penalty",                      boost)?;
    d.set_item("event_penalty",                      event)?;
    d.set_item("lr_delta_penalty",                   lr_pen)?;
    d.set_item("dsp_penalty",                        dsp_pen)?;
    d.set_item("exc_penalty",                        exc_pen)?;
    d.set_item("bass_integration_penalty",           bass_pen)?;
    d.set_item("bass_feasibility_penalty",           bass_feas)?;
    d.set_item("bass_preference_bonus",              bass_bon)?;
    d.set_item("mode_penalty",                       mode_pen)?;
    d.set_item("decay_penalty",                      decay_pen)?;
    d.set_item("residual_peak_penalty",              rp_pen)?;
    d.set_item("correction_sharpness_penalty",       sharp_pen)?;
    d.set_item("dip_fill_risk_penalty",              dip_pen)?;
    d.set_item("channel_overfit_penalty",            ch_of_pen)?;
    d.set_item("target_tracking_penalty",            tt_pen)?;
    d.set_item("voice_clarity_penalty",              voice_pen)?;
    d.set_item("phase_risk_penalty",                 phase_risk)?;
    d.set_item("phase_limit_penalty",                phase_pen)?;
    d.set_item("thd_boost_penalty",                  thd_pen)?;
    d.set_item("stereo_coherence_penalty",           stereo_pen)?;
    d.set_item("phantom_center_stability_penalty",   center_pen)?;
    d.set_item("policy_divergence_penalty",          policy_pen)?;
    d.set_item("asymmetry_budget_overflow_penalty",  asym_pen)?;
    d.set_item("worst_channel_relief_bonus",         worst_bon)?;
    d.set_item("shared_preference_bias",             shared_pen)?;
    d.set_item("shared_preference_penalty",          shared_pen)?;
    d.set_item("rt60_policy_penalty",                rt60_pen)?;
    d.set_item("harmonic_local_boost_penalty",       harm_pen)?;
    d.set_item("rank_score_raw",                     rank_raw)?;
    d.set_item("rank_score_gain",                    g)?;
    d.set_item("rank_score_bias",                    b)?;

    let ctx_d = PyDict::new_bound(py);
    ctx_d.set_item("score_kind",  score_kind)?;
    ctx_d.set_item("score_label", score_label)?;
    ctx_d.set_item("score_min",   lo)?;
    ctx_d.set_item("score_max",   hi)?;
    d.set_item("context", ctx_d)?;

    Ok(d.into())
}

// ---------------------------------------------------------------------------
// calibrated_auto_quality
// ---------------------------------------------------------------------------
#[pyfunction]
#[pyo3(signature = (rank_score_0_100, metrics = None))]
fn calibrated_auto_quality(
    rank_score_0_100: f64,
    metrics: Option<&Bound<'_, PyDict>>,
) -> f64 {
    if !rank_score_0_100.is_finite() {
        return f64::NAN;
    }
    let raw_c = clamp(rank_score_0_100, 0.0, 100.0);
    let mut score = 100.0 * (1.0 - (1.0 - raw_c / 100.0_f64).powf(2.35));
    score = clamp(score, 0.0, 100.0);

    if let Some(m) = metrics {
        if let Ok(Some(v)) = m.get_item("max_net_boost_db") {
            if let Ok(vf) = v.extract::<f64>() {
                if vf.is_finite() && vf > 6.5 {
                    score = score.min(69.0);
                }
            }
        }
        let mut hard_gate_failed = false;
        if let Ok(Some(v)) = m.get_item("hard_gate_failed") {
            hard_gate_failed |= v.is_truthy().unwrap_or(false);
        }
        if let Ok(Some(v)) = m.get_item("stereo_policy_gate_failed") {
            hard_gate_failed |= v.is_truthy().unwrap_or(false);
        }
        if let (Ok(Some(rp_v)), Ok(Some(rg_v))) = (
            m.get_item("worst_residual_peak_db"),
            m.get_item("residual_peak_hard_gate_db"),
        ) {
            if let (Ok(rp), Ok(rg)) = (rp_v.extract::<f64>(), rg_v.extract::<f64>()) {
                if rp.is_finite() && rg.is_finite() && rp > rg {
                    hard_gate_failed = true;
                }
            }
        }
        if hard_gate_failed {
            score = score.min(59.0);
        }
        if let Ok(Some(v)) = m.get_item("bass_cancellation_risk") {
            if let Ok(vf) = v.extract::<f64>() {
                if vf.is_finite() && vf > 0.5 {
                    score = score.min(70.0);
                }
            }
        }
        if let Ok(Some(v)) = m.get_item("bass_xo_gd_mismatch_delta_ms") {
            if let Ok(vf) = v.extract::<f64>() {
                if vf.is_finite() && vf.abs() > 5.0 {
                    score = score.min(65.0);
                }
            }
        }
    }

    clamp(score, 0.0, 100.0)
}

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------
#[pymodule]
fn camillafir_scoring(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_rank_score_components, m)?)?;
    m.add_function(wrap_pyfunction!(calibrated_auto_quality, m)?)?;
    m.add("__version__", "0.1.0")?;
    Ok(())
}
