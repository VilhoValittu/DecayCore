// DecayCore
// Copyright (c) 2026 Vilho Valittu.
// All rights reserved except as expressly granted in the LICENSE file.
//
// This file is part of the public source-available DecayCore repository.
// Non-commercial use is permitted under the terms of the LICENSE file.
// Commercial use requires separate written permission.
//
// SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

// DecayCore scoring extension — Rust/PyO3 0.22
// Mirrors rank_score.py: compute_rank_score_components() and calibrated_auto_quality()

#![allow(clippy::useless_conversion)]

use numpy::{PyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyModule, PyTuple};

const RANK_SCORE_BATCH_FIELDS: [&str; 31] = [
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
    "rt60_policy_penalty",
    "harmonic_local_boost_penalty",
    "shared_preference_penalty",
];
const RANK_SCORE_BATCH_WIDTH: usize = RANK_SCORE_BATCH_FIELDS.len();
const GIL_RELEASE_MIN_BATCH_ROWS: usize = 64;

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

#[inline]
fn safe_f64(v: f64) -> f64 {
    if v.is_finite() {
        v
    } else {
        0.0
    }
}

#[inline]
fn clamp(v: f64, lo: f64, hi: f64) -> f64 {
    v.max(lo).min(hi)
}

#[inline]
fn metric_f64(value: &Bound<'_, PyAny>) -> Option<f64> {
    let parsed = value.extract::<f64>().ok().or_else(|| {
        value
            .extract::<String>()
            .ok()
            .and_then(|text| text.trim().parse::<f64>().ok())
    })?;
    parsed.is_finite().then_some(parsed)
}

#[inline]
fn rank_raw_from_values(values: &[f64]) -> f64 {
    debug_assert_eq!(values.len(), RANK_SCORE_BATCH_WIDTH);
    let positive = |index: usize| safe_f64(values[index]).max(0.0);
    safe_f64(values[0]) + positive(1)
        - positive(2)
        - positive(3)
        - positive(4)
        - positive(5)
        - positive(6)
        - positive(7)
        - positive(8)
        - positive(9)
        + positive(10)
        - positive(11)
        - positive(12)
        - positive(13)
        - positive(14)
        - positive(15)
        - positive(16)
        - positive(17)
        - positive(18)
        - positive(19)
        - positive(20)
        - positive(21)
        - positive(22)
        - positive(23)
        - positive(24)
        - positive(25)
        + positive(26)
        - positive(27).max(positive(30))
        - positive(28)
        - positive(29)
}

#[inline]
fn score_bounds(gain: f64, bias: f64, score_min: f64, score_max: f64) -> (f64, f64, f64, f64) {
    let gain = if gain.is_finite() { gain } else { 1.0 };
    let bias = if bias.is_finite() { bias } else { 0.0 };
    let lo = if score_min.is_finite() {
        score_min
    } else {
        0.0
    };
    let hi = if score_max.is_finite() {
        score_max
    } else {
        100.0
    };
    let (lo, hi) = if hi < lo { (hi, lo) } else { (lo, hi) };
    (gain, bias, lo, hi)
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
    bass_prering_penalty = 0.0,
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
    bass_prering_penalty: f64,
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
    let avg = safe_f64(avg_score);
    let phase_bon = safe_f64(phase_benefit_bonus).max(0.0);
    let boost = safe_f64(boost_penalty).max(0.0);
    let event = safe_f64(event_penalty).max(0.0);
    let lr_pen = safe_f64(lr_delta_penalty).max(0.0);
    let dsp_pen = safe_f64(dsp_penalty).max(0.0);
    let bass_prering = safe_f64(bass_prering_penalty).max(0.0);
    let exc_pen = safe_f64(exc_penalty).max(0.0);
    let bass_pen = safe_f64(bass_integration_penalty).max(0.0);
    let bass_feas = safe_f64(bass_feasibility_penalty).max(0.0);
    let bass_bon = safe_f64(bass_preference_bonus).max(0.0);
    let mode_pen = safe_f64(mode_penalty).max(0.0);
    let decay_pen = safe_f64(decay_penalty).max(0.0);
    let rp_pen = safe_f64(residual_peak_penalty).max(0.0);
    let sharp_pen = safe_f64(correction_sharpness_penalty).max(0.0);
    let dip_pen = safe_f64(dip_fill_risk_penalty).max(0.0);
    let ch_of_pen = safe_f64(channel_overfit_penalty).max(0.0);
    let tt_pen = safe_f64(target_tracking_penalty).max(0.0);
    let voice_pen = safe_f64(voice_clarity_penalty).max(0.0);
    let phase_risk = safe_f64(phase_risk_penalty).max(0.0);
    let phase_pen = safe_f64(phase_limit_penalty).max(0.0);
    let thd_pen = safe_f64(thd_boost_penalty).max(0.0);
    let stereo_pen = safe_f64(stereo_coherence_penalty).max(0.0);
    let center_pen = safe_f64(phantom_center_stability_penalty).max(0.0);
    let policy_pen = safe_f64(policy_divergence_penalty).max(0.0);
    let asym_pen = safe_f64(asymmetry_budget_overflow_penalty).max(0.0);
    let worst_bon = safe_f64(worst_channel_relief_bonus).max(0.0);
    let shared_pen = safe_f64(shared_preference_bias)
        .max(0.0)
        .max(safe_f64(shared_preference_penalty).max(0.0));
    let rt60_pen = safe_f64(rt60_policy_penalty).max(0.0);
    let harm_pen = safe_f64(harmonic_local_boost_penalty).max(0.0);

    let (g, b, lo, hi) = score_bounds(gain, bias, score_min, score_max);
    let rank_raw = rank_raw_from_values(&[
        avg,
        phase_bon,
        boost,
        event,
        lr_pen,
        dsp_pen,
        bass_prering,
        exc_pen,
        bass_pen,
        bass_feas,
        bass_bon,
        mode_pen,
        decay_pen,
        rp_pen,
        sharp_pen,
        dip_pen,
        ch_of_pen,
        tt_pen,
        voice_pen,
        phase_risk,
        phase_pen,
        thd_pen,
        stereo_pen,
        center_pen,
        policy_pen,
        asym_pen,
        worst_bon,
        shared_pen,
        rt60_pen,
        harm_pen,
        shared_pen,
    ]);

    let rank_score = clamp(g * rank_raw + b, lo, hi);

    let official_ctx = "preset_objective_score";
    let score_kind = context.unwrap_or(official_ctx).trim();
    let score_kind = if score_kind.is_empty() {
        official_ctx
    } else {
        score_kind
    };
    let score_label = if score_kind == official_ctx {
        "Best rank score"
    } else {
        "Run ranking score"
    };

    let d = PyDict::new_bound(py);
    d.set_item("rank_score", rank_score)?;
    d.set_item("avg_score", avg)?;
    d.set_item("phase_benefit_bonus", phase_bon)?;
    d.set_item("boost_penalty", boost)?;
    d.set_item("event_penalty", event)?;
    d.set_item("lr_delta_penalty", lr_pen)?;
    d.set_item("dsp_penalty", dsp_pen)?;
    d.set_item("bass_prering_penalty", bass_prering)?;
    d.set_item("exc_penalty", exc_pen)?;
    d.set_item("bass_integration_penalty", bass_pen)?;
    d.set_item("bass_feasibility_penalty", bass_feas)?;
    d.set_item("bass_preference_bonus", bass_bon)?;
    d.set_item("mode_penalty", mode_pen)?;
    d.set_item("decay_penalty", decay_pen)?;
    d.set_item("residual_peak_penalty", rp_pen)?;
    d.set_item("correction_sharpness_penalty", sharp_pen)?;
    d.set_item("dip_fill_risk_penalty", dip_pen)?;
    d.set_item("channel_overfit_penalty", ch_of_pen)?;
    d.set_item("target_tracking_penalty", tt_pen)?;
    d.set_item("voice_clarity_penalty", voice_pen)?;
    d.set_item("phase_risk_penalty", phase_risk)?;
    d.set_item("phase_limit_penalty", phase_pen)?;
    d.set_item("thd_boost_penalty", thd_pen)?;
    d.set_item("stereo_coherence_penalty", stereo_pen)?;
    d.set_item("phantom_center_stability_penalty", center_pen)?;
    d.set_item("policy_divergence_penalty", policy_pen)?;
    d.set_item("asymmetry_budget_overflow_penalty", asym_pen)?;
    d.set_item("worst_channel_relief_bonus", worst_bon)?;
    d.set_item("shared_preference_bias", shared_pen)?;
    d.set_item("shared_preference_penalty", shared_pen)?;
    d.set_item("rt60_policy_penalty", rt60_pen)?;
    d.set_item("harmonic_local_boost_penalty", harm_pen)?;
    d.set_item("rank_score_raw", rank_raw)?;
    d.set_item("rank_score_gain", g)?;
    d.set_item("rank_score_bias", b)?;

    let ctx_d = PyDict::new_bound(py);
    ctx_d.set_item("score_kind", score_kind)?;
    ctx_d.set_item("score_label", score_label)?;
    ctx_d.set_item("score_min", lo)?;
    ctx_d.set_item("score_max", hi)?;
    d.set_item("context", ctx_d)?;

    Ok(d.into())
}

// ---------------------------------------------------------------------------
// compute_rank_scores_batch
// ---------------------------------------------------------------------------
/// Compute rank scores for a `(candidates, 31)` component matrix.
///
/// Column order is exposed as `RANK_SCORE_BATCH_FIELDS`. Safety gates remain
/// outside this numeric combiner and must still be applied by winner selection.
#[pyfunction]
#[pyo3(signature = (
    component_rows,
    gain = 1.0,
    bias = 0.0,
    score_min = 0.0,
    score_max = 100.0
))]
fn compute_rank_scores_batch<'py>(
    py: Python<'py>,
    component_rows: PyReadonlyArray2<f64>,
    gain: f64,
    bias: f64,
    score_min: f64,
    score_max: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let view = component_rows.as_array();
    let rows = view.nrows();
    let columns = view.ncols();
    if columns != RANK_SCORE_BATCH_WIDTH {
        return Err(PyValueError::new_err(format!(
            "component_rows must have {} columns, got {}",
            RANK_SCORE_BATCH_WIDTH, columns
        )));
    }

    let values: Vec<f64> = view.iter().copied().collect();
    let (gain, bias, lo, hi) = score_bounds(gain, bias, score_min, score_max);
    let compute = move || {
        values
            .chunks_exact(RANK_SCORE_BATCH_WIDTH)
            .map(|row| clamp(gain * rank_raw_from_values(row) + bias, lo, hi))
            .collect::<Vec<_>>()
    };
    let scores = if rows >= GIL_RELEASE_MIN_BATCH_ROWS {
        py.allow_threads(compute)
    } else {
        compute()
    };
    Ok(PyArray1::from_vec_bound(py, scores))
}

// ---------------------------------------------------------------------------
// calibrated_auto_quality
// ---------------------------------------------------------------------------
#[pyfunction]
#[pyo3(signature = (rank_score_0_100, metrics = None))]
fn calibrated_auto_quality(rank_score_0_100: f64, metrics: Option<&Bound<'_, PyDict>>) -> f64 {
    if !rank_score_0_100.is_finite() {
        return f64::NAN;
    }
    let raw_c = clamp(rank_score_0_100, 0.0, 100.0);
    let mut score = 100.0 * (1.0 - (1.0 - raw_c / 100.0_f64).powf(2.35));
    score = clamp(score, 0.0, 100.0);

    if let Some(m) = metrics {
        if let Ok(Some(v)) = m.get_item("max_net_boost_db") {
            if let Some(vf) = metric_f64(&v) {
                if vf > 6.5 {
                    score = score.min(69.0);
                }
            }
        }
        if let Ok(Some(v)) = m.get_item("event_penalty") {
            if let Some(vf) = metric_f64(&v) {
                if vf >= 3.0 {
                    score = score.min(100.0);
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
            if let (Some(rp), Some(rg)) = (metric_f64(&rp_v), metric_f64(&rg_v)) {
                if rp > rg {
                    hard_gate_failed = true;
                }
            }
        }
        if hard_gate_failed {
            score = score.min(59.0);
        }
        if let Ok(Some(v)) = m.get_item("residual_peak_penalty") {
            if let Some(vf) = metric_f64(&v) {
                if vf > 1.5 {
                    score = score.min(100.0);
                }
            }
        }
        if let Ok(Some(v)) = m.get_item("bass_cancellation_risk") {
            if let Some(vf) = metric_f64(&v) {
                if vf > 0.5 {
                    score = score.min(70.0);
                }
            }
        }
        if let Ok(Some(v)) = m.get_item("bass_xo_gd_mismatch_delta_ms") {
            if let Some(vf) = metric_f64(&v) {
                if vf.abs() > 5.0 {
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
fn decaycore_scoring(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_rank_score_components, m)?)?;
    m.add_function(wrap_pyfunction!(compute_rank_scores_batch, m)?)?;
    m.add_function(wrap_pyfunction!(calibrated_auto_quality, m)?)?;
    m.add(
        "RANK_SCORE_BATCH_FIELDS",
        PyTuple::new_bound(m.py(), RANK_SCORE_BATCH_FIELDS),
    )?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{clamp, rank_raw_from_values, safe_f64, RANK_SCORE_BATCH_WIDTH};

    #[test]
    fn safe_float_neutralizes_non_finite_values() {
        assert_eq!(safe_f64(f64::NAN), 0.0);
        assert_eq!(safe_f64(f64::INFINITY), 0.0);
        assert_eq!(safe_f64(3.5), 3.5);
    }

    #[test]
    fn clamp_respects_requested_bounds() {
        assert_eq!(clamp(-1.0, 0.0, 100.0), 0.0);
        assert_eq!(clamp(101.0, 0.0, 100.0), 100.0);
        assert_eq!(clamp(42.0, 0.0, 100.0), 42.0);
    }

    #[test]
    fn batch_row_uses_same_bonus_penalty_directions_and_shared_max() {
        let mut row = [0.0; RANK_SCORE_BATCH_WIDTH];
        row[0] = 80.0;
        row[1] = 2.0;
        row[2] = 3.0;
        row[27] = 1.0;
        row[30] = 4.0;
        assert_eq!(rank_raw_from_values(&row), 75.0);
    }
}
