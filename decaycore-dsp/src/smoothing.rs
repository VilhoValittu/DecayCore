// DecayCore
// Copyright (c) 2026 Vilho Valittu.
// All rights reserved except as expressly granted in the LICENSE file.
//
// This file is part of the public source-available DecayCore repository.
// Non-commercial use is permitted under the terms of the LICENSE file.
// Commercial use requires separate written permission.
//
// SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

// Smoothing core — mirrors smoothing.py::_smooth_mag_core

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use crate::utils::safe_f64;

// ---------------------------------------------------------------------------
// smooth_mag_core_rs: Core log-grid smoothing kernel
// ---------------------------------------------------------------------------
/// Fused interpolation → edge-padding → convolution → interpolation back.
/// Mirrors numba::_smooth_mag_core.
///
/// Args:
///   freqs: original frequency axis (must be strictly increasing)
///   mags: magnitude values at freqs
///   log_freqs: log-grid frequency axis (strictly increasing)
///   window: Hanning window, normalized (sum = 1.0)
///   pad_len: padding length on each side
///
/// Returns: magnitude values interpolated back to original freqs
#[pyfunction]
pub fn smooth_mag_core_rs<'py>(
    py: Python<'py>,
    freqs: PyReadonlyArray1<f64>,
    mags: PyReadonlyArray1<f64>,
    log_freqs: PyReadonlyArray1<f64>,
    window: PyReadonlyArray1<f64>,
    pad_len: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let freqs = freqs.as_slice()?;
    let mags = mags.as_slice()?;
    let log_freqs = log_freqs.as_slice()?;
    let window = window.as_slice()?;

    if freqs.is_empty() || mags.is_empty() || log_freqs.is_empty() || window.is_empty() {
        return Ok(PyArray1::from_vec_bound(py, mags.to_vec()));
    }

    // Step 1: Interpolate magnitudes onto log frequency grid
    let mut log_mags = vec![0.0; log_freqs.len()];
    for (i, &lf) in log_freqs.iter().enumerate() {
        log_mags[i] = interp1_linear(freqs, mags, lf);
    }

    // Step 2: Edge-pad with boundary values
    let n = log_mags.len();
    let w = window.len();
    let p = n + 2 * pad_len;
    let mut padded = vec![0.0; p];

    // Left edge padding
    for i in 0..pad_len {
        padded[i] = log_mags[0];
    }
    // Center data
    for i in 0..n {
        padded[pad_len + i] = log_mags[i];
    }
    // Right edge padding
    for i in 0..pad_len {
        padded[pad_len + n + i] = log_mags[n - 1];
    }

    // Step 3: Convolve padded with window (same-mode style)
    // output[ii] = sum_k padded[half_w + pad_len + ii - k] * window[k]
    let half_w = (w.saturating_sub(1)) / 2;
    let mut sm = vec![0.0; n];
    for ii in 0..n {
        let mut acc = 0.0;
        let base = half_w + pad_len + ii;
        for k in 0..w {
            acc += padded.get(base.saturating_sub(k)).copied().unwrap_or(0.0) * window[k];
        }
        sm[ii] = acc;
    }

    // Step 4: Interpolate smoothed values back to original frequency grid
    let mut result = vec![0.0; freqs.len()];
    for (j, &f) in freqs.iter().enumerate() {
        result[j] = interp1_linear(log_freqs, &sm, f);
    }

    Ok(PyArray1::from_vec_bound(py, result))
}

// Simple 1D linear interpolation helper (no extrapolation)
#[inline]
fn interp1_linear(x: &[f64], y: &[f64], xi: f64) -> f64 {
    if x.is_empty() || y.is_empty() || x.len() != y.len() {
        return 0.0;
    }

    // Boundary cases
    if xi <= x[0] {
        return y[0];
    }
    if xi >= x[x.len() - 1] {
        return y[x.len() - 1];
    }

    // Binary search for interval [x[lo], x[hi]] containing xi
    let mut lo = 0;
    let mut hi = x.len() - 1;
    while lo + 1 < hi {
        let mid = (lo + hi) / 2;
        if x[mid] <= xi {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    // Linear interpolation between lo and hi
    let x_lo = x[lo];
    let x_hi = x[lo + 1];
    if (x_hi - x_lo).abs() < 1e-14 {
        return (y[lo] + y[lo + 1]) * 0.5;
    }

    let y_lo = y[lo];
    let y_hi = y[lo + 1];
    y_lo + (xi - x_lo) * (y_hi - y_lo) / (x_hi - x_lo)
}

// Placeholder function (Phase 2/3)
#[pyfunction]
pub fn apply_afdw_stack_rs<'py>(
    py: Python<'py>,
    freqs: PyReadonlyArray1<f64>,
    mags: PyReadonlyArray1<f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let _freqs = freqs.as_slice()?;
    let mags = mags.as_slice()?;
    Ok(PyArray1::from_vec_bound(py, mags.to_vec()))
}
