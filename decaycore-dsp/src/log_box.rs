// DecayCore
// Copyright (c) 2026 Vilho Valittu.
// All rights reserved except as expressly granted in the LICENSE file.
//
// This file is part of the public source-available DecayCore repository.
// Non-commercial use is permitted under the terms of the LICENSE file.
// Commercial use requires separate written permission.
//
// SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

// Log-space box smoothing — mirrors modal_preparation.py::_smooth_log_box_kernel

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

// ---------------------------------------------------------------------------
// smooth_log_box_kernel_rs: box-average over a ±half window in log space.
// ---------------------------------------------------------------------------
/// Average finite `y_raw` values within `[x[i]-half, x[i]+half]` using prefix
/// sums and binary search for the window edges.
///
/// Mirrors modal_preparation.py::_smooth_log_box_kernel. `x` must be sorted
/// ascending. Bin boundaries: left edge inclusive of `lo` (`x[mid] < lo`),
/// right edge inclusive of `hi` (`x[mid] <= hi`).
#[pyfunction]
pub fn smooth_log_box_kernel_rs<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<f64>,
    y_raw: PyReadonlyArray1<f64>,
    half: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let x = x.as_slice()?;
    let y_raw = y_raw.as_slice()?;
    let n = x.len();
    if y_raw.len() != n {
        return Err(PyValueError::new_err(format!(
            "x and y_raw must have the same length, got {} and {}",
            n,
            y_raw.len()
        )));
    }

    // Prefix sums of finite values and their weights.
    let mut sum_y = vec![0.0; n + 1];
    let mut sum_w = vec![0.0; n + 1];
    for i in 0..n {
        if y_raw[i].is_finite() {
            sum_y[i + 1] = sum_y[i] + y_raw[i];
            sum_w[i + 1] = sum_w[i] + 1.0;
        } else {
            sum_y[i + 1] = sum_y[i];
            sum_w[i + 1] = sum_w[i];
        }
    }

    let mut out = vec![0.0; n];
    for i in 0..n {
        let lo = x[i] - half;
        let hi = x[i] + half;

        // Lower bound: first index with x[idx] >= lo.
        let mut left = 0usize;
        let mut right = n;
        while left < right {
            let mid = (left + right) >> 1;
            if x[mid] < lo {
                left = mid + 1;
            } else {
                right = mid;
            }
        }

        // Upper bound: first index with x[idx] > hi.
        let mut rl = 0usize;
        let mut rr = n;
        while rl < rr {
            let mid = (rl + rr) >> 1;
            if x[mid] <= hi {
                rl = mid + 1;
            } else {
                rr = mid;
            }
        }

        let count = sum_w[rl] - sum_w[left];
        out[i] = if count > 0.0 {
            (sum_y[rl] - sum_y[left]) / count
        } else {
            0.0
        };
    }

    Ok(PyArray1::from_vec_bound(py, out))
}
