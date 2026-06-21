// DecayCore
// Copyright (c) 2026 Vilho Valittu.
// All rights reserved except as expressly granted in the LICENSE file.
//
// This file is part of the public source-available DecayCore repository.
// Non-commercial use is permitted under the terms of the LICENSE file.
// Commercial use requires separate written permission.
//
// SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

// Slope-per-octave limiters — mirrors dsp/limits.py::_slope_passes + _slope_passes_asym

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

// ---------------------------------------------------------------------------
// Symmetric slope limiting: forward + backward pass.
// Mirrors limits.py::_slope_passes (in-place over a working copy).
// ---------------------------------------------------------------------------
fn slope_passes(g: &mut [f64], x: &[f64], max_db_per_oct: f64) {
    let n = x.len();
    if n == 0 {
        return;
    }

    // Forward pass
    for k in 1..n {
        let dx = x[k] - x[k - 1];
        if dx <= 0.0 {
            continue;
        }
        let lim = max_db_per_oct * dx;
        let dg = g[k] - g[k - 1];
        if dg > lim {
            g[k] = g[k - 1] + lim;
        } else if dg < -lim {
            g[k] = g[k - 1] - lim;
        }
    }

    // Backward pass
    for k in (0..n - 1).rev() {
        let dx = x[k + 1] - x[k];
        if dx <= 0.0 {
            continue;
        }
        let lim = max_db_per_oct * dx;
        let dg = g[k] - g[k + 1];
        if dg > lim {
            g[k] = g[k + 1] + lim;
        } else if dg < -lim {
            g[k] = g[k + 1] - lim;
        }
    }
}

// ---------------------------------------------------------------------------
// Asymmetric slope limiting helpers.
// Mirrors limits.py::_slope_passes_asym_forward / _slope_passes_asym_backward.
// ---------------------------------------------------------------------------
fn slope_passes_asym_forward(g: &mut [f64], x: &[f64], boost: f64, cut: f64) {
    let n = x.len();
    for k in 1..n {
        let dx = x[k] - x[k - 1];
        if dx <= 0.0 {
            continue;
        }
        let dg = g[k] - g[k - 1];
        let lim = (if dg > 0.0 { boost } else { cut }) * dx;
        if (dg > 0.0 && boost <= 0.0) || (dg < 0.0 && cut <= 0.0) {
            continue;
        }
        if dg > lim {
            g[k] = g[k - 1] + lim;
        } else if dg < -lim {
            g[k] = g[k - 1] - lim;
        }
    }
}

fn slope_passes_asym_backward(g: &mut [f64], x: &[f64], boost: f64, cut: f64) {
    let n = x.len();
    if n == 0 {
        return;
    }
    for k in (0..n - 1).rev() {
        let dx = x[k + 1] - x[k];
        if dx <= 0.0 {
            continue;
        }
        let dg = g[k] - g[k + 1];
        let lim = (if dg > 0.0 { boost } else { cut }) * dx;
        if (dg > 0.0 && boost <= 0.0) || (dg < 0.0 && cut <= 0.0) {
            continue;
        }
        if dg > lim {
            g[k] = g[k + 1] + lim;
        } else if dg < -lim {
            g[k] = g[k + 1] - lim;
        }
    }
}

// ---------------------------------------------------------------------------
// slope_passes_rs: symmetric dB/octave slope limiter.
// ---------------------------------------------------------------------------
/// Limit the rate of change of `g` (dB) symmetrically to `max_db_per_oct`.
///
/// Mirrors limits.py::_slope_passes. Returns a new array (the numba version
/// mutates in place and returns the same buffer).
#[pyfunction]
pub fn slope_passes_rs<'py>(
    py: Python<'py>,
    g: PyReadonlyArray1<f64>,
    x: PyReadonlyArray1<f64>,
    max_db_per_oct: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let x = x.as_slice()?;
    let mut gv = g.as_slice()?.to_vec();
    slope_passes(&mut gv, x, max_db_per_oct);
    Ok(PyArray1::from_vec_bound(py, gv))
}

// ---------------------------------------------------------------------------
// slope_passes_asym_rs: asymmetric (separate boost/cut) slope limiter.
// ---------------------------------------------------------------------------
/// Limit the rate of change of `g` (dB) asymmetrically with separate `boost`
/// (rising) and `cut` (falling) limits.
///
/// Mirrors limits.py::_slope_passes_asym (forward + backward passes).
#[pyfunction]
pub fn slope_passes_asym_rs<'py>(
    py: Python<'py>,
    g: PyReadonlyArray1<f64>,
    x: PyReadonlyArray1<f64>,
    boost: f64,
    cut: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let x = x.as_slice()?;
    let mut gv = g.as_slice()?.to_vec();
    slope_passes_asym_forward(&mut gv, x, boost, cut);
    slope_passes_asym_backward(&mut gv, x, boost, cut);
    Ok(PyArray1::from_vec_bound(py, gv))
}
