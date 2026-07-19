// DecayCore
// Copyright (c) 2026 Vilho Valittu.
// All rights reserved except as expressly granted in the LICENSE file.
//
// This file is part of the public source-available DecayCore repository.
// Non-commercial use is permitted under the terms of the LICENSE file.
// Commercial use requires separate written permission.
//
// SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

// Robust per-row RMS — mirrors repeat_representative.py::_rms_rows_kernel

use numpy::{PyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

const GIL_RELEASE_MIN_VALUES: usize = 4096;

// ---------------------------------------------------------------------------
// rms_rows_kernel_rs: robust RMS per row via median of squared values.
// ---------------------------------------------------------------------------
/// For each row, take the finite values, square them, and return the square
/// root of their median. Empty rows yield 0.0.
///
/// Mirrors repeat_representative.py::_rms_rows_kernel.
#[pyfunction]
pub fn rms_rows_kernel_rs<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray2<f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let view = arr.as_array();
    let nrows = view.nrows();
    let ncols = view.ncols();
    let values: Vec<f64> = view.iter().copied().collect();
    let out = if values.len() >= GIL_RELEASE_MIN_VALUES {
        py.allow_threads(move || rms_rows_kernel(&values, nrows, ncols))
    } else {
        rms_rows_kernel(&values, nrows, ncols)
    };

    Ok(PyArray1::from_vec_bound(py, out))
}

fn rms_rows_kernel(values: &[f64], nrows: usize, ncols: usize) -> Vec<f64> {
    let mut out = vec![0.0; nrows];
    let mut tmp: Vec<f64> = Vec::with_capacity(ncols);

    for row in 0..nrows {
        tmp.clear();
        for &v in &values[row * ncols..(row + 1) * ncols] {
            if v.is_finite() {
                tmp.push(v * v);
            }
        }
        let count = tmp.len();
        if count == 0 {
            continue;
        }
        // Sort ascending; matches numpy/numba median semantics on finite data.
        tmp.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let med = if count % 2 == 1 {
            tmp[count / 2]
        } else {
            0.5 * (tmp[count / 2 - 1] + tmp[count / 2])
        };
        out[row] = med.max(0.0).sqrt();
    }

    out
}
