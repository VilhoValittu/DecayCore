// GD gradient limiter — mirrors dsp_ops.py::_gd_smooth_loop + helpers

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use std::f64;

// ---------------------------------------------------------------------------
// Gradient computation: central differences with spacing
// ---------------------------------------------------------------------------
fn gradient1d(arr: &[f64], spacing: &[f64]) -> Vec<f64> {
    let n = arr.len();
    let mut out = vec![0.0; n];

    if n <= 1 {
        return out;
    }

    // Forward difference at start
    out[0] = (arr[1] - arr[0]) / (spacing[1] - spacing[0]);

    // Backward difference at end
    out[n - 1] = (arr[n - 1] - arr[n - 2]) / (spacing[n - 1] - spacing[n - 2]);

    // Central differences in the middle
    for i in 1..n - 1 {
        out[i] = (arr[i + 1] - arr[i - 1]) / (spacing[i + 1] - spacing[i - 1]);
    }

    out
}

// ---------------------------------------------------------------------------
// Gaussian blur: reflect boundary conditions (nearest-neighbor extrapolation)
// ---------------------------------------------------------------------------
fn gaussian1d_nearest(arr: &[f64], sigma: f64) -> Vec<f64> {
    let radius = ((4.0 * sigma + 0.5) as i32).max(1) as usize;
    let n = arr.len();
    let s2 = 2.0 * sigma * sigma;
    let ksize = 2 * radius + 1;

    // Build Gaussian kernel
    let mut k = vec![0.0; ksize];
    for i in 0..ksize {
        let x = (i as f64) - (radius as f64);
        k[i] = (-x * x / s2).exp();
    }

    // Normalize kernel
    let k_sum: f64 = k.iter().sum();
    if k_sum > 0.0 {
        for ki in &mut k {
            *ki /= k_sum;
        }
    }

    // Convolve with boundary reflection
    let mut out = vec![0.0; n];
    for i in 0..n {
        let mut acc = 0.0;
        for j in 0..ksize {
            let offset = (j as i32) - (radius as i32);
            let idx = ((i as i32) + offset).max(0).min((n - 1) as i32) as usize;
            acc += arr[idx] * k[j];
        }
        out[i] = acc;
    }

    out
}

// ---------------------------------------------------------------------------
// gd_smooth_loop_rs: Main GD smoothing loop
// ---------------------------------------------------------------------------
/// Iterative group delay smoothing with gradient and curvature limiting.
///
/// Mirrors dsp_ops.py::_gd_smooth_loop
///
/// Args:
///   gd_l: group delay (ms) values
///   log2f: log2(frequency) axis (spacing for gradient computation)
///   lim_arr: gradient limit for each bin (ms/oct)
///   curv_lim_arr: curvature limit for each bin (ms/oct²)
///   sigma: initial Gaussian blur sigma
///
/// Returns: smoothed group delay array
#[pyfunction]
pub fn gd_smooth_loop_rs<'py>(
    py: Python<'py>,
    gd_l: PyReadonlyArray1<f64>,
    log2f: PyReadonlyArray1<f64>,
    lim_arr: PyReadonlyArray1<f64>,
    curv_lim_arr: PyReadonlyArray1<f64>,
    sigma: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let gd_l = gd_l.as_slice()?;
    let log2f = log2f.as_slice()?;
    let lim_arr = lim_arr.as_slice()?;
    let curv_lim_arr = curv_lim_arr.as_slice()?;

    if gd_l.is_empty() || log2f.is_empty() || lim_arr.is_empty() || curv_lim_arr.is_empty() {
        return Ok(PyArray1::from_vec_bound(py, gd_l.to_vec()));
    }

    let mut gd = gd_l.to_vec();
    let mut current_sigma = sigma.max(0.25);

    for _iteration in 0..14 {
        // Compute gradient
        let mut gd_grad = gradient1d(&gd, log2f);

        // Clean non-finite values
        for v in &mut gd_grad {
            if !v.is_finite() {
                *v = 0.0;
            }
        }

        // Compute curvature (gradient of gradient)
        let mut gd_curv = gradient1d(&gd_grad, log2f);

        // Clean non-finite values
        for v in &mut gd_curv {
            if !v.is_finite() {
                *v = 0.0;
            }
        }

        // Find max violation ratio
        let mut max_ratio = 0.0;
        for i in 0..gd_grad.len() {
            if lim_arr[i] > 0.0 {
                let r = (gd_grad[i] / lim_arr[i]).abs();
                if r > max_ratio {
                    max_ratio = r;
                }
            }

            if curv_lim_arr[i] > 0.0 {
                let rc = (gd_curv[i] / curv_lim_arr[i]).abs();
                if rc > max_ratio {
                    max_ratio = rc;
                }
            }
        }

        // Check convergence
        if max_ratio <= 1.001 {
            break;
        }

        // Apply Gaussian blur and increase sigma
        gd = gaussian1d_nearest(&gd, current_sigma);
        current_sigma *= 1.25;
    }

    Ok(PyArray1::from_vec_bound(py, gd))
}
