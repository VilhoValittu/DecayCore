// DecayCore
// Copyright (c) 2026 Vilho Valittu.
// All rights reserved except as expressly granted in the LICENSE file.
//
// This file is part of the public source-available DecayCore repository.
// Non-commercial use is permitted under the terms of the LICENSE file.
// Commercial use requires separate written permission.
//
// SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

// DecayCore DSP extension — Rust/PyO3 0.22
// Selective Rust hotpath acceleration: smoothing, phase/IR stage

mod utils;
mod smoothing;
mod gd_limit;
mod slope_limit;
mod rms;
mod log_box;

use pyo3::prelude::*;
use pyo3::types::PyModule;

use pyo3::wrap_pyfunction;

#[pymodule]
fn decaycore_dsp(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(smoothing::smooth_mag_core_rs, m)?)?;
    m.add_function(wrap_pyfunction!(smoothing::apply_afdw_stack_rs, m)?)?;
    m.add_function(wrap_pyfunction!(gd_limit::gd_smooth_loop_rs, m)?)?;
    m.add_function(wrap_pyfunction!(slope_limit::slope_passes_rs, m)?)?;
    m.add_function(wrap_pyfunction!(slope_limit::slope_passes_asym_rs, m)?)?;
    m.add_function(wrap_pyfunction!(rms::rms_rows_kernel_rs, m)?)?;
    m.add_function(wrap_pyfunction!(log_box::smooth_log_box_kernel_rs, m)?)?;
    m.add("__version__", "0.2.0")?;
    Ok(())
}
