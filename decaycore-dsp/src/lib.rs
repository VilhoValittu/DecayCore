// DecayCore DSP extension — Rust/PyO3 0.22
// Selective Rust hotpath acceleration: smoothing, phase/IR stage

mod utils;
mod smoothing;
mod gd_limit;

use pyo3::prelude::*;
use pyo3::types::PyModule;

use pyo3::wrap_pyfunction;

#[pymodule]
fn decaycore_dsp(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(smoothing::smooth_mag_core_rs, m)?)?;
    m.add_function(wrap_pyfunction!(smoothing::apply_afdw_stack_rs, m)?)?;
    m.add_function(wrap_pyfunction!(gd_limit::gd_smooth_loop_rs, m)?)?;
    m.add("__version__", "0.1.0")?;
    Ok(())
}
