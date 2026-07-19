from typing import Never

import numpy as np
from numpy.typing import NDArray

Float64Array = NDArray[np.float64]

__version__: str

def smooth_mag_core_rs(
    freqs: Float64Array,
    mags: Float64Array,
    log_freqs: Float64Array,
    window: Float64Array,
    pad_len: int,
) -> Float64Array: ...
def apply_afdw_stack_rs(freqs: Float64Array, mags: Float64Array) -> Never: ...
def gd_smooth_loop_rs(
    gd_l: Float64Array,
    log2f: Float64Array,
    lim_arr: Float64Array,
    curv_lim_arr: Float64Array,
    sigma: float,
) -> Float64Array: ...
def slope_passes_rs(g: Float64Array, x: Float64Array, max_db_per_oct: float) -> Float64Array: ...
def slope_passes_asym_rs(
    g: Float64Array,
    x: Float64Array,
    boost: float,
    cut: float,
) -> Float64Array: ...
def rms_rows_kernel_rs(arr: Float64Array) -> Float64Array: ...
def smooth_log_box_kernel_rs(x: Float64Array, y_raw: Float64Array, half: float) -> Float64Array: ...
