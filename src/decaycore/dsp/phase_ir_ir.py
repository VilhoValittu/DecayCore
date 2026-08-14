# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

from __future__ import annotations

import numpy as np
import scipy.fft

try:
    from decaycore_scoring import phase_feedback_ir_batch_rs as _phase_feedback_ir_batch_rs
except (ImportError, OSError):
    _phase_feedback_ir_batch_rs = None


# RustFFT wins for the paired mixed-phase transform on the filter sizes used by
# normal AUTO finalists. SciPy remains faster for a single row and for the very
# large transforms in current measurements, so keep those on pocketfft.
_PRIVATE_PHASE_IR_MIN_ROWS = 2
_PRIVATE_PHASE_IR_MAX_FFT = 65_536


def _build_complex_spectrum(corr_mag, phase_final) -> np.ndarray:
    return np.asarray(corr_mag, dtype=float) * np.exp(1j * np.asarray(phase_final, dtype=float))


def _ifft_to_ir(H, n: int | None = None) -> np.ndarray:
    if n is None:
        n = max(2, int((len(H) - 1) * 2))
    return scipy.fft.irfft(np.asarray(H), n=int(n))


def _phase_batch_to_ir(
    magnitude: np.ndarray,
    phases: np.ndarray,
    *,
    n: int,
) -> np.ndarray:
    mag = np.ascontiguousarray(np.asarray(magnitude, dtype=np.float64).reshape(-1))
    phase_rows = np.ascontiguousarray(np.atleast_2d(np.asarray(phases, dtype=np.float64)))
    n_fft = int(n)
    if (
        _phase_feedback_ir_batch_rs is not None
        and _PRIVATE_PHASE_IR_MIN_ROWS <= int(phase_rows.shape[0]) <= 5
        and n_fft <= _PRIVATE_PHASE_IR_MAX_FFT
        and mag.size == n_fft // 2 + 1
        and phase_rows.shape[1] == mag.size
        and np.all(np.isfinite(mag))
        and np.all(mag >= 0.0)
        and np.all(np.isfinite(phase_rows))
    ):
        try:
            return np.asarray(
                _phase_feedback_ir_batch_rs(mag, phase_rows, n_fft),
                dtype=float,
            )
        except (RuntimeError, TypeError, ValueError, FloatingPointError):
            pass
    return np.asarray(
        [_ifft_to_ir(_build_complex_spectrum(mag, phase), n=n_fft) for phase in phase_rows],
        dtype=float,
    )


def _phase_to_ir(magnitude: np.ndarray, phase: np.ndarray, *, n: int) -> np.ndarray:
    return np.asarray(_phase_batch_to_ir(magnitude, phase, n=n)[0], dtype=float)


def _normalize_ir(ir, cfg) -> np.ndarray:
    if not bool(getattr(cfg, "phase_ir_stage_normalize", False)):
        return np.asarray(ir, dtype=float)
    out = np.asarray(ir, dtype=float).copy()
    mx = float(np.max(np.abs(out))) if out.size else 0.0
    if mx > 0.0:
        out *= 0.89 / mx
    return out
