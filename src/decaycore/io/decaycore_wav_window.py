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
import logging
import numpy as np

logger = logging.getLogger("DecayCore")

scipy = None  # type: ignore
try:
    import scipy
    import scipy.signal  # noqa: F401
except (
    AttributeError,
    TypeError,
    ValueError,
    KeyError,
    IndexError,
    RuntimeError,
    OSError,
    ImportError,
    ModuleNotFoundError,
    NameError,
):  # pragma: no cover
    scipy = None  # type: ignore

from ..dsp.dsp_utils import choose_fft_len as _next_pow2


def _tukey_window(n: int, alpha: float = 0.25) -> np.ndarray:
    """Sisainen apufunktio: tukey window."""
    n = int(n)
    if n <= 1:
        return np.ones(max(n, 1), dtype=np.float32)

    a = float(alpha)
    a = 0.0 if a < 0.0 else (min(a, 1.0))

    if scipy is not None:
        try:
            return scipy.signal.windows.tukey(n, alpha=a).astype(np.float32)
        except (
            AttributeError,
            TypeError,
            ValueError,
            KeyError,
            IndexError,
            RuntimeError,
            OSError,
            ImportError,
            ModuleNotFoundError,
            NameError,
        ):
            logger.exception("scipy tukey window")

    w = np.ones(n, dtype=np.float32)
    if a <= 0.0:
        return w
    if a >= 1.0:
        return np.hanning(n).astype(np.float32)

    t = np.linspace(0.0, 1.0, n, endpoint=True, dtype=np.float64)
    edge = a / 2.0
    m1 = t < edge
    w[m1] = 0.5 * (1.0 + np.cos(np.pi * (2.0 * t[m1] / a - 1.0)))
    m2 = t > (1.0 - edge)
    w[m2] = 0.5 * (1.0 + np.cos(np.pi * (2.0 * t[m2] / a - 2.0 / a + 1.0)))
    return w.astype(np.float32)


def _octave_smooth_loggrid(freqs_hz: np.ndarray, mags_db: np.ndarray, smoothing_level: int) -> np.ndarray:
    """Sisainen apufunktio: octave smooth loggrid."""
    f = np.asarray(freqs_hz, dtype=float)
    m = np.asarray(mags_db, dtype=float)
    sl = int(smoothing_level)

    if sl <= 0 or f.size != m.size or f.size < 64:
        return m.astype(float)

    mask = f > 0.0
    if np.count_nonzero(mask) < 64:
        return m.astype(float)

    ff = f[mask]
    mm = m[mask]
    logf = np.log10(ff)

    n = ff.size
    grid = np.linspace(logf.min(), logf.max(), n)
    mg = np.interp(grid, logf, mm)

    sigma_pts = max(1.0, (n / 2000.0) * (24.0 / max(float(sl), 1.0)))

    half = int(max(3, round(4.0 * sigma_pts)))
    x = np.arange(-half, half + 1, dtype=float)
    k = np.exp(-0.5 * (x / sigma_pts) ** 2)
    k /= np.sum(k)

    mg_s = np.convolve(mg, k, mode="same")
    mm_s = np.interp(logf, grid, mg_s)

    out = m.copy()
    out[mask] = mm_s
    return out.astype(float)


_WAV_WINDOW_RECOVERABLE_EXC = (
    AttributeError,
    TypeError,
    ValueError,
    KeyError,
    IndexError,
    RuntimeError,
    OSError,
    ImportError,
    ModuleNotFoundError,
    NameError,
)


def _wav_window_extract_peak_segment(sig: np.ndarray, *, fs_i: int, pre_ms: float, post_ms: float) -> np.ndarray:
    peak = int(np.argmax(np.abs(sig)))
    pre_s = max(int(round((float(pre_ms) / 1000.0) * fs_i)), 0)
    post_s = max(int(round((float(post_ms) / 1000.0) * fs_i)), 64)
    i0 = max(0, peak - pre_s)
    i1 = min(sig.size, peak + post_s)
    seg = sig[i0:i1]
    return seg if seg.size >= 64 else sig


def _wav_window_extract_anchor_segment(
    sig: np.ndarray,
    *,
    fs_i: int,
    pre_ms: float,
    post_ms: float,
    anchor_sample: int | None,
) -> np.ndarray:
    if anchor_sample is None:
        return sig.copy()

    pre_s = max(int(round((float(pre_ms) / 1000.0) * fs_i)), 0)
    post_s = max(int(round((float(post_ms) / 1000.0) * fs_i)), 64)
    seg_len = int(max(64, pre_s + post_s))
    seg = np.zeros(seg_len, dtype=np.float32)
    try:
        anchor_i = int(anchor_sample)
    except _WAV_WINDOW_RECOVERABLE_EXC:
        anchor_i = -1
    src_i0 = max(0, anchor_i - pre_s)
    src_i1 = min(int(sig.size), anchor_i + post_s)
    dst_i0 = max(0, pre_s - anchor_i)
    n_copy = max(0, src_i1 - src_i0)
    if n_copy > 0 and dst_i0 < seg.size:
        n_copy = min(int(n_copy), int(seg.size - dst_i0))
        seg[dst_i0 : dst_i0 + n_copy] = sig[src_i0 : src_i0 + n_copy]
    return seg if seg.size >= 64 else sig.copy()


def _wav_window_apply_detrend(seg: np.ndarray, *, detrend: str, logger: logging.Logger, context: str) -> np.ndarray:
    dt = (detrend or "").lower().strip()
    if dt == "linear":
        try:
            return seg - np.linspace(float(seg[0]), float(seg[-1]), seg.size, dtype=np.float32)
        except _WAV_WINDOW_RECOVERABLE_EXC:
            logger.exception(context)
    elif dt == "mean":
        return seg - float(np.mean(seg))
    return seg


def _wav_window_apply_window(
    seg: np.ndarray,
    *,
    window: str,
    tukey_alpha: float,
    logger: logging.Logger,
    context: str,
) -> np.ndarray:
    wtype = (window or "").lower().strip()
    try:
        if wtype == "hann" or wtype == "hanning":
            w = np.hanning(seg.size).astype(np.float32)
        elif wtype == "rect" or wtype == "rectangular" or wtype == "none":
            w = np.ones(seg.size, dtype=np.float32)
        else:
            w = _tukey_window(seg.size, alpha=float(tukey_alpha))
        return seg * w
    except _WAV_WINDOW_RECOVERABLE_EXC:
        logger.exception(context)
        return seg


def _wav_window_resolve_n_fft(
    seg_size: int,
    *,
    zero_pad_pow2: bool,
    min_n_fft: int | None,
    context: str,
) -> int:
    n_fft = int(seg_size)
    if zero_pad_pow2:
        n_fft = _next_pow2(n_fft)
    n_fft = max(n_fft, seg_size)
    if min_n_fft is None:
        return n_fft
    try:
        mn = int(min_n_fft)
        if mn > 0:
            mn = _next_pow2(mn)
            n_fft = max(n_fft, mn)
    except _WAV_WINDOW_RECOVERABLE_EXC:
        logger.exception(context)
    return n_fft


def _wav_window_apply_mag_smoothing(
    freqs: np.ndarray,
    mag_db: np.ndarray,
    *,
    smoothing_level: int | None,
    logger: logging.Logger,
    context: str,
) -> np.ndarray:
    if smoothing_level is None:
        return mag_db
    try:
        sl = int(smoothing_level)
        if sl > 0:
            return _octave_smooth_loggrid(freqs, mag_db, sl)
    except _WAV_WINDOW_RECOVERABLE_EXC:
        logger.exception(context)
    return mag_db


def _wav_window_apply_phase_hf_hold(
    freqs: np.ndarray,
    phase_deg: np.ndarray,
    *,
    spec: np.ndarray | None,
    phase_hf_hold: bool,
    fs_i: int,
    logger: logging.Logger,
    context: str,
) -> tuple[np.ndarray, np.ndarray | None]:
    if not phase_hf_hold:
        return phase_deg, spec
    try:
        hf = freqs > min(0.45 * fs_i, 18000.0)
        if np.any(hf) and np.any(~hf):
            phase_hold_deg = phase_deg[np.where(~hf)[0][-1]]
            phase_deg = np.asarray(phase_deg, dtype=float).copy()
            phase_deg[hf] = phase_hold_deg
            if spec is not None:
                phase_hold_rad = np.deg2rad(float(phase_hold_deg))
                spec = np.asarray(spec, dtype=np.complex128).copy()
                spec[hf] = np.maximum(np.abs(spec[hf]), 1e-12) * np.exp(1j * phase_hold_rad)
    except _WAV_WINDOW_RECOVERABLE_EXC:
        logger.exception(context)
    return phase_deg, spec


def ir_wav_to_freq_response(
    fs: int,
    x: np.ndarray,
    *,
    pre_ms: float = 5.0,
    post_ms: float = 500.0,
    smoothing_level: int | None = None,
    window: str = "tukey",
    tukey_alpha: float = 0.25,
    zero_pad_pow2: bool = True,
    min_n_fft: int | None = None,
    detrend: str = "linear",
    phase_hf_hold: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Funktio: ir wav to freq response."""
    fs_i = int(fs) if fs else 0
    if fs_i <= 0:
        raise ValueError("Invalid WAV sample rate.")

    sig = np.asarray(x, dtype=np.float32).copy()
    if sig.size < 64:
        raise ValueError("WAV too short.")

    sig -= float(np.mean(sig))
    seg = _wav_window_extract_peak_segment(sig, fs_i=fs_i, pre_ms=pre_ms, post_ms=post_ms)
    seg = _wav_window_apply_detrend(seg, detrend=detrend, logger=logger, context="linear detrend")
    seg = _wav_window_apply_window(
        seg, window=window, tukey_alpha=tukey_alpha, logger=logger, context="window apply in wav freq response"
    )
    n_fft = _wav_window_resolve_n_fft(
        seg.size, zero_pad_pow2=zero_pad_pow2, min_n_fft=min_n_fft, context="min_n_fft apply"
    )

    spec = np.fft.rfft(seg, n=n_fft)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / float(fs_i))

    mag = np.abs(spec)
    mag_db = 20.0 * np.log10(np.maximum(mag, 1e-12))

    phase_rad = np.unwrap(np.angle(spec))
    phase_deg = np.rad2deg(phase_rad)
    mag_db = _wav_window_apply_mag_smoothing(
        freqs, mag_db, smoothing_level=smoothing_level, logger=logger, context="octave smoothing in wav freq response"
    )
    phase_deg, _ = _wav_window_apply_phase_hf_hold(
        freqs,
        phase_deg,
        spec=None,
        phase_hf_hold=phase_hf_hold,
        fs_i=fs_i,
        logger=logger,
        context="phase hf hold in wav freq response",
    )

    return freqs.astype(float), mag_db.astype(float), phase_deg.astype(float)


def ir_wav_to_complex_response(
    fs: int,
    x: np.ndarray,
    *,
    pre_ms: float = 5.0,
    post_ms: float = 500.0,
    smoothing_level: int | None = None,
    anchor_sample: int | None = None,
    window: str = "tukey",
    tukey_alpha: float = 0.25,
    zero_pad_pow2: bool = True,
    min_n_fft: int | None = None,
    detrend: str = "linear",
    phase_hf_hold: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return a coherent complex response without per-file peak-centering."""
    fs_i = int(fs) if fs else 0
    if fs_i <= 0:
        raise ValueError("Invalid WAV sample rate.")

    sig = np.asarray(x, dtype=np.float32).copy()
    if sig.size < 64:
        raise ValueError("WAV too short.")

    sig -= float(np.mean(sig))
    seg = _wav_window_extract_anchor_segment(
        sig, fs_i=fs_i, pre_ms=pre_ms, post_ms=post_ms, anchor_sample=anchor_sample
    )
    seg = _wav_window_apply_detrend(seg, detrend=detrend, logger=logger, context="linear detrend in complex response")
    seg = _wav_window_apply_window(
        seg, window=window, tukey_alpha=tukey_alpha, logger=logger, context="window apply in complex response"
    )
    n_fft = _wav_window_resolve_n_fft(
        seg.size, zero_pad_pow2=zero_pad_pow2, min_n_fft=min_n_fft, context="min_n_fft apply in complex response"
    )

    spec = np.fft.rfft(seg, n=n_fft)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / float(fs_i))
    mag_db = 20.0 * np.log10(np.maximum(np.abs(spec), 1e-12))

    phase_rad = np.unwrap(np.angle(spec))
    phase_deg = np.rad2deg(phase_rad)
    mag_db = _wav_window_apply_mag_smoothing(
        freqs, mag_db, smoothing_level=smoothing_level, logger=logger, context="octave smoothing in complex response"
    )
    phase_deg, spec = _wav_window_apply_phase_hf_hold(
        freqs,
        phase_deg,
        spec=spec,
        phase_hf_hold=phase_hf_hold,
        fs_i=fs_i,
        logger=logger,
        context="phase hf hold in complex response",
    )

    return (
        freqs.astype(float),
        np.asarray(spec, dtype=np.complex128),
        mag_db.astype(float),
        phase_deg.astype(float),
    )
