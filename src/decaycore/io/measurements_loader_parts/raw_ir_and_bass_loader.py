from .lr_measurement_loader import _silent_transfer_like
from .measurement_source_helpers import (
    _clean_local_path,
    _detect_coherent_slot_anchor_sample,
    _detect_shared_coherent_anchor_sample,
    _get_local_path,
    _get_uploaded_file,
    _get_wav_window_params,
    _load_coherent_transfer_slot,
)

# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

import io
import logging
import os

import numpy as np

logger = logging.getLogger("DecayCore")
import scipy.io.wavfile

from ...auto_mode.shared import (
    AUTO_MODE_BASS_INTEGRATION_GUARD_HI_RATIO,
    AUTO_MODE_BASS_INTEGRATION_GUARD_LO_RATIO,
)
from ...dsp.bass_integration import (
    build_combined_sub_transfer,
    compute_bass_integration_diagnostics,
    compute_direct_dac_bass_integration_diagnostics,
    normalize_sub_combine_mode,
    prepare_dual_sub_peak_aligned_average,
    sum_complex_responses,
)
from ..measurement_bundle import BassIntegrationBundle, TransferData

_RECOVERABLE_WAV_LOAD_EXCEPTIONS = (
    AttributeError,
    TypeError,
    ValueError,
    OverflowError,
    OSError,
    EOFError,
    RuntimeError,
)


def _wav_content_from_upload(file_dict) -> bytes | bytearray | None:
    if not isinstance(file_dict, dict):
        return None
    name = str(file_dict.get("filename", "") or "")
    raw = file_dict.get("content", None)
    if raw is None:
        return None
    ext = os.path.splitext(name)[1].lower()
    if ext != ".wav":
        is_riff = isinstance(raw, (bytes, bytearray)) and len(raw) >= 4 and raw[:4] == b"RIFF"
        if not is_riff:
            return None
    return raw


def _wav_content_from_local_path(local_path: str) -> bytes | None:
    lp = _clean_local_path(local_path)
    if not lp:
        return None
    ext = os.path.splitext(lp)[1].lower()
    if ext != ".wav":
        return None
    with open(lp, "rb") as f:
        return f.read()


def _wav_to_float_channel(data_raw) -> np.ndarray:
    x = np.asarray(data_raw)
    if x.ndim == 2:
        x = x[:, 0]
    if x.dtype.kind != "f":
        if x.dtype == np.int16:
            x = x.astype(np.float32) / 32768.0
        elif x.dtype == np.int32:
            x = x.astype(np.float32) / 2147483648.0
        else:
            x = x.astype(np.float32)
    else:
        x = x.astype(np.float32, copy=False)
    return np.asarray(x - float(np.mean(x)), dtype=np.float32)


def _load_raw_wav_from_source(file_dict=None, local_path: str = "") -> tuple:
    """Lataa raaka WAV-data (ilman ikkunointia/tasoitusta) ylöslatauksesta tai
    paikallisesta tiedostosta. Palauttaa (ir_array, sample_rate) tai (None, 0).

    ir_array on float32-vektori (kanava 0), DC poistettu.
    """
    try:
        content = None
        if file_dict is not None:
            content = _wav_content_from_upload(file_dict)
            if content is None:
                return None, 0
        elif local_path:
            content = _wav_content_from_local_path(local_path)
            if content is None:
                return None, 0
        else:
            return None, 0

        fs, data_raw = scipy.io.wavfile.read(io.BytesIO(bytes(content)))
        x = _wav_to_float_channel(data_raw)
        return x, int(fs)
    except _RECOVERABLE_WAV_LOAD_EXCEPTIONS:
        return None, 0

def load_raw_irs_lr(
    data: dict,
    *,
    file_key_l: str = "file_l",
    path_key_l: str = "local_path_l",
    file_key_r: str = "file_r",
    path_key_r: str = "local_path_r",
    logger=None,
) -> tuple:
    """Lataa vasemman ja oikean kanavan raaka WAV IR -datat (ilman ikkunointia)
    annetuista L/R-sloteista.

    Palauttaa (raw_ir_l, fs_l, raw_ir_r, fs_r).
    Palauttaa (None, 0, None, 0) jos kumpaakaan ei löydy WAV-muodossa.
    """
    up_l = _get_uploaded_file(data, file_key_l)
    up_r = _get_uploaded_file(data, file_key_r)

    if up_l is not None and up_r is not None:
        ir_l, fs_l = _load_raw_wav_from_source(file_dict=up_l)
        ir_r, fs_r = _load_raw_wav_from_source(file_dict=up_r)
        if ir_l is not None and ir_r is not None:
            return ir_l, fs_l, ir_r, fs_r

    lp_l = _get_local_path(data, path_key_l)
    lp_r = _get_local_path(data, path_key_r)
    if lp_l and lp_r:
        ir_l, fs_l = _load_raw_wav_from_source(local_path=lp_l)
        ir_r, fs_r = _load_raw_wav_from_source(local_path=lp_r)
        if ir_l is not None and ir_r is not None:
            return ir_l, fs_l, ir_r, fs_r

    return None, 0, None, 0

def load_raw_ir_sub(data: dict, *, logger=None) -> tuple:
    """Lataa sub-kanavan raaka WAV IR -data bass integration -sloteista
    (file_l_sub / local_path_l_sub, fallback file_r_sub / local_path_r_sub).

    Palauttaa (raw_ir_sub, fs_sub) tai (None, 0).
    """
    up_ls = _get_uploaded_file(data, "file_l_sub")
    if up_ls is not None:
        ir, fs = _load_raw_wav_from_source(file_dict=up_ls)
        if ir is not None:
            return ir, fs

    lp_ls = _get_local_path(data, "local_path_l_sub")
    if lp_ls:
        ir, fs = _load_raw_wav_from_source(local_path=lp_ls)
        if ir is not None:
            return ir, fs

    up_rs = _get_uploaded_file(data, "file_r_sub")
    if up_rs is not None:
        ir, fs = _load_raw_wav_from_source(file_dict=up_rs)
        if ir is not None:
            return ir, fs

    lp_rs = _get_local_path(data, "local_path_r_sub")
    if lp_rs:
        ir, fs = _load_raw_wav_from_source(local_path=lp_rs)
        if ir is not None:
            return ir, fs

    return None, 0


def _load_bass_integration_transfers(
    data: dict,
    *,
    pre_ms: float,
    post_ms: float,
    smoothing_level: str,
    anchor_sample: int | None,
    logger=None,
):
    l_main = _load_coherent_transfer_slot(
        data,
        file_key="file_l_main",
        path_key="local_path_l_main",
        label="L main only",
        pre_ms=pre_ms,
        post_ms=post_ms,
        smoothing_level=smoothing_level,
        anchor_sample=anchor_sample,
        logger=logger,
    )
    r_main = _load_coherent_transfer_slot(
        data,
        file_key="file_r_main",
        path_key="local_path_r_main",
        label="R main only",
        pre_ms=pre_ms,
        post_ms=post_ms,
        smoothing_level=smoothing_level,
        anchor_sample=anchor_sample,
        logger=logger,
    )
    l_sub = _load_coherent_transfer_slot(
        data,
        file_key="file_l_sub",
        path_key="local_path_l_sub",
        label="L sub only",
        pre_ms=pre_ms,
        post_ms=post_ms,
        smoothing_level=smoothing_level,
        anchor_sample=anchor_sample,
        logger=logger,
    )
    r_sub_source_present = bool(
        _get_uploaded_file(data, "file_r_sub") is not None
        or _get_local_path(data, "local_path_r_sub")
    )
    r_sub = _load_coherent_transfer_slot(
        data,
        file_key="file_r_sub",
        path_key="local_path_r_sub",
        label="R sub only",
        pre_ms=pre_ms,
        post_ms=post_ms,
        smoothing_level=smoothing_level,
        anchor_sample=anchor_sample,
        logger=logger,
    )
    return l_main, r_main, l_sub, r_sub, r_sub_source_present


def _validate_bass_integration_transfers(
    *,
    l_main: TransferData | None,
    r_main: TransferData | None,
    l_sub: TransferData | None,
    r_sub: TransferData | None,
    r_sub_source_present: bool,
) -> tuple[TransferData, TransferData, TransferData, TransferData] | None:
    if any(v is None for v in (l_main, r_main, l_sub)):
        return None
    if r_sub is None:
        if r_sub_source_present:
            return None
        r_sub = _silent_transfer_like(l_sub, label="R sub absent")
    if any(v is None for v in (l_main, r_main, l_sub, r_sub)):
        return None
    return l_main, r_main, l_sub, r_sub


def _shared_sample_rate_or_none(
    *,
    l_main: TransferData,
    r_main: TransferData,
    l_sub: TransferData,
    r_sub: TransferData,
    logger=None,
) -> int | None:
    sample_rates = {
        int(l_main.sample_rate),
        int(r_main.sample_rate),
        int(l_sub.sample_rate),
        int(r_sub.sample_rate),
    }
    if len(sample_rates) == 1:
        return int(next(iter(sample_rates)))
    if logger:
        logger.error(f"Bass Integration sample rates do not match: {sorted(sample_rates)}")
    return None


def _prepare_dual_sub_response_if_needed(
    *,
    data: dict,
    l_sub: TransferData,
    r_sub: TransferData,
    r_sub_source_present: bool,
    combine_mode: str,
    is_direct_dac: bool,
    logger=None,
) -> tuple[TransferData, TransferData, list[TransferData], str, dict]:
    real_subs = [l_sub, r_sub] if r_sub_source_present else [l_sub]
    dual_sub_diag: dict = {}
    if not (is_direct_dac and r_sub_source_present):
        return l_sub, r_sub, real_subs, combine_mode, dual_sub_diag

    sub1_peak = _detect_coherent_slot_anchor_sample(
        data,
        file_key="file_l_sub",
        path_key="local_path_l_sub",
        logger=logger,
    )
    sub2_peak = _detect_coherent_slot_anchor_sample(
        data,
        file_key="file_r_sub",
        path_key="local_path_r_sub",
        logger=logger,
    )
    if sub1_peak is None or sub2_peak is None:
        if logger:
            logger.warning(
                "Bass Integration: 2 subwoofers detected but impulse peak detection failed; using configured sub combine mode."
            )
        return l_sub, r_sub, real_subs, combine_mode, dual_sub_diag

    if logger:
        fs = int(l_sub.sample_rate)
        logger.info("Bass Integration: 2 subwoofers detected.")
        logger.info(
            "Bass Integration: SUB1 impulse peak at %d samples / %.3f ms.",
            int(sub1_peak),
            float(sub1_peak) / float(fs) * 1000.0,
        )
        logger.info(
            "Bass Integration: SUB2 impulse peak at %d samples / %.3f ms.",
            int(sub2_peak),
            float(sub2_peak) / float(fs) * 1000.0,
        )
        logger.info(
            "Bass Integration: applied SUB2 -> SUB1 alignment delay: %d samples / %.3f ms.",
            int(sub1_peak) - int(sub2_peak),
            (float(sub1_peak) - float(sub2_peak)) / float(fs) * 1000.0,
        )
    combined_sub, dual_sub_diag = prepare_dual_sub_peak_aligned_average(
        l_sub,
        r_sub,
        sub1_peak_samples=int(sub1_peak),
        sub2_peak_samples=int(sub2_peak),
        label="Direct-DAC dual-sub peak-aligned vector average",
    )
    dual_sub_diag["dual_sub_original_sub_combine_mode"] = str(combine_mode)
    if logger:
        logger.info("Bass Integration: created vector-averaged combined subwoofer response.")
        logger.info("Bass Integration: using combined subwoofer response for main/sub integration.")
    l_sub = combined_sub
    r_sub = _silent_transfer_like(l_sub, label="Direct-DAC inactive sub slot after dual-sub preprocessing")
    real_subs = [l_sub]
    combine_mode = "average"
    return l_sub, r_sub, real_subs, combine_mode, dual_sub_diag


def _safe_float_from_data(data: dict, key: str, default: float) -> float:
    try:
        return float(data.get(key, default) or default)
    except _RECOVERABLE_WAV_LOAD_EXCEPTIONS:
        return float(default)


def _build_bass_integration_base_diagnostics(
    *,
    l_combine_diag: dict,
    r_combine_diag: dict,
    dual_sub_diag: dict,
    r_sub_source_present: bool,
    combine_mode: str,
) -> dict:
    return {
        "sub_slots_present": ["l_sub"]
        if bool(dual_sub_diag.get("dual_sub_preprocessing_applied", False))
        else (["l_sub", "r_sub"] if r_sub_source_present else ["l_sub"]),
        "sub_combine_mode": "dual_sub_peak_aligned_average"
        if bool(dual_sub_diag.get("dual_sub_preprocessing_applied", False))
        else str(combine_mode),
        "sub_combined_level_delta_db_20_120": float(
            l_combine_diag.get(
                "sub_combined_level_delta_db_20_120",
                r_combine_diag.get("sub_combined_level_delta_db_20_120", 0.0),
            )
            or 0.0
        ),
        "sub_combined_level_delta_db_30_90": float(
            l_combine_diag.get(
                "sub_combined_level_delta_db_30_90",
                r_combine_diag.get("sub_combined_level_delta_db_30_90", 0.0),
            )
            or 0.0
        ),
        "whether_alignment_applied": bool(
            l_combine_diag.get("whether_alignment_applied", False)
            or r_combine_diag.get("whether_alignment_applied", False)
        ),
        "alignment_offset_ms": float(
            l_combine_diag.get("alignment_offset_ms", r_combine_diag.get("alignment_offset_ms", 0.0))
            or 0.0
        ),
        "alignment_confidence": float(
            max(
                float(l_combine_diag.get("alignment_confidence", 0.0) or 0.0),
                float(r_combine_diag.get("alignment_confidence", 0.0) or 0.0),
            )
        ),
        **dict(dual_sub_diag or {}),
    }


def _compute_bass_integration_diagnostics_for_bundle(
    *,
    data: dict,
    bundle_base: BassIntegrationBundle,
    fc_hz: float,
    profile: str,
    combine_mode: str,
    guard_lo_ratio: float,
    guard_hi_ratio: float,
    is_direct_dac: bool,
):
    if is_direct_dac:
        xo_order = max(1, int(round(_safe_float_from_data(data, "sub_crossover_slope", 24.0))) // 6)
        sub_hpf_hz = _safe_float_from_data(data, "sub_hpf_freq", 20.0)
        sub_hpf_order = max(1, int(round(_safe_float_from_data(data, "sub_hpf_slope", 12.0))) // 6)
        sub_delay_ms = _safe_float_from_data(data, "bass_integration_sub_delay_ms", 0.0)
        sub_gain_trim_db = _safe_float_from_data(data, "bass_integration_sub_gain_trim_db", 0.0)
        return compute_direct_dac_bass_integration_diagnostics(
            bundle_base,
            float(fc_hz),
            profile,
            main_hpf_order=int(xo_order),
            sub_lpf_order=int(xo_order),
            sub_hpf_hz=float(sub_hpf_hz),
            sub_hpf_order=int(sub_hpf_order),
            sub_combine_mode=combine_mode,
            sub_delay_ms=float(sub_delay_ms),
            sub_polarity_invert=bool(data.get("bass_integration_sub_polarity_invert", False)),
            sub_gain_trim_db=float(sub_gain_trim_db),
            guard_lo_ratio=float(guard_lo_ratio),
            guard_hi_ratio=float(guard_hi_ratio),
        )
    return compute_bass_integration_diagnostics(
        bundle_base,
        float(fc_hz),
        profile,
        sub_combine_mode=combine_mode,
        guard_lo_ratio=float(guard_lo_ratio),
        guard_hi_ratio=float(guard_hi_ratio),
    )


def load_bass_integration_measurements(data: dict, *, logger=None):
    """Load decomposed bass-integration WAV measurements and build complex predicted totals.

    Predicted left/right totals always use the summed sub field, so each main
    channel is evaluated against `sub_1 + sub_2` rather than only its matching
    sub slot.

    Returns a 7-value tuple:
    `(bundle, f_l, m_l, p_l, f_r, m_r, p_r)`.
    """
    is_direct_dac = True

    pre_ms, post_ms, sl = _get_wav_window_params(data)
    anchor_sample = _detect_shared_coherent_anchor_sample(data, logger=logger)
    l_main, r_main, l_sub, r_sub, r_sub_source_present = _load_bass_integration_transfers(
        data,
        pre_ms=pre_ms,
        post_ms=post_ms,
        smoothing_level=sl,
        anchor_sample=anchor_sample,
        logger=logger,
    )
    validated = _validate_bass_integration_transfers(
        l_main=l_main,
        r_main=r_main,
        l_sub=l_sub,
        r_sub=r_sub,
        r_sub_source_present=r_sub_source_present,
    )
    if validated is None:
        return None, None, None, None, None, None, None
    l_main, r_main, l_sub, r_sub = validated
    if _shared_sample_rate_or_none(
        l_main=l_main,
        r_main=r_main,
        l_sub=l_sub,
        r_sub=r_sub,
        logger=logger,
    ) is None:
        return None, None, None, None, None, None, None

    combine_mode = normalize_sub_combine_mode(data.get("bass_integration_sub_combine_mode", "average"))
    l_sub, r_sub, real_subs, combine_mode, dual_sub_diag = _prepare_dual_sub_response_if_needed(
        data=data,
        l_sub=l_sub,
        r_sub=r_sub,
        r_sub_source_present=r_sub_source_present,
        combine_mode=combine_mode,
        is_direct_dac=is_direct_dac,
        logger=logger,
    )
    l_combined_sub, l_combine_diag = build_combined_sub_transfer(
        l_main,
        *real_subs,
        mode=combine_mode,
        label="L combined sub predicted",
    )
    r_combined_sub, r_combine_diag = build_combined_sub_transfer(
        r_main,
        *real_subs,
        mode=combine_mode,
        label="R combined sub predicted",
    )
    l_total = sum_complex_responses(l_main, l_combined_sub, label="L total predicted")
    r_total = sum_complex_responses(r_main, r_combined_sub, label="R total predicted")

    fc_hz = _safe_float_from_data(data, "avr_crossover_hz", 80.0)
    profile = str(data.get("bass_integration_profile", "safe") or "safe").strip().lower()
    guard_lo_ratio = _safe_float_from_data(
        data,
        "bass_integration_guard_lo_ratio",
        AUTO_MODE_BASS_INTEGRATION_GUARD_LO_RATIO,
    )
    guard_hi_ratio = _safe_float_from_data(
        data,
        "bass_integration_guard_hi_ratio",
        AUTO_MODE_BASS_INTEGRATION_GUARD_HI_RATIO,
    )
    base_diagnostics = _build_bass_integration_base_diagnostics(
        l_combine_diag=dict(l_combine_diag or {}),
        r_combine_diag=dict(r_combine_diag or {}),
        dual_sub_diag=dict(dual_sub_diag or {}),
        r_sub_source_present=bool(r_sub_source_present),
        combine_mode=str(combine_mode),
    )
    bundle_base = BassIntegrationBundle(
        l_main=l_main,
        r_main=r_main,
        l_sub=l_sub,
        r_sub=r_sub,
        l_total=l_total,
        r_total=r_total,
        avr_crossover_hz=float(fc_hz),
        profile=profile,
        diagnostics=base_diagnostics,
    )
    diagnostics = _compute_bass_integration_diagnostics_for_bundle(
        data=data,
        bundle_base=bundle_base,
        fc_hz=float(fc_hz),
        profile=profile,
        combine_mode=str(combine_mode),
        guard_lo_ratio=float(guard_lo_ratio),
        guard_hi_ratio=float(guard_hi_ratio),
        is_direct_dac=bool(is_direct_dac),
    )
    final_diagnostics = {**base_diagnostics, **dict(diagnostics or {})}
    if bool(dual_sub_diag.get("dual_sub_preprocessing_applied", False)):
        final_diagnostics.update(dict(dual_sub_diag or {}))
        final_diagnostics["sub_slots_present"] = ["l_sub"]
        final_diagnostics["sub_combine_mode"] = "dual_sub_peak_aligned_average"
    bundle = BassIntegrationBundle(
        l_main=l_main,
        r_main=r_main,
        l_sub=l_sub,
        r_sub=r_sub,
        l_total=l_total,
        r_total=r_total,
        avr_crossover_hz=float(fc_hz),
        profile=profile,
        diagnostics=final_diagnostics,
    )
    return (
        bundle,
        l_main.freqs_hz,
        l_main.mag_db,
        l_main.phase_deg,
        r_main.freqs_hz,
        r_main.mag_db,
        r_main.phase_deg,
    )


__all__ = ['_load_raw_wav_from_source', 'load_raw_irs_lr', 'load_raw_ir_sub', 'load_bass_integration_measurements']

