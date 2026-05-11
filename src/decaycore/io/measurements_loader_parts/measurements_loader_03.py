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
import json
import logging
import os

import numpy as np

logger = logging.getLogger("DecayCore")
import scipy.io.wavfile

from ...auto_mode.shared import (
    AUTO_MODE_BASS_INTEGRATION_GUARD_HI_RATIO,
    AUTO_MODE_BASS_INTEGRATION_GUARD_LO_RATIO,
)
from ...common.measurement_features import normalize_rt60_bands, normalize_rt60_value
from ...dsp.bass_integration import (
    build_combined_sub_transfer,
    compute_bass_integration_diagnostics,
    compute_direct_dac_bass_integration_diagnostics,
    normalize_sub_combine_mode,
    sum_complex_responses,
)
from ..measurement_bundle import BassIntegrationBundle, TransferData
from ..measurements_txt import parse_measurements_from_path as parse_txt_path
from ..measurements_txt import parse_measurements_from_bytes as parse_txt_bytes
from ..measurements_wav import (
    detect_coherent_anchor_sample_from_wav_bytes,
    detect_coherent_anchor_sample_from_wav_path,
    parse_coherent_transfer_from_wav_bytes,
    parse_coherent_transfer_from_wav_path,
    parse_measurements_from_wav_bytes,
    parse_measurements_from_wav_path,
)

def _load_raw_wav_from_source(file_dict=None, local_path: str = "") -> tuple:
    """
    Lataa raaka WAV-data (ilman ikkunointia/tasoitusta) ylöslatauksesta tai
    paikallisesta tiedostosta. Palauttaa (ir_array, sample_rate) tai (None, 0).

    ir_array on float32-vektori (kanava 0), DC poistettu.
    """
    try:
        content = None
        if file_dict is not None:
            if not isinstance(file_dict, dict):
                return None, 0
            name = str(file_dict.get("filename", "") or "")
            raw = file_dict.get("content", None)
            if raw is None:
                return None, 0
            ext = os.path.splitext(name)[1].lower()
            if ext != ".wav":
                # Tarkista RIFF-header
                if not (isinstance(raw, (bytes, bytearray)) and len(raw) >= 4 and raw[:4] == b"RIFF"):
                    return None, 0
            content = raw
        elif local_path:
            lp = _clean_local_path(local_path)
            if not lp:
                return None, 0
            ext = os.path.splitext(lp)[1].lower()
            if ext != ".wav":
                return None, 0
            with open(lp, "rb") as f:
                content = f.read()
        else:
            return None, 0

        fs, data_raw = scipy.io.wavfile.read(io.BytesIO(bytes(content)))
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
        x = x - float(np.mean(x))
        return x, int(fs)
    except Exception:
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
    """
    Lataa vasemman ja oikean kanavan raaka WAV IR -datat (ilman ikkunointia)
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
    """
    Lataa sub-kanavan raaka WAV IR -data bass integration -sloteista
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

def load_bass_integration_measurements(data: dict, *, logger=None):
    """
    Load decomposed bass-integration WAV measurements and build complex predicted totals.

    Predicted left/right totals always use the summed sub field, so each main
    channel is evaluated against `sub_1 + sub_2` rather than only its matching
    sub slot.

    Returns a 7-value tuple:
    `(bundle, f_l, m_l, p_l, f_r, m_r, p_r)`.
    """
    bi_mode = str(
        data.get("bass_integration_mode", "avr_lfe_main_decomposed") or "avr_lfe_main_decomposed"
    ).strip().lower()
    is_direct_dac = bi_mode == "direct_dac"

    pre_ms, post_ms, sl = _get_wav_window_params(data)
    anchor_sample = _detect_shared_coherent_anchor_sample(data, logger=logger)
    l_main = _load_coherent_transfer_slot(
        data,
        file_key="file_l_main",
        path_key="local_path_l_main",
        label="L main only",
        pre_ms=pre_ms,
        post_ms=post_ms,
        smoothing_level=sl,
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
        smoothing_level=sl,
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
        smoothing_level=sl,
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
        smoothing_level=sl,
        anchor_sample=anchor_sample,
        logger=logger,
    )
    if any(v is None for v in (l_main, r_main, l_sub)):
        return None, None, None, None, None, None, None
    if r_sub is None:
        if r_sub_source_present:
            return None, None, None, None, None, None, None
        r_sub = _silent_transfer_like(l_sub, label="R sub absent")

    if any(v is None for v in (l_main, r_main, l_sub, r_sub)):
        return None, None, None, None, None, None, None

    sample_rates = {
        int(l_main.sample_rate),
        int(r_main.sample_rate),
        int(l_sub.sample_rate),
        int(r_sub.sample_rate),
    }
    if len(sample_rates) != 1:
        if logger:
            logger.error(f"Bass Integration sample rates do not match: {sorted(sample_rates)}")
        return None, None, None, None, None, None, None

    combine_mode = normalize_sub_combine_mode(data.get("bass_integration_sub_combine_mode", "average"))
    _real_subs = [l_sub, r_sub] if r_sub_source_present else [l_sub]
    l_combined_sub, l_combine_diag = build_combined_sub_transfer(
        l_main,
        *_real_subs,
        mode=combine_mode,
        label="L combined sub predicted",
    )
    r_combined_sub, r_combine_diag = build_combined_sub_transfer(
        r_main,
        *_real_subs,
        mode=combine_mode,
        label="R combined sub predicted",
    )
    l_total = sum_complex_responses(l_main, l_combined_sub, label="L total predicted")
    r_total = sum_complex_responses(r_main, r_combined_sub, label="R total predicted")

    try:
        fc_hz = float(data.get("avr_crossover_hz", 80.0) or 80.0)
    except Exception:
        fc_hz = 80.0
    profile = str(data.get("bass_integration_profile", "safe") or "safe").strip().lower()
    try:
        guard_lo_ratio = float(
            data.get(
                "bass_integration_guard_lo_ratio",
                AUTO_MODE_BASS_INTEGRATION_GUARD_LO_RATIO,
            )
            or AUTO_MODE_BASS_INTEGRATION_GUARD_LO_RATIO
        )
    except Exception:
        guard_lo_ratio = AUTO_MODE_BASS_INTEGRATION_GUARD_LO_RATIO
    try:
        guard_hi_ratio = float(
            data.get(
                "bass_integration_guard_hi_ratio",
                AUTO_MODE_BASS_INTEGRATION_GUARD_HI_RATIO,
            )
            or AUTO_MODE_BASS_INTEGRATION_GUARD_HI_RATIO
        )
    except Exception:
        guard_hi_ratio = AUTO_MODE_BASS_INTEGRATION_GUARD_HI_RATIO

    base_diagnostics = {
        "sub_slots_present": ["l_sub", "r_sub"] if r_sub_source_present else ["l_sub"],
        "sub_combine_mode": str(combine_mode),
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
            l_combine_diag.get("alignment_offset_ms", r_combine_diag.get("alignment_offset_ms", 0.0)) or 0.0
        ),
        "alignment_confidence": float(
            max(
                float(l_combine_diag.get("alignment_confidence", 0.0) or 0.0),
                float(r_combine_diag.get("alignment_confidence", 0.0) or 0.0),
            )
        ),
    }
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
    if is_direct_dac:
        try:
            xo_order = max(1, int(round(float(data.get("sub_crossover_slope", 24) or 24.0))) // 6)
        except Exception:
            xo_order = 4
        try:
            sub_hpf_hz = float(data.get("sub_hpf_freq", 20.0) or 20.0)
        except Exception:
            sub_hpf_hz = 20.0
        try:
            sub_hpf_order = max(1, int(round(float(data.get("sub_hpf_slope", 12) or 12.0))) // 6)
        except Exception:
            sub_hpf_order = 2
        try:
            sub_delay_ms = float(data.get("bass_integration_sub_delay_ms", 0.0) or 0.0)
        except Exception:
            sub_delay_ms = 0.0
        try:
            sub_gain_trim_db = float(data.get("bass_integration_sub_gain_trim_db", 0.0) or 0.0)
        except Exception:
            sub_gain_trim_db = 0.0
        diagnostics = compute_direct_dac_bass_integration_diagnostics(
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
    else:
        diagnostics = compute_bass_integration_diagnostics(
            bundle_base,
            float(fc_hz),
            profile,
            sub_combine_mode=combine_mode,
            guard_lo_ratio=float(guard_lo_ratio),
            guard_hi_ratio=float(guard_hi_ratio),
        )
    bundle = BassIntegrationBundle(
        l_main=l_main,
        r_main=r_main,
        l_sub=l_sub,
        r_sub=r_sub,
        l_total=l_total,
        r_total=r_total,
        avr_crossover_hz=float(fc_hz),
        profile=profile,
        diagnostics={**base_diagnostics, **dict(diagnostics or {})},
    )
    if is_direct_dac:
        return (
            bundle,
            l_main.freqs_hz,
            l_main.mag_db,
            l_main.phase_deg,
            r_main.freqs_hz,
            r_main.mag_db,
            r_main.phase_deg,
        )
    return (
        bundle,
        l_total.freqs_hz,
        l_total.mag_db,
        l_total.phase_deg,
        r_total.freqs_hz,
        r_total.mag_db,
        r_total.phase_deg,
    )


__all__ = ['_load_raw_wav_from_source', 'load_raw_irs_lr', 'load_raw_ir_sub', 'load_bass_integration_measurements']


def _load_sibling_symbols() -> None:
    import importlib
    package = __package__
    for module_name in ['measurements_loader_01', 'measurements_loader_02', 'measurements_loader_03']:
        if module_name == __name__.rsplit('.', 1)[-1]:
            continue
        module = importlib.import_module(f"{package}.{module_name}")
        for symbol in getattr(module, "__all__", ()):
            globals().setdefault(symbol, getattr(module, symbol))


_load_sibling_symbols()
