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

import io
import zipfile
from typing import Any

import numpy as np
import scipy.io.wavfile

from ..config.decaycore_convolver_configs import (
    bypass_filter_wav_export_spec,
    bypass_sub_filter_wav_export_spec,
)

BYPASS_EXPORT_DIR = "bypass"


def bypass_zip_path(filename: str) -> str:
    return f"{BYPASS_EXPORT_DIR}/{filename}"


def _bypass_ir_like(reference_ir: Any) -> np.ndarray:
    ref = np.asarray(reference_ir, dtype=float).reshape(-1)
    n = int(ref.size)
    bypass = np.zeros(max(1, n), dtype=np.float32)
    if n <= 0:
        bypass[0] = 1.0
        return bypass
    finite_abs = np.where(np.isfinite(ref), np.abs(ref), 0.0)
    peak_idx = int(np.argmax(finite_abs)) if np.any(finite_abs > 0.0) else 0
    peak_idx = min(max(0, peak_idx), n - 1)
    bypass[peak_idx] = 1.0
    return bypass


def _convert_bypass_ir(ir: np.ndarray, fmt: str) -> np.ndarray:
    if fmt == "S32_LE":
        return (np.clip(ir, -1.0, 1.0) * 2147483647).astype(np.int32)
    if fmt == "S16_LE":
        return (np.clip(ir, -1.0, 1.0) * 32767).astype(np.int16)
    return ir.astype(np.float32)


def write_bypass_fir_wavs(
    zf: zipfile.ZipFile,
    *,
    result: Any,
    fs: int,
    ft_short: str,
    file_ts: str,
    irw_tag: str = "auto",
    target_curve_tag: str = "",
    layout: str | None = "Mono",
    wav_fmt: str = "FLOAT32",
) -> None:
    spec = bypass_filter_wav_export_spec(
        fs,
        ft_short,
        file_ts,
        irw_tag=irw_tag,
        target_curve_tag=target_curve_tag,
        layout=layout,
    )
    l_bypass = _bypass_ir_like(getattr(result, "l_ir", []))
    r_bypass = _bypass_ir_like(getattr(result, "r_ir", []))

    if str(spec.get("layout")) == "Stereo":
        stereo_wav = io.BytesIO()
        n = max(int(l_bypass.size), int(r_bypass.size))
        if int(l_bypass.size) != n:
            l_bypass = np.pad(l_bypass, (0, n - int(l_bypass.size)))
        if int(r_bypass.size) != n:
            r_bypass = np.pad(r_bypass, (0, n - int(r_bypass.size)))
        scipy.io.wavfile.write(
            stereo_wav,
            int(fs),
            np.column_stack((_convert_bypass_ir(l_bypass, wav_fmt), _convert_bypass_ir(r_bypass, wav_fmt))),
        )
        zf.writestr(bypass_zip_path(str(spec["bundle_names"][0])), stereo_wav.getvalue())
    else:
        wav_l = io.BytesIO()
        wav_r = io.BytesIO()
        scipy.io.wavfile.write(wav_l, int(fs), _convert_bypass_ir(l_bypass, wav_fmt))
        scipy.io.wavfile.write(wav_r, int(fs), _convert_bypass_ir(r_bypass, wav_fmt))
        zf.writestr(bypass_zip_path(str(spec["bundle_names"][0])), wav_l.getvalue())
        zf.writestr(bypass_zip_path(str(spec["bundle_names"][1])), wav_r.getvalue())

    sub_ir = getattr(result, "sub_ir", None)
    if sub_ir is not None and getattr(sub_ir, "size", 0) > 0:
        wav_sub = io.BytesIO()
        scipy.io.wavfile.write(wav_sub, int(fs), _convert_bypass_ir(_bypass_ir_like(sub_ir), wav_fmt))
        sub_name = str(
            bypass_sub_filter_wav_export_spec(
                fs,
                ft_short,
                file_ts,
                irw_tag=irw_tag,
            )["bundle_name"]
        )
        zf.writestr(bypass_zip_path(sub_name), wav_sub.getvalue())
