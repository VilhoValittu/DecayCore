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
import time
import zipfile
import numpy as np
import scipy.io.wavfile

from ..app_paths import safe_filters_dir
from ..config.decaycore_convolver_configs import (
    filter_wav_export_spec,
    generate_bypass_hlc_config,
    generate_raspberry_yaml,
    sub_filter_wav_export_spec,
)
from ..config.results import FilterResult
from ..io.bypass_fir_export import bypass_zip_path, write_bypass_fir_wavs
from .export_outputs import (
    _camilladsp_yaml_name,
    _direct_dac_yaml_export_settings,
    _export_version_tag,
    _hybrid_iir_biquads_from_result,
    _write_fs_outputs,
)
from .export_scoring import _build_export_ranking

logger = logging.getLogger("DecayCore")

_VALID_FILTER_WAV_FORMATS = ("FLOAT32", "S32_LE", "S16_LE")


def _convert_ir_for_export(ir: np.ndarray, fmt: str) -> np.ndarray:
    if fmt == "S32_LE":
        return (np.clip(ir, -1.0, 1.0) * 2147483647).astype(np.int32)
    if fmt == "S16_LE":
        return (np.clip(ir, -1.0, 1.0) * 32767).astype(np.int16)
    return ir.astype(np.float32)


def build_export_zip(
    *,
    data: dict,
    results: list[FilterResult],
    ft_short: str,
    file_ts: str,
    irw_tag: str = "auto",
) -> tuple[io.BytesIO, dict]:
    """Build full export ZIP from pipeline results.

    Returns:
    - zip_buffer: in-memory ZIP payload
    - perf: {"zip_png_s": float, "per_fs_stats": {fs: {"zip_png_s": float}}}

    """
    zip_buffer = io.BytesIO()
    perf = {"zip_png_s": 0.0, "per_fs_stats": {}}
    target_curve_tag = str(data.get("target_curve_tag", "") or "").strip()
    multi_rate_on = bool(data.get("multi_rate_opt", False))
    ranking_context = _build_export_ranking(results)

    wav_fmt = str(data.get("filter_wav_format", "FLOAT32") or "FLOAT32").upper()
    if wav_fmt not in _VALID_FILTER_WAV_FORMATS:
        wav_fmt = "FLOAT32"
    device_fmt = str(data.get("device_audio_format", "S32_LE") or "S32_LE").upper()
    if device_fmt not in ("S32_LE", "S16_LE"):
        device_fmt = "S32_LE"

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for result in list(results or []):
            fs_v = int(result.fs)
            t0 = time.perf_counter()
            spec = filter_wav_export_spec(
                fs_v,
                ft_short,
                file_ts,
                irw_tag=irw_tag,
                target_curve_tag=target_curve_tag,
                layout=data.get("layout", "Mono"),
            )

            if str(spec.get("layout")) == "Stereo":
                stereo_wav = io.BytesIO()
                stereo_ir = np.column_stack(
                    (
                        _convert_ir_for_export(result.l_ir, wav_fmt),
                        _convert_ir_for_export(result.r_ir, wav_fmt),
                    )
                )
                scipy.io.wavfile.write(stereo_wav, fs_v, stereo_ir)
                zf.writestr(str(spec["bundle_names"][0]), stereo_wav.getvalue())
            else:
                wav_l = io.BytesIO()
                wav_r = io.BytesIO()
                scipy.io.wavfile.write(wav_l, fs_v, _convert_ir_for_export(result.l_ir, wav_fmt))
                scipy.io.wavfile.write(wav_r, fs_v, _convert_ir_for_export(result.r_ir, wav_fmt))
                zf.writestr(str(spec["bundle_names"][0]), wav_l.getvalue())
                zf.writestr(str(spec["bundle_names"][1]), wav_r.getvalue())

            if getattr(result, "sub_ir", None) is not None and result.sub_ir.size > 0:
                wav_sub = io.BytesIO()
                scipy.io.wavfile.write(wav_sub, fs_v, _convert_ir_for_export(result.sub_ir, wav_fmt))
                sub_name = str(
                    sub_filter_wav_export_spec(
                        fs_v,
                        ft_short,
                        file_ts,
                        irw_tag=irw_tag,
                    )["bundle_name"]
                )
                zf.writestr(sub_name, wav_sub.getvalue())

            write_bypass_fir_wavs(
                zf,
                result=result,
                fs=fs_v,
                ft_short=ft_short,
                file_ts=file_ts,
                irw_tag=irw_tag,
                target_curve_tag=target_curve_tag,
                layout=data.get("layout", "Mono"),
                wav_fmt=wav_fmt,
            )
            zf.writestr(
                bypass_zip_path(f"Bypass_Config_{ft_short}_{fs_v}Hz_{irw_tag}.cfg"),
                generate_bypass_hlc_config(
                    fs_v,
                    ft_short,
                    file_ts,
                    irw_tag=irw_tag,
                    target_curve_tag=target_curve_tag,
                    layout=data.get("layout", "Mono"),
                ),
            )

            _write_fs_outputs(
                zf,
                data,
                fs_v,
                ft_short,
                file_ts,
                None,
                None,
                None,
                None,
                {},
                None,
                None,
                None,
                None,
                {},
                result=result,
                irw_tag=irw_tag,
                ranking_context=ranking_context,
            )

            dt = max(0.0, float(time.perf_counter() - t0))
            perf["zip_png_s"] = float(perf.get("zip_png_s", 0.0)) + dt
            slot = perf["per_fs_stats"].setdefault(int(fs_v), {})
            slot["zip_png_s"] = float(slot.get("zip_png_s", 0.0)) + dt

        if multi_rate_on:
            include_sub = bool(results) and all(
                getattr(getattr(result, "sub_ir", None), "size", 0) > 0 for result in list(results or [])
            )
            yaml_settings = _direct_dac_yaml_export_settings(
                data,
                include_sub=include_sub,
                result_taps=int(getattr(results[0], "taps", 0) or 0) if results else None,
            )
            yaml_content = generate_raspberry_yaml(
                int(data.get("fs") or 44100),
                ft_short,
                file_ts,
                master_gain_db=0.0,
                irw_tag=irw_tag,
                target_curve_tag=target_curve_tag,
                layout=data.get("layout", "Mono"),
                program_version=str(data.get("program_version", "") or "").strip(),
                include_sub=bool(yaml_settings.get("include_sub", False)),
                sub_allpass_freq_hz=yaml_settings.get("sub_allpass_freq_hz"),
                sub_allpass_q=yaml_settings.get("sub_allpass_q"),
                sub_delay_ms=yaml_settings.get("sub_delay_ms"),
                sub_polarity_invert=bool(yaml_settings.get("sub_polarity_invert", False)),
                sub_gain_trim_db=yaml_settings.get("sub_gain_trim_db"),
                main_hpf_hz=yaml_settings.get("main_hpf_hz"),
                sub_hpf_hz=yaml_settings.get("sub_hpf_hz"),
                sub_lpf_hz=yaml_settings.get("sub_lpf_hz"),
                main_hpf_order=yaml_settings.get("main_hpf_order"),
                sub_hpf_order=yaml_settings.get("sub_hpf_order"),
                sub_lpf_order=yaml_settings.get("sub_lpf_order"),
                device_format=device_fmt,
                left_iir_biquads=_hybrid_iir_biquads_from_result(results[0] if results else None, "left"),
                right_iir_biquads=_hybrid_iir_biquads_from_result(results[0] if results else None, "right"),
            )
            zf.writestr(
                _camilladsp_yaml_name(data=data, ft_short=ft_short, irw_tag=irw_tag),
                yaml_content,
            )

    return zip_buffer, perf


def save_export_bundle(
    zip_buffer: io.BytesIO,
    *,
    data: dict | None = None,
    ft_short: str,
    irw_tag: str,
    target_curve_tag: str,
    ts: str,
    output_dir: str | None = None,
    program_version: str | None = None,
) -> tuple[str, str, str]:
    ver_tag = _export_version_tag(data, program_version=program_version)
    filters_dir = safe_filters_dir(output_dir, program_version=program_version)
    logger.info(f"Export filters directory: {filters_dir}")
    parts = ["DecayCore", str(ft_short), str(irw_tag), str(target_curve_tag), str(ver_tag), str(ts)]
    fname = "_".join(parts) + ".zip"
    out_path = os.path.join(filters_dir, fname)

    try:
        with open(out_path, "wb") as f:
            f.write(zip_buffer.getvalue())
        save_msg = f"Saved: {os.path.abspath(out_path)}"
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
        save_msg = "Zip saving failed."
    return fname, os.path.abspath(filters_dir), save_msg
