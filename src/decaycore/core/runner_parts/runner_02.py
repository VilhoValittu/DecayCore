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
import json
import logging
import math
import os
import shutil
import subprocess
import tempfile
import time
import traceback
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from ...application.run_request import RunRequest
from ...auto_mode.api import AUTO_MODE_COMPAT_VERSION
from ...auto_mode.rank_score import attach_official_rank_score, official_rank_score
from ...config.decaycore_config import load_config
from ...version import VERSION
from ...workflow.auto_flow import _run_auto_mode_search_if_needed, _run_auto_mode_seed_phases
from ...workflow.pipeline_flow import _run_pipeline
from ...workflow.process_run_flow import ProcessRunSupport
from ...workflow.process_support import (
    auto_target_mode_norm,
    auto_target_selection_method_text,
    has_uploaded_target_file,
    pick_target_curve_label,
    slugify_filename_token,
)
from ...workflow.run_finalize import _finalize_run_outputs
from ...workflow.run_prepare import _prepare_target_curve_and_run_context, _prepare_ui_and_measurements

logger = logging.getLogger("DecayCore")
























































__all__ = [
    "ConsoleProgressSink",
    "ProgressSink",
    "prepare_headless_config",
    "run_batch",
]

def _headless_camilladsp_yaml_name(*, data: dict | None, ft_short: str, irw_tag: str, fs_v: int | None = None) -> str:
    parts = ["camilladsp", str(ft_short)]
    if fs_v is not None:
        parts.append(f"{int(fs_v)}Hz")
    parts.append(str(irw_tag))
    parts.append(_safe_filename_token((data or {}).get("program_version", VERSION), default="v0"))
    rank = _headless_winner_rank_score(data)
    if math.isfinite(rank):
        parts.append(f"rank{rank:.3f}")
    return "_".join(parts) + ".yml"

def _headless_summary_content(data: dict, result: Any) -> str:
    l_st = dict(getattr(result, "l_st", {}) or {})
    r_st = dict(getattr(result, "r_st", {}) or {})
    auto_meta = dict(data.get("_auto_mode_meta", {}) or {})
    best_metrics = attach_official_rank_score(auto_meta.get("best_metrics", {}) or {})
    rank = official_rank_score(best_metrics)
    lines = [
        "DecayCore headless export summary",
        f"Version: {data.get('program_version', VERSION)}",
        f"Mode: {data.get('mode', '')}",
        f"Sample rate: {int(getattr(result, 'fs', 0) or 0)} Hz",
        f"Taps: {int(getattr(result, 'taps', data.get('taps', 0)) or 0)}",
        f"Filter type: {data.get('filter_type', '')}",
        f"Selected house curve: {data.get('hc_mode', '')}",
    ]
    if math.isfinite(float(rank)):
        lines.append(f"Auto score: {float(rank):.6f}")
    for side, st in (("Left", l_st), ("Right", r_st)):
        score = _f(st.get("score"), None)
        boost = _f(st.get("net_boost_peak_db", st.get("max_net_boost_db")), None)
        gd = _f(st.get("gd_abs_max_20_500_ms", st.get("gd_grad_limiter_after_max_ms_per_oct")), None)
        lines.append(f"{side} score: {'n/a' if score is None else f'{score:.6f}'}")
        lines.append(f"{side} max boost: {'n/a' if boost is None else f'{boost:.3f} dB'}")
        lines.append(f"{side} GD max: {'n/a' if gd is None else f'{gd:.3f} ms'}")
    return "\n".join(lines) + "\n"

def _build_headless_export_zip(
    *,
    data: dict,
    results: list[Any],
    ft_short: str,
    file_ts: str,
    irw_tag: str = "auto",
    write_dashboards: bool = False,
    dash_fs: int | None = None,
) -> tuple[io.BytesIO, dict, dict]:
    import scipy.io.wavfile

    from ...config.decaycore_convolver_configs import (
        filter_wav_export_spec,
        generate_hlc_config,
        generate_raspberry_yaml,
        sub_filter_wav_export_spec,
    )

    zip_buffer = io.BytesIO()
    perf = {"zip_png_s": 0.0, "per_fs_stats": {}}
    target_curve_tag = str(data.get("target_curve_tag", "") or "").strip()
    multi_rate_on = bool(data.get("multi_rate_opt", False))

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for result in list(results or []):
            fs_v = int(getattr(result, "fs", data.get("fs", 44100)) or 44100)
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
                stereo_ir = np.column_stack((result.l_ir.astype("float32"), result.r_ir.astype("float32")))
                scipy.io.wavfile.write(stereo_wav, fs_v, stereo_ir)
                zf.writestr(str(spec["bundle_names"][0]), stereo_wav.getvalue())
            else:
                wav_l = io.BytesIO()
                wav_r = io.BytesIO()
                scipy.io.wavfile.write(wav_l, fs_v, result.l_ir.astype("float32"))
                scipy.io.wavfile.write(wav_r, fs_v, result.r_ir.astype("float32"))
                zf.writestr(str(spec["bundle_names"][0]), wav_l.getvalue())
                zf.writestr(str(spec["bundle_names"][1]), wav_r.getvalue())

            sub_ir = getattr(result, "sub_ir", None)
            include_sub = bool(sub_ir is not None and getattr(sub_ir, "size", 0) > 0)
            if include_sub:
                wav_sub = io.BytesIO()
                scipy.io.wavfile.write(wav_sub, fs_v, sub_ir.astype("float32"))
                sub_name = str(sub_filter_wav_export_spec(fs_v, ft_short, file_ts, irw_tag=irw_tag)["bundle_name"])
                zf.writestr(sub_name, wav_sub.getvalue())

            zf.writestr(f"Summary_{ft_short}_{fs_v}Hz_{file_ts}.txt", _headless_summary_content(data, result))
            zf.writestr(
                f"Config_{ft_short}_{fs_v}Hz_{irw_tag}.cfg",
                generate_hlc_config(
                    fs_v,
                    ft_short,
                    file_ts,
                    irw_tag=irw_tag,
                    target_curve_tag=target_curve_tag,
                    layout=data.get("layout", "Mono"),
                ),
            )
            if not multi_rate_on:
                zf.writestr(
                    _headless_camilladsp_yaml_name(data=data, ft_short=ft_short, irw_tag=irw_tag, fs_v=fs_v),
                    generate_raspberry_yaml(
                        fs_v,
                        ft_short,
                        file_ts,
                        master_gain_db=0.0,
                        irw_tag=irw_tag,
                        target_curve_tag=target_curve_tag,
                        layout=data.get("layout", "Mono"),
                        program_version=str(data.get("program_version", VERSION) or VERSION),
                        winner_rank_score=_headless_winner_rank_score(data),
                        include_sub=include_sub,
                        sub_allpass_freq_hz=data.get("bass_integration_allpass_freq_hz"),
                        sub_allpass_q=data.get("bass_integration_allpass_q"),
                        sub_delay_ms=data.get("bass_integration_sub_delay_ms"),
                        sub_polarity_invert=bool(data.get("bass_integration_sub_polarity_invert", False)),
                        sub_gain_trim_db=data.get("bass_integration_sub_gain_trim_db"),
                    ),
                )

        if multi_rate_on:
            include_sub_multi = bool(results) and all(
                getattr(getattr(result, "sub_ir", None), "size", 0) > 0 for result in list(results or [])
            )
            zf.writestr(
                _headless_camilladsp_yaml_name(data=data, ft_short=ft_short, irw_tag=irw_tag),
                generate_raspberry_yaml(
                    int(data.get("fs") or 44100),
                    ft_short,
                    file_ts,
                    master_gain_db=0.0,
                    irw_tag=irw_tag,
                    target_curve_tag=target_curve_tag,
                    layout=data.get("layout", "Mono"),
                    program_version=str(data.get("program_version", VERSION) or VERSION),
                    winner_rank_score=_headless_winner_rank_score(data),
                    include_sub=include_sub_multi,
                    sub_allpass_freq_hz=data.get("bass_integration_allpass_freq_hz"),
                    sub_allpass_q=data.get("bass_integration_allpass_q"),
                ),
            )

    return zip_buffer, {}, perf

def _save_headless_export_bundle(
    zip_buffer: io.BytesIO,
    *,
    output_dir: Path,
    data: dict | None = None,
    ft_short: str,
    irw_tag: str,
    target_curve_tag: str,
    ts: str,
    program_version: str | None = None,
) -> tuple[str, str, str]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ver_tag = _safe_filename_token(program_version or (data or {}).get("program_version", VERSION), default="v0")
    rank = _headless_winner_rank_score(data)
    parts = ["DecayCore", str(ft_short), str(irw_tag), str(target_curve_tag or "").strip(), str(ver_tag)]
    if math.isfinite(rank):
        parts.append(f"rank{rank:.3f}")
    parts.append(str(ts))
    fname = "_".join([p for p in parts if p]) + ".zip"
    path = out_dir / fname
    path.write_bytes(zip_buffer.getvalue())
    return fname, str(out_dir.resolve()), f"Saved: {path.resolve()}"

def _read_json(path: Path) -> dict:
    with Path(path).open("r", encoding="utf-8-sig") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return data

def _resolve_path(value: Any, base_dir: Path) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    path = Path(text)
    if not path.is_absolute():
        path = base_dir / path
    return str(path.resolve())

def _first_existing(base_dir: Path, names: list[str]) -> str:
    for name in names:
        path = base_dir / name
        if path.exists():
            return str(path.resolve())
    return ""

def _normalize_headless_config(config: dict, *, config_dir: Path, output_dir: Path) -> dict:
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory(prefix="camillafir-headless-defaults-") as tmp:
        try:
            os.chdir(tmp)
            data = load_config()
        finally:
            os.chdir(cwd)
    data.update(dict(config or {}))
    data["_headless"] = True
    data["mode"] = str(config.get("mode", data.get("mode", "auto")) or "auto").strip().upper()
    if data["mode"] == "AUTO":
        data["camillafir_automatic_mode"] = True
    data["program_version"] = str(VERSION)
    data["auto_mode_compat_version"] = str(AUTO_MODE_COMPAT_VERSION)
    data["output_dir"] = str(output_dir)

    if "sample_rate" in config and "fs" not in config:
        data["fs"] = int(config.get("sample_rate") or data.get("fs", 48000))
    if "house_curve" in config and "hc_mode" not in config:
        data["hc_mode"] = str(config.get("house_curve") or data.get("hc_mode", "Harman10"))
    if "correction_min_hz" in config and "mag_c_min" not in config:
        data["mag_c_min"] = float(config.get("correction_min_hz") or data.get("mag_c_min", 20.0))
    if "correction_max_hz" in config and "mag_c_max" not in config:
        data["mag_c_max"] = float(config.get("correction_max_hz") or data.get("mag_c_max", 250.0))
    if ("house_curve" in config or "hc_mode" in config) and "auto_target_mode" not in config:
        data["auto_target_mode"] = "selected"

    left = config.get("left_wav", config.get("local_path_l", ""))
    right = config.get("right_wav", config.get("local_path_r", ""))
    data["local_path_l"] = _resolve_path(left, config_dir) or _first_existing(config_dir, ["L.wav", "left.wav"])
    data["local_path_r"] = _resolve_path(right, config_dir) or _first_existing(config_dir, ["R.wav", "right.wav"])

    bi = config.get("bass_integration", {})
    if isinstance(bi, dict) and bool(bi.get("enabled", False)):
        data["bass_integration_enable"] = True
        data["bass_integration_mode"] = str(bi.get("mode", data.get("bass_integration_mode", "avr_lfe_main_decomposed")) or "avr_lfe_main_decomposed")
        data["local_path_l_main"] = _resolve_path(bi.get("main_L", bi.get("left_main", "")), config_dir) or _first_existing(config_dir, ["main_L.wav"])
        data["local_path_r_main"] = _resolve_path(bi.get("main_R", bi.get("right_main", "")), config_dir) or _first_existing(config_dir, ["main_R.wav"])
        data["local_path_l_sub"] = _resolve_path(bi.get("sub_L", bi.get("left_sub", "")), config_dir) or _first_existing(config_dir, ["sub_L.wav"])
        data["local_path_r_sub"] = _resolve_path(bi.get("sub_R", bi.get("right_sub", "")), config_dir) or _first_existing(config_dir, ["sub_R.wav"])
        if data["local_path_l_main"]:
            data["local_path_l"] = data["local_path_l_main"]
        if data["local_path_r_main"]:
            data["local_path_r"] = data["local_path_r_main"]

    for key in ("local_path_l", "local_path_r", "local_path_l_main", "local_path_r_main", "local_path_l_sub", "local_path_r_sub", "local_path_house"):
        if key in data:
            data[key] = _resolve_path(data.get(key), config_dir)
    return data


__all__ = ['_headless_camilladsp_yaml_name', '_headless_summary_content', '_build_headless_export_zip', '_save_headless_export_bundle', '_read_json', '_resolve_path', '_first_existing', '_normalize_headless_config']


def _load_sibling_symbols() -> None:
    import importlib
    package = __package__
    for module_name in ['runner_01', 'runner_02', 'runner_03', 'runner_04']:
        if module_name == __name__.rsplit('.', 1)[-1]:
            continue
        module = importlib.import_module(f"{package}.{module_name}")
        for symbol in getattr(module, "__all__", ()):
            globals().setdefault(symbol, getattr(module, symbol))


_load_sibling_symbols()
