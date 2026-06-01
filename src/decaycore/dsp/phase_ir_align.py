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

from typing import Any

import numpy as np

from .phase_ir_utils import _ms_value, _resolve_ir_anchor_mode, _resolve_ir_window_mode, _shift_zeropad


def _anchor_index(ir, anchor_mode: str) -> int:
    x = np.abs(np.asarray(ir, dtype=float).reshape(-1))
    n = int(x.size)
    if n <= 0:
        return 0
    mode = str(anchor_mode or "peak").strip().lower()
    if mode == "centroid":
        w_sum = float(np.sum(x))
        if np.isfinite(w_sum) and w_sum > 1e-30:
            idx = float(np.sum(np.arange(n, dtype=float) * x) / w_sum)
            if np.isfinite(idx):
                return int(np.clip(int(round(idx)), 0, n - 1))
    return int(np.argmax(x))


def _record_alignment_modes(st, *, anchor_mode: str, requested_win_mode: str, win_mode: str) -> None:
    try:
        if isinstance(st, dict):
            st["ir_anchor_mode"] = str(anchor_mode)
            st["ir_window_mode_requested"] = str(requested_win_mode)
            st["ir_window_mode_effective"] = str(win_mode)
    except (TypeError, ValueError):
        pass


def _pre_window_alignment_target(*, anchor_mode: str, is_min_filter: bool, win_mode: str) -> int | None:
    if str(anchor_mode) == "min_causal":
        return None
    if bool(is_min_filter) and str(win_mode) == "auto":
        return 0
    return None


def _min_causal_alignment_target(*, cfg, st, n: int, is_mixed: bool, is_min_filter: bool, win_mode: str) -> int | None:
    if not is_mixed:
        return None
    if bool(is_min_filter) and str(win_mode) == "auto":
        return 0
    min_causal_ms = float(getattr(cfg, "min_causal_ms", 80.0) or 80.0)
    if not np.isfinite(min_causal_ms):
        min_causal_ms = 80.0
    min_causal_ms = float(max(0.0, min_causal_ms))
    target = int(np.clip(int(round(min_causal_ms * float(cfg.fs) / 1000.0)), 0, int(n) - 1))
    try:
        if isinstance(st, dict):
            st["min_causal_ms"] = float(min_causal_ms)
            st["min_causal_target_samples"] = int(target)
    except (TypeError, ValueError):
        pass
    return int(target)


def _mixed_alignment_target(*, cfg, n: int) -> int:
    return int(np.clip(int(round(90.0 * cfg.fs / 1000.0)), 0, int(n) - 1))


def _rew_asym_alignment_target(*, cfg, n: int) -> int:
    left_ms = _ms_value(cfg, "ir_window_ms_left", "ir_window_left", 0.0)
    return int(np.clip(int(round(left_ms * cfg.fs / 1000.0)), 0, int(n) - 1))


def _compute_alignment_target(ir, cfg, st, *, stage: str = "post_window") -> int | None:
    n = int(len(ir))
    if n <= 0:
        return None

    anchor_mode = _resolve_ir_anchor_mode(cfg)
    is_mixed = ("Mixed" in str(getattr(cfg, "filter_type_str", "")))

    requested_win_mode, win_mode, is_min_filter = _resolve_ir_window_mode(cfg, logger=None)
    _record_alignment_modes(
        st,
        anchor_mode=str(anchor_mode),
        requested_win_mode=str(requested_win_mode),
        win_mode=str(win_mode),
    )

    if stage == "pre_window":
        return _pre_window_alignment_target(
            anchor_mode=str(anchor_mode),
            is_min_filter=bool(is_min_filter),
            win_mode=str(win_mode),
        )

    if str(anchor_mode) == "min_causal":
        return _min_causal_alignment_target(
            cfg=cfg,
            st=st,
            n=int(n),
            is_mixed=bool(is_mixed),
            is_min_filter=bool(is_min_filter),
            win_mode=str(win_mode),
        )

    if is_mixed:
        return _mixed_alignment_target(cfg=cfg, n=int(n))

    if str(requested_win_mode).strip().lower() == "rew_asym":
        return _rew_asym_alignment_target(cfg=cfg, n=int(n))

    return None


def _shift_ir(
    ir,
    desired_idx,
    allow_late,
    *,
    st=None,
    key_prefix: str = "align",
    anchor_mode: str = "peak",
) -> tuple[np.ndarray, dict[str, Any]]:
    x = np.asarray(ir, dtype=float)
    n = int(len(x))
    mode = str(anchor_mode or "peak").strip().lower()
    if mode not in ("peak", "centroid", "min_causal"):
        mode = "peak"
    anchor_mode_eff = "peak" if mode == "min_causal" else mode
    if desired_idx is None or n == 0:
        return x, {
            "applied": False,
            "shift_samples": 0,
            "peak_before_samples": 0,
            "peak_after_samples": 0,
            "anchor_before_samples": 0,
            "anchor_after_samples": 0,
            "anchor_mode": str(anchor_mode_eff),
        }

    peak_now = int(np.argmax(np.abs(x)))
    anchor_now = int(_anchor_index(x, anchor_mode_eff))
    shift_samp = int(desired_idx) - anchor_now
    if (not bool(allow_late)) and shift_samp > 0:
        shift_samp = 0
    y = _shift_zeropad(x, shift_samp) if shift_samp != 0 else x
    peak_after = int(np.argmax(np.abs(y))) if n > 0 else 0
    anchor_after = int(_anchor_index(y, anchor_mode_eff)) if n > 0 else 0
    info = {
        "applied": bool(shift_samp != 0),
        "shift_samples": int(shift_samp),
        "peak_before_samples": int(peak_now),
        "peak_after_samples": int(peak_after),
        "anchor_before_samples": int(anchor_now),
        "anchor_after_samples": int(anchor_after),
        "anchor_mode": str(anchor_mode_eff),
        "desired_samples": int(desired_idx),
    }
    try:
        if isinstance(st, dict):
            st[f"{key_prefix}_desired_samples"] = int(desired_idx)
            st[f"{key_prefix}_shift_samples"] = int(shift_samp)
            st[f"{key_prefix}_peak_before_samples"] = int(peak_now)
            st[f"{key_prefix}_peak_after_samples"] = int(peak_after)
            st[f"{key_prefix}_anchor_before_samples"] = int(anchor_now)
            st[f"{key_prefix}_anchor_after_samples"] = int(anchor_after)
            st[f"{key_prefix}_anchor_mode"] = str(anchor_mode_eff)
    except (TypeError, ValueError):
        pass
    return y, info
