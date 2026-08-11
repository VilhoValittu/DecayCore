# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""Target tab preview figure assembly (Plotly dict + draggable shape points)."""
from __future__ import annotations

import logging

from ...features import has_packaged_bass_engine

logger = logging.getLogger("DecayCore")

from .. import ng_controls as ctrl
from ..target_preview_interaction import (
    build_draggable_tilt_handle_shape,
    build_draggable_target_shape,
    build_level_window_trace,
    build_target_curve_path,
    build_tilt_handle_path,
    build_vertical_marker_trace,
    parse_svg_path_points,
)
from .preview_curve import _current_target_preview_curve
from .preview_state import STATE

def _build_target_preview_fig():  # noqa: C901 - target preview figure is assembled from many UI states
    """Build the target curve preview Plotly figure from current ctrl values.

    Returns a Plotly dict plus the base points of the draggable target curve and tilt handle.
    """
    try:
        import math  # noqa: PLC0415
        from ...config.auto_mode_policy import auto_goal_forced_level_window  # noqa: PLC0415
        from ...config.value_normalization import normalize_sub_combine_mode  # noqa: PLC0415


        def _cv(name, default=None):
            return ctrl.value(name, default)

        def _to_float(v, default):
            try:
                x = float(v)
                if math.isfinite(x):
                    return x
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
                logger.exception("float parse in target preview")
            return float(default)

        # --- collect ctrl values ---
        preview_curve = _current_target_preview_curve(_cv)
        if preview_curve is None:
            return None, [], []
        lvl_min = _to_float(_cv("lvl_min", 500.0), 500.0)
        lvl_max = _to_float(_cv("lvl_max", 2000.0), 2000.0)
        mag_c_min = _to_float(_cv("mag_c_min", 10.0), 10.0)
        mag_c_max = _to_float(_cv("mag_c_max", 200.0), 200.0)
        auto_goal = str(_cv("auto_goal", "balanced") or "balanced")
        app_mode = str(_cv("mode", "BASIC") or "BASIC").upper()
        pre_ms = _to_float(_cv("ir_window_left", 85.0), 85.0)
        post_ms = _to_float(_cv("ir_window_right") or _cv("ir_window", 500.0), 500.0)
        smoothing_level = 96
        bass_integration_enabled = bool(
            _cv("bass_integration_enable", False)
        ) and has_packaged_bass_engine()
        bass_integration_sub_combine_mode = normalize_sub_combine_mode(
            _cv("bass_integration_sub_combine_mode", "average")
        )

        freq_axis = preview_curve.frequency_hz
        target_curve = preview_curve.base_magnitude_db

        # AUTO mode forced level window
        if app_mode in ("BASIC", "AUTO"):
            forced = auto_goal_forced_level_window(auto_goal)
            if forced is not None:
                lvl_min, lvl_max = float(forced[0]), float(forced[1])

        speaker_interp, speaker_components, _speaker_align_offset, _harmonic_sources = (
            _build_target_preview_speaker_curves(
                freq_axis=freq_axis,
                target_curve=target_curve,
                lvl_min=lvl_min,
                lvl_max=lvl_max,
                pre_ms=pre_ms,
                post_ms=post_ms,
                smoothing_level=smoothing_level,
                bass_integration_enabled=bass_integration_enabled,
                bass_integration_sub_combine_mode=bass_integration_sub_combine_mode,
                _cv=_cv,
            )
        )

        return _assemble_target_preview_figure(
            freq_axis=freq_axis,
            target_curve_display=preview_curve.display_magnitude_db,
            is_manual_level=preview_curve.is_manual_level,
            lvl_min=lvl_min,
            lvl_max=lvl_max,
            mag_c_min=mag_c_min,
            mag_c_max=mag_c_max,
            hc_mode_raw=preview_curve.mode_label,
            lvl_manual_db=preview_curve.manual_level_db,
            manual_target_tilt_db_per_oct=preview_curve.manual_tilt_db_per_oct,
            speaker_interp=speaker_interp,
            speaker_components=speaker_components,
            speaker_align_offset=_speaker_align_offset,
            harmonic_sources=_harmonic_sources,
            bass_integration_enabled=bass_integration_enabled,
        )

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
        import logging  # noqa: PLC0415
        logging.getLogger("DecayCore").warning("_build_target_preview_fig failed", exc_info=True)
        return None, [], []


def _assemble_target_preview_figure(
    *,
    freq_axis,
    target_curve_display,
    is_manual_level,
    lvl_min,
    lvl_max,
    mag_c_min,
    mag_c_max,
    hc_mode_raw,
    lvl_manual_db,
    manual_target_tilt_db_per_oct,
    speaker_interp,
    speaker_components,
    speaker_align_offset,
    harmonic_sources,
    bass_integration_enabled,
):
    """Assemble the target-preview Plotly figure dict from precomputed curves."""
    import math  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    import plotly.graph_objects as go  # noqa: PLC0415

    target_curve_path = build_target_curve_path(freq_axis, target_curve_display)
    drag_base_points = []
    tilt_handle_points = []
    if is_manual_level and target_curve_path:
        drag_base_points = [
            (float(x), float(y))
            for x, y in zip(freq_axis, target_curve_display)
            if math.isfinite(float(x)) and math.isfinite(float(y))
        ]
        tilt_handle_y = float(np.interp(16000.0, freq_axis, target_curve_display))
        tilt_handle_points = parse_svg_path_points(build_tilt_handle_path(tilt_handle_y))

    # --- build figure ---
    fig = go.Figure()
    fig.add_trace(build_level_window_trace(lvl_min, lvl_max))
    fig.add_trace(build_vertical_marker_trace(mag_c_min))
    fig.add_trace(build_vertical_marker_trace(mag_c_max))

    target_trace = dict(
        x=freq_axis,
        y=target_curve_display,
        mode="lines",
        name=f"Target ({hc_mode_raw})",
        line=dict(color="#4caf50", width=2.0),
    )
    if is_manual_level:
        target_trace["opacity"] = 0.45
        target_trace["hoverinfo"] = "skip"
    fig.add_trace(go.Scatter(**target_trace))
    if is_manual_level:
        fig.add_trace(go.Scatter(
            x=[freq_axis[0], freq_axis[-1]],
            y=[lvl_manual_db, lvl_manual_db],
            mode="lines",
            name=f"Manual level ({lvl_manual_db:+.1f} dB)",
            line=dict(color="rgba(15,23,42,0.55)", width=1.2, dash="dot"),
            hoverinfo="skip",
            visible="legendonly",
        ))
        fig.add_trace(go.Scatter(
            x=[freq_axis[0], freq_axis[-1]],
            y=[0.0, 0.0],
            mode="lines",
            name=f"Manual tilt ({manual_target_tilt_db_per_oct:+.1f} dB/oct @ 1 kHz)",
            line=dict(color="#f4a261", width=1.8, dash="dot"),
            hoverinfo="skip",
            visible="legendonly",
        ))
    if "L" in speaker_interp:
        fig.add_trace(go.Scatter(
            x=freq_axis, y=speaker_interp["L"], mode="lines",
            name="Predicted L total" if bass_integration_enabled else "Speaker L",
            line=dict(color="rgba(102,187,255,0.55)", width=1.2),
        ))
    if "R" in speaker_interp:
        fig.add_trace(go.Scatter(
            x=freq_axis, y=speaker_interp["R"], mode="lines",
            name="Predicted R total" if bass_integration_enabled else "Speaker R",
            line=dict(color="rgba(255,167,102,0.55)", width=1.2),
        ))
    for comp_key, trace_name, color in (
        ("L_main", "L main only", "rgba(102,187,255,0.35)"),
        ("L_sub", "L sub only", "rgba(30,64,175,0.35)"),
        ("R_main", "R main only", "rgba(255,167,102,0.35)"),
        ("R_sub", "R sub only", "rgba(185,28,28,0.35)"),
    ):
        if comp_key in speaker_components:
            fig.add_trace(go.Scatter(
                x=freq_axis,
                y=speaker_components[comp_key],
                mode="lines",
                name=trace_name,
                line=dict(color=color, width=1.0, dash="dot"),
                visible="legendonly",
            ))
    if len(speaker_interp) > 0:
        avg = np.mean(np.vstack([speaker_interp[k] for k in sorted(speaker_interp)]), axis=0)
        fig.add_trace(go.Scatter(
            x=freq_axis, y=avg, mode="lines",
            name="Predicted avg" if bass_integration_enabled else "Speaker avg",
            line=dict(color="#dc2626", width=2.0),
        ))
    _HARM_CLRS = {
        2: {"L": "rgba(40,120,255,0.65)", "R": "rgba(255,110,40,0.65)"},
        3: {"L": "rgba(0,80,200,0.60)",   "R": "rgba(200,60,10,0.60)"},
        4: {"L": "rgba(0,50,160,0.55)",   "R": "rgba(160,35,0,0.55)"},
        5: {"L": "rgba(0,20,120,0.50)",   "R": "rgba(120,15,0,0.50)"},
    }
    for _hch in ("L", "R"):
        if _hch not in harmonic_sources:
            continue
        _hf_arr = np.asarray(harmonic_sources[_hch][0], dtype=float)
        _hm_dict = harmonic_sources[_hch][1]
        _h_off = speaker_align_offset.get(_hch, 0.0)
        for _hord, _harr in sorted(_hm_dict.items()):
            _ha = np.asarray(_harr, dtype=float)
            _hmsk = (_hf_arr > 10.0) & np.isfinite(_hf_arr) & np.isfinite(_ha)
            if _hmsk.sum() < 4:
                continue
            _hi = np.interp(freq_axis, _hf_arr[_hmsk], _ha[_hmsk],
                            left=_ha[_hmsk][0], right=_ha[_hmsk][-1])
            _hcolor = _HARM_CLRS.get(_hord, {}).get(_hch, "rgba(128,128,128,0.50)")
            fig.add_trace(go.Scatter(
                x=freq_axis,
                y=_hi + _h_off,
                mode="lines",
                name=f"H{_hord} {_hch}",
                line=dict(color=_hcolor, width=1.0, dash="dot"),
                visible="legendonly",
            ))
    fig.update_xaxes(
        type="log",
        title_text="Hz",
        range=[math.log10(10.0), math.log10(20000.0)],
        fixedrange=True,
        gridcolor="rgba(15,23,42,0.10)",
        linecolor="rgba(15,23,42,0.22)",
        zerolinecolor="rgba(15,23,42,0.10)",
    )
    _y_sources = [target_curve_display, *speaker_interp.values()]
    _all_y = [v for arr in _y_sources for v in arr if math.isfinite(float(v))]
    _y_max = (max(_all_y) + 3.0) if _all_y else 20.0
    fig.update_yaxes(
        title_text="dB",
        range=[-20.0, max(-19.0, _y_max)],
        fixedrange=False,
        gridcolor="rgba(15,23,42,0.10)",
        linecolor="rgba(15,23,42,0.22)",
        zerolinecolor="rgba(15,23,42,0.10)",
    )
    fig.update_layout(height=320, margin=dict(l=40, r=20, t=30, b=35),
                      showlegend=True, template="plotly_white",
                      paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
                      font=dict(color="#1f2937"),
                      uirevision="target_preview_lock")
    if is_manual_level and target_curve_path:
        drag_shape = build_draggable_target_shape(freq_axis, target_curve_display)
        drag_shape["path"] = target_curve_path
        tilt_handle_shape = build_draggable_tilt_handle_shape(float(np.interp(16000.0, freq_axis, target_curve_display)))
        fig.update_layout(shapes=[drag_shape, tilt_handle_shape])

    fig_dict = fig.to_plotly_json()
    layout = fig_dict.setdefault("layout", {})
    layout["uirevision"] = "target_preview_lock"
    layout["editrevision"] = (
        f"target_preview_manual_{STATE.refresh_token}_{int(STATE.drag_active)}"
        if is_manual_level else
        "target_preview_static"
    )
    fig_dict["config"] = {
        "responsive": True,
        "displayModeBar": True,
        "modeBarButtonsToRemove": ["select2d", "lasso2d"],
        "editable": False,
        "edits": {"shapePosition": bool(is_manual_level)},
        "doubleClick": False,
        "scrollZoom": False,
    }
    return fig_dict, drag_base_points, tilt_handle_points


def _build_target_preview_speaker_curves(
    *,
    freq_axis,
    target_curve,
    lvl_min,
    lvl_max,
    pre_ms,
    post_ms,
    smoothing_level,
    bass_integration_enabled,
    bass_integration_sub_combine_mode,
    _cv,
):
    """Load + align speaker/sub measurement curves for the target preview."""
    import numpy as np  # noqa: PLC0415
    from ...io.generated_measurement_source import generated_source_matches_upload, parse_generated_source  # noqa: PLC0415
    from ...dsp.smoothing import psychoacoustic_smoothing as _psycho_smooth  # noqa: PLC0415
    from ..target_preview_cache import (  # noqa: PLC0415
        load_path_measurement_curve,
        load_path_measurement_transfer,
        load_upload_measurement_curve,
        load_upload_measurement_transfer,
    )

    def _smooth_for_preview(freq_axis, m):
        try:
            return _psycho_smooth(freq_axis, np.asarray(m, dtype=float))
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
            return m

    # --- speaker measurements ---
    def _align(m_curve, t_curve, fa, fmin, fmax):
        try:
            mask = (fa >= fmin) & (fa <= fmax)
            if not mask.any():
                return m_curve
            diff = np.nanmedian(t_curve[mask] - np.asarray(m_curve, dtype=float)[mask])
            return np.asarray(m_curve, dtype=float) + diff
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
            return m_curve

    speaker_interp = {}
    speaker_components = {}
    _speaker_align_offset: dict[str, float] = {}
    _harmonic_sources: dict[str, tuple] = {}
    if bass_integration_enabled:
        from ...dsp.bass_integration import (  # noqa: PLC0415
            build_combined_sub_transfer,
            sum_complex_responses,
        )

        def _load_transfer(up_key: str, path_key: str, label: str):
            tr = load_upload_measurement_transfer(
                _cv(up_key, None),
                pre_ms=pre_ms,
                post_ms=post_ms,
                smoothing_level=smoothing_level,
                label=label,
            )
            if tr is None:
                tr = load_path_measurement_transfer(
                    _cv(path_key, ""),
                    pre_ms=pre_ms,
                    post_ms=post_ms,
                    smoothing_level=smoothing_level,
                    label=label,
                )
            return tr

        bi_slots = {
            "L_main": _load_transfer("file_l_main", "local_path_l_main", "L main only"),
            "R_main": _load_transfer("file_r_main", "local_path_r_main", "R main only"),
            "L_sub": _load_transfer("file_l_sub", "local_path_l_sub", "L sub only"),
            "R_sub": _load_transfer("file_r_sub", "local_path_r_sub", "R sub only"),
        }

        def _interp_transfer(tr):
            if tr is None:
                return None
            try:
                ff = np.asarray(tr.freqs_hz, dtype=float)
                mm = np.asarray(tr.mag_db, dtype=float)
                if ff.size < 8 or mm.size != ff.size:
                    return None
                return np.interp(freq_axis, ff, mm, left=mm[0], right=mm[-1])
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
                return None

        for ch in ("L", "R"):
            main_tr = bi_slots.get(f"{ch}_main")
            l_sub_tr = bi_slots.get("L_sub")
            r_sub_tr = bi_slots.get("R_sub")
            active_subs = [tr for tr in (l_sub_tr, r_sub_tr) if tr is not None]
            if main_tr is None or not active_subs:
                continue
            combined_sub_tr, _combine_diag = build_combined_sub_transfer(
                main_tr,
                *active_subs,
                mode=bass_integration_sub_combine_mode,
                label=f"{ch} combined sub predicted",
            )
            total_tr = sum_complex_responses(
                main_tr,
                combined_sub_tr,
                label=f"{ch} total predicted",
            )
            total_curve = _interp_transfer(total_tr)
            if total_curve is None:
                continue
            total_aligned = _align(total_curve, target_curve, freq_axis, lvl_min, lvl_max)
            _speaker_align_offset[ch] = float(np.nanmedian(total_aligned - total_curve)) if np.isfinite(total_curve).any() else 0.0
            speaker_interp[ch] = _smooth_for_preview(freq_axis, total_aligned)

            main_curve = _interp_transfer(main_tr)
            if main_curve is not None:
                speaker_components[f"{ch}_main"] = _smooth_for_preview(
                    freq_axis, main_curve + _speaker_align_offset[ch]
                )
            _individual_sub_tr = bi_slots.get(f"{ch}_sub")
            _individual_sub_curve = _interp_transfer(_individual_sub_tr) if _individual_sub_tr is not None else None
            if _individual_sub_curve is not None:
                speaker_components[f"{ch}_sub"] = _smooth_for_preview(
                    freq_axis, _individual_sub_curve + _speaker_align_offset[ch]
                )
    else:
        generated_sources = {
            "L": _cv("generated_measurement_l", None),
            "R": _cv("generated_measurement_r", None),
        }
        for ch, up_key, path_key in (("L", "file_l", "local_path_l"), ("R", "file_r", "local_path_r")):
            ff = None
            mm = None
            generated = generated_sources.get(ch)
            upload_value = _cv(up_key, None)
            if generated_source_matches_upload(generated, upload_value):
                ff_gen, mm_gen, _pp_gen, _raw_ir, _raw_ir_fs, _h_freq, _h_mags = parse_generated_source(
                    generated,
                    pre_ms=pre_ms,
                    post_ms=post_ms,
                    smoothing_level=smoothing_level,
                )
                if ff_gen is not None and mm_gen is not None:
                    ff = ff_gen
                    mm = mm_gen
                    if _h_freq is not None and _h_mags:
                        _harmonic_sources[ch] = (_h_freq, _h_mags)
            if ff is None:
                ff, mm = load_upload_measurement_curve(
                    upload_value,
                    pre_ms=pre_ms,
                    post_ms=post_ms,
                    smoothing_level=smoothing_level,
                )
            if ff is None:
                ff, mm = load_path_measurement_curve(
                    _cv(path_key, ""),
                    pre_ms=pre_ms,
                    post_ms=post_ms,
                    smoothing_level=smoothing_level,
                )
            if ff is not None and mm is not None:
                m_interp = np.interp(freq_axis, ff, mm, left=mm[0], right=mm[-1])
                m_aligned = _align(m_interp, target_curve, freq_axis, lvl_min, lvl_max)
                _speaker_align_offset[ch] = float(np.nanmedian(m_aligned - m_interp))
                speaker_interp[ch] = _smooth_for_preview(freq_axis, m_aligned)

    from ...io.measurements_loader_parts import _try_load_harmonic_sidecar  # noqa: PLC0415
    for _hch in ("L", "R"):
        if _hch in _harmonic_sources:
            continue
        _hlp = str(_cv(
            ("local_path_l_main" if _hch == "L" else "local_path_r_main")
            if bass_integration_enabled else
            ("local_path_l" if _hch == "L" else "local_path_r"),
            "",
        ) or "")
        if not _hlp:
            continue
        _hf, _hm = _try_load_harmonic_sidecar(_hlp)
        if _hf is not None and _hm:
            _harmonic_sources[_hch] = (_hf, _hm)

    return speaker_interp, speaker_components, _speaker_align_offset, _harmonic_sources
