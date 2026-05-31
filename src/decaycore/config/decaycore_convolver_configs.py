# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

import math

_NUMERIC_PARSE_EXCEPTIONS = (TypeError, ValueError, OverflowError)


def _tc_segment(target_curve_tag: str | None) -> str:
    tag = str(target_curve_tag or "").strip()
    return f"_{tag}" if tag else ""


def _title_suffix(program_version: str | None = None, winner_rank_score: float | None = None) -> str:
    parts: list[str] = []
    ver = str(program_version or "").strip()
    if ver:
        parts.append(ver)
    try:
        rank = float(winner_rank_score) if winner_rank_score is not None else float("nan")
    except _NUMERIC_PARSE_EXCEPTIONS:
        rank = float("nan")
    if math.isfinite(rank):
        parts.append(f"rank {rank:.3f}")
    return "" if not parts else " " + " ".join(parts)


def _normalize_layout(layout: str | None) -> str:
    try:
        layout_s = str(layout or "").strip().lower()
    except (TypeError, ValueError):
        layout_s = ""
    return "Stereo" if "stereo" in layout_s else "Mono"


def filter_wav_export_spec(
    fs_token,
    ft_short,
    file_ts,
    *,
    irw_tag: str = "auto",
    target_curve_tag: str = "",
    layout: str | None = "Mono",
    prefix: str = "",
):
    tc = _tc_segment(target_curve_tag)
    fs_s = str(fs_token)
    layout_norm = _normalize_layout(layout)

    if layout_norm == "Stereo":
        stereo_name = f"Stereo_{ft_short}_{fs_s}Hz{tc}_{file_ts}_{irw_tag}.wav"
        stereo_path = f"{prefix}{stereo_name}"
        return {
            "layout": layout_norm,
            "bundle_names": [stereo_name],
            "left_filename": stereo_path,
            "right_filename": stereo_path,
            "left_channel": 0,
            "right_channel": 1,
        }

    l_name = f"L_{ft_short}_{fs_s}Hz{tc}_{file_ts}_{irw_tag}.wav"
    r_name = f"R_{ft_short}_{fs_s}Hz{tc}_{file_ts}_{irw_tag}.wav"
    return {
        "layout": layout_norm,
        "bundle_names": [l_name, r_name],
        "left_filename": f"{prefix}{l_name}",
        "right_filename": f"{prefix}{r_name}",
        "left_channel": 0,
        "right_channel": 0,
    }


def bypass_filter_wav_export_spec(
    fs_token,
    ft_short,
    file_ts,
    *,
    irw_tag: str = "auto",
    target_curve_tag: str = "",
    layout: str | None = "Mono",
    prefix: str = "",
):
    spec = filter_wav_export_spec(
        fs_token,
        ft_short,
        file_ts,
        irw_tag=irw_tag,
        target_curve_tag=target_curve_tag,
        layout=layout,
        prefix=prefix,
    )
    names = [f"Bypass_{name}" for name in list(spec.get("bundle_names", []) or [])]
    if str(spec.get("layout")) == "Stereo":
        stereo_name = names[0] if names else ""
        return {
            **spec,
            "bundle_names": names,
            "left_filename": f"{prefix}{stereo_name}",
            "right_filename": f"{prefix}{stereo_name}",
        }
    left_name = names[0] if len(names) > 0 else ""
    right_name = names[1] if len(names) > 1 else ""
    return {
        **spec,
        "bundle_names": names,
        "left_filename": f"{prefix}{left_name}",
        "right_filename": f"{prefix}{right_name}",
    }


def sub_filter_wav_export_spec(
    fs_token,
    ft_short,
    file_ts,
    *,
    irw_tag: str = "auto",
    prefix: str = "",
):
    fs_s = str(fs_token)
    sub_name = f"Sub_{ft_short}_{fs_s}Hz_{file_ts}_{irw_tag}.wav"
    return {
        "bundle_name": sub_name,
        "filename": f"{prefix}{sub_name}",
        "channel": 0,
    }


def bypass_sub_filter_wav_export_spec(
    fs_token,
    ft_short,
    file_ts,
    *,
    irw_tag: str = "auto",
    prefix: str = "",
):
    spec = sub_filter_wav_export_spec(
        fs_token,
        ft_short,
        file_ts,
        irw_tag=irw_tag,
        prefix=prefix,
    )
    sub_name = f"Bypass_{spec['bundle_name']}"
    return {
        **spec,
        "bundle_name": sub_name,
        "filename": f"{prefix}{sub_name}",
    }


def _camilladsp_crossover_order(value: int | float | str | None, default: int = 2) -> int:
    try:
        order = int(round(float(value))) if value is not None else int(default)
    except _NUMERIC_PARSE_EXCEPTIONS:
        order = int(default)
    if order <= 2:
        return 2
    if order % 2:
        order -= 1
    return max(2, order)


def _extend_camilladsp_crossover_filter(
    lines: list[str],
    *,
    name: str,
    kind: str,
    freq_hz: float,
    order: int,
) -> None:
    if int(order) <= 2:
        lines.extend(
            [
                "",
                f"  {name}:",
                "    type: Biquad",
                "    parameters:",
                f"      type: {kind}",
                f"      freq: {freq_hz:.3f}",
                "      q: 0.707107",
            ]
        )
        return

    combo_type = "LinkwitzRileyHighpass" if str(kind).lower() == "highpass" else "LinkwitzRileyLowpass"
    lines.extend(
        [
            "",
            f"  {name}:",
            "    type: BiquadCombo",
            "    parameters:",
            f"      type: {combo_type}",
            f"      freq: {freq_hz:.3f}",
            f"      order: {int(order)}",
        ]
    )


def _normalize_camilladsp_biquads(value) -> list[dict[str, float]]:
    out: list[dict[str, float]] = []
    for raw in list(value or []):
        if not isinstance(raw, dict):
            continue
        try:
            freq = float(raw.get("freq", raw.get("freq_hz")))
            q = float(raw.get("q"))
            gain = float(raw.get("gain", raw.get("gain_db")))
        except (TypeError, ValueError, OverflowError):
            continue
        if not (freq == freq and q == q and gain == gain):
            continue
        if freq <= 0.0 or q <= 0.0 or gain > 0.0:
            continue
        out.append({"freq": freq, "q": q, "gain": gain})
    return out


def generate_raspberry_yaml(
    fs,
    ft_short,
    file_ts,
    master_gain_db=0.0,
    irw_tag: str = "auto",
    target_curve_tag: str = "",
    layout: str | None = "Mono",
    program_version: str | None = None,
    winner_rank_score: float | None = None,
    include_sub: bool = False,
    sub_allpass_freq_hz: float | None = None,
    sub_allpass_q: float | None = None,
    sub_delay_ms: float | None = None,
    sub_polarity_invert: bool = False,
    sub_gain_trim_db: float | None = None,
    main_hpf_hz: float | None = None,
    sub_hpf_hz: float | None = None,
    sub_lpf_hz: float | None = None,
    main_hpf_order: int | float | str | None = None,
    sub_hpf_order: int | float | str | None = None,
    sub_lpf_order: int | float | str | None = None,
    device_format: str = "S32_LE",
    left_iir_biquads=None,
    right_iir_biquads=None,
):
    if device_format not in ("S32_LE", "S16_LE"):
        raise ValueError(f"device_format must be 'S32_LE' or 'S16_LE', got {device_format!r}")
    yaml_device_format = str(device_format).replace("_", "_")
    tc = _tc_segment(target_curve_tag)
    title_meta = _title_suffix(program_version=program_version, winner_rank_score=winner_rank_score)
    spec = filter_wav_export_spec(
        "$samplerate$",
        ft_short,
        file_ts,
        irw_tag=irw_tag,
        target_curve_tag=target_curve_tag,
        layout=layout,
        prefix="../coeffs/",
    )
    l_wav = str(spec["left_filename"])
    r_wav = str(spec["right_filename"])
    l_ch = int(spec["left_channel"])
    r_ch = int(spec["right_channel"])
    sub_spec = sub_filter_wav_export_spec(
        "$samplerate$",
        ft_short,
        file_ts,
        irw_tag=irw_tag,
        prefix="../coeffs/",
    )
    sub_wav = str(sub_spec["filename"])
    sub_ch = int(sub_spec["channel"])

    include_sub = bool(include_sub)
    try:
        ap_freq_hz = float(sub_allpass_freq_hz) if sub_allpass_freq_hz is not None else float("nan")
    except _NUMERIC_PARSE_EXCEPTIONS:
        ap_freq_hz = float("nan")
    try:
        ap_q = float(sub_allpass_q) if sub_allpass_q is not None else float("nan")
    except _NUMERIC_PARSE_EXCEPTIONS:
        ap_q = float("nan")
    try:
        raw_sub_delay_ms = float(sub_delay_ms) if sub_delay_ms is not None else 0.0
    except _NUMERIC_PARSE_EXCEPTIONS:
        raw_sub_delay_ms = 0.0
    if not (raw_sub_delay_ms == raw_sub_delay_ms) or abs(raw_sub_delay_ms) == float("inf"):
        raw_sub_delay_ms = 0.0
    try:
        sub_gain_db = float(sub_gain_trim_db) if sub_gain_trim_db is not None else 0.0
    except _NUMERIC_PARSE_EXCEPTIONS:
        sub_gain_db = 0.0
    if not (sub_gain_db == sub_gain_db) or abs(sub_gain_db) == float("inf"):
        sub_gain_db = 0.0
    try:
        main_hpf = float(main_hpf_hz) if main_hpf_hz is not None else float("nan")
    except _NUMERIC_PARSE_EXCEPTIONS:
        main_hpf = float("nan")
    try:
        sub_hpf = float(sub_hpf_hz) if sub_hpf_hz is not None else float("nan")
    except _NUMERIC_PARSE_EXCEPTIONS:
        sub_hpf = float("nan")
    try:
        sub_lpf = float(sub_lpf_hz) if sub_lpf_hz is not None else float("nan")
    except _NUMERIC_PARSE_EXCEPTIONS:
        sub_lpf = float("nan")
    sub_polarity_invert = bool(sub_polarity_invert)
    use_main_hpf = bool(include_sub and main_hpf == main_hpf and abs(main_hpf) != float("inf") and main_hpf > 0.0)
    use_sub_hpf = bool(include_sub and sub_hpf == sub_hpf and abs(sub_hpf) != float("inf") and sub_hpf > 0.0)
    use_sub_lpf = bool(include_sub and sub_lpf == sub_lpf and abs(sub_lpf) != float("inf") and sub_lpf > 0.0)
    use_sub_allpass = bool(
        include_sub
        and ap_freq_hz == ap_freq_hz
        and ap_q == ap_q
        and abs(ap_freq_hz) != float("inf")
        and abs(ap_q) != float("inf")
        and ap_freq_hz > 0.0
        and ap_q > 0.0
    )
    # Delay filters only accept non-negative values. Preserve the requested
    # relative main/sub timing by shifting the mains when the sub needs
    # an effective phase advance.
    main_delay_ms = float(max(0.0, -raw_sub_delay_ms)) if include_sub else 0.0
    sub_delay_pos_ms = float(max(0.0, raw_sub_delay_ms + main_delay_ms)) if include_sub else 0.0
    use_main_delay = bool(include_sub and main_delay_ms > 1e-9)
    use_sub_delay = bool(include_sub and sub_delay_pos_ms > 1e-9)
    use_sub_gain = bool(include_sub and (abs(sub_gain_db) > 1e-9 or sub_polarity_invert))
    main_hpf_order_i = _camilladsp_crossover_order(main_hpf_order, 2)
    sub_hpf_order_i = _camilladsp_crossover_order(sub_hpf_order, 2)
    sub_lpf_order_i = _camilladsp_crossover_order(sub_lpf_order, 2)
    left_iir = _normalize_camilladsp_biquads(left_iir_biquads)
    right_iir = _normalize_camilladsp_biquads(right_iir_biquads)

    playback_channels = 3 if include_sub else 2
    left_filter_names = ["mastergain"]
    right_filter_names = ["mastergain"]
    if use_main_delay:
        left_filter_names.append("main_delay")
        right_filter_names.append("main_delay")
    if use_main_hpf:
        left_filter_names.append("main_hpf")
        right_filter_names.append("main_hpf")
    left_filter_names.extend([f"l_hybrid_iir_{idx}" for idx in range(1, len(left_iir) + 1)])
    right_filter_names.extend([f"r_hybrid_iir_{idx}" for idx in range(1, len(right_iir) + 1)])
    left_filter_names.append("ir_left")
    right_filter_names.append("ir_right")

    sub_filter_names = ["mastergain"]
    if use_sub_gain:
        sub_filter_names.append("sub_gain")
    if use_sub_delay:
        sub_filter_names.append("sub_delay")
    if use_sub_hpf:
        sub_filter_names.append("sub_hpf")
    if use_sub_lpf:
        sub_filter_names.append("sub_lpf")
    if use_sub_allpass:
        sub_filter_names.append("sub_allpass")
    sub_filter_names.append("ir_sub")

    lines = [
        "description: null",
        "devices:",
        "  capture:",
        "    type: Stdin",
        "    channels: 2",
        f"    format: {yaml_device_format}",
        "  playback:",
        "    type: Alsa",
        "    device: plughw:0,0",
        f"    channels: {playback_channels}",
        f"    format: {yaml_device_format}",
        f"  samplerate: {int(fs)}",
        "  enable_rate_adjust: true",
        "  chunksize: 2048",
        "  queuelimit: 1",
        "  volume_ramp_time: 150",
        "",
        "filters:",
        "  ir_left:",
        "    type: Conv",
        "    parameters:",
        "      type: Wav",
        f"      filename: {l_wav}",
        f"      channel: {l_ch}",
        "",
        "  ir_right:",
        "    type: Conv",
        "    parameters:",
        "      type: Wav",
        f"      filename: {r_wav}",
        f"      channel: {r_ch}",
    ]
    for idx, biquad in enumerate(left_iir, start=1):
        lines.extend(
            [
                "",
                f"  l_hybrid_iir_{idx}:",
                "    type: Biquad",
                "    parameters:",
                "      type: Peaking",
                f"      freq: {float(biquad['freq']):.3f}",
                f"      q: {float(biquad['q']):.6f}",
                f"      gain: {float(biquad['gain']):.3f}",
            ]
        )
    for idx, biquad in enumerate(right_iir, start=1):
        lines.extend(
            [
                "",
                f"  r_hybrid_iir_{idx}:",
                "    type: Biquad",
                "    parameters:",
                "      type: Peaking",
                f"      freq: {float(biquad['freq']):.3f}",
                f"      q: {float(biquad['q']):.6f}",
                f"      gain: {float(biquad['gain']):.3f}",
            ]
        )
    if include_sub:
        lines.extend(
            [
                "",
                "  ir_sub:",
                "    type: Conv",
                "    parameters:",
                "      type: Wav",
                f"      filename: {sub_wav}",
                f"      channel: {sub_ch}",
            ]
        )
        if use_sub_allpass:
            lines.extend(
                [
                    "",
                    "  sub_allpass:",
                    "    type: Biquad",
                    "    parameters:",
                    "      type: Allpass",
                    f"      freq: {ap_freq_hz:.3f}",
                    f"      q: {ap_q:.6f}",
                ]
            )
        if use_sub_hpf:
            _extend_camilladsp_crossover_filter(
                lines,
                name="sub_hpf",
                kind="Highpass",
                freq_hz=sub_hpf,
                order=sub_hpf_order_i,
            )
        if use_sub_lpf:
            _extend_camilladsp_crossover_filter(
                lines,
                name="sub_lpf",
                kind="Lowpass",
                freq_hz=sub_lpf,
                order=sub_lpf_order_i,
            )
    if use_main_hpf:
        _extend_camilladsp_crossover_filter(
            lines,
            name="main_hpf",
            kind="Highpass",
            freq_hz=main_hpf,
            order=main_hpf_order_i,
        )
    if use_main_delay:
        lines.extend(
            [
                "",
                "  main_delay:",
                "    type: Delay",
                "    parameters:",
                f"      delay: {main_delay_ms:.3f}",
                "      unit: ms",
                "      subsample: true",
            ]
        )
    if use_sub_gain:
        lines.extend(
            [
                "",
                "  sub_gain:",
                "    type: Gain",
                "    parameters:",
                f"      gain: {sub_gain_db:.3f}",
                "      scale: dB",
                f"      inverted: {'true' if sub_polarity_invert else 'false'}",
            ]
        )
    if use_sub_delay:
        lines.extend(
            [
                "",
                "  sub_delay:",
                "    type: Delay",
                "    parameters:",
                f"      delay: {sub_delay_pos_ms:.3f}",
                "      unit: ms",
                "      subsample: true",
            ]
        )
    lines.extend(
        [
            "",
            "  mastergain:",
            "    type: Gain",
            "    parameters:",
            "      gain: -6",
            "",
            "mixers:",
            "  stereo:",
            "    channels:",
            "      in: 2",
            f"      out: {playback_channels}",
            "    mapping:",
            "      - dest: 0",
            "        sources:",
            "          - channel: 0",
            "            gain: 0",
            "      - dest: 1",
            "        sources:",
            "          - channel: 1",
            "            gain: 0",
        ]
    )
    if include_sub:
        lines.extend(
            [
                "      - dest: 2",
                "        sources:",
                "          - channel: 0",
                "            gain: 0",
                "          - channel: 1",
                "            gain: 0",
            ]
        )
    lines.extend(
        [
            "",
            "pipeline:",
            "  - type: Mixer",
            "    name: stereo",
            "  - type: Filter",
            "    channels: [0]",
            f"    names: [{', '.join(left_filter_names)}]",
            "  - type: Filter",
            "    channels: [1]",
            f"    names: [{', '.join(right_filter_names)}]",
        ]
    )
    if include_sub:
        lines.extend(
            [
                "  - type: Filter",
                "    channels: [2]",
                f"    names: [{', '.join(sub_filter_names)}]",
            ]
        )
    lines.extend(
        [
            "",
            "processors: null",
            f"title: {ft_short} Window {irw_tag}{tc} {file_ts} {title_meta}",
        ]
    )
    return "\n".join(lines).strip()




def generate_hlc_config(
    fs,
    ft_short,
    file_ts,
    irw_tag: str = "auto",
    target_curve_tag: str = "",
    layout: str | None = "Mono",
):
    """Rakentaa tai generoi: generate hlc config."""
    spec = filter_wav_export_spec(
        fs,
        ft_short,
        file_ts,
        irw_tag=irw_tag,
        target_curve_tag=target_curve_tag,
        layout=layout,
    )
    l_name = str(spec["left_filename"])
    r_name = str(spec["right_filename"])
    l_ch = int(spec["left_channel"])
    r_ch = int(spec["right_channel"])

    config = [
        f"{int(fs)} 2 2 0",
        "0 0",
        "0 0",
        f"{l_name}",
        f"{l_ch}",
        "0.0",
        "0.0",
        f"{r_name}",
        f"{r_ch}",
        "1.0",
        "1.0"
    ]
    return "\n".join(config)


def generate_bypass_hlc_config(
    fs,
    ft_short,
    file_ts,
    irw_tag: str = "auto",
    target_curve_tag: str = "",
    layout: str | None = "Mono",
):
    spec = bypass_filter_wav_export_spec(
        fs,
        ft_short,
        file_ts,
        irw_tag=irw_tag,
        target_curve_tag=target_curve_tag,
        layout=layout,
    )
    l_name = str(spec["left_filename"])
    r_name = str(spec["right_filename"])
    l_ch = int(spec["left_channel"])
    r_ch = int(spec["right_channel"])

    config = [
        f"{int(fs)} 2 2 0",
        "0 0",
        "0 0",
        f"{l_name}",
        f"{l_ch}",
        "0.0",
        "0.0",
        f"{r_name}",
        f"{r_ch}",
        "1.0",
        "1.0"
    ]
    return "\n".join(config)
