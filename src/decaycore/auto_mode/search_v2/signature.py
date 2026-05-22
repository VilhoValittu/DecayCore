# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""Canonical Auto Search v2 signatures."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from .input_model import AutoSearchInput
from ..cache_signature import _auto_signature_payload


@dataclass(frozen=True)
class AutoSearchSignature:
    signature: str
    payload: dict
    measurement_identity: str
    frequency_grid_identity: str
    diff_reason: str


def _signature_diff_reason(payload: dict) -> str:
    sections = [
        "schema/compat",
        "measurement identity",
        "measurement metadata identity",
        "frequency grid summary",
        "DSP/scorer policy versions",
        "target identity",
        "correction band",
        "smoothing",
        "TDC/scoring policy",
        "phase/group-delay policy",
        "bass integration flags",
        "filter type/goal/fs/taps",
    ]
    ft = str(payload.get("filter_type", "") or "")
    goal = str(payload.get("auto_goal", "") or "")
    fs = str(payload.get("fs", "") or "")
    taps = str(payload.get("taps", "") or "")
    return (
        "signature includes "
        + ", ".join(sections)
        + f" (filter_type={ft}, goal={goal}, fs={fs}, taps={taps})"
    )


def compute_auto_search_signature_object(search_input: AutoSearchInput) -> AutoSearchSignature:
    payload = _auto_signature_payload(
        base_data=search_input.base_data(),
        measurements={"_auto_measurement_signature": search_input.measurement_identity},
        fs_v=int(search_input.fs_v),
        taps_v=int(search_input.taps_v),
        xos=list(search_input.xos_items or []),
        hpf=search_input.hpf(),
        hc_mode=search_input.hc_mode,
        include_hc_mode=bool(search_input.include_hc_mode),
    )
    payload["measurement_sig"] = str(search_input.measurement_identity)
    payload["measurement_metadata_identity"] = str(search_input.measurement_metadata_identity)
    payload["frequency_grid_sig"] = str(search_input.frequency_grid_identity)
    h = hashlib.sha256()
    try:
        h.update(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8"))
    except Exception:
        h.update(str(sorted(payload.items())).encode("utf-8", "ignore"))
    return AutoSearchSignature(
        signature=str(h.hexdigest()),
        payload=dict(payload),
        measurement_identity=str(search_input.measurement_identity),
        frequency_grid_identity=str(search_input.frequency_grid_identity),
        diff_reason=_signature_diff_reason(payload),
    )


def compute_auto_search_signature(search_input: AutoSearchInput) -> str:
    return compute_auto_search_signature_object(search_input).signature
