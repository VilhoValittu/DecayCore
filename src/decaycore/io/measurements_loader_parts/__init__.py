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

from importlib import import_module
from typing import Any

__all__ = [
    "_clean_local_path",
    "_detect_coherent_slot_anchor_sample",
    "_detect_shared_coherent_anchor_sample",
    "_get_local_path",
    "_get_uploaded_file",
    "_get_wav_window_params",
    "_is_wav_upload",
    "_load_coherent_transfer_slot",
    "_load_raw_wav_from_source",
    "_measurement_sidecar_candidates",
    "_measurement_sidecar_stems",
    "_silent_transfer_like",
    "_try_load_harmonic_sidecar",
    "_try_load_measurement_metadata_sidecar",
    "_try_load_rt60_sidecar",
    "load_bass_integration_measurements",
    "load_measurements_lr",
    "load_raw_ir_sub",
    "load_raw_irs_lr",
    "parse_measurements_from_upload",
]

_EXPORT_MODULES = {
    "_silent_transfer_like": "lr_measurement_loader",
    "parse_measurements_from_upload": "lr_measurement_loader",
    "_try_load_harmonic_sidecar": "lr_measurement_loader",
    "_measurement_sidecar_stems": "lr_measurement_loader",
    "_measurement_sidecar_candidates": "lr_measurement_loader",
    "_try_load_measurement_metadata_sidecar": "lr_measurement_loader",
    "_try_load_rt60_sidecar": "lr_measurement_loader",
    "load_measurements_lr": "lr_measurement_loader",
    "_clean_local_path": "measurement_source_helpers",
    "_get_uploaded_file": "measurement_source_helpers",
    "_get_local_path": "measurement_source_helpers",
    "_get_wav_window_params": "measurement_source_helpers",
    "_is_wav_upload": "measurement_source_helpers",
    "_load_coherent_transfer_slot": "measurement_source_helpers",
    "_detect_coherent_slot_anchor_sample": "measurement_source_helpers",
    "_detect_shared_coherent_anchor_sample": "measurement_source_helpers",
    "_load_raw_wav_from_source": "raw_ir_and_bass_loader",
    "load_raw_irs_lr": "raw_ir_and_bass_loader",
    "load_raw_ir_sub": "raw_ir_and_bass_loader",
    "load_bass_integration_measurements": "raw_ir_and_bass_loader",
}


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.{module_name}"), name)
    globals()[name] = value
    return value
