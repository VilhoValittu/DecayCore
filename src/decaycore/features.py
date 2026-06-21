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

import importlib.util


def has_measurement_module() -> bool:
    """Return True when the private measurement feature package is installed."""
    try:
        return importlib.util.find_spec("decaycore.measurement") is not None
    except (ImportError, AttributeError, ValueError):
        return False


# Native Rust acceleration modules. When these are missing the program still
# runs using the pure-Python fallbacks, but noticeably slower.
RUST_EXTENSION_MODULES = ("decaycore_scoring", "decaycore_dsp")

# User-facing installation guide for the native extensions.
RUST_INSTALL_URL = "https://vilhovalittu.github.io/DecayCore/installation/"


def missing_rust_modules() -> list[str]:
    """Return the names of native Rust extension modules that are not installed."""
    missing: list[str] = []
    for name in RUST_EXTENSION_MODULES:
        try:
            found = importlib.util.find_spec(name) is not None
        except (ImportError, AttributeError, ValueError):
            found = False
        if not found:
            missing.append(name)
    return missing


def has_rust_acceleration() -> bool:
    """Return True only when every native Rust extension module is available."""
    return not missing_rust_modules()


__all__ = [
    "has_measurement_module",
    "has_rust_acceleration",
    "missing_rust_modules",
    "RUST_EXTENSION_MODULES",
    "RUST_INSTALL_URL",
]
