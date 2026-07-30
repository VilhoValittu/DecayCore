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

import importlib
import importlib.util


class PackagedFeatureUnavailableError(RuntimeError):
    """Raised when a packaged-only native capability is requested from source."""


def has_measurement_module() -> bool:
    """Return True when the private measurement feature package is installed."""
    try:
        return importlib.util.find_spec("decaycore.measurement") is not None
    except (ImportError, AttributeError, ValueError):
        return False


# Native Rust acceleration modules. When these are missing the program still
# runs using the pure-Python fallbacks, but noticeably slower.
RUST_EXTENSION_MODULES = ("decaycore_scoring", "decaycore_dsp")

# This engine is intentionally built only for packaged DecayCore releases.  It
# is kept separate from RUST_EXTENSION_MODULES because source installs cannot
# obtain it from the public native-acceleration installation path.
PACKAGED_BASS_ENGINE_MODULE = "decaycore_bass_engine"

# Engine policy versions this build knows how to talk to.  The engine ships
# inside the package rather than being installed separately, so a version
# outside this window means a mismatched build and must fail the capability
# gate — an open-ended ">= 1" would silently accept a future breaking engine.
# Raised to 2 when robust_score_rs stopped collapsing a candidate to +inf on a
# single infeasible scenario and the dominance band started propagating NaN.
PACKAGED_BASS_ENGINE_MIN_POLICY_VERSION = 2
PACKAGED_BASS_ENGINE_MAX_POLICY_VERSION = 2

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


def has_packaged_bass_engine() -> bool:
    """Return True when the packaged-only automatic bass engine is available."""
    try:
        if importlib.util.find_spec(PACKAGED_BASS_ENGINE_MODULE) is None:
            return False
        module = importlib.import_module(PACKAGED_BASS_ENGINE_MODULE)
        policy_version = int(getattr(module, "ENGINE_POLICY_VERSION", 0) or 0)
        return bool(
            PACKAGED_BASS_ENGINE_MIN_POLICY_VERSION
            <= policy_version
            <= PACKAGED_BASS_ENGINE_MAX_POLICY_VERSION
            and callable(getattr(module, "channel_overlap_metrics_rs", None))
            and callable(getattr(module, "transform_sub_and_sum_rs", None))
            and callable(getattr(module, "robust_score_rs", None))
        )
    except (ImportError, OSError, AttributeError, TypeError, ValueError):
        return False


def require_packaged_bass_engine() -> None:
    """Fail explicitly when package-only automatic bass integration is unavailable."""
    if not has_packaged_bass_engine():
        raise PackagedFeatureUnavailableError(
            "Automatic Bass Integration requires the packaged DecayCore application."
        )


__all__ = [
    "has_measurement_module",
    "has_packaged_bass_engine",
    "has_rust_acceleration",
    "missing_rust_modules",
    "require_packaged_bass_engine",
    "PACKAGED_BASS_ENGINE_MAX_POLICY_VERSION",
    "PACKAGED_BASS_ENGINE_MIN_POLICY_VERSION",
    "PACKAGED_BASS_ENGINE_MODULE",
    "PackagedFeatureUnavailableError",
    "RUST_EXTENSION_MODULES",
    "RUST_INSTALL_URL",
]
