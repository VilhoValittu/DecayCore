# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""Canonical acoustic policy values for Direct DAC bass integration."""

from __future__ import annotations

from ..config.value_normalization import normalize_bass_integration_profile

BASS_INTEGRATION_GUARD_LO_RATIO = 0.60
BASS_INTEGRATION_GUARD_HI_RATIO = 1.40
BASS_INTEGRATION_PROFILE_WEIGHTS: dict[str, dict[str, float]] = {
    "safe": {
        "cancellation": 6.0,
        "overlap_ripple": 1.4,
        "anti_null_boost": 1.0,
        "sub_dominance": 0.7,
        "xo_gd_continuity": 0.6,
        "main_activity": 4.5,
    },
    "normal": {
        "cancellation": 4.2,
        "overlap_ripple": 0.9,
        "anti_null_boost": 0.65,
        "sub_dominance": 0.45,
        "xo_gd_continuity": 0.45,
        "main_activity": 3.0,
    },
    "assertive": {
        "cancellation": 3.0,
        "overlap_ripple": 0.7,
        "anti_null_boost": 0.4,
        "sub_dominance": 0.25,
        "xo_gd_continuity": 0.3,
        "main_activity": 1.5,
    },
}


def bass_integration_profile_weights(profile: str | None) -> dict[str, float]:
    """Return an isolated weight mapping for a normalized policy profile."""
    profile_norm = normalize_bass_integration_profile(profile)
    return dict(BASS_INTEGRATION_PROFILE_WEIGHTS[profile_norm])


__all__ = [
    "BASS_INTEGRATION_GUARD_HI_RATIO",
    "BASS_INTEGRATION_GUARD_LO_RATIO",
    "BASS_INTEGRATION_PROFILE_WEIGHTS",
    "bass_integration_profile_weights",
]
