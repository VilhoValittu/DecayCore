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

_BUTTER_RESPONSE_CACHE: dict = {}


def clear_bass_integration_caches() -> None:
    """Clear module-level bass-integration caches shared across runs."""
    _BUTTER_RESPONSE_CACHE.clear()


def _get_filtered_branch_cache(bundle) -> dict:
    """Return the session-scope filtered-branch cache attached to the bundle."""
    try:
        return object.__getattribute__(bundle, "_camillafir_filtered_branch_cache")
    except AttributeError:
        cache: dict = {}
        object.__setattr__(bundle, "_camillafir_filtered_branch_cache", cache)
        return cache


def _metrics_cache_key(
    fc_hz: float,
    profile: str,
    *,
    mode: str,
    main_hpf_order: int,
    sub_lpf_order: int,
    sub_hpf_hz: float,
    sub_hpf_order: int,
    sub_combine_mode: str,
    sub_delay_ms: float,
    sub_polarity_invert: bool,
    sub_gain_trim_db: float,
    sub_lpf_hz: float | None,
    sub_allpass_freq_hz: float | None,
    sub_allpass_q: float | None,
    guard_lo_ratio: float,
    guard_hi_ratio: float,
    realized_fir_signature: tuple | None = None,
    robust: bool = False,
) -> tuple:
    return (
        round(float(fc_hz), 4),
        str(profile),
        str(mode),
        int(main_hpf_order),
        int(sub_lpf_order),
        round(float(sub_hpf_hz), 4),
        int(sub_hpf_order),
        str(sub_combine_mode),
        round(float(sub_delay_ms), 4),
        bool(sub_polarity_invert),
        round(float(sub_gain_trim_db), 4),
        round(float(sub_lpf_hz), 4) if sub_lpf_hz is not None else None,
        round(float(sub_allpass_freq_hz), 4) if sub_allpass_freq_hz is not None else None,
        round(float(sub_allpass_q), 5) if sub_allpass_q is not None else None,
        round(float(guard_lo_ratio), 4),
        round(float(guard_hi_ratio), 4),
        tuple(realized_fir_signature or ()),
        bool(robust),
    )


def _get_metrics_cache(bundle) -> dict:
    """Return the session-scope metrics cache dict attached to the bundle object."""
    try:
        return object.__getattribute__(bundle, "_camillafir_metrics_cache")
    except AttributeError:
        cache: dict = {}
        object.__setattr__(bundle, "_camillafir_metrics_cache", cache)
        object.__setattr__(bundle, "_camillafir_metrics_cache_hits", 0)
        object.__setattr__(bundle, "_camillafir_metrics_cache_misses", 0)
        return cache


def _get_combined_sub_cache(bundle) -> dict:
    """Return session-scope combined-sub transfer cache attached to the bundle."""
    try:
        return object.__getattribute__(bundle, "_camillafir_combined_sub_cache")
    except AttributeError:
        cache: dict = {}
        object.__setattr__(bundle, "_camillafir_combined_sub_cache", cache)
        return cache


def _combined_sub_cache_key(
    channel: str,
    mode_norm: str,
    max_lag_ms: float,
    min_confidence: float,
) -> tuple:
    return (str(channel), str(mode_norm), round(float(max_lag_ms), 4), round(float(min_confidence), 5))


def increment_metrics_cache_hit(bundle) -> None:
    try:
        hits = object.__getattribute__(bundle, "_camillafir_metrics_cache_hits")
        object.__setattr__(bundle, "_camillafir_metrics_cache_hits", hits + 1)
    except (AttributeError, TypeError, ValueError):
        return


def increment_metrics_cache_miss(bundle) -> None:
    try:
        misses = object.__getattribute__(bundle, "_camillafir_metrics_cache_misses")
        object.__setattr__(bundle, "_camillafir_metrics_cache_misses", misses + 1)
    except (AttributeError, TypeError, ValueError):
        return
