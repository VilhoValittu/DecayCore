from __future__ import annotations

import importlib
import logging
from typing import Callable

logger = logging.getLogger("DecayCore")


def reset_runtime_caches() -> None:
    """Clear process-local runtime caches before a new run starts.

    This resets in-memory helper caches only. It does not remove on-disk
    auto-mode cache files or ``__pycache__`` bytecode folders.
    """
    try:
        importlib.invalidate_caches()
    except Exception:
        logger.exception("Python import cache invalidation failed")

    from ..auto_mode import cache_signature, filter_priors, optuna_backend_storage
    from ..dsp import (
        bass_integration,
        decaycore_analysis,
        decaycore_leveling,
        correction_baseline,
        dsp_preprocess,
        smoothing,
    )

    clearers: list[tuple[str, Callable[[], None]]] = [
        ("auto_mode_filter_priors", filter_priors.clear_auto_mode_filter_priors_cache),
        ("auto_mode_synth_target", cache_signature.clear_auto_mode_runtime_caches),
        ("optuna_backend", optuna_backend_storage.clear_optuna_runtime_caches),
        ("analysis", decaycore_analysis.clear_analysis_cache),
        ("bass_integration", bass_integration.clear_bass_integration_caches),
        ("leveling", decaycore_leveling._clear_leveling_cache),
        ("level_window", decaycore_leveling._clear_level_window_cache),
        ("rt60", correction_baseline._clear_rt60_cache),
        ("smoothing", smoothing.clear_smoothing_cache),
        ("preprocess", dsp_preprocess.clear_preprocess_cache),
    ]

    cleared = 0
    for cache_name, clearer in clearers:
        try:
            clearer()
            cleared += 1
        except Exception:
            logger.exception("Runtime cache reset failed: %s", cache_name)

    logger.debug("Reset %d runtime caches before run", cleared)
