from __future__ import annotations

import importlib
import logging
from typing import Callable

logger = logging.getLogger("DecayCore")


def reset_runtime_caches() -> None:
    """Clear run-scoped runtime caches before a new run starts.

    This resets selected in-memory helper caches only. It does not remove on-disk
    auto-mode cache files or ``__pycache__`` bytecode folders. The bounded,
    content-addressed fallback RT60 cache intentionally survives across runs.
    """
    import gc

    try:
        importlib.invalidate_caches()
    except Exception:
        logger.exception("Python import cache invalidation failed")

    from ..dsp import (
        decaycore_analysis,
        leveling_parts as decaycore_leveling,
        dsp_preprocess,
        smoothing,
    )
    from ..ui.results_sections.plots_export import clear_plot_render_cache
    from ..ui.target_preview_cache import clear_target_preview_caches

    clearers: list[tuple[str, Callable[[], None]]] = [
        ("analysis", decaycore_analysis.clear_analysis_cache),
        ("leveling", decaycore_leveling._clear_leveling_cache),
        ("level_window", decaycore_leveling._clear_level_window_cache),
        ("smoothing", smoothing.clear_smoothing_cache),
        ("preprocess", dsp_preprocess.clear_preprocess_cache),
        ("plot_render_cache", clear_plot_render_cache),
        ("target_preview_cache", clear_target_preview_caches),
    ]

    from ..features import has_packaged_auto_engine, has_packaged_bass_engine

    if has_packaged_auto_engine():
        from ..auto_mode import cache_signature, filter_priors, optuna_backend_storage

        clearers[:0] = [
            ("auto_mode_filter_priors", filter_priors.clear_auto_mode_filter_priors_cache),
            ("auto_mode_synth_target", cache_signature.clear_auto_mode_runtime_caches),
            ("optuna_backend", optuna_backend_storage.clear_optuna_runtime_caches),
        ]
    if has_packaged_bass_engine():
        from ..dsp import bass_integration

        clearers.append(("bass_integration", bass_integration.clear_bass_integration_caches))

    cleared = 0
    for cache_name, clearer in clearers:
        try:
            clearer()
            cleared += 1
        except Exception:
            logger.exception("Runtime cache reset failed: %s", cache_name)

    gc.collect()
    logger.debug("Reset %d runtime caches before run", cleared)
