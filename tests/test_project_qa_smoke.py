import importlib
import importlib.util

import pytest


def test_core_modules_import_smoke():
    modules = (
        "decaycore.decaycore",
        "decaycore.auto_mode.rank_score",
        "decaycore.auto_mode.scoring_ranking",
        "decaycore.dsp.dsp_preprocess",
        "decaycore.dsp.decaycore_dsp",
        "decaycore.dsp.smoothing",
        "decaycore.dsp.bass_cache",
        "decaycore.dsp.bass_metrics",
        "decaycore.dsp.bass_search",
        "decaycore.dsp.bass_diagnostics",
    )

    for module_name in modules:
        assert importlib.import_module(module_name) is not None



@pytest.mark.gui
@pytest.mark.requires_nicegui
def test_gui_modules_import_smoke():
    if importlib.util.find_spec("nicegui") is None:
        pytest.skip("nicegui is not installed")

    modules = (
        "decaycore.__main__",
        "decaycore.ui.ng_app",
    )

    for module_name in modules:
        assert importlib.import_module(module_name) is not None
