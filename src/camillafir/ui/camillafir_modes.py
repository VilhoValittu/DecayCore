import importlib
import sys

_module = importlib.import_module("decaycore.ui.decaycore_modes")
sys.modules[__name__] = _module
