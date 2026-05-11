import importlib
import sys

_module = importlib.import_module("decaycore.ui.decaycore_housecurve")
sys.modules[__name__] = _module
