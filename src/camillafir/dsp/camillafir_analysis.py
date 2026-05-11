import importlib
import sys

_module = importlib.import_module("decaycore.dsp.decaycore_analysis")
sys.modules[__name__] = _module
