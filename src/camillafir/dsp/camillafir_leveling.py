import importlib
import sys

_module = importlib.import_module("decaycore.dsp.decaycore_leveling")
sys.modules[__name__] = _module
