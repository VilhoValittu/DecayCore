import importlib
import sys

_module = importlib.import_module("decaycore.dsp.decaycore_dsp_parts")
sys.modules[__name__] = _module
