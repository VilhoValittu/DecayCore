import importlib
import sys

_module = importlib.import_module("decaycore.io.decaycore_wav_window")
sys.modules[__name__] = _module
