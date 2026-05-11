import importlib
import sys

_module = importlib.import_module("decaycore.io.decaycore_automatic_mode")
sys.modules[__name__] = _module
