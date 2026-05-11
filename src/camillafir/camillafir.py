import importlib
import sys

_module = importlib.import_module("decaycore.decaycore")
sys.modules[__name__] = _module
