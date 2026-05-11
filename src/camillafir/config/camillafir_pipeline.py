import importlib
import sys

_module = importlib.import_module("decaycore.config.decaycore_pipeline")
sys.modules[__name__] = _module
