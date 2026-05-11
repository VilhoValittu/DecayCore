import importlib
import sys

_module = importlib.import_module("decaycore.resources.i8n.decaycore_i18n")
sys.modules[__name__] = _module
