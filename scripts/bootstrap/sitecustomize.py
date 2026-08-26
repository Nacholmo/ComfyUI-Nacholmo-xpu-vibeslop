"""Early environment bootstrap for Intel Arc XPU & ComfyUI.

Executed automatically by Python on startup before any script or custom node.
Initializes early guards (e.g. torchaudio CUDA fallback) to prevent import-time crashes.
"""

import os
import sys
import importlib.util

_BOOTSTRAP_DIR = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS_DIR = os.path.dirname(_BOOTSTRAP_DIR)
_SUITE_DIR = os.path.dirname(_SCRIPTS_DIR)
_GUARD_PATH = os.path.join(_SUITE_DIR, "patches", "torchaudio_guard.py")

if os.path.exists(_GUARD_PATH):
    try:
        spec = importlib.util.spec_from_file_location("_nacholmo_early_torchaudio_guard", _GUARD_PATH)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if hasattr(mod, "apply"):
            mod.apply()
    except Exception:
        pass
