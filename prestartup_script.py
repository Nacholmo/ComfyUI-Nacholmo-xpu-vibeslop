import os
import sys
import importlib.util

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_and_run(rel_path, func_name):
    full_path = os.path.join(_THIS_DIR, rel_path)
    if os.path.exists(full_path):
        mod_name = "_nacholmo_prestartup_" + rel_path.replace("/", "_").replace(".", "_")
        spec = importlib.util.spec_from_file_location(mod_name, full_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        func = getattr(mod, func_name, None)
        if func:
            func()


try:
    _load_and_run("patches/torchaudio_guard.py", "apply")
except Exception:
    pass

try:
    _load_and_run("patches/xpu_vram_guard.py", "install_deferred")
except Exception:
    pass

# Version-check hold: must run here, not only at custom-node import time.
# Core's frontend init (which emits the version-compatibility banner) runs
# BEFORE custom nodes load, so registering in apply_all_patches() alone is
# too late. This module is torch-free, hence prestartup-safe.
try:
    _load_and_run("patches/comfy_version_check.py", "apply")
except Exception:
    pass

# Note: minimax_upscaler_xpu is intentionally NOT loaded here.
# It imports torch at the top level, which would trigger
# main.py:244 "Torch already imported" warning.
# It is applied lazily via patches/__init__.py:apply_all_patches()
# during custom_nodes loading (after the warning check) via its
# _MinimaxUpscalerMetaFinder + sys.modules scan, so early load
# is unnecessary.

