"""Early environment bootstrap for Intel Arc XPU & ComfyUI.

Executed automatically by Python on startup before any script or custom node.
Initializes early guards (e.g. torchaudio CUDA fallback) to prevent import-time crashes.
"""

try:
    import os as _os
    import importlib.util as _ilu
    _guard = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "patches", "torchaudio_guard.py")
    if _os.path.exists(_guard):
        _spec = _ilu.spec_from_file_location("_nacholmo_early_torchaudio_guard", _guard)
        _mod = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        if hasattr(_mod, "apply"):
            _mod.apply()
    else:
        # Fallback for PYTHONPATH layouts where repo root is on sys.path.
        from patches.torchaudio_guard import apply as apply_torchaudio_guard
        apply_torchaudio_guard()
except Exception:
    pass
