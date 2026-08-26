"""Early environment bootstrap for Intel Arc XPU & ComfyUI.

Executed automatically by Python on startup before any script or custom node.
Initializes early guards (e.g. torchaudio CUDA fallback) to prevent import-time crashes.
"""

try:
    from patches.torchaudio_guard import apply as apply_torchaudio_guard
    apply_torchaudio_guard()
except Exception:
    pass
