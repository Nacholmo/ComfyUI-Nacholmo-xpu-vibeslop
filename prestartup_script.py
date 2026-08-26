"""ComfyUI Prestartup Script for Intel Arc & XPU Toolkit.

Automatically executed by ComfyUI server at startup before loading models or graph.
Caps the PyTorch XPU caching allocator to protect against driver-level hard freezes.
"""

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

try:
    from patches.torchaudio_guard import apply as apply_torchaudio_guard
    apply_torchaudio_guard()
except Exception:
    pass

try:
    from patches.xpu_vram_guard import install_deferred as install_deferred_vram_guard
    install_deferred_vram_guard()
except Exception:
    pass
