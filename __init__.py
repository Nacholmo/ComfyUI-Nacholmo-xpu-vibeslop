"""
ComfyUI Intel Arc & XPU Suite
─────────────────────────────
Unified enhancement and performance toolkit for Intel Arc GPUs & PyTorch XPU.

Included custom nodes:
  - ArcSuperResolution         → OpenVINO XMX-accelerated AI Super Resolution for images & video
  - ArcResampleFPS             → Duration-locked audio-synced video frame rate resampler
  - TorchCompileBlockwise      → LowVRAM-safe blockwise Dynamo/Inductor compilation
  - UpscaleVideoWithModel      → Batched video frame upscaler with torch.compile acceleration
  - VideoCombineSync           → Video combine node with pitch-preserving atempo audio sync
  - WINT8ModelQuantizer        → UNet BF16/FP16/FP8 to INT8 quantizer
  - WINT8ModelLoader           → INT8 UNet loader with fast XPU kernels
  - WINT8LoRALoader            → Standalone INT8 LoRA loader
  - WINT8LoRAStack             → Multi-LoRA stack for INT8 models (up to 5)

Runtime enhancements:
  - VRAM Guard (auto-caps XPU allocator to avoid driver lockups on Level Zero)
  - MiniMax-H3 memory factor calibration override
"""

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

# 1. Apply system patches & stability guards
from patches import apply_all_patches
apply_all_patches()

# 2. Import and expose node mappings
from nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
