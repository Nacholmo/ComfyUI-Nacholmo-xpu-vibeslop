"""
ComfyUI Nacholmo XPU Vibeslop
─────────────────────────────
Unified enhancement and performance toolkit for Intel Arc GPUs & PyTorch XPU.

Included custom nodes:
  - ArcSuperResolution         → OpenVINO XMX-accelerated AI Super Resolution for images & video
  - ArcResampleFPS             → Duration-locked audio-synced video frame rate resampler
  - UpscaleVideoWithModel      → Batched video frame upscaler with torch.compile acceleration
  - VideoCombineSync           → Video combine node with pitch-preserving atempo audio sync
  - MiniMax-H3 Turbo/Stride/Extend → Turbo LoRA + sampler, stride ref-to-video, extend-split
  - ApplySolAttn               → Sparse block attention opt-in (OmniXPU CUTE dense is default)

Deprecated (opt-in via env, superseded by OmniXPU/Kitchen):
  - TorchCompileBlockwise      → NACHOLMO_TORCHCOMPILE=1 (use OmniXPU CUTE instead)
  - WINT8 quantizer/loader/LoRA → NACHOLMO_WINT8=1 (use Kitchen GGUF/SVDQuant + OmniXPU INT8/FP8)

Runtime enhancements:
  - VRAM Guard (auto-caps XPU allocator to avoid driver lockups on Level Zero)
  - MiniMax-H3 memory factor calibration override
"""

import os
import sys

# 1. Apply system patches & stability guards
from .patches import apply_all_patches
apply_all_patches()

# 2. Import and expose node mappings
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# 3. Web extensions & DarkComfyX Theme
WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]


