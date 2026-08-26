"""Unified node exporter for Intel Arc & XPU ComfyUI Toolkit."""

import logging

log = logging.getLogger("ComfyUI-Nacholmo-xpu-vibeslop")

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# 1. Arc OpenVINO Nodes (Super Resolution, FPS Resampling)
try:
    from .arc_nodes import (
        NODE_CLASS_MAPPINGS as _ARC_CLASS_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS as _ARC_DISPLAY_NAME_MAPPINGS,
    )
    NODE_CLASS_MAPPINGS.update(_ARC_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(_ARC_DISPLAY_NAME_MAPPINGS)
except Exception as e:
    log.warning(f"[Intel-Arc-Suite] Could not load Arc OpenVINO nodes: {e}")

# 2. TorchCompile Blockwise (LowVRAM-Safe)
try:
    from .torch_compile_model import (
        NODE_CLASS_MAPPINGS as _COMPILE_CLASS_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS as _COMPILE_DISPLAY_NAME_MAPPINGS,
    )
    NODE_CLASS_MAPPINGS.update(_COMPILE_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(_COMPILE_DISPLAY_NAME_MAPPINGS)
except Exception as e:
    log.warning(f"[Intel-Arc-Suite] Could not load TorchCompile node: {e}")

# 3. Upscale Video With Model
try:
    from .upscale_model_video import (
        NODE_CLASS_MAPPINGS as _UPSCALE_CLASS_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS as _UPSCALE_DISPLAY_NAME_MAPPINGS,
    )
    NODE_CLASS_MAPPINGS.update(_UPSCALE_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(_UPSCALE_DISPLAY_NAME_MAPPINGS)
except Exception as e:
    log.warning(f"[Intel-Arc-Suite] Could not load UpscaleVideo node: {e}")

# 4. Video Combine Sync (atempo audio duration sync)
try:
    from .video_combine_sync import (
        NODE_CLASS_MAPPINGS as _SYNC_CLASS_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS as _SYNC_DISPLAY_NAME_MAPPINGS,
    )
    NODE_CLASS_MAPPINGS.update(_SYNC_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(_SYNC_DISPLAY_NAME_MAPPINGS)
except Exception as e:
    log.warning(f"[Intel-Arc-Suite] Could not load VideoCombineSync node: {e}")

# 5. WINT8 Suite (Quantizer, Loader, LoRA Loader, LoRA Stack)
try:
    from .wint8 import (
        NODE_CLASS_MAPPINGS as _WINT8_CLASS_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS as _WINT8_DISPLAY_NAME_MAPPINGS,
    )
    NODE_CLASS_MAPPINGS.update(_WINT8_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(_WINT8_DISPLAY_NAME_MAPPINGS)
except Exception as e:
    log.warning(f"[Intel-Arc-Suite] Could not load WINT8 suite: {e}")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
