"""Unified node exporter for Intel Arc & XPU ComfyUI Toolkit."""

import logging
import os

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

# 2. TorchCompile Blockwise (LowVRAM-Safe) — DEPRECATED, opt-in only.
# Omni's CUTE fused attention is the default compile path now, and this
# module's import-time torch._dynamo.disable patches on comfy.ops fight
# Kitchen/Omni dispatch. Set NACHOLMO_TORCHCOMPILE=1 to re-enable.
if os.environ.get("NACHOLMO_TORCHCOMPILE", "0") != "0":
    try:
        from .torch_compile_model import (
            NODE_CLASS_MAPPINGS as _COMPILE_CLASS_MAPPINGS,
            NODE_DISPLAY_NAME_MAPPINGS as _COMPILE_DISPLAY_NAME_MAPPINGS,
        )
        NODE_CLASS_MAPPINGS.update(_COMPILE_CLASS_MAPPINGS)
        NODE_DISPLAY_NAME_MAPPINGS.update(_COMPILE_DISPLAY_NAME_MAPPINGS)
    except Exception as e:
        log.warning(f"[Intel-Arc-Suite] Could not load TorchCompile node: {e}")
else:
    log.info("[Intel-Arc-Suite] TorchCompileBlockwise disabled (deprecated: use OmniXPU CUTE attention; opt in with NACHOLMO_TORCHCOMPILE=1)")

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

# 5. WINT8 Suite (Quantizer, Loader, LoRA Loader, LoRA Stack) — DEPRECATED, opt-in only.
# Kitchen XPU (GGUF/SVDQuant/FP8/INT8 via GGUF-XPU + nunchaku-XPU companions)
# plus OmniXPU INT8-FFN/FP8 adapters supersede this path, and wint8_xpu_ops'
# import-time torch dtype + inductor patches fight Kitchen dispatch.
# Set NACHOLMO_WINT8=1 to re-enable for legacy workflows.
if os.environ.get("NACHOLMO_WINT8", "0") != "0":
    try:
        from .wint8 import (
            NODE_CLASS_MAPPINGS as _WINT8_CLASS_MAPPINGS,
            NODE_DISPLAY_NAME_MAPPINGS as _WINT8_DISPLAY_NAME_MAPPINGS,
        )
        NODE_CLASS_MAPPINGS.update(_WINT8_CLASS_MAPPINGS)
        NODE_DISPLAY_NAME_MAPPINGS.update(_WINT8_DISPLAY_NAME_MAPPINGS)
    except Exception as e:
        log.warning(f"[Intel-Arc-Suite] Could not load WINT8 suite: {e}")
else:
    log.info("[Intel-Arc-Suite] WINT8 suite disabled (deprecated: use Kitchen GGUF/SVDQuant + OmniXPU INT8/FP8; opt in with NACHOLMO_WINT8=1)")

# 6. MiniMax-H3 Turbo (Turbo LoRA & 4-Step Sampler)
try:
    from .minimax_turbo import (
        NODE_CLASS_MAPPINGS as _MINIMAX_CLASS_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS as _MINIMAX_DISPLAY_NAME_MAPPINGS,
    )
    NODE_CLASS_MAPPINGS.update(_MINIMAX_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(_MINIMAX_DISPLAY_NAME_MAPPINGS)
except Exception as e:
    log.warning(f"[Intel-Arc-Suite] Could not load MiniMax-H3 Turbo nodes: {e}")

# 7. MiniMax-H3 Reference to Video (Stride / Lite) – VRAM saver
try:
    from .minimax_reference_stride import (
        NODE_CLASS_MAPPINGS as _MINIMAX_STRIDE_CLASS_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS as _MINIMAX_STRIDE_DISPLAY_NAME_MAPPINGS,
    )
    NODE_CLASS_MAPPINGS.update(_MINIMAX_STRIDE_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(_MINIMAX_STRIDE_DISPLAY_NAME_MAPPINGS)
except Exception as e:
    log.warning(f"[Intel-Arc-Suite] Could not load MiniMax-H3 Reference Stride node: {e}")

# 8. Sol-Attn Sparse Block Attention Node (Intel Arc XPU 3-4x Speedup)
try:
    from .sol_attn_node import (
        NODE_CLASS_MAPPINGS as _SOL_ATTN_CLASS_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS as _SOL_ATTN_DISPLAY_NAME_MAPPINGS,
    )
    NODE_CLASS_MAPPINGS.update(_SOL_ATTN_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(_SOL_ATTN_DISPLAY_NAME_MAPPINGS)
except Exception as e:
    log.warning(f"[Intel-Arc-Suite] Could not load Sol-Attn node: {e}")

# 9. MiniMax-H3 Latent Split & Extend Upscaling Suite
try:
    from .minimax_extend_split import (
        NODE_CLASS_MAPPINGS as _EXTEND_SPLIT_CLASS_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS as _EXTEND_SPLIT_DISPLAY_NAME_MAPPINGS,
    )
    NODE_CLASS_MAPPINGS.update(_EXTEND_SPLIT_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(_EXTEND_SPLIT_DISPLAY_NAME_MAPPINGS)
except Exception as e:
    log.warning(f"[Intel-Arc-Suite] Could not load MiniMax-H3 Extend Split nodes: {e}")

# 10. Video Concat Extend (temporal A/V join + color match)
try:
    from .video_concat_extend import (
        NODE_CLASS_MAPPINGS as _CONCAT_EXTEND_CLASS_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS as _CONCAT_EXTEND_DISPLAY_NAME_MAPPINGS,
    )
    NODE_CLASS_MAPPINGS.update(_CONCAT_EXTEND_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(_CONCAT_EXTEND_DISPLAY_NAME_MAPPINGS)
except Exception as e:
    log.warning(f"[Intel-Arc-Suite] Could not load VideoConcatExtend node: {e}")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
