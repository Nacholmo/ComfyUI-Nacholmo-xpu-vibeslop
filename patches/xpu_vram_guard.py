"""Cap the torch XPU caching allocator below physical VRAM.

Level Zero sometimes hangs the whole system when an allocation exceeds
available VRAM. Raising a clean OutOfMemoryError from the allocator lets
ComfyUI recover (unload models, retry) instead of freezing the desktop.
"""

import logging
import os

log = logging.getLogger("ComfyUI-Nacholmo-xpu-vibeslop.VRAMGuard")

_APPLIED = False


def apply():
    global _APPLIED
    if _APPLIED:
        return
    try:
        import torch
        if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
            return
        frac = float(os.environ.get("XPU_VRAM_FRACTION", "0.75"))
        dev_count = torch.xpu.device_count()
        for dev in range(dev_count):
            torch.xpu.set_per_process_memory_fraction(frac, dev)
        _APPLIED = True
        print(f"[xpu-vram-guard] allocator capped at {frac:.0%} of VRAM across {dev_count} device(s) (override with XPU_VRAM_FRACTION)")
    except Exception as e:
        log.debug(f"[xpu-vram-guard] could not set memory fraction: {e}")


__all__ = ["apply"]
