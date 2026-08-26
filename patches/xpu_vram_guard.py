"""Cap the torch XPU caching allocator below physical VRAM.

Level Zero sometimes hangs the whole system when an allocation exceeds
available VRAM. Raising a clean OutOfMemoryError from the allocator lets
ComfyUI recover (unload models, retry) instead of freezing the desktop.
"""

import logging
import os
import sys

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


def install_deferred():
    """Install hook so VRAM guard is applied as soon as torch is imported, without eagerly importing torch in prestartup."""
    if "torch" in sys.modules:
        apply()
        return

    from importlib.machinery import PathFinder

    class _TorchGuardMetaFinder:
        @classmethod
        def find_spec(cls, fullname, path=None, target=None):
            if fullname == "torch":
                spec = PathFinder.find_spec(fullname, path, target)
                if spec and spec.loader:
                    orig_exec = spec.loader.exec_module

                    def exec_module_patched(module):
                        orig_exec(module)
                        try:
                            apply()
                        except Exception:
                            pass

                    spec.loader.exec_module = exec_module_patched
                return spec
            return None

    for finder in sys.meta_path:
        if getattr(finder, "__name__", "") == "_TorchGuardMetaFinder" or finder is _TorchGuardMetaFinder:
            return
    sys.meta_path.insert(0, _TorchGuardMetaFinder)


__all__ = ["apply", "install_deferred"]
