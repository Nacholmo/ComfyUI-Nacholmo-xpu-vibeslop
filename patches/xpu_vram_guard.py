"""Cap the torch XPU caching allocator below physical VRAM.

Level Zero sometimes hangs the whole system when an allocation exceeds
available VRAM. Raising a clean OutOfMemoryError from the allocator lets
ComfyUI recover (unload models, retry) instead of freezing the desktop.

Hybrid policy (Omni-aligned): when ComfyUI runs with --enable-dynamic-vram
and the OmniXPU provider bootstrap is active, AIMDO VBAR owns the allocator
and our fraction cap stays dormant (it would only fight AIMDO's headroom
management). The debug_memory_summary override still applies in both modes.
"""

import logging
import os
import sys

log = logging.getLogger("ComfyUI-Nacholmo-xpu-vibeslop.VRAMGuard")

_APPLIED = False


def _aimdo_owns_allocator():
    """True when VRAM mode hands allocation to the AIMDO XPU provider.

    Order-independent check (no dependence on prestartup script ordering):
    VRAM mode is requested via --enable-dynamic-vram and the OmniXPU master
    switch + provider bootstrap are not disabled. Also returns True if the
    AIMDO XPU control module is already live in this process.
    """
    try:
        control = sys.modules.get("comfy_aimdo.control")
        if control is not None and (
            getattr(control, "_xpu_allocator_ready", False)
            or getattr(control, "_torch_allocator", None) is not None
        ):
            return True
    except Exception:
        pass
    try:
        if os.environ.get("OMNIXPU_ENABLE", "1") == "0":
            return False
        if os.environ.get("OMNIXPU_PROVIDER_BOOTSTRAP", "auto").strip().lower() == "off":
            return False
        if "--enable-dynamic-vram" in sys.argv:
            return True
        cli_args = sys.modules.get("comfy.cli_args")
        args = getattr(cli_args, "args", None)
        if args is not None and bool(getattr(args, "enable_dynamic_vram", False)):
            return True
    except Exception:
        pass
    return False


def apply():
    global _APPLIED
    if _APPLIED:
        return
    try:
        import torch
    except ImportError:
        return
    # The prestartup hook exec's this file as a standalone module, so a second
    # copy runs again at custom-node import time. Dedupe on the shared torch
    # module object instead of this file's globals.
    if getattr(torch, "_nacholmo_vram_guard_applied", False):
        _APPLIED = True
        return
    torch._nacholmo_vram_guard_applied = True
    if _aimdo_owns_allocator():
        _APPLIED = True
        print("[xpu-vram-guard] dormant: AIMDO DynamicVRAM owns the XPU allocator (VRAM mode)")
    else:
        try:
            if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
                return
            frac = float(os.environ.get("XPU_VRAM_FRACTION", "0.90"))
            dev_count = torch.xpu.device_count()
            for dev in range(dev_count):
                torch.xpu.set_per_process_memory_fraction(frac, dev)
            _APPLIED = True
            print(f"[xpu-vram-guard] allocator capped at {frac:.0%} of VRAM across {dev_count} device(s) (override with XPU_VRAM_FRACTION)")
        except Exception as e:
            log.debug(f"[xpu-vram-guard] could not set memory fraction: {e}")

    try:
        import comfy.model_management as mm

        def _safe_debug_memory_summary():
            try:
                import torch
                if hasattr(torch, "xpu") and torch.xpu.is_available():
                    return torch.xpu.memory_summary()
                elif hasattr(torch, "cuda") and torch.cuda.is_available():
                    return torch.cuda.memory.memory_summary()
            except Exception:
                pass
            return ""

        mm.debug_memory_summary = _safe_debug_memory_summary
    except Exception:
        pass


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
