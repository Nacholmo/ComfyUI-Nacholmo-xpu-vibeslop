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


def _safe_xpu_memory_summary(device=None, abbreviated=False):
    """Format an XPU memory summary equivalent to torch.cuda.memory_summary()."""
    try:
        import torch
        if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
            return ""
        if device is None:
            dev_idx = torch.xpu.current_device() if hasattr(torch.xpu, "current_device") else 0
        elif isinstance(device, int):
            dev_idx = device
        else:
            dev_idx = getattr(device, "index", 0) or 0

        dev_name = torch.xpu.get_device_name(dev_idx) if hasattr(torch.xpu, "get_device_name") else f"device {dev_idx}"
        free_bytes, total_bytes = torch.xpu.mem_get_info(dev_idx) if hasattr(torch.xpu, "mem_get_info") else (0, 0)
        alloc_bytes = torch.xpu.memory_allocated(dev_idx) if hasattr(torch.xpu, "memory_allocated") else 0
        max_alloc_bytes = torch.xpu.max_memory_allocated(dev_idx) if hasattr(torch.xpu, "max_memory_allocated") else 0
        res_bytes = torch.xpu.memory_reserved(dev_idx) if hasattr(torch.xpu, "memory_reserved") else 0
        max_res_bytes = torch.xpu.max_memory_reserved(dev_idx) if hasattr(torch.xpu, "max_memory_reserved") else 0

        lines = [
            "=" * 75,
            f"  PyTorch XPU memory summary, device ID {dev_idx} ({dev_name})",
            "-" * 75,
            f"  VRAM Total: {total_bytes / (1024**3):.2f} GiB | Free: {free_bytes / (1024**3):.2f} GiB",
            f"  Allocated:  {alloc_bytes / (1024**2):.2f} MiB (Peak: {max_alloc_bytes / (1024**2):.2f} MiB)",
            f"  Reserved:   {res_bytes / (1024**2):.2f} MiB (Peak: {max_res_bytes / (1024**2):.2f} MiB)",
        ]
        if hasattr(torch.xpu, "memory_stats"):
            try:
                stats = torch.xpu.memory_stats(dev_idx)
                if stats and not abbreviated:
                    lines.append("-" * 75)
                    act = stats.get("active_bytes.all.current", 0) / (1024**2)
                    act_pk = stats.get("active_bytes.all.peak", 0) / (1024**2)
                    lines.append(f"  Active:     {act:.2f} MiB (Peak: {act_pk:.2f} MiB)")
            except Exception:
                pass
        lines.append("=" * 75)
        return "\n".join(lines) + "\n"
    except Exception as e:
        return f"XPU memory summary error: {e}\n"


def _safe_debug_memory_summary():
    try:
        import torch
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            if hasattr(torch.xpu, "memory_summary") and torch.xpu.memory_summary is not _safe_xpu_memory_summary:
                try:
                    return torch.xpu.memory_summary()
                except Exception:
                    pass
            return _safe_xpu_memory_summary()
        elif hasattr(torch, "cuda") and torch.cuda.is_available():
            try:
                return torch.cuda.memory_summary()
            except Exception:
                return torch.cuda.memory.memory_summary()
    except Exception:
        pass
    return ""


def patch_debug_memory_summary():
    try:
        import torch
        if hasattr(torch, "xpu") and not hasattr(torch.xpu, "memory_summary"):
            torch.xpu.memory_summary = _safe_xpu_memory_summary
    except Exception:
        pass

    try:
        import comfy.model_management as mm
        mm.debug_memory_summary = _safe_debug_memory_summary
    except Exception:
        pass


def apply():
    global _APPLIED
    try:
        import torch
    except ImportError:
        return

    # Always re-apply debug_memory_summary patch even if allocator fraction
    # was already capped (e.g. during prestartup before comfy.model_management
    # was loaded or overwritten).
    patch_debug_memory_summary()

    if _APPLIED or getattr(torch, "_nacholmo_vram_guard_applied", False):
        _APPLIED = True
        return
    torch._nacholmo_vram_guard_applied = True
    if _aimdo_owns_allocator():
        _APPLIED = True
        print("[xpu-vram-guard] dormant: AIMDO DynamicVRAM owns the XPU allocator (VRAM mode)")
    else:
        try:
            if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
                log.debug("[xpu-vram-guard] no XPU device; skipping allocator cap (memory-summary override still applies)")
            else:
                frac = float(os.environ.get("XPU_VRAM_FRACTION", "0.90"))
                dev_count = torch.xpu.device_count()
                for dev in range(dev_count):
                    torch.xpu.set_per_process_memory_fraction(frac, dev)
                print(f"[xpu-vram-guard] allocator capped at {frac:.0%} of VRAM across {dev_count} device(s) (override with XPU_VRAM_FRACTION)")
            _APPLIED = True
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


__all__ = ["apply", "install_deferred", "patch_debug_memory_summary", "_safe_debug_memory_summary", "_safe_xpu_memory_summary"]
