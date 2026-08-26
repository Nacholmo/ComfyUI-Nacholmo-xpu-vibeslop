"""System and runtime patches for Intel Arc XPU stability and calibration."""

import logging

log = logging.getLogger("ComfyUI-Intel-Arc-Suite.Patches")


def apply_vram_guard():
    try:
        from .xpu_vram_guard import apply
        apply()
    except Exception as e:
        log.debug(f"VRAM guard not applied: {e}")


def apply_minimax_factor():
    try:
        from .minimax_h3_factor import apply
        apply()
    except Exception as e:
        log.debug(f"MiniMax factor not applied: {e}")


def apply_all_patches():
    apply_vram_guard()
    apply_minimax_factor()


__all__ = ["apply_vram_guard", "apply_minimax_factor", "apply_all_patches"]
