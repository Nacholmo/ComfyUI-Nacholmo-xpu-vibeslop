"""System and runtime patches for Intel Arc XPU stability and calibration."""

import logging

log = logging.getLogger("ComfyUI-Nacholmo-xpu-vibeslop.Patches")


def apply_torchaudio_guard():
    try:
        from .torchaudio_guard import apply
        apply()
    except Exception as e:
        log.debug(f"TorchAudio guard not applied: {e}")


def apply_vram_guard():
    try:
        from .xpu_vram_guard import apply, patch_debug_memory_summary
        apply()
        patch_debug_memory_summary()
    except Exception as e:
        log.debug(f"VRAM guard not applied: {e}")


def install_deferred_vram_guard():
    try:
        from .xpu_vram_guard import install_deferred
        install_deferred()
    except Exception as e:
        log.debug(f"Deferred VRAM guard hook not installed: {e}")


def apply_minimax_factor():
    try:
        from .minimax_h3_factor import apply
        apply()
    except Exception as e:
        log.debug(f"MiniMax factor not applied: {e}")


def apply_minimax_upscaler_xpu():
    try:
        from .minimax_upscaler_xpu import apply
        apply()
    except Exception as e:
        log.debug(f"MiniMax upscaler XPU patch not applied: {e}")


def apply_xpu_lerp_promotion():
    try:
        from .xpu_lerp_promotion import apply
        apply()
    except Exception as e:
        log.debug(f"XPU lerp promotion not applied: {e}")


def apply_version_check_hold():
    try:
        from .comfy_version_check import apply
        apply()
    except Exception as e:
        log.debug(f"Version-check hold not applied: {e}")


def apply_all_patches():
    apply_torchaudio_guard()
    apply_vram_guard()
    apply_minimax_factor()
    apply_minimax_upscaler_xpu()
    apply_xpu_lerp_promotion()
    apply_version_check_hold()


__all__ = [
    "apply_torchaudio_guard",
    "apply_vram_guard",
    "install_deferred_vram_guard",
    "apply_minimax_factor",
    "apply_minimax_upscaler_xpu",
    "apply_xpu_lerp_promotion",
    "apply_version_check_hold",
    "apply_all_patches",
]
