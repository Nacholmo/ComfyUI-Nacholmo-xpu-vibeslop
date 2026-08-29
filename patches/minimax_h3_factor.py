"""MiniMax-H3 memory calibration and conditioning keyframe auto-rescaling patches.

1. Memory Usage Factor:
   Keeps the measured MiniMaxH3 memory_usage_factor calibration across ComfyUI updates.
   The sampling working set measured on Arc B580 with int8 weights needs ~0.18
   (upstream still ships 0.114).

2. Conditioning Keyframe Auto-Rescale:
   When upscaled video latents (e.g. 18x32 -> 36x62 via MiniMaxH3LatentUpscaler) are passed
   to a MiniMax sampler alongside image-to-video / keyframe conditioning, the conditioning
   latents default to the old spatial size (18x32). This patch automatically rescales
   the conditioning latents to match the target canvas layout, preventing shape mismatch crashes:
   `cannot broadcast [144, 96] to indexing result of shape [558, 96]`.
"""

import logging
import sys
import torch
import torch.nn.functional as F
from importlib.machinery import PathFinder

log = logging.getLogger("ComfyUI-Nacholmo-xpu-vibeslop.MiniMaxH3Patches")

MEASURED_FACTOR = 0.18
_INSTALLED = False


def _patch_minimax_model_module(mod):
    if getattr(mod, "_nacholmo_keyframe_rescale_patched", False):
        return

    diff_model_cls = getattr(mod, "MiniMaxH3Model", None) or getattr(mod, "DiffusionModel", None)
    if diff_model_cls and hasattr(diff_model_cls, "_cond_video_rows"):
        orig_cond_video_rows = diff_model_cls._cond_video_rows

        def safe_cond_video_rows(self, payload, device):
            layout = payload.get("layout")
            target_hw = None
            if layout is not None and hasattr(layout, "signature") and len(layout.signature) >= 4:
                target_hw = (layout.signature[2], layout.signature[3])

            rows = []
            aug = payload.get("visual_cond_noise_aug", getattr(mod, "VISUAL_COND_TIMESTEP", 0.999))
            seed = int(payload.get("seed", 0))

            for z in payload.get("cond_video_latents", []):
                if target_hw is not None and z.shape[-2:] != target_hw:
                    B, C, T, H, W = z.shape
                    z = F.interpolate(
                        z.view(B * T, C, H, W).to(torch.float32),
                        size=target_hw,
                        mode="bilinear",
                        align_corners=False
                    ).view(B, C, T, target_hw[0], target_hw[1])

                r = mod.patchify_video(z.to(torch.float32), self.patch_size)
                if aug < 1.0:
                    gen = torch.Generator("cpu").manual_seed(seed)
                    noise = torch.randn(r.shape, generator=gen, dtype=torch.float32)
                    r = aug * r + (1.0 - aug) * noise.to(r.device)
                rows.append(r.to(device))

            return torch.cat(rows, dim=0) if rows else None

        diff_model_cls._cond_video_rows = safe_cond_video_rows
        mod._nacholmo_keyframe_rescale_patched = True
        print("[MiniMaxH3-Patches] Enabled automatic keyframe conditioning resolution auto-scaling")


class _MiniMaxModelMetaFinder:
    @classmethod
    def find_spec(cls, fullname, path=None, target=None):
        if "comfy.ldm.minimax.model" in fullname:
            spec = PathFinder.find_spec(fullname, path, target)
            if spec and spec.loader:
                orig_exec = spec.loader.exec_module

                def exec_module_patched(module):
                    orig_exec(module)
                    try:
                        _patch_minimax_model_module(module)
                    except Exception as e:
                        log.debug(f"Failed to patch {fullname}: {e}")

                spec.loader.exec_module = exec_module_patched
            return spec
        return None


def apply():
    global _INSTALLED

    # 1. Memory Factor Override
    try:
        import comfy.supported_models
        if hasattr(comfy.supported_models, "MiniMaxH3"):
            _current = getattr(comfy.supported_models.MiniMaxH3, "memory_usage_factor", None)
            if _current is not None and _current != MEASURED_FACTOR:
                if _current != 0.114:
                    log.warning(
                        "[minimax-h3-memory-factor] upstream MiniMaxH3.memory_usage_factor changed to %s, "
                        "revisit this override.", _current)
                comfy.supported_models.MiniMaxH3.memory_usage_factor = MEASURED_FACTOR
    except Exception as e:
        log.debug(f"[minimax-h3-memory-factor] could not apply override: {e}")

    # 2. Keyframe auto-rescaling patch
    if "comfy.ldm.minimax.model" in sys.modules:
        mod = sys.modules["comfy.ldm.minimax.model"]
        if mod is not None:
            try:
                _patch_minimax_model_module(mod)
            except Exception as e:
                log.debug(f"Could not patch already loaded comfy.ldm.minimax.model: {e}")

    if not _INSTALLED:
        already_in_meta = any(
            getattr(finder, "__name__", "") == "_MiniMaxModelMetaFinder" or finder is _MiniMaxModelMetaFinder
            for finder in sys.meta_path
        )
        if not already_in_meta:
            sys.meta_path.insert(0, _MiniMaxModelMetaFinder)
        _INSTALLED = True


apply()

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ["apply", "NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
