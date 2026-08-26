"""
Upscale Video With Model for ComfyUI

Video-focused rework of the vanilla "Upscale Image (using Model)" node:
- Frames are batched through the upscale model in chunks instead of
  one-at-a-time (vanilla comfy.utils.tiled_scale loops the batch dim per frame)
- Whole-frame passes, no tiling
- Custom scale factor on top of the model scale (run a 4x model as 2x)
- Optional torch.compile of the upscale model, which pays off here because
  the weights stay resident and frame shapes repeat every call
"""

import logging
import weakref

import torch
import torch.nn.functional as F

import comfy.model_management
import comfy.utils

log = logging.getLogger("UpscaleVideo")

_COMPILED_UPSCALERS = weakref.WeakKeyDictionary()


def _compiled_for(model):
    compiled = _COMPILED_UPSCALERS.get(model)
    if compiled is None:
        compiled = torch.compile(model, dynamic=False)
        _COMPILED_UPSCALERS[model] = compiled
        log.info("[UpscaleVideo] Compiled upscale model")
    return compiled


class UpscaleVideoWithModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "upscale_model": ("UPSCALE_MODEL",),
                "image": ("IMAGE",),
                "scale_factor": ("FLOAT", {
                    "default": 1.0, "min": 0.25, "max": 8.0, "step": 0.05,
                    "tooltip": "Extra factor on top of the model scale, applied with antialiased bicubic after the model. Use 0.5 to turn a 4x model into a 2x upscaler.",
                }),
                "batch_size": ("INT", {
                    "default": 8, "min": 1, "max": 4096,
                    "tooltip": "Frames per forward pass. Lower this if you run out of memory.",
                }),
                "torch_compile": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Compile the upscale model with torch.compile. First run compiles, later runs reuse it.",
                }),
                "fp16_output": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Return half-precision frames. Halves system RAM for long videos; fine for video encoding, avoid if feeding float-sensitive nodes.",
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = "image/upscaling"

    def upscale(self, upscale_model, image, scale_factor, batch_size, torch_compile, fp16_output):
        device = upscale_model.patcher.load_device
        total_frames = image.shape[0]
        eff_scale = upscale_model.scale * scale_factor

        memory_required = (image.shape[1] * image.shape[2] * 3) * image.element_size() * max(eff_scale, 1.0) * 384.0 * min(batch_size, total_frames)
        comfy.model_management.load_models_gpu([upscale_model.patcher], memory_required=memory_required, force_full_load=True)

        orig_call_fn = None
        if torch_compile:
            compiled = _compiled_for(upscale_model.model)
            orig_call_fn = upscale_model._call_fn
            upscale_model._call_fn = lambda model, img: compiled(img)
        output_device = comfy.model_management.intermediate_device()
        out_dtype = torch.float16 if fp16_output else comfy.model_management.intermediate_dtype()
        s = torch.empty((total_frames, round(image.shape[1] * eff_scale), round(image.shape[2] * eff_scale), upscale_model.output_channels),
                        device=output_device, dtype=out_dtype)

        def run(x):
            out = upscale_model(x.float())
            if scale_factor != 1.0:
                size = (round(x.shape[2] * eff_scale), round(x.shape[3] * eff_scale))
                out = F.interpolate(out, size=size, mode="bicubic", antialias=True)
            return out

        try:
            bs = batch_size
            oom = True
            while oom:
                try:
                    pbar = comfy.utils.ProgressBar((total_frames + bs - 1) // bs)
                    for start in range(0, total_frames, bs):
                        x = image[start:start + bs].movedim(-1, -3).to(device)
                        frame_count = x.shape[0]
                        if torch_compile and frame_count < bs:
                            # pad the tail chunk up to full batch size so a compiled model
                            # only ever sees one input shape instead of recompiling for the remainder
                            x = torch.cat([x, x[-1:].expand(bs - frame_count, -1, -1, -1)])
                        chunk = run(x)[:frame_count].clamp_(0, 1.0)
                        # one fused copy: device transfer + dtype cast + NCHW->NHWC
                        s[start:start + frame_count].movedim(-1, -3).copy_(chunk)
                        pbar.update(1)
                    oom = False
                except Exception as e:
                    comfy.model_management.raise_non_oom(e)
                    if bs > 1:
                        bs = max(1, bs // 2)
                    else:
                        raise
        finally:
            if orig_call_fn is not None:
                upscale_model._call_fn = orig_call_fn

        return (s,)


NODE_CLASS_MAPPINGS = {
    "UpscaleVideoWithModel": UpscaleVideoWithModel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UpscaleVideoWithModel": "Upscale Video (using Model)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
