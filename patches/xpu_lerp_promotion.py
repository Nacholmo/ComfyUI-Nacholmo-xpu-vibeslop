"""XPU torch.lerp dtype-promotion shim.

OneDNN (XPU backend) does not type-promote mixed-dtype `torch.lerp` calls,
while CUDA/CPU silently promote. Core's MiniMax-H3 forward hits this in the
adaLN curve lookup (`comfy/ldm/minimax/model.py`, DiffusionModel._forward):

    t_emb = torch.lerp(table[i0], table[i0 + 1], (pos - i0).unsqueeze(1))

whenever the table dtype (e.g. F16 in GGUF files) differs from the float32
blend weight, aborting sampling with:
`RuntimeError: expected dtype c10::Half for 'weight' but got dtype float`.

This shim wraps `torch.lerp` and explicitly promotes mixed-dtype *tensor*
inputs to their common type first, reproducing CUDA/CPU promotion semantics.
Same-dtype calls (the hot path) pass through untouched: zero behavior or
performance change. Scalar-weight calls are never touched.
"""

import functools
import logging

import torch

log = logging.getLogger("ComfyUI-Nacholmo-xpu-vibeslop.XPULerp")

_INSTALLED = False


def _tensor_dtypes(args):
    return {a.dtype for a in args if isinstance(a, torch.Tensor)}


def apply():
    global _INSTALLED
    if _INSTALLED or getattr(torch.lerp, "_nacholmo_lerp_patched", False):
        _INSTALLED = True
        return

    orig_lerp = torch.lerp

    @functools.wraps(orig_lerp)
    def lerp_promoted(input, end, weight, *, out=None):
        # In-place variant: never intervene, preserve exact semantics.
        if out is not None:
            return orig_lerp(input, end, weight, out=out)
        dtypes = _tensor_dtypes((input, end, weight))
        if len(dtypes) > 1:
            target = input.dtype
            for a in (end, weight):
                if isinstance(a, torch.Tensor):
                    target = torch.promote_types(target, a.dtype)
            try:
                if isinstance(input, torch.Tensor) and input.dtype != target:
                    input = input.to(target)
                if isinstance(end, torch.Tensor) and end.dtype != target:
                    end = end.to(target)
                if isinstance(weight, torch.Tensor) and weight.dtype != target:
                    weight = weight.to(target)
            except Exception as e:
                log.debug(f"[xpu-lerp] promotion to {target} failed, calling through: {e}")
        return orig_lerp(input, end, weight)

    lerp_promoted._nacholmo_lerp_patched = True
    torch.lerp = lerp_promoted
    _INSTALLED = True
    print("[XPU-Lerp] Enabled torch.lerp dtype-promotion shim for Intel XPU")


NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ["apply", "NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
