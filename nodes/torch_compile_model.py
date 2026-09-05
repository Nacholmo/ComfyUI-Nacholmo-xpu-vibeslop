"""
TorchCompile Model Node for ComfyUI

Provides optimized torch.compile support with:
- Individual transformer block compilation (faster compilation, lower memory overhead)
- Automatic eager graph-breaks for ComfyUI weight-casting in LowVRAM / Offload mode
- Guard filtering for transformer_options to prevent recompiles across sampling steps
"""

import logging
import os
import weakref
import torch
from torch._inductor import list_mode_options
import comfy.utils
import comfy.ops
from comfy.patcher_extension import WrappersMP, WrapperExecutor

log = logging.getLogger("TorchCompileModel")

_COMPILE_KEY = "torch.compile"
_COMPILED_MODELS_CACHE = weakref.WeakKeyDictionary()
_ops_patched_for_compile = False

# launch_xpu.sh sets COMFY_TORCH_COMPILE_DYNAMIC=1 to avoid recompiles on shape changes
_DYNAMIC_DEFAULT = "true" if os.environ.get("COMFY_TORCH_COMPILE_DYNAMIC", "") in ("1", "true", "yes") else "false"


def _ensure_ops_patched_for_compile():
    """Mark ComfyUI weight casting functions as dynamo-disabled.

    This forces Dynamo to create an eager graph-break at memory boundaries so
    host-to-device transfers (LowVRAM / CPU offloading) occur in eager Python
    before compiled GPU kernels are executed. Idempotent; set
    NACHOLMO_TORCH_COMPILE_EAGER_BREAK=0 to opt out.
    """
    global _ops_patched_for_compile
    if _ops_patched_for_compile:
        return
    if os.environ.get("NACHOLMO_TORCH_COMPILE_EAGER_BREAK", "1") in ("0", "false", "no", "off"):
        log.info("[TorchCompile] Eager graph-break patch disabled via env")
        _ops_patched_for_compile = True
        return
    _ops_patched_for_compile = True

    cast_fn_names = (
        "cast_bias_weight",
        "uncast_bias_weight",
        "cast_modules_with_vbar",
        "resolve_cast_module_with_vbar",
    )
    for name in cast_fn_names:
        fn = getattr(comfy.ops, name, None)
        if fn is not None and not getattr(fn, "_nacholmo_dynamo_disabled", False):
            wrapped = torch._dynamo.disable(fn)
            wrapped._nacholmo_dynamo_disabled = True
            setattr(comfy.ops, name, wrapped)

    try:
        import comfy_aimdo.torch as at
        if hasattr(at, "get_tensor_from_raw_ptr") and not getattr(at.get_tensor_from_raw_ptr, "_nacholmo_dynamo_disabled", False):
            wrapped = torch._dynamo.disable(at.get_tensor_from_raw_ptr)
            wrapped._nacholmo_dynamo_disabled = True
            at.get_tensor_from_raw_ptr = wrapped
    except Exception:
        pass

    log.info("[TorchCompile] Patched comfy.ops memory casting for eager graph breaks (LowVRAM-safe)")


def _guard_filter_fn(guard_entries):
    """Prevent recompilation when transformer_options dict values change."""
    return [("transformer_options" not in entry.name) for entry in guard_entries]


def _apply_torch_compile_wrapper(executor: WrapperExecutor, *args, **kwargs):
    compiled = _COMPILED_MODELS_CACHE.get(executor.class_obj)
    if not compiled:
        return executor(*args, **kwargs)
    orig = {}
    try:
        for key, value in compiled.items():
            orig[key] = comfy.utils.get_attr(executor.class_obj, key)
            comfy.utils.set_attr(executor.class_obj, key, value)
        return executor(*args, **kwargs)
    finally:
        for key, value in orig.items():
            comfy.utils.set_attr(executor.class_obj, key, value)


class TorchCompileBlockwise:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "backend": (["inductor", "eager", "aot_eager"], {"default": "inductor"}),
                "mode": (["default", "max-autotune", "max-autotune-no-cudagraphs", "reduce-overhead"], {"default": "default"}),
                "dynamic": (["auto", "true", "false"], {"default": _DYNAMIC_DEFAULT, "tooltip": "Dynamic shape tracing. Defaults to true when COMFY_TORCH_COMPILE_DYNAMIC=1 is set at launch."}),
                "compile_blocks_only": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Compile transformer blocks individually instead of the whole model (recommended for LowVRAM and faster compilation)",
                }),
                "fullgraph": ("BOOLEAN", {"default": False, "tooltip": "Require full graph tracing without graph breaks (not recommended for LowVRAM)"}),
                "cache_size_limit": ("INT", {"default": 64, "min": 1, "max": 1024, "tooltip": "Dynamo frame cache size limit"}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_compile"
    CATEGORY = "model_patches/torch_compile"

    def apply_compile(self, model, backend, mode, dynamic, compile_blocks_only, fullgraph, cache_size_limit):
        m = model.clone(disable_dynamic=True)
        diffusion_model = m.get_model_object("diffusion_model")
        try:
            cache_size_limit = int(cache_size_limit)
        except (TypeError, ValueError):
            cache_size_limit = 64
        cache_size_limit = max(1, min(1024, cache_size_limit))
        torch._dynamo.config.cache_size_limit = cache_size_limit

        # QuantizedTensor subclasses can't be pickled into AOT autograd cache keys,
        # so every entry fails with a huge traceback while contributing nothing;
        # restart warmup comes from the fx_graph/triton caches regardless.
        # Set once; this is a process-global Dynamo flag.
        try:
            if torch._functorch.config.enable_autograd_cache:
                torch._functorch.config.enable_autograd_cache = False
        except Exception:
            pass

        # Always ensure weight casting runs eagerly in LowVRAM / Offload mode
        if not fullgraph:
            _ensure_ops_patched_for_compile()

        dynamic_map = {"true": True, "false": False, "auto": None}
        dynamic_val = dynamic_map.get(dynamic, None)

        compile_keys = []
        if compile_blocks_only:
            block_attrs = (
                "double_blocks",
                "single_blocks",
                "layers",
                "transformer_blocks",
                "blocks",
                "visual_transformer_blocks",
                "text_transformer_blocks",
                "patch_blocks",
                "pixel_blocks",
            )
            for attr in block_attrs:
                if hasattr(diffusion_model, attr):
                    blocks = getattr(diffusion_model, attr)
                    try:
                        if isinstance(blocks, dict):
                            indices = sorted(blocks.keys())
                            for i in indices:
                                compile_keys.append(f"diffusion_model.{attr}.{i}")
                        else:
                            for i in range(len(blocks)):
                                compile_keys.append(f"diffusion_model.{attr}.{i}")
                    except TypeError:
                        # Not a sized container (custom arch): fall through to whole-model.
                        continue

            if not compile_keys:
                log.warning("[TorchCompile] No transformer blocks detected, compiling entire diffusion_model")
                compile_keys = ["diffusion_model"]
            else:
                log.info(f"[TorchCompile] Compiling {len(compile_keys)} transformer blocks")
        else:
            compile_keys = ["diffusion_model"]

        # guard_filter_fn must ride inside options, and torch.compile rejects mode+options,
        # so expand the mode into its concrete inductor config patches instead
        compile_options = {"guard_filter_fn": _guard_filter_fn}
        if backend == "inductor" and mode and mode != "default":
            try:
                compile_options.update(list_mode_options(mode))
            except Exception as e:
                log.warning(f"[TorchCompile] Unknown mode '{mode}', using default inductor options: {e}")
        compile_kwargs = {
            "backend": backend,
            "fullgraph": fullgraph,
            "dynamic": dynamic_val,
            "options": compile_options,
        }

        compiled_modules = {}
        for key in compile_keys:
            target_module = m.get_model_object(key)
            compiled_modules[key] = torch.compile(target_module, **compile_kwargs)

        try:
            existed = m.model in _COMPILED_MODELS_CACHE
        except TypeError:
            existed = False
        if existed:
            log.warning("[TorchCompile] This model already has compiled modules from another TorchCompile node, replacing them")
        try:
            _COMPILED_MODELS_CACHE[m.model] = compiled_modules
        except TypeError:
            # m.model not weakrefable (rare custom patcher): skip cache, wrapper still works via closure.
            log.warning("[TorchCompile] Model object not weakrefable; skipping compiled-module cache")

        m.remove_wrappers_with_key(WrappersMP.APPLY_MODEL, _COMPILE_KEY)
        m.add_wrapper_with_key(WrappersMP.APPLY_MODEL, _COMPILE_KEY, _apply_torch_compile_wrapper)
        m.model_options["torch_compile_kwargs"] = compile_kwargs

        return (m,)


NODE_CLASS_MAPPINGS = {
    "TorchCompileBlockwise": TorchCompileBlockwise,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TorchCompileBlockwise": "TorchCompile Blockwise (LowVRAM-Safe)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
