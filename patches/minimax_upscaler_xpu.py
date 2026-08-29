"""Dynamic Intel XPU patch for Comfyui_Minimax_h3_latent_Upscaler.

Injects Intel XPU support, device resolution, schema options, and VRAM management
into Minimax H3 2D and 3D Latent Upscalers at runtime without modifying the original
node repository.
"""

import logging
import sys
import torch
from importlib.machinery import PathFinder

log = logging.getLogger("ComfyUI-Nacholmo-xpu-vibeslop.MinimaxUpscalerXPU")

_INSTALLED = False


def _wrap_load_model(mod):
    if hasattr(mod, "load_model") and not getattr(mod.load_model, "_nacholmo_wrapped", False):
        orig_load_model = mod.load_model

        def safe_load_model(name, device, precision):
            try:
                return orig_load_model(name, device, precision)
            except RuntimeError as e:
                err_str = str(e)
                if "Missing key(s) in state_dict" in err_str and (
                    "initial_conv" in err_str or "post_upsample_res_blocks" in err_str or "final_conv" in err_str
                ):
                    raise ValueError(
                        f"\n\n❌ Incompatible Model Selected: '{name}'\n"
                        f"This checkpoint is an LTX-Video latent upscaler, NOT a MiniMax-H3 latent upscaler.\n"
                        f"👉 Please select a MiniMax-H3 model in the node (e.g. 'minimax_h3_latent_upscaler_3d_bf16.safetensors').\n"
                    ) from e
                raise

        safe_load_model._nacholmo_wrapped = True
        mod.load_model = safe_load_model


def _patch_3d_module(mod):
    if getattr(mod, "_nacholmo_xpu_patched", False):
        return

    # 1. Patch _resolve_device
    if hasattr(mod, "_resolve_device"):
        orig_resolve = mod._resolve_device

        def safe_resolve_device(backend):
            if backend == "xpu":
                if hasattr(torch, "xpu") and torch.xpu.is_available():
                    return torch.device("xpu")
                raise RuntimeError("XPU was selected, but PyTorch cannot access an Intel XPU device.")
            if backend == "cuda":
                if torch.cuda.is_available():
                    return torch.device("cuda")
                if hasattr(torch, "xpu") and torch.xpu.is_available():
                    return torch.device("xpu")
                return torch.device("cpu")
            return orig_resolve(backend)

        mod._resolve_device = safe_resolve_device

    # 2. Patch _backend_label
    if hasattr(mod, "_backend_label"):
        orig_label = mod._backend_label

        def safe_backend_label(device):
            if device.type == "xpu":
                dev_name = (
                    torch.xpu.get_device_name(device)
                    if hasattr(torch, "xpu") and torch.xpu.is_available()
                    else "Intel XPU"
                )
                return f"XPU ({dev_name})"
            return orig_label(device)

        mod._backend_label = safe_backend_label

    # 3. Friendly model loading validation
    _wrap_load_model(mod)

    # 4. Patch MinimaxH3LatentUpscaler3D
    cls_3d = getattr(mod, "MinimaxH3LatentUpscaler3D", None)
    if cls_3d:
        # Patch define_schema for new V3 API
        if hasattr(cls_3d, "define_schema"):
            orig_schema_fn = cls_3d.define_schema

            @classmethod
            def patched_define_schema(cls):
                schema = orig_schema_fn()
                for inp in schema.inputs:
                    inp_id = getattr(inp, "id", getattr(inp, "name", None))
                    if inp_id == "device" and hasattr(inp, "options"):
                        opts = list(inp.options)
                        if "xpu" not in opts:
                            if "cuda" in opts:
                                idx = opts.index("cuda") + 1
                                opts.insert(idx, "xpu")
                            else:
                                opts.append("xpu")
                            inp.options = opts
                return schema

            cls_3d.define_schema = patched_define_schema
        elif hasattr(cls_3d, "INPUT_TYPES"):
            # Only patch INPUT_TYPES for legacy non-V3 nodes
            orig_input_types = cls_3d.INPUT_TYPES

            @classmethod
            def patched_input_types(cls):
                res = orig_input_types()
                if "required" in res and "device" in res["required"]:
                    dev_entry = res["required"]["device"]
                    if isinstance(dev_entry[0], list):
                        opts = list(dev_entry[0])
                        if "xpu" not in opts:
                            if "cuda" in opts:
                                idx = opts.index("cuda") + 1
                                opts.insert(idx, "xpu")
                            else:
                                opts.append("xpu")
                            res["required"]["device"] = (opts, dev_entry[1])
                return res

            cls_3d.INPUT_TYPES = patched_input_types

        # Patch execute for XPU VRAM cleanup
        if hasattr(cls_3d, "execute"):
            orig_execute = cls_3d.execute

            @classmethod
            def patched_execute(cls, *args, **kwargs):
                out = orig_execute(*args, **kwargs)
                if hasattr(torch, "xpu") and torch.xpu.is_available():
                    force_unload = kwargs.get("force_unload", True)
                    if force_unload and hasattr(mod, "MODEL_CACHE"):
                        for m in mod.MODEL_CACHE.values():
                            try:
                                m.to("cpu", non_blocking=True)
                            except Exception:
                                pass
                    try:
                        import comfy.model_management as mm
                        mm.soft_empty_cache()
                    except Exception:
                        torch.xpu.empty_cache()
                    import gc
                    gc.collect()
                return out

            cls_3d.execute = patched_execute

    mod._nacholmo_xpu_patched = True
    print("[MinimaxUpscaler-XPU] Patched Minimax H3 3D Latent Upscaler for Intel XPU")


def _patch_2d_module(mod):
    if getattr(mod, "_nacholmo_xpu_patched", False):
        return

    # 1. Friendly model loading validation
    _wrap_load_model(mod)

    cls_2d = getattr(mod, "MinimaxH3LatentUpscalerNode2D", None)
    if cls_2d:
        # Patch INPUT_TYPES (Legacy API)
        if hasattr(cls_2d, "INPUT_TYPES"):
            orig_input_types = cls_2d.INPUT_TYPES

            @classmethod
            def patched_input_types(cls):
                res = orig_input_types()
                if "required" in res and "device" in res["required"]:
                    dev_entry = res["required"]["device"]
                    if isinstance(dev_entry[0], list):
                        opts = list(dev_entry[0])
                        if "xpu" not in opts:
                            if "cuda" in opts:
                                idx = opts.index("cuda") + 1
                                opts.insert(idx, "xpu")
                            else:
                                opts.append("xpu")
                            res["required"]["device"] = (opts, dev_entry[1])
                return res

            cls_2d.INPUT_TYPES = patched_input_types

        # Patch run method
        if hasattr(cls_2d, "run"):
            def patched_run(self, latent, model_name, scale, device, precision):
                if model_name.startswith('('):
                    raise ValueError("请将模型文件放入 latent_upscale_models 目录")

                if abs(scale - 1.0) < 1e-6:
                    return (latent,)

                if scale < 1.0:
                    raise ValueError("仅支持放大 (scale >= 1.0)")

                if device == "xpu":
                    if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
                        raise RuntimeError("XPU was selected, but PyTorch cannot access an Intel XPU device.")
                    dev = torch.device("xpu")
                elif device == "cuda":
                    if torch.cuda.is_available():
                        dev = torch.device("cuda")
                    elif hasattr(torch, "xpu") and torch.xpu.is_available():
                        dev = torch.device("xpu")
                    else:
                        dev = torch.device("cpu")
                else:
                    dev = torch.device("cpu")

                model = mod.load_model(model_name, dev, precision)

                s = latent["samples"].clone()
                orig_dtype = s.dtype
                if len(s.shape) == 4:
                    s = s.unsqueeze(2)

                compute_dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[precision]
                s = s.to(dev, compute_dtype)

                norm_mean, norm_std = mod._make_norm_tensors(dev, compute_dtype)
                s = (s - norm_mean) / norm_std

                with torch.no_grad():
                    T, H, W = s.shape[2], s.shape[3], s.shape[4]
                    target_hw = (int(round(H * scale)), int(round(W * scale)))
                    out = model(s, scale=scale, target_hw=target_hw)

                out = out * norm_std + norm_mean

                if len(latent["samples"].shape) == 4:
                    out = out.squeeze(2)

                out = out.cpu().to(orig_dtype)

                if dev.type == "xpu" and hasattr(torch, "xpu"):
                    torch.xpu.empty_cache()
                elif dev.type == "cuda" and hasattr(torch, "cuda"):
                    torch.cuda.empty_cache()

                return ({"samples": out},)

            cls_2d.run = patched_run

    mod._nacholmo_xpu_patched = True
    print("[MinimaxUpscaler-XPU] Patched Minimax H3 2D Latent Upscaler for Intel XPU")


class _MinimaxUpscalerMetaFinder:
    @classmethod
    def find_spec(cls, fullname, path=None, target=None):
        if "minimax_h3_latent_upscaler_3d" in fullname or "minimax_h3_latent_upscaler_2d" in fullname:
            spec = PathFinder.find_spec(fullname, path, target)
            if spec and spec.loader:
                orig_exec = spec.loader.exec_module

                def exec_module_patched(module):
                    orig_exec(module)
                    try:
                        if "minimax_h3_latent_upscaler_3d" in fullname:
                            _patch_3d_module(module)
                        elif "minimax_h3_latent_upscaler_2d" in fullname:
                            _patch_2d_module(module)
                    except Exception as e:
                        log.debug(f"Failed to patch {fullname}: {e}")

                spec.loader.exec_module = exec_module_patched
            return spec
        return None


def apply():
    global _INSTALLED
    if _INSTALLED:
        return

    # 1. Patch already-loaded modules in sys.modules
    for name, mod in list(sys.modules.items()):
        if mod is not None:
            if "minimax_h3_latent_upscaler_3d" in name:
                try:
                    _patch_3d_module(mod)
                except Exception as e:
                    log.debug(f"Could not patch loaded 3d module {name}: {e}")
            elif "minimax_h3_latent_upscaler_2d" in name:
                try:
                    _patch_2d_module(mod)
                except Exception as e:
                    log.debug(f"Could not patch loaded 2d module {name}: {e}")

    # 2. Register import meta finder for deferred loading
    already_in_meta = any(
        getattr(finder, "__name__", "") == "_MinimaxUpscalerMetaFinder" or finder is _MinimaxUpscalerMetaFinder
        for finder in sys.meta_path
    )
    if not already_in_meta:
        sys.meta_path.insert(0, _MinimaxUpscalerMetaFinder)

    _INSTALLED = True


__all__ = ["apply"]
