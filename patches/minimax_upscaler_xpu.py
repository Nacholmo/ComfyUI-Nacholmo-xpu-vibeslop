"""Dynamic Intel XPU patch for Comfyui_Minimax_h3_latent_Upscaler.

Injects Intel XPU support, device resolution, schema options, and VRAM management
into Minimax H3 2D and 3D Latent Upscalers at runtime without modifying the original
node repository.
"""

import gc
import logging
import sys
import torch
from importlib.machinery import PathFinder

log = logging.getLogger("ComfyUI-Nacholmo-xpu-vibeslop.MinimaxUpscalerXPU")

_INSTALLED = False


def _is_nested_samples(s):
    """Check if s is a comfy.nested_tensor.NestedTensor (AV latent)."""
    return getattr(s, "is_nested", False) and hasattr(s, "tensors") and isinstance(getattr(s, "tensors", None), (list, tuple))


def _tensor_ndim(t):
    """Get ndim for both torch.Tensor and NestedTensor (which has ndim, not dim)."""
    if hasattr(t, "dim"):
        try:
            return t.dim()
        except Exception:
            pass
    return getattr(t, "ndim", len(getattr(t, "shape", [])))


def _ensure_nested_dim_alias():
    """Add .dim() to NestedTensor so src.dim() doesn't crash even outside our wrapper."""
    try:
        import comfy.nested_tensor as nt

        if not hasattr(nt.NestedTensor, "dim"):
            # define as method returning ndim property
            def _dim(self):
                return self.ndim

            nt.NestedTensor.dim = _dim
            print("[MinimaxUpscaler-XPU] Patched NestedTensor.dim alias -> ndim")

        # also ensure .clone exists? Not needed, but provide fallback via _copy
        if not hasattr(nt.NestedTensor, "clone"):
            def _clone(self):
                return self._copy()

            nt.NestedTensor.clone = _clone
    except Exception as e:
        log.debug(f"Could not patch NestedTensor dim alias: {e}")


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

    # Ensure NestedTensor.dim exists globally
    _ensure_nested_dim_alias()

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

        # Patch execute for XPU VRAM cleanup + NestedTensor (H3 AV) support
        if hasattr(cls_3d, "execute"):
            orig_execute = cls_3d.execute

            @classmethod
            def patched_execute(cls, *args, **kwargs):
                # Detect NestedTensor AV latent (is_nested)
                latent = kwargs.get("latent")
                if latent is None and len(args) >= 1:
                    latent = args[0]
                samples = None
                is_nested = False
                try:
                    if isinstance(latent, dict) and "samples" in latent:
                        samples = latent["samples"]
                        is_nested = _is_nested_samples(samples)
                except Exception:
                    is_nested = False

                if is_nested:
                    # Handle AV NestedTensor manually - only upscale video stream, preserve audio
                    param_names = ["latent", "model_name", "mode", "align", "enable_temporal_chunking", "force_unload", "device", "precision"]
                    vals = {}
                    for idx, name in enumerate(param_names):
                        if name in kwargs:
                            vals[name] = kwargs[name]
                        elif idx < len(args):
                            vals[name] = args[idx]
                        else:
                            vals[name] = None

                    latent_arg = vals["latent"]
                    model_name = vals["model_name"]
                    mode = vals["mode"]
                    align = vals["align"]
                    enable_temporal_chunking = vals["enable_temporal_chunking"]
                    force_unload = vals["force_unload"] if vals["force_unload"] is not None else True
                    device = vals["device"]
                    precision = vals["precision"]

                    if model_name.startswith("("):
                        raise ValueError("Please place model files into the latent_upscale_models directory")

                    src_nested = latent_arg["samples"]
                    video = src_nested.tensors[0]
                    audio = src_nested.tensors[1]
                    orig_dtype = video.dtype

                    # was_4d check on video stream only (image latent vs video latent)
                    try:
                        was_4d = (video.dim() == 4)
                    except AttributeError:
                        was_4d = (_tensor_ndim(video) == 4)

                    # Resolve device using patched _resolve_device
                    try:
                        dev = mod._resolve_device(device)
                    except Exception:
                        # fallback manual
                        if device == "xpu":
                            dev = torch.device("xpu") if hasattr(torch, "xpu") and torch.xpu.is_available() else torch.device("cpu")
                        elif device == "cuda":
                            dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("xpu") if hasattr(torch, "xpu") and torch.xpu.is_available() else torch.device("cpu")
                        else:
                            dev = torch.device("cpu")

                    compute_dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[precision]

                    # Move video to device (clone to avoid mutating original)
                    s = video.to(device=dev, dtype=compute_dtype)
                    # ensure copy
                    try:
                        s = s.clone()
                    except Exception:
                        pass

                    if was_4d:
                        s = s.unsqueeze(2)

                    b, c, t, h_in, w_in = s.shape
                    downsample = getattr(mod, "VAE_DOWNSAMPLE", 16)
                    UpscaleMode = getattr(mod, "UpscaleMode", None)

                    # Determine selected mode / effective scale
                    selected_mode = None
                    if isinstance(mode, dict):
                        selected_mode = mode.get("mode")
                    else:
                        selected_mode = mode

                    # Normalize to string compare if enum not available
                    def _is_scale_by(m):
                        if UpscaleMode is not None:
                            try:
                                return m == UpscaleMode.SCALE_BY
                            except Exception:
                                pass
                        return str(m) == "scale by multiplier" or str(m) == str(getattr(UpscaleMode, "SCALE_BY", ""))

                    def _is_target_dims(m):
                        if UpscaleMode is not None:
                            try:
                                return m == UpscaleMode.TARGET_DIMENSIONS
                            except Exception:
                                pass
                        return str(m) == "target dimensions"

                    def _is_megapixels(m):
                        if UpscaleMode is not None:
                            try:
                                return m == UpscaleMode.MEGAPIXELS
                            except Exception:
                                pass
                        return str(m) == "megapixels"

                    if _is_scale_by(selected_mode):
                        scale_val = mode["scale"]
                        w_pixel_target = w_in * downsample * scale_val
                        h_pixel_target = h_in * downsample * scale_val
                        effective_scale = scale_val
                    elif _is_target_dims(selected_mode):
                        w_pixel_target = float(mode["width"])
                        h_pixel_target = float(mode["height"])
                        effective_scale = (w_pixel_target / (w_in * downsample) + h_pixel_target / (h_in * downsample)) / 2.0
                    elif _is_megapixels(selected_mode):
                        mp = mode["megapixels"]
                        target_pixels = mp * 1024 * 1024
                        aspect_ratio = w_in / h_in
                        h_pixel_target = (target_pixels / aspect_ratio) ** 0.5
                        w_pixel_target = h_pixel_target * aspect_ratio
                        effective_scale = (w_pixel_target / (w_in * downsample) + h_pixel_target / (h_in * downsample)) / 2.0
                    else:
                        raise ValueError(f"Unsupported mode: {selected_mode}")

                    alignment = max(1, align)
                    w_pixel_aligned = round(w_pixel_target / alignment) * alignment
                    h_pixel_aligned = round(h_pixel_target / alignment) * alignment

                    w_pixel_final = round(w_pixel_aligned / downsample) * downsample
                    h_pixel_final = round(h_pixel_aligned / downsample) * downsample

                    w_out = max(1, int(w_pixel_final // downsample))
                    h_out = max(1, int(h_pixel_final // downsample))

                    if effective_scale < 1.0 and (w_out < w_in or h_out < h_in):
                        raise ValueError("This model only supports upscaling (effective scale >= 1.0).")

                    if w_out == w_in and h_out == h_in:
                        # No-op, return original latent untouched
                        return mod.io.NodeOutput(latent_arg) if hasattr(mod, "io") else orig_execute(*args, **kwargs)

                    print(f"[MinimaxH3-3D] Latent {w_in}x{h_in} -> {w_out}x{h_out} | Pixels {w_out * downsample}x{h_out * downsample} | scale={effective_scale:.3f} [H3 AV NestedTensor]")

                    model = mod.load_model(model_name, dev, precision)
                    norm_mean, norm_std = mod._make_norm_tensors(dev, compute_dtype)

                    with torch.inference_mode():
                        s_norm = (s - norm_mean) / norm_std
                        # free s reference
                        del s
                        out_v = model(s_norm, scale=effective_scale, target_size=(t, h_out, w_out), enable_chunking=enable_temporal_chunking)
                        del s_norm
                        out_v = out_v * norm_std + norm_mean

                    if was_4d:
                        out_v = out_v.squeeze(2)

                    out_v = out_v.to(device="cpu", dtype=orig_dtype, non_blocking=True)
                    # Keep audio on CPU with original dtype
                    try:
                        audio_cpu = audio.to(device="cpu", dtype=audio.dtype)
                    except Exception:
                        audio_cpu = audio

                    # VRAM management for both XPU and CUDA
                    try:
                        if dev.type == "xpu" and hasattr(torch, "xpu") and torch.xpu.is_available():
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
                                try:
                                    torch.xpu.empty_cache()
                                except Exception:
                                    pass
                            gc.collect()
                        elif dev.type == "cuda":
                            if force_unload:
                                try:
                                    model.to("cpu", non_blocking=True)
                                except Exception:
                                    pass
                                print("[MinimaxH3-3D] ✅ Model offloaded to CPU. VRAM released.")
                            try:
                                import comfy.model_management as mm
                                if getattr(mod, "HAS_COMFY_MM", False):
                                    mm.soft_empty_cache()
                                else:
                                    torch.cuda.empty_cache()
                            except Exception:
                                try:
                                    torch.cuda.empty_cache()
                                except Exception:
                                    pass
                            gc.collect()
                        else:
                            if force_unload:
                                try:
                                    model.to("cpu", non_blocking=True)
                                except Exception:
                                    pass
                    except Exception:
                        pass

                    # Re-wrap as NestedTensor
                    try:
                        import comfy.nested_tensor as nt
                        out_nested = nt.NestedTensor((out_v, audio_cpu))
                    except Exception:
                        # fallback to io path
                        import comfy.nested_tensor as nt2
                        out_nested = nt2.NestedTensor((out_v, audio_cpu))

                    # Return via comfy_api io.NodeOutput if available
                    try:
                        from comfy_api.latest import io as _io
                        return _io.NodeOutput({"samples": out_nested})
                    except Exception:
                        # fallback: try mod.io
                        if hasattr(mod, "io"):
                            return mod.io.NodeOutput({"samples": out_nested})
                        # last fallback: plain dict
                        return {"samples": out_nested}

                # Non-nested path: original execute + XPU VRAM cleanup
                try:
                    out = orig_execute(*args, **kwargs)
                except AttributeError as e:
                    # Provide clearer error for NestedTensor-like objects that slipped through is_nested check
                    # e.g. 'NestedTensor' object has no attribute 'dim'
                    if "has no attribute 'dim'" in str(e) and samples is not None:
                        raise AttributeError(
                            f"{e}. This usually means a Minimax H3 AV latent (NestedTensor) was passed to the 3D upscaler "
                            f"but was not detected as nested. Ensure comfy.nested_tensor.NestedTensor is used and update the XPU vibeslop patch."
                        ) from e
                    raise
                # XPU/CUDA cleanup
                try:
                    # Determine device for cleanup - use kwargs device or infer
                    force_unload = kwargs.get("force_unload", True)
                    # also check positional force_unload (index 5)
                    if "force_unload" not in kwargs and len(args) >= 6:
                        force_unload = args[5]
                    if hasattr(torch, "xpu") and torch.xpu.is_available():
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
                            try:
                                torch.xpu.empty_cache()
                            except Exception:
                                pass
                        gc.collect()
                    else:
                        # For CUDA path the original already does cleanup, but ensure
                        pass
                except Exception:
                    pass
                return out

            cls_3d.execute = patched_execute

    mod._nacholmo_xpu_patched = True
    print("[MinimaxUpscaler-XPU] Patched Minimax H3 3D Latent Upscaler for Intel XPU")


def _patch_2d_module(mod):
    if getattr(mod, "_nacholmo_xpu_patched", False):
        return

    _ensure_nested_dim_alias()
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

        # Patch run method - with NestedTensor support and XPU device handling
        if hasattr(cls_2d, "run"):

            def patched_run(self, latent, model_name, scale, device, precision):
                if model_name.startswith("("):
                    raise ValueError("请将模型文件放入 latent_upscale_models 目录")

                if abs(scale - 1.0) < 1e-6:
                    return (latent,)

                if scale < 1.0:
                    raise ValueError("仅支持放大 (scale >= 1.0)")

                # Check for H3 AV NestedTensor
                samples = latent["samples"] if isinstance(latent, dict) and "samples" in latent else None
                is_nested = _is_nested_samples(samples)

                # Device resolution (with XPU fallback)
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

                if is_nested:
                    # AV path: only upscale video stream
                    video = samples.tensors[0]
                    audio = samples.tensors[1]
                    orig_dtype = video.dtype
                    # Determine was_4d from video shape
                    try:
                        was_4d = (video.dim() == 4)
                    except AttributeError:
                        was_4d = (_tensor_ndim(video) == 4)
                    # Also handle len(shape) check
                    s = video.clone() if hasattr(video, "clone") else video
                    if _tensor_ndim(s) == 4 or len(getattr(s, "shape", [])) == 4:
                        # Need to check shape length; for video 5D: [B,C,T,H,W], 4D would be [B,C,H,W]
                        if len(s.shape) == 4:
                            s = s.unsqueeze(2)
                            was_4d = True
                        # else keep

                    compute_dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[precision]
                    s = s.to(dev, compute_dtype)

                    norm_mean, norm_std = mod._make_norm_tensors(dev, compute_dtype)
                    s = (s - norm_mean) / norm_std

                    with torch.no_grad():
                        T, H, W = s.shape[2], s.shape[3], s.shape[4]
                        target_hw = (int(round(H * scale)), int(round(W * scale)))
                        out_v = model(s, scale=scale, target_hw=target_hw)

                    out_v = out_v * norm_std + norm_mean

                    if was_4d:
                        out_v = out_v.squeeze(2)

                    out_v = out_v.cpu().to(orig_dtype)
                    try:
                        audio_cpu = audio.to(device="cpu", dtype=audio.dtype)
                    except Exception:
                        audio_cpu = audio

                    if dev.type == "xpu" and hasattr(torch, "xpu"):
                        try:
                            torch.xpu.empty_cache()
                        except Exception:
                            pass
                    elif dev.type == "cuda" and hasattr(torch, "cuda"):
                        try:
                            torch.cuda.empty_cache()
                        except Exception:
                            pass

                    import comfy.nested_tensor as nt
                    out_nested = nt.NestedTensor((out_v, audio_cpu))
                    return ({"samples": out_nested},)
                else:
                    # Original plain tensor path
                    s = latent["samples"].clone() if hasattr(latent["samples"], "clone") else latent["samples"]
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
                        try:
                            torch.xpu.empty_cache()
                        except Exception:
                            pass
                    elif dev.type == "cuda" and hasattr(torch, "cuda"):
                        try:
                            torch.cuda.empty_cache()
                        except Exception:
                            pass

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

    _ensure_nested_dim_alias()

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
