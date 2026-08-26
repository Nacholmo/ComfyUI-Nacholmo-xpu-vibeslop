# Arc Super Resolution: OpenVINO-accelerated image and video super resolution
# for Intel Arc GPUs. The Intel counterpart to Comfy-Org/Nvidia_RTX_Nodes_ComfyUI.

import os
import re
import tempfile
import time
import weakref
from enum import Enum

import numpy as np
import psutil
import torch

import folder_paths
import comfy.utils
import comfy.model_management

try:
    import openvino as ov
    HAS_OPENVINO = True
except ImportError:
    ov = None
    HAS_OPENVINO = False


class UpscaleType(str, Enum):
    MODEL_NATIVE = "model native (original scale)"
    SCALE_BY = "scale by multiplier"
    TARGET_DIMENSIONS = "target dimensions"


class PrecisionMode(str, Enum):
    AUTO_FP16 = "auto / fp16 (fastest)"
    FP32 = "fp32 (full precision)"


class PerformanceMode(str, Enum):
    THROUGHPUT = "throughput (video / batch)"
    LATENCY = "latency (single frame)"


_UPSCALER_DIR = os.path.join(folder_paths.models_dir, "openvino_upscalers")
folder_paths.add_model_folder_path("openvino_upscalers", _UPSCALER_DIR)
try:
    os.makedirs(_UPSCALER_DIR, exist_ok=True)
    _CACHE_DIR = os.path.join(_UPSCALER_DIR, ".cache")
    os.makedirs(_CACHE_DIR, exist_ok=True)
except Exception:
    _CACHE_DIR = None

# Converted models are kept next to their .pth sources in the regular
# upscale_models folders; .onnx/.xml are invisible to the core loader.
for path in folder_paths.get_folder_paths("upscale_models"):
    folder_paths.add_model_folder_path("openvino_upscalers", path)

_MODEL_EXTENSIONS = (".xml", ".onnx")

_SCALE_REGEXES = [
    re.compile(r"(?:^|[\W_])([1-8])x(?:[\W_]|$)", re.IGNORECASE),
    re.compile(r"(?:^|[\W_])x([1-8])(?:[\W_]|$)", re.IGNORECASE),
    re.compile(r"[-_]x([1-8])(?:plus|v\d+|anime)?(?:[-_.]|$)", re.IGNORECASE),
    re.compile(r"([1-8])x[-_]", re.IGNORECASE),
]


def _cleanup_spill(path):
    try:
        os.unlink(path)
    except OSError:
        pass


_core = None
_loaded = None  # ((path, mtime_ns, precision, performance_mode), compiled, layout, scale)


def _get_core():
    global _core
    if not HAS_OPENVINO:
        raise RuntimeError("OpenVINO is not installed. Please run 'pip install openvino'.")
    if _core is None:
        _core = ov.Core()
        if _CACHE_DIR:
            _core.set_property({"CACHE_DIR": _CACHE_DIR})
    return _core


def detect_scale_from_filename(filename: str) -> int | None:
    stem = os.path.splitext(os.path.basename(filename))[0]
    for pattern in _SCALE_REGEXES:
        match = pattern.search(stem)
        if match:
            return int(match.group(1))
    return None


def _detect_layout(model, name):
    shape = model.input(0).partial_shape
    if len(shape) != 4:
        raise ValueError(f"{name}: expected a 4D NCHW/NHWC input, got rank {len(shape)}")
    c2 = shape[1].get_length() if not shape[1].is_dynamic else None
    c4 = shape[3].get_length() if not shape[3].is_dynamic else None
    if c2 == 3:
        return "NCHW"
    if c4 == 3:
        return "NHWC"
    if c2 is None and c4 is None:
        return "NCHW"
    raise ValueError(f"{name}: could not find 3 RGB channels in input shape {shape}")


def _infer_patch(request, patch, layout):
    x = np.ascontiguousarray(patch, dtype=np.float32)
    if layout == "NCHW":
        x = np.ascontiguousarray(x.transpose(2, 0, 1))[None]
    else:
        x = x[None]
    request.infer(ov.Tensor(x))
    y = request.get_output_tensor(0).data[0]
    if layout == "NCHW":
        y = y.transpose(1, 2, 0)
    return np.ascontiguousarray(y, dtype=np.float32)


def _load_model(path, precision: str = PrecisionMode.AUTO_FP16, performance_mode: str = PerformanceMode.THROUGHPUT):
    global _loaded
    key = (path, os.stat(path).st_mtime_ns, precision, performance_mode)
    if _loaded is not None and _loaded[0] == key:
        return _loaded[1:]

    core = _get_core()
    if "GPU" not in core.get_available_devices():
        raise RuntimeError(
            "OpenVINO reports no GPU device. Install the Intel compute runtime "
            "(OpenCL) for your Arc GPU, e.g. 'intel-opencl-icd' or 'intel-compute-runtime'."
        )

    name = os.path.basename(path)
    model = core.read_model(path)
    if len(model.inputs) != 1 or len(model.outputs) != 1:
        raise ValueError(f"{name}: node supports models with exactly one input and one output")
    layout = _detect_layout(model, name)

    config = {}
    if precision == PrecisionMode.AUTO_FP16:
        config["INFERENCE_PRECISION_HINT"] = "f16"
    elif precision == PrecisionMode.FP32:
        config["INFERENCE_PRECISION_HINT"] = "f32"

    if performance_mode == PerformanceMode.THROUGHPUT:
        config["PERFORMANCE_HINT"] = "THROUGHPUT"
    else:
        config["PERFORMANCE_HINT"] = "LATENCY"

    compiled = core.compile_model(model, "GPU", config)
    req = compiled.create_infer_request()
    probe = _infer_patch(req, np.zeros((128, 128, 3), np.float32), layout)
    scale = probe.shape[0] // 128
    if scale < 1 or probe.shape[:2] != (128 * scale, 128 * scale):
        raise ValueError(f"{name}: unexpected output shape {probe.shape} for a 128x128 RGB input")

    _loaded = (key, compiled, layout, scale)
    return compiled, layout, scale


def _edge_ramp(size, edge):
    r = np.ones(size, np.float32)
    e = min(edge, size // 2)
    if e > 0:
        r[:e] = (np.arange(e, dtype=np.float32) + 0.5) / e
        r[-e:] = np.minimum(r[-e:], r[:e][::-1])
    return r


def _super_resolve(compiled, layout, scale, image, tile, overlap, request=None):
    if request is None:
        request = compiled.create_infer_request()

    h, w, c = image.shape
    th, tw = h * scale, w * scale

    # Fast path: full frame without tiling
    if tile <= 0 or (tile >= h and tile >= w):
        return _infer_patch(request, image, layout)

    overlap = min(overlap, tile // 4)
    step = max(tile - 2 * overlap, 16)
    edge = overlap * scale

    out = np.zeros((th, tw, c), dtype=np.float32)
    weight = np.zeros((th, tw, 1), dtype=np.float32)

    for y in range(0, h, step):
        for x in range(0, w, step):
            y0 = max(0, y - overlap)
            x0 = max(0, x - overlap)
            y1 = min(h, y + step + overlap)
            x1 = min(w, x + step + overlap)

            patch = image[y0:y1, x0:x1]
            up = _infer_patch(request, patch, layout)

            ph, pw = up.shape[:2]
            ry = _edge_ramp(ph, edge) if y0 > 0 or y1 < h else np.ones(ph, np.float32)
            rx = _edge_ramp(pw, edge) if x0 > 0 or x1 < w else np.ones(pw, np.float32)
            wpatch = (ry[:, None] * rx[None, :])[:, :, None]

            oy0, ox0 = y0 * scale, x0 * scale
            oy1, ox1 = oy0 + ph, ox0 + pw
            out[oy0:oy1, ox0:ox1] += up * wpatch
            weight[oy0:oy1, ox0:ox1] += wpatch

    np.maximum(weight, 1e-6, out=weight)
    out /= weight
    return out


def _get_model_names():
    names = []
    for f in folder_paths.get_filename_list("openvino_upscalers"):
        if f.lower().endswith(_MODEL_EXTENSIONS):
            names.append(f)
    return names if names else ["None"]


class ArcSuperResolution:
    @classmethod
    def INPUT_TYPES(cls):
        names = _get_model_names()
        return {
            "required": {
                "images": ("IMAGE",),
                "model_name": (names,),
                "resize_type": ([UpscaleType.MODEL_NATIVE.value, UpscaleType.SCALE_BY.value, UpscaleType.TARGET_DIMENSIONS.value],),
                "scale": ("FLOAT", {"default": 4.0, "min": 1.0, "max": 8.0, "step": 0.01}),
                "target_width": ("INT", {"default": 1920, "min": 16, "max": 8192, "step": 1}),
                "target_height": ("INT", {"default": 1080, "min": 16, "max": 8192, "step": 1}),
                "tile_size": ("INT", {"default": 512, "min": 0, "max": 4096, "step": 16}),
                "tile_overlap": ("INT", {"default": 64, "min": 8, "max": 256, "step": 8}),
                "precision": ([PrecisionMode.AUTO_FP16.value, PrecisionMode.FP32.value], {"default": PrecisionMode.AUTO_FP16.value}),
                "performance_mode": ([PerformanceMode.THROUGHPUT.value, PerformanceMode.LATENCY.value], {"default": PerformanceMode.THROUGHPUT.value}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("upscaled_images",)
    FUNCTION = "upscale"
    CATEGORY = "image/upscaling"

    def upscale(
        self,
        images: torch.Tensor,
        model_name: str,
        resize_type: str = UpscaleType.MODEL_NATIVE.value,
        scale: float = 4.0,
        target_width: int = 1920,
        target_height: int = 1080,
        tile_size: int = 512,
        tile_overlap: int = 64,
        precision: str = PrecisionMode.AUTO_FP16.value,
        performance_mode: str = PerformanceMode.THROUGHPUT.value,
    ):
        b, h, w, c = images.shape
        path = folder_paths.get_full_path("openvino_upscalers", model_name)
        if path is None:
            raise FileNotFoundError(f"'{model_name}' not found in models/openvino_upscalers")

        compiled, layout, model_scale = _load_model(path, precision=precision, performance_mode=performance_mode)

        if resize_type == UpscaleType.MODEL_NATIVE.value:
            out_w, out_h = int(w * model_scale), int(h * model_scale)
        elif resize_type == UpscaleType.SCALE_BY.value:
            out_w, out_h = int(w * scale), int(h * scale)
        elif resize_type == UpscaleType.TARGET_DIMENSIONS.value:
            out_w, out_h = target_width, target_height
        else:
            raise ValueError(f"Unsupported resize type: {resize_type}")

        out_bytes = b * out_h * out_w * c * 4
        available = psutil.virtual_memory().available
        spill = None
        if out_bytes > available // 2:
            fd, spill_path = tempfile.mkstemp(suffix=".f32", dir=folder_paths.get_temp_directory())
            os.close(fd)
            spill = np.memmap(spill_path, dtype=np.float32, mode="w+", shape=(b, out_h, out_w, c))
        else:
            out_tensor = torch.empty((b, out_h, out_w, c), dtype=torch.float32)

        pbar = comfy.utils.ProgressBar(b)
        req = compiled.create_infer_request()

        t_start = time.perf_counter()
        last_log_time = t_start

        for i in range(b):
            comfy.model_management.throw_exception_if_processing_interrupted()

            frame_np = images[i].cpu().numpy()
            up = _super_resolve(compiled, layout, model_scale, frame_np, tile_size, tile_overlap, request=req)

            if (up.shape[1], up.shape[0]) != (out_w, out_h):
                t = torch.from_numpy(up).permute(2, 0, 1)[None]
                t = torch.nn.functional.interpolate(t, size=(out_h, out_w), mode="bicubic", align_corners=False, antialias=True)
                up = t[0].permute(1, 2, 0).numpy()

            np.clip(up, 0.0, 1.0, out=up)
            if spill is None:
                out_tensor[i] = torch.from_numpy(np.ascontiguousarray(up))
            else:
                spill[i] = up
            pbar.update(1)

            now = time.perf_counter()
            if (now - last_log_time >= 1.0) or (i == b - 1):
                elapsed = now - t_start
                fps = (i + 1) / elapsed if elapsed > 0 else 0.0
                pct = ((i + 1) / b) * 100.0
                remaining = (b - (i + 1)) / fps if fps > 0 else 0.0
                print(f"[Arc Super Resolution] Frame {i + 1}/{b} ({pct:.1f}%) | {fps:.2f} fps | ETA: {remaining:.1f}s")
                last_log_time = now

        if spill is not None:
            spill.flush()
            del spill
            storage = torch.UntypedStorage.from_file(spill_path, False, out_bytes)
            out_tensor = torch.empty((b, out_h, out_w, c), dtype=torch.float32)
            out_tensor.set_(storage, 0, (b, out_h, out_w, c))
            weakref.finalize(out_tensor, _cleanup_spill, spill_path)

        return (out_tensor,)


class ArcResampleFPS:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "source_fps": ("FLOAT", {"default": 72.0, "min": 0.1, "max": 1000.0, "step": 0.01}),
                "target_fps": ("FLOAT", {"default": 60.0, "min": 0.1, "max": 1000.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE", "FLOAT")
    RETURN_NAMES = ("images", "fps")
    FUNCTION = "resample"
    CATEGORY = "video"

    def resample(self, images: torch.Tensor, source_fps: float, target_fps: float):
        num_frames = images.shape[0]
        if num_frames <= 1 or abs(source_fps - target_fps) < 1e-4:
            return (images, float(target_fps))

        duration = (num_frames - 1) / source_fps
        target_count = max(1, round(duration * target_fps) + 1)

        indices = [min(num_frames - 1, max(0, round(i * (source_fps / target_fps)))) for i in range(target_count)]
        resampled_images = images[indices]
        return (resampled_images, float(target_fps))


NODE_CLASS_MAPPINGS = {
    "ArcSuperResolution": ArcSuperResolution,
    "ArcResampleFPS": ArcResampleFPS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ArcSuperResolution": "Arc Super Resolution",
    "ArcResampleFPS": "Arc Resample FPS",
}

__all__ = ["ArcSuperResolution", "ArcResampleFPS", "NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
