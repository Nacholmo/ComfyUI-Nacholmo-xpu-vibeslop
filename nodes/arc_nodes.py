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


import gc
import logging
import multiprocessing as mp
from multiprocessing import shared_memory
import subprocess
import sys
import traceback

log = logging.getLogger("ComfyUI-Nacholmo-xpu-vibeslop.ArcNodes")


def _cleanup_spill(path):
    try:
        os.unlink(path)
    except OSError:
        pass


def detect_scale_from_filename(filename: str) -> int | None:
    stem = os.path.splitext(os.path.basename(filename))[0]
    for pattern in _SCALE_REGEXES:
        match = pattern.search(stem)
        if match:
            return int(match.group(1))
    return None


# ─── Standalone OpenVINO Worker Process Client ─────────────────────────────────

_WORKER_SCRIPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools", "arc_openvino_worker.py")


class _ArcWorkerClient:
    def __init__(self):
        self.proc = None
        self.conn = None
        self.listener = None
        self.sock_path = None
        self.loaded_key = None
        self.layout = None
        self.model_scale = None

    def is_alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None and self.conn is not None

    def start(self):
        if self.is_alive():
            return
        self.stop()

        from multiprocessing.connection import Listener
        fd, self.sock_path = tempfile.mkstemp(prefix="arc_worker_", suffix=".sock")
        os.close(fd)
        os.unlink(self.sock_path)

        self.listener = Listener(self.sock_path, family="AF_UNIX")

        args = [sys.executable, _WORKER_SCRIPT, self.sock_path]
        if _CACHE_DIR:
            args.append(_CACHE_DIR)

        self.proc = subprocess.Popen(args)
        self.conn = self.listener.accept()
        self.listener.close()
        self.listener = None

    def stop(self):
        had_proc = self.proc is not None
        if self.conn is not None:
            try:
                self.conn.send({"cmd": "exit"})
                self.conn.close()
            except Exception:
                pass
            self.conn = None
        if self.listener is not None:
            try:
                self.listener.close()
            except Exception:
                pass
            self.listener = None
        if self.proc is not None:
            try:
                self.proc.wait(timeout=1.0)
            except Exception:
                try:
                    self.proc.terminate()
                    self.proc.wait(timeout=1.0)
                except Exception:
                    pass
            self.proc = None
        if self.sock_path and os.path.exists(self.sock_path):
            try:
                os.unlink(self.sock_path)
            except Exception:
                pass
            self.sock_path = None
        self.loaded_key = None
        self.layout = None
        self.model_scale = None
        return had_proc

    def load_model(self, path: str, precision: str, performance_mode: str):
        key = (path, os.stat(path).st_mtime_ns, precision, performance_mode)
        if self.is_alive() and self.loaded_key == key:
            return self.layout, self.model_scale

        self.start()
        self.conn.send({
            "cmd": "load_model",
            "path": path,
            "precision": precision,
            "performance_mode": performance_mode
        })
        resp = self.conn.recv()
        if resp.get("status") != "ok":
            self.stop()
            raise RuntimeError(f"Arc OpenVINO worker failed to load model: {resp.get('error')}")

        self.loaded_key = key
        self.layout = resp["layout"]
        self.model_scale = resp["scale"]
        return self.layout, self.model_scale


_WORKER = _ArcWorkerClient()
_loaded_wrapper = None


def unload_arc_models():
    """Completely terminate the isolated OpenVINO worker process and release 100% of GPU VRAM."""
    global _loaded_wrapper
    had_worker = _WORKER.stop()
    _loaded_wrapper = None
    gc.collect()
    try:
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.empty_cache()
    except Exception:
        pass
    if had_worker:
        print("[Arc Super Resolution] Unloaded OpenVINO worker process and reclaimed 100% GPU VRAM.")


class ArcOpenVINOInnerModel:
    """Inner model stub so ComfyUI logging and diagnostics can inspect the model."""
    def __init__(self, name: str = "ArcSuperResolutionModel"):
        self.model_name = name
        self.__class__.__name__ = f"ArcSuperResolution({name})"


class ArcOpenVINOModelWrapper:
    """Wrapper that satisfies ComfyUI's ModelPatcher interface for model management."""
    def __init__(self, path: str, layout: str, scale: int, precision: str, performance_mode: str):
        self.path = path
        self.layout = layout
        self.scale = scale
        self.precision = precision
        self.performance_mode = performance_mode
        self.load_device = comfy.model_management.get_torch_device()
        self.parent = None
        self.clone_base_uuid = f"arc_openvino_{path}"
        self.model = ArcOpenVINOInnerModel(os.path.basename(path))
        try:
            self._size = os.path.getsize(path) * 2  # approximate weights + workspace
        except Exception:
            self._size = 250 * 1024 * 1024

    def is_dynamic(self) -> bool:
        return False

    def loaded_size(self) -> int:
        return self._size

    def model_size(self) -> int:
        return self._size

    def current_loaded_device(self):
        return self.load_device

    def is_clone(self, other) -> bool:
        return hasattr(other, "clone_base_uuid") and other.clone_base_uuid == self.clone_base_uuid

    def detach(self, unpatch_all: bool = True):
        pass


class ArcLoadedModel:
    """ComfyUI LoadedModel compatible wrapper tracked in comfy.model_management.current_loaded_models."""
    def __init__(self, wrapper: ArcOpenVINOModelWrapper):
        self._patcher = wrapper
        self.device = wrapper.load_device
        self.currently_used = True
        self.model_finalizer = None
        self._patcher_finalizer = None
        self._real_model = weakref.ref(wrapper.model)

    @property
    def model(self):
        return self._patcher

    def real_model(self):
        return self._real_model()

    def is_dead(self) -> bool:
        return self._patcher is None

    def model_memory(self) -> int:
        return self._patcher.model_size() if self._patcher else 0

    def model_loaded_memory(self) -> int:
        return self._patcher.loaded_size() if self._patcher else 0

    def model_offloaded_memory(self) -> int:
        return 0

    def model_memory_required(self, device) -> int:
        return self.model_memory()

    def model_load(self, lowvram_model_memory=0, force_patch_weights=False):
        return self.real_model()

    def should_reload_model(self, force_patch_weights=False) -> bool:
        return False

    def model_unload(self, memory_to_free=None, unpatch_weights=True) -> bool:
        unload_arc_models()
        return True

    def model_use_more_vram(self, extra_memory, force_patch_weights=False) -> int:
        return 0

    def __eq__(self, other):
        if not hasattr(other, "model"):
            return False
        return self.model is other.model


if hasattr(comfy.model_management, "unload_all_models"):
    _orig_unload_all_models = comfy.model_management.unload_all_models

    def _patched_unload_all_models():
        try:
            unload_arc_models()
        except Exception:
            pass
        return _orig_unload_all_models()

    comfy.model_management.unload_all_models = _patched_unload_all_models


def _load_model(path, precision: str = PrecisionMode.AUTO_FP16.value, performance_mode: str = PerformanceMode.THROUGHPUT.value):
    global _loaded_wrapper
    key = (path, os.stat(path).st_mtime_ns, precision, performance_mode)
    if _WORKER.is_alive() and _WORKER.loaded_key == key and _loaded_wrapper is not None:
        for m in comfy.model_management.current_loaded_models:
            if isinstance(m, ArcLoadedModel) and m.model is _loaded_wrapper:
                m.currently_used = True
                break
        return _WORKER.layout, _WORKER.model_scale

    # Free memory from other models before starting worker / compiling
    try:
        approx_size = os.path.getsize(path) * 2
    except Exception:
        approx_size = 250 * 1024 * 1024
    comfy.model_management.free_memory(approx_size, comfy.model_management.get_torch_device())

    layout, model_scale = _WORKER.load_model(path, precision=precision, performance_mode=performance_mode)
    _loaded_wrapper = ArcOpenVINOModelWrapper(path, layout, model_scale, precision, performance_mode)

    comfy.model_management.current_loaded_models[:] = [
        m for m in comfy.model_management.current_loaded_models if not isinstance(m, ArcLoadedModel)
    ]
    loaded_entry = ArcLoadedModel(_loaded_wrapper)
    comfy.model_management.current_loaded_models.insert(0, loaded_entry)

    return layout, model_scale


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

        layout, model_scale = _load_model(path, precision=precision, performance_mode=performance_mode)

        if resize_type == UpscaleType.MODEL_NATIVE.value:
            out_w, out_h = int(w * model_scale), int(h * model_scale)
        elif resize_type == UpscaleType.SCALE_BY.value:
            out_w, out_h = int(w * scale), int(h * scale)
        elif resize_type == UpscaleType.TARGET_DIMENSIONS.value:
            out_w, out_h = target_width, target_height
        else:
            raise ValueError(f"Unsupported resize type: {resize_type}")

        in_bytes = b * h * w * c * 4
        out_bytes = b * out_h * out_w * c * 4
        available = psutil.virtual_memory().available

        spill_path = None
        out_shm = None
        if out_bytes > available // 2:
            fd, spill_path = tempfile.mkstemp(suffix=".f32", dir=folder_paths.get_temp_directory())
            os.close(fd)
            with open(spill_path, "wb") as f:
                f.truncate(out_bytes)
        else:
            out_shm = shared_memory.SharedMemory(create=True, size=out_bytes)

        in_shm = shared_memory.SharedMemory(create=True, size=in_bytes)
        try:
            in_arr = np.ndarray((b, h, w, c), dtype=np.float32, buffer=in_shm.buf)
            in_arr[:] = images.cpu().numpy()

            _WORKER.conn.send({
                "cmd": "upscale",
                "in_shm_name": in_shm.name,
                "in_shape": (b, h, w, c),
                "out_shm_name": out_shm.name if out_shm else None,
                "out_shape": (b, out_h, out_w, c),
                "spill_path": spill_path,
                "tile_size": tile_size,
                "tile_overlap": tile_overlap,
                "target_w": out_w,
                "target_h": out_h
            })

            pbar = comfy.utils.ProgressBar(b)
            t_start = time.perf_counter()

            # Single-line terminal bar — clean like sampler: `100%|████| 3/3 [01:39<00:00, 33.33s/it]`
            # Uses tqdm when available (auto handles rate/ETA), fallback mimics same format without duplication.
            term_pbar = None
            use_tqdm = False
            last_fallback = t_start
            try:
                from tqdm.auto import tqdm as _tqdm  # type: ignore

                # tqdm auto-detects non-tty and disables itself gracefully
                term_pbar = _tqdm(
                    total=b,
                    desc="Arc Super Resolution",
                    unit="frame",
                    dynamic_ncols=True,
                    leave=True,
                    mininterval=0.2,
                )
                use_tqdm = True
            except Exception:
                term_pbar = None
                use_tqdm = False

            def _fmt_time(s: float) -> str:
                s = int(s)
                m, sec = divmod(s, 60)
                h, m = divmod(m, 60)
                return f"{h:02d}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"

            try:
                while True:
                    comfy.model_management.throw_exception_if_processing_interrupted()
                    if _WORKER.conn.poll(0.05):
                        msg = _WORKER.conn.recv()
                        if msg.get("status") == "error":
                            raise RuntimeError(f"Arc Super Resolution worker error: {msg.get('error')}\n{msg.get('traceback', '')}")
                        elif msg.get("status") == "progress":
                            i = msg["frame"]
                            pbar.update(1)
                            if use_tqdm and term_pbar is not None:
                                # Let tqdm render rate + ETA itself — no manual postfix (avoids `1.55frame/s, 1.52 fps, ETA 99.9s` duplication)
                                term_pbar.update(1)
                            else:
                                # Fallback: throttled in-place bar via \r (one line, overwrites itself) — same clean format as tqdm
                                now = time.perf_counter()
                                if (now - last_fallback >= 0.5) or (i == b - 1):
                                    elapsed = now - t_start
                                    fps = (i + 1) / elapsed if elapsed > 0 else 0.0
                                    remaining = (b - (i + 1)) / fps if fps > 0 else 0.0
                                    pct = ((i + 1) / b) * 100
                                    bar_w = 30
                                    filled = int(bar_w * (i + 1) / b)
                                    bar = "█" * filled + "░" * (bar_w - filled)
                                    msg_str = f"\rArc Super Resolution: {pct:3.0f}%|{bar}| {i + 1}/{b} [{_fmt_time(elapsed)}<{_fmt_time(remaining)}, {fps:.2f} frame/s]"
                                    # pad to clear previous longer line
                                    print(msg_str.ljust(100), end="", flush=True)
                                    last_fallback = now
                                    if i == b - 1:
                                        print()
                        elif msg.get("status") == "done":
                            break
            finally:
                if term_pbar is not None:
                    try:
                        term_pbar.close()
                    except Exception:
                        pass

            if spill_path is not None:
                storage = torch.UntypedStorage.from_file(spill_path, False, out_bytes)
                out_tensor = torch.empty((b, out_h, out_w, c), dtype=torch.float32)
                out_tensor.set_(storage, 0, (b, out_h, out_w, c))
                weakref.finalize(out_tensor, _cleanup_spill, spill_path)
            else:
                out_arr = np.ndarray((b, out_h, out_w, c), dtype=np.float32, buffer=out_shm.buf)
                out_tensor = torch.from_numpy(out_arr.copy())

            return (out_tensor,)

        finally:
            in_shm.close()
            in_shm.unlink()
            if out_shm is not None:
                out_shm.close()
                out_shm.unlink()


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

__all__ = ["ArcSuperResolution", "ArcResampleFPS", "NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "unload_arc_models"]
