"""Standalone OpenVINO worker process for Arc Super Resolution.

Executed as an independent subprocess communicating over AF_UNIX socket IPC.
This guarantees clean process isolation and 100% GPU VRAM reclamation upon unload.
"""

import os
import sys
import traceback
from multiprocessing import shared_memory, resource_tracker
from multiprocessing.connection import Client
import numpy as np
import torch

try:
    import openvino as ov
except ImportError:
    ov = None


def _unregister_shm(name):
    if not name:
        return
    try:
        shm_name = f"/{name.lstrip('/')}"
        resource_tracker.unregister(shm_name, "shared_memory")
    except Exception:
        pass


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


def main():
    if len(sys.argv) < 2:
        sys.exit(1)

    sock_path = sys.argv[1]
    cache_dir = sys.argv[2] if len(sys.argv) > 2 else None

    if ov is None:
        try:
            conn = Client(sock_path, family="AF_UNIX")
            conn.send({"status": "error", "error": "OpenVINO is not installed. Please run 'pip install openvino'."})
            conn.close()
        except Exception:
            pass
        sys.exit(1)

    conn = Client(sock_path, family="AF_UNIX")

    core = ov.Core()
    if cache_dir:
        try:
            core.set_property({"CACHE_DIR": cache_dir})
        except Exception:
            pass

    current_model_key = None
    compiled = None
    layout = None
    model_scale = None

    while True:
        try:
            msg = conn.recv()
        except (EOFError, KeyboardInterrupt):
            break
        except Exception as e:
            try:
                conn.send({"status": "error", "error": f"IPC recv failed: {e}", "traceback": traceback.format_exc()})
            except Exception:
                break
            continue

        if not isinstance(msg, dict):
            try:
                conn.send({"status": "error", "error": f"Invalid IPC message (expected dict, got {type(msg).__name__})"})
            except Exception:
                break
            continue
        cmd = msg.get("cmd")
        if cmd == "exit":
            break

        elif cmd == "load_model":
            try:
                path = msg["path"]
                precision = msg["precision"]
                performance_mode = msg["performance_mode"]
                key = (path, os.stat(path).st_mtime_ns, precision, performance_mode)

                if current_model_key != key:
                    if "GPU" not in core.get_available_devices():
                        raise RuntimeError(
                            "OpenVINO reports no GPU device. Install the Intel compute runtime "
                            "(OpenCL) for your Arc GPU, e.g. 'intel-opencl-icd' or 'intel-compute-runtime'."
                        )

                    name = os.path.basename(path)
                    model = core.read_model(path)
                    if len(model.inputs) != 1 or len(model.outputs) != 1:
                        raise ValueError(f"{name}: node supports models with exactly one input and one output")

                    shape = model.input(0).partial_shape
                    if len(shape) != 4:
                        raise ValueError(f"{name}: expected a 4D NCHW/NHWC input, got rank {len(shape)}")
                    c2 = shape[1].get_length() if not shape[1].is_dynamic else None
                    c4 = shape[3].get_length() if not shape[3].is_dynamic else None
                    if c2 == 3:
                        layout = "NCHW"
                        probe_channels = 3
                    elif c4 == 3:
                        layout = "NHWC"
                        probe_channels = 3
                    elif c2 is not None and c2 > 0:
                        layout = "NCHW"
                        probe_channels = int(c2)
                    elif c4 is not None and c4 > 0:
                        layout = "NHWC"
                        probe_channels = int(c4)
                    else:
                        layout = "NCHW"
                        probe_channels = 3

                    config = {}
                    if precision == "auto / fp16 (fastest)":
                        config["INFERENCE_PRECISION_HINT"] = "f16"
                    elif precision == "fp32 (full precision)":
                        config["INFERENCE_PRECISION_HINT"] = "f32"

                    if performance_mode == "throughput (video / batch)":
                        config["PERFORMANCE_HINT"] = "THROUGHPUT"
                    else:
                        config["PERFORMANCE_HINT"] = "LATENCY"

                    compiled = core.compile_model(model, "GPU", config)
                    req = compiled.create_infer_request()
                    probe = _infer_patch(req, np.zeros((128, 128, probe_channels), np.float32), layout)
                    del req
                    model_scale = probe.shape[0] // 128
                    if model_scale < 1 or probe.shape[:2] != (128 * model_scale, 128 * model_scale):
                        raise ValueError(f"{name}: unexpected output shape {probe.shape} for a 128x128 input with {probe_channels} channels")

                    current_model_key = key

                try:
                    conn.send({"status": "ok", "layout": layout, "scale": model_scale})
                except Exception:
                    pass
            except Exception as e:
                try:
                    conn.send({"status": "error", "error": str(e), "traceback": traceback.format_exc()})
                except Exception:
                    pass

        elif cmd == "upscale":
            in_shm = None
            out_shm = None
            req = None
            in_shm_name = msg.get("in_shm_name")
            out_shm_name = msg.get("out_shm_name")
            if compiled is None:
                try:
                    conn.send({"status": "error", "error": "No model loaded (send load_model before upscale)"})
                except Exception:
                    pass
                continue
            try:
                in_shape = tuple(int(v) for v in msg["in_shape"])
                out_shape = tuple(int(v) for v in msg["out_shape"])
                if len(in_shape) != 4 or len(out_shape) != 4:
                    raise ValueError(f"expected 4D in/out shapes, got {msg['in_shape']} / {msg['out_shape']}")
                spill_path = msg.get("spill_path")
                tile_size = int(msg.get("tile_size", 512))
                tile_overlap = int(msg.get("tile_overlap", 64))
                target_w = int(msg["target_w"])
                target_h = int(msg["target_h"])
                if target_w < 1 or target_h < 1:
                    raise ValueError(f"invalid target dims {target_w}x{target_h}")

                in_shm = shared_memory.SharedMemory(name=in_shm_name)
                _unregister_shm(in_shm_name)
                in_arr = np.ndarray(in_shape, dtype=np.float32, buffer=in_shm.buf)

                if out_shm_name:
                    out_shm = shared_memory.SharedMemory(name=out_shm_name)
                    _unregister_shm(out_shm_name)
                    out_arr = np.ndarray(out_shape, dtype=np.float32, buffer=out_shm.buf)
                else:
                    if not spill_path:
                        raise ValueError("upscale request has neither out_shm_name nor spill_path")
                    out_arr = np.memmap(spill_path, dtype=np.float32, mode="r+", shape=out_shape)

                b, h, w, c = in_shape
                req = compiled.create_infer_request()

                for i in range(b):
                    frame_np = in_arr[i]
                    up = _super_resolve(compiled, layout, model_scale, frame_np, tile_size, tile_overlap, request=req)

                    if (up.shape[1], up.shape[0]) != (target_w, target_h):
                        t = torch.from_numpy(up).permute(2, 0, 1)[None]
                        t = torch.nn.functional.interpolate(t, size=(target_h, target_w), mode="bicubic", align_corners=False, antialias=True)
                        up = t[0].permute(1, 2, 0).numpy()

                    np.clip(up, 0.0, 1.0, out=up)
                    out_arr[i] = up
                    try:
                        conn.send({"status": "progress", "frame": i, "total": b})
                    except Exception:
                        # Parent went away; stop work and let outer loop exit.
                        break

                if spill_path:
                    try:
                        out_arr.flush()
                    except Exception:
                        pass

                try:
                    conn.send({"status": "done"})
                except Exception:
                    pass
            except Exception as e:
                try:
                    conn.send({"status": "error", "error": str(e), "traceback": traceback.format_exc()})
                except Exception:
                    pass
            finally:
                if req is not None:
                    del req
                if in_shm is not None:
                    try:
                        in_shm.close()
                    except Exception:
                        pass
                if out_shm is not None:
                    try:
                        out_shm.close()
                    except Exception:
                        pass

    conn.close()


if __name__ == "__main__":
    main()
