# Converts WINT8-quantized MiniMax-H3 diffusion models (.safetensors holding int8
# weights + weight_scale + comfy_quant markers, optionally QuaRot-rotated) back
# to a full-precision GGUF ready for llama.cpp quantization.
#
# Do NOT feed the raw int8 file to a plain safetensors->GGUF converter: that
# silently bakes rotated int8 values as if they were real weights. This tool
# inverts this suite's own WINT8 quantizer math instead:
#   W = (int8.float() * per_row_scale) @ H_block   (per convrot_groupsize group)
# which satisfies x @ W^T == (x @ H) @ W_rot^T as executed by Int8XPUOps.
#
# HADAMARD VARIANT (critical): third-party WINT8 files (Dasiwa, ErosMax, ...,
# anything marked convrot from the ConvRot ecosystem, including core's native
# int8 path, comfy-kitchen and the OMNI kernels) use the REGULAR Hadamard
# (4x4-Kronecker construction, power-of-4 sizes). The suite's own quantizer
# historically used Sylvester (2x2) Hadamards via scipy; the two matrices are
# both orthogonal/symmetric but DIFFER, and un-rotating with the wrong one
# silently yields plausible-looking garbage (verified: spikiness restored with
# regular-H, mosaic output with Sylvester-H). Default is regular; --hadamard
# sylvester keeps the legacy behavior for files made with this suite's own
# quantizer.
#
# GGUF tensor-layout conventions (arch tag, ftype selection, F32 rules for
# 1-D/small tensors) are adapted from city96/ComfyUI-GGUF tools/convert.py
# (Copyright (c) City96, Apache-2.0).
#
# Usage (run with the ComfyUI venv python):
#   python tools/convert_wint8_minimax_h3.py --src model.safetensors [--dst model-BF16.gguf]

import argparse
import gc
import importlib.util
import json
import logging
import os
import sys

import gguf
import torch
from safetensors.torch import load_file
from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

QUANTIZATION_THRESHOLD = 1024
MAX_TENSOR_NAME_LENGTH = 127
MAX_TENSOR_DIMS = 4
# 2-D float tables that must keep source precision: the adaLN curve table is
# consumed by a mixed-dtype torch.lerp in core's MiniMax forward, and other
# backends silently promote while XPU oneDNN rejects the mix.
HIPREC_KEYS = ["adaln_t_table"]


def _load_quarot():
    path = os.path.join(REPO_ROOT, "nodes", "wint8", "wint8_quarot.py")
    spec = importlib.util.spec_from_file_location("wint8_quarot", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_regular_hadamard(size, dtype=torch.float32):
    """Normalized REGULAR orthogonal Hadamard (ConvRot family): Kronecker
    products of the 4x4 base, sizes must be powers of 4. Same construction as
    comfy-kit's tensor/int8_utils and the OMNI convrot kernels (which consume
    third-party convrot-quantized files), and NOT the same matrix as the
    Sylvester (2x2) Hadamard from scipy."""
    import math
    if size < 4 or (size & (size - 1)) != 0 or math.log(size, 4) % 1 != 0:
        raise ValueError(f"Regular Hadamard size must be a power of 4, got {size}")
    h4 = torch.tensor(
        [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]],
        dtype=dtype)
    h = h4
    n = 4
    while n < size:
        h = torch.kron(h, h4)
        n *= 4
    return h / (size ** 0.5)


def _strip_prefix(state_dict):
    for pfx in ["model.diffusion_model.", "model."]:
        if any(k.startswith(pfx) for k in state_dict.keys()):
            logging.info(f"State dict prefix found: '{pfx}'")
            return {k[len(pfx):] if k.startswith(pfx) else k: v for k, v in state_dict.items()}
    if all(k.startswith("net.") for k in state_dict.keys()):
        logging.info("State dict prefix found: 'net.'")
        return {k[len("net."):]: v for k, v in state_dict.items()}
    return state_dict


def _parse_marker(raw):
    try:
        return json.loads(bytes(raw.tolist()).decode("utf-8"))
    except Exception:
        return {}


def dequantize_state_dict(sd, quarot, hadamard="regular"):
    """Replace WINT8 int8 weights with full-precision unrotated weights.

    Consumes sibling weight_scale/comfy_quant/input_scale tensors.
    hadamard: "regular" (ConvRot ecosystem default) or "sylvester" (this
    suite's own legacy quantizer). Returns the number of dequantized layers.
    """
    use_regular = (hadamard == "regular")
    hadamard_cache = {}
    n_dq = 0
    for key in list(sd.keys()):
        if key.endswith(".comfy_quant") or key.endswith("weight_scale") or key.endswith("input_scale"):
            continue
        t = sd[key]
        if t.dtype != torch.int8 or not key.endswith(".weight"):
            continue
        base = key[: -len("weight")]  # keeps trailing dot
        scale = sd.pop(base + "weight_scale", None)
        if scale is None:
            raise KeyError(f"missing scale for blaze int8 weight '{key}'")
        meta = _parse_marker(sd.pop(base + "comfy_quant", torch.zeros(1, dtype=torch.uint8)))
        with torch.no_grad():
            w = t.to(torch.float32) * scale.to(torch.float32).view(-1, 1)
            if meta.get("quarot", False) or meta.get("convrot", False):
                gs = int(meta.get("group_size", meta.get("convrot_groupsize", 128)))
                out_f, in_f = w.shape
                if in_f % gs != 0:
                    raise ValueError(f"in_features {in_f} not divisible by group_size {gs} ({key})")
                if gs not in hadamard_cache:
                    if use_regular:
                        hadamard_cache[gs] = build_regular_hadamard(gs, dtype=torch.float32)
                    else:
                        hadamard_cache[gs] = quarot.build_hadamard(gs, device="cpu", dtype=torch.float32)
                H = hadamard_cache[gs]
                w = (w.view(out_f, in_f // gs, gs) @ H).reshape(out_f, in_f)
            orig = str(meta.get("orig_dtype", "torch.bfloat16"))
            dtype = {"torch.bfloat16": torch.bfloat16, "torch.float16": torch.float16,
                     "torch.float32": torch.float32}.get(orig, torch.bfloat16)
            sd[key] = w.to(dtype)
            del w, t, scale
            n_dq += 1
            if n_dq % 25 == 0:
                gc.collect()
    for key in list(sd.keys()):
        if key.endswith(".comfy_quant") or key.endswith("weight_scale") or key.endswith("input_scale"):
            del sd[key]
    gc.collect()
    return n_dq


def write_gguf(sd, dst_path):
    names = sorted(sd.keys(), key=len, reverse=True)
    if names and len(names[0]) > MAX_TENSOR_NAME_LENGTH:
        bad = ", ".join(f"{k!r}" for k in names if len(k) > MAX_TENSOR_NAME_LENGTH)
        raise ValueError(f"Tensor names exceed {MAX_TENSOR_NAME_LENGTH} chars: {bad}")

    dtypes = [x.dtype for x in sd.values()]
    main_dtype = max(set(dtypes), key=dtypes.count)
    if main_dtype == torch.bfloat16:
        ftype_gguf = gguf.LlamaFileType.MOSTLY_BF16
    else:
        ftype_gguf = gguf.LlamaFileType.MOSTLY_F16

    writer = gguf.GGUFWriter(path=None, arch="minimax_h3")
    writer.add_quantization_version(gguf.GGML_QUANT_VERSION)
    writer.add_file_type(ftype_gguf)

    max_name_len = len(names[0]) if names else 0
    for key in tqdm(sd.keys()):
        data = sd[key]
        old_dtype = data.dtype
        if data.dtype == torch.bfloat16:
            data = data.to(torch.float32).numpy()
            data_qtype = gguf.GGMLQuantizationType.BF16
        elif data.dtype in [d for d in
                             (getattr(torch, "float8_e4m3fn", None),
                              getattr(torch, "float8_e5m2", None)) if isinstance(d, torch.dtype)]:
            data = data.to(torch.float16).numpy()
            data_qtype = gguf.GGMLQuantizationType.F16
        else:
            data = data.numpy()
            data_qtype = gguf.GGMLQuantizationType.F16

        if len(data.shape) > MAX_TENSOR_DIMS:
            raise NotImplementedError(f"Tensor exceeds GGUF dims: {key} {data.shape}")

        n_params, n_dims = 1, len(data.shape)
        for d in data.shape:
            n_params *= d
        if old_dtype in (torch.float32, torch.bfloat16):
            if n_dims == 1 or n_params <= QUANTIZATION_THRESHOLD:
                data_qtype = gguf.GGMLQuantizationType.F32
            elif any(h in key for h in HIPREC_KEYS):
                data_qtype = gguf.GGMLQuantizationType.F32

        try:
            data = gguf.quants.quantize(data, data_qtype)
        except (AttributeError, gguf.QuantError) as e:
            tqdm.write(f"falling back to F16: {e}")
            data_qtype = gguf.GGMLQuantizationType.F16
            data = gguf.quants.quantize(data, data_qtype)

        shape_str = f"{{{', '.join(str(n) for n in reversed(data.shape))}}}"
        tqdm.write(f"{f'%-{max_name_len + 4}s' % key} {old_dtype} --> {data_qtype.name}, shape = {shape_str}")
        writer.add_tensor(key, data, raw_dtype=data_qtype)

    writer.write_header_to_file(path=dst_path)
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()
    return dst_path


def convert_file(src, dst=None, hadamard="regular"):
    quarot = _load_quarot()
    logging.info("Loading safetensors (full model into RAM)...")
    sd = _strip_prefix(load_file(src))
    if not ("audio_patch_proj.weight" in sd and "video_patch_proj.weight" in sd):
        raise RuntimeError("Not a MiniMax-H3 state dict (missing audio/video patch proj)")
    logging.info(f"* Architecture detected from input: minimax_h3 ({len(sd)} tensors)")
    logging.info(f"* Hadamard variant: {hadamard}")
    n_dq = dequantize_state_dict(sd, quarot, hadamard=hadamard)
    logging.info(f"Dequantized {n_dq} int8 layers, {len(sd)} tensors remain")
    if dst is None:
        base, _ = os.path.splitext(src)
        dst = f"{base}-BF16.gguf"
    write_gguf(sd, dst)
    logging.info(f"WROTE {dst}")
    return dst


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description="WINT8 MiniMax-H3 safetensors -> BF16 GGUF")
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", default=None)
    ap.add_argument("--hadamard", default="regular", choices=["regular", "sylvester"],
                    help="Hadamard family used at quantize time (default: regular/ConvRot)")
    args = ap.parse_args()
    if not os.path.isfile(args.src):
        ap.error("No input provided!")
    convert_file(args.src, args.dst, hadamard=args.hadamard)


if __name__ == "__main__":
    main()
