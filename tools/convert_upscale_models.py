# Converts ComfyUI upscale models (.pth/.safetensors) to ONNX so they can run
# on Intel Arc GPUs through the Arc Super Resolution node (OpenVINO).
#
# Usage:
#   python convert_upscale_models.py                # convert everything found
#   python convert_upscale_models.py path [path..]  # convert specific files
#   python convert_upscale_models.py --force        # force re-conversion
#
# The ONNX file is written next to the source model and verified against the
# original network before being kept.

import argparse
import inspect
import os
import sys

import numpy as np
import openvino as ov
import torch
from spandrel import ImageModelDescriptor, ModelLoader

def _find_comfy_root():
    cur = os.path.abspath(__file__)
    while cur and cur != os.path.dirname(cur):
        cur = os.path.dirname(cur)
        if os.path.exists(os.path.join(cur, "main.py")) and os.path.exists(os.path.join(cur, "folder_paths.py")):
            return cur
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_ROOT = _find_comfy_root()
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import folder_paths  # noqa: E402
from utils.extra_config import load_extra_path_config  # noqa: E402

_EXTRA_PATHS = os.path.join(_ROOT, "extra_model_paths.yaml")
if os.path.isfile(_EXTRA_PATHS):
    load_extra_path_config(_EXTRA_PATHS)

_SOURCE_EXTENSIONS = (".pth", ".safetensors")
_VERIFY_SIZE = (96, 112)  # h, w


def _export_onnx(model, dummy, out_path):
    kwargs = {}
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        kwargs["dynamo"] = False
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            out_path,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch", 2: "height", 3: "width"}, "output": {0: "batch", 2: "height", 3: "width"}},
            opset_version=17,
            **kwargs,
        )


def convert(path, force: bool = False):
    name = os.path.basename(path)
    stem = os.path.splitext(name)[0]
    out_path = os.path.join(os.path.dirname(path), stem + ".onnx")
    if os.path.exists(out_path) and not force:
        try:
            if os.path.getmtime(out_path) >= os.path.getmtime(path):
                print(f"skip   {name}: {stem}.onnx already exists and is newer")
                return True
            print(f"redo   {name}: source newer than existing {stem}.onnx, reconverting")
        except OSError:
            print(f"skip   {name}: {stem}.onnx already exists")
            return True
    try:
        descriptor = ModelLoader().load_from_file(path)
    except Exception as error:
        print(f"fail   {name}: cannot load ({error})")
        return False
    if not isinstance(descriptor, ImageModelDescriptor):
        print(f"fail   {name}: not a single-image upscale model")
        return False

    arch_name = getattr(descriptor, "architecture", "Unknown")
    scale_desc = getattr(descriptor, "scale", None)

    net = descriptor.to(device="cpu", dtype=torch.float32).eval().model
    h, w = _VERIFY_SIZE
    dummy = torch.rand(1, descriptor.input_channels, h, w)
    try:
        _export_onnx(net, dummy, out_path)
    except Exception as error:
        if os.path.exists(out_path):
            os.remove(out_path)
        print(f"fail   {name}: export failed ({error})")
        return False

    with torch.no_grad():
        reference = net(dummy).numpy()
    core = ov.Core()
    compiled = core.compile_model(core.read_model(out_path), "CPU")
    result = np.asarray(compiled([dummy.numpy()])[compiled.output(0)])
    diff = float(np.abs(result - reference).max())
    scale = result.shape[-1] // w
    result_shape = tuple(result.shape)
    # Release large temporaries before the next model (RAM spike on big nets).
    try:
        del net, reference, compiled, result
        import gc as _gc
        _gc.collect()
    except Exception:
        pass
    if diff > 2e-4 or scale < 1 or list(result_shape[-2:]) != [h * scale, w * scale]:
        try:
            if os.path.exists(out_path):
                os.remove(out_path)
        except OSError:
            pass
        print(f"fail   {name}: verification failed (max diff {diff:g}, output {result_shape})")
        return False

    arch_info = f"[{arch_name}]" if arch_name else ""
    print(f"ok     {name} -> {stem}.onnx ({arch_info} x{scale}, max diff {diff:g})")
    return True


def main():
    parser = argparse.ArgumentParser(description="Convert ComfyUI upscale models (.pth/.safetensors) to ONNX for Intel Arc GPUs")
    parser.add_argument("paths", nargs="*", help="Specific model files to convert (default: scan upscale_models folders)")
    parser.add_argument("-f", "--force", action="store_true", help="Force overwrite existing .onnx files")
    args = parser.parse_args()

    paths = args.paths
    if not paths:
        for folder in folder_paths.get_folder_paths("upscale_models"):
            if not os.path.exists(folder):
                continue
            for entry in sorted(os.listdir(folder)):
                full = os.path.join(folder, entry)
                if os.path.isfile(full) and entry.lower().endswith(_SOURCE_EXTENSIONS):
                    paths.append(full)
    if not paths:
        print("no upscale models found")
        return 1
    results = [convert(path, force=args.force) for path in paths]
    failed = results.count(False)
    print(f"\n{len(results) - failed}/{len(results)} converted")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

