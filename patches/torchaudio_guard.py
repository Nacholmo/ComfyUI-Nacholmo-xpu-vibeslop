"""Torchaudio CUDA fallback guard for Intel Arc XPU & Non-CUDA Linux.

PyPI torchaudio wheels for Linux include C++ extensions (_torchaudio.abi3.so)
compiled against CUDA (libcudart.so). On Intel Arc (XPU) systems without CUDA,
importing torchaudio crashes with 'libcudart.so: cannot open shared object file'.

This patch intercepts torchaudio's extension loader to catch missing CUDA/C++
runtime libraries gracefully and fall back to pure PyTorch/CPU audio processing,
preventing fatal startup crashes in ComfyUI audio nodes and VAEs.
"""

import logging
import sys
from importlib.machinery import PathFinder

log = logging.getLogger("ComfyUI-Nacholmo-xpu-vibeslop.TorchAudioGuard")

_INSTALLED = False


class _TorchAudioMetaFinder:
    @classmethod
    def find_spec(cls, fullname, path=None, target=None):
        if fullname == "torchaudio._extension.utils":
            spec = PathFinder.find_spec(fullname, path, target)
            if spec and spec.loader:
                orig_exec = spec.loader.exec_module

                def exec_module_patched(module):
                    orig_exec(module)
                    if hasattr(module, "_load_lib"):
                        orig_load_lib = module._load_lib

                        def safe_load_lib(lib):
                            try:
                                return orig_load_lib(lib)
                            except Exception as e:
                                log.debug(f"Intercepted torchaudio extension load failure ({lib}): {e}")
                                return False

                        module._load_lib = safe_load_lib

                spec.loader.exec_module = exec_module_patched
            return spec
        return None


def apply():
    global _INSTALLED
    if _INSTALLED:
        return

    # 1. Install meta finder for future imports of torchaudio
    already_in_meta = any(
        getattr(finder, "__name__", "") == "_TorchAudioMetaFinder" or finder is _TorchAudioMetaFinder
        for finder in sys.meta_path
    )
    if not already_in_meta:
        sys.meta_path.insert(0, _TorchAudioMetaFinder)

    # 2. Patch torchaudio._extension.utils directly if already imported
    mod = sys.modules.get("torchaudio._extension.utils")
    if mod and hasattr(mod, "_load_lib"):
        orig_load_lib = mod._load_lib
        if not getattr(orig_load_lib, "_nacholmo_patched", False):
            def safe_load_lib(lib):
                try:
                    return orig_load_lib(lib)
                except Exception as e:
                    log.debug(f"Intercepted torchaudio extension load failure ({lib}): {e}")
                    return False
            safe_load_lib._nacholmo_patched = True
            mod._load_lib = safe_load_lib

    # 3. Patch torch.ops.load_library if torch is present
    torch_mod = sys.modules.get("torch")
    if torch_mod and hasattr(torch_mod, "ops") and hasattr(torch_mod.ops, "load_library"):
        orig_load_library = torch_mod.ops.load_library
        if not getattr(orig_load_library, "_nacholmo_patched", False):
            def safe_load_library(path):
                try:
                    return orig_load_library(path)
                except Exception as e:
                    if "_torchaudio" in str(path):
                        log.debug(f"Intercepted torchaudio torch.ops.load_library failure ({path}): {e}")
                        return
                    raise
            safe_load_library._nacholmo_patched = True
            torch_mod.ops.load_library = safe_load_library

    _INSTALLED = True


__all__ = ["apply"]

