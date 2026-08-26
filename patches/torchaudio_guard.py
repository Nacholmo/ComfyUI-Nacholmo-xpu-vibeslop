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
    for finder in sys.meta_path:
        if getattr(finder, "__name__", "") == "_TorchAudioMetaFinder" or finder is _TorchAudioMetaFinder:
            _INSTALLED = True
            return
    sys.meta_path.insert(0, _TorchAudioMetaFinder)
    _INSTALLED = True


__all__ = ["apply"]
