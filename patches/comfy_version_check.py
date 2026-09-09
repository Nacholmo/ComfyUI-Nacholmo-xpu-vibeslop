"""Exempt provider-locked packages from core's version-compatibility warning.

ComfyUI core (>= the malloc_graph era) warns for every comfy* package older
than requirements.txt — including comfy-kitchen/comfy-aimdo, which this
suite intentionally HOLDS below requirements.txt for the XPU provider hash
contract (see docs/rolling-release.md §1, lockstep triple). Following the
remedy printed with the warning (pip install -r requirements.txt) would
float officials past provider-accepted versions and silently disable
Kitchen XPU dispatch + AIMDO VBAR.

The patch wraps utils.install_util.get_required_packages_versions (the
single choke point feeding both the boot banner and the server/UI version
API) so the two held packages report required == installed. Everything
else passes through untouched. Idempotent via _INSTALLED guard.
"""

import logging
import sys

log = logging.getLogger("ComfyUI-Nacholmo-xpu-vibeslop.VersionCheck")

_INSTALLED = False
_LOGGED_HOLDS = set()

# Packages intentionally held below requirements.txt for the provider
# contract. Values here are informational only — the wrapper clamps to
# whatever is actually installed.
HELD_PACKAGES = ("comfy-kitchen", "comfy-aimdo")


def _wrap_module(module):
    if getattr(module, "_nacholmo_version_check_wrapped", False):
        return True
    orig_fn = getattr(module, "get_required_packages_versions", None)
    if not callable(orig_fn):
        return False

    def get_required_packages_versions_patched():
        versions = orig_fn()
        if not versions:
            return versions
        try:
            import importlib.metadata as metadata
        except ImportError:
            return versions
        for name in HELD_PACKAGES:
            if name not in versions:
                continue
            try:
                installed = metadata.version(name)
            except Exception:
                continue
            if versions[name] != installed:
                if name not in _LOGGED_HOLDS:
                    log.info(
                        "[version-check] holding %s at installed %s "
                        "(requirements wants %s; XPU provider contract — see docs/rolling-release.md)",
                        name, installed, versions[name],
                    )
                    _LOGGED_HOLDS.add(name)
                versions[name] = installed
        return versions

    module.get_required_packages_versions = get_required_packages_versions_patched
    module._nacholmo_version_check_wrapped = True
    return True


def apply():
    global _INSTALLED
    if _INSTALLED:
        return
    # Already-loaded module (e.g. double import in tests): patch in place.
    existing = sys.modules.get("utils.install_util")
    if existing is not None:
        try:
            if _wrap_module(existing):
                _INSTALLED = True
                return
        except Exception as e:
            log.debug(f"[version-check] in-place wrap failed: {e}")
    # Meta-path hook: patch before the module executes (house pattern —
    # never edit comfy/ or upstream files directly).
    from importlib.machinery import PathFinder

    class _VersionCheckMetaFinder:
        @classmethod
        def find_spec(cls, fullname, path=None, target=None):
            if fullname != "utils.install_util":
                return None
            spec = PathFinder.find_spec(fullname, path, target)
            if spec and spec.loader:
                orig_exec = spec.loader.exec_module

                def exec_module_patched(module):
                    orig_exec(module)
                    try:
                        _wrap_module(module)
                    except Exception as e:
                        log.debug(f"[version-check] exec wrap failed: {e}")

                spec.loader.exec_module = exec_module_patched
            return spec

    for finder in sys.meta_path:
        if finder is _VersionCheckMetaFinder or getattr(finder, "__name__", "") == "_VersionCheckMetaFinder":
            _INSTALLED = True
            return
    sys.meta_path.insert(0, _VersionCheckMetaFinder)
    _INSTALLED = True


__all__ = ["apply", "HELD_PACKAGES"]
