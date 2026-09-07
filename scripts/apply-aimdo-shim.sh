#!/bin/bash
# AIMDO malloc_graph compat shim (Intel Arc XPU).
#
# Newer ComfyUI core hard-imports comfy_aimdo.malloc_graph (NVIDIA CUDA-graph
# recording surface, official comfy-aimdo>=0.5), while the XPU provider stack
# tracks official 0.4.15, which lacks that module. On XPU the module is
# import-only: every use is gated behind is_device_cuda(), so its native
# calls never execute here.
#
# Provenance & licensing: the file is extracted at install time from the
# official PyPI wheel (default AIMDO_SHIM_VERSION=0.5.2, GPLv3 — same licence
# as the installed comfy-aimdo package). Nothing GPL is stored in this repo;
# no upstream files are modified. Idempotent: re-running refreshes the shim.
#
# Usage: apply-aimdo-shim.sh  (runs inside the ComfyUI venv)
set -uo pipefail

SHIM_VERSION="${AIMDO_SHIM_VERSION:-0.5.2}"
SHIM_MODULE="comfy_aimdo/malloc_graph.py"

# comfy_aimdo is a namespace package (__file__ is None) — locate the
# installed official package dir via importlib.metadata (never the
# provider's _vendor tree, which must stay pristine per its contract).
AIMDO_DIR="$(python -c 'import importlib.metadata as m; print(str(m.distribution("comfy-aimdo").locate_file("comfy_aimdo")))' 2>/dev/null || true)"
if [ -z "${AIMDO_DIR:-}" ] || [ ! -d "$AIMDO_DIR" ]; then
    echo "[aimdo-shim] comfy_aimdo not importable; skipping." >&2
    exit 0
fi

if [ -f "$AIMDO_DIR/malloc_graph.py" ]; then
    echo "[aimdo-shim] already present at $AIMDO_DIR/malloc_graph.py — refreshing."
fi

_TMPD="$(mktemp -d)"
trap 'rm -rf "$_TMPD"' EXIT
if ! pip download --no-deps -d "$_TMPD" "comfy-aimdo==$SHIM_VERSION" > /dev/null 2>&1; then
    if [ -f "$AIMDO_DIR/malloc_graph.py" ]; then
        echo "[aimdo-shim] WARNING: download failed, keeping existing shim." >&2
        exit 0
    fi
    echo "[aimdo-shim] ERROR: could not download comfy-aimdo==$SHIM_VERSION." >&2
    exit 1
fi

_WHEEL=( "$_TMPD"/comfy_aimdo-"$SHIM_VERSION"-*.whl )
if python -c "
import zipfile, sys
z = zipfile.ZipFile('${_WHEEL[0]}')
z.extract('$SHIM_MODULE', '$_TMPD/extracted')
" && cp "$_TMPD/extracted/$SHIM_MODULE" "$AIMDO_DIR/malloc_graph.py"; then
    echo "[aimdo-shim] installed $SHIM_MODULE from official comfy-aimdo==$SHIM_VERSION."
else
    echo "[aimdo-shim] ERROR: extraction failed." >&2
    exit 1
fi

if python -c "import comfy_aimdo.malloc_graph; print('[aimdo-shim] import OK:', comfy_aimdo.malloc_graph.__file__)"; then
    exit 0
else
    echo "[aimdo-shim] ERROR: shim import failed." >&2
    exit 1
fi
