#!/bin/bash
# Launch ComfyUI on Intel Arc (XPU) with optimized environment settings.
#
# VRAM mode is the DEFAULT (Omni-aligned): AIMDO DynamicVRAM VBAR owns
# allocation with --reserve-vram headroom; our fraction cap stays dormant and
# PYTORCH_ALLOC_CONF is left unset (refused with the UR hook).
#   Default:  ./launch_xpu.sh
#     → python main.py --enable-dynamic-vram --reserve-vram 4 "$@"
#   Direct:   XPU_VRAM_MODE=direct ./launch_xpu.sh
#     Native torch XPU allocator capped at XPU_VRAM_FRACTION (default 0.88).
#   Override: XPU_VRAM_MODE=vram|direct, OMNI_COMFYUI_RESERVE_VRAM_GB=N.
#     Explicit --enable-dynamic-vram / --reserve-vram flags are never duplicated.
#
# Required in both modes: oneAPI setvars + LD_LIBRARY_PATH ordering
# (venv/lib : torch/lib : rest) so AIMDO sees the XPU devices.

# Locate ComfyUI root directory
COMFY_ROOT=""
if [ -f "./main.py" ] && [ -f "./execution.py" ]; then
    COMFY_ROOT="$PWD"
else
    DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    while [ "$DIR" != "/" ]; do
        if [ -f "$DIR/main.py" ] && [ -f "$DIR/execution.py" ]; then
            COMFY_ROOT="$DIR"
            break
        fi
        DIR="$(dirname "$DIR")"
    done
fi

if [ -z "$COMFY_ROOT" ]; then
    echo "[launch_xpu] Error: Could not locate ComfyUI root directory (main.py not found)." >&2
    exit 1
fi

_INVOCATION_DIR="$PWD"
cd "$COMFY_ROOT" || exit 1

# Activate virtual environment if present
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# --- oneAPI runtime (Omni): compiler/tool discovery for the XPU stack ---
# Sourced with errexit/nounset relaxed: oneAPI vars.sh is not `set -u` clean
# (OCL_ICD_FILENAMES) and must not kill this script on nonzero return.
if [ -f "/opt/intel/oneapi/setvars.sh" ]; then
    _launch_flags="$-"
    set +e +u
    # shellcheck disable=SC1091
    . /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 || true
    case "$_launch_flags" in *e*) set -e ;; *) set +e ;; esac
    case "$_launch_flags" in *u*) set -u ;; *) set +u ;; esac
    unset _launch_flags
fi

# --- Shared-library ordering (Omni, mandatory for DynamicVRAM/AIMDO) ---
# The venv's Torch-matched SYCL/UR runtime (venv/lib + torch/lib) must stay
# FIRST on LD_LIBRARY_PATH. Without this ordering, --enable-dynamic-vram
# (AIMDO init) leaves the process with XPU device count zero.
_VENV_DIR="$COMFY_ROOT/venv"
[ -d "$COMFY_ROOT/.venv" ] && [ ! -d "$_VENV_DIR" ] && _VENV_DIR="$COMFY_ROOT/.venv"
_TORCH_LIB="$(find "$_VENV_DIR/lib" -maxdepth 4 -type d -path '*site-packages/torch/lib' 2>/dev/null | head -n 1)"
if [ -n "${_TORCH_LIB:-}" ] && [ -d "$_TORCH_LIB" ]; then
    export LD_LIBRARY_PATH="$_VENV_DIR/lib:$_TORCH_LIB:${LD_LIBRARY_PATH:-}"
else
    echo "[launch_xpu] Warning: torch/lib not found under $_VENV_DIR; AIMDO VRAM mode may report zero XPU devices." >&2
fi
unset _VENV_DIR _TORCH_LIB

# --- Omni runtime policy (defaults; override via environment) ---
export OMNI_IMAGE_XPU_TARGET="${OMNI_IMAGE_XPU_TARGET:-bmg}"
export OMNIXPU_ENABLE="${OMNIXPU_ENABLE:-1}"
export OMNIXPU_PROVIDER_BOOTSTRAP="${OMNIXPU_PROVIDER_BOOTSTRAP:-auto}"
export SOL_ATTN_XPU_EXPERIMENTAL="${SOL_ATTN_XPU_EXPERIMENTAL:-1}"
export OMNIXPU_INTERPOLATE_FIX="${OMNIXPU_INTERPOLATE_FIX:-0}"
export OMNI_COMFYUI_RESERVE_VRAM_GB="${OMNI_COMFYUI_RESERVE_VRAM_GB:-4}"
export GIT_DISCOVERY_ACROSS_FILESYSTEM="${GIT_DISCOVERY_ACROSS_FILESYSTEM:-1}"

# --- VRAM-mode policy: VRAM by default, opt out with XPU_VRAM_MODE=direct ---
# --enable-dynamic-vram hands the allocator to AIMDO VBAR. Explicit user flags
# are respected and never duplicated.
XPU_VRAM_MODE="${XPU_VRAM_MODE:-vram}"
_HAVE_DYNAMIC_VRAM=0
_HAVE_RESERVE_VRAM=0
for _arg in "$@"; do
    if [ "$_arg" = "--enable-dynamic-vram" ]; then
        _HAVE_DYNAMIC_VRAM=1
    fi
    if [ "$_arg" = "--reserve-vram" ]; then
        _HAVE_RESERVE_VRAM=1
    fi
done
_VRAM_MODE=0
if [ "$_HAVE_DYNAMIC_VRAM" -eq 1 ]; then
    _VRAM_MODE=1
elif [ "$XPU_VRAM_MODE" = "vram" ]; then
    _VRAM_MODE=1
fi

# --- Driver stability: prevent "No device available" crashes ---
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
export ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE
unset ONEAPI_DEVICE_SELECTOR

# --- Level Zero: allow larger single allocations on Arc GPUs ---
export UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1
export UR_L0_USE_RELAXED_ALLOCATION_LIMITS=1

# --- Inductor: persistent compile cache across restarts ---
export TORCHINDUCTOR_FX_GRAPH_CACHE=1
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${HOME}/.cache/torch_compile}"

# --- torch.compile dynamic shapes: defaults to true to avoid recompiles on shape changes ---
export COMFY_TORCH_COMPILE_DYNAMIC=1

# --- PyTorch XPU allocator tuning (direct mode only) ---
# In VRAM mode the AIMDO UR hook owns allocation and refuses
# PYTORCH_ALLOC_CONF=expandable_segments, so leave it unset there and let
# AIMDO + --reserve-vram manage headroom instead.
if [ "$_VRAM_MODE" -eq 0 ]; then
    export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True,garbage_collection_threshold:0.85}"
fi

# --- Telemetry & warnings noise reduction ---
export OPENVINO_TELEMETRY=0
export UV_LINK_MODE=copy
export PYTHONWARNINGS="ignore"

# --- Early bootstrap & companion node support ---
# NOTE: COMFY_ROOT cd above breaks relative BASH_SOURCE paths, so anchor them
# to the invocation directory captured before cd.
SCRIPT_SOURCE="${BASH_SOURCE[0]}"
if [[ "$SCRIPT_SOURCE" != /* ]]; then
    SCRIPT_SOURCE="${_INVOCATION_DIR:-$PWD}/$SCRIPT_SOURCE"
fi
while [ -h "$SCRIPT_SOURCE" ]; do
    SCRIPT_DIR="$(cd -P "$(dirname "$SCRIPT_SOURCE")" && pwd)"
    SCRIPT_SOURCE="$(readlink "$SCRIPT_SOURCE")"
    [[ $SCRIPT_SOURCE != /* ]] && SCRIPT_SOURCE="$SCRIPT_DIR/$SCRIPT_SOURCE"
done
BOOTSTRAP_DIR="$(cd -P "$(dirname "$SCRIPT_SOURCE")/bootstrap" && pwd)"

export PYTHONPATH="$BOOTSTRAP_DIR${PYTHONPATH:+:$PYTHONPATH}"


# --- VRAM guard: cap the torch XPU allocator below physical VRAM ---
# Hybrid policy: in VRAM mode AIMDO VBAR owns the allocator (fraction cap
# would only fight it), so our guard stays dormant — see prestartup_script.py.
# In direct mode the cap raises clean OOMs instead of Level Zero hard-locks.
export XPU_VRAM_FRACTION="${XPU_VRAM_FRACTION:-0.88}"

if [ "$_VRAM_MODE" -eq 1 ]; then
    _reserve="${OMNI_COMFYUI_RESERVE_VRAM_GB:-4}"
    if [[ ! "$_reserve" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
        echo "[launch_xpu] Error: OMNI_COMFYUI_RESERVE_VRAM_GB must be a nonnegative number (got '$_reserve')." >&2
        exit 2
    fi
    echo "[launch_xpu] VRAM mode: AIMDO DynamicVRAM owns allocation (reserve ${_reserve} GiB); XPU_VRAM_FRACTION guard dormant."
    # Prepend missing VRAM flags (explicit user flags are never duplicated).
    if [ "$_HAVE_DYNAMIC_VRAM" -eq 0 ]; then
        set -- --enable-dynamic-vram "$@"
    fi
    if [ "$_HAVE_RESERVE_VRAM" -eq 0 ]; then
        set -- --reserve-vram "$_reserve" "$@"
    fi
else
    echo "[launch_xpu] Direct mode (XPU_VRAM_MODE=direct): native allocator capped at ${XPU_VRAM_FRACTION}."
fi
unset _VRAM_MODE _arg _HAVE_DYNAMIC_VRAM _HAVE_RESERVE_VRAM _reserve

exec python main.py "$@"
