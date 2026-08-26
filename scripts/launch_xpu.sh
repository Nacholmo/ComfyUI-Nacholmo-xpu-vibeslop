#!/bin/bash
# Launch ComfyUI on Intel Arc (XPU) with optimized environment settings.

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

cd "$COMFY_ROOT" || exit 1

# Activate virtual environment if present
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
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

# --- PyTorch XPU allocator tuning ---
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True,garbage_collection_threshold:0.85}"

# --- Telemetry & warnings noise reduction ---
export OPENVINO_TELEMETRY=0
export UV_LINK_MODE=copy
export PYTHONWARNINGS="ignore"

# --- Early bootstrap & companion node support ---
SCRIPT_SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SCRIPT_SOURCE" ]; do
    SCRIPT_DIR="$(cd -P "$(dirname "$SCRIPT_SOURCE")" && pwd)"
    SCRIPT_SOURCE="$(readlink "$SCRIPT_SOURCE")"
    [[ $SCRIPT_SOURCE != /* ]] && SCRIPT_SOURCE="$SCRIPT_DIR/$SCRIPT_SOURCE"
done
BOOTSTRAP_DIR="$(cd -P "$(dirname "$SCRIPT_SOURCE")/bootstrap" && pwd)"

export PYTHONPATH="$BOOTSTRAP_DIR:$PYTHONPATH"
if [ -d "$PWD/custom_nodes/ComfyUI-AIMDO-XPU" ]; then
    export PYTHONPATH="$PWD/custom_nodes/ComfyUI-AIMDO-XPU:$PYTHONPATH"
fi


# --- VRAM guard: cap the torch XPU allocator below physical VRAM ---
export XPU_VRAM_FRACTION="${XPU_VRAM_FRACTION:-0.90}"

exec python main.py "$@"
