#!/bin/bash
set -uo pipefail

echo "======================================================="
echo "   ComfyUI Nacholmo XPU Vibeslop - Setup & Restore     "
echo "======================================================="

usage() {
    echo "Usage: setup.sh [--with-aimdo] [--with-vhs] [--with-minimax-extend] [--all] [--help]"
    echo "  --with-aimdo           Clone ComfyUI-AIMDO-XPU companion"
    echo "  --with-vhs             Clone comfyui-videohelpersuite + deps"
    echo "  --with-minimax-extend  Clone ComfyUI-MiniMax-H3-Extend companion"
    echo "  --all                  All of the above"
}

# Parse arguments
WITH_AIMDO=0
WITH_VHS=0
WITH_MINIMAX_EXTEND=0
for arg in "$@"; do
    case $arg in
        --with-aimdo)
            WITH_AIMDO=1
            ;;
        --with-vhs)
            WITH_VHS=1
            ;;
        --with-minimax-extend)
            WITH_MINIMAX_EXTEND=1
            ;;
        --all)
            WITH_AIMDO=1
            WITH_VHS=1
            WITH_MINIMAX_EXTEND=1
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "[!] Unknown argument: $arg" >&2
            usage >&2
            exit 1
            ;;
    esac
done

# Find ComfyUI root directory
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

SUITE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ -z "$COMFY_ROOT" ]; then
    echo "[!] Notice: ComfyUI root not detected in parent directory."
    echo "    Running standalone dependency setup in current environment."
else
    echo "[+] Detected ComfyUI installation at: $COMFY_ROOT"
    cd "$COMFY_ROOT"
fi

# Activate virtual environment if present
if [ -d "venv" ]; then
    echo "[+] Activating venv/..."
    source venv/bin/activate
elif [ -d ".venv" ]; then
    echo "[+] Activating .venv/..."
    source .venv/bin/activate
fi

# 1. Install Suite dependencies (non-fatal: report but continue to symlink/theme)
echo "[+] Installing toolkit Python dependencies..."
if ! pip install -r "$SUITE_DIR/requirements.txt"; then
    echo "[!] Warning: suite requirements failed to install; continuing with symlink/theme setup." >&2
fi

# 2. Symlink / copy launch script to ComfyUI root if applicable
if [ -n "$COMFY_ROOT" ]; then
    _link_target="$SUITE_DIR/scripts/launch_xpu.sh"
    if [ -L "$COMFY_ROOT/launch_xpu.sh" ]; then
        _current="$(readlink "$COMFY_ROOT/launch_xpu.sh")"
        if [ "$_current" != "$_link_target" ]; then
            echo "[+] Updating stale launch_xpu.sh symlink ($_current -> $_link_target)..."
            ln -sf "$_link_target" "$COMFY_ROOT/launch_xpu.sh"
            chmod +x "$COMFY_ROOT/launch_xpu.sh"
        else
            echo "[*] launch_xpu.sh symlink already up to date."
        fi
    elif [ ! -f "$COMFY_ROOT/launch_xpu.sh" ]; then
        echo "[+] Creating launch_xpu.sh in ComfyUI root..."
        ln -sf "$_link_target" "$COMFY_ROOT/launch_xpu.sh"
        chmod +x "$COMFY_ROOT/launch_xpu.sh"
    else
        echo "[*] launch_xpu.sh already exists as a regular file; leaving untouched."
    fi

    # 3. Install DarkComfyX Theme
    if [ -f "$SUITE_DIR/tools/install_theme.py" ]; then
        echo "[+] Installing and configuring DarkComfyX Theme..."
        if ! python "$SUITE_DIR/tools/install_theme.py"; then
            echo "[!] Warning: theme installer failed; continuing." >&2
        fi
    fi

    # 4. Optional companion nodes
    if [ "$WITH_AIMDO" -eq 1 ] && [ ! -d "$COMFY_ROOT/custom_nodes/ComfyUI-AIMDO-XPU" ]; then
        echo "[+] Cloning ComfyUI-AIMDO-XPU companion repository..."
        if ! git clone https://github.com/allanmeng/ComfyUI-AIMDO-XPU "$COMFY_ROOT/custom_nodes/ComfyUI-AIMDO-XPU"; then
            echo "[!] Warning: failed to clone ComfyUI-AIMDO-XPU." >&2
        fi
    fi

    if [ "$WITH_VHS" -eq 1 ] && [ ! -d "$COMFY_ROOT/custom_nodes/comfyui-videohelpersuite" ]; then
        echo "[+] Cloning comfyui-videohelpersuite companion repository..."
        if git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite "$COMFY_ROOT/custom_nodes/comfyui-videohelpersuite"; then
            if [ -f "$COMFY_ROOT/custom_nodes/comfyui-videohelpersuite/requirements.txt" ]; then
                if ! pip install -r "$COMFY_ROOT/custom_nodes/comfyui-videohelpersuite/requirements.txt"; then
                    echo "[!] Warning: VHS requirements failed to install." >&2
                fi
            fi
        else
            echo "[!] Warning: failed to clone comfyui-videohelpersuite." >&2
        fi
    fi

    if [ "$WITH_MINIMAX_EXTEND" -eq 1 ] && [ ! -d "$COMFY_ROOT/custom_nodes/ComfyUI-MiniMax-H3-Extend" ]; then
        echo "[+] Cloning ComfyUI-MiniMax-H3-Extend companion repository..."
        if ! git clone https://github.com/kat3ri/ComfyUI-MiniMax-H3-Extend "$COMFY_ROOT/custom_nodes/ComfyUI-MiniMax-H3-Extend"; then
            echo "[!] Warning: failed to clone ComfyUI-MiniMax-H3-Extend." >&2
        fi
    fi
fi

# 4. Environment verification
echo ""
echo "--- Environment Check ---"
python -c "
import torch
print('PyTorch Version:', torch.__version__)
if hasattr(torch, 'xpu') and torch.xpu.is_available():
    print('PyTorch XPU:', 'Available (Device:', torch.xpu.get_device_name(0), ')')
else:
    print('PyTorch XPU: Not detected (ensure intel-compute-runtime and torch-xpu are installed)')

try:
    import torchaudio
    print('TorchAudio:', 'Available (Version:', torchaudio.__version__, ')')
except Exception as e:
    print('TorchAudio:', 'Error loading:', e)

try:
    import openvino as ov
    core = ov.Core()
    devs = core.get_available_devices()
    print('OpenVINO Devices:', devs)
    if 'GPU' in devs:
        print('OpenVINO GPU Acceleration: Ready!')
    else:
        print('OpenVINO GPU: CPU only detected (install intel-opencl-icd for Arc GPU upscaling)')
except ImportError:
    print('OpenVINO: Not installed')
"

echo ""
echo "[✓] Setup complete! You can now start ComfyUI using ./launch_xpu.sh"
