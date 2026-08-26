#!/bin/bash
set -e

echo "======================================================="
echo "   ComfyUI Intel Arc & XPU Suite - Setup & Restore     "
echo "======================================================="

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

# 1. Install Suite dependencies
echo "[+] Installing toolkit Python dependencies..."
pip install -r "$SUITE_DIR/requirements.txt"

# 2. Symlink / copy launch script to ComfyUI root if applicable
if [ -n "$COMFY_ROOT" ]; then
    if [ ! -f "$COMFY_ROOT/launch_xpu.sh" ]; then
        echo "[+] Creating launch_xpu.sh in ComfyUI root..."
        ln -sf "$SUITE_DIR/scripts/launch_xpu.sh" "$COMFY_ROOT/launch_xpu.sh"
        chmod +x "$COMFY_ROOT/launch_xpu.sh"
    else
        echo "[*] launch_xpu.sh already exists in ComfyUI root."
    fi
fi

# 3. Environment verification
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
