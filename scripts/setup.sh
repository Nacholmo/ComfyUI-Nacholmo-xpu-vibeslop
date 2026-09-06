#!/bin/bash
set -uo pipefail

echo "======================================================="
echo "   ComfyUI Nacholmo XPU Vibeslop - Setup & Restore     "
echo "======================================================="

# Omni-aligned torch nightly pin.
# The local omni_xpu_kernel + provider wheels are hash-pinned to an EXACT
# torch build (provider manifest runtime.torch_version). If nightly moves
# past this pin, OmniXPU rejects both providers (see boot log "runtime
# provider rejected ... does not match provider") and Kitchen-XPU/AIMDO go
# silent — attention/norm adapters still apply, but no XPU dispatch or VBAR.
# After any torch upgrade, re-run with --with-omni to verify the match.
TORCH_PIN="2.15.0.dev20260830"
TORCHVISION_PIN="0.30.0.dev20260831"
TORCH_INDEX="https://download.pytorch.org/whl/nightly/xpu"
# Local wheel source (torch215 bmg builds). Falls back to $HOME/llm-scaler.
WHEELS_DIR="${WHEELS_DIR:-/home/sundae/llm-scaler/wheels}"

# Omni XPU custom-node pins (intel/llm-scaler omni 0.2.0-b2 image contract).
# nunchaku needs --ignore-requires-python on Python 3.14 (declares <3.14).
OMNI_NODES_PATCH="/home/sundae/Drives/Fenix/Comfy-omni/llm-scaler/omni/patches/comfyui_controlnet_aux_depth_anything_v2_xpu.patch"

usage() {
    echo "Usage: setup.sh [--with-aimdo] [--with-vhs] [--with-minimax-extend] [--all] [--fresh-venv] [--with-omni] [--with-omni-nodes] [--help]"
    echo "  --with-aimdo           Clone ComfyUI-AIMDO-XPU companion"
    echo "  --with-vhs             Clone comfyui-videohelpersuite + deps"
    echo "  --with-minimax-extend  Clone ComfyUI-MiniMax-H3-Extend companion"
    echo "  --all                  All of the above"
    echo "  --fresh-venv           Snapshot + rename venv, rebuild from torch pin + local Omni wheels"
    echo "  --with-omni            Verify OmniXPU stack (kernel probe, Kitchen XPU backend, AIMDO)"
    echo "  --with-omni-nodes      Clone Omni XPU nodes at pinned commits (nunchaku/SolAttn/easy-use/CacheDiT/controlnet_aux+XPU patch)"
}

# Parse arguments
WITH_AIMDO=0
WITH_VHS=0
WITH_MINIMAX_EXTEND=0
FRESH_VENV=0
WITH_OMNI=0
WITH_OMNI_NODES=0
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
        --fresh-venv)
            FRESH_VENV=1
            ;;
        --with-omni)
            WITH_OMNI=1
            ;;
        --with-omni-nodes)
            WITH_OMNI_NODES=1
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

# Rolling tip: seed versions from the last verified roll when present,
# so fresh installs reproduce the floating tip instead of stale built-ins.
# Values are charset-validated before use; built-ins remain the fallback.
NUNCHAKU_COMMIT="cc0f6236b6c329178ad4ef58452a874e774c7b8e"
SOLATTN_COMMIT="5f1c4aac3ca32a00b0b4c15ddbb7cb53fa43344d"
EASYUSE_COMMIT="b5e31ef12ad9d0b187b545c2707735cc7d581c52"
CACHEDIT_COMMIT="1d92bbd86ec59aa6223fe2368849b7413a1acb93"
CTRLAUX_COMMIT="e8b689a513c3e6b63edc44066560ca5919c0576e"
if [ -f "$SUITE_DIR/manifests/last-good.json" ]; then
    while IFS='=' read -r _k _v; do
        case "$_k" in
            TORCH_PIN|TORCHVISION_PIN|NUNCHAKU_COMMIT|SOLATTN_COMMIT|EASYUSE_COMMIT|CACHEDIT_COMMIT|CTRLAUX_COMMIT)
                if [[ "$_v" =~ ^[A-Za-z0-9.+_:/-]+$ ]]; then
                    printf -v "$_k" '%s' "$_v"
                else
                    echo "[!] Warning: ignoring suspicious last-good value for $_k." >&2
                fi
                ;;
        esac
    done < <(python3 -c "
import json
try:
    s = json.load(open('$SUITE_DIR/manifests/last-good.json'))
except Exception:
    raise SystemExit
nodes = s.get('nodes', {})
print('TORCH_PIN=' + str(s.get('torch', '')))
print('TORCHVISION_PIN=' + str(s.get('torchvision', '')))
print('NUNCHAKU_COMMIT=' + str(nodes.get('ComfyUI-nunchaku-XPU', '')))
print('SOLATTN_COMMIT=' + str(nodes.get('ComfyUI-SolAttn', '')))
print('EASYUSE_COMMIT=' + str(nodes.get('comfyui-easy-use', '')))
print('CACHEDIT_COMMIT=' + str(nodes.get('ComfyUI-CacheDiT', '')))
print('CTRLAUX_COMMIT=' + str(nodes.get('comfyui_controlnet_aux', '')))
" 2>/dev/null)
    unset _k _v
    echo "[+] Seeding versions from manifests/last-good.json (torch $TORCH_PIN)."
fi

if [ -z "$COMFY_ROOT" ]; then
    echo "[!] Notice: ComfyUI root not detected in parent directory."
    echo "    Running standalone dependency setup in current environment."
else
    echo "[+] Detected ComfyUI installation at: $COMFY_ROOT"
    cd "$COMFY_ROOT"
fi

# Activate virtual environment if present
if [ "$FRESH_VENV" -eq 1 ]; then
    echo "[+] Rebuilding fresh venv (torch pin $TORCH_PIN)..."
    STAMP="$(date +%Y%m%d)"
    if [ -d "venv" ]; then
        venv/bin/pip freeze > "$SUITE_DIR/manifests/aurora-venv-freeze-$STAMP.txt" 2>/dev/null || true
        echo "[+] Snapshot saved to manifests/aurora-venv-freeze-$STAMP.txt"
        mv venv "venv.bak-$STAMP"
        echo "[+] Old venv renamed to venv.bak-$STAMP (delete after validation)"
    fi
    /usr/bin/python3.14 -m venv venv
    # shellcheck disable=SC1091
    source venv/bin/activate
    echo "[+] Installing torch nightly pin + XPU deps..."
    pip install --pre --upgrade torch torchaudio torchvision triton-xpu --extra-index-url "$TORCH_INDEX"
    pip install "torch==$TORCH_PIN" "torchvision==$TORCHVISION_PIN" --extra-index-url "$TORCH_INDEX"
    if [ -n "${COMFY_ROOT:-}" ] && [ -f "$COMFY_ROOT/requirements.txt" ]; then
        echo "[+] Installing ComfyUI requirements..."
        pip install -r "$COMFY_ROOT/requirements.txt" || echo "[!] Warning: ComfyUI requirements failed." >&2
    fi
    if [ -f "$SUITE_DIR/manifests/companion-pins.txt" ]; then
        echo "[+] Installing companion custom-node dependency pins..."
        pip install -r "$SUITE_DIR/manifests/companion-pins.txt" || echo "[!] Warning: companion pins failed." >&2
    fi
    # Local wheels LAST so vendored provider files win over PyPI copies.
    if [ -d "$WHEELS_DIR" ]; then
        echo "[+] Installing local Omni wheels from $WHEELS_DIR ..."
        pip install --no-deps \
            "$WHEELS_DIR/kitchen-source"/comfy_kitchen-*.whl \
            "$WHEELS_DIR/kitchen-provider"/comfy_kitchen_xpu_runtime-*.whl \
            "$WHEELS_DIR/aimdo-source"/comfy_aimdo-*.whl \
            "$WHEELS_DIR/aimdo-provider"/comfy_aimdo_xpu_runtime-*.whl \
            "$WHEELS_DIR"/omni_xpu_kernel-*torch215*.whl
        pip install "onednn==2026.0.0"
    else
        echo "[!] Warning: WHEELS_DIR not found ($WHEELS_DIR); skipping Omni wheels." >&2
    fi
elif [ -d "venv" ]; then
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

    # 4b. Omni XPU nodes at pinned commits (+ DepthAnythingV2 XPU patch).
    if [ "$WITH_OMNI_NODES" -eq 1 ]; then
        _omni_clone() { # $1=repo $2=dir $3=commit
            if [ -d "$COMFY_ROOT/custom_nodes/$2" ]; then
                echo "[*] $2 already present; leaving untouched."
                return 0
            fi
            echo "[+] Cloning $2 @ $3 ..."
            git clone --filter=blob:none --no-checkout "$1" "$COMFY_ROOT/custom_nodes/$2" \
                && git -C "$COMFY_ROOT/custom_nodes/$2" fetch --depth 1 origin "$3" \
                && git -C "$COMFY_ROOT/custom_nodes/$2" checkout --detach FETCH_HEAD
        }
        _omni_clone https://github.com/xiangyuT/ComfyUI-nunchaku-XPU.git ComfyUI-nunchaku-XPU "$NUNCHAKU_COMMIT"
        _omni_clone https://github.com/xiangyuT/ComfyUI-SolAttn_xpu.git ComfyUI-SolAttn "$SOLATTN_COMMIT"
        _omni_clone https://github.com/yolain/ComfyUI-Easy-Use.git comfyui-easy-use "$EASYUSE_COMMIT"
        _omni_clone https://github.com/Jasonzzt/ComfyUI-CacheDiT.git ComfyUI-CacheDiT "$CACHEDIT_COMMIT"
        _omni_clone https://github.com/Fannovel16/comfyui_controlnet_aux.git comfyui_controlnet_aux "$CTRLAUX_COMMIT"
        if [ -f "$OMNI_NODES_PATCH" ] && [ -d "$COMFY_ROOT/custom_nodes/comfyui_controlnet_aux" ]; then
            if git -C "$COMFY_ROOT/custom_nodes/comfyui_controlnet_aux" status --short | grep -q "depth_anything_v2/dpt.py"; then
                echo "[*] controlnet_aux XPU patch already applied."
            elif git -C "$COMFY_ROOT/custom_nodes/comfyui_controlnet_aux" apply "$OMNI_NODES_PATCH"; then
                echo "[+] Applied DepthAnythingV2 XPU patch to controlnet_aux."
            else
                echo "[!] Warning: controlnet_aux XPU patch failed to apply." >&2
            fi
        fi
        for _nd in ComfyUI-nunchaku-XPU comfyui-easy-use ComfyUI-CacheDiT comfyui_controlnet_aux; do
            if [ -f "$COMFY_ROOT/custom_nodes/$_nd/requirements.txt" ]; then
                pip install -r "$COMFY_ROOT/custom_nodes/$_nd/requirements.txt" \
                    || echo "[!] Warning: $_nd requirements failed." >&2
            fi
        done
        # nunchaku_torch runtime: same checkout as one distribution (XPU).
        # --ignore-requires-python: declares <3.14, verified working on 3.14.
        if [ -d "$COMFY_ROOT/custom_nodes/ComfyUI-nunchaku-XPU" ]; then
            pip install --no-deps --no-build-isolation --ignore-requires-python \
                "$COMFY_ROOT/custom_nodes/ComfyUI-nunchaku-XPU" \
                || echo "[!] Warning: nunchaku dist install failed." >&2
        fi
        unset -f _omni_clone
        unset _nd
    fi
fi

# 5. OmniXPU stack verification
if [ "$WITH_OMNI" -eq 1 ]; then
    echo ""
    echo "--- OmniXPU Check ---"
    python - <<'PYEOF' || echo "[!] OmniXPU check failed (see errors above)."
import importlib.util
import sys

sys.argv = ['main.py']  # direct mode: Kitchen expected active, AIMDO skipped

spec = importlib.util.spec_from_file_location(
    '_omnixpu_bootstrap', 'custom_nodes/ComfyUI-OmniXPU/runtime_bootstrap.py')
mod = importlib.util.module_from_spec(spec)
sys.modules['_omnixpu_bootstrap'] = mod
spec.loader.exec_module(mod)
state = mod.bootstrap()
print('provider bootstrap:', state['status'], '(mode ' + str(state['mode']) + ')')
for pid, ps in state['providers'].items():
    print(' ', pid, '->', ps['status'], ps['reason'])
kitchen = state['providers'].get('comfy_kitchen.xpu', {})
if kitchen.get('status') != 'active':
    print('[!] Kitchen XPU provider not active: XPU dispatch disabled.')
    print('    Likely cause: torch build drifted past the wheel pin')
    print('    (see TORCH_PIN at top of setup.sh); rebuild with --fresh-venv.')

import comfy_kitchen as ck
xpu = ck.list_backends().get('xpu') or {}
print('kitchen xpu backend available:', xpu.get('available'),
      '| caps:', len(xpu.get('capabilities', [])))

pspec = importlib.util.spec_from_file_location(
    '_omnixpu_probe', 'custom_nodes/ComfyUI-OmniXPU/probe.py')
probe = importlib.util.module_from_spec(pspec)
sys.modules['_omnixpu_probe'] = probe
pspec.loader.exec_module(probe)
probe.probe()
print('kernel summary:', probe.summary())
PYEOF
fi
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
