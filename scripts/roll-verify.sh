#!/bin/bash
# Rolling-release verification gate for Aurora ComfyUI (Intel Arc XPU).
# Read-only against the live install except: starts ComfyUI twice on an
# ephemeral port (VRAM default + XPU_VRAM_MODE=direct) and kills both.
# Exit 0 = releasable, non-zero = do NOT roll / roll back.
#
# Usage: roll-verify.sh [--port 8399] [--boot-timeout 420]
set -uo pipefail

PORT="${1:-8399}"
if [ "${1:-}" = "--port" ]; then PORT="${2:-8399}"; fi
BOOT_TIMEOUT=420

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
    echo "[verify] FAIL: ComfyUI root not found." >&2
    exit 1
fi
cd "$COMFY_ROOT" || exit 1
if [ -d "venv" ]; then
    # shellcheck disable=SC1091
    source venv/bin/activate
fi

FAIL=0
fail() { echo "[verify] FAIL: $1" >&2; FAIL=1; }
pass() { echo "[verify] ok: $1"; }

# --- 1. Provider bootstrap + kernel probe (no torch pre-import issues: fresh process) ---
echo "[verify] stage 1: provider bootstrap + kernel probe"
python - <<'PYEOF' > /tmp/opencode-roll-verify-stage1.log 2>&1 || true
import importlib.util
import sys

sys.argv = ['main.py']  # direct mode: Kitchen expected active, AIMDO skipped

spec = importlib.util.spec_from_file_location(
    '_rv_bootstrap', 'custom_nodes/ComfyUI-OmniXPU/runtime_bootstrap.py')
mod = importlib.util.module_from_spec(spec)
sys.modules['_rv_bootstrap'] = mod
spec.loader.exec_module(mod)
state = mod.bootstrap()
print('BOOTSTRAP_STATUS=' + str(state['status']))
for pid, ps in state['providers'].items():
    print('PROVIDER ' + pid + ' ' + ps['status'] + ' ' + ps['reason'])

import comfy_kitchen as ck
xpu = ck.list_backends().get('xpu') or {}
print('KITCHEN_XPU_AVAILABLE=' + str(bool(xpu.get('available'))))
print('KITCHEN_XPU_CAPS=' + str(len(xpu.get('capabilities', []))))

pspec = importlib.util.spec_from_file_location(
    '_rv_probe', 'custom_nodes/ComfyUI-OmniXPU/probe.py')
probe = importlib.util.module_from_spec(pspec)
sys.modules['_rv_probe'] = probe
pspec.loader.exec_module(probe)
probe.probe()
print('KERNEL_SUMMARY=' + str(probe.summary()))
PYEOF
grep -a -q "PROVIDER comfy_kitchen.xpu active" /tmp/opencode-roll-verify-stage1.log \
    && pass "Kitchen XPU provider active" \
    || fail "Kitchen XPU provider not active (see /tmp/opencode-roll-verify-stage1.log)"
grep -a -q "KITCHEN_XPU_AVAILABLE=True" /tmp/opencode-roll-verify-stage1.log \
    && pass "Kitchen XPU backend available" \
    || fail "Kitchen XPU backend unavailable"
for _mod in "'sdp': True" "'norm': True" "'rotary': True" "'linear_fp8': True" "'int8': True" "'layout': True"; do
    grep -a -q "$_mod" /tmp/opencode-roll-verify-stage1.log \
        || fail "kernel submodule missing: $_mod"
done
grep -a "KERNEL_SUMMARY" /tmp/opencode-roll-verify-stage1.log | head -n 1

# --- 2+3. Boot-to-GUI in both modes ---
_boot_one() { # $1 = vram|direct
    _label="$1"; shift
    _log="/tmp/opencode-roll-verify-boot-${_label}.log"
    echo "[verify] stage 2: boot-to-GUI (${_label}) on port $PORT"
    rm -f "$_log"
    if [ "$_label" = "direct" ]; then
        env XPU_VRAM_MODE=direct nohup bash custom_nodes/ComfyUI-Nacholmo-xpu-vibeslop/scripts/launch_xpu.sh \
            --listen 127.0.0.1 --port "$PORT" > "$_log" 2>&1 &
    else
        nohup bash custom_nodes/ComfyUI-Nacholmo-xpu-vibeslop/scripts/launch_xpu.sh \
            --listen 127.0.0.1 --port "$PORT" > "$_log" 2>&1 &
    fi
    _pid=$!
    _waited=0
    while [ "$_waited" -lt "$BOOT_TIMEOUT" ]; do
        sleep 15
        _waited=$((_waited + 15))
        if ! kill -0 "$_pid" 2>/dev/null; then
            break  # exited early (crash or --help-like fast exit)
        fi
        if grep -a -q "To see the GUI" "$_log" 2>/dev/null; then
            break
        fi
    done
    if grep -a -q "To see the GUI" "$_log" 2>/dev/null; then
        pass "boot-to-GUI (${_label})"
    else
        fail "boot-to-GUI (${_label}) missing (log: $_log)"
    fi
    if grep -a -q "Traceback" "$_log" 2>/dev/null; then
        fail "tracebacks in ${_label} boot log"
    else
        pass "no tracebacks (${_label})"
    fi
    if grep -a -q "provider rejected" "$_log" 2>/dev/null; then
        fail "provider rejection in ${_label} boot log"
    else
        pass "no provider rejections (${_label})"
    fi
    if [ "$_label" = "vram" ]; then
        grep -a -q "inited for GPU" "$_log" 2>/dev/null \
            && pass "AIMDO XPU init (${_label})" \
            || fail "AIMDO XPU init line missing (${_label})"
    else
        grep -a -q "xpu-vram-guard.*capped" "$_log" 2>/dev/null \
            && pass "fraction-cap guard (${_label})" \
            || fail "fraction-cap guard line missing (${_label})"
    fi
    # --- 4. API spot-checks (same booted server) ---
    for _node in ArcSuperResolution MiniMaxH3TurboLoRA OmniXPUStatus VideoCombineSync SolAttnPatch ApplySolAttn SolAttnPatchMiniMax; do
        if curl -s "http://127.0.0.1:${PORT}/object_info/${_node}" 2>/dev/null | grep -a -q "\"${_node}\""; then
            pass "API serves ${_node} (${_label})"
        else
            fail "API missing ${_node} (${_label})"
        fi
    done
    kill "$_pid" 2>/dev/null || true
    sleep 3
    unset _label _log _pid _waited _node
}

_boot_one vram
_boot_one direct

if [ "$FAIL" -ne 0 ]; then
    echo "[verify] RESULT: RED — do not release." >&2
    exit 1
fi
echo "[verify] RESULT: GREEN — releasable."
