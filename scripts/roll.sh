#!/bin/bash
# Rolling-release runner for Aurora ComfyUI (Intel Arc XPU, fully floating).
# Floats torch nightly + providers + ComfyUI core + custom nodes, reinstalls
# deps unpinned, runs the verify gate, and rolls back automatically on RED.
#
# See docs/rolling-release.md for the model, constraints, and runbook.
#
# Usage:
#   roll.sh [--dry-run] [--yes] [--skip-torch] [--skip-core] [--skip-nodes]
# Safety: snapshot + instant venv rename before any mutation; --dry-run
# mutates nothing (no network, no installs, no checkouts).
set -uo pipefail

DRY_RUN=0
ASSUME_YES=0
SKIP_TORCH=0
SKIP_CORE=0
SKIP_NODES=0
for arg in "$@"; do
    case $arg in
        --dry-run) DRY_RUN=1 ;;
        --yes) ASSUME_YES=1 ;;
        --skip-torch) SKIP_TORCH=1 ;;
        --skip-core) SKIP_CORE=1 ;;
        --skip-nodes) SKIP_NODES=1 ;;
        --help|-h)
            sed -n '2,12p' "${BASH_SOURCE[0]}"
            exit 0 ;;
        *) echo "[roll] Unknown argument: $arg" >&2; exit 2 ;;
    esac
done

SUITE_DIR="$(cd -P "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMFY_ROOT=""
if [ -f "./main.py" ] && [ -f "./execution.py" ]; then
    COMFY_ROOT="$PWD"
else
    DIR="$SUITE_DIR"
    while [ "$DIR" != "/" ]; do
        if [ -f "$DIR/main.py" ] && [ -f "$DIR/execution.py" ]; then
            COMFY_ROOT="$DIR"
            break
        fi
        DIR="$(dirname "$DIR")"
    done
fi
if [ -z "$COMFY_ROOT" ]; then
    echo "[roll] Error: ComfyUI root not found." >&2
    exit 1
fi

STAMP="$(date +%Y%m%d-%H%M)"
SNAP_DIR="$SUITE_DIR/manifests/roll-snapshots/$STAMP"
HOLDS_FILE="$SUITE_DIR/manifests/roll-holds.conf"
PATCH_FILE="/home/sundae/Drives/Fenix/Comfy-omni/llm-scaler/omni/patches/comfyui_controlnet_aux_depth_anything_v2_xpu.patch"
TORCH_INDEX="https://download.pytorch.org/whl/nightly/xpu"
LLM_SCALER="/home/sundae/llm-scaler"
WHEELS_SRC="$LLM_SCALER/wheels"

# dir|expected-origin-remote (float guard: skip on remote mismatch)
FLOAT_NODES="
ComfyUI-nunchaku-XPU|https://github.com/xiangyuT/ComfyUI-nunchaku-XPU.git
ComfyUI-SolAttn|https://github.com/xiangyuT/ComfyUI-SolAttn_xpu.git
comfyui-easy-use|https://github.com/yolain/ComfyUI-Easy-Use.git
ComfyUI-CacheDiT|https://github.com/Jasonzzt/ComfyUI-CacheDiT.git
comfyui_controlnet_aux|https://github.com/Fannovel16/comfyui_controlnet_aux.git
ComfyUI-GGUF-XPU|https://github.com/analytics-zoo/ComfyUI-GGUF-XPU.git
comfyui-videohelpersuite|https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git
ComfyUI-MiniMax-H3-Extend|https://github.com/kat3ri/ComfyUI-MiniMax-H3-Extend.git
Comfyui_Minimax_h3_latent_Upscaler|https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler.git
ComfyUI-LTXVideo|https://github.com/Lightricks/ComfyUI-LTXVideo.git
comfyui-frame-interpolation|https://github.com/Fannovel16/comfyui-frame-interpolation.git
comfyui-krea2edit|https://github.com/lbouaraba/comfyui-krea2edit.git
ComfyUI-Flux2Klein-Enhancer|https://github.com/capitan01R/ComfyUI-Flux2Klein-Enhancer.git
comfyui-unload-model|https://github.com/Nacholmo/comfyui-unload-model.git
rgthree-comfy|https://github.com/rgthree/rgthree-comfy.git
RES4LYF|https://github.com/ClownsharkBatwing/RES4LYF.git
"

# Remotes vary in trailing '.git' — normalize before comparing.
_norm_remote() {
    _r="$1"
    _r="${_r%/}"
    _r="${_r%.git}"
    printf '%s' "$_r"
}

_hold_for() { # $1=dirname -> commit or empty
    [ -f "$HOLDS_FILE" ] || return 0
    grep -a -E "^${1}=" "$HOLDS_FILE" 2>/dev/null | tail -n 1 | cut -d= -f2-
}

if [ "$DRY_RUN" -eq 1 ]; then
    echo "[roll] DRY RUN — planned actions (nothing will change):"
    [ "$SKIP_TORCH" -eq 0 ] && echo "  - float torch nightly (unpinned) + rebuild provider wheels + -U comfy-kitchen/comfy-aimdo"
    [ "$SKIP_CORE" -eq 0 ] && echo "  - float ComfyUI core to origin HEAD $([ -n "$(_hold_for comfy-core)" ] && echo "(HELD at $(_hold_for comfy-core))")"
    if [ "$SKIP_NODES" -eq 0 ]; then
        echo "  - float custom nodes:"
        echo "$FLOAT_NODES" | while IFS='|' read -r _d _r; do
            [ -z "$_d" ] && continue
            _h="$(_hold_for "$_d")"
            if [ -d "$COMFY_ROOT/custom_nodes/$_d" ]; then
                if [ -e "$COMFY_ROOT/custom_nodes/$_d/.git" ]; then
                    echo "      $_d @ $(git -C "$COMFY_ROOT/custom_nodes/$_d" rev-parse --short HEAD 2>/dev/null || echo missing) -> origin HEAD $([ -n "$_h" ] && echo "(HELD at $_h)")"
                else
                    echo "      $_d (no-git checkout, will skip)"
                fi
            else
                echo "      $_d (not installed, skip)"
            fi
        done
    fi
    echo "  - reinstall root/suite/node requirements (unpinned) + nunchaku dist"
    echo "  - run scripts/roll-verify.sh; rollback on RED"
    echo "  - snapshot -> manifests/roll-snapshots/<date>/, record manifests/last-good.json"
    exit 0
fi

if [ "$ASSUME_YES" -eq 0 ]; then
    echo "[roll] This floats torch + ComfyUI core + $(echo "$FLOAT_NODES" | grep -c '|') custom nodes on the LIVE install."
    echo "[roll] Snapshot + instant rollback are automatic, but budget ~30 min."
    read -r -p "[roll] Proceed? [y/N] " _ans
    if [ "$_ans" != "y" ] && [ "$_ans" != "Y" ]; then
        echo "[roll] Aborted."
        exit 0
    fi
    unset _ans
fi

cd "$COMFY_ROOT" || exit 1
mkdir -p "$SNAP_DIR"

# --- 0. Snapshot (pre-mutation) ---
echo "[roll] snapshotting to $SNAP_DIR ..."
if [ -d "venv" ]; then
    venv/bin/pip freeze > "$SNAP_DIR/freeze.txt" 2>/dev/null || true
else
    echo "[roll] Error: no venv/ to roll." >&2
    exit 1
fi
git rev-parse HEAD > "$SNAP_DIR/comfy-core.txt" 2>/dev/null || echo unknown > "$SNAP_DIR/comfy-core.txt"
git branch --show-current > "$SNAP_DIR/comfy-core-branch.txt" 2>/dev/null || echo DETACHED > "$SNAP_DIR/comfy-core-branch.txt"
: > "$SNAP_DIR/nodes.txt"
for _d in "$COMFY_ROOT"/custom_nodes/*/; do
    _n="$(basename "$_d")"
    if [ -e "$_d/.git" ]; then
        _c="$(git -C "$_d" rev-parse HEAD 2>/dev/null || echo unknown)"
        _b="$(git -C "$_d" branch --show-current 2>/dev/null || true)"
        echo "$_n $_c ${_b:-DETACHED}" >> "$SNAP_DIR/nodes.txt"
    else
        echo "$_n (no-git)" >> "$SNAP_DIR/nodes.txt"
    fi
done
cp "$SUITE_DIR/manifests/last-good.json" "$SNAP_DIR/last-good.json" 2>/dev/null || true
unset _d _n

rollback() { # $1=reason
    echo "[roll] ROLLBACK: $1" >&2
    rm -rf venv
    mv "venv.pre-roll-$STAMP" venv
    while read -r _n _c _b; do
        # Never touch the suite repo itself: roll.sh modifies none of its
        # tracked files (only untracked snapshots/), and detaching it
        # strands the working checkout (observed on the first RED roll).
        if [ "$_n" = "ComfyUI-Nacholmo-xpu-vibeslop" ]; then
            continue
        fi
        if [ -e "custom_nodes/$_n/.git" ] && [ "$_c" != "(no-git)" ] && [ "$_c" != "unknown" ]; then
            git -C "custom_nodes/$_n" checkout --detach "$_c" 2>/dev/null || true
            # Reattach the recorded branch when it still points at the
            # snapshot commit (same-commit checkout: working tree untouched,
            # local mods like GGUF-XPU loader.py survive).
            if [ -n "${_b:-}" ] && [ "$_b" != "DETACHED" ] \
                && [ "$(git -C "custom_nodes/$_n" rev-parse "$_b" 2>/dev/null)" = "$_c" ]; then
                git -C "custom_nodes/$_n" checkout "$_b" 2>/dev/null || true
            fi
        fi
    done < "$SNAP_DIR/nodes.txt"
    _core="$(cat "$SNAP_DIR/comfy-core.txt")"
    _core_branch="$(cat "$SNAP_DIR/comfy-core-branch.txt" 2>/dev/null || echo DETACHED)"
    if [ "$_core" != "unknown" ]; then
        git checkout --detach "$_core" 2>/dev/null || true
        if [ "$_core_branch" != "DETACHED" ] && [ "$_core_branch" != "" ] \
            && [ "$(git rev-parse "$_core_branch" 2>/dev/null)" = "$_core" ]; then
            git checkout "$_core_branch" 2>/dev/null || true
        fi
    fi
    unset _n _c _b _core _core_branch
    echo "[roll] Restored venv + node commits from $SNAP_DIR" >&2
    exit 3
}

mv venv "venv.pre-roll-$STAMP"
# shellcheck disable=SC1091
/usr/bin/python3.14 -m venv venv && source venv/bin/activate

# --- 1. Torch nightly (floating, unpinned) ---
if [ "$SKIP_TORCH" -eq 0 ]; then
    echo "[roll] floating torch nightly ..."
    pip install --pre --upgrade torch torchaudio torchvision triton-xpu --extra-index-url "$TORCH_INDEX" \
        || rollback "torch nightly install failed"
    NEW_TORCH="$(python -c 'import torch; print(torch.__version__)')"
    echo "[roll] torch now: $NEW_TORCH"

    # --- 2. Provider re-stamp for the new torch ---
    echo "[roll] rebuilding provider wheels for $NEW_TORCH ..."
    _provdir="$(mktemp -d)"
    _stamp_ok=1
    for _prov in "kitchen|comfy-kitchen-xpu" "aimdo|comfy-aimdo-xpu"; do
        IFS='|' read -r _p _src <<< "$_prov"
        _srcwheel="$(ls "$WHEELS_SRC/${_p}"-source/*.whl 2>/dev/null | head -n 1)"
        _script="$LLM_SCALER/omni/${_src}/packaging/xpu_runtime_provider/build_wheel.py"
        _srcrev="$(git -C "$LLM_SCALER/omni/${_src}" rev-parse HEAD 2>/dev/null || echo unknown)"
        if [ -z "$_srcwheel" ] || [ ! -f "$_script" ] || [ "$_srcrev" = "unknown" ]; then
            echo "[roll] provider rebuild inputs missing for $_p (wheel/script/revision)" >&2
            _stamp_ok=0
            break
        fi
        python "$_script" --source-wheel "$_srcwheel" --output-dir "$_provdir" \
            --source-revision "$_srcrev" \
            --torch-version "$NEW_TORCH" --xpu-target bmg \
            || { _stamp_ok=0; break; }
        unset _p _src _srcwheel _script _srcrev
    done
    if [ "$_stamp_ok" -eq 0 ]; then
        rm -rf "$_provdir"
        rollback "provider wheel rebuild failed"
    fi
    pip install --no-deps "$_provdir"/*.whl || { rm -rf "$_provdir"; rollback "provider wheel install failed"; }
    rm -rf "$_provdir"
    unset _provdir _stamp_ok _prov

    # Kernel wheel reused as-is (no kernel rebuild in a roll — see docs).
    _kernel=( "$WHEELS_SRC"/omni_xpu_kernel-*torch215*.whl )
    pip install --no-deps "$_kernel" || rollback "kernel wheel install failed"
    pip install "onednn==2026.0.0" || rollback "onednn install failed"
    unset _kernel
else
    # --skip-torch keeps the snapshot torch stack, but the venv above is
    # still rebuilt fresh — so reinstall it explicitly. (Previously this
    # branch left a bare venv: no torch/providers/kernel -> both boots
    # died at import. Observed on the core-only roll.)
    echo "[roll] keeping snapshot torch stack ..."
    _snap_torch="$(grep -a '^torch==' "$SNAP_DIR/freeze.txt" | head -n 1)"
    _snap_tv="$(grep -a '^torchvision==' "$SNAP_DIR/freeze.txt" | head -n 1)"
    _snap_triton="$(grep -a '^triton-xpu==' "$SNAP_DIR/freeze.txt" | head -n 1)"
    if [ -z "$_snap_torch" ]; then
        rollback "snapshot has no torch pin"
    fi
    pip install "$_snap_torch" "$_snap_tv" "$_snap_triton" --extra-index-url "$TORCH_INDEX" \
        || rollback "snapshot torch stack install failed"
    NEW_TORCH="$(python -c 'import torch; print(torch.__version__)')"
    echo "[roll] torch now: $NEW_TORCH (snapshot)"
    pip install --no-deps \
        "$WHEELS_SRC/kitchen-source"/comfy_kitchen-*.whl \
        "$WHEELS_SRC/kitchen-provider"/comfy_kitchen_xpu_runtime-*.whl \
        "$WHEELS_SRC/aimdo-source"/comfy_aimdo-*.whl \
        "$WHEELS_SRC/aimdo-provider"/comfy_aimdo_xpu_runtime-*.whl \
        "$WHEELS_SRC"/omni_xpu_kernel-*torch215*.whl \
        || rollback "Omni wheel reinstall failed"
    pip install "onednn==2026.0.0" || rollback "onednn install failed"
    unset _snap_torch _snap_tv _snap_triton
fi

# --- Root packages (official kitchen/aimdo HELD by requirements.txt) ---
# Official comfy-kitchen/comfy-aimdo must move in lockstep with the provider
# SOURCES (provider manifest compatible_versions). Floating them alone
# (pip install -U) gets both providers rejected -> RED. They advance only
# via a manual provider-source rebuild (see docs/rolling-release.md §3),
# so here they are intentionally re-pinned by requirements.txt, never -U'd.
echo "[roll] installing ComfyUI requirements ..."
pip install -r requirements.txt || rollback "ComfyUI requirements failed"
pip install -r "$SUITE_DIR/requirements.txt" || rollback "suite requirements failed"
if [ -f "$SUITE_DIR/manifests/companion-pins.txt" ]; then
    # Pins file doubles as the companion package list; versions float on rolls.
    sed 's/==.*//' "$SUITE_DIR/manifests/companion-pins.txt" | grep -a -v '^$' | xargs pip install -U \
        || rollback "companion packages failed"
fi

# --- 4. ComfyUI core float ---
if [ "$SKIP_CORE" -eq 0 ]; then
    _core_hold="$(_hold_for comfy-core)"
    if [ -n "$_core_hold" ]; then
        echo "[roll] core HELD at $_core_hold"
        git checkout --detach "$_core_hold" || rollback "core hold checkout failed"
    else
        _branch="$(git remote show origin 2>/dev/null | sed -n 's/.*HEAD branch: //p')"
        git fetch --depth 50 origin "$_branch" || rollback "core fetch failed"
        git checkout --detach FETCH_HEAD || rollback "core checkout failed"
        echo "[roll] core now: $(git rev-parse --short HEAD)"
        unset _branch
    fi
    unset _core_hold
fi

# --- 5. Custom-node float ---
if [ "$SKIP_NODES" -eq 0 ]; then
    echo "$FLOAT_NODES" | while IFS='|' read -r _d _repo; do
        [ -z "$_d" ] && continue
        # controlnet_aux floats via the dedicated patch-safe block below.
        if [ "$_d" = "comfyui_controlnet_aux" ]; then
            continue
        fi
        if [ ! -d "$COMFY_ROOT/custom_nodes/$_d" ]; then
            echo "[roll] skip $_d (not installed)"
            continue
        fi
        if [ ! -d "$COMFY_ROOT/custom_nodes/$_d/.git" ] && [ ! -f "$COMFY_ROOT/custom_nodes/$_d/.git" ]; then
            echo "[roll] skip $_d (no-git checkout, cannot float safely)"
            continue
        fi
        _actual="$(git -C "$COMFY_ROOT/custom_nodes/$_d" remote get-url origin 2>/dev/null || echo none)"
        if [ "$(_norm_remote "$_actual")" != "$(_norm_remote "$_repo")" ]; then
            echo "[roll] skip $_d (remote mismatch: $_actual) — fix manually, see docs."
            continue
        fi
        _hold="$(_hold_for "$_d")"
        if [ -n "$_hold" ]; then
            echo "[roll] $_d HELD at $_hold"
            git -C "$COMFY_ROOT/custom_nodes/$_d" checkout --detach "$_hold" || echo "[roll] WARNING: hold checkout failed for $_d" >&2
            continue
        fi
        _branch="$(git -C "$COMFY_ROOT/custom_nodes/$_d" remote show origin 2>/dev/null | sed -n 's/.*HEAD branch: //p')"
        if git -C "$COMFY_ROOT/custom_nodes/$_d" fetch --depth 1 origin "$_branch" \
            && git -C "$COMFY_ROOT/custom_nodes/$_d" checkout --detach FETCH_HEAD; then
            echo "[roll] $_d now: $(git -C "$COMFY_ROOT/custom_nodes/$_d" rev-parse --short HEAD)"
        else
            echo "[roll] WARNING: float failed for $_d, restoring snapshot commit" >&2
            _snap="$(grep -a "^${_d} " "$SNAP_DIR/nodes.txt" | awk '{print $2}')"
            git -C "$COMFY_ROOT/custom_nodes/$_d" checkout --detach "$_snap" 2>/dev/null || true
            unset _snap
        fi
        unset _branch _hold _actual
    done
    # controlnet_aux float: clean checkout first (never carry the dirty
    # patch tree across commits — it can silently half-apply), then apply
    # the canonical XPU patch. Hold on reject, never ship unpatched.
    _caux="$COMFY_ROOT/custom_nodes/comfyui_controlnet_aux"
    if [ -e "$_caux/.git" ]; then
        _caux_hold="$(_hold_for comfyui_controlnet_aux)"
        _caux_snap="$(grep -a "^comfyui_controlnet_aux " "$SNAP_DIR/nodes.txt" | awk '{print $2}')"
        git -C "$_caux" diff -- src/custom_controlnet_aux/depth_anything_v2/dpt.py > "$SNAP_DIR/caux-patch-backup.diff" 2>/dev/null || true
        git -C "$_caux" checkout -- src/custom_controlnet_aux/depth_anything_v2/dpt.py 2>/dev/null || true
        if [ -n "$_caux_hold" ]; then
            echo "[roll] comfyui_controlnet_aux HELD at $_caux_hold"
            git -C "$_caux" checkout --detach "$_caux_hold" || echo "[roll] WARNING: hold checkout failed for controlnet_aux" >&2
            git -C "$_caux" apply "$PATCH_FILE" 2>/dev/null || git -C "$_caux" apply "$SNAP_DIR/caux-patch-backup.diff" 2>/dev/null || true
        else
            _branch="$(git -C "$_caux" remote show origin 2>/dev/null | sed -n 's/.*HEAD branch: //p')"
            if git -C "$_caux" fetch --depth 1 origin "$_branch" && git -C "$_caux" checkout --detach FETCH_HEAD; then
                echo "[roll] comfyui_controlnet_aux now: $(git -C "$_caux" rev-parse --short HEAD)"
            else
                echo "[roll] WARNING: float failed for controlnet_aux, restoring snapshot" >&2
                git -C "$_caux" checkout --detach "$_caux_snap" 2>/dev/null || true
                git -C "$_caux" apply "$SNAP_DIR/caux-patch-backup.diff" 2>/dev/null || true
            fi
            unset _branch
        fi
        if git -C "$_caux" status --short | grep -a -q "depth_anything_v2/dpt.py"; then
            echo "[roll] controlnet_aux XPU patch present."
        elif git -C "$_caux" apply "$PATCH_FILE" 2>/dev/null; then
            echo "[roll] controlnet_aux XPU patch applied."
        else
            git -C "$_caux" checkout --detach "$_caux_snap" 2>/dev/null || true
            git -C "$_caux" apply "$SNAP_DIR/caux-patch-backup.diff" 2>/dev/null || git -C "$_caux" apply "$PATCH_FILE" 2>/dev/null || true
            echo "comfyui_controlnet_aux=$_caux_snap" >> "$HOLDS_FILE"
            echo "[roll] WARNING: patch rejected upstream — held comfyui_controlnet_aux at $_caux_snap (added to roll-holds.conf)" >&2
        fi
        unset _caux _caux_hold _caux_snap
    fi
fi

# --- 6. Node requirements: ensure-installed, ALWAYS (fresh venv needs
# them even when the git float is skipped; no -U here, so this completes
# the venv without floating versions — companion-pins (-U) is the floater).
while IFS='|' read -r _d _repo; do
    [ -z "$_d" ] && continue
    if [ -f "$COMFY_ROOT/custom_nodes/$_d/requirements.txt" ]; then
        pip install -r "$COMFY_ROOT/custom_nodes/$_d/requirements.txt" \
            || echo "[roll] WARNING: $_d requirements failed (fix forward)" >&2
    fi
done <<< "$FLOAT_NODES"
unset _d _repo
if [ -d "$COMFY_ROOT/custom_nodes/ComfyUI-nunchaku-XPU" ]; then
    pip install --no-deps --no-build-isolation --ignore-requires-python \
        "$COMFY_ROOT/custom_nodes/ComfyUI-nunchaku-XPU" \
        || echo "[roll] WARNING: nunchaku dist rebuild failed (fix forward)" >&2
fi

# --- 6b. AIMDO malloc_graph shim (idempotent; required once core floats
# past the hard `import comfy_aimdo.malloc_graph`; import-only on XPU) ---
bash "$SUITE_DIR/scripts/apply-aimdo-shim.sh" || rollback "AIMDO shim failed"

# --- 7. Verify gate ---
echo "[roll] running verify gate ..."
if bash "$SUITE_DIR/scripts/roll-verify.sh"; then
    echo "[roll] gate GREEN."
else
    rollback "verify gate RED"
fi

# --- 8. Record last-good + prune ---
python - <<PYEOF || echo "[roll] WARNING: last-good record failed" >&2
import json, subprocess

def _out(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, text=True).strip()
    except Exception:
        return "unknown"

nodes = {}
try:
    with open("$SNAP_DIR/nodes.txt") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2 and parts[1] not in ("(no-git)", "unknown"):
                d = parts[0]
                nodes[d] = _out(f"git -C custom_nodes/{d} rev-parse HEAD")
except OSError:
    pass

state = {
    "date": "$STAMP",
    "python": _out("python --version"),
    "torch": _out("python -c 'import torch; print(torch.__version__)'"),
    "torchvision": _out("python -c 'import torchvision; print(torchvision.__version__)'"),
    "triton_xpu": _out("python -c 'import importlib.metadata as m; print(m.version(\"triton-xpu\"))'"),
    "kernel": _out("python -c 'import omni_xpu_kernel as k; print(getattr(k, \"__version__\", \"unknown\"))'"),
    "kitchen": _out("python -c 'import importlib.metadata as m; print(m.version(\"comfy-kitchen\"))'"),
    "aimdo": _out("python -c 'import importlib.metadata as m; print(m.version(\"comfy-aimdo\"))'"),
    "comfyui_core": _out("git rev-parse HEAD"),
    "nodes": nodes,
}
with open("$SUITE_DIR/manifests/last-good.json", "w") as f:
    json.dump(state, f, indent=2, sort_keys=True)
print("last-good.json written.")
PYEOF

# Prune pre-roll venvs, keep newest 1.
ls -dt "$COMFY_ROOT"/venv.pre-roll-* 2>/dev/null | tail -n +2 | xargs -r rm -rf
echo "[roll] ROLL COMPLETE — live venv is the new tip. Review + push manifests."
