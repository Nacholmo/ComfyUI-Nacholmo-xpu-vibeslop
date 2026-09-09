#!/bin/bash
# Painless updater for Aurora ComfyUI (Intel Arc XPU).
#
# Floats custom nodes (and optionally ComfyUI core) WITHOUT rebuilding the
# venv: no torch reinstall, no provider re-stamp, no dependency re-resolve.
# Typical run finishes in ~1-3 minutes plus verification.
#
# Safety: node/core commits are snapshotted to manifests/update-snapshots/
# before any mutation; a failed verification restores them automatically
# (venv is never touched, so there is nothing else to restore).
#
# Usage:
#   update.sh [--dry-run] [--yes] [--core] [--verify=quick|full|none]
#             [--rollback=SNAP] [--help]
#   --dry-run   Preview planned floats (no network, no checkouts).
#   --yes       Skip the confirmation prompt.
#   --core      Also float ComfyUI core (default: nodes only).
#   --verify=   quick (default): provider probe + --help import smoke, seconds.
#               full: delegate to scripts/roll-verify.sh (two boots, ~10+ min).
#               none: skip verification (snapshot still recorded for rollback).
#   --rollback= Restore node/core commits from a previous snapshot dir or
#               stamp under manifests/update-snapshots/ (or roll-snapshots/).
set -uo pipefail

VERIFY=quick
ASSUME_YES=0
WITH_CORE=0
DRY_RUN=0
ROLLBACK=""
for arg in "$@"; do
    case $arg in
        --dry-run) DRY_RUN=1 ;;
        --yes) ASSUME_YES=1 ;;
        --core) WITH_CORE=1 ;;
        --verify=*) VERIFY="${arg#--verify=}" ;;
        --rollback=*) ROLLBACK="${arg#--rollback=}" ;;
        --help|-h)
            sed -n '2,17p' "${BASH_SOURCE[0]}"
            exit 0 ;;
        *) echo "[update] Unknown argument: $arg" >&2; exit 2 ;;
    esac
done
case "$VERIFY" in
    quick|full|none) ;;
    *) echo "[update] --verify must be quick|full|none (got '$VERIFY')" >&2; exit 2 ;;
esac

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
    echo "[update] Error: ComfyUI root not found." >&2
    exit 1
fi

HOLDS_FILE="$SUITE_DIR/manifests/roll-holds.conf"
PATCH_FILE="${PATCH_FILE:-${OMNI_NODES_PATCH:-/home/sundae/Drives/Fenix/Comfy-omni/llm-scaler/omni/patches/comfyui_controlnet_aux_depth_anything_v2_xpu.patch}}"
# shellcheck disable=SC1091
. "$SUITE_DIR/scripts/roll-common.sh"

# --- Manual rollback: restore commits, touch nothing else. ---
if [ -n "$ROLLBACK" ]; then
    if [ -d "$ROLLBACK" ]; then
        SNAP_DIR="$ROLLBACK"
    elif [ -d "$SUITE_DIR/manifests/update-snapshots/$ROLLBACK" ]; then
        SNAP_DIR="$SUITE_DIR/manifests/update-snapshots/$ROLLBACK"
    elif [ -d "$SUITE_DIR/manifests/roll-snapshots/$ROLLBACK" ]; then
        SNAP_DIR="$SUITE_DIR/manifests/roll-snapshots/$ROLLBACK"
    else
        echo "[update] Snapshot not found: $ROLLBACK" >&2
        exit 1
    fi
    if [ ! -f "$SNAP_DIR/nodes.txt" ]; then
        echo "[update] No nodes.txt in $SNAP_DIR" >&2
        exit 1
    fi
    while read -r _n _c _b; do
        [ "$_n" = "ComfyUI-Nacholmo-xpu-vibeslop" ] && continue
        if [ -e "$COMFY_ROOT/custom_nodes/$_n/.git" ] && [ "$_c" != "(no-git)" ] && [ "$_c" != "unknown" ]; then
            git -C "$COMFY_ROOT/custom_nodes/$_n" checkout --detach "$_c" 2>/dev/null \
                && echo "[update] restored $_n @ $(git -C "$COMFY_ROOT/custom_nodes/$_n" rev-parse --short HEAD)" \
                || echo "[update] WARNING: could not restore $_n" >&2
        fi
    done < "$SNAP_DIR/nodes.txt"
    if [ -f "$SNAP_DIR/comfy-core.txt" ]; then
        _core="$(cat "$SNAP_DIR/comfy-core.txt")"
        if [ "$_core" != "unknown" ] && [ -n "$_core" ]; then
            git -C "$COMFY_ROOT" checkout --detach "$_core" 2>/dev/null \
                && echo "[update] restored core @ $(git -C "$COMFY_ROOT" rev-parse --short HEAD)" \
                || echo "[update] WARNING: could not restore core" >&2
        fi
        unset _core
    fi
    unset _n _c _b
    echo "[update] Rollback from $SNAP_DIR complete. Restart ComfyUI."
    exit 0
fi

# --- Dry run: plan only (no network, no checkouts, no snapshot dir). ---
if [ "$DRY_RUN" -eq 1 ]; then
    echo "[update] DRY RUN — planned floats (venv untouched, verify=$VERIFY):"
    [ "$WITH_CORE" -eq 1 ] && echo "  - float ComfyUI core to origin HEAD $([ -n "$(_hold_for comfy-core)" ] && echo "(HELD at $(_hold_for comfy-core))")"
    echo "$FLOAT_NODES" | while IFS='|' read -r _d _r; do
        [ -z "$_d" ] && continue
        [ "$_d" = "comfyui_controlnet_aux" ] && { echo "      $_d (patch-safe path, see roll.sh)"; continue; }
        _h="$(_hold_for "$_d")"
        if [ -d "$COMFY_ROOT/custom_nodes/$_d" ] && [ -e "$COMFY_ROOT/custom_nodes/$_d/.git" ]; then
            echo "      $_d @ $(git -C "$COMFY_ROOT/custom_nodes/$_d" rev-parse --short HEAD 2>/dev/null || echo missing) -> origin HEAD $([ -n "$_h" ] && echo "(HELD at $_h)")"
        else
            echo "      $_d (missing or no-git, will skip)"
        fi
    done
    echo "  - pip install requirements only for nodes that actually moved"
    echo "  - snapshot -> manifests/update-snapshots/<date>/, restore on verify RED"
    exit 0
fi

if [ "$ASSUME_YES" -eq 0 ]; then
    echo "[update] This floats $(echo "$FLOAT_NODES" | grep -c '|') custom nodes$([ "$WITH_CORE" -eq 1 ] && echo ' + ComfyUI core') in place (no venv rebuild)."
    read -r -p "[update] Proceed? [y/N] " _ans
    if [ "$_ans" != "y" ] && [ "$_ans" != "Y" ]; then
        echo "[update] Aborted."
        exit 0
    fi
    unset _ans
fi

STAMP="$(date +%Y%m%d-%H%M)"
SNAP_DIR="$SUITE_DIR/manifests/update-snapshots/$STAMP"
mkdir -p "$SNAP_DIR"

# --- 0. Snapshot commits (no venv copy — venv is never modified). ---
git -C "$COMFY_ROOT" rev-parse HEAD > "$SNAP_DIR/comfy-core.txt" 2>/dev/null || echo unknown > "$SNAP_DIR/comfy-core.txt"
: > "$SNAP_DIR/nodes.txt"
for _d in "$COMFY_ROOT"/custom_nodes/*/; do
    _n="$(basename "$_d")"
    if [ -e "$_d/.git" ]; then
        echo "$_n $(git -C "$_d" rev-parse HEAD 2>/dev/null || echo unknown)" >> "$SNAP_DIR/nodes.txt"
    else
        echo "$_n (no-git)" >> "$SNAP_DIR/nodes.txt"
    fi
done
unset _d _n

restore_snapshot() { # $1=reason — checkout snapshot commits, exit 3 like roll.sh
    echo "[update] RESTORE: $1" >&2
    while read -r _n _c; do
        [ "$_n" = "ComfyUI-Nacholmo-xpu-vibeslop" ] && continue
        if [ -e "$COMFY_ROOT/custom_nodes/$_n/.git" ] && [ "$_c" != "(no-git)" ] && [ "$_c" != "unknown" ]; then
            git -C "$COMFY_ROOT/custom_nodes/$_n" checkout --detach "$_c" 2>/dev/null || true
        fi
    done < "$SNAP_DIR/nodes.txt"
    _core="$(cat "$SNAP_DIR/comfy-core.txt")"
    [ "$_core" != "unknown" ] && [ -n "$_core" ] && git -C "$COMFY_ROOT" checkout --detach "$_core" 2>/dev/null || true
    unset _n _c _core
    echo "[update] Restored commits from $SNAP_DIR" >&2
    exit 3
}

cd "$COMFY_ROOT" || exit 1
if [ -d "venv" ]; then
    # shellcheck disable=SC1091
    source venv/bin/activate
fi

# --- 1. Core float (opt-in). ---
if [ "$WITH_CORE" -eq 1 ]; then
    _core_hold="$(_hold_for comfy-core)"
    if [ -n "$_core_hold" ]; then
        echo "[update] core HELD at $_core_hold"
        git checkout --detach "$_core_hold" || restore_snapshot "core hold checkout failed"
    else
        _branch="$(git remote show origin 2>/dev/null | sed -n 's/.*HEAD branch: //p')"
        git fetch --depth 50 origin "$_branch" || restore_snapshot "core fetch failed"
        git checkout --detach FETCH_HEAD || restore_snapshot "core checkout failed"
        echo "[update] core now: $(git rev-parse --short HEAD)"
        unset _branch
    fi
    unset _core_hold
    bash "$SUITE_DIR/scripts/apply-aimdo-shim.sh" || restore_snapshot "AIMDO shim failed"
fi

# --- 2. Node float (in place; collect the ones that moved). ---
CHANGED=""
echo "$FLOAT_NODES" | while IFS='|' read -r _d _repo; do
    [ -z "$_d" ] && continue
    [ "$_d" = "comfyui_controlnet_aux" ] && continue # patch-safe block below
    if [ ! -d "$COMFY_ROOT/custom_nodes/$_d" ]; then
        echo "[update] skip $_d (not installed)"
        continue
    fi
    if [ ! -e "$COMFY_ROOT/custom_nodes/$_d/.git" ]; then
        echo "[update] skip $_d (no-git checkout)"
        continue
    fi
    _actual="$(git -C "$COMFY_ROOT/custom_nodes/$_d" remote get-url origin 2>/dev/null || echo none)"
    if [ "$(_norm_remote "$_actual")" != "$(_norm_remote "$_repo")" ]; then
        echo "[update] skip $_d (remote mismatch: $_actual)"
        continue
    fi
    _hold="$(_hold_for "$_d")"
    if [ -n "$_hold" ]; then
        echo "[update] $_d HELD at $_hold"
        git -C "$COMFY_ROOT/custom_nodes/$_d" checkout --detach "$_hold" || echo "[update] WARNING: hold checkout failed for $_d" >&2
        continue
    fi
    _before="$(git -C "$COMFY_ROOT/custom_nodes/$_d" rev-parse HEAD)"
    _branch="$(git -C "$COMFY_ROOT/custom_nodes/$_d" remote show origin 2>/dev/null | sed -n 's/.*HEAD branch: //p')"
    if git -C "$COMFY_ROOT/custom_nodes/$_d" fetch --depth 1 origin "$_branch" \
        && git -C "$COMFY_ROOT/custom_nodes/$_d" checkout --detach FETCH_HEAD; then
        _after="$(git -C "$COMFY_ROOT/custom_nodes/$_d" rev-parse HEAD)"
        if [ "$_before" = "$_after" ]; then
            echo "[update] $_d already at tip ($(git -C "$COMFY_ROOT/custom_nodes/$_d" rev-parse --short HEAD))"
        else
            echo "[update] $_d now: $(git -C "$COMFY_ROOT/custom_nodes/$_d" rev-parse --short HEAD)"
            echo "$_d" >> "$SNAP_DIR/changed.txt"
        fi
    else
        echo "[update] WARNING: float failed for $_d, keeping current commit" >&2
    fi
    unset _branch _hold _actual _before _after
done
# controlnet_aux: float from a clean tree, re-apply the XPU patch, hold on reject.
_caux="$COMFY_ROOT/custom_nodes/comfyui_controlnet_aux"
if [ -e "$_caux/.git" ]; then
    _caux_hold="$(_hold_for comfyui_controlnet_aux)"
    _caux_before="$(git -C "$_caux" rev-parse HEAD 2>/dev/null)"
    git -C "$_caux" diff -- src/custom_controlnet_aux/depth_anything_v2/dpt.py > "$SNAP_DIR/caux-patch-backup.diff" 2>/dev/null || true
    git -C "$_caux" checkout -- src/custom_controlnet_aux/depth_anything_v2/dpt.py 2>/dev/null || true
    if [ -n "$_caux_hold" ]; then
        echo "[update] comfyui_controlnet_aux HELD at $_caux_hold"
        git -C "$_caux" checkout --detach "$_caux_hold" || echo "[update] WARNING: hold checkout failed" >&2
    else
        _branch="$(git -C "$_caux" remote show origin 2>/dev/null | sed -n 's/.*HEAD branch: //p')"
        git -C "$_caux" fetch --depth 1 origin "$_branch" && git -C "$_caux" checkout --detach FETCH_HEAD \
            && echo "[update] comfyui_controlnet_aux now: $(git -C "$_caux" rev-parse --short HEAD)" \
            || echo "[update] WARNING: float failed for controlnet_aux, keeping current" >&2
        unset _branch
    fi
    if [ -f "$PATCH_FILE" ] && git -C "$_caux" apply "$PATCH_FILE" 2>/dev/null; then
        echo "[update] controlnet_aux XPU patch applied."
    else
        git -C "$_caux" apply "$SNAP_DIR/caux-patch-backup.diff" 2>/dev/null || true
        echo "[update] controlnet_aux kept previous patch state."
    fi
    if [ "$(git -C "$_caux" rev-parse HEAD 2>/dev/null)" != "$_caux_before" ]; then
        echo "comfyui_controlnet_aux" >> "$SNAP_DIR/changed.txt"
    fi
    unset _caux _caux_hold _caux_before
fi
# NOTE: the float loop above runs in a subshell (pipe), so CHANGED is read
# back from changed.txt rather than a shell variable.
if [ -f "$SNAP_DIR/changed.txt" ]; then
    echo "[update] moved: $(tr '\n' ' ' < "$SNAP_DIR/changed.txt")"
else
    echo "[update] everything already at tip — nothing moved."
fi

# --- 3. Requirements only for nodes that moved (fresh venv needs them all;
# here the venv already has them, so this just completes newly added deps). ---
if [ -f "$SNAP_DIR/changed.txt" ]; then
    while read -r _d; do
        [ -z "$_d" ] && continue
        if [ -f "$COMFY_ROOT/custom_nodes/$_d/requirements.txt" ]; then
            pip install -r "$COMFY_ROOT/custom_nodes/$_d/requirements.txt" \
                || echo "[update] WARNING: $_d requirements failed (fix forward)" >&2
        fi
    done < "$SNAP_DIR/changed.txt"
    unset _d
    if grep -qx "ComfyUI-nunchaku-XPU" "$SNAP_DIR/changed.txt" && [ -d "$COMFY_ROOT/custom_nodes/ComfyUI-nunchaku-XPU" ]; then
        pip install --no-deps --no-build-isolation --ignore-requires-python \
            "$COMFY_ROOT/custom_nodes/ComfyUI-nunchaku-XPU" \
            || echo "[update] WARNING: nunchaku dist rebuild failed (fix forward)" >&2
    fi
fi

# --- 4. Verify. ---
_verify_quick() { # provider probe + --help import smoke in both modes, seconds
    python - <<'PYEOF' || return 1
import importlib.util
import sys
sys.argv = ['main.py']
spec = importlib.util.spec_from_file_location(
    '_uq_bootstrap', 'custom_nodes/ComfyUI-OmniXPU/runtime_bootstrap.py')
mod = importlib.util.module_from_spec(spec)
sys.modules['_uq_bootstrap'] = mod
spec.loader.exec_module(mod)
state = mod.bootstrap()
kitchen = state['providers'].get('comfy_kitchen.xpu', {})
assert kitchen.get('status') == 'active', f"Kitchen XPU not active: {kitchen}"
print('[update] quick verify: Kitchen XPU provider active')
PYEOF
    XPU_VRAM_MODE=direct bash "$SUITE_DIR/scripts/launch_xpu.sh" --help >/dev/null 2>&1 \
        || { echo "[update] quick verify: direct-mode import smoke failed" >&2; return 1; }
    bash "$SUITE_DIR/scripts/launch_xpu.sh" --help >/dev/null 2>&1 \
        || { echo "[update] quick verify: vram-mode import smoke failed" >&2; return 1; }
    echo "[update] quick verify: import smoke clean in both modes"
}

if [ "$VERIFY" = "quick" ]; then
    echo "[update] running quick verify ..."
    _verify_quick || restore_snapshot "quick verify RED"
    echo "[update] verify GREEN (quick)."
elif [ "$VERIFY" = "full" ]; then
    echo "[update] running full verify gate ..."
    bash "$SUITE_DIR/scripts/roll-verify.sh" --snapshot-dir "$SNAP_DIR" || restore_snapshot "full verify RED"
    echo "[update] verify GREEN (full)."
else
    echo "[update] verify skipped (--verify=none). Snapshot at $SNAP_DIR if you need --rollback."
fi

echo "[update] COMPLETE — snapshot at $SNAP_DIR. Restart ComfyUI."
