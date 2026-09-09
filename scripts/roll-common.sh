# Shared float list + helpers for the Aurora update tooling.
#
# Sourced (not executed) by scripts/roll.sh (full release: venv rebuild +
# double-boot verify) and scripts/update.sh (painless: no venv touch, quick
# verify). Canonical home of FLOAT_NODES — edit the node list here and both
# runners pick it up.
#
# Callers must set HOLDS_FILE (path to manifests/roll-holds.conf) before
# using _hold_for.

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
