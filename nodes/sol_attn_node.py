"""Sol-Attn sparse block attention node for Intel Arc / XPU."""

import logging
import math
import re
import sys
import weakref

import torch

log = logging.getLogger("ComfyUI-Nacholmo-xpu-vibeslop")

_BLOCK_INDEX_HOOKED = weakref.WeakSet()
_INSTALLED = weakref.WeakSet()
_PATCHED_LAYOUTS = weakref.WeakSet()
# position_ids tensor -> (layout, bounds, span); WeakKeyDictionary so entries
# vanish with the tensor instead of leaking via id() reuse-after-GC.
_SPANS = weakref.WeakKeyDictionary()
# Fallback for tensors that cannot be weak-referenced (rare): id -> entry.
_SPANS_BY_ID = {}
_PERM_CACHE = {}
_DEVICE_CACHE = {}
_PERM_CACHE_LIMIT = 64


def parse_blocks(spec, count):
    """Parse '0-3,47,-1' into absolute block indices; negatives count from end."""
    out = set()
    for part in "".join(str(spec).split()).split(","):
        if not part:
            continue
        match = re.fullmatch(r"(-?\d+)(?:-(-?\d+))?", part)
        if match is None:
            continue
        first = int(match.group(1))
        last = first if match.group(2) is None else int(match.group(2))
        first = first if first >= 0 else count + first
        last = last if last >= 0 else count + last
        if first > last:
            first, last = last, first
        out.update(range(max(first, 0), min(last, count - 1) + 1))
    return frozenset(out)


def parse_tau_profile(spec, count):
    """Parse '0-30=2.0; 39-42=0.9' into {block: tau}."""
    profile = {}
    for entry in re.split(r"[;\n]", str(spec)):
        entry = entry.split("#", 1)[0].strip()
        if not entry:
            continue
        blocks, sep, value = entry.partition("=")
        if not sep:
            continue
        try:
            level = float(value.strip())
        except ValueError:
            continue
        for block in parse_blocks(blocks, count):
            profile[block] = level
    return profile


def _install_block_index(model):
    """Publish running block index into transformer_options."""
    blocks = getattr(model, "blocks", None)
    if blocks is None:
        return False
    if model in _BLOCK_INDEX_HOOKED:
        return True

    def make_hook(index):
        def hook(_module, _args, kwargs):
            options = kwargs.get("transformer_options")
            if isinstance(options, dict):
                options["sol_block"] = index
            return None
        return hook

    for index, block in enumerate(blocks):
        block.register_forward_pre_hook(make_hook(index), with_kwargs=True)
    _BLOCK_INDEX_HOOKED.add(model)
    return True


def morton_perm(grid, device, curve="2d_frame"):
    key = (tuple(int(x) for x in grid), curve)
    hit = _PERM_CACHE.get(key)
    if hit is None:
        frames, height, width = key[0]
        linear = torch.arange(frames * height * width, dtype=torch.int64)
        area = height * width
        z = linear // area
        rem = linear - z * area
        y = rem // width
        x = rem - y * width

        def part1by2(value):
            value = value & 0x1FFFFF
            value = (value | (value << 32)) & 0x1F00000000FFFF
            value = (value | (value << 16)) & 0x1F0000FF0000FF
            value = (value | (value << 8)) & 0x100F00F00F00F00F
            value = (value | (value << 4)) & 0x10C30C30C30C30C3
            value = (value | (value << 2)) & 0x1249249249249249
            return value

        if curve == "2d_frame":
            code = (z << 42) | part1by2(x) | (part1by2(y) << 1)
        else:
            code = part1by2(x) | (part1by2(y) << 1) | (part1by2(z) << 2)
        perm = linear[torch.argsort(code)]
        hit = (perm, torch.argsort(perm))
        if len(_PERM_CACHE) >= _PERM_CACHE_LIMIT:
            # Evict oldest entry; grids vary per resolution.
            _PERM_CACHE.pop(next(iter(_PERM_CACHE)))
        _PERM_CACHE[key] = hit
    return hit[0].to(device), hit[1].to(device)


def _perm_for(grid, curve, device, start):
    pad = (-int(start)) % 64
    key = (tuple(grid), curve, str(device), pad)
    hit = _DEVICE_CACHE.get(key)
    if hit is None:
        perm, inverse = morton_perm(grid, device, curve)
        if pad:
            perm = torch.roll(perm, pad)
            inverse = torch.argsort(perm)
        if len(_DEVICE_CACHE) >= _PERM_CACHE_LIMIT:
            _DEVICE_CACHE.pop(next(iter(_DEVICE_CACHE)))
        hit = (perm, inverse)
        _DEVICE_CACHE[key] = hit
    return hit


def _video_span(layout, latent_t, latent_h, latent_w):
    segments = getattr(layout, "segments", None)
    if not segments:
        return None
    span = next(((a, b) for a, b, kind in segments if kind == "video"), None)
    if span is None:
        return None
    start, stop = span
    grid = (int(latent_t), int(latent_h) // 2, int(latent_w) // 2)
    if grid[0] * grid[1] * grid[2] != stop - start:
        return None
    return start, stop, grid


def _span_get(position_ids):
    try:
        return _SPANS.get(position_ids)
    except TypeError:
        return _SPANS_BY_ID.get(id(position_ids))


def _span_set(position_ids, value):
    try:
        _SPANS[position_ids] = value
    except TypeError:
        _SPANS_BY_ID[id(position_ids)] = value
        # Bound fallback dict; id-reuse risk is small but cap it.
        if len(_SPANS_BY_ID) > 256:
            _SPANS_BY_ID.pop(next(iter(_SPANS_BY_ID)))


def _patch_packed_layout(module):
    layout_cls = getattr(module, "PackedLayout", None)
    if layout_cls is None or layout_cls in _PATCHED_LAYOUTS:
        return
    original_init = layout_cls.__init__

    def __init__(self, text_len, latent_t, latent_h, latent_w, audio_t, *args, **kwargs):
        original_init(self, text_len, latent_t, latent_h, latent_w, audio_t, *args, **kwargs)
        try:
            span = _video_span(self, latent_t, latent_h, latent_w)
        except Exception:
            span = None
        bounds = next(((a, b) for a, b, kind in getattr(self, "segments", []) or [] if kind == "video"), None)
        if torch.is_tensor(getattr(self, "position_ids", None)) and bounds is not None:
            _span_set(self.position_ids, (self, bounds, span))

    layout_cls.__init__ = __init__
    _PATCHED_LAYOUTS.add(layout_cls)


def install_h3_morton(model):
    if model in _INSTALLED:
        return
    for attr in ("rope_freqs", "_forward", "blocks"):
        if not hasattr(model, attr):
            return

    try:
        mod = sys.modules.get(type(model).__module__)
        if mod:
            _patch_packed_layout(mod)
    except Exception:
        pass

    original_forward = model._forward
    original_rope_freqs = model.rope_freqs

    def _forward(x, timestep, context, transformer_options={}, **kwargs):
        previous = getattr(model, "_sol_morton_active", False)
        model._sol_morton_active = bool(transformer_options.get("sol_morton"))
        model._sol_morton_curve = transformer_options.get("sol_morton_curve", "2d_frame")
        model._sol_transformer_options = transformer_options
        try:
            return original_forward(x, timestep, context, transformer_options=transformer_options, **kwargs)
        finally:
            model._sol_morton_active = previous
            model._sol_morton_span = None
            model._sol_morton_state = None
            model._sol_transformer_options = None
            transformer_options.pop("sol_h3_video_span", None)

    def rope_freqs(position_ids, device):
        model._sol_morton_span = None
        model._sol_morton_state = None
        entry = _span_get(position_ids)
        if entry is not None:
            _layout, bounds, span = entry
            options = getattr(model, "_sol_transformer_options", None)
            if options is not None:
                options["sol_h3_video_span"] = bounds
            if getattr(model, "_sol_morton_active", False) and span is not None:
                model._sol_morton_span = span
        return original_rope_freqs(position_ids, device)

    def pre_hook(module, args):
        if len(args) < 4:
            return None
        first = model.blocks[0]
        if module is not first:
            state = getattr(model, "_sol_morton_state", None)
            if state is None:
                return None
            return args[:3] + (state[1],) + tuple(args[4:])

        model._sol_morton_state = None
        span = getattr(model, "_sol_morton_span", None)
        if span is None:
            return None
        start, stop, grid = span
        h, rope = args[0], args[3]
        if not torch.is_tensor(h) or h.ndim != 2 or h.shape[0] < stop:
            return None
        if not torch.is_tensor(rope) or rope.ndim < 2 or rope.shape[1] != h.shape[0]:
            return None

        curve = getattr(model, "_sol_morton_curve", "2d_frame")
        perm, inverse = _perm_for(grid, curve, h.device, start)
        h = h.clone()
        h[start:stop] = h[start:stop][perm]

        full = torch.arange(rope.shape[1], device=rope.device)
        full[start:stop] = perm + start
        rope = rope.index_select(1, full)

        model._sol_morton_state = (inverse, rope)
        return (h,) + args[1:3] + (rope,) + tuple(args[4:])

    def post_hook(_module, _args, output):
        state = getattr(model, "_sol_morton_state", None)
        span = getattr(model, "_sol_morton_span", None)
        model._sol_morton_state = None
        if state is None or span is None:
            return None
        start, stop, _grid = span
        inverse, _rope = state
        if not torch.is_tensor(output) or output.shape[0] < stop:
            return None
        output = output.clone()
        output[start:stop] = output[start:stop][inverse]
        return output

    model._forward = _forward
    model.rope_freqs = rope_freqs
    for block in model.blocks:
        block.register_forward_pre_hook(pre_hook)
    model.blocks[-1].register_forward_hook(post_hook)
    _INSTALLED.add(model)


def install_wan_morton(diffusion_model):
    if diffusion_model in _INSTALLED:
        return
    if not hasattr(diffusion_model, "blocks"):
        return
    model = diffusion_model
    blocks = model.blocks
    first, last = blocks[0], blocks[-1]

    def _decide(x, freqs, transformer_options):
        grid = transformer_options.get("grid_sizes")
        if grid is None:
            return None
        tokens = int(math.prod(int(g) for g in grid))
        if not torch.is_tensor(x) or x.ndim != 3 or x.shape[1] != tokens:
            return None
        if not torch.is_tensor(freqs) or freqs.shape[1] != tokens:
            return None
        curve = transformer_options.get("sol_morton_curve", "3d")
        perm, inverse = morton_perm(grid, x.device, curve)
        return perm, inverse, freqs.index_select(1, perm)

    def pre_hook(module, args, kwargs):
        transformer_options = kwargs.get("transformer_options") or {}
        if not transformer_options.get("sol_morton") or not args:
            return None

        if module is first:
            model._sol_morton_state = None
            decision = _decide(args[0], kwargs.get("freqs"), transformer_options)
            if decision is None:
                return None
            perm, inverse, freqs = decision
            model._sol_morton_state = (perm, inverse, freqs)
            kwargs = {**kwargs, "freqs": freqs}
            return (args[0].index_select(1, perm),) + tuple(args[1:]), kwargs

        state = getattr(model, "_sol_morton_state", None)
        if state is None:
            return None
        kwargs = {**kwargs, "freqs": state[2]}
        return args, kwargs

    def post_hook(_module, _args, _kwargs, output):
        state = getattr(model, "_sol_morton_state", None)
        if state is None:
            return None
        model._sol_morton_state = None
        _perm, inverse, _freqs = state
        if not torch.is_tensor(output) or output.shape[1] != inverse.numel():
            return None
        return output.index_select(1, inverse)

    for block in blocks:
        block.register_forward_pre_hook(pre_hook, with_kwargs=True)
    last.register_forward_hook(post_hook, with_kwargs=True)
    _INSTALLED.add(diffusion_model)


class ApplySolAttn:
    """Enables accelerated sparse Sol-Attn on Intel Arc XPU."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "tau": (
                    "FLOAT",
                    {
                        "default": 1.30,
                        "min": 0.0,
                        "max": 4.0,
                        "step": 0.05,
                        "tooltip": "Threshold tau for Sol-Attn pruning. Higher is sparser. 1.0 ~ 16% blocks kept, 1.30 is sweet spot, 1.50 ~ 7%.",
                    },
                ),
                "start_percent": (
                    "FLOAT",
                    {
                        "default": 0.20,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Run dense before this point (e.g. 0.20 for warm-up steps).",
                    },
                ),
                "end_percent": (
                    "FLOAT",
                    {
                        "default": 0.90,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Run dense after this point (e.g. 0.90 for cool-down steps).",
                    },
                ),
                "min_tokens": (
                    "INT",
                    {
                        "default": 12288,
                        "min": 512,
                        "max": 131072,
                        "step": 512,
                        "tooltip": "Minimum sequence length to activate sparse Sol-Attn. Below this sequence length, standard dense CUTE FMHA is used.",
                    },
                ),
                "sink_conditioning": (
                    ["exact_kv_and_rows", "exact_kv", "off"],
                    {
                        "default": "exact_kv_and_rows",
                        "tooltip": "MiniMax-H3 conditioning sinks. exact_kv_and_rows: keeps text/audio/ref KV exact and query rows dense for audio stream sync. exact_kv: keeps KV exact only. off: no sinks.",
                    },
                ),
                "morton": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Reorder video tokens into Morton (Z-order) so 64-token tiles form compact 3D spatial-temporal blocks instead of 2-row strips.",
                    },
                ),
                "morton_curve": (
                    ["2d_frame", "3d"],
                    {
                        "default": "2d_frame",
                        "tooltip": "2d_frame: Z-order within each frame (recommended for MiniMax-H3). 3d: interleave t/h/w equally (recommended for Wan).",
                    },
                ),
                "centroid_tail": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Preserve centroid tail tokens / boundary tokens.",
                    },
                ),
                "routed_cap_percent": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 100,
                        "step": 1,
                        "tooltip": "Cap the maximum percentage of routed blocks (0 = uncapped).",
                    },
                ),
                "reuse_qkv_memory": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Avoid extra intermediate allocations when shapes match.",
                    },
                ),
                "verbose": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Log detailed Sol-Attn execution diagnostics.",
                    },
                ),
                "dense_blocks": (
                    "STRING",
                    {
                        "default": "0-2,-1",
                        "tooltip": "Transformer block indices or ranges to keep dense, e.g. '0-2,-1'. First and last blocks are sensitive to pruning.",
                    },
                ),
            },
            "optional": {
                "tau_profile": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "forceInput": True,
                        "tooltip": "Per-block tau, overriding base tau. Syntax: '0-30=2.0; 39-42=0.9'. '#' starts comments.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "Intel-Arc/Attention"
    DESCRIPTION = "Enables Sol-Attn sparse block attention for massive 3-4x attention speedup on Intel Arc XPU."

    def apply(
        self,
        model,
        tau=1.30,
        start_percent=0.20,
        end_percent=0.90,
        min_tokens=12288,
        sink_conditioning="exact_kv_and_rows",
        morton=False,
        morton_curve="2d_frame",
        centroid_tail=True,
        routed_cap_percent=0,
        reuse_qkv_memory=False,
        verbose=False,
        dense_blocks="0-2,-1",
        tau_profile="",
        enabled=True,
        min_seq=None,
        sink_blocks=None,
        **kwargs,
    ):
        if min_seq is not None:
            min_tokens = int(min_seq)
        if not enabled:
            m = model.clone()
            m.model_options = m.model_options.copy()
            to = m.model_options["transformer_options"] = m.model_options.get("transformer_options", {}).copy()
            to["sol_attn_enabled"] = False
            log.info("[ApplySolAttn] Disabled Sol-Attn")
            return (m,)

        diffusion_model = model.get_model_object("diffusion_model")
        num_blocks = len(getattr(diffusion_model, "blocks", [])) or 50
        dense_set = parse_blocks(dense_blocks, num_blocks) if dense_blocks else frozenset()
        profile_map = parse_tau_profile(tau_profile, num_blocks) if tau_profile else {}

        if diffusion_model is not None:
            _install_block_index(diffusion_model)
            model_type = getattr(diffusion_model, "__class__", type(diffusion_model)).__name__.lower()
            is_wan = "wan" in model_type
            is_h3 = ("h3" in model_type or "minimax" in model_type) or (
                hasattr(diffusion_model, "patch_size") and not is_wan
            )
            if is_wan and morton:
                install_wan_morton(diffusion_model)
            elif is_h3:
                install_h3_morton(diffusion_model)

        sigma_start = None
        sigma_end = None
        try:
            model_sampling = model.get_model_object("model_sampling")
            if model_sampling is not None:
                sigma_start = float(model_sampling.percent_to_sigma(start_percent))
                sigma_end = float(model_sampling.percent_to_sigma(end_percent))
        except Exception as e:
            log.debug(f"[ApplySolAttn] Could not derive sigmas from model_sampling: {e}")

        m = model.clone()
        m.model_options = m.model_options.copy()
        to = m.model_options["transformer_options"] = m.model_options.get("transformer_options", {}).copy()
        to["sol_attn_enabled"] = True
        to["sol_attn_tau"] = float(tau)
        to["sol_attn_min_tokens"] = int(min_tokens)
        to["sol_attn_min_seq"] = int(min_tokens)
        to["sol_attn_sigma_start"] = sigma_start
        to["sol_attn_sigma_end"] = sigma_end
        to["sol_attn_sink_conditioning"] = str(sink_conditioning)
        to["sol_attn_morton"] = bool(morton)
        to["sol_attn_morton_curve"] = str(morton_curve)
        to["sol_attn_centroid_tail"] = bool(centroid_tail)
        to["sol_attn_routed_cap_percent"] = int(routed_cap_percent)
        to["sol_attn_reuse_qkv_memory"] = bool(reuse_qkv_memory)
        to["sol_attn_verbose"] = bool(verbose)
        to["sol_attn_dense_blocks"] = dense_set
        to["sol_attn_tau_profile"] = profile_map
        if morton:
            to["sol_morton"] = True
            to["sol_morton_curve"] = str(morton_curve)
        if sink_blocks is not None:
            to["sol_attn_sink_blocks"] = int(sink_blocks)

        log.info(
            f"[ApplySolAttn] Configured Sol-Attn (tau={tau}, min_tokens={min_tokens}, "
            f"range=[{start_percent:.2f}, {end_percent:.2f}], sinks={sink_conditioning}, "
            f"dense_blocks='{dense_blocks}', morton={morton} ({morton_curve}))"
        )
        return (m,)


NODE_CLASS_MAPPINGS = {
    "ApplySolAttn": ApplySolAttn,
    "SolAttnPatch": ApplySolAttn,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ApplySolAttn": "Apply Sol-Attn Sparse Attention (Intel Arc XPU)",
    "SolAttnPatch": "Patch Sol-Attn (MiniMax)",
}
