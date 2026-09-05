"""MiniMax-H3 Latent Split, Context Extend & Seamless Stitching Toolkit.

Allows splitting long (e.g. 10s) 0.5MP latents into shorter (e.g. 5s) chunks,
upscaling each to 2MP independently to prevent VRAM OOM on Intel Arc B580 (12GB),
and seamlessly connecting the chunks using MiniMax-H3 Extend context conditioning
followed by cosine-fade latent & audio stitching.
"""

import logging
import math
import torch

import comfy.nested_tensor

log = logging.getLogger("ComfyUI-Nacholmo-xpu-vibeslop")


def _is_nested_samples(samples):
    return (
        samples is not None
        and hasattr(samples, "is_nested")
        and samples.is_nested
        and hasattr(samples, "tensors")
        and len(samples.tensors) >= 2
    )


def _extract_video_audio(latent):
    samples = latent.get("samples")
    if _is_nested_samples(samples):
        video = samples.tensors[0]
        audio = samples.tensors[1]
        is_av = True
    elif torch.is_tensor(samples):
        video = samples
        audio = None
        is_av = False
    else:
        raise ValueError("Invalid latent structure: expected tensor or AV NestedTensor")
    return video, audio, is_av


def _wrap_latent(video, audio, is_av):
    if is_av and audio is not None:
        return {"samples": comfy.nested_tensor.NestedTensor((video, audio))}
    return {"samples": video}


def _snap_to_h3_cycle(t):
    """Ensure temporal length t satisfies MiniMax H3's (t - 2) % 5 == 0 rule."""
    if t <= 2:
        return 2
    rem = (t - 2) % 5
    if rem != 0:
        t = t - rem
    return max(2, t)


def _snap_overlap(overlap):
    """Snap overlap to 2 mod 5 (2, 7, 12, ...) so both split chunks keep valid H3 cycle length.

    Split cut s is cycle-aligned (s = 2 + 5*k1) and chunk 2 length is
    T2 = T_total - s + overlap, so (T2 - 2) % 5 == (overlap - 2) % 5.
    Only overlap % 5 == 2 keeps both chunks valid. 0 means no overlap
    (hard cut) and is passed through for Stitch plain-concat paths.
    """
    overlap = int(overlap)
    if overlap <= 0:
        return 0
    overlap = max(1, overlap)
    snapped = 2 + 5 * round((overlap - 2) / 5)
    return max(1, snapped)


def _audio_overlap_for_video_overlap(overlap):
    """Exact audio-latent overlap for a cycle-aligned trailing window of `overlap` video latents.

    Trailing rem frames of a 5-group map to 4px each, full groups to 17px,
    audio latent runs at 5/3 per pixel frame (mirrors the split slicing).
    """
    full, rem = divmod(int(overlap), 5)
    px = full * 17 + 4 * rem
    return int(round(px * (5.0 / 3.0)))


def _context_span(n_frames):
    """Calculate cursor-axis duration spanned by n_frames trailing latent frames ending at target origin."""
    try:
        import comfy.ldm.minimax.model as h3model
        return sum(h3model.FRAME_RESCALE * h3model.FRAME_PER_TOKEN[k % 5] for k in range(-n_frames, 0))
    except Exception:
        return n_frames * 4.0 * (5.0 / 3.0)


class MiniMaxH3LatentSplit:
    """Splits a long MiniMax H3 AV latent into two chunks with exact token cycle alignment."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT", {"tooltip": "Input long MiniMax H3 AV latent (e.g. 10s 0.5MP)."}),
                "split_frame": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 512,
                        "step": 1,
                        "tooltip": "Latent frame index where to split (0 = auto split roughly in half aligned to MiniMax H3 5-token cycle).",
                    },
                ),
                "overlap_latent_frames": (
                    "INT",
                    {
                        "default": 2,
                        "min": 1,
                        "max": 8,
                        "step": 1,
                        "tooltip": "Overlapping latent frames at the boundary (snapped to 2 mod 5 for H3 cycle: 2 or 7; use 7 for a wider seamless window).",
                    },
                ),
            },
        }

    RETURN_TYPES = ("LATENT", "LATENT", "LATENT", "LATENT", "LATENT", "LATENT", "INT", "INT")
    RETURN_NAMES = (
        "chunk_1",
        "chunk_2",
        "chunk_1_video",
        "chunk_1_audio",
        "chunk_2_video",
        "chunk_2_audio",
        "chunk_1_frame_count",
        "chunk_2_frame_count",
    )
    FUNCTION = "split"
    CATEGORY = "Intel-Arc/MiniMax"
    DESCRIPTION = "Splits a long MiniMax-H3 latent into two temporally aligned chunks with context overlap to enable VRAM-safe 2MP upscaling."

    def split(self, latent, split_frame=0, overlap_latent_frames=2):
        video, audio, is_av = _extract_video_audio(latent)
        T_total = video.shape[2]
        overlap = _snap_overlap(overlap_latent_frames)
        if overlap != int(overlap_latent_frames):
            log.info(f"[MiniMaxH3LatentSplit] Snapped overlap {overlap_latent_frames} -> {overlap} for H3 cycle alignment")

        if T_total < 4:
            raise ValueError(f"Latent is too short to split (T={T_total})")
        if T_total - overlap < 4:
            raise ValueError(
                f"Latent too short for overlap={overlap} (T={T_total}); "
                "use a longer latent or smaller overlap."
            )

        # Snap total to valid k if possible
        if (T_total - 2) % 5 == 0:
            k_total = (T_total - 2) // 5
            k1 = k_total // 2 if split_frame == 0 else max(1, (split_frame - 2) // 5)
            s = 2 + 5 * k1
        else:
            s = T_total // 2 if split_frame == 0 else split_frame
            s = _snap_to_h3_cycle(s)

        s = min(max(s, overlap + 2), T_total - 2)

        v1 = video[:, :, :s].contiguous()
        v2 = video[:, :, s - overlap:].contiguous()

        # Calculate pixel frame count for each chunk
        k1 = (v1.shape[2] - 2) // 5 if (v1.shape[2] - 2) % 5 == 0 else (v1.shape[2] - 2) / 5
        k2 = (v2.shape[2] - 2) // 5 if (v2.shape[2] - 2) % 5 == 0 else (v2.shape[2] - 2) / 5
        f1 = int(round(5 + 17 * k1))
        f2 = int(round(5 + 17 * k2))

        # Audio split matching exact pixel frame durations
        if is_av and audio is not None:
            A_total = audio.shape[-1]
            a1_len = min(int(round(f1 * (5.0 / 3.0))), A_total)
            a2_len = min(int(round(f2 * (5.0 / 3.0))), A_total)

            a1 = audio[..., :a1_len].contiguous()
            a2 = audio[..., A_total - a2_len:].contiguous()
        else:
            a1, a2 = None, None

        log.info(
            f"[MiniMaxH3LatentSplit] Split T={T_total} into Chunk 1 (T={v1.shape[2]}, ~{f1} frames) "
            f"and Chunk 2 (T={v2.shape[2]}, ~{f2} frames) with overlap={overlap}"
        )

        chunk_1_av = _wrap_latent(v1, a1, is_av)
        chunk_2_av = _wrap_latent(v2, a2, is_av)
        c1_v = {"samples": v1}
        c1_a = {"samples": a1} if a1 is not None else {"samples": torch.empty(0)}
        c2_v = {"samples": v2}
        c2_a = {"samples": a2} if a2 is not None else {"samples": torch.empty(0)}

        return (
            chunk_1_av,
            chunk_2_av,
            c1_v,
            c1_a,
            c2_v,
            c2_a,
            f1,
            f2,
        )


class MiniMaxH3AttachContext:
    """Injects trailing frames from the prior refined chunk into conditioning as Extend context."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING", {"tooltip": "Base conditioning to attach context to."}),
                "context_latent": ("LATENT", {"tooltip": "Prior refined high-res chunk (provides 2MP context frames)."}),
                "context_frames": (
                    "INT",
                    {
                        "default": 2,
                        "min": 1,
                        "max": 8,
                        "step": 1,
                        "tooltip": "Number of trailing latent frames to carry over as context.",
                    },
                ),
            },
            "optional": {
                "frame_count": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 1000,
                        "step": 1,
                        "tooltip": "Pixel frame count for this chunk (0 = auto-detected).",
                    },
                ),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "attach"
    CATEGORY = "Intel-Arc/MiniMax"
    DESCRIPTION = "Injects trailing context keyframes from a prior refined chunk to condition the next chunk seamlessly via MiniMax-H3-Extend."

    def attach(self, conditioning, context_latent, context_frames=2, frame_count=0):
        video, audio, is_av = _extract_video_audio(context_latent)
        ctx_t = video.shape[2]
        n_frames = min(int(context_frames), ctx_t)

        ctx_video = video[:, :, ctx_t - n_frames:].contiguous()
        keyframes = [
            {"kind": "context", "num_frames": n_frames, "latent": ctx_video}
        ]

        if is_av and audio is not None:
            ctx_audio_t = audio.shape[-1]
            # Map latent-frame overlap to audio-latent overlap with the same
            # pixel-frame math used by split/stitch (not RoPE cursor time).
            n_audio = min(int(_audio_overlap_for_video_overlap(n_frames)), ctx_audio_t)
            if n_audio > 0:
                ctx_audio = audio[..., ctx_audio_t - n_audio:].contiguous()
                keyframes.append(
                    {"kind": "context_audio", "num_frames": n_audio, "audio_latent": ctx_audio}
                )

        # Clone and update conditioning
        out_cond = []
        for tensor, d in conditioning:
            nd = dict(d)
            existing_kfs = nd.get("minimax_keyframes") or []
            # Keep any non-context keyframes (or replace existing context keyframes)
            filtered_kfs = [kf for kf in existing_kfs if kf.get("kind") not in ("context", "context_audio")]
            nd["minimax_keyframes"] = keyframes + filtered_kfs
            if frame_count > 0:
                nd["minimax_frame_count"] = int(frame_count)
            out_cond.append([tensor, nd])

        log.info(
            f"[MiniMaxH3AttachContext] Attached {n_frames} video context frames "
            f"({tuple(ctx_video.shape)}) to conditioning"
        )
        return (out_cond,)


class MiniMaxH3AssembleAV:
    """Assembles video latent and audio latent into a single MiniMax H3 AV latent."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_latent": ("LATENT", {"tooltip": "Video latent (e.g. upscaled 2MP)."}),
            },
            "optional": {
                "audio_latent": ("LATENT", {"tooltip": "Audio latent (optional)."}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "assemble"
    CATEGORY = "Intel-Arc/MiniMax"
    DESCRIPTION = "Merges video and audio latents into a single MiniMax H3 AV NestedTensor latent."

    def assemble(self, video_latent, audio_latent=None):
        video = video_latent.get("samples")
        if _is_nested_samples(video):
            video = video.tensors[0]

        def _is_empty(samples):
            if samples is None:
                return True
            try:
                if _is_nested_samples(samples):
                    return all(t.numel() == 0 for t in samples.tensors)
                return samples.numel() == 0
            except Exception:
                return False

        if audio_latent is None or _is_empty(audio_latent.get("samples")):
            return ({"samples": video},)
        audio = audio_latent["samples"]
        if _is_nested_samples(audio):
            audio = audio.tensors[1]
        return ({"samples": comfy.nested_tensor.NestedTensor((video, audio))},)


class MiniMaxH3LatentStitch:
    """Seamlessly blends and stitches two refined MiniMax H3 latent chunks into a continuous video."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "chunk_1": ("LATENT", {"tooltip": "Refined first half latent (2MP)."}),
                "chunk_2": ("LATENT", {"tooltip": "Refined second half latent (2MP, conditioned on chunk 1)."}),
                "overlap_latent_frames": (
                    "INT",
                    {
                        "default": 2,
                        "min": 0,
                        "max": 8,
                        "step": 1,
                        "tooltip": "Number of overlapping latent frames at the seam.",
                    },
                ),
                "blend_mode": (
                    ["seamless_handoff", "variance_preserving_fade", "cosine_fade", "linear_fade"],
                    {
                        "default": "variance_preserving_fade",
                        "tooltip": "variance_preserving_fade: cross-fades while maintaining contrast to prevent milky frames (recommended, hides VAE-decode pop). seamless_handoff: hard cut preserving Chunk 1, discards Chunk 2 boundary tokens.",
                    },
                ),
            },
            "optional": {
                "color_match": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Channel-wise DC offset alignment between chunks. Recommended True: independent per-chunk upscales drift in tone.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "stitch"
    CATEGORY = "Intel-Arc/MiniMax"
    DESCRIPTION = "Stitches two refined 2MP chunks with boundary-artifact elimination for seamless continuity."

    def stitch(self, chunk_1, chunk_2, overlap_latent_frames=2, blend_mode="seamless_handoff", color_match=False):
        v1, a1, is_av1 = _extract_video_audio(chunk_1)
        v2, a2, is_av2 = _extract_video_audio(chunk_2)
        overlap = _snap_overlap(overlap_latent_frames)
        if overlap != int(overlap_latent_frames):
            log.info(f"[MiniMaxH3LatentStitch] Snapped overlap {overlap_latent_frames} -> {overlap} for H3 cycle alignment")

        # 1. Color Matching: Channel-wise DC median offset alignment
        if color_match and overlap > 0 and v1.shape[2] >= overlap and v2.shape[2] >= overlap:
            c = v2.shape[1]
            ref_ov = v1[:, :, -overlap:].float()
            new_ov = v2[:, :, :overlap].float()
            p_ref = ref_ov.permute(0, 2, 3, 4, 1).reshape(-1, c)
            p_new = new_ov.permute(0, 2, 3, 4, 1).reshape(-1, c)
            dc = (p_new - p_ref).median(dim=0).values.clamp(-0.5, 0.5)
            v2 = v2 - dc.view(1, c, 1, 1, 1).to(v2.device, v2.dtype)
            log.info(f"[MiniMaxH3LatentStitch] Applied DC color matching (|dc| max={dc.abs().max().item():.4f})")

        # Audio alignment parameters
        if is_av1 and is_av2 and a1 is not None and a2 is not None:
            a1_t = a1.shape[-1]
            a2_t = a2.shape[-1]
            k_st = (v1.shape[2] + v2.shape[2] - overlap - 2) // 5 if (v1.shape[2] + v2.shape[2] - overlap - 2) % 5 == 0 else (v1.shape[2] + v2.shape[2] - overlap - 2) / 5
            expected_a_t = int(round((5 + 17 * k_st) * (5.0 / 3.0)))
            a_ov = a1_t + a2_t - expected_a_t
            if a_ov <= 0 or a_ov > min(a1_t, a2_t):
                a_ov = min(_audio_overlap_for_video_overlap(overlap), min(a1_t, a2_t))
        else:
            a_ov = 0

        # 2. Latent Stitching
        if overlap <= 0 or blend_mode == "seamless_handoff":
            # Seamless handoff: Keep Chunk 1 pristine up to its end, discard Chunk 2's initial
            # boundary frames (which suffer from the token-0 lightness artifact), and append Chunk 2
            # starting at frame 'overlap'. Motion flows seamlessly because Chunk 2 was conditioned on Chunk 1.
            v_stitched = torch.cat([v1, v2[:, :, overlap:]], dim=2)
            if is_av1 and is_av2 and a1 is not None and a2 is not None:
                a_stitched = torch.cat([a1, a2[..., a_ov:]], dim=-1)
            else:
                a_stitched = None

        elif blend_mode == "variance_preserving_fade":
            w = torch.linspace(0.0, 1.0, overlap + 2, device=v1.device, dtype=v1.dtype)[1:-1]
            w = 0.5 * (1.0 - torch.cos(w * math.pi)).view(1, 1, overlap, 1, 1)
            ov1 = v1[:, :, -overlap:]
            ov2 = v2[:, :, :overlap]
            raw_blend = (1.0 - w) * ov1 + w * ov2
            # Clamp std floor so flat patches don't blow up (1e-6 -> 1e-3).
            raw_std = raw_blend.std(dim=(-2, -1), keepdim=True).clamp_min(1e-3)
            target_std = (1.0 - w) * ov1.std(dim=(-2, -1), keepdim=True) + w * ov2.std(dim=(-2, -1), keepdim=True)
            target_mean = (1.0 - w) * ov1.mean(dim=(-2, -1), keepdim=True) + w * ov2.mean(dim=(-2, -1), keepdim=True)
            v_overlap = (raw_blend - raw_blend.mean(dim=(-2, -1), keepdim=True)) / raw_std * target_std + target_mean
            v_stitched = torch.cat([v1[:, :, :-overlap], v_overlap, v2[:, :, overlap:]], dim=2)

            if is_av1 and is_av2 and a1 is not None and a2 is not None and a_ov > 0:
                w_a = torch.linspace(0.0, 1.0, a_ov + 2, device=a1.device, dtype=a1.dtype)[1:-1]
                w_a = 0.5 * (1.0 - torch.cos(w_a * math.pi)).view(1, 1, 1, a_ov)
                a_overlap = (1.0 - w_a) * a1[..., -a_ov:] + w_a * a2[..., :a_ov]
                a_stitched = torch.cat([a1[..., :-a_ov], a_overlap, a2[..., a_ov:]], dim=-1)
            else:
                a_stitched = torch.cat([a1, a2], dim=-1) if (is_av1 and is_av2 and a1 is not None and a2 is not None) else None

        else:  # cosine_fade or linear_fade
            if blend_mode == "cosine_fade":
                w = torch.linspace(0.0, 1.0, overlap + 2, device=v1.device, dtype=v1.dtype)[1:-1]
                w = 0.5 * (1.0 - torch.cos(w * math.pi))
            else:
                w = torch.linspace(0.0, 1.0, overlap + 2, device=v1.device, dtype=v1.dtype)[1:-1]
            w = w.view(1, 1, overlap, 1, 1)

            v_overlap = (1.0 - w) * v1[:, :, -overlap:] + w * v2[:, :, :overlap]
            v_stitched = torch.cat([v1[:, :, :-overlap], v_overlap, v2[:, :, overlap:]], dim=2)

            if is_av1 and is_av2 and a1 is not None and a2 is not None and a_ov > 0:
                if blend_mode == "cosine_fade":
                    w_a = torch.linspace(0.0, 1.0, a_ov + 2, device=a1.device, dtype=a1.dtype)[1:-1]
                    w_a = 0.5 * (1.0 - torch.cos(w_a * math.pi)).view(1, 1, 1, a_ov)
                else:
                    w_a = torch.linspace(0.0, 1.0, a_ov + 2, device=a1.device, dtype=a1.dtype)[1:-1].view(1, 1, 1, a_ov)
                a_overlap = (1.0 - w_a) * a1[..., -a_ov:] + w_a * a2[..., :a_ov]
                a_stitched = torch.cat([a1[..., :-a_ov], a_overlap, a2[..., a_ov:]], dim=-1)
            else:
                a_stitched = torch.cat([a1, a2], dim=-1) if (is_av1 and is_av2 and a1 is not None and a2 is not None) else None

        log.info(
            f"[MiniMaxH3LatentStitch] Stitched Chunk 1 (T={v1.shape[2]}) and Chunk 2 (T={v2.shape[2]}) "
            f"-> Total T={v_stitched.shape[2]} with mode={blend_mode} (color_match={color_match})"
        )

        return (_wrap_latent(v_stitched, a_stitched, is_av1 and is_av2),)


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3LatentSplit": MiniMaxH3LatentSplit,
    "MiniMaxH3AttachContext": MiniMaxH3AttachContext,
    "MiniMaxH3AssembleAV": MiniMaxH3AssembleAV,
    "MiniMaxH3LatentStitch": MiniMaxH3LatentStitch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3LatentSplit": "MiniMax-H3 Latent Split (2MP Tiling)",
    "MiniMaxH3AttachContext": "MiniMax-H3 Attach Extend Context",
    "MiniMaxH3AssembleAV": "MiniMax-H3 Assemble AV Latent",
    "MiniMaxH3LatentStitch": "MiniMax-H3 Latent Stitch (Seamless)",
}
