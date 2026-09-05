"""Video Concat Extend for ComfyUI.

Joins a source clip and its AI-generated continuation (e.g. MiniMax-H3
VideoExtend output) into one clip, temporally:

- Overlap is **auto-detected**: the continuation head re-plays the pinned /
  context frames, so the node frame-matches the head against the source tail
  (grayscale, per-frame normalized so color drift can't fool it) and trims
  the duplicated replay. No manual tuning; ``overlap_frames`` is only a
  fallback when detection is uncertain.
- Images: concatenates source + trimmed continuation on the batch dim, so
  the output is (input + generated - replay) frames long.
- Audio: trims the same duration off the continuation head, resamples to a
  common rate, applies a short crossfade, concatenates.
- Color: optional Reinhard-style per-channel mean/std transfer so the
  continuation (which often drifts in tone) matches the source tail.

Pure tensor ops, no new dependencies.
"""

import logging
import math

import torch
import torch.nn.functional as F

log = logging.getLogger("ComfyUI-Nacholmo-xpu-vibeslop.VideoConcatExtend")


def _detect_overlap(before, after, max_overlap, probe_frames=8, size=64):
    """Return how many head frames of ``after`` duplicate the tail of ``before``.

    Grayscale, downscaled, per-frame zero-mean/unit-std so grade shifts don't
    affect the match. For each candidate k, compares after[0:W] against
    before[N-k:N-k+W] and takes the argmin MSE.
    """
    n_before = before.shape[0]
    n_after = after.shape[0]
    max_k = min(int(max_overlap), n_after - 1, n_before - 1)
    if max_k < 1:
        return 0, float("inf")

    def prep(x):
        g = x.float()
        if g.shape[-1] >= 3:
            g = 0.299 * g[..., 0] + 0.587 * g[..., 1] + 0.114 * g[..., 2]
        else:
            g = g[..., 0]
        g = F.interpolate(g[:, None], size=(size, size), mode="area")[:, 0]
        mu = g.mean(dim=(1, 2), keepdim=True)
        sd = g.std(dim=(1, 2), keepdim=True).clamp_min(1e-3)
        return (g - mu) / sd

    b = prep(before)
    best_k, best_err = 0, float("inf")
    for k in range(0, max_k + 1):
        w = min(probe_frames, n_after if k == 0 else k, n_before - k)
        if w < 1:
            continue
        a = prep(after[:w])
        # candidate k means after[0:k] duplicates before[N-k:N], so probe
        # after[0:W] against before[N-k:N-k+W]
        seg = b[n_before - k:n_before - k + w] if k > 0 else b[n_before - w:]
        err = float(((a - seg) ** 2).mean())
        if err < best_err:
            best_err, best_k = err, k
    return best_k, best_err


def _to_stereo(waveform):
    """[B, C, L] -> [B, 2, L]."""
    c = waveform.shape[1]
    if c == 2:
        return waveform
    if c == 1:
        return waveform.repeat(1, 2, 1)
    return waveform[:, :2, :]


def _resample(waveform, src_rate, dst_rate):
    if src_rate == dst_rate:
        return waveform
    try:
        import torchaudio
        resampled = torchaudio.functional.resample(waveform, src_rate, dst_rate)
        # resample pads with garbage past the true length; trim exactly.
        expect = math.ceil(waveform.shape[-1] * dst_rate / src_rate)
        return resampled[..., :expect].contiguous()
    except Exception as e:
        raise RuntimeError(
            f"VideoConcatExtend: need torchaudio to resample {src_rate} -> {dst_rate} Hz: {e}"
        ) from e


def _color_transfer(source, ref, strength=1.0, eps=1e-3):
    """Match per-channel mean/std of ``source`` [N,H,W,C] to ``ref`` [M,H,W,C]."""
    s = source.float()
    r = ref.float()
    mu_s = s.mean(dim=(0, 1, 2), keepdim=True)
    mu_r = r.mean(dim=(0, 1, 2), keepdim=True)
    std_s = s.std(dim=(0, 1, 2), keepdim=True).clamp_min(eps)
    std_r = r.std(dim=(0, 1, 2), keepdim=True)
    matched = (s - mu_s) / std_s * std_r + mu_r
    out = s + (matched - s) * float(strength)
    return out.clamp(0.0, 1.0).to(source.dtype)


class VideoConcatExtend:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "before_images": ("IMAGE", {"tooltip": "Source clip frames (full length, kept in full)."}),
                "before_audio": ("AUDIO", {"tooltip": "Source clip audio."}),
                "after_images": ("IMAGE", {"tooltip": "Continuation frames (head trimmed by overlap_frames)."}),
                "after_audio": ("AUDIO", {"tooltip": "Continuation audio."}),
                "overlap_frames": ("INT", {
                    "default": 24, "min": 0, "max": 4096, "step": 1,
                    "tooltip": "Manual head trim, used only when auto_overlap is off or detection is uncertain.",
                }),
                "fps": ("FLOAT", {
                    "default": 24.0, "min": 1.0, "max": 120.0, "step": 0.01,
                    "tooltip": "Clip frame rate; converts overlap_frames to the audio trim length.",
                }),
                "color_match": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Match continuation colors to the source tail (mean/std transfer).",
                }),
                "color_strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Blend between raw continuation (0) and fully matched (1).",
                }),
                "color_ref_frames": ("INT", {
                    "default": 8, "min": 1, "max": 256, "step": 1,
                    "tooltip": "Tail frames of the source used as the color reference.",
                }),
                "audio_crossfade_seconds": ("FLOAT", {
                    "default": 0.1, "min": 0.0, "max": 5.0, "step": 0.01,
                    "tooltip": "Linear crossfade at the audio joint. 0 = hard cut.",
                }),
                "auto_overlap": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Frame-match the continuation head against the source tail and trim the replayed overlap automatically. Falls back to overlap_frames when uncertain.",
                }),
                "max_overlap_frames": ("INT", {
                    "default": 48, "min": 1, "max": 4096, "step": 1,
                    "tooltip": "Upper bound for auto-detected overlap (~2s at 24fps).",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("images", "audio")
    FUNCTION = "concat"
    CATEGORY = "video"

    def concat(self, before_images, before_audio, after_images, after_audio,
               overlap_frames=24, fps=24.0, color_match=True, color_strength=1.0,
               color_ref_frames=8, audio_crossfade_seconds=0.1,
               auto_overlap=True, max_overlap_frames=48):
        if before_images.shape[1:] != after_images.shape[1:]:
            raise ValueError(
                f"VideoConcatExtend: frame shape mismatch {tuple(before_images.shape)} vs "
                f"{tuple(after_images.shape)} (HWC must match; resize one side first)."
            )
        n_after = after_images.shape[0]
        overlap = 0
        if auto_overlap:
            detected, err = _detect_overlap(before_images, after_images, max_overlap_frames)
            # Normalized-MSE of unrelated frames sits ~2.0; above ~1.0 the
            # match is not trustworthy (e.g. hard cut), use the manual value.
            if err <= 1.0:
                overlap = detected
                log.info(f"[VideoConcatExtend] Auto overlap: {overlap} frames (mse {err:.3f})")
            else:
                overlap = int(overlap_frames)
                log.info(
                    f"[VideoConcatExtend] Detection uncertain (mse {err:.3f}), "
                    f"falling back to manual overlap_frames={overlap}"
                )
        else:
            overlap = int(overlap_frames)
        if overlap >= n_after:
            raise ValueError(
                f"VideoConcatExtend: overlap={overlap} >= continuation length {n_after}."
            )
        after_t = after_images[overlap:] if overlap > 0 else after_images

        if color_match and float(color_strength) > 0.0:
            ref = before_images[-int(color_ref_frames):] if before_images.shape[0] else before_images
            after_t = _color_transfer(after_t, ref, strength=color_strength)
            log.info("[VideoConcatExtend] Applied color transfer to continuation")

        out_images = torch.cat([before_images, after_t], dim=0)

        # --- audio ---
        w1 = before_audio["waveform"]
        w2 = after_audio["waveform"]
        r1 = int(before_audio["sample_rate"])
        r2 = int(after_audio["sample_rate"])
        if w1.shape[0] != w2.shape[0]:
            raise ValueError(
                f"VideoConcatExtend: audio batch mismatch {w1.shape[0]} vs {w2.shape[0]}."
            )
        rate = max(r1, r2)
        w1 = _to_stereo(_resample(w1, r1, rate))
        w2 = _to_stereo(_resample(w2, r2, rate))

        trim = int(round(overlap / float(fps) * rate)) if overlap > 0 else 0
        if trim >= w2.shape[-1]:
            raise ValueError(
                f"VideoConcatExtend: audio trim {trim} samples >= continuation audio {w2.shape[-1]}."
            )
        w2t = w2[..., trim:] if trim > 0 else w2

        cf = int(round(float(audio_crossfade_seconds) * rate))
        cf = max(0, min(cf, w1.shape[-1], w2t.shape[-1]))
        if cf > 1:
            ramp = torch.linspace(0.0, 1.0, cf, device=w1.device, dtype=w1.dtype).view(1, 1, cf)
            head = w1[..., :-cf]
            tail = w2t[..., cf:]
            mix = (1.0 - ramp) * w1[..., -cf:] + ramp * w2t[..., :cf]
            out_w = torch.cat([head, mix, tail], dim=-1)
        else:
            out_w = torch.cat([w1, w2t], dim=-1)

        log.info(
            f"[VideoConcatExtend] {tuple(before_images.shape)} + {tuple(after_images.shape)} "
            f"(trim {overlap}) -> {tuple(out_images.shape)}, "
            f"audio {w1.shape[-1]/rate:.2f}s + {w2t.shape[-1]/rate:.2f}s @ {rate}Hz"
        )
        return (out_images, {"waveform": out_w.contiguous(), "sample_rate": rate})


NODE_CLASS_MAPPINGS = {
    "VideoConcatExtend": VideoConcatExtend,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoConcatExtend": "Video Concat Extend (A/V + Color)",
}

__all__ = ["VideoConcatExtend", "NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
