"""Video Concat Extend for ComfyUI.

Joins a source clip and its AI-generated continuation (e.g. MiniMax-H3
VideoExtend output) into one clip, temporally:

- Images: drops the first ``overlap_frames`` of the continuation (they
  re-play the pinned/context frames) and concatenates on the batch dim.
- Audio: trims ``overlap_frames / fps`` seconds off the continuation head,
  resamples to a common rate, applies a short crossfade, concatenates.
- Color: optional Reinhard-style per-channel mean/std transfer so the
  continuation (which often drifts in tone) matches the source tail.

Pure tensor ops, no new dependencies.
"""

import logging
import math

import torch

log = logging.getLogger("ComfyUI-Nacholmo-xpu-vibeslop.VideoConcatExtend")


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
                    "tooltip": "Frames to drop from the start of the continuation (they re-play the pinned/context frames). ~24 ~= 1s at 24fps.",
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
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("images", "audio")
    FUNCTION = "concat"
    CATEGORY = "video"

    def concat(self, before_images, before_audio, after_images, after_audio,
               overlap_frames=24, fps=24.0, color_match=True, color_strength=1.0,
               color_ref_frames=8, audio_crossfade_seconds=0.1):
        if before_images.shape[1:] != after_images.shape[1:]:
            raise ValueError(
                f"VideoConcatExtend: frame shape mismatch {tuple(before_images.shape)} vs "
                f"{tuple(after_images.shape)} (HWC must match; resize one side first)."
            )
        n_after = after_images.shape[0]
        if overlap_frames >= n_after:
            raise ValueError(
                f"VideoConcatExtend: overlap_frames={overlap_frames} >= continuation length {n_after}."
            )
        after_t = after_images[overlap_frames:] if overlap_frames > 0 else after_images

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

        trim = int(round(overlap_frames / float(fps) * rate)) if overlap_frames > 0 else 0
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
            f"(trim {overlap_frames}) -> {tuple(out_images.shape)}, "
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
