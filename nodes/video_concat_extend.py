"""Video Concat Extend for ComfyUI.

Joins a source clip and its AI-generated continuation (e.g. MiniMax-H3
VideoExtend output) into one clip, temporally:

- Overlap is **auto-detected**: the continuation head re-plays the pinned /
  context frames, so the node frame-matches the head against the source tail
  (grayscale, per-frame normalized so color drift can't fool it) and trims
  the duplicated replay. No manual tuning; ``overlap_frames`` is only a
  fallback when detection is uncertain. Detection is coarse-to-fine (64px
  scan, 128px verify of the top-2 candidates) with a best-vs-runner-up
  margin check and flat-scene guard, plus sample-accurate audio
  cross-correlation refinement of the trim point.
- Images: concatenates source + trimmed continuation on the batch dim, so
  the output is (input + generated - replay) frames long. The joint can use
  a centered true-overlap dissolve (no duplicated action) with linear,
  cosine, or variance-preserving fades.
- Audio: trims the same duration off the continuation head, resamples to a
  common rate, applies an equal-power crossfade, concatenates. Audio inputs
  are optional (video-only chains pass through with ``None`` audio).
- Color: optional mean/std transfer (RGB, luma-only, or exposure gain) so
  the continuation (which often drifts in tone) matches the source tail,
  with an optional temporal decay so the correction relaxes away from the
  seam instead of tinting the whole continuation.

Pure tensor ops, no new dependencies.
"""

import logging
import math

import torch
import torch.nn.functional as F

import comfy.model_management

log = logging.getLogger("ComfyUI-Nacholmo-xpu-vibeslop.VideoConcatExtend")

_LUMA_W = (0.299, 0.587, 0.114)
# Raw grayscale std below this (0-1 scale) means a flat tail (sky, fade to
# black) where normalized frame matching just amplifies noise.
_FLAT_STD = 0.02


def _to_gray(x):
    """[N,H,W,C] float -> [N,H,W] luma."""
    if x.shape[-1] >= 3:
        return (_LUMA_W[0] * x[..., 0] + _LUMA_W[1] * x[..., 1] + _LUMA_W[2] * x[..., 2])
    return x[..., 0]


def _prep_gray(x, size):
    """Downscale + per-frame zero-mean/unit-std normalize. Returns [N,S,S]."""
    g = _to_gray(x.float())
    g = F.interpolate(g[:, None], size=(size, size), mode="area")[:, 0]
    mu = g.mean(dim=(1, 2), keepdim=True)
    sd = g.std(dim=(1, 2), keepdim=True).clamp_min(1e-3)
    return (g - mu) / sd


def _detect_overlap(before, after, max_overlap, probe_frames=8, size=64, fine_size=128):
    """Return ``(best_k, best_err, margin, flat)`` for the replayed head.

    Coarse scan at ``size`` px over all ``k``, then fine verify of the top-2
    candidates at ``fine_size`` px. ``margin`` is best-vs-runner-up MSE gap
    (bigger = more confident). ``flat`` flags a low-texture tail where the
    match can't be trusted.
    """
    n_before = before.shape[0]
    n_after = after.shape[0]
    max_k = min(int(max_overlap), n_after - 1, n_before - 1)
    if max_k < 1:
        return 0, float("inf"), 0.0, False

    before_f = before.float()
    after_f = after.float()

    # Flat-scene guard on raw (pre-normalization) tail texture. Plain tensor
    # ops here (no no_grad/inference_mode per suite policy).
    raw = _to_gray(before_f[-min(max_k + int(probe_frames), n_before):])
    raw = F.interpolate(raw[:, None], size=(size, size), mode="area")[:, 0]
    flat = bool(float(raw.std(dim=(1, 2)).mean()) < _FLAT_STD)

    b = _prep_gray(before_f, size)
    a_full = _prep_gray(after_f[:min(int(probe_frames), n_after)], size)

    coarse = {}
    for k in range(0, max_k + 1):
        comfy.model_management.throw_exception_if_processing_interrupted()
        w = min(int(probe_frames), n_after if k == 0 else k, n_before - k)
        if w < 1:
            continue
        a = a_full[:w]
        seg = b[n_before - k:n_before - k + w] if k > 0 else b[n_before - w:]
        coarse[k] = float(((a - seg) ** 2).mean())

    if not coarse:
        return 0, float("inf"), 0.0, flat

    ranked = sorted(coarse.items(), key=lambda kv: kv[1])
    top = [k for k, _ in ranked[:2]]

    # Fine verify top-2 at higher resolution.
    b_fine = _prep_gray(before_f, fine_size)
    a_fine_full = _prep_gray(after_f[:min(int(probe_frames), n_after)], fine_size)
    fine = {}
    for k in top:
        comfy.model_management.throw_exception_if_processing_interrupted()
        w = min(int(probe_frames), n_after if k == 0 else k, n_before - k)
        if w < 1:
            fine[k] = coarse[k]
            continue
        a = a_fine_full[:w]
        seg = b_fine[n_before - k:n_before - k + w] if k > 0 else b_fine[n_before - w:]
        fine[k] = float(((a - seg) ** 2).mean())

    best_k = min(fine, key=lambda k: fine[k])
    best_err = fine[best_k]
    if len(fine) > 1:
        runner = min(e for k, e in fine.items() if k != best_k)
        margin = runner - best_err
    elif len(ranked) > 1:
        margin = ranked[1][1] - ranked[0][1]
    else:
        margin = 0.0
    return best_k, best_err, margin, flat


def _mono_mix(waveform):
    """[B,C,L] -> [B,1,L] mono mix for correlation."""
    return waveform.float().mean(dim=1, keepdim=True)


def _refine_audio_trim(w1_mono, w2_mono, rate, overlap_samples, window_samples):
    """Nudge the audio trim by cross-correlating the replayed overlap region.

    The continuation head replays ``overlap_samples`` of audio, so the
    ``w1`` tail should match the ``w2`` head there. Tries candidate overlap
    lengths in ``[L-W, L+W]`` (normalized cross-correlation of
    ``w1[-L':]`` vs ``w2[:L']`` at ~4kHz for speed) and returns the delta
    in full-rate samples (0 when the overlap is too short to correlate).
    Positive delta = replay is longer than nominal = trim more.
    """
    try:
        n1 = w1_mono.shape[-1]
        n2 = w2_mono.shape[-1]
        L = int(min(overlap_samples, n1, n2))
        W = int(min(window_samples, L - int(rate * 0.05)))
        if L < int(rate * 0.1) or W <= 0:
            return 0
        target_ds = 4000
        factor = max(1, int(rate // target_ds))
        t1 = w1_mono[..., n1 - L - W:n1] if n1 >= L + W else w1_mono
        t2 = w2_mono[..., :L + W] if n2 >= L + W else w2_mono
        if factor > 1:
            t1 = F.avg_pool1d(t1, kernel_size=factor, stride=factor)
            t2 = F.avg_pool1d(t2, kernel_size=factor, stride=factor)
        # Candidate overlap length Lp_ds in [L-W, L+W] (downsampled units):
        # w1 tail Lp <-> w2 head Lp.
        L_ds = L // factor
        W_ds = max(1, W // factor)
        best_d, best_corr, corr0 = 0, float("-inf"), None
        for d_ds in range(-W_ds, W_ds + 1):
            comfy.model_management.throw_exception_if_processing_interrupted()
            Lp = L_ds + d_ds
            if Lp < 1 or Lp > t1.shape[-1] or Lp > t2.shape[-1]:
                continue
            a = t1[..., t1.shape[-1] - Lp:]
            b = t2[..., :Lp]
            a_z = a - a.mean(dim=-1, keepdim=True)
            b_z = b - b.mean(dim=-1, keepdim=True)
            denom = float((a_z ** 2).sum().sqrt() * (b_z ** 2).sum().sqrt()) + 1e-8
            corr = float(((a_z * b_z).sum() / denom).mean())
            if d_ds == 0:
                corr0 = corr
            if corr > best_corr:
                best_corr, best_d = corr, d_ds
        # Hysteresis: shorter windows can score equal-or-higher by chance, so
        # only nudge when clearly better than nominal (else we'd add seam
        # error instead of removing it).
        if corr0 is None or best_corr <= corr0 + 0.01:
            return 0
        return int(best_d * factor)
    except Exception as e:
        log.info(f"[VideoConcatExtend] Audio align skipped: {e}")
        return 0


def _to_stereo(waveform, mode="stereo"):
    """[B, C, L] -> [B, 2, L], or preserved channels with ``preserve``."""
    c = waveform.shape[1]
    if mode == "preserve":
        return waveform
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


def _color_transfer(source, ref, strength=1.0, mode="luma_only", std_max_ratio=2.0,
                    decay_weights=None, eps=1e-3):
    """Match ``source`` [M,H,W,C] tone to ``ref`` [R,H,W,C].

    - ``rgb``: per-channel Reinhard mean/std (legacy behavior).
    - ``luma_only``: match luma mean/std, apply the delta equally to all
      channels (preserves chroma / skin tones).
    - ``exposure_gain``: single scalar gain from luma means (no contrast
      change; safest for mild drift).
    ``std_max_ratio`` clamps contrast change to ``[1/r, r]`` so flat refs
    can't collapse detail. ``decay_weights`` is an optional ``[M]`` vector
    scaling strength per continuation frame (1 at the seam).
    """
    s = source.float()
    r = ref.float()
    m = s.shape[0]
    strength = float(strength)
    if decay_weights is not None:
        w = decay_weights.to(device=s.device, dtype=torch.float32).view(m, 1, 1, 1)
    else:
        w = None

    def _blend(out_matched):
        if w is None:
            out = s + (out_matched - s) * strength
        else:
            out = s + (out_matched - s) * (strength * w)
        return out.clamp(0.0, 1.0).to(source.dtype)

    if mode == "exposure_gain":
        mu_s = _to_gray(s).mean().clamp_min(eps)
        mu_r = _to_gray(r).mean().clamp_min(eps)
        gain = (mu_r / mu_s).clamp(1.0 / std_max_ratio, std_max_ratio)
        return _blend(s * gain)

    if mode == "luma_only":
        ls = _to_gray(s)
        lr = _to_gray(r)
        mu_s, mu_r = ls.mean(), lr.mean()
        std_s = ls.std().clamp_min(eps)
        std_r = lr.std()
        ratio = (std_r / std_s).clamp(1.0 / std_max_ratio, std_max_ratio)
        matched_luma = (ls - mu_s) * ratio + mu_r
        delta = (matched_luma - ls)[..., None]
        return _blend(s + delta)

    # rgb (legacy Reinhard, with contrast clamp)
    mu_s = s.mean(dim=(0, 1, 2), keepdim=True)
    mu_r = r.mean(dim=(0, 1, 2), keepdim=True)
    std_s = s.std(dim=(0, 1, 2), keepdim=True).clamp_min(eps)
    std_r = r.std(dim=(0, 1, 2), keepdim=True)
    ratio = (std_r / std_s).clamp(1.0 / std_max_ratio, std_max_ratio)
    matched = (s - mu_s) * ratio + mu_r
    return _blend(matched)


def _blend_ramp(nb, mode, device, dtype):
    t = torch.linspace(0.0, 1.0, nb, device=device, dtype=torch.float32)
    if mode in ("cosine", "variance_preserving"):
        t = 0.5 * (1.0 - torch.cos(t * math.pi))
    # "linear" keeps t as-is; variance_preserving reuses the cosine ramp
    # for the raw mix, then restores contrast (see _blend_pair).
    return t.to(dtype)


def _blend_pair(ov1, ov2, mode):
    """Blend two [nb,H,W,C] overlap windows into [nb,H,W,C]."""
    nb = ov1.shape[0]
    ramp = _blend_ramp(nb, mode, ov1.device, torch.float32).view(nb, 1, 1, 1)
    a = ov1.float()
    b = ov2.float()
    raw = (1.0 - ramp) * a + ramp * b
    if mode == "variance_preserving" and nb > 0:
        # Port of MiniMaxH3LatentStitch variance-preserving fade: keep the
        # interpolated mean/std instead of letting the mix go milky.
        raw_std = raw.std(dim=(1, 2), keepdim=True).clamp_min(1e-3)
        target_std = (1.0 - ramp) * a.std(dim=(1, 2), keepdim=True) + ramp * b.std(dim=(1, 2), keepdim=True)
        target_mean = (1.0 - ramp) * a.mean(dim=(1, 2), keepdim=True) + ramp * b.mean(dim=(1, 2), keepdim=True)
        raw = (raw - raw.mean(dim=(1, 2), keepdim=True)) / raw_std * target_std + target_mean
    return raw.to(ov1.dtype)


def _fade_gains(length, curve, device, dtype):
    t = torch.linspace(0.0, 1.0, length, device=device, dtype=torch.float32)
    if curve == "equal_power":
        fade_out = torch.cos(t * math.pi / 2.0)
        fade_in = torch.sin(t * math.pi / 2.0)
    else:
        fade_out, fade_in = 1.0 - t, t
    return fade_out.to(dtype), fade_in.to(dtype)


class VideoConcatExtend:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "before_images": ("IMAGE", {"tooltip": "Source clip frames (full length, kept in full)."}),
                "after_images": ("IMAGE", {"tooltip": "Continuation frames (head trimmed by overlap_frames)."}),
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
                    "default": 0.25, "min": 0.0, "max": 5.0, "step": 0.01,
                    "tooltip": "Equal-power crossfade at the audio joint. 0 = hard cut.",
                }),
                "auto_overlap": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Frame-match the continuation head against the source tail and trim the replayed overlap automatically. Falls back to overlap_frames when uncertain.",
                }),
                "max_overlap_frames": ("INT", {
                    "default": 48, "min": 1, "max": 4096, "step": 1,
                    "tooltip": "Upper bound for auto-detected overlap (~2s at 24fps).",
                }),
                "video_blend_frames": ("INT", {
                    "default": 6, "min": 0, "max": 64, "step": 1,
                    "tooltip": "Dissolve across the joint (frames). Hides residual pops from pose/re-lighting jumps. 0 = hard cut.",
                }),
                # --- P1: detection confidence (appended; old workflows keep defaults) ---
                "overlap_threshold": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 4.0, "step": 0.05,
                    "tooltip": "Max normalized-MSE for trusting auto-detection. Above this, falls back to overlap_frames.",
                }),
                "overlap_margin": ("FLOAT", {
                    "default": 0.10, "min": 0.0, "max": 2.0, "step": 0.01,
                    "tooltip": "Min best-vs-runner-up MSE gap for trusting auto-detection. Below this the match is ambiguous.",
                }),
                # --- P3: color ---
                "color_mode": (["luma_only", "rgb", "exposure_gain"], {
                    "default": "luma_only",
                    "tooltip": "luma_only: match brightness/contrast, keep chroma (best for skin). rgb: legacy per-channel. exposure_gain: single gain, safest.",
                }),
                "color_decay_seconds": ("FLOAT", {
                    "default": 2.0, "min": 0.0, "max": 30.0, "step": 0.1,
                    "tooltip": "Relax the correction away from the seam over this many seconds (floor 25%). 0 = legacy global correction.",
                }),
                # --- P2: video joint ---
                "video_blend_mode": (["cosine", "linear", "variance_preserving"], {
                    "default": "cosine",
                    "tooltip": "Fade curve for the dissolve. variance_preserving keeps contrast (no milky ghost) on re-lit joints.",
                }),
                "video_blend_position": (["centered", "tail_only"], {
                    "default": "centered",
                    "tooltip": "centered: true-overlap dissolve (length N+M-nb, no duplicated action). tail_only: legacy length-preserving overwrite.",
                }),
                # --- P4: audio ---
                "audio_fade_curve": (["equal_power", "linear"], {
                    "default": "equal_power",
                    "tooltip": "equal_power keeps loudness constant through the fade (no dip/click).",
                }),
                "audio_align_window_frames": ("INT", {
                    "default": 2, "min": 0, "max": 24, "step": 1,
                    "tooltip": "Sample-accurate trim refinement search window (±frames) via overlap-region correlation. 0 = off.",
                }),
                "loudness_match": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Match continuation head RMS to source tail RMS (clamped 0.5x-2x). Enable when extends pop louder/quieter.",
                }),
                "match_size": (["disabled", "bicubic"], {
                    "default": "disabled",
                    "tooltip": "Resize continuation frames to the source HxW when they differ. disabled = legacy strict error.",
                }),
                "stereo_mode": (["stereo", "preserve"], {
                    "default": "stereo",
                    "tooltip": "stereo: legacy mono-duplicate / multichannel-truncate to stereo. preserve: keep channels as-is (must match).",
                }),
            },
            "optional": {
                "before_audio": ("AUDIO", {"tooltip": "Source clip audio (optional; video-only chains pass None through)."}),
                "after_audio": ("AUDIO", {"tooltip": "Continuation audio (optional)."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "INT", "FLOAT")
    RETURN_NAMES = ("images", "audio", "overlap_detected", "overlap_mse")
    FUNCTION = "concat"
    CATEGORY = "video"

    def concat(self, before_images, after_images,
               overlap_frames=24, fps=24.0, color_match=True, color_strength=1.0,
               color_ref_frames=8, audio_crossfade_seconds=0.25,
               auto_overlap=True, max_overlap_frames=48, video_blend_frames=6,
               overlap_threshold=1.0, overlap_margin=0.10,
               color_mode="luma_only", color_decay_seconds=2.0,
               video_blend_mode="cosine", video_blend_position="centered",
               audio_fade_curve="equal_power", audio_align_window_frames=2,
               loudness_match=False, match_size="disabled", stereo_mode="stereo",
               before_audio=None, after_audio=None):
        # --- device/dtype unify (forgiving concat across CPU/XPU mixes) ---
        target_device = before_images.device
        target_dtype = before_images.dtype
        if after_images.device != target_device or after_images.dtype != target_dtype:
            after_images = after_images.to(device=target_device, dtype=target_dtype)
            log.info(f"[VideoConcatExtend] Moved continuation to {target_device}/{target_dtype}")

        if before_images.shape[1:] != after_images.shape[1:]:
            if match_size == "bicubic" and before_images.shape[-1] == after_images.shape[-1]:
                hb, wb = before_images.shape[1], before_images.shape[2]
                a = after_images.movedim(-1, -3).float()
                a = F.interpolate(a, size=(hb, wb), mode="bicubic", align_corners=False, antialias=True)
                after_images = a.movedim(-3, -1).to(dtype=target_dtype)
                log.info(f"[VideoConcatExtend] Resized continuation to {hb}x{wb} (bicubic)")
            else:
                raise ValueError(
                    f"VideoConcatExtend: frame shape mismatch {tuple(before_images.shape)} vs "
                    f"{tuple(after_images.shape)} (HWC must match; enable match_size=bicubic to auto-resize)."
                )
        n_before = before_images.shape[0]
        n_after = after_images.shape[0]
        if n_before < 1 or n_after < 1:
            raise ValueError("VideoConcatExtend: need at least 1 frame on each side.")

        # --- P1: overlap detection ---
        overlap = 0
        det_err = float("inf")
        if auto_overlap:
            detected, err, margin, flat = _detect_overlap(
                before_images, after_images, max_overlap_frames)
            det_err = err
            confident = (err <= float(overlap_threshold)
                         and margin >= float(overlap_margin)
                         and not flat)
            if confident:
                overlap = detected
                log.info(f"[VideoConcatExtend] Auto overlap: {overlap} frames (mse {err:.3f}, margin {margin:.3f})")
            else:
                overlap = int(overlap_frames)
                reason = "flat tail" if flat else f"mse {err:.3f}/margin {margin:.3f}"
                log.info(
                    f"[VideoConcatExtend] Detection uncertain ({reason}), "
                    f"falling back to manual overlap_frames={overlap}"
                )
        else:
            overlap = int(overlap_frames)
        if overlap >= n_after:
            raise ValueError(
                f"VideoConcatExtend: overlap={overlap} >= continuation length {n_after}."
            )
        after_t = after_images[overlap:] if overlap > 0 else after_images

        # --- P3: color with temporal decay ---
        if color_match and float(color_strength) > 0.0 and after_t.shape[0] > 0:
            ref = before_images[-int(color_ref_frames):] if n_before else before_images
            m = after_t.shape[0]
            weights = None
            decay_s = float(color_decay_seconds)
            if decay_s > 0 and m > 1:
                decay_frames = max(1, int(round(decay_s * float(fps))))
                idx = torch.arange(m, device=after_t.device, dtype=torch.float32)
                w = 1.0 - 0.75 * (idx / max(1, min(decay_frames, m - 1))).clamp(0.0, 1.0)
                weights = w
            after_t = _color_transfer(after_t, ref, strength=color_strength,
                                      mode=color_mode, decay_weights=weights)
            log.info(f"[VideoConcatExtend] Applied color transfer ({color_mode}, decay {decay_s}s)")

        # --- P2: video joint ---
        if target_device != after_t.device or target_dtype != after_t.dtype:
            after_t = after_t.to(device=target_device, dtype=target_dtype)
        nb = max(0, min(int(video_blend_frames), n_before, after_t.shape[0]))
        if nb > 1:
            if video_blend_position == "centered" and n_before - nb >= 0 and after_t.shape[0] - nb >= 0:
                # True-overlap dissolve: blend before-tail with after-head into
                # nb frames counted once (length N+M-nb, like the audio fade).
                ov1 = before_images[n_before - nb:]
                ov2 = after_t[:nb]
                joint = _blend_pair(ov1, ov2, video_blend_mode)
                out_images = torch.cat([before_images[:n_before - nb], joint, after_t[nb:]], dim=0)
                log.info(f"[VideoConcatExtend] Centered {video_blend_mode} dissolve over {nb} frames")
            else:
                # Legacy tail_only: length-preserving overwrite of before tail.
                out_images = torch.cat([before_images, after_t], dim=0)
                n_b = before_images.shape[0]
                joint = _blend_pair(out_images[n_b - nb:n_b],
                                    out_images[n_b:n_b + nb], video_blend_mode)
                out_images = out_images.clone()
                out_images[n_b - nb:n_b] = joint.to(out_images.dtype)
                log.info(f"[VideoConcatExtend] Tail {video_blend_mode} dissolve over {nb} frames")
        else:
            out_images = torch.cat([before_images, after_t], dim=0)
        log.info(
            f"[VideoConcatExtend] before={n_before} frames, after(trimmed)={after_t.shape[0]} frames"
        )

        # --- P4: audio (optional, forgiving) ---
        if before_audio is None and after_audio is None:
            log.info("[VideoConcatExtend] No audio inputs; passing video only (audio=None)")
            return (out_images, None, int(overlap), float(det_err))
        if before_audio is None:
            log.info("[VideoConcatExtend] Only continuation audio present; passing through")
            return (out_images, after_audio, int(overlap), float(det_err))
        if after_audio is None:
            log.info("[VideoConcatExtend] Only source audio present; passing through")
            return (out_images, before_audio, int(overlap), float(det_err))

        w1 = before_audio["waveform"]
        w2 = after_audio["waveform"]
        r1 = int(before_audio["sample_rate"])
        r2 = int(after_audio["sample_rate"])
        if w1.shape[0] != w2.shape[0]:
            if w1.shape[0] == 1:
                w1 = w1.repeat(w2.shape[0], 1, 1)
                log.info("[VideoConcatExtend] Broadcast source audio batch to match")
            elif w2.shape[0] == 1:
                w2 = w2.repeat(w1.shape[0], 1, 1)
                log.info("[VideoConcatExtend] Broadcast continuation audio batch to match")
            else:
                raise ValueError(
                    f"VideoConcatExtend: audio batch mismatch {w1.shape[0]} vs {w2.shape[0]}."
                )
        rate = max(r1, r2)
        a_device, a_dtype = w1.device, w1.dtype
        w1 = _to_stereo(_resample(w1, r1, rate), stereo_mode).to(device=a_device, dtype=a_dtype)
        w2 = _to_stereo(_resample(w2, r2, rate), stereo_mode).to(device=a_device, dtype=a_dtype)
        if stereo_mode == "preserve" and w1.shape[1] != w2.shape[1]:
            raise ValueError(
                f"VideoConcatExtend: audio channel mismatch {w1.shape[1]} vs {w2.shape[1]} "
                "(stereo_mode=preserve; use stereo to force stereo)."
            )

        trim = int(round(overlap / float(fps) * rate)) if overlap > 0 else 0
        # Sample-accurate refinement via overlap-region correlation.
        if overlap > 0 and int(audio_align_window_frames) > 0:
            overlap_samples = int(round(overlap / float(fps) * rate))
            window_samples = int(round(int(audio_align_window_frames) / float(fps) * rate))
            delta = _refine_audio_trim(_mono_mix(w1), _mono_mix(w2), rate,
                                       overlap_samples, window_samples)
            if delta:
                log.info(f"[VideoConcatExtend] Audio align nudged trim by {delta} samples "
                         f"({delta / rate * 1000.0:.1f}ms)")
                trim += delta
        trim = max(0, trim)
        if trim >= w2.shape[-1]:
            raise ValueError(
                f"VideoConcatExtend: audio trim {trim} samples >= continuation audio {w2.shape[-1]}."
            )
        w2t = w2[..., trim:] if trim > 0 else w2

        if loudness_match and w1.shape[-1] > 0 and w2t.shape[-1] > 0:
            seg = min(rate, w1.shape[-1], w2t.shape[-1])
            rms1 = float((w1[..., -seg:].float() ** 2).mean().sqrt()) + 1e-8
            rms2 = float((w2t[..., :seg].float() ** 2).mean().sqrt()) + 1e-8
            gain = max(0.5, min(2.0, rms1 / rms2))
            w2t = (w2t.float() * gain).to(w2t.dtype)
            log.info(f"[VideoConcatExtend] Loudness matched continuation x{gain:.3f}")

        cf = int(round(float(audio_crossfade_seconds) * rate))
        cf = max(0, min(cf, w1.shape[-1], w2t.shape[-1]))
        if cf > 1:
            fade_out, fade_in = _fade_gains(cf, audio_fade_curve, w1.device, w1.dtype)
            fade_out = fade_out.view(1, 1, cf)
            fade_in = fade_in.view(1, 1, cf)
            head = w1[..., :-cf]
            tail = w2t[..., cf:]
            mix = fade_out * w1[..., -cf:] + fade_in * w2t[..., :cf]
            out_w = torch.cat([head, mix, tail], dim=-1)
        else:
            out_w = torch.cat([w1, w2t], dim=-1)

        log.info(
            f"[VideoConcatExtend] {tuple(before_images.shape)} + {tuple(after_images.shape)} "
            f"(trim {overlap}) -> {tuple(out_images.shape)}, "
            f"audio {w1.shape[-1]/rate:.2f}s + {w2t.shape[-1]/rate:.2f}s @ {rate}Hz"
        )
        return (out_images, {"waveform": out_w.contiguous(), "sample_rate": rate},
                int(overlap), float(det_err))


NODE_CLASS_MAPPINGS = {
    "VideoConcatExtend": VideoConcatExtend,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoConcatExtend": "Video Concat Extend (A/V + Color)",
}

__all__ = ["VideoConcatExtend", "NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
