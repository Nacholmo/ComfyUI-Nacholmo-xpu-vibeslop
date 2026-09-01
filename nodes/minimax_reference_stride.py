"""MiniMax H3 Reference to Video — VRAM-saver variant with frame striding.

Vanilla node: comfy_extras/nodes_minimax_h3.py:239 MiniMaxH3ReferenceToVideo

Problem:
  Reference videos dominate the packed DiT sequence length S in
  comfy/ldm/minimax/model.py:321 PackedLayout.
  Each video ref adds vt*frame_rows tokens where
    vt ~ latent_t = vae.encode(frames).shape[2]
    frame_rows = (latent_h//2)*(latent_w//2)
  For a 5s 1344x768 ref (119 frames -> vt=32 -> 32k tokens) the sequence
  doubles (target ~37k + ref 32k ~69k) -> OOM on low VRAM / XPU.

Hypothesis verified:
  The vanilla node does NOT require dense frames. It already discards frames:
    - truncates if ref longer than generation: frames[:frame_count]
    - snaps down to valid 17k+5 length: while n%17!=5: n-=1
  Any vt that satisfies the VAE shape is accepted by PackedLayout
  (comfy/ldm/minimax/model.py:390-405, model_base.py:2205-2212).

Solution:
  Insert a uniform stride on the reference video *before* VAE encoding:

    frames = frames[::stride]   # stride=2 -> half frames

  then re-snap to 17k+5 and encode. vt and therefore token count shrink
  ~1/stride. DiT memory scales with S (flash) / S² (naive) -> ~22% / ~40%
  saving for stride=2 on a single 5s ref.

  Two Qwen handling modes are exposed:
    - "strided":  Qwen sees the same strided frames (max saving)
    - "full":     Qwen sees the pre-stride frames at 2 fps (better identity,
                 marginal extra text tokens)

  An optional post-VAE latent slicing mode is also available for cases where
  the 3D VAE's temporal kernel prefers contiguous input.

  This file is intentionally self-contained (copies adapt_canvas/_resize/etc.)
  so it does not depend on vanilla internals staying stable.
"""

import math

import torch
try:
    import torchaudio
    _HAS_TORCHAUDIO = True
except Exception as _e:
    torchaudio = None
    _HAS_TORCHAUDIO = False

import nodes
import comfy.model_management
import comfy.utils
import comfy.nested_tensor
import node_helpers
from comfy.ldm.minimax.model import FRAME_PER_TOKEN
import comfy.ldm.minimax.model as _minimax_model
from comfy_api.latest import ComfyExtension, io

# ---- runtime patch to preserve reference duration when striding ----
# Without this, stride halves the reference timeline (double speed) because
# PackedLayout places strided tokens contiguously. We patch _ref_t_span and
# PackedLayout to spread strided tokens over the original duration.
# Use a global flag on the target module to make patch idempotent across
# multiple imports of this file (Comfy may load the module twice in tests).
def _apply_stride_patch():
    if getattr(_minimax_model.PackedLayout, "_nacholmo_stride_patched", False):
        return
    orig_ref_t_span = _minimax_model._ref_t_span
    orig_video_t_spans = _minimax_model._video_t_spans
    orig_video_t_grid = _minimax_model._video_t_grid
    orig_video_grid = _minimax_model._video_grid
    orig_PackedLayout_init = _minimax_model.PackedLayout.__init__

    def _patched_ref_t_span(blk):
        kind = blk.get("kind")
        if kind == "image":
            return 1.0
        if kind == "audio":
            return float(blk.get("ref_audio_t", 0))
        if kind in ("video", "video_audio"):
            if blk.get("preserve_duration") and blk.get("orig_latent_t") is not None:
                orig_vt = blk["orig_latent_t"]
                try:
                    return max(float(blk.get("ref_audio_t", 0)), sum(orig_video_t_spans(orig_vt)))
                except Exception:
                    pass
            return orig_ref_t_span(blk)
        return 0.0

    def _refs_cursor_delta_patched(refs):
        delta = 0.0
        for blk in refs or ():
            delta += _patched_ref_t_span(blk)
        return delta

    def _context_k_distance(k):
        if k >= 0:
            return 0.0
        m = -k
        return sum(_minimax_model.FRAME_RESCALE * _minimax_model.FRAME_PER_TOKEN[(-i) % 5] for i in range(1, m + 1))

    def _patched_PackedLayout_init(self, text_len, latent_t, latent_h, latent_w, audio_t, keyframes=None, refs=None, frame_count=None, **kwargs):
        has_strided = False
        if refs:
            for blk in refs:
                if blk.get("preserve_duration") and blk.get("frame_stride", 1) > 1 and blk.get("orig_latent_t") is not None:
                    has_strided = True
                    break
        if not has_strided:
            try:
                return orig_PackedLayout_init(self, text_len, latent_t, latent_h, latent_w, audio_t, keyframes=keyframes, refs=refs, frame_count=frame_count, **kwargs)
            except TypeError:
                if frame_count is not None:
                    try:
                        return orig_PackedLayout_init(self, text_len, latent_t, latent_h, latent_w, audio_t, keyframes=keyframes, refs=refs, **kwargs)
                    except TypeError:
                        return orig_PackedLayout_init(self, text_len, latent_t, latent_h, latent_w, audio_t, keyframes, refs)
                return orig_PackedLayout_init(self, text_len, latent_t, latent_h, latent_w, audio_t, keyframes, refs)

        # has_strided == True: custom init merging Extend context handling with stride scaling
        frame, w_grid = _minimax_model._frame_grid(latent_h, latent_w)
        frame_rows = frame.shape[0]

        segments = [("text", text_len)]
        g = torch.zeros(text_len, 3, dtype=torch.float64)
        g[:, 0] = torch.arange(text_len, dtype=torch.float64)
        pos = [g]

        img_pos, img_update = [], []
        audio_pos, audio_update = [], []
        row = text_len

        target_audio_w = (float(w_grid[0]), float(w_grid[-1]))
        target_origin = float(text_len) + _refs_cursor_delta_patched(refs)
        cursor_for_keyframes = target_origin
        context_k_cursor = 0
        audio_context_cursor = target_origin

        if keyframes:
            for kf in keyframes:
                if kf.get("kind") == "context":
                    n_frames = kf["num_frames"]
                    ks = range(context_k_cursor - n_frames + 1, context_k_cursor + 1)
                    t_grid = torch.tensor([target_origin - _context_k_distance(k) for k in ks], dtype=torch.float64)
                    context_k_cursor -= n_frames
                    g2 = torch.empty(n_frames, frame_rows, 3, dtype=torch.float64)
                    g2[:, :, 0] = t_grid[:, None]
                    g2[:, :, 1:] = frame[None]
                    n = n_frames * frame_rows
                    segments.append(("cond", n))
                    pos.append(g2.reshape(-1, 3))
                    img_pos.append(torch.arange(row, row + n))
                    img_update.append(torch.zeros(n, dtype=torch.bool))
                    row += n
                    continue
                if kf.get("kind") == "context_audio":
                    rt = kf["num_frames"]
                    segments.append(("ref_audio", rt * 2))
                    pos.append(_minimax_model._audio_grid(audio_context_cursor - rt, rt, float(w_grid[0]), float(w_grid[-1])))
                    audio_pos.append(torch.arange(row, row + rt * 2))
                    audio_update.append(torch.zeros(rt * 2, dtype=torch.bool))
                    audio_context_cursor -= rt
                    row += rt * 2
                    continue
                if "resolved_frame_index" in kf:
                    pixel_index = kf["resolved_frame_index"]
                    if pixel_index == 0:
                        cond_t = target_origin
                    elif frame_count is not None and pixel_index == frame_count - 1:
                        try:
                            cond_t = target_origin + sum(_minimax_model._video_t_spans(latent_t)) - _minimax_model.FRAME_RESCALE
                        except Exception:
                            cond_t = cursor_for_keyframes + _minimax_model.FRAME_RESCALE * pixel_index
                    else:
                        # generic anchor (Extend would raise for non-first/last, but we allow)
                        if pixel_index != 0 and not (frame_count is not None and pixel_index == frame_count - 1):
                            # keep strict for Extend compatibility: raise if not first/last unless we are in stride mode where intermediate might be used
                            # For now, compute generic position to avoid crash
                            pass
                        cond_t = cursor_for_keyframes + _minimax_model.FRAME_RESCALE * pixel_index
                    video_latent = kf.get("latent")
                    if video_latent is not None:
                        # Extend's resolved_frame_index uses single frame_rows (image keyframe)
                        n = frame_rows
                        g_single = torch.empty(frame_rows, 3, dtype=torch.float64)
                        g_single[:, 0] = cond_t
                        g_single[:, 1:] = frame
                        segments.append(("cond", n))
                        pos.append(g_single)
                        img_pos.append(torch.arange(row, row + n))
                        img_update.append(torch.zeros(n, dtype=torch.bool))
                        row += n
                    audio_latent = kf.get("audio_latent")
                    if audio_latent is not None:
                        rt = audio_latent.shape[-1]
                        segments.append(("cond_audio", rt * 2))
                        pos.append(_minimax_model._audio_grid(cond_t, rt, *target_audio_w))
                        audio_pos.append(torch.arange(row, row + rt * 2))
                        audio_update.append(torch.zeros(rt * 2, dtype=torch.bool))
                        row += rt * 2
                    continue
                # fallback for unknown kf shape
                cond_t = cursor_for_keyframes + _minimax_model.FRAME_RESCALE * kf.get("resolved_frame_index", 0)
                video_latent = kf.get("latent")
                if video_latent is not None:
                    vt = video_latent.shape[2]
                    n = vt * frame_rows
                    segments.append(("cond", n))
                    pos.append(_minimax_model._video_grid(vt, frame, cond_t))
                    img_pos.append(torch.arange(row, row + n))
                    img_update.append(torch.zeros(n, dtype=torch.bool))
                    row += n
                audio_latent = kf.get("audio_latent")
                if audio_latent is not None:
                    rt = audio_latent.shape[-1]
                    segments.append(("cond_audio", rt * 2))
                    pos.append(_minimax_model._audio_grid(cond_t, rt, *target_audio_w))
                    audio_pos.append(torch.arange(row, row + rt * 2))
                    audio_update.append(torch.zeros(rt * 2, dtype=torch.bool))
                    row += rt * 2

        if refs:
            cursor = float(text_len)
            for blk in refs:
                kind = blk["kind"]
                if kind == "image":
                    r_frame, _ = _minimax_model._frame_grid(blk["latent_h"], blk["latent_w"])
                    n = r_frame.shape[0]
                    g = torch.empty(n, 3, dtype=torch.float64)
                    g[:, 0] = cursor
                    g[:, 1:] = r_frame
                    segments.append(("ref_img", n))
                    pos.append(g)
                    img_pos.append(torch.arange(row, row + n))
                    img_update.append(torch.zeros(n, dtype=torch.bool))
                    row += n
                    cursor += 1.0
                elif kind == "audio":
                    rt = blk["ref_audio_t"]
                    if rt > 0:
                        segments.append(("ref_audio", rt * 2))
                        pos.append(_minimax_model._audio_grid(cursor, rt, *target_audio_w))
                        audio_pos.append(torch.arange(row, row + rt * 2))
                        audio_update.append(torch.zeros(rt * 2, dtype=torch.bool))
                        row += rt * 2
                    cursor += float(rt)
                elif kind in ("video", "video_audio"):
                    rt = blk["ref_audio_t"]
                    vt = blk["latent_t"]
                    stride = int(blk.get("frame_stride", 1))
                    orig_vt = blk.get("orig_latent_t", vt)
                    preserve = blk.get("preserve_duration", False)
                    r_frame, r_w_grid = _minimax_model._frame_grid(blk["latent_h"], blk["latent_w"])
                    if rt > 0:
                        segments.append(("ref_audio", rt * 2))
                        pos.append(_minimax_model._audio_grid(cursor, rt, float(r_w_grid[0]), float(r_w_grid[-1])))
                        audio_pos.append(torch.arange(row, row + rt * 2))
                        audio_update.append(torch.zeros(rt * 2, dtype=torch.bool))
                        row += rt * 2
                    n = vt * r_frame.shape[0]
                    segments.append(("ref_img", n))
                    if preserve and stride > 1 and orig_vt is not None and orig_vt != vt:
                        try:
                            orig_spans = orig_video_t_spans(orig_vt)
                            orig_total = sum(orig_spans)
                            cur_spans = orig_video_t_spans(vt)
                            cur_total = sum(cur_spans)
                            scale = orig_total / cur_total if cur_total != 0 else 1.0
                            scaled_spans_t = torch.tensor([s * scale for s in cur_spans], dtype=torch.float64)
                            t_grid = float(cursor) + torch.cat([torch.zeros(1, dtype=torch.float64), scaled_spans_t[:-1].cumsum(0)])
                        except Exception:
                            t_grid = orig_video_t_grid(vt, cursor)
                        g = torch.empty(vt, r_frame.shape[0], 3, dtype=torch.float64)
                        g[:, :, 0] = t_grid[:, None]
                        g[:, :, 1:] = r_frame[None]
                        pos_grid = g.reshape(-1, 3)
                        pos.append(pos_grid)
                    else:
                        pos.append(orig_video_grid(vt, r_frame, cursor))
                    img_pos.append(torch.arange(row, row + n))
                    img_update.append(torch.zeros(n, dtype=torch.bool))
                    row += n
                    if preserve and orig_vt is not None:
                        try:
                            cursor += max(float(rt), sum(orig_video_t_spans(orig_vt)))
                        except Exception:
                            cursor += max(float(rt), sum(orig_video_t_spans(vt)))
                    else:
                        cursor += max(float(rt), sum(orig_video_t_spans(vt)))

        segments.append(("audio", audio_t * 2))
        pos.append(_minimax_model._audio_grid(cursor, audio_t, *target_audio_w))
        audio_pos.append(torch.arange(row, row + audio_t * 2))
        audio_update.append(torch.ones(audio_t * 2, dtype=torch.bool))
        row += audio_t * 2

        n_video = latent_t * frame_rows
        segments.append(("video", n_video))
        pos.append(_minimax_model._video_grid(latent_t, frame, cursor))
        img_pos.append(torch.arange(row, row + n_video))
        img_update.append(torch.ones(n_video, dtype=torch.bool))
        row += n_video

        self.seq_len = row
        self.position_ids = torch.cat(pos)
        self.img_pos = torch.cat(img_pos)
        self.img_update = torch.cat(img_update)
        self.audio_pos = torch.cat(audio_pos)
        self.audio_update = torch.cat(audio_update)
        self.signature = (text_len, latent_t, latent_h, latent_w, audio_t)
        seg_abs = []
        off = 0
        for kind, n in segments:
            seg_abs.append((off, off + n, kind))
            off += n
        self.segments = seg_abs

    _minimax_model._ref_t_span = _patched_ref_t_span
    _minimax_model.PackedLayout.__init__ = _patched_PackedLayout_init
    _minimax_model.PackedLayout._nacholmo_stride_patched = True
    _minimax_model._ref_t_span._nacholmo_stride_patched = True
    print("[MiniMaxH3-Stride] Patched PackedLayout to preserve reference duration for strided videos (with Extend compat, frame_count support)", flush=True)

# Apply at import time
try:
    _apply_stride_patch()
except Exception as e:
    print(f"[MiniMaxH3-Stride] Failed to patch PackedLayout: {e}", flush=True)

# ---- copied helpers from comfy_extras/nodes_minimax_h3.py for self-containment ----
CANVAS_MULTIPLE = 32
BASE_SHORT_EDGE = 768
MAX_PIXELS = 768 * 1344
REF_IMAGE_SHORT_EDGE = 2048
FPS = 24
AUDIO_LATENT_FPS = 40


def align_frame_count(n):
    while n % 17 != 5:
        n += 1
    return n


def video_latent_t(frame_count):
    return 2 if frame_count <= 5 else ((frame_count - 5) // 17) * 5 + 2


def temporal_shape(length):
    frame_count = align_frame_count(max(5, length))
    duration = frame_count / FPS
    return frame_count, video_latent_t(frame_count), round(duration * AUDIO_LATENT_FPS)


def adapt_canvas(width, height):
    """768-short-edge canvas with 768*1344 area cap, per-axis round to 32."""
    ratio = width / height
    if ratio >= 1.0:
        nom_w, nom_h = BASE_SHORT_EDGE * ratio, BASE_SHORT_EDGE
    else:
        nom_w, nom_h = BASE_SHORT_EDGE, BASE_SHORT_EDGE / ratio
    if nom_w * nom_h > MAX_PIXELS:
        s = math.sqrt(MAX_PIXELS / (nom_w * nom_h))
        nom_w, nom_h = nom_w * s, nom_h * s
    return (max(CANVAS_MULTIPLE, round(nom_w / CANVAS_MULTIPLE) * CANVAS_MULTIPLE),
            max(CANVAS_MULTIPLE, round(nom_h / CANVAS_MULTIPLE) * CANVAS_MULTIPLE))


def _resize(image, width, height, crop):
    # image [B, H, W, C] -> [B, height, width, 3]
    samples = image[..., :3].movedim(-1, 1)
    samples = comfy.utils.common_upscale(samples, width, height, "lanczos", crop)
    return samples.movedim(1, -1)


def _encode_ref_audio(audio_vae, audio):
    waveform = audio["waveform"]  # [B, C, L]
    sr = audio["sample_rate"]
    vae_sr = getattr(audio_vae, "audio_sample_rate", 32000)
    if sr != vae_sr:
        if _HAS_TORCHAUDIO and hasattr(torchaudio, "functional"):
            try:
                waveform = torchaudio.functional.resample(waveform, sr, vae_sr)
            except Exception:
                # Fallback to torch interpolate if torchaudio missing or fails (e.g. XPU w/o CUDA)
                waveform = torch.nn.functional.interpolate(waveform.float(), scale_factor=vae_sr / sr, mode="linear", align_corners=False).to(waveform.dtype)
        else:
            waveform = torch.nn.functional.interpolate(waveform.float(), scale_factor=vae_sr / sr, mode="linear", align_corners=False).to(waveform.dtype)
    z = audio_vae.encode(waveform[:1].movedim(1, -1))  # [1, 32, 2, T]
    return z, z.shape[-1]


def _empty_av_latent(width, height, length, batch_size=1):
    frame_count, latent_t, audio_t = temporal_shape(length)
    video = torch.zeros([batch_size, 24, latent_t, height // 16, width // 16],
                        device=comfy.model_management.intermediate_device())
    audio = torch.zeros([batch_size, 32, 2, audio_t],
                        device=comfy.model_management.intermediate_device())
    return {"samples": comfy.nested_tensor.NestedTensor((video, audio))}, frame_count


# ---- VRAM-saver node ----

class MiniMaxH3ReferenceToVideoStride(io.ComfyNode):
    """ref2va with reference-video frame striding for VRAM reduction.

    Stride is applied *before* VAE encoding so the packed DiT sequence
    shrinks proportionally. See module docstring for memory analysis.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3ReferenceToVideoStride",
            description=(
                "VRAM-saver variant of MiniMax H3 Reference to Video. "
                "Adds `ref_video_stride` to uniformly skip reference-video frames before "
                "VAE encoding, halving (stride=2) or more the reference token count. "
                "The packed DiT sequence length S is dominated by vt*frame_rows per ref video, "
                "so stride reduces VRAM ~linearly (flash) / ~quadratically (naive). "
                "Vanilla uses stride=1. Use 2 for half VRAM, 3-4 for aggressive saving. "
                "Qwen vision tokens can be kept at full density (better identity) or strided (max saving)."
            ),
            display_name="MiniMax H3 Reference to Video (Stride / Lite)",
            category="model/conditioning/minimax",
            inputs=[
                io.Clip.Input("clip"),
                io.Vae.Input("vae"),
                io.Vae.Input("audio_vae"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.Int.Input("width", default=1344, min=32, max=nodes.MAX_RESOLUTION, step=32),
                io.Int.Input("height", default=768, min=32, max=nodes.MAX_RESOLUTION, step=32),
                io.Int.Input("length", default=124, min=5, max=3600, step=17, tooltip="Frame count at 24 fps, (124 = ~5s, trained range is ~124-362)"),
                io.Combo.Input("ref_image_size", options=["match", "max"], default="match",
                    tooltip="Reference image sizing. 'match' scales each ref (down only, keeping aspect) to the generation's pixel area; 'max' uses 2048px short edge for best fidelity. Tokens ride every step, so 'max' is slower."),
                io.Int.Input("ref_video_stride", default=2, min=1, max=8, step=1,
                    tooltip="Frame stride for reference VIDEOS only (images/audio unaffected). 1 = vanilla (all frames), 2 = every other frame (~50% VRAM for refs), 3 = every third, etc. Applied before VAE encode, then re-snapped to valid 17k+5. Saves ~1/stride tokens."),
                io.Combo.Input("qwen_stride_mode", options=["strided", "full"], default="strided",
                    tooltip="'strided': Qwen-VL sees the strided frames (max VRAM saving, fewer vision tokens). 'full': Qwen sees the original (pre-stride) frames at 2 fps (better identity, ~same DiT saving, slightly more text tokens)."),
                io.Combo.Input("stride_mode", options=["pre_vae", "post_vae"], default="pre_vae",
                    tooltip="'pre_vae': subsample pixels then encode (max saving, recommended). 'post_vae': encode full then slice latent t-dim (preserves VAE temporal kernel, but encode cost not saved)."),
                io.Boolean.Input("preserve_duration", default=True, label_on="preserve (normal speed)", label_off="compress (2× speed)",
                    tooltip="When ON (default), strided reference keeps original duration by spreading its tokens over the full time span – fixes double-speed. When OFF, strided reference is treated as contiguous (old behavior, double speed for stride=2)."),
                io.Autogrow.Input("ref_images", optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input("ref_image", tooltip="Reference image (downscaled to 2048 short edge if larger, never upscaled)"),
                        prefix="ref_image_", min=0, max=9)),
                io.Autogrow.Input("ref_videos", optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input("ref_video", tooltip="Reference video frames at 24 fps (2-15s). Stride will uniformly drop frames."),
                        prefix="ref_video_", min=0, max=3)),
                io.Autogrow.Input("ref_video_audios", optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Audio.Input("ref_video_audio", tooltip="Soundtrack of the same-numbered reference video"),
                        prefix="ref_video_audio_", min=0, max=3)),
                io.Autogrow.Input("ref_audios", optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Audio.Input("ref_audio", tooltip="Standalone reference audio"),
                        prefix="ref_audio_", min=0, max=3)),
            ],
            outputs=[io.Conditioning.Output(display_name="positive"), io.Latent.Output()],
        )

    @classmethod
    def execute(cls, clip, vae, audio_vae, prompt, width, height, length,
                ref_image_size="match", ref_video_stride=2, qwen_stride_mode="strided",
                stride_mode="pre_vae", preserve_duration=True,
                ref_images=None, ref_videos=None, ref_video_audios=None, ref_audios=None) -> io.NodeOutput:
        # Clamp stride to sane range even if UI allows 1..8
        try:
            stride = int(ref_video_stride)
        except Exception:
            stride = 1
        stride = max(1, min(8, stride))
        latent, frame_count = _empty_av_latent(width, height, length)

        ref_items = []   # for tokenizer (Qwen) presentation, in request order
        ref_blocks = []  # for DiT payload, same order

        # ---- reference images (unaffected by stride) ----
        for img in (ref_images or {}).values():
            if img is None:
                continue
            h, w = img.shape[1], img.shape[2]
            if ref_image_size == "match":
                scale = min(1.0, math.sqrt((width * height) / (w * h)))
            else:
                scale = min(1.0, REF_IMAGE_SHORT_EDGE / min(w, h))
            tw = max(CANVAS_MULTIPLE, round(w * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
            th = max(CANVAS_MULTIPLE, round(h * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
            resized = _resize(img[:1], tw, th, "disabled")
            z = vae.encode(resized)
            ref_items.append({"type": "image", "data": resized})
            ref_blocks.append({"kind": "image", "latent_h": th // 16, "latent_w": tw // 16, "latent": z})

        # ---- reference videos (with striding) ----
        ref_video_audios = ref_video_audios or {}
        for name, video_frames in (ref_videos or {}).items():
            if video_frames is None:
                continue
            suffix = name.rsplit("_", 1)[-1]
            soundtrack = ref_video_audios.get("ref_video_audio_" + suffix)
            orig_n = int(video_frames.shape[0])
            vh, vw = video_frames.shape[1], video_frames.shape[2]
            cw, ch = adapt_canvas(vw, vh)
            if vw * vh < cw * ch:
                cw = max(CANVAS_MULTIPLE, round(vw / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
                ch = max(CANVAS_MULTIPLE, round(vh / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)

            # Keep a copy for Qwen "full" mode before striding
            # Resize first (as vanilla does), then stride
            frames_full = _resize(video_frames, cw, ch, "disabled")
            if frames_full.shape[0] > frame_count:
                frames_full = frames_full[:frame_count]

            # Choose frames for VAE path - compute orig metrics for preserve_duration
            # orig_n_snap is the valid 17k+5 length the vanilla node would have used (no stride)
            orig_n_full = int(frames_full.shape[0])
            _orig_n_snap = orig_n_full
            if _orig_n_snap >= 5:
                while _orig_n_snap % 17 != 5:
                    _orig_n_snap -= 1
                if _orig_n_snap < 5:
                    _orig_n_snap = 5
            else:
                _orig_n_snap = 5
            orig_latent_t_est = video_latent_t(_orig_n_snap)

            if stride_mode == "post_vae":
                # Encode full then slice latent
                n_pre = frames_full.shape[0]
                if n_pre < 5:
                    raise ValueError("MiniMax H3 reference videos need at least 5 frames (~0.2s at 24 fps)")
                while n_pre % 17 != 5:
                    n_pre -= 1
                frames_for_encode = frames_full[:n_pre]
                z_full = vae.encode(frames_for_encode)
                latent_t_full = int(z_full.shape[2])
                # Use actual encoded full as orig for accurate preserve
                orig_latent_t_est = latent_t_full
                idx = torch.arange(0, latent_t_full, stride, device=z_full.device)
                if idx.shape[0] < 2:
                    idx = torch.arange(0, min(2, latent_t_full), device=z_full.device)
                z = z_full[:, :, idx]
                # Qwen handling
                if qwen_stride_mode == "full":
                    frames_qwen = frames_full
                    sample_idx = list(range(0, frames_qwen.shape[0], FPS // 2))
                    qwen_frames = frames_qwen[sample_idx]
                    qwen_timestamps = [i / 2.0 for i in range(len(sample_idx))]
                    qwen_log = "full"
                else:
                    frames_qwen = frames_full[::stride]
                    if frames_qwen.shape[0] == 0:
                        frames_qwen = frames_full[:1]
                    sample_idx = list(range(0, frames_qwen.shape[0], FPS // 2))
                    qwen_frames = frames_qwen[sample_idx]
                    # Preserve duration: timestamps spaced by stride
                    if preserve_duration:
                        qwen_timestamps = [i * stride / 2.0 for i in range(len(sample_idx))]
                    else:
                        qwen_timestamps = [i / 2.0 for i in range(len(sample_idx))]
                    qwen_log = f"strided(preserve={preserve_duration})"
                print(f"[MiniMaxH3-Stride] ref_video={name} orig_frames={orig_n} "
                      f"resized={frames_full.shape[0]} stride={stride} mode=post_vae "
                      f"latent_t {latent_t_full}->{z.shape[2]} orig_latent_t={orig_latent_t_est} "
                      f"qwen_frames {len(sample_idx)} qwen_mode={qwen_stride_mode} preserve={preserve_duration} "
                      f"cw={cw} ch={ch} latent_h={ch//16} latent_w={cw//16}",
                      flush=True)
                # stash for ref_blocks
                _vt = int(z.shape[2])
                _orig_vt = int(orig_latent_t_est)
                _use_preserve = bool(preserve_duration and stride > 1 and _orig_vt != _vt)
            else:
                # pre_vae: stride pixels before encode (recommended)
                frames = frames_full
                if stride > 1:
                    frames = frames[::stride]
                n = frames.shape[0]
                if n < 5:
                    raise ValueError(
                        f"MiniMax H3 reference videos need at least 5 frames after stride "
                        f"(got {n} from {orig_n} with stride {stride}); try a smaller stride or longer clip")
                while n % 17 != 5:
                    n -= 1
                frames = frames[:n]
                z = vae.encode(frames)
                _vt = int(z.shape[2])
                _orig_vt = int(orig_latent_t_est)
                _use_preserve = bool(preserve_duration and stride > 1 and _orig_vt != _vt)
                if qwen_stride_mode == "full":
                    frames_qwen = frames_full
                    sample_idx = list(range(0, frames_qwen.shape[0], FPS // 2))
                    qwen_frames = frames_qwen[sample_idx]
                    qwen_timestamps = [i / 2.0 for i in range(len(sample_idx))]
                    qwen_log = "full"
                else:
                    sample_idx = list(range(0, frames.shape[0], FPS // 2))
                    qwen_frames = frames[sample_idx]
                    if preserve_duration:
                        qwen_timestamps = [i * stride / 2.0 for i in range(len(sample_idx))]
                    else:
                        qwen_timestamps = [i / 2.0 for i in range(len(sample_idx))]
                    qwen_log = f"strided(preserve={preserve_duration})"
                print(f"[MiniMaxH3-Stride] ref_video={name} orig_frames={orig_n} "
                      f"resized_full={frames_full.shape[0]} stride={stride} mode=pre_vae "
                      f"kept={frames.shape[0]} (n snapped {n}) orig_n_snap={_orig_n_snap} "
                      f"latent_t {_vt} orig_latent_t={_orig_vt} preserve={_use_preserve} "
                      f"qwen_frames={len(sample_idx)} qwen_log={qwen_log} cw={cw} ch={ch}",
                      flush=True)

            audio_latent, ref_audio_t = (None, 0)
            if soundtrack is not None:
                audio_latent, ref_audio_t = _encode_ref_audio(audio_vae, soundtrack)
                ref_items.append({"type": "audio"})

            # Qwen video entry with preserve-aware timestamps
            ref_items.append({"type": "video", "data": qwen_frames,
                              "timestamps": qwen_timestamps})
            # Store stride metadata for the DiT patch to preserve duration
            blk = {"kind": "video_audio" if ref_audio_t else "video",
                               "latent_t": _vt, "latent_h": ch // 16, "latent_w": cw // 16,
                               "ref_audio_t": ref_audio_t, "latent": z, "audio_latent": audio_latent}
            if _use_preserve:
                blk["frame_stride"] = int(stride)
                blk["orig_latent_t"] = int(_orig_vt)
                blk["preserve_duration"] = True
            ref_blocks.append(blk)

        # ---- standalone audio (unaffected) ----
        for audio in (ref_audios or {}).values():
            if audio is None:
                continue
            audio_latent, ref_audio_t = _encode_ref_audio(audio_vae, audio)
            ref_items.append({"type": "audio"})
            ref_blocks.append({"kind": "audio", "ref_audio_t": ref_audio_t, "audio_latent": audio_latent})

        tokens = clip.tokenize(prompt, minimax_ref_items=ref_items)
        cond = clip.encode_from_tokens_scheduled(tokens)
        if ref_blocks:
            cond = node_helpers.conditioning_set_values(cond, {"minimax_refs": ref_blocks})

        # Optional: log sequence length estimate for VRAM debugging
        try:
            # Estimate packed seq len similar to PackedLayout for logging
            # text_len from tokens is not directly available here, but we can approximate via cond
            # Instead we log per-ref token counts
            for i, blk in enumerate(ref_blocks):
                if blk["kind"] in ("video", "video_audio"):
                    fr = (blk["latent_h"] // 2) * (blk["latent_w"] // 2) if blk["latent_h"] % 2 == 0 and blk["latent_w"] % 2 == 0 else 0
                    # more precise: frame_rows = (latent_h//2)*(latent_w//2) via _frame_grid
                    # approximate
                    tokens_est = blk["latent_t"] * (blk["latent_h"] // 2) * (blk["latent_w"] // 2)  # placeholder
                    # Actual frame_rows may differ due to area norm, but this is indicative
                    print(f"[MiniMaxH3-Stride] ref_block {i} kind={blk['kind']} "
                          f"latent_t={blk['latent_t']} latent_h={blk['latent_h']} latent_w={blk['latent_w']} "
                          f"~tokens~{tokens_est} ref_audio_t={blk.get('ref_audio_t',0)}", flush=True)
        except Exception:
            pass

        return io.NodeOutput(cond, latent)


# Keep extension compatible with Comfy's new API, but also expose legacy mappings
NODE_CLASS_MAPPINGS = {
    "MiniMaxH3ReferenceToVideoStride": MiniMaxH3ReferenceToVideoStride,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3ReferenceToVideoStride": "MiniMax H3 Reference to Video (Stride / Lite)",
}


class MiniMaxH3StrideExtension(ComfyExtension):
    async def get_node_list(self):
        return [MiniMaxH3ReferenceToVideoStride]


async def comfy_entrypoint() -> MiniMaxH3StrideExtension:
    return MiniMaxH3StrideExtension()
