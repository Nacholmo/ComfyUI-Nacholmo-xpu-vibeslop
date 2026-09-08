# WINT8 Hadamard Families — Why the Wrong One Makes Mosaic Video

## The one-paragraph version

Third-party WINT8 MiniMax-H3 files (Dasiwa, ErosMax, anything from the
ConvRot ecosystem: core native int8, comfy-kitchen, OMNI kernels) rotate
weights with the **regular Hadamard** (4x4-Kronecker construction, power-of-4
sizes). This suite's own quantizer historically used the **Sylvester
Hadamard** (scipy 2x2 recursion). Both are orthogonal and symmetric, but they
are DIFFERENT matrices — and un-rotating with the wrong one silently yields
plausible-looking garbage: correct tensor shapes, healthy norms, passing
naive checks, but mosaic video and sampler `d_rms` ~0.55 instead of ~1.55.

## How it manifests

- Output: patch-granularity mosaic (each DiT token decodes independently,
  no spatial coherence) instead of random-pixel noise.
- Sampler canary (`MiniMaxH3TurboSampler` logs): `d_rms` ~0.5 and flat across
  steps, vs ~1.5 and evolving on a working model. `denoised_rms` looks sane
  (~1.15) in both cases — it does NOT discriminate.
- Everything else checks out: tensor count/shapes/names, per-tensor values
  vs an independent recompute, Q4_K dequant on CPU and XPU, LoRA attach/fire,
  schedule, conds, sampler, VAE/TE. All of these pass because they are either
  self-consistent (same wrong H on both sides of the comparison) or
  unaffected (non-rotated tensors).

## Why value-level checks can't catch it

For stored `R = W @ H_true^T` and a candidate `H`:

- File audit compares against `(int8*scale) @ H_candidate` — passes for ANY
  orthogonal `H_candidate`.
- The functional check `x@W^T == (x@H)@(W_rot)^T` passes for any H used
  consistently on both sides.
- Per-module outputs vs the WINT8 runtime can even match closely, because
  both sides share the same (wrong) H.

The only checks that discriminate:

1. **Spikiness restoration**: correctly un-rotated DiT linears recover
   spiky column structure (top-column-max / median ratio ~10, matching
   never-quantized layers). Wrong-H output stays flat (~1.5).
2. **End-to-end**: sampler `d_rms` curve + decoded video.

## The fix

`tools/convert_wint8_minimax_h3.py` defaults to `--hadamard regular`
(vendored builder, bit-identical to `comfy_kitchen.tensor.int8_utils`).
Use `--hadamard sylvester` only for files made with this suite's own
(scipy-based) quantizer.

Reference values (DasiwaMinimaxH3 hybrid, `blocks.25`):

| stage | qkv col ratio | fc2 col ratio |
|---|---|---|
| stored int8 x scale | 1.6 | 1.6 |
| un-rotated, Sylvester-H (wrong) | 1.5 | 1.5 |
| un-rotated, regular-H (correct) | 10.8 | 5.0 |
| never-quantized refiner (native) | — | 10.8 |

## Open follow-up

This suite's own quantizer (`nodes/wint8/wint8_quarot.py`, scipy
Sylvester-H) disagrees with the live XPU kernels (kitchen/OMNI,
regular-H). Files quantized by our own tool and run through the kernel
fast path may hit the same mismatch from the other side. Our quantizer,
runtime rotation, and converter should be aligned on one family
(recommendation: regular-H, matching the ecosystem) — but that requires
re-quantizing existing suite-made files and its own validation pass, so it
is deliberately NOT part of this fix.
