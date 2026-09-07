# Rolling Release — Design & Policy

Aurora runs **fully floating**: latest torch nightly XPU, latest ComfyUI core,
latest custom-node commits, unpinned companion deps. Fix-forward when things
break. This doc describes the model, the tooling, and the recovery runbook.

Tooling: `scripts/roll.sh` (the update runner), `scripts/roll-verify.sh`
(the release gate), `manifests/last-good.json` (last verified state),
`manifests/roll-holds.conf` (surgical holds), `manifests/roll-snapshots/`
(per-roll evidence).

## 1. Why floating needs machinery

Three hard constraints fight floating versions (a lockstep triple):

1. **Provider hash contract.** `omni_xpu_kernel` + both provider wheels are
   built against an EXACT torch build (`runtime.torch_version` in the
   provider manifest). A newer nightly makes OmniXPU reject both providers
   (`"does not match provider"` in the boot log) → no Kitchen XPU dispatch,
   no AIMDO VBAR. Attention/norm adapters still apply, so the failure is
   *silent degradation*, not a crash. `roll.sh` therefore **rebuilds the
   provider wheels** against the new torch (see §3) and the verify gate
   fails the roll if Kitchen XPU is not `active`.
2. **Core ↔ official ↔ provider lockstep.** ComfyUI core imports provider
   surface directly (observed: core `d03a2430` needs
   `comfy_aimdo.malloc_graph`, only in official ≥0.5.x) while the provider
   sources accept only the older official (0.4.15). Floating any one side
   breaks the other two — proven by two RED rolls (first: official 0.5.2
   rejected by providers; second: new core crashing on held 0.4.15).
   Resolved without moving officials/providers: `scripts/apply-aimdo-shim.sh`
   extracts `malloc_graph.py` from the official PyPI wheel at install time
   (GPLv3, same licence as installed comfy-aimdo; nothing vendored in this
   repo). On XPU the module is import-only — every use is gated behind
   `is_device_cuda()` — so the shim never executes natively here.
3. **Local patch drift.** `comfyui_controlnet_aux` carries the
   DepthAnythingV2 XPU patch. A new upstream commit can reject the patch.
   `roll.sh` floats it from a *clean* tree and holds it at last-good on
   reject instead of shipping unpatched (unpatched = wrong-device crash
   on XPU). Never carry the dirty patch tree across the float checkout —
   it can silently half-apply.

Everything else floats freely; pip resolves companion deps unpinned.

## 2. What rolls, in order

`scripts/roll.sh` (default: everything; `--skip-*` to narrow):

| # | Layer | How | Hold mechanism |
|---|---|---|---|
| 0 | Snapshot | `pip freeze` + all node commits → `manifests/roll-snapshots/<date>/`; `venv` → `venv.pre-roll-<date>` (instant `mv`) | rollback source |
| 1 | Torch nightly | `pip install --pre -U torch torchaudio torchvision triton-xpu` (nightly index, **no pin**) | full rollback |
| 1b | Torch keep (`--skip-torch`) | Reinstall the **snapshot** torch/torchvision/triton-xpu pins + Omni wheels into the fresh venv (a skipped torch still gets a complete stack — a bare venv kills both boots; observed) | full rollback |
| 2 | Providers | Re-stamp provider wheels for the new torch via `build_wheel.py` from existing source wheels (`--source-wheel/--source-revision/--torch-version/--xpu-target bmg`), install `--no-deps`; kernel wheel reused `--no-deps` (no kernel rebuild — see §4) | full rollback |
| 3 | Official kitchen/aimdo | **HELD** — re-pinned by root `requirements.txt`, never `-U`'d (see §3) | advance only with provider sources (manual) |
| 4 | ComfyUI core | `git fetch origin` + checkout `origin/HEAD` (detached — see note below) | `roll-holds.conf`: `comfy-core=<commit>` |
| 5 | Custom nodes | Same float per dir in `FLOAT_NODES`; controlnet_aux patch re-applied, hold-on-reject | `roll-holds.conf`: `<dirname>=<commit>` |

> Note: after a green core float, core is detached at the recorded commit
> (`last-good.json: comfyui_core`). Audit position via `git log`, not branch
> — floating tracks commits, not branches.
| 6 | Node requirements | Ensure-installed for present node dirs (no `-U`: completes the fresh venv without floating; `companion-pins -U` is the version floater) + nunchaku dist rebuild | warnings only (fix forward) |
| 7 | Verify gate | `scripts/roll-verify.sh` (see §5) | fail → automatic rollback |
| 8 | Record | Write `manifests/last-good.json`; prune pre-roll venvs (keep 1) | `setup.sh --fresh-venv` seeds from it |

`manifests/roll-holds.conf` format (one per line, `#` comments):

```ini
# <dirname>=<commit>  — float everything except these
comfyui_controlnet_aux=e8b689a513c3e6b63edc44066560ca5919c0576e
```

## 3. Provider rebuild details

Official `comfy-kitchen`/`comfy-aimdo` (PyPI) and the XPU provider wheels
form a lockstep triple with the provider *sources*: the provider manifest
pins both `compatible_versions` (official) and `torch_version`. Floating the
official packages alone gets both providers rejected (`official ... is
incompatible; provider accepts [...]`) — observed on the first live roll
(official 0.2.33/0.5.2 vs accepted 0.2.31/0.4.15). So `roll.sh` never `-U`s
them; they advance only together with a manual provider-source rebuild:

Sources live outside this repo (Intel llm-scaler checkout):

- Vendor scripts:
  - `/home/sundae/llm-scaler/omni/comfy-kitchen-xpu/packaging/xpu_runtime_provider/build_wheel.py`
  - `/home/sundae/llm-scaler/omni/comfy-aimdo-xpu/packaging/xpu_runtime_provider/build_wheel.py`
- Inputs: existing source wheels in `/home/sundae/llm-scaler/wheels/{kitchen,aimdo}-source/`
  (rebuilt from source checkouts only when the provider *source* must move —
  manual, not part of a roll), current source revisions, `torch.__version__`,
  `--xpu-target bmg`.
- Output: fresh provider wheels in a temp dir → `pip install --no-deps`.
- The AIMDO **native lib** (UR hook `.so` inside the aimdo-source wheel) is
  NOT rebuilt by a roll. If a new torch breaks it, the verify gate (AIMDO
  init line missing in VRAM boot) fails the roll → automatic rollback.
  Rebuilding the native lib (`comfy-aimdo-xpu/scripts/build-linux-xpu.sh`,
  needs oneAPI `icx` + UR headers) is a manual task.

## 4. What does NOT roll automatically

- **`omni_xpu_kernel`**: full SYCL/CUTLASS rebuild (`icx`, `CUTLASS_SYCL_ROOT`,
  `OMNI_XPU_REQUIRE_CUTE=1`, hours of compile). Held at the last working
  wheel; installed `--no-deps` over new torch. If the kernel's native ABI
  breaks against a new torch (verify gate: probe import / adapter apply /
  boot), the roll fails back and the kernel becomes the blocking item —
  rebuild manually per `llm-scaler/omni/docker/Dockerfile` (`kernel-wheel`
  stage), then re-roll.
- **Python itself / `requires-python` caps** (e.g. nunchaku declares `<3.14`;
  currently forced with `--ignore-requires-python`, verified working).
  A Python upgrade is a manual migration, not a roll.
- **This suite's own deprecations/gates** (`NACHOLMO_*`): policy changes,
  not version bumps — normal commits.

## 5. Verify gate (`scripts/roll-verify.sh`)

Fails (non-zero) on the first red item; `roll.sh` rolls back automatically:

1. Provider bootstrap: `comfy_kitchen.xpu == active` (AIMDO `skipped` in
   direct mode is expected).
2. Kitchen XPU backend `available` with capabilities; kernel probe:
   `sdp/norm/rotary/linear_fp8/int8/layout` all present.
3. Boot-to-GUI on an ephemeral port in **both** modes (default VRAM +
   `XPU_VRAM_MODE=direct`): `To see the GUI` present, zero `Traceback`s,
   no `provider rejected` lines.
4. API spot-checks: `ArcSuperResolution`, `MiniMaxH3TurboLoRA`,
   `OmniXPUStatus`, `VideoCombineSync`, `SolAttnPatch` (official),
   `ApplySolAttn` + `SolAttnPatchMiniMax` (ours) all resolve with distinct
   schemas.

## 6. Cadence & policy

- Rolls are **manual, never scheduled**: `./scripts/roll.sh` (add `--dry-run`
  to preview, `--yes` to skip confirmation).
- One layer at a time when diagnosing (`--skip-torch`, `--skip-core`,
  `--skip-nodes`), everything together when confident.
- After a green roll: review `git diff` of *this suite* (roll only touches
  `manifests/`), `git push`.
- After a red roll: the live `venv` is byte-identical to pre-roll
  (instant `mv` back). Fix forward via `roll-holds.conf`, then re-roll.
- `setup.sh --fresh-venv` seeds torch/node versions from
  `manifests/last-good.json` when present (else built-in defaults), so fresh
  installs reproduce the rolling tip instead of a stale pin.

## 7. Recovery runbook (no roll.sh involved)

- **Providers rejected after a manual torch upgrade**: either rebuild
  providers per §3, or `pip install torch==<last-good torch>` (nightly
  retention is limited — days, not weeks).
- **Single node broken after float**: add `<dir>=<last-good commit>` to
  `manifests/roll-holds.conf`, `git -C custom_nodes/<dir> checkout <commit>`,
  restart. Remove the hold on the next roll to retry upstream.
- **Everything broken, no idea why**: `mv venv venv.bad-<date> &&
  mv venv.pre-roll-<latest> venv` (if kept), restore node commits from
  `manifests/roll-snapshots/<date>/nodes.txt`, restart.
- **Last-good state**: `manifests/last-good.json`
  (`torch/torchvision/triton_xpu/kernel/kitchen/aimdo/comfyui_core/nodes/python/date`).
