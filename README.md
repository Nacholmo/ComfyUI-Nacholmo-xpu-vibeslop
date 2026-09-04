> **Note**: Stuff that I vibecoded for myself. If you are looking for hand-written code, look somewhere else lmao
> (Derived only from Apache-2.0, MIT, and CC0-1.0 licensed code, attributed at the end of this README).

# ComfyUI Nacholmo XPU Vibeslop

A comprehensive, unified performance toolkit, custom node suite, and launcher environment for running **ComfyUI** on **Intel Arc GPUs** (Alchemist, Battlemage Xe2, Xe-LPG/HPG) and PyTorch XPU.

<a href="https://www.youtube.com/watch?v=zbgWUK_tidI" target="_blank" rel="noopener noreferrer">
  <img width="1024" height="384" alt="Image" src="https://github.com/user-attachments/assets/38a30e36-85b4-4dc0-8610-3bc5942cb1b7" />
</a>

---

## Features

### 1. Custom Nodes

- **Arc Super Resolution (`ArcSuperResolution`)**:
  - OpenVINO XMX FP16 accelerated super resolution for single images and video batches (`[B, H, W, C]`).
  - **>5x faster** than standard PyTorch upscalers on Intel Arc GPUs.
  - Automatic scale factor detection from model filenames (`4x_...`, `2x_...`, `_x4plus`, etc.).
  - Seamless tiled feathered blending for large resolutions with RAM-safe memmap spillover.
  - Zero-latency compiled binary disk caching (`.cache/`).
- **Arc Resample FPS (`ArcResampleFPS`)**:
  - Resamples image/video batches to a target frame rate (e.g. 72fps down to 60fps) while locking exact duration and preserving audio sync.
- **WINT8 Suite (Linux Port)** (`WINT8ModelQuantizer`, `WINT8ModelLoader`, `WINT8LoRALoader`, `WINT8LoRAStack`):
  - Pure PyTorch INT8 per-row UNet quantization and loading (50% VRAM reduction).
  - Multi-LoRA stacking (up to 5 LoRAs) baked directly onto INT8 weights.
  - Custom XPU-accelerated GEMM kernels with Hadamard rotations (QuaRot).
  - **Linux Port Enhancements**: Auto-detects Intel oneAPI `icpx`, removes Windows BAT/.pth injection requirements, and natively supports modern Linux Triton >= 3.8 / PyTorch XPU.
- **TorchCompile Blockwise (`TorchCompileBlockwise`)**:
  - Compiles transformer blocks individually with `torch.compile` / Inductor.
  - Automatically adds Dynamo graph-breaks at ComfyUI memory boundaries for LowVRAM / offloading safety.
  - Filters `transformer_options` guards to prevent recompiles during sampling steps.
- **Upscale Video With Model (`UpscaleVideoWithModel`)**:
  - Batched frame upscale inference (processes multiple frames per forward pass).
  - Optional `torch.compile` acceleration with batch padding.
  - FP16 output toggle to halve system RAM on long video sequences.
- **Video Combine Sync (`VideoCombineSync`)**:
  - Video Combine node with pitch-preserving `atempo` audio duration matching to keep audio perfectly synchronized with video timing.
- **Sol-Attn Sparse Block Attention (`ApplySolAttn`)**:
  - Block-sparse attention patching with dynamic head allocation, local window attention, and Intel Arc XPU acceleration.
  - Achieves **3-4x speedups** over naive SDPA on long sequence lengths while preventing VRAM exhaustion.
- **MiniMax-H3 Extend Split & Seamless Stitching Suite (`MiniMaxH3LatentSplit`, `MiniMaxH3AttachContext`, `MiniMaxH3AssembleAV`, `MiniMaxH3LatentStitch`)**:
  - Solves the 2MP VRAM bottleneck on 12GB GPUs (Intel Arc B580) by splitting long latents (e.g. 10s 0.5MP) into temporally aligned chunks (e.g. 5s each), upscaling and refining each within safe VRAM limits.
  - Mathematically aligned to MiniMax H3's $(T - 2) \% 5 == 0$ token schedule and duration-locked audio scaling.
  - Uses `ComfyUI-MiniMax-H3-Extend` context conditioning: extracts trailing 2MP frames from Chunk 1 and injects them as negative RoPE-time context keyframes into Chunk 2's conditioning, guaranteeing seamless lighting, motion, and visual continuity across seams.
  - Stitches refined chunks with cosine cross-fade blending for both video and audio.

### 2. Runtime Stability & Calibration Patches

- **TorchAudio Guard (`torchaudio_guard.py` / `sitecustomize.py`)**:
  - Intercepts `torchaudio`'s C++ extension loader to gracefully handle missing CUDA runtime libraries (`libcudart.so`).
  - Enables pure PyTorch/CPU audio fallback so audio VAEs, audio nodes, and companion extensions run without fatal `libcudart` import crashes on Intel Arc / XPU systems.
- **VRAM Guard (`xpu_vram_guard.py` / `prestartup_script.py`)**:
  - Automatically caps the PyTorch XPU caching allocator fraction (default: `0.75` / `75%` of VRAM).
  - On Intel Level Zero drivers, exceeding physical VRAM during sudden activation spikes can hard-lock the GPU kernel and freeze the Linux desktop. VRAM Guard intercepts spikes and raises clean, recoverable Python `OutOfMemoryError`s so ComfyUI can safely unload models and recover.
- **MiniMax-H3 Memory Override & Keyframe Auto-Rescale (`patches/minimax_h3_factor.py`)**:
  - Calibrates `MiniMaxH3.memory_usage_factor` from `0.114` to the measured `0.18` working set on Arc B580 to prevent mid-generation OOMs.
  - Automatically rescales conditioning keyframes/image references when sampling upscaled video latents (e.g. 18x32 -> 36x62), eliminating `shape mismatch [144, 96] vs [558, 96]` errors.
- **MiniMax-H3 Latent Upscaler XPU Patch (`minimax_upscaler_xpu.py`)**:
  - Dynamically patches `Comfyui_Minimax_h3_latent_Upscaler` (both 2D and 3D nodes) at runtime via meta-path import hooks and memory guards without modifying the original node files.
  - Automatically resolves `"xpu"` and seamlessly routes default `"cuda"` calls to Intel Arc XPU, updates node device schemas, and handles XPU VRAM caching cleanup.

### 3. DarkComfyX Theme & Appearance Port

- **DarkComfyX Theme (`web/js/darkcomfyx_theme.js`, `web/css/darkcomfyx.css`)**:
- **Based on the userstyle theme [Dark4chanX](https://uso.kkx.one/style/229613)**
  - Full authentic port of the DarkComfyX theme for ComfyUI based on the official [ComfyUI Appearance Guidelines](https://docs.comfy.org/interface/appearance).
  - **Charcoal Surfaces**: Base canvas `#191919`, node layers `#212121`, input card backgrounds `#181818`.
  - **4chan X Header Accent**: Clover header banner styling with `#498152` brand green top border.
  - **Signature Green-on-Dark Scrollbar**: WebKit & Firefox custom scrollbars with `#54cc66` rgba / `#527d4c` thumb and `#2c2c2c` track.
  - **2px Sharp Rounded Corners**: Subtle, clean 2px borders across dialogs, buttons, context menus, and node cards.
  - **Periwinkle Links & Gold Warnings**: Soft `#9BAED2` links and `#CB975B` gold headings / warning highlights.
  - **Live Greentext Highlighter**: Auto-highlights `>...` lines in green (`#85B76F`) across markdown and text prompts.
  - **Seamless ComfyUI Appearance Integration**:
    - Auto-registers `darkcomfyx` in `Comfy.CustomColorPalettes`.
    - Selectable directly via ComfyUI **Settings -> Appearance -> Color Palette -> DarkComfyX**.
    - Command Palette shortcut: `Apply DarkComfyX Theme`.
    - Standalone `user.css` and `darkcomfyx_palette.json` files for direct manual/portable use.

### 4. Toolchain & Dotfiles Launcher

- **`tools/install_theme.py`**:
  - One-click installer that registers DarkComfyX into `comfy.settings.json` and copies `user.css` into your ComfyUI user profile.
- **`scripts/launch_xpu.sh`**:
  - Optimized driver environment flags (`SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1`, `ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE`, unset `ONEAPI_DEVICE_SELECTOR`).
  - Relaxed Level Zero single-allocation limits (`UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1`).
  - Persistent Inductor compilation cache (`TORCHINDUCTOR_FX_GRAPH_CACHE=1`).
  - Tuned allocator configuration (`PYTORCH_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.85"`).
  - Auto-configures companion paths (`ComfyUI-AIMDO-XPU`).
  - Default launch arguments: `--enable-triton-backend --disable-dynamic-vram --reserve-vram 5`.
- **`tools/convert_upscale_models.py`**:
  - CLI batch tool to convert PyTorch upscale models (`.pth` / `.safetensors`) into OpenVINO ONNX models with automated output verification.
- **`scripts/setup.sh`**:
  - Automated one-command setup script for fresh vanilla ComfyUI installations.

---

## Installation & Restoring to Vanilla ComfyUI

### Quick Setup on a New ComfyUI Install

1. Clone vanilla ComfyUI and create your virtual environment:
   ```bash
   git clone https://github.com/comfyanonymous/ComfyUI.git
   cd ComfyUI
   python3 -m venv venv
   source venv/bin/activate
   pip install --pre --upgrade torch torchaudio torchvision triton-xpu --extra-index-url https://download.pytorch.org/whl/nightly/xpu
   pip install -r requirements.txt
   ```

2. Clone this suite into `custom_nodes/`:
   ```bash
   git clone https://github.com/Nacholmo/ComfyUI-Nacholmo-xpu-vibeslop custom_nodes/ComfyUI-Nacholmo-xpu-vibeslop
   ```

3. Run the setup script:
   ```bash
   # Standard setup:
   ./custom_nodes/ComfyUI-Nacholmo-xpu-vibeslop/scripts/setup.sh

   # Or setup with optional companions (AIMDO DynamicVRAM + VideoHelperSuite):
   ./custom_nodes/ComfyUI-Nacholmo-xpu-vibeslop/scripts/setup.sh --all
   ```

4. Launch ComfyUI:
   ```bash
   ./launch_xpu.sh --enable-triton-backend
   
    # Or reserve memory to run big models like minimax h3 and ltx2.5
   ./launch_xpu.sh --enable-triton-backend --reserve-vram 5
   ```

---

## Recommended Companion Custom Nodes

While this suite is fully standalone, the following companion custom nodes are recommended for specific workflows:

1. **[ComfyUI-AIMDO-XPU](https://github.com/allanmeng/ComfyUI-AIMDO-XPU)** (by Allan Meng):
   - Intel XPU replacement for `comfy-aimdo` DynamicVRAM offloading.
   - Install with: `git clone https://github.com/allanmeng/ComfyUI-AIMDO-XPU custom_nodes/ComfyUI-AIMDO-XPU`
   - `scripts/launch_xpu.sh` automatically detects and injects it into `PYTHONPATH`.

2. **[ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite)** (by Kosinkadink):
   - Required by `VideoCombineSync` for full video loading and frame sequence combining.
   - Install with: `git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite custom_nodes/comfyui-videohelpersuite`

---

## Directory Structure

```text
ComfyUI-Nacholmo-xpu-vibeslop/
├── .gitignore
├── LICENSE
├── README.md
├── pyproject.toml
├── requirements.txt
├── sitecustomize.py                           # Early Python bootstrap hook
├── prestartup_script.py                       # ComfyUI startup hook (activates guards)
├── __init__.py                                # Root custom node entry point
│
├── nodes/                                     # Node Implementations
│   ├── __init__.py
│   ├── arc_nodes.py                           # Arc Super Resolution & Resample FPS
│   ├── torch_compile_model.py                 # LowVRAM blockwise torch.compile
│   ├── upscale_model_video.py                 # Batched video upscaling + compile
│   ├── video_combine_sync.py                  # Video Combine with atempo audio sync
│   └── wint8/                                 # WINT8 Suite (Linux Port)
│       ├── LICENSE                            # MIT License (JWLHS)
│       ├── __init__.py
│       ├── wint8_model_quantizer.py
│       ├── wint8_model_loader.py
│       ├── wint8_lora_loader.py
│       ├── wint8_lora_stack.py
│       ├── wint8_xpu_ops.py
│       ├── wint8_quarot.py
│       └── wint8_lora_common.py
│
├── patches/                                   # Stability & Memory Patches
│   ├── __init__.py
│   ├── torchaudio_guard.py                    # TorchAudio CUDA fallback patch
│   ├── xpu_vram_guard.py                      # VRAM allocator guard
│   └── minimax_h3_factor.py                   # MiniMax H3 memory factor calibration
│
├── tools/                                     # CLI Utilities
│   ├── convert_upscale_models.py              # PyTorch to OpenVINO ONNX model converter
│   └── install_theme.py                       # DarkComfyX theme installer & configurator
│
├── web/                                       # Web Frontend Extensions
│   ├── js/
│   │   └── darkcomfyx_theme.js                # Extension logic, settings & greentext
│   ├── css/
│   │   └── darkcomfyx.css                     # Complete DarkComfyX stylesheet
│   └── darkcomfyx_palette.json                # Standalone ComfyUI palette JSON
│
├── scripts/                                   # Launch & Setup Automation
│   ├── launch_xpu.sh                          # Production Intel Arc B580 launcher
│   └── setup.sh                               # One-command installer & restorer
│
└── config/                                    # Configurations & Theme Presets
    ├── extra_model_paths.yaml.example         # External storage preset template
    ├── darkcomfyx_palette.json                # Standalone palette export
    └── darkcomfyx.user.css                    # Standalone user.css drop-in
```

---

## Converting Upscale Models for Arc Super Resolution

To convert PyTorch upscale models (`.pth` or `.safetensors`) into OpenVINO ONNX format for hardware acceleration:

```bash
# Convert all models found in upscale_models directories:
python custom_nodes/ComfyUI-Nacholmo-xpu-vibeslop/tools/convert_upscale_models.py

# Convert specific models:
python custom_nodes/ComfyUI-Nacholmo-xpu-vibeslop/tools/convert_upscale_models.py path/to/model.pth

# Force re-conversion:
python custom_nodes/ComfyUI-Nacholmo-xpu-vibeslop/tools/convert_upscale_models.py --force
```

---

## Acknowledgments & Licenses

- **Toolkit Core & Integrations**: Licensed under Apache-2.0.
- **DarkComfyX Theme**: Color palette and userstyle design inspired by & ported from [Dark4chanX](https://uso.kkx.one/style/229613) by myi pfdll / eXoNecro, dedicated under the **Creative Commons Zero v1.0 Universal (CC0-1.0)** Public Domain Dedication.
- **WINT8 Suite**: Derived from [JWLHS/ComfyUI-WINT8-XPU](https://github.com/JWLHS/ComfyUI-WINT8-XPU) under the **MIT License** (Copyright (c) 2026 JWLHS), ported to Linux and tuned for Intel Arc B580.
- **Video Combine Sync**: Extends [ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite) with duration-matching atempo audio filter chaining.
