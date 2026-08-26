# ComfyUI Intel Arc & XPU Suite

A comprehensive, unified performance toolkit, custom node suite, and launcher environment for running **ComfyUI** on **Intel Arc GPUs** (Alchemist, Battlemage Xe2, Xe-LPG/HPG) and PyTorch XPU.

Designed as an all-in-one "dotfiles & node pack" to restore your complete custom Intel Arc setup onto any fresh, vanilla ComfyUI installation with a single command.

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
- **WINT8 Suite (`WINT8ModelQuantizer`, `WINT8ModelLoader`, `WINT8LoRALoader`, `WINT8LoRAStack`)**:
  - Pure PyTorch INT8 per-row UNet quantization and loading.
  - Multi-LoRA stacking (up to 5 LoRAs) baked directly onto INT8 weights.
  - Custom XPU-accelerated GEMM kernels with Hadamard rotations (QuaRot).
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

### 2. Runtime Stability & Calibration Patches

- **VRAM Guard (`xpu_vram_guard.py` / `prestartup_script.py`)**:
  - Automatically caps the PyTorch XPU caching allocator fraction (default: `0.75` / `75%` of VRAM).
  - On Intel Level Zero drivers, exceeding physical VRAM during sudden activation spikes can hard-lock the GPU kernel and freeze the Linux desktop. VRAM Guard intercepts spikes and raises clean, recoverable Python `OutOfMemoryError`s so ComfyUI can safely unload models and recover.
- **MiniMax-H3 Memory Override**:
  - Calibrates `MiniMaxH3.memory_usage_factor` from `0.114` to the measured `0.18` working set on Arc B580 to prevent mid-generation OOMs.

### 3. Toolchain & Dotfiles Launcher

- **`scripts/launch_xpu.sh`**:
  - Optimized driver environment flags (`SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1`, `ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE`, unset `ONEAPI_DEVICE_SELECTOR`).
  - Relaxed Level Zero single-allocation limits (`UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1`).
  - Persistent Inductor compilation cache (`TORCHINDUCTOR_FX_GRAPH_CACHE=1`).
  - Tuned allocator configuration (`PYTORCH_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.85"`).
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
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
   pip install -r requirements.txt
   ```

2. Clone this suite into `custom_nodes/`:
   ```bash
   git clone <YOUR_REPO_URL> custom_nodes/ComfyUI-Intel-Arc-Suite
   ```

3. Run the setup script:
   ```bash
   ./custom_nodes/ComfyUI-Intel-Arc-Suite/scripts/setup.sh
   ```

4. Launch ComfyUI:
   ```bash
   ./launch_xpu.sh
   ```

---

## Directory Structure

```text
ComfyUI-Intel-Arc-Suite/
├── .gitignore
├── LICENSE
├── README.md
├── pyproject.toml
├── requirements.txt
├── prestartup_script.py                       # Startup hook (activates VRAM Guard)
├── __init__.py                                # Root custom node entry point
│
├── nodes/                                     # Node Implementations
│   ├── __init__.py
│   ├── arc_nodes.py                           # Arc Super Resolution & Resample FPS
│   ├── torch_compile_model.py                 # LowVRAM blockwise torch.compile
│   ├── upscale_model_video.py                 # Batched video upscaling + compile
│   ├── video_combine_sync.py                  # Video Combine with atempo audio sync
│   └── wint8/                                 # WINT8 Quantization & LoRA Suite
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
│   ├── xpu_vram_guard.py                      # VRAM allocator guard
│   └── minimax_h3_factor.py                   # MiniMax H3 memory factor calibration
│
├── tools/                                     # CLI Utilities
│   └── convert_upscale_models.py              # PyTorch to OpenVINO ONNX model converter
│
├── scripts/                                   # Launch & Setup Automation
│   ├── launch_xpu.sh                          # Production Intel Arc B580 launcher
│   └── setup.sh                               # One-command installer & restorer
│
└── config/                                    # Path Configurations
    └── extra_model_paths.yaml.example         # External storage preset template
```

---

## Converting Upscale Models for Arc Super Resolution

To convert PyTorch upscale models (`.pth` or `.safetensors`) into OpenVINO ONNX format for hardware acceleration:

```bash
# Convert all models found in upscale_models directories:
python custom_nodes/ComfyUI-Intel-Arc-Suite/tools/convert_upscale_models.py

# Convert specific models:
python custom_nodes/ComfyUI-Intel-Arc-Suite/tools/convert_upscale_models.py path/to/model.pth

# Force re-conversion:
python custom_nodes/ComfyUI-Intel-Arc-Suite/tools/convert_upscale_models.py --force
```

---

## System Requirements

- **OS**: Linux (kernel 6.8+ recommended for Intel Arc Battlemage / Alchemist)
- **Intel Compute Runtime**: `intel-opencl-icd` / `intel-compute-runtime`
- **PyTorch**: PyTorch with XPU support (`torch>=2.4.0` with `--index-url https://download.pytorch.org/whl/xpu`)
- **OpenVINO**: `openvino>=2024.0.0`

---

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.
