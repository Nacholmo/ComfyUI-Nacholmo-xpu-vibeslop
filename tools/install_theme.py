#!/usr/bin/env python3
"""
Install and Activate DarkComfyX Theme for ComfyUI
─────────────────────────────────────────────────
Copies darkcomfyx.user.css and updates comfy.settings.json
with the DarkComfyX color palette.
"""

import json
import os
import shutil
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
CONFIG_DIR = REPO_ROOT / "config"
PALETTE_FILE = CONFIG_DIR / "darkcomfyx_palette.json"
USER_CSS_FILE = CONFIG_DIR / "darkcomfyx.user.css"

def find_comfyui_user_dirs():
    candidates = [
        Path.home() / ".config" / "ComfyUI" / "user",
        Path.home() / "ComfyUI" / "user" / "default",
        Path.home() / "ComfyUI" / "user",
        Path("/home/sundae/ComfyUI/user/default"),
        Path("/home/sundae/Drives/Aurora/ComfyUI/user/default"),
        Path("/home/sundae/Drives/Aurora/ComfyUI/user"),
    ]
    found = []
    for p in candidates:
        if p.exists() and p.is_dir() and p not in found:
            found.append(p)
    return found

def install_theme(activate=True):
    print("=" * 60)
    print(" DarkComfyX Theme Installer for ComfyUI")
    print("=" * 60)

    if not PALETTE_FILE.exists():
        print(f"[!] Palette file not found at {PALETTE_FILE}")
        return False
    if not USER_CSS_FILE.exists():
        print(f"[!] user.css file not found at {USER_CSS_FILE}")
        return False

    with open(PALETTE_FILE, "r", encoding="utf-8") as f:
        palette_data = json.load(f)

    user_dirs = find_comfyui_user_dirs()
    if not user_dirs:
        print("[!] No ComfyUI user directories detected.")
        default_target = Path.home() / "ComfyUI" / "user" / "default"
        default_target.mkdir(parents=True, exist_ok=True)
        user_dirs = [default_target]

    for udir in user_dirs:
        print(f"\n[*] Processing ComfyUI user directory: {udir}")

        # 1. Copy user.css
        target_css = udir / "user.css"
        shutil.copy2(USER_CSS_FILE, target_css)
        print(f"  [✓] Installed user.css -> {target_css}")

        # 2. Update comfy.settings.json
        settings_file = udir / "comfy.settings.json"
        settings = {}
        if settings_file.exists():
            try:
                with open(settings_file, "r", encoding="utf-8") as f:
                    settings = json.load(f)
            except Exception as e:
                print(f"  [!] Warning: Failed to read {settings_file}: {e}")
                settings = {}

        custom_palettes = settings.setdefault("Comfy.CustomColorPalettes", {})
        # Remove old key if present
        custom_palettes.pop("dark4chanx", None)
        custom_palettes["darkcomfyx"] = palette_data

        if activate:
            settings["Comfy.ColorPalette"] = "darkcomfyx"
            print("  [✓] Set active Comfy.ColorPalette = 'darkcomfyx'")

        with open(settings_file, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=4)
        print(f"  [✓] Updated {settings_file}")

    print("\n[✓] DarkComfyX Theme successfully installed and configured!")
    print("    Refresh your browser or restart ComfyUI to see the changes.")
    return True


if __name__ == "__main__":
    activate = "--no-activate" not in sys.argv
    install_theme(activate=activate)
