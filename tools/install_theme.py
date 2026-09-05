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

def _find_comfy_root(start=None):
    """Walk up from CWD/script to find ComfyUI root (main.py + folder_paths.py)."""
    candidates = []
    if start:
        candidates.append(Path(start))
    candidates.append(Path.cwd())
    candidates.append(SCRIPT_DIR)
    for base in candidates:
        cur = base.resolve()
        while True:
            if (cur / "main.py").exists() and (cur / "folder_paths.py").exists():
                return cur
            parent = cur.parent
            if parent == cur:
                break
            cur = parent
    env_root = os.environ.get("COMFYUI_ROOT") or os.environ.get("COMFY_ROOT")
    if env_root:
        p = Path(env_root)
        if (p / "main.py").exists():
            return p
    return None


def find_comfyui_user_dirs(explicit=None):
    if explicit:
        p = Path(explicit).expanduser()
        return [p]
    candidates = []
    root = _find_comfy_root()
    if root is not None:
        candidates.append(root / "user" / "default")
        candidates.append(root / "user")
    # Portable home-based layouts (no hardcoded usernames).
    candidates.extend([
        Path.home() / ".config" / "ComfyUI" / "user" / "default",
        Path.home() / "ComfyUI" / "user" / "default",
        Path.home() / "ComfyUI" / "user",
    ])
    found = []
    for p in candidates:
        try:
            if p.exists() and p.is_dir() and p not in found:
                found.append(p)
        except Exception:
            continue
    return found


def _backup(path: Path):
    try:
        if path.exists():
            bak = path.with_suffix(path.suffix + ".bak")
            shutil.copy2(path, bak)
            print(f"  [•] Backed up {path.name} -> {bak.name}")
    except Exception as e:
        print(f"  [!] Warning: could not back up {path}: {e}")


def _atomic_write_json(path: Path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
        f.flush()
        try:
            os.fsync(f.fileno())
        except Exception:
            pass
    os.replace(tmp, path)

def install_theme(activate=True, target=None):
    print("=" * 60)
    print(" DarkComfyX Theme Installer for ComfyUI")
    print("=" * 60)

    if not PALETTE_FILE.exists():
        print(f"[!] Palette file not found at {PALETTE_FILE}")
        return False
    if not USER_CSS_FILE.exists():
        print(f"[!] user.css file not found at {USER_CSS_FILE}")
        return False

    try:
        with open(PALETTE_FILE, "r", encoding="utf-8") as f:
            palette_data = json.load(f)
    except Exception as e:
        print(f"[!] Invalid palette JSON at {PALETTE_FILE}: {e}")
        return False
    if not isinstance(palette_data, dict):
        print(f"[!] Invalid palette JSON at {PALETTE_FILE}: expected object")
        return False

    user_dirs = find_comfyui_user_dirs(explicit=target)
    if not user_dirs:
        print("[!] No ComfyUI user directories detected.")
        root = _find_comfy_root()
        default_target = (root / "user" / "default") if root else (Path.home() / "ComfyUI" / "user" / "default")
        try:
            default_target.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"[!] Could not create {default_target}: {e}")
            return False
        user_dirs = [default_target]

    for udir in user_dirs:
        print(f"\n[*] Processing ComfyUI user directory: {udir}")
        try:
            udir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"  [!] Skipping {udir}: cannot create directory ({e})")
            continue

        # 1. Copy user.css (with backup, atomic replace)
        target_css = udir / "user.css"
        _backup(target_css)
        try:
            tmp_css = udir / "user.css.tmp"
            shutil.copy2(USER_CSS_FILE, tmp_css)
            os.replace(tmp_css, target_css)
        except Exception as e:
            print(f"  [!] Failed to install user.css -> {target_css}: {e}")
            continue
        print(f"  [✓] Installed user.css -> {target_css}")

        # 2. Update comfy.settings.json (with backup, atomic write)
        settings_file = udir / "comfy.settings.json"
        settings = {}
        if settings_file.exists():
            try:
                with open(settings_file, "r", encoding="utf-8") as f:
                    settings = json.load(f)
                if not isinstance(settings, dict):
                    print(f"  [!] Warning: {settings_file} is not a JSON object; starting fresh")
                    settings = {}
            except Exception as e:
                print(f"  [!] Warning: Failed to read {settings_file}: {e}")
                settings = {}

        custom_palettes = settings.setdefault("Comfy.CustomColorPalettes", {})
        if not isinstance(custom_palettes, dict):
            custom_palettes = {}
            settings["Comfy.CustomColorPalettes"] = custom_palettes
        # Remove old key if present
        custom_palettes.pop("dark4chanx", None)
        custom_palettes["darkcomfyx"] = palette_data

        if activate:
            settings["Comfy.ColorPalette"] = "darkcomfyx"
            print("  [✓] Set active Comfy.ColorPalette = 'darkcomfyx'")

        _backup(settings_file)
        try:
            _atomic_write_json(settings_file, settings)
        except Exception as e:
            print(f"  [!] Failed to write {settings_file}: {e}")
            continue
        print(f"  [✓] Updated {settings_file}")

    print("\n[✓] DarkComfyX Theme successfully installed and configured!")
    print("    Refresh your browser or restart ComfyUI to see the changes.")
    return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Install DarkComfyX theme into ComfyUI user dir")
    parser.add_argument("--no-activate", action="store_true", help="Install palette without setting it active")
    parser.add_argument("--target", default=None, help="Explicit ComfyUI user directory (default: auto-detect)")
    args = parser.parse_args()
    install_theme(activate=not args.no_activate, target=args.target)
