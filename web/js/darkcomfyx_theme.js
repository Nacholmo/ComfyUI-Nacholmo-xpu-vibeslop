import { app } from "../../scripts/app.js";

/**
 * Nacholmo XPU Vibeslop - DarkComfyX Theme Extension
 * Authentic DarkComfyX port for ComfyUI: charcoal surfaces (#191919/#212121),
 * vanilla dot grid background texture, signature green-on-dark scrollbar,
 * 2px sharp corners, clover header accents, periwinkle links, and live greentext.
 */

const THEME_ID = "darkcomfyx";
const THEME_NAME = "DarkComfyX";
const STYLE_TAG_ID = "nacholmo-darkcomfyx-theme-style";

const BACKGROUND_GRID_BASE64 = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAGQAAABkCAIAAAD/gAIDAAAAGXRFWHRTb2Z0d2FyZQBBZG9iZSBJbWFnZVJlYWR5ccllPAAAAQBJREFUeNrs1rEKwjAUhlETUkj3vP9rdmr1Ysammk2w5wdxuLgcMHyptfawuZX4pJSWZTnfnu/lnIe/jNNxHHGNn//HNbbv+4dr6V+11uF527arU7+u63qfa/bnmh8sWLBgwYJlqRf8MEptXPBXJXa37BSl3ixYsGDBMliwFLyCV/DeLIMFCxYsWLBMwSt4Be/NggXLYMGCBUvBK3iNruC9WbBgwYJlsGApeAWv4L1ZBgsWLFiwYJmCV/AK3psFC5bBggULloJX8BpdwXuzYMGCBctgwVLwCl7Be7MMFixYsGDBsu8FH1FaSmExVfAxBa/gvVmwYMGCZbBg/W4vAQYA5tRF9QYlv/QAAAAASUVORK5CYII=";

const DARKCOMFYX_PALETTE = {
  id: THEME_ID,
  name: THEME_NAME,
  colors: {
    node_slot: {
      CLIP: "#E5C07B",
      CLIP_VISION: "#85B76F",
      CLIP_VISION_OUTPUT: "#CB975B",
      CONDITIONING: "#CB975B",
      CONTROL_NET: "#527D4C",
      IMAGE: "#729FCF",
      LATENT: "#B8A7C3",
      MASK: "#85B76F",
      MODEL: "#9BAED2",
      STYLE_MODEL: "#A6C48A",
      VAE: "#D03D3D",
      NOISE: "#838383",
      GUIDER: "#6EA477",
      SAMPLER: "#C3C3C3",
      SIGMAS: "#A0C090",
      TAESD: "#CB975B",
      PIPE_LINE: "#9BAED2",
      PIPE_LINE_SDXL: "#9BAED2",
      INT: "#527D4C",
      FLOAT: "#85B76F",
      STRING: "#C3C3C3",
      BOOLEAN: "#CB975B",
      LORA_STACK: "#527D4C",
      CONTROL_NET_STACK: "#527D4C",
      FAST_MODEL_LOADER: "#CB975B",
      SAMPLING: "#729FCF",
      AUDIO: "#CB975B",
      VIDEO: "#729FCF",
      SEGS: "#85B76F",
      BBOX: "#E5C07B"
    },
    litegraph_base: {
      BACKGROUND_IMAGE: BACKGROUND_GRID_BASE64,
      CLEAR_BACKGROUND_COLOR: "#191919",
      NODE_TITLE_COLOR: "#C3C3C3",
      NODE_SELECTED_TITLE_COLOR: "#FFFFFF",
      NODE_TEXT_SIZE: 14,
      NODE_TEXT_COLOR: "#C3C3C3",
      NODE_TEXT_HIGHLIGHT_COLOR: "#FFFFFF",
      NODE_SUBTEXT_SIZE: 12,
      NODE_DEFAULT_COLOR: "#212121",
      NODE_DEFAULT_BGCOLOR: "rgba(25, 25, 25, 0.95)",
      NODE_DEFAULT_BOXCOLOR: "#353535",
      NODE_DEFAULT_SHAPE: 2,
      NODE_BOX_OUTLINE_COLOR: "#527D4C",
      NODE_BYPASS_BGCOLOR: "rgba(82, 125, 76, 0.4)",
      NODE_ERROR_COLOUR: "#D03D3D",
      DEFAULT_SHADOW_COLOR: "rgba(0,0,0,0.5)",
      DEFAULT_GROUP_FONT: 24,
      WIDGET_BGCOLOR: "#1D1D1D",
      WIDGET_OUTLINE_COLOR: "#2C2C2C",
      WIDGET_TEXT_COLOR: "#C3C3C3",
      WIDGET_SECONDARY_TEXT_COLOR: "#838383",
      WIDGET_DISABLED_TEXT_COLOR: "#555555",
      LINK_COLOR: "#527D4C",
      EVENT_LINK_COLOR: "#CB975B",
      CONNECTING_LINK_COLOR: "#85B76F",
      BADGE_FG_COLOR: "#FFFFFF",
      BADGE_BG_COLOR: "#1B2A1A"
    },
    comfy_base: {
      "fg-color": "#C3C3C3",
      "bg-color": "#191919",
      "comfy-menu-bg": "#212121",
      "comfy-menu-secondary-bg": "#1D1D1D",
      "comfy-input-bg": "#181818",
      "input-text": "#C3C3C3",
      "descrip-text": "#838383",
      "drag-text": "#838383",
      "error-text": "#D03D3D",
      "border-color": "#262626",
      "tr-even-bg-color": "#191919",
      "tr-odd-bg-color": "#212121",
      "content-bg": "#212121",
      "content-fg": "#C3C3C3",
      "content-hover-bg": "#282828",
      "content-hover-fg": "#FFFFFF",
      "bar-shadow": "rgba(0, 0, 0, 0.4) 0 2px 6px"
    }
  }
};

/** Ensure the DarkComfyX palette is registered into ComfyUI CustomColorPalettes */
function registerColorPalette() {
  try {
    const customPalettes = app.ui.settings.getSettingValue("Comfy.CustomColorPalettes", {}) || {};
    customPalettes[THEME_ID] = DARKCOMFYX_PALETTE;
    app.ui.settings.setSettingValue("Comfy.CustomColorPalettes", customPalettes);
  } catch (err) {
    console.debug("[DarkComfyX] Palette registration notice:", err);
  }
}

/** Injects or removes the DarkComfyX CSS stylesheet */
function updateStyles() {
  const enableCSS = app.ui.settings.getSettingValue("Nacholmo.DarkComfyX.EnableCSS", true);
  const cloverHeader = app.ui.settings.getSettingValue("Nacholmo.DarkComfyX.CloverHeader", true);

  if (cloverHeader) {
    document.body.classList.add("darkcomfyx-clover-header");
  } else {
    document.body.classList.remove("darkcomfyx-clover-header");
  }

  let linkTag = document.getElementById(STYLE_TAG_ID);
  if (enableCSS) {
    if (!linkTag) {
      linkTag = document.createElement("link");
      linkTag.id = STYLE_TAG_ID;
      linkTag.rel = "stylesheet";
      linkTag.href = new URL("../css/darkcomfyx.css", import.meta.url).href;
      document.head.appendChild(linkTag);
    }
  } else if (linkTag) {
    linkTag.remove();
  }
}

/** Activate DarkComfyX as active Color Palette in ComfyUI */
function applyDarkComfyXTheme() {
  registerColorPalette();
  try {
    app.ui.settings.setSettingValue("Comfy.ColorPalette", THEME_ID);
    if (app.canvas) {
      app.canvas.background_image = BACKGROUND_GRID_BASE64;
      app.canvas.clear_background_color = "#191919";
      app.canvas.setDirty(true, true);
    }
    updateStyles();
  } catch (err) {
    console.error("[DarkComfyX] Failed to apply theme:", err);
  }
}

/** Greentext processor for markdown and textareas */
const GREENTEXT_CLASS = "darkcomfyx-greentext";
function processGreentext(scope = document.body) {
  const enabled = app.ui.settings.getSettingValue("Nacholmo.DarkComfyX.GreentextEngine", true);
  if (!enabled || !scope) return;

  const markdownBlocks = scope.querySelectorAll(".comfy-markdown-content, .markdown, .p-dialog-content");
  for (const block of markdownBlocks) {
    const walker = document.createTreeWalker(block, NodeFilter.SHOW_TEXT, {
      acceptNode(node) {
        if (!node.nodeValue || node.nodeValue.indexOf(">") === -1) return NodeFilter.FILTER_REJECT;
        if (node.parentElement && node.parentElement.closest("." + GREENTEXT_CLASS)) return NodeFilter.FILTER_REJECT;
        return NodeFilter.FILTER_ACCEPT;
      }
    });

    const targets = [];
    while (walker.nextNode()) targets.push(walker.currentNode);

    for (const textNode of targets) {
      const val = textNode.nodeValue;
      if (!/^(\s*)>/m.test(val)) continue;
      const lines = val.split("\n");
      const frag = document.createDocumentFragment();
      let modified = false;

      lines.forEach((line, idx) => {
        if (line.trimStart().startsWith(">")) {
          modified = true;
          const span = document.createElement("span");
          span.className = GREENTEXT_CLASS;
          span.textContent = line;
          frag.appendChild(span);
        } else if (line.length > 0) {
          frag.appendChild(document.createTextNode(line));
        }
        if (idx < lines.length - 1) frag.appendChild(document.createTextNode("\n"));
      });

      if (modified && textNode.parentNode) {
        textNode.parentNode.replaceChild(frag, textNode);
      }
    }
  }
}

app.registerExtension({
  name: "Nacholmo.DarkComfyXTheme",

  commands: [
    {
      id: "Nacholmo.DarkComfyX.ApplyTheme",
      label: "Apply DarkComfyX Theme",
      menubarLabel: "Apply DarkComfyX Theme",
      function: applyDarkComfyXTheme
    }
  ],

  menuCommands: [
    {
      path: ["Appearance", "DarkComfyX Theme"],
      commands: ["Nacholmo.DarkComfyX.ApplyTheme"]
    }
  ],

  settings: [
    {
      id: "Nacholmo.DarkComfyX.EnableCSS",
      name: "Enable DarkComfyX UI Styling",
      category: ["Nacholmo XPU", "DarkComfyX Theme", "Enable CSS"],
      tooltip: "Apply DarkComfyX styles: green-on-dark scrollbar, 2px borders, periwinkle links, and charcoal panels.",
      type: "boolean",
      defaultValue: true,
      onChange: updateStyles
    },
    {
      id: "Nacholmo.DarkComfyX.CloverHeader",
      name: "4chan Clover Header Accent",
      category: ["Nacholmo XPU", "DarkComfyX Theme", "Clover Header"],
      tooltip: "Show the subtle 4chan clover background gradient along the top header bar.",
      type: "boolean",
      defaultValue: true,
      onChange: updateStyles
    },
    {
      id: "Nacholmo.DarkComfyX.GreentextEngine",
      name: "Greentext Syntax Highlighting",
      category: ["Nacholmo XPU", "DarkComfyX Theme", "Greentext"],
      tooltip: "Highlight lines starting with '>' in green (#85B76F) across prompt notes and markdown text.",
      type: "boolean",
      defaultValue: true,
      onChange: () => processGreentext(document.body)
    }
  ],

  async setup() {
    registerColorPalette();
    updateStyles();

    // If active palette is DarkComfyX, enforce canvas grid background
    const activePalette = app.ui.settings.getSettingValue("Comfy.ColorPalette");
    if (activePalette === THEME_ID && app.canvas) {
      app.canvas.background_image = BACKGROUND_GRID_BASE64;
      app.canvas.clear_background_color = "#191919";
      app.canvas.setDirty(true, true);
    }

    // Debounced observer for live greentext formatting
    let scheduled = false;
    const observer = new MutationObserver(() => {
      if (scheduled) return;
      scheduled = true;
      requestAnimationFrame(() => {
        scheduled = false;
        processGreentext(document.body);
      });
    });

    observer.observe(document.body, { childList: true, subtree: true });
    processGreentext(document.body);
  }
});

export { DARKCOMFYX_PALETTE, applyDarkComfyXTheme };
