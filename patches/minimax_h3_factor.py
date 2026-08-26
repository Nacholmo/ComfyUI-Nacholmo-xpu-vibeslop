"""Keeps the measured MiniMaxH3 memory_usage_factor calibration across ComfyUI updates.

The sampling working set measured on an Arc B580 with int8 weights needs ~0.18
(issue #15781 shortfall); upstream still ships 0.114. Only applied while
upstream still has 0.114, so a real upstream fix is never clobbered.
"""

import logging

MEASURED_FACTOR = 0.18


def apply():
    try:
        import comfy.supported_models
        if not hasattr(comfy.supported_models, "MiniMaxH3"):
            return
        _current = getattr(comfy.supported_models.MiniMaxH3, "memory_usage_factor", None)
        if _current is not None and _current != MEASURED_FACTOR:
            if _current != 0.114:
                logging.warning(
                    "[minimax-h3-memory-factor] upstream MiniMaxH3.memory_usage_factor changed to %s, "
                    "revisit this override.", _current)
            comfy.supported_models.MiniMaxH3.memory_usage_factor = MEASURED_FACTOR
    except Exception as e:
        logging.debug(f"[minimax-h3-memory-factor] could not apply override: {e}")


apply()

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ["apply", "NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
