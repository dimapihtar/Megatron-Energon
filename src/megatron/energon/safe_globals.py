import torch


SAFE_GLOBALS = [
]


def register_safe_globals():
    """Register megatron-energon safe classes with torch serialization."""
    for cls in SAFE_GLOBALS:
        torch.serialization.add_safe_globals([cls])
