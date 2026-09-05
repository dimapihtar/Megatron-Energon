import torch

from megatron.energon.rng import SystemRngState
from megatron.energon.savable_loader import (
    SavableDataLoaderState,
    SavableDatasetCheckpoint,
    SavableDatasetState,
)
from megatron.energon.state import FlexState

SAFE_GLOBALS = [
    SavableDataLoaderState,
    SavableDatasetCheckpoint,
    SavableDatasetState,
    SystemRngState,
    FlexState,
]


def register_safe_globals():
    """Register megatron-energon safe classes with torch serialization."""
    for cls in SAFE_GLOBALS:
        torch.serialization.add_safe_globals([cls])
