# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import pickle
import threading
from unittest.mock import patch

import numpy

# Guards `pickle.load` patching in `safe_numpy_load`, so concurrent calls from multiple
# threads don't race on the monkeypatch.
_pickle_patch_lock = threading.Lock()


def _safe_pickle_load(file, **kwargs):
    """Safe version of `pickle.load`."""
    return SafeUnpickler(file, **kwargs).load()


def safe_numpy_load(path, **kwargs):
    """Safe version of `numpy.load` which calls `pickle.load`."""
    with _pickle_patch_lock:
        with patch('pickle.load', _safe_pickle_load):
            return numpy.load(path, **kwargs)


class SafeUnpickler(pickle.Unpickler):
    """A `pickle.Unpickler` that only constructs classes/functions explicitly allowlisted
    in `_SAFE_CLASSES`, to protect against arbitrary code execution when unpickling data
    from disk. Use `add_safe_classes` to extend the allowlist for custom payload types.

    Usage: `SafeUnpickler(file).load()` instead of `pickle.load(file)`.
    """

    _SAFE_CLASSES: set = {
        ("numpy", "ndarray"),
        ("numpy", "dtype"),
        ("numpy._core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "scalar"),
        ("PIL.Image", "Image"),
    }

    def find_class(self, module: str, name: str):
        if (module, name) not in self._SAFE_CLASSES:
            raise pickle.UnpicklingError(
                f"Refusing to unpickle disallowed class '{module}.{name}' "
            )
        return super().find_class(module, name)
