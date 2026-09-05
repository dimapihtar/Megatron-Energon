# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import numpy
import pickle
from unittest.mock import patch


def _safe_pickle_load(file, **kwargs):
    """Safe version of `pickle.load`."""
    return SafeUnpickler(file, **kwargs).load()


def safe_numpy_load(path, **kwargs):
    """Safe version of `numpy.load` which calls `pickle.load`."""
    with _pickle_patch_lock:
        with patch('pickle.load', _safe_pickle_load):
            return numpy.load(path, **kwargs)


class SafeUnpickler(pickle.Unpickler):
    """ """

    _SAFE_CLASSES: frozenset = frozenset(
        {
        }
    )

    def find_class(self, module: str, name: str):
        if (module, name) not in self._SAFE_CLASSES:
            raise pickle.UnpicklingError(
                f"Refusing to unpickle disallowed class '{module}.{name}' "
            )
        return super().find_class(module, name)
