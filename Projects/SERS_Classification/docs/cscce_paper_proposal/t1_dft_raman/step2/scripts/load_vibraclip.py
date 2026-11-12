"""Load the VibraCLIP experimental pickle with an unpickler that only allows numpy array reconstruction."""
import pickle
import sys

import numpy as np

ALLOWED = {("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"),
           ("numpy", "ndarray"), ("numpy", "dtype"), ("numpy.core.multiarray", "scalar"), ("numpy._core.multiarray", "scalar")}


class Safe(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) not in ALLOWED:
            raise pickle.UnpicklingError(f"blocked {module}.{name}")
        if module.startswith("numpy.core"):
            module = module.replace("numpy.core", "numpy._core") if not hasattr(np, "core") else module
        return super().find_class(module, name)


def load(path):
    with open(path, "rb") as f:
        return Safe(f).load()


if __name__ == "__main__":
    d = load(sys.argv[1])
    print(type(d), len(d) if hasattr(d, "__len__") else "")
    first = d[0] if isinstance(d, list) else d
    if isinstance(first, dict):
        for k, v in first.items():
            print(k, type(v).__name__, getattr(v, "shape", v if not hasattr(v, "__len__") or len(str(v)) < 80 else str(v)[:80]))
    if isinstance(d, dict):
        for k, v in d.items():
            print(k, type(v).__name__, len(v) if hasattr(v, "__len__") else v)
