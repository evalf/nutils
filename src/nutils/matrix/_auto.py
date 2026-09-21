from ._base import BackendNotAvailable

try:
    from . import _mkl as _backend
except BackendNotAvailable:
    try:
        from . import _scipy as _backend
    except BackendNotAvailable:
        from . import _numpy as _backend

assemble = _backend.assemble
