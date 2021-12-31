from ._base import BackendNotAvailable

try:
    from ._mkl import assemble
except BackendNotAvailable:
    try:
        from ._scipy import assemble
    except BackendNotAvailable:
        from ._numpy import assemble
