from ._base import BackendNotAvailable

try:
    from ._mkl import assemble, assemble_csr
except BackendNotAvailable:
    try:
        from ._scipy import assemble, assemble_csr
    except BackendNotAvailable:
        from ._numpy import assemble, assemble_csr
