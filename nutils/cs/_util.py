import numpy

def overdimensioned(*args):
    ncoeffs = len(args[0]) - 1
    matrix = []
    values = []
    for v, *coeffs in args:
        assert len(coeffs) == ncoeffs
        if v is not None:
            matrix.append(coeffs)
            values.append(v)
    if len(values) != ncoeffs:
        raise ValueError(f'exactly {ncoeffs} arguments should be specified')
    mat = numpy.linalg.solve(matrix, values).T
    return [mat @ coeffs for v, *coeffs in args]
