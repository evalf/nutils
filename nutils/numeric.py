"""
The numeric module provides methods that are lacking from the numpy module.
"""

from . import types, warnings
import numpy
import numbers
import builtins
import collections.abc

_abc = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'  # indices for einsum


def round(arr):
    return numpy.round(arr).astype(int)


def floor(arr):
    return numpy.floor(arr).astype(int)


def ceil(arr):
    return numpy.ceil(arr).astype(int)


def overlapping(arr, axis=-1, n=2):
    'reinterpret data with overlaps'

    arr = numpy.asarray(arr)
    axis = normdim(arr.ndim, axis)
    overlapping = numpy.ndarray(buffer=arr, dtype=arr.dtype,
                                shape=(*arr.shape[:axis], arr.shape[axis]-n+1, n, *arr.shape[axis+1:]),
                                strides=arr.strides[:axis+1] + arr.strides[axis:])
    overlapping.flags.writeable = False
    return overlapping


def full(shape, fill_value, dtype):
    'read-only equivalent to :func:`numpy.full`'

    z = (0,)*len(shape)
    f = numpy.ndarray(shape=shape, strides=z, dtype=dtype)
    if f.size:
        f[z] = fill_value
    f.flags.writeable = False
    return f


def normdim(ndim: int, n: int) -> int:
    'check bounds and make positive'

    assert isint(ndim) and ndim >= 0, 'ndim must be positive integer, got {}'.format(ndim)
    if n < 0:
        n += ndim
    if n < 0 or n >= ndim:
        raise IndexError('index out of bounds: {} not in [0,{})'.format(n, ndim))
    return n


def get(arr, axis, item):
    'take single item from array axis'

    arr = numpy.asarray(arr)
    axis = normdim(arr.ndim, axis)
    return arr[(slice(None),) * axis + (item,)]


def contract(A, B, axis=-1):
    'contract'

    A = numpy.asarray(A)
    B = numpy.asarray(B)

    maxdim = max(A.ndim, B.ndim)
    m = _abc[maxdim-A.ndim:maxdim]
    n = _abc[maxdim-B.ndim:maxdim]

    axes = sorted([normdim(maxdim, axis)] if isinstance(axis, int) else [normdim(maxdim, ax) for ax in axis])
    o = _abc[:maxdim-len(axes)] if axes == range(maxdim-len(axes), maxdim) \
        else ''.join(_abc[a+1:b] for a, b in zip([-1]+axes, axes+[maxdim]) if a+1 != b)

    return numpy.einsum('{},{}->{}'.format(m, n, o), A, B, optimize=False)


def dot(A, B, axis=-1):
    '''Transform axis of A by contraction with first axis of B and inserting
       remaining axes. Note: with default axis=-1 this leads to multiplication of
       vectors and matrices following linear algebra conventions.'''

    A = numpy.asarray(A)
    B = numpy.asarray(B)

    m = _abc[:A.ndim]
    x = _abc[A.ndim:A.ndim+B.ndim-1]
    n = m[axis] + x
    o = m[:axis] + x
    if axis != -1:
        o += m[axis+1:]

    return numpy.einsum('{},{}->{}'.format(m, n, o), A, B, optimize=False)


def meshgrid(*args, dtype=None):
    '''Multi-dimensional meshgrid generalisation.

    Meshgrid stacks ``n`` arbitry-dimensional arrays into an array that is one
    dimension higher than all dimensions combined, such that ``retval[i]`` equals
    ``args[i]`` broadcasted to consecutive dimension slices. For two vector
    arguments this is almost equal to :func:`numpy.meshgrid`, with the main
    difference that dimensions are not swapped in the return values. The other
    difference is that the return value is a single array, but since the stacked
    axis is the first dimension the result can always be tuple unpacked.

    Parameters
    ----------
    args : sequence of :class:`numpy.ndarray` objects or equivalent
      The arrays that are to be grid-stacked.
    dtype : :class:`type` of output array
      If unspecified the dtype is determined automatically from the input arrays
      using :func:`numpy.result_type`.

    Returns
    -------
    :class:`numpy.ndarray`
    '''

    args = [numpy.asarray(arg) for arg in args]
    shape = [len(args)]
    for arg in args:
        shape.extend(arg.shape)
    if dtype is None:
        dtype = numpy.result_type(*(arg.dtype for arg in args))
    grid = numpy.empty(shape, dtype=dtype)
    n = len(shape)-1
    for i, arg in enumerate(args):
        n -= arg.ndim
        grid[i] = arg[(...,)+(numpy.newaxis,)*n]
    assert n == 0
    return grid


def _simplex_grid_helper(bounds):
    if bounds.ndim != 1 or len(bounds) == 0:
        raise ValueError
    nd = len(bounds)
    spacing = [numpy.sqrt((1+i/2)/(1+i)) for i in range(nd)]  # simplex height orthogonal to lower dimension
    grid = meshgrid(*[numpy.arange(bound, step=step) for step, bound in zip(spacing, bounds)])
    out_of_bounds = []
    for idim in range(nd-1):
        stripes = grid[(idim,)+(slice(None),)*(idim+1)+(slice(1, None, 2),)]
        stripes += spacing[idim] * (idim+1) / (idim+2)
        if stripes.size and stripes.flat[-1] >= bounds[idim]:
            out_of_bounds.append(idim)
    if out_of_bounds:
        select = numpy.ones(grid.shape[1:], dtype=bool)
        for idim in out_of_bounds:
            select[(slice(None),)*(idim)+(-1,)+(slice(1, None, 2),)] = False
        points = grid[:, select].T
    else:
        points = grid.reshape(nd, -1).T
    d = numpy.subtract(bounds, points.max(axis=0))
    assert (d > 0).all()
    points += d / 2
    return points


def simplex_grid(shape, spacing):
    '''Multi-dimensional generator for equilateral simplex grids.

    Simplex_grid generates a point cloud within an n-dimensional orthotope, which
    ranges from zero to a specified shape. The point coordinates are spaced in
    such a way that the nearest neighbours are at distance `spacing`, thus
    forming vertices of regular simplices. The returned array is two-dimensional,
    with the first axis being the spatial dimension (matching `shape`) and the
    second a stacking of the generated points.

    Parameters
    ----------
    shape : :class:`tuple`
      list or tuple of dimensions of the orthotope to be filled.
    spacing : :class:`float`
      minimum spacing in the generated point cloud.

    Returns
    -------
    :class:`numpy.ndarray`
    '''

    return _simplex_grid_helper(numpy.divide(shape, spacing)) * spacing


def takediag(A, axis=-2, rmaxis=-1):
    axis = normdim(A.ndim, axis)
    rmaxis = normdim(A.ndim, rmaxis)
    assert axis < rmaxis
    fmt = _abc[:rmaxis] + _abc[axis] + _abc[rmaxis:A.ndim-1] + '->' + _abc[:A.ndim-1]
    return numpy.einsum(fmt, A, optimize=False)


def normalize(A, axis=-1):
    'devide by normal'

    s = [slice(None)] * A.ndim
    s[axis] = numpy.newaxis
    return A / numpy.linalg.norm(A, axis=axis)[tuple(s)]


def diagonalize(arg, axis=-1, newaxis=-1):
    'insert newaxis, place axis on diagonal of axis and newaxis'
    axis = normdim(arg.ndim, axis)
    newaxis = normdim(arg.ndim+1, newaxis)
    assert 0 <= axis < newaxis <= arg.ndim
    diagonalized = numpy.zeros(arg.shape[:newaxis]+(arg.shape[axis],)+arg.shape[newaxis:], arg.dtype)
    diag = takediag(diagonalized, axis, newaxis)
    assert diag.base is diagonalized
    diag.flags.writeable = True
    diag[:] = arg
    return diagonalized


def inv(A):
    '''Matrix inverse.

    Fully equivalent to :func:`numpy.linalg.inv`, with the exception that upon
    singular systems :func:`~nutils.numeric.inv` does not raise a ``LinAlgError``, but rather
    issues a ``RuntimeWarning`` and returns NaN (not a number) values. For
    arguments of dimension >2 the return array contains NaN values only for those
    entries that correspond to singular matrices.
    '''

    try:
        Ainv = numpy.linalg.inv(A)
    except numpy.linalg.LinAlgError:
        warnings.warn('singular matrix', RuntimeWarning)
        Ainv = numpy.empty(A.shape, dtype=complex if A.dtype.kind == 'c' else float)
        for index in numpy.ndindex(A.shape[:-2]):
            try:
                Ainv[index] = numpy.linalg.inv(A[index])
            except numpy.linalg.LinAlgError:
                Ainv[index] = numpy.nan
    return Ainv


isarray = lambda a: isinstance(a, numpy.ndarray)
isboolarray = lambda a: isarray(a) and a.dtype == bool
isbool = lambda a: isboolarray(a) and a.ndim == 0 or isinstance(a, (bool, numpy.bool_))
isint = lambda a: isinstance(a, numbers.Integral)
isnumber = lambda a: isinstance(a, numbers.Number)
isintarray = lambda a: isarray(a) and numpy.issubdtype(a.dtype, numpy.integer)
asobjvector = lambda v: numpy.array((None,)+tuple(v), dtype=object)[1:]  # 'None' prevents interpretation of objects as axes


def blockdiag(args):
    args = [numpy.asarray(arg) for arg in args]
    args = [arg[numpy.newaxis, numpy.newaxis] if arg.ndim == 0 else arg for arg in args]
    assert all(arg.ndim == 2 for arg in args)
    shapes = numpy.array([arg.shape for arg in args])
    blockdiag = numpy.zeros(shapes.sum(0))
    for arg, (i, j) in zip(args, shapes.cumsum(0)):
        blockdiag[i-arg.shape[0]:i, j-arg.shape[1]:j] = arg
    return blockdiag


def nanjoin(args, axis=0):
    args = [numpy.asarray(arg) for arg in args]
    assert args
    assert axis >= 0
    shape = list(args[0].shape)
    shape[axis] = sum(arg.shape[axis] for arg in args) + len(args) - 1
    concat = numpy.empty(shape, dtype=float)
    concat[:] = numpy.nan
    i = 0
    for arg in args:
        j = i + arg.shape[axis]
        concat[(slice(None),)*axis+(slice(i, j),)] = arg
        i = j + 1
    return concat


def ix(args):
    'version of :func:`numpy.ix_` that allows for scalars'
    args = tuple(numpy.asarray(arg) for arg in args)
    assert all(0 <= arg.ndim <= 1 for arg in args)
    idims = numpy.cumsum([0] + [arg.ndim for arg in args])
    ndims = idims[-1]
    return [arg.reshape((1,)*idim+(arg.size,)+(1,)*(ndims-idim-1)) for idim, arg in zip(idims, args)]


class Broadcast1D:
    def __init__(self, arg):
        self.arg = numpy.asarray(arg)
        self.shape = self.arg.shape
        self.size = self.arg.size

    def __iter__(self):
        return ((item,) for item in self.arg.flat)


broadcast = lambda *args: numpy.broadcast(*args) if len(args) > 1 else Broadcast1D(args[0])


def ext(A):
    """Exterior
    For array of shape (n,n-1) return n-vector ex such that ex.array = 0 and
    det(arr;ex) = ex.ex"""
    A = numpy.asarray(A)
    assert A.ndim == 2 and A.shape[0] == A.shape[1]+1
    if len(A) == 1:
        ext = numpy.ones(1)
    elif len(A) == 2:
        ((a,), (b,)) = A
        ext = numpy.array((b, -a))
    elif len(A) == 3:
        ((a, b), (c, d), (e, f)) = A
        ext = numpy.array((c*f-e*d, e*b-a*f, a*d-c*b))
    else:
        raise NotImplementedError('shape={}'.format(A.shape))
    return ext


def unpack(n, atol, rtol):
    '''Convert packed representation to floating point data.

    The packed binary form is a floating point interpretation of signed integer
    data, such that any integer ``n`` maps onto float ``a`` as follows:

    .. code-block:: none

        a = nan                       if n = -N-1
        a = -inf                      if n = -N
        a = sinh(n*rtol)*atol/rtol    if -N < n < N
        a = +inf                      if n = N,

    where ``N = 2**(nbits-1)-1`` is the largest representable signed integer.

    Note that packing is both order and zero preserving. The transformation is
    designed such that the spacing around zero equals ``atol``, while the
    relative spacing for most of the data range is approximately constant at
    ``rtol``. Precisely, the spacing between a value ``a`` and the adjacent value
    is ``sqrt(atol**2 + (a*rtol)**2)``. Note that the truncation error equals
    half the spacing.

    The representable data range depends on the values of ``atol`` and ``rtol``
    and the bitsize of ``n``. Useful values for different data types are:

    =====  ====  =====  =====
    dtype  rtol  atol   range
    =====  ====  =====  =====
    int8   2e-1  2e-06  4e+05
    int16  2e-3  2e-15  1e+16
    int32  2e-7  2e-96  2e+97
    =====  ====  =====  =====

    Args
    ----
    n : :class:`int` array
        Integer data.
    atol : :class:`float`
        Absolute tolerance.
    rtol : :class:`float`
        Relative tolerance.

    Returns
    -------
    :class:`float` array
    '''

    iinfo = numpy.iinfo(n.dtype)
    assert iinfo.dtype.kind == 'i', 'data should be of signed integer type'
    a = numpy.asarray(numpy.sinh(n*rtol)*(atol/rtol))
    a[numpy.equal(n, iinfo.max)] = numpy.inf
    a[numpy.equal(n, -iinfo.max)] = -numpy.inf
    a[numpy.equal(n, iinfo.min)] = numpy.nan
    return a[()]


def pack(a, atol, rtol, dtype):
    '''Lossy compression of floating point data.

    See :func:`unpack` for the definition of the packed binary form. The converse
    transformation uses rounding in packed domain to determine the closest
    matching value. In particular this may lead to values falling outside the
    representable data range to be clipped to infinity. Some examples of packed
    truncation:

    >>> def truncate(a, dtype, **tol):
    ...   return unpack(pack(a, dtype=dtype, **tol), **tol)
    >>> truncate(0.5, dtype='int16', atol=2e-15, rtol=2e-3)
    0.5004...
    >>> truncate(1, dtype='int16', atol=2e-15, rtol=2e-3)
    0.9998...
    >>> truncate(2, dtype='int16', atol=2e-15, rtol=2e-3)
    2.0013...
    >>> truncate(2, dtype='int16', atol=2e-15, rtol=2e-4)
    inf
    >>> truncate(2, dtype='int32', atol=2e-15, rtol=2e-4)
    2.00013...

    Args
    ----
    a : :class:`float` array
      Input data.
    atol : :class:`float`
      Absolute tolerance.
    rtol : :class:`float`
      Relative tolerance.
    dtype : :class:`str` or numpy dtype
      Target dtype for packed data.

    Returns
    -------
    :class:`int` array.
    '''

    iinfo = numpy.iinfo(dtype)
    assert iinfo.dtype.kind == 'i', 'dtype should be a signed integer'
    amax = numpy.sinh(iinfo.max*rtol)*(atol/rtol)
    a = numpy.asarray(a)
    n = numpy.asarray((numpy.arcsinh(a.clip(-amax, amax)*(rtol/atol))/rtol).round().astype(iinfo.dtype))
    if numpy.logical_and(numpy.equal(abs(n), iinfo.max), numpy.isfinite(a)).any():
        warnings.warn('some values are clipped to infinity', RuntimeWarning)
    n[numpy.isnan(a)] = iinfo.min
    return n[()]


def binom(n, k):
    a = b = 1
    for i in range(1, k+1):
        a *= n+1-i
        b *= i
    return a // b


def accumulate(data, index, shape):
    '''accumulate scattered data in dense array.

    Accumulates values from ``data`` in an array of shape ``shape`` at positions
    ``index``, equivalent with:

    >>> def accumulate(data, index, shape):
    ...   array = numpy.zeros(shape, data.dtype)
    ...   for v, *ij in zip(data, *index):
    ...     array[ij] += v
    ...   return array
    '''

    ndim = len(shape)
    assert data.ndim == 1
    assert len(index) == ndim and all(isintarray(ind) and ind.shape == data.shape for ind in index)
    if not ndim:
        return data.sum()
    retval = numpy.zeros(shape, data.dtype)
    numpy.add.at(retval, tuple(index), data)
    return retval


def _sorted_index_mask(sorted_array, values):
    values = numpy.asarray(values)
    assert sorted_array.ndim == 1 and values.ndim == 1
    if len(sorted_array):
        # searchsorted always returns an array with dtype np.int64 regardless of its arguments
        indices = numpy.searchsorted(sorted_array[:-1], values)
        mask = numpy.equal(sorted_array[indices], values)
    else:
        indices = numpy.zeros(values.shape, dtype=int)
        mask = numpy.zeros(values.shape, dtype=bool)
    return indices, mask


def sorted_index(sorted_array, values, *, missing=None):
    indices, found = _sorted_index_mask(sorted_array, values)
    if missing is None:
        if not found.all():
            raise ValueError
    elif isint(missing):
        indices[~found] = missing
    elif missing == 'mask':
        indices = indices[found]
    else:
        raise ValueError
    return types.frozenarray(indices, copy=False)


def sorted_contains(sorted_array, values):
    return types.frozenarray(_sorted_index_mask(sorted_array, values)[1], copy=False)


def asboolean(array, size, ordered=True):
    '''convert index array to boolean.

    A boolean array is returned as-is after confirming that the length is correct.

    >>> asboolean([True, False], size=2)
    array([ True, False], dtype=bool)

    A strictly increasing integer array is converted to the equivalent boolean
    array such that ``asboolean(array, n).nonzero()[0] == array``.

    >>> asboolean([1,3], size=4)
    array([False,  True, False,  True], dtype=bool)

    In case the order of integers is not important this must be explicitly
    specified using the ``ordered`` argument.

    >>> asboolean([3,1,1], size=4, ordered=False)
    array([False,  True, False,  True], dtype=bool)

    Args
    ----
    array : :class:`int` or :class:`bool` array_like or None
        Integer or boolean index data.
    size : :class:`int`
        Target array length.
    ordered : :class:`bool`
        Assert that integers are strictly increasing.
    '''

    if array is None or isinstance(array, (list, tuple)) and len(array) == 0:
        return numpy.zeros(size, dtype=bool)
    array = numpy.asarray(array)
    if array.ndim != 1:
        raise Exception('cannot convert array of dimension {} to boolean'.format(array.ndim))
    if array.dtype.kind == 'b':
        if array.size != size:
            raise Exception('array is already boolean but has the wrong length')
        return array
    if array.dtype.kind != 'i':
        raise Exception('cannot convert array of type {!r} to boolean'.format(array.dtype))
    barray = numpy.zeros(size, dtype=bool)
    if array.size:
        if ordered and not numpy.greater(array[1:], array[:-1]).all():
            raise Exception('indices are not strictly increasing')
        if (array[0] if ordered else array.min()) < 0 or (array[-1] if ordered else array.max()) >= size:
            raise Exception('indices are out of bounds')
        barray[array] = True
    return barray


def invmap(indices, length, missing=-1):
    '''Create inverse index array.

    Create the index array ``inverse`` with the given ``length`` such that
    ``inverse[indices[i]] == i`` and ``inverse[j] == missing`` for all ``j`` not
    in ``indices``. It is an error to pass an ``indices`` array with repeated
    indices, in which case the result is undefined.

    >>> m = invmap([3,1], length=5)
    >>> m[3]
    0
    >>> m[1]
    1

    Args
    ----
    indices : :class:`int` array_like
        Integer or index data.
    length : :class:`int`
        Target array length; must be larger than max(indices).
    missing : :class:`int` (default: -1)
        Value to insert for missing indices.

    Returns
    -------
    :class:`numpy.ndarray`
    '''

    invmap = numpy.full(length, missing)
    invmap[numpy.asarray(indices)] = numpy.arange(len(indices))
    return invmap


def levicivita(n: int, dtype=float):
    'n-dimensional Levi-Civita symbol.'
    if n < 2:
        raise ValueError('The Levi-Civita symbol is undefined for dimensions lower than 2.')
    # Generate all possible permutations of `{0,1,...,n-1}` in array `I`, where
    # the second axis runs over the permutations, and determine the number of
    # permutations (`nperms`). First, `I[k] ∈ {k,...,n-1}` becomes the index of
    # dimension `k` for the partial permutation `I[k:]`.
    I = numpy.mgrid[tuple(slice(k, n) for k in range(n))].reshape(n, -1)
    # The number of permutations is equal to the number of deviations from the
    # unpermuted case.
    nperms = numpy.sum(numpy.not_equal(I, numpy.arange(n)[:, None]), 0)
    # Make all partial permutations `I[k+1:]` unique by replacing `I[j]` with `k`
    # if `I[j]` equals `I[k]`, `j > k`. Example with `n = 4`: if `I[2:] = [3,2]` and
    # `I[1] = 2` then `I[3]` must be replaced with `1` to give `I[1:] = [2,3,1]`.
    for k in reversed(range(n-1)):
        I[k+1:][numpy.equal(I[k+1:], I[k, None])] = k
    # Inflate with `1` if `nperms` is even and `-1` if odd.
    result = numpy.zeros((n,)*n, dtype=dtype)
    result[tuple(I)] = 1 - 2*(nperms % 2)
    return result


def sinc(x, n=0):
    '''
    Evaluates the n-th derivative of the unnormalized sinc function:

        sinc(x) = sin(x) / x
    '''

    x = numpy.asarray(x)
    f = numpy.asarray(numpy.sinc(x / numpy.pi)) # Numpy's implementation is normalized to π
    if n == 0:
        return f
    # Derivatives are evaluated using either a recurrence relation or a Taylor
    # series expansion, depending on proximity to the origin.
    m = abs(x) >= 1
    if m.any(): # points outside unit ball
        # sinc'i(x) = (sin'i(x) - i sinc'i-1(x)) / x
        fm = f[m]
        xm = x[m]
        for i in range(1, n+1): # traverse derivatives
            fm *= -i
            fm += [numpy.sin, numpy.cos][i%2](xm) * [1, 1, -1, -1][i % 4]
            fm /= xm
        f[m] = fm
    if not m.all(): # points inside unit ball
        # sinc'n(x) = Σ_i cos(½π(i+n)) x^i / i! / (i+n+1)
        xm = x[~m]
        xm2 = xm**2
        fm = numpy.zeros(xm.shape, dtype=f.dtype)
        imax = 32 # 1/32! = 4e-36
        for i in reversed(range(n % 2, imax, 4)):
            fm *= xm2 / ((i + 4) * (i + 3))
            fm -= 1 / (i + n + 3)
            fm *= xm2 / ((i + 2) * (i + 1))
            fm += 1 / (i + n + 1)
        if n % 2:
            fm *= xm
        if 1 <= n % 4 <= 2:
            fm = -fm
        f[~m] = fm
    return f


def sanitize_einsum_subscripts(subscripts, *shapes):
    '''Sanitize einsum's subscript labels.

    This helper function checks that a subscripts string is consistent with
    argument shapes to be used by Numpy's einsum function, and expands implicit
    output and/or ellipses. The expanded subscript labels are returned as a
    tuple of strings, the last of which contains the output labels.'''

    einsum_symbols_set = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ') # from Numpy's source code
    if not isinstance(subscripts, str):
        raise ValueError('first einsum argument must be a string of subscript labels')
    in_, explicit, out = subscripts.partition('->')
    if not explicit:
        out = ''.join(sorted(c for c in einsum_symbols_set.intersection(subscripts) if subscripts.count(c) == 1))
    in_ = in_.split(',')
    if len(in_) != len(shapes):
        raise ValueError('number of arguments does not match subscript labels')
    if '...' in subscripts: # expand ellipses
        unused_symbol = iter(sorted(einsum_symbols_set.difference(subscripts)))
        ell = ''
        for i, shape in enumerate(shapes):
            if '...' in in_[i]:
                n = builtins.max(len(shape) - (len(in_[i])-3), 0)
                while len(ell) < n:
                    ell += next(unused_symbol)
                in_[i] = in_[i].replace('...', ell[:n][::-1], 1)
        if not explicit:
            out = ell[::-1] + out
        elif '...' in out:
            out = out.replace('...', ell[::-1], 1)
        elif ell:
            raise ValueError('non-empty ellipses in input require ellipsis in output')
    if not all(einsum_symbols_set.issuperset(s) for s in (*in_, out)):
        raise ValueError('invalid subscripts argument')
    if any(len(s) != len(shape) for s, shape in zip(in_, shapes)):
        raise ValueError('argument dimensions are inconsistent with subscript labels')
    axis_shapes = {}
    if any(axis_shapes.setdefault(c, n) != n for s, shape in zip(in_, shapes) for c, n in zip(s, shape) if n != 1):
        raise ValueError('argument shapes are inconsistent with subscript labels')
    for index in set(out) - set(''.join(in_)):
        raise ValueError(f'einstein sum subscripts string included output subscript {index!r} which never appeared in an input')
    return (*in_, out)


# vim:sw=4:sts=4:et
