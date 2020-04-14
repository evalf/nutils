# Copyright (c) 2014 Evalf
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""
The numeric module provides methods that are lacking from the numpy module.
"""

from . import types, warnings
import numpy, numbers, builtins, collections.abc, functools

_abc = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ' # indices for einsum

def round(arr):
  return numpy.round(arr).astype(int)

def sign(arr):
  return numpy.sign(arr).astype(int)

def floor(arr):
  return numpy.floor(arr).astype(int)

def ceil(arr):
  return numpy.ceil(arr).astype(int)

def overlapping(arr, axis=-1, n=2):
  'reinterpret data with overlaps'

  arr = numpy.asarray(arr)
  if axis < 0:
    axis += arr.ndim
  assert 0 <= axis < arr.ndim
  shape = arr.shape[:axis] + (arr.shape[axis]-n+1,n) + arr.shape[axis+1:]
  strides = arr.strides[:axis] + (arr.strides[axis],arr.strides[axis]) + arr.strides[axis+1:]
  overlapping = numpy.lib.stride_tricks.as_strided(arr, shape, strides)
  overlapping.flags.writeable = False
  return overlapping

def normdim(ndim, n):
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

  axes = sorted([normdim(maxdim,axis)] if isinstance(axis,int) else [normdim(maxdim,ax) for ax in axis])
  o = _abc[:maxdim-len(axes)] if axes == range(maxdim-len(axes), maxdim) \
    else ''.join(_abc[a+1:b] for a, b in zip([-1]+axes, axes+[maxdim]) if a+1 != b)

  return numpy.einsum('{},{}->{}'.format(m,n,o), A, B, optimize=False)

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

  return numpy.einsum('{},{}->{}'.format(m,n,o), A, B, optimize=False)

def meshgrid(*args):
  'multi-dimensional meshgrid generalisation'

  args = [numpy.asarray(arg) for arg in args]
  shape = [len(args)] + [arg.size for arg in args if arg.ndim]
  dtype = int if all(isintarray(a) for a in args) else float
  grid = numpy.empty(shape, dtype=dtype)
  n = len(shape)-1
  for i, arg in enumerate(args):
    if arg.ndim:
      n -= 1
      grid[i] = arg[(slice(None),)+(numpy.newaxis,)*n]
    else:
      grid[i] = arg
  assert n == 0
  return grid

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
  singular systems :func:`inv` does not raise a ``LinAlgError``, but rather
  issues a ``RuntimeWarning`` and returns NaN (not a number) values. For
  arguments of dimension >2 the return array contains NaN values only for those
  entries that correspond to singular matrices.
  '''

  try:
    Ainv = numpy.linalg.inv(A)
  except numpy.linalg.LinAlgError:
    warnings.warn('singular matrix', RuntimeWarning)
    Ainv = numpy.empty(A.shape, dtype=float)
    for index in numpy.ndindex(A.shape[:-2]):
      try:
        Ainv[index] = numpy.linalg.inv(A[index])
      except numpy.linalg.LinAlgError:
        Ainv[index] = numpy.nan
  return Ainv

isarray = lambda a: isinstance(a, (numpy.ndarray, types.frozenarray))
isboolarray = lambda a: isarray(a) and a.dtype == bool
isbool = lambda a: isboolarray(a) and a.ndim == 0 or type(a) == bool
isint = lambda a: isinstance(a, (numbers.Integral,numpy.integer))
isnumber = lambda a: isinstance(a, (numbers.Number,numpy.generic))
isintarray = lambda a: isarray(a) and numpy.issubdtype(a.dtype, numpy.integer)
asobjvector = lambda v: numpy.array((None,)+tuple(v), dtype=object)[1:] # 'None' prevents interpretation of objects as axes

def blockdiag(args):
  args = [numpy.asarray(arg) for arg in args]
  args = [arg[numpy.newaxis,numpy.newaxis] if arg.ndim == 0 else arg for arg in args]
  assert all(arg.ndim == 2 for arg in args)
  shapes = numpy.array([arg.shape for arg in args])
  blockdiag = numpy.zeros(shapes.sum(0))
  for arg, (i,j) in zip(args, shapes.cumsum(0)):
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
    concat[(slice(None),)*axis+(slice(i,j),)] = arg
    i = j + 1
  return concat

def ix(args):
  'version of :func:`numpy.ix_` that allows for scalars'
  args = tuple(numpy.asarray(arg) for arg in args)
  assert all(0 <= arg.ndim <= 1 for arg in args)
  idims = numpy.cumsum([0] + [arg.ndim for arg in args])
  ndims = idims[-1]
  return [arg.reshape((1,)*idim+(arg.size,)+(1,)*(ndims-idim-1)) for idim, arg in zip(idims, args)]

def kronecker(arr, axis, length, pos):
  axis = normdim(arr.ndim+1, axis)
  kron = numpy.zeros(arr.shape[:axis]+(length,)+arr.shape[axis:], arr.dtype)
  kron[(slice(None),)*axis + (pos,)] = arr
  return kron

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
    ((a,),(b,)) = A
    ext = numpy.array((b,-a))
  elif len(A) == 3:
    ((a,b),(c,d),(e,f)) = A
    ext = numpy.array((c*f-e*d,e*b-a*f,a*d-c*b))
  else:
    raise NotImplementedError('shape={}'.format(A.shape))
  return ext

def power(a, b):
  a = numpy.asarray(a)
  b = numpy.asarray(b)
  if a.dtype == int and b.dtype == int:
    b = b.astype(float)
  return numpy.power(a, b)

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
  n = numpy.asarray((numpy.arcsinh(a.clip(-amax,amax)*(rtol/atol))/rtol).round().astype(iinfo.dtype))
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

def poly_outer_product(left, right):
  left, right = numpy.asarray(left), numpy.asarray(right)
  nleft, nright = left.ndim-1, right.ndim-1
  pshape = left.shape[1:] if not nright else right.shape[1:] if not nleft else (max(left.shape[1:])+max(right.shape[1:])-1,) * (nleft + nright)
  outer = numpy.zeros((left.shape[0], right.shape[0], *pshape), dtype=numpy.common_type(left, right))
  a = slice(None)
  outer[(a,a,*(map(slice, left.shape[1:]+right.shape[1:])))] = left[(a,None)+(a,)*nleft+(None,)*nright]*right[(None,a)+(None,)*nleft+(a,)*nright]
  return types.frozenarray(outer.reshape(left.shape[0] * right.shape[0], *pshape), copy=False)

def poly_concatenate(coeffs):
  n = max(c.shape[1] for c in coeffs)
  coeffs = [numpy.pad(c, [(0,0)]+[(0,n-c.shape[1])]*(c.ndim-1), 'constant', constant_values=0) if c.shape[1] < n else c for c in coeffs]
  return numpy.concatenate(coeffs)

@types.apply_annotations
@functools.lru_cache()
def poly_grad(coeffs:types.frozenarray, ndim:int):
  I = range(ndim)
  dcoeffs = [coeffs[(...,*(slice(1,None) if i==j else slice(0,-1) for j in I))] for i in I]
  if coeffs.shape[-1] > 2:
    a = numpy.arange(1, coeffs.shape[-1])
    dcoeffs = [a[tuple(slice(None) if i==j else numpy.newaxis for j in I)] * c for i, c in enumerate(dcoeffs)]
  dcoeffs = numpy.stack(dcoeffs, axis=coeffs.ndim-ndim)
  return types.frozenarray(dcoeffs, copy=False)

@types.apply_annotations
@functools.lru_cache()
def poly_eval(coeffs:types.frozenarray, points:types.frozenarray):
  assert points.ndim == 2
  if coeffs.shape[-1] == 0:
    return types.frozenarray.full((points.shape[0],)+coeffs.shape[1:coeffs.ndim-points.shape[-1]], 0.)
  for dim in reversed(range(points.shape[-1])):
    result = numpy.empty((points.shape[0], *coeffs.shape[1:-1]), dtype=float)
    result[:] = coeffs[...,-1]
    points_dim = points[(slice(None),dim,*(numpy.newaxis,)*(result.ndim-1))]
    for j in reversed(range(coeffs.shape[-1]-1)):
      result *= points_dim
      result += coeffs[...,j]
    coeffs = result
  return types.frozenarray(coeffs, copy=False)

def poly_mul(p, q):
  assert p.ndim == q.ndim
  pq = numpy.zeros([n+m-1 for n, m in zip(p.shape, q.shape)])
  if q.size < p.size:
    p, q = q, p # loop over the smallest of the two arrays
  for i, pi in numpy.ndenumerate(p):
    if pi:
      pq[tuple(slice(o, o+m) for o, m in zip(i, q.shape))] += pi * q
  return pq

def poly_pow(p, n):
  assert isint(n) and n >= 0
  if n == 0:
    return numpy.ones((1,)*p.ndim)
  if n == 1:
    return p
  q = poly_pow(poly_mul(p, p), n//2)
  if n%2:
    return poly_mul(q, p)
  return q

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

# vim:sw=2:sts=2:et
