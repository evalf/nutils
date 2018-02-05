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

import numpy, numbers, builtins, collections.abc

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
  assert 0 <= n < ndim, 'argument out of bounds: {} not in [0,{})'.format(n, ndim)
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

def eig(A):
  '''If A has repeated eigenvalues, numpy.linalg.eig sometimes fails to produce
  the complete eigenbasis. This function aims to fix that by identifying the
  problem and completing the basis where necessary.'''

  L, V = numpy.linalg.eig(A)

  # check repeated eigenvalues
  for index in numpy.ndindex(A.shape[:-2]):
    unique, inverse = numpy.unique(L[index], return_inverse=True)
    if len(unique) < len(inverse): # have repeated eigenvalues
      repeated, = numpy.where(numpy.bincount(inverse) > 1)
      vectors = V[index].T
      for i in repeated: # indices pointing into unique corresponding to repeated eigenvalues
        where, = numpy.where(inverse == i) # corresponding eigenvectors
        for j, n in enumerate(where):
          W = vectors[where[:j]]
          vectors[n] -= numpy.dot(numpy.dot(W, vectors[n]), W) # gram schmidt orthonormalization
          scale = numpy.linalg.norm(vectors[n])
          if scale < 1e-8: # vectors are near linearly dependent
            u, s, vh = numpy.linalg.svd(A[index] - unique[i] * numpy.eye(len(inverse)))
            nnz = numpy.argsort(abs(s))[:len(where)]
            vectors[where] = vh[nnz].conj()
            break
          vectors[n] /= scale

  return L, V

isarray = lambda a: isinstance(a, (numpy.ndarray, const))
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

def det_exact(A):
  # for some reason, numpy.linalg.det suffers from rounding errors
  A = numpy.asarray(A)
  assert A.ndim == 2 and A.shape[0] == A.shape[1]
  if len(A) == 0:
    det = 1.
  elif len(A) == 1:
    det = A[0,0]
  elif len(A) == 2:
    ((a,b),(c,d)) = A
    det = a*d - b*c
  elif len(A) == 3:
    ((a,b,c),(d,e,f),(g,h,i)) = A
    det = a*e*i + b*f*g + c*d*h - c*e*g - b*d*i - a*f*h
  else:
    raise NotImplementedError('shape=' + str(A.shape))
  return det

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

def serialized(array, nsig, ndec):
  if array.ndim > 0:
    return '[{}]'.format(','.join(serialized(a, nsig, ndec) for a in array))
  if not numpy.isfinite(array): # nan, inf
    return str(array)
  a = builtins.round(float(array) * 10**ndec)
  if a == 0:
    return '0'
  while abs(a) >= 10**nsig:
    a //= 10
    ndec -= 1
  return '{}e{}'.format(a, -ndec)

def encode64(array, nsig, ndec):
  import zlib, binascii
  assert isinstance(array, numpy.ndarray) and array.dtype == float
  binary = zlib.compress('{},{},{}'.format(nsig, ndec, serialized(array, nsig, ndec)).encode(), 9)
  data = binascii.b2a_base64(binary).decode().rstrip()
  assert_allclose64(array, data)
  return data

def decode64(data):
  import zlib, binascii
  serialized = zlib.decompress(binascii.a2b_base64(data))
  nsig, ndec, array = eval(serialized, numpy.__dict__)
  return nsig, ndec, numpy.array(array, dtype=float)

def assert_allclose64(actual, data=None):
  try:
    nsig, ndec, desired = decode64(data)
  except Exception as e:
    status = str(e)
    nsig = 4
    ndec = 15
  else:
    try:
      numpy.testing.assert_allclose(actual, desired, atol=1.5*10**-ndec, rtol=10**(1-nsig))
    except Exception as e:
      status = str(e)
    else:
      return
  status += '\n\nIf this is expected, use the following base64 string to test up to nsig={}, ndec={}:'.format(nsig, ndec)
  data = encode64(actual, nsig=nsig, ndec=ndec)
  while data:
    status += '\n{!r}'.format(data[:80])
    data = data[80:]
  raise Exception(status)

class const(collections.abc.Sequence):
  __slots__ = '__base', '__hash'

  @staticmethod
  def full(shape, fill_value):
    return const(numpy.lib.stride_tricks.as_strided(fill_value, shape, [0]*len(shape)), copy=False)

  def __new__(cls, base, copy=True, dtype=None):
    if isinstance(base, const):
      return base
    self = object.__new__(cls)
    self.__base = numpy.array(base, dtype=dtype) if copy or not isinstance(base, numpy.ndarray) or dtype and dtype != base.dtype else base
    self.__base.flags.writeable = False
    self.__hash = hash((self.__base.shape, self.__base.dtype, tuple(self.__base.flat[::self.__base.size//32+1]) if self.__base.size else ())) # NOTE special case self.__base.size == 0 necessary for numpy<1.12
    return self

  @property
  def __array_struct__(self):
    return self.__base.__array_struct__

  def __reduce__(self):
    return const, (self.__base, False)

  def __eq__(self, other):
    if self is other:
      return True
    if not isinstance(other, const):
      return False
    if self.__base is other.__base:
      return True
    if self.__hash != other.__hash or self.__base.dtype != other.__base.dtype or self.__base.shape != other.__base.shape or numpy.not_equal(self.__base, other.__base).any():
      return False
    # deduplicate
    self.__base = other.__base
    return True

  def __lt__(self, other):
    if not isinstance(other, const):
      return NotImplemented
    return self != other and (self.dtype < other.dtype
      or self.dtype == other.dtype and (self.shape < other.shape
        or self.shape == other.shape and self.__base.tolist() < other.__base.tolist()))

  def __le__(self, other):
    if not isinstance(other, const):
      return NotImplemented
    return self == other or (self.dtype < other.dtype
      or self.dtype == other.dtype and (self.shape < other.shape
        or self.shape == other.shape and self.__base.tolist() < other.__base.tolist()))

  def __gt__(self, other):
    if not isinstance(other, const):
      return NotImplemented
    return self != other and (self.dtype > other.dtype
      or self.dtype == other.dtype and (self.shape > other.shape
        or self.shape == other.shape and self.__base.tolist() > other.__base.tolist()))

  def __ge__(self, other):
    if not isinstance(other, const):
      return NotImplemented
    return self == other or (self.dtype > other.dtype
      or self.dtype == other.dtype and (self.shape > other.shape
        or self.shape == other.shape and self.__base.tolist() > other.__base.tolist()))

  def __getitem__(self, item):
    retval = self.__base.__getitem__(item)
    return const(retval, copy=False) if isinstance(retval, numpy.ndarray) else retval

  dtype = property(lambda self: self.__base.dtype)
  shape = property(lambda self: self.__base.shape)
  size = property(lambda self: self.__base.size)
  ndim = property(lambda self: self.__base.ndim)
  flat = property(lambda self: self.__base.flat)
  T = property(lambda self: const(self.__base.T, copy=False))

  __len__ = lambda self: self.__base.__len__()
  __repr__ = lambda self: 'const'+self.__base.__repr__()[5:]
  __str__ = lambda self: self.__base.__str__()
  __add__ = lambda self, other: self.__base.__add__(other)
  __radd__ = lambda self, other: self.__base.__radd__(other)
  __sub__ = lambda self, other: self.__base.__sub__(other)
  __rsub__ = lambda self, other: self.__base.__rsub__(other)
  __mul__ = lambda self, other: self.__base.__mul__(other)
  __rmul__ = lambda self, other: self.__base.__rmul__(other)
  __truediv__ = lambda self, other: self.__base.__truediv__(other)
  __rtruediv__ = lambda self, other: self.__base.__rtruediv__(other)
  __floordiv__ = lambda self, other: self.__base.__floordiv__(other)
  __rfloordiv__ = lambda self, other: self.__base.__rfloordiv__(other)
  __pow__ = lambda self, other: self.__base.__pow__(other)
  __hash__ = lambda self: self.__hash
  __int__ = lambda self: self.__base.__int__()
  __float__ = lambda self: self.__base.__float__()
  __abs__ = lambda self: self.__base.__abs__()
  __neg__ = lambda self: self.__base.__neg__()

  tolist = lambda self, *args, **kwargs: self.__base.tolist(*args, **kwargs)
  copy = lambda self, *args, **kwargs: self.__base.copy(*args, **kwargs)
  astype = lambda self, *args, **kwargs: self.__base.astype(*args, **kwargs)
  take = lambda self, *args, **kwargs: self.__base.take(*args, **kwargs)
  any = lambda self, *args, **kwargs: self.__base.any(*args, **kwargs)
  all = lambda self, *args, **kwargs: self.__base.all(*args, **kwargs)
  sum = lambda self, *args, **kwargs: self.__base.sum(*args, **kwargs)
  min = lambda self, *args, **kwargs: self.__base.min(*args, **kwargs)
  max = lambda self, *args, **kwargs: self.__base.max(*args, **kwargs)
  prod = lambda self, *args, **kwargs: self.__base.prod(*args, **kwargs)
  dot = lambda self, *args, **kwargs: self.__base.dot(*args, **kwargs)
  swapaxes = lambda self, *args, **kwargs: const(self.__base.swapaxes(*args, **kwargs), copy=False)
  ravel = lambda self, *args, **kwargs: const(self.__base.ravel(*args, **kwargs), copy=False)
  reshape = lambda self, *args, **kwargs: const(self.__base.reshape(*args, **kwargs), copy=False)
  transpose = lambda self, *args, **kwargs: const(self.__base.transpose(*args, **kwargs), copy=False)
  cumsum = lambda self, *args, **kwargs: const(self.__base.cumsum(*args, **kwargs), copy=False)
  nonzero = lambda self, *args, **kwargs: const(self.__base.nonzero(*args, **kwargs), copy=False)

  def insertaxis(self, axis, length):
    base = self.__base
    return const(numpy.lib.stride_tricks.as_strided(base,
      shape=base.shape[:axis]+(length,)+base.shape[axis:],
      strides=base.strides[:axis]+(0,)+base.strides[axis:]))

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
  return const(outer.reshape(left.shape[0] * right.shape[0], *pshape), copy=False)

def poly_stack(coeffs):
  coeffs = tuple(coeffs)
  n = max(icoeffs.shape[0] for icoeffs in coeffs)
  ndim = coeffs[0].ndim
  dest = numpy.zeros((len(coeffs),)+(n,)*ndim, dtype=float)
  for i, j in enumerate(coeffs):
    dest[(i,*map(slice, j.shape))] = j
  return const(dest, copy=False)

def poly_grad(coeffs, ndim):
  I = range(ndim)
  dcoeffs = [coeffs[(...,*(slice(1,None) if i==j else slice(0,-1) for j in I))] for i in I]
  if coeffs.shape[-1] > 2:
    a = numpy.arange(1, coeffs.shape[-1])
    dcoeffs = [a[tuple(slice(None) if i==j else numpy.newaxis for j in I)] * c for i, c in enumerate(dcoeffs)]
  dcoeffs = numpy.stack(dcoeffs, axis=coeffs.ndim-ndim)
  return const(dcoeffs, copy=False)

def poly_eval(coeffs, points):
  assert points.ndim == 2
  if coeffs.shape[-1] == 0:
    return const.full((points.shape[0],)+coeffs.shape[1:coeffs.ndim-points.shape[-1]], 0.)
  for dim in reversed(range(points.shape[-1])):
    result = numpy.empty((points.shape[0], *coeffs.shape[1:-1]), dtype=float)
    result[:] = coeffs[...,-1]
    points_dim = points[(slice(None),dim,*(numpy.newaxis,)*(result.ndim-1))]
    for j in reversed(range(coeffs.shape[-1]-1)):
      result *= points_dim
      result += coeffs[...,j]
    coeffs = result
  return const(coeffs, copy=False)

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

# vim:shiftwidth=2:softtabstop=2:expandtab:foldmethod=indent:foldnestmax=2
