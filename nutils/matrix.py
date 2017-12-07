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
The matrix module defines a number of 2D matrix objects, notably the
:func:`ScipyMatrix` and :func:`NumpyMatrix`. Matrix objects support basic
addition and subtraction operations and provide a consistent insterface for
solving linear systems. Matrices can be converted to numpy arrays via
``toarray`` or scipy matrices via ``toscipy``.
"""

from . import util, numpy, log, numeric
import functools


class MyCallback:

  def __init__ (self, matrix, rhs, tol, callback):
    self.matrix = matrix
    self.rhs = rhs
    self.norm = max(numpy.linalg.norm(rhs), 1) # scipy terminates on minimum of relative and absolute residual
    self.niter = 0
    self.tol = tol
    self.callback = callback

  def __call__(self, arg):
    self.niter += 1
    # some solvers provide the residual, others the left hand side vector
    res = float(numpy.linalg.norm(self.rhs - self.matrix * arg) / self.norm if numpy.ndim(arg) == 1 else arg)
    if self.callback:
      self.callback(res)
    with log.context('residual {:.2e} ({:.0f}%)'.format(res, 100. * numpy.log10(res) / numpy.log10(self.tol) if res > 0 else 0)):
      pass

class Matrix( object ):
  'matrix base class'

  def __init__( self, shape ):
    'constructor'

    assert len(shape) == 2
    self.shape = shape

  @property
  def size( self ):
    return numpy.prod( self.shape )

  def cond( self, constrain=None, lconstrain=None, rconstrain=None ):
    'condition number'

    x, I, J = parsecons( constrain, lconstrain, rconstrain, self.shape )
    matrix = self.toarray()[numpy.ix_(I,J)]
    return numpy.linalg.cond( matrix )

  def res( self, x, b=0, constrain=None, lconstrain=None, rconstrain=None, scaled=True ):
    'residual'

    x0, I, J = parsecons( constrain, lconstrain, rconstrain, self.shape )
    res = numpy.linalg.norm( (self.matvec(x)-b)[I] )
    if scaled:
      res /= numpy.linalg.norm( (self.matvec(x0)-b)[I] )
    return res

  def clone( self ):
    warnings.warn( 'warning: arrays are immutable; clone returns self for backwards compatibility', DeprecationWarning )
    return self

class ScipyMatrix( Matrix ):
  '''matrix based on any of scipy's sparse matrices'''

  def __init__( self, core ):
    self.core = core
    Matrix.__init__( self, core.shape )

  matvec = lambda self, vec: self.core.dot( vec )
  toarray = lambda self: self.core.toarray()
  toscipy = lambda self: self.core
  __add__ = lambda self, other: ScipyMatrix( self.core + (other.toscipy() if isinstance(other,Matrix) else other) )
  __sub__ = lambda self, other: ScipyMatrix( self.core - (other.toscipy() if isinstance(other,Matrix) else other) )
  __mul__ = lambda self, other: ScipyMatrix( self.core * (other.toscipy() if isinstance(other,Matrix) else other) )
  __radd__ = __add__
  __rmul__ = __mul__
  __div__ = lambda self, other: ScipyMatrix( self.core / other )
  T = property( lambda self: ScipyMatrix( self.core.transpose() ) )

  def rowsupp( self, tol=0 ):
    'return row indices with nonzero/non-small entries'

    supp = numpy.empty( self.shape[0], dtype=bool )
    for irow in range( self.shape[0] ):
      a, b = self.core.indptr[irow:irow+2]
      supp[irow] = a != b and numpy.any( numpy.abs( self.core.data[a:b] ) > tol )
    return supp

  @log.title
  def solve(self, rhs=None, constrain=None, lconstrain=None, rconstrain=None, tol=0, lhs0=None, solver=None, symmetric=False, callback=None, precon=None, **solverargs):
    'solve'

    import scipy.sparse.linalg

    lhs, I, J = parsecons( constrain, lconstrain, rconstrain, self.shape )
    A = self.core
    if not I.all():
      A = A[I,:]
    b = ( rhs[I] if rhs is not None else 0 ) - A * lhs
    if not J.all():
      A = A[:,J]

    if not b.any():
      log.info('right hand side is zero')
      return lhs

    if solver is None:
      solver = 'spsolve' if tol == 0 else 'cg' if symmetric else 'gmres'

    x0 = lhs0[J] if lhs0 is not None else numpy.zeros(A.shape[0])
    res0 = numpy.linalg.norm(b - A * x0) / numpy.linalg.norm(b)
    if res0 < tol:
      log.info('initial residual is below tolerance')
      x = x0
    elif solver == 'spsolve':
      log.info('solving system using sparse direct solver')
      x = scipy.sparse.linalg.spsolve(A, b)
    else:
      assert tol, 'tolerance must be specified for iterative solver'
      log.info('solving system using {} iterative solver'.format(solver))
      solverfun = getattr(scipy.sparse.linalg, solver)
      # keep scipy from making things circular by shielding the nature of A
      A = scipy.sparse.linalg.LinearOperator( A.shape, A.__mul__, dtype=float )
      if isinstance(precon, str):
        M = self.getprecon(precon, constrain, lconstrain, rconstrain)
      elif callable(precon):
        M = precon(A)
      elif precon is None: # create identity operator, because scipy's native identity operator has circular references
        M = scipy.sparse.linalg.LinearOperator(A.shape, matvec=lambda x:x, rmatvec=lambda x:x, matmat=lambda x:x, dtype=float)
      else:
        M = precon
      assert isinstance(M, scipy.sparse.linalg.LinearOperator)
      mycallback = MyCallback(matrix=A, rhs=b, tol=tol, callback=callback)
      x, status = solverfun(A, b, M=M, tol=tol, x0=x0, callback=mycallback, **solverargs)
      assert status == 0, '{} solver failed with status {}'.format(solver, status)
      res = numpy.linalg.norm(b - A * x) / mycallback.norm
      log.info('solver converged in {} iterations to residual {:.1e}'.format(mycallback.niter, res))

    lhs[J] = x
    return lhs

  def getprecon( self, name='SPLU', constrain=None, lconstrain=None, rconstrain=None ):

    import scipy.sparse.linalg

    name = name.lower()
    x, I, J = parsecons( constrain, lconstrain, rconstrain, self.shape )
    A = self.core[I,:][:,J]
    assert A.shape[0] == A.shape[1], 'constrained matrix must be square'
    log.info( 'building %s preconditioner' % name )
    if name == 'splu':
      precon = scipy.sparse.linalg.splu( A.tocsc() ).solve
    elif name == 'spilu':
      precon = scipy.sparse.linalg.spilu( A.tocsc(), drop_tol=1e-5, fill_factor=None, drop_rule=None, permc_spec=None, diag_pivot_thresh=None, relax=None, panel_size=None, options=None ).solve
    elif name == 'diag':
      precon = numpy.reciprocal( A.diagonal() ).__mul__
    else:
      raise Exception( 'invalid preconditioner %r' % name )
    return scipy.sparse.linalg.LinearOperator( A.shape, precon, dtype=float )

class NumpyMatrix( Matrix ):
  '''matrix based on numpy array'''

  def __init__( self, core ):
    assert numeric.isarray(core)
    self.core = core
    Matrix.__init__( self, core.shape )

  matvec = lambda self, vec: numpy.dot( self.core, vec )
  toarray = lambda self: self.core
  __add__ = lambda self, other: NumpyMatrix( self.core + other.toarray() )
  __sub__ = lambda self, other: NumpyMatrix( self.core - other.toarray() )
  __mul__ = lambda self, other: NumpyMatrix( self.core * other )
  __rmul__ = __mul__
  __div__ = lambda self, other: NumpyMatrix( self.core / other )
  T = property( lambda self: NumpyMatrix( self.core.T ) )

  @log.title
  def solve(self, b=None, constrain=None, lconstrain=None, rconstrain=None, tol=0):
    'solve'

    if b is None:
      b = numpy.zeros( self.shape[0] )
    else:
      b = numpy.asarray( b, dtype=float )
      assert b.ndim == 1, 'right-hand-side has shape %s, expected a vector' % (b.shape,)
      assert b.shape == self.shape[:1]

    x, I, J = parsecons( constrain, lconstrain, rconstrain, self.shape )
    if I.all() and J.all():
      return numpy.linalg.solve( self.core, b )

    data = self.core[I]
    x[J] = numpy.linalg.solve( data[:,J], b[I] - numpy.dot( data[:,~J], x[~J] ) )
    return x


# UTILITY FUNCTIONS

def assemble( data, index, shape, force_dense=False ):
  'create data from values and indices'

  if len(shape) == 0:
    retval = data.sum()
  elif len(shape) == 2 and not force_dense:
    import scipy.sparse.linalg
    csr = scipy.sparse.csr_matrix( (data,index), shape )
    retval = ScipyMatrix( csr )
  else:
    flatindex = numpy.dot( numpy.cumprod( (1,)+shape[:0:-1] )[::-1], index )
    retval = numpy.bincount( flatindex, data, numpy.prod(shape) ).reshape( shape ).astype( data.dtype, copy=False )
    if retval.ndim == 2:
      retval = NumpyMatrix( retval )
  assert retval.shape == shape
  log.debug( 'assembled', '%s(%s)' % ( retval.__class__.__name__, ','.join( str(n) for n in shape ) ) )
  return retval

def parsecons( constrain, lconstrain, rconstrain, shape ):
  'parse constraints'

  I = numpy.ones( shape[0], dtype=bool )
  x = numpy.empty( shape[1] )
  x[:] = numpy.nan
  if constrain is not None:
    assert lconstrain is None
    assert rconstrain is None
    assert numeric.isarray(constrain)
    I[:] = numpy.isnan( constrain )
    x[:] = constrain
  if lconstrain is not None:
    assert numeric.isarray(lconstrain)
    x[:] = lconstrain
  if rconstrain is not None:
    assert numeric.isarray(rconstrain)
    I[:] = rconstrain
  J = numpy.isnan(x)
  assert numpy.sum(I) == numpy.sum(J), 'constrained matrix is not square: %dx%d' % ( numpy.sum(I), numpy.sum(J) )
  x[J] = 0
  return x, I, J


# vim:shiftwidth=2:softtabstop=2:expandtab:foldmethod=indent:foldnestmax=2
