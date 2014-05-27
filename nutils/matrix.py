from . import util, numpy, log, numeric, parallel, _
import scipy.sparse
# from scipy.sparse.sparsetools.csr import _csr
from scipy.sparse import sparsetools as _csr
from scipy.sparse.linalg.isolve import _iterative
import time

def krylov( matvec, b, x0=None, tol=1e-5, restart=None, maxiter=0, precon=None, callback=None ):
  '''solve linear system iteratively

  restart=None: CG
  restart=integer: GMRES'''

  assert isinstance( b, numpy.ndarray ) and b.dtype == numpy.float64 and b.ndim == 1
  n = b.size
  if x0 is None:
    x0 = numpy.zeros( n, dtype=numpy.float64 )
  else:
    assert isinstance( x0, numpy.ndarray ) and x0.dtype == numpy.float64 and x0.ndim == 1 and x0.size == n
  assert isinstance( tol, float ) and tol > 0
  res = tol
  ndx1 = 1
  ndx2 = -1
  ijob = 1
  info = 0
  firsttime = True
  bnrm2 = -1.0
  assert isinstance( maxiter, int ) and maxiter >= 0
  iiter = maxiter

  if restart is None:
    log.context( 'CG' )
    work = numpy.zeros( 4*n, dtype=numpy.float64 )
    ijob_matvecx = 3
    revcom = lambda x, iiter, res, info, ndx1, ndx2, ijob: \
      _iterative.dcgrevcom( b, x, work, iiter, res, info, ndx1, ndx2, ijob )
  else:
    if restart > n:
      restart = n
    log.context( 'GMRES%d' % restart )
    work = numpy.zeros( (6+restart)*n, dtype=numpy.float64 )
    work2 = numpy.zeros( (restart+1)*(2*restart+2), dtype=numpy.float64 )
    ijob_matvecx = 1
    revcom = lambda x, iiter, res, info, ndx1, ndx2, ijob: \
      _iterative.dgmresrevcom( b, x, restart, work, work2, iiter, res, info, ndx1, ndx2, ijob )
  stoptest = lambda vec1, bnrm2, info: \
    _iterative.dstoptest2( vec1, b, bnrm2, tol, info )

  x = x0
  progress = log.progress( 'residual', target=numpy.log(tol) )
  t0 = time.clock()
  while True:
    x, iiter, res, info, ndx1, ndx2, sclr1, sclr2, ijob = \
      revcom( x, iiter, res, info, ndx1, ndx2, ijob )
    vec1 = work[ndx1-1:ndx1-1+n]
    vec2 = work[ndx2-1:ndx2-1+n]
    if ijob == 1 or ijob == 3:
      vec2 *= sclr2
      vec2 += sclr1 * matvec( x if ijob == ijob_matvecx else vec1 )
    elif ijob == 2:
      vec1[:] = precon(vec2) if precon else vec2
    elif ijob == 4:
      if firsttime:
        info = -1
        firsttime = False
      bnrm2, res, info = stoptest( vec1, bnrm2, info )
      if callback:
        callback( (iiter,res) )
    else:
      assert ijob == -1
      break
    ijob = 2
    progress.update( numpy.log(res) )
  dt = time.clock() - t0
  progress.disable()

  assert info == 0
  log.info( 'converged in %.1f seconds, %d iterations' % ( dt, iiter ) )
  return x

def parsecons( constrain, lconstrain, rconstrain, shape ):
  'parse constraints'

  I = numpy.ones( shape[0], dtype=bool )
  x = numpy.empty( shape[1] )
  x[:] = numpy.nan
  if constrain is not None:
    assert lconstrain is None
    assert rconstrain is None
    assert isinstance( constrain, numpy.ndarray )
    I[:] = numpy.isnan( constrain )
    x[:] = constrain
  if lconstrain is not None:
    assert isinstance( lconstrain, numpy.ndarray )
    x[:] = lconstrain
  if rconstrain is not None:
    assert isinstance( rconstrain, numpy.ndarray )
    I[:] = rconstrain
  J = numpy.isnan(x)
  assert numpy.sum(I) == numpy.sum(J), 'constrained matrix is not square: %dx%d' % ( numpy.sum(I), numpy.sum(J) )
  x[J] = 0
  return x, I, J

class Matrix( object ):
  'matrix base class'

  def __init__( self, (nrows,ncols) ):
    'constructor'

    self.shape = int(nrows), int(ncols) # need exact type because of _csr funcs
    self.size = nrows * ncols

  def __sub__( self, other ):
    'add'

    if other == 0:
      return self

    A = self.clone()
    A -= other
    return A

  def __add__( self, other ):
    'add'

    if other == 0:
      return self

    A = self.clone()
    A += other
    return A

  def __mul__( self, other ):
    'multiply'

    A = self.clone()
    A *= other
    return A

  __rmul__ = __mul__

  def cond( self, constrain=None, lconstrain=None, rconstrain=None ):
    'condition number'

    x, I, J = parsecons( constrain, lconstrain, rconstrain, self.shape )
    matrix = self.toarray()[numpy.ix_(I,J)]
    return numpy.linalg.cond( matrix )

  def res( self, x, b=0, constrain=None, lconstrain=None, rconstrain=None ):
    'residual'

    x0, I, J = parsecons( constrain, lconstrain, rconstrain, self.shape )
    return numeric.norm2( (self.matvec(x)-b)[I] ) / numeric.norm2( (self.matvec(x0)-b)[I] )

class DenseSubMatrix( Matrix ):
  'dense but non-contiguous data'

  def __init__( self, data, indices ):
    'constructor'

    self.data = data
    self.indices = indices
    if isinstance( indices, numpy.ndarray ):
      nrows, ncols = indices.shape
    else:
      I, J = indices
      nrows = I.stop - I.start if isinstance(I,slice) else I.size
      ncols = J.stop - J.start if isinstance(J,slice) else J.size
    Matrix.__init__( self, (nrows,ncols) )

  def __iadd__( self, other ):
    'in place addition'

    assert self.shape == other.shape
    self.data[ self.indices ] += other
    return self

class SparseMatrix( Matrix ):
  'sparse matrix'

  def __init__( self, graph, ncols=None ):
    'constructor'

    if isinstance( graph, tuple ):
      self.data, self.indices, self.indptr = graph
      assert self.indices.dtype == numpy.intc
      assert self.indptr.dtype == numpy.intc
      assert len(self.indices) == len(self.data) == self.indptr[-1]
      nrows = len(self.indptr) - 1
    else:
      nrows = len(graph)
      nzrow = map(len,graph)
      count = sum( nzrow )
      assert numpy.sum( nzrow ) <= numpy.iinfo( numpy.intc ).max, 'matrix overflow: length %d > max intc %d' % ( numpy.sum( nzrow ), numpy.iinfo( numpy.intc ).max )
      self.data = parallel.shzeros( count, dtype=float )
      self.indptr = numpy.cumsum( [0] + nzrow, dtype=numpy.intc )
      self.indices = numpy.empty( count, dtype=numpy.intc )
      for irow, icols in enumerate( graph ):
        a, b = self.indptr[irow:irow+2]
        self.indices[a:b] = icols
    self.splu_cache = {}
    Matrix.__init__( self, (nrows, ncols or nrows) )

  def reshape( self, (nrows,ncols) ):
    'reshape matrix'

    assert nrows >= self.shape[0] and ncols >= self.shape[1]
    indptr = self.indptr
    if nrows > self.shape[1]:
      indptr = numpy.concatenate([ indptr, numeric.fastrepeat( indptr[-1:], nrows-self.shape[1] ) ])
    return self.__class__( (self.data,self.indices,indptr), ncols )

  def clone( self ):
    'clone matrix'

    return self.__class__( (self.data.copy(),self.indices,self.indptr), self.shape[1] )

  def matvec( self, other ):
    'matrix-vector multiplication'

    assert other.shape == self.shape[1:]
    result = numpy.zeros( self.shape[0] )
    _csr.csr_matvec( self.shape[0], self.shape[1], self.indptr, self.indices, self.data, other, result )
    return result

  def __getitem__( self, (rows,cols) ):
    'get submatrix'

    if isinstance(cols,numpy.ndarray) and cols.dtype == bool:
      assert len(cols) == self.shape[1]
      cols, = cols.nonzero()
    elif isinstance(cols,numpy.ndarray) and cols.dtype == int:
      pass
    else:
      raise Exception, 'invalid column argument'
    ncols = len(cols)

    if isinstance(rows,numpy.ndarray) and rows.dtype == bool:
      assert len(rows) == self.shape[0]
      rows, = rows.nonzero()
    elif isinstance(rows,numpy.ndarray) and rows.dtype == int:
      pass
    else:
      raise Exception, 'invalid row argument'
    nrows = len(rows)

    indptr = numpy.empty( nrows+1, dtype=int )
    I = numpy.empty( numpy.minimum( self.indptr[rows+1] - self.indptr[rows], ncols ).sum(), dtype=int ) # allocate for worst case
    indptr[0] = 0
    for n, irow in enumerate( rows ):
      a, b = self.indptr[irow:irow+2]
      where = a + numpy.searchsorted( self.indices[a:b], cols )
      assert ( self.indices[where] == cols ).all()
      c = indptr[n]
      d = c + where.size
      I[c:d] = where
      indptr[n+1] = d

    return DenseSubMatrix( self.data, I.reshape(nrows,ncols) ) if d == nrows * ncols \
      else SparseMatrix( (self.data[I],self.indices[I],indptr), ncols )

  def __setitem__( self, item, value ):
    'set submatrix'

    assert self.data is value.data # apparently we are assigning ourselves

  def _binary( self, other, op ):
    'binary operation'

    assert isinstance( other, SparseMatrix )
    assert self.shape == other.shape
    maxcount = len(self.data) + len(other.data)
    indptr = numpy.empty( self.shape[0]+1, dtype=numpy.intc )
    indices = numpy.empty( maxcount, dtype=numpy.intc )
    data = numpy.empty( maxcount, dtype=float )
    op( self.shape[0], self.shape[1], self.indptr, self.indices, self.data, other.indptr, other.indices, other.data, indptr, indices, data )
    nz = indptr[-1]
    return SparseMatrix( (data[:nz],indices[:nz],indptr), ncols=self.shape[1] )

  def __add__( self, other ):
    'add'

    if other == 0:
      return self

    return self._binary( other, _csr.csr_plus_csr )

  def __sub__( self, other ):
    'subtract'

    if other == 0:
      return self

    return self._binary( other, _csr.csr_minus_csr )

  def _indices_into( self, other ):
    'locate indices of other into self'

    assert isinstance( other, self.__class__ ) and self.shape == other.shape
    if numpy.all( self.indptr == other.indptr ) and numpy.all( self.indices == other.indices ):
      return slice(None)
    I = numpy.empty( other.data.shape, dtype=int )
    for irow in range( self.shape[0] ):
      s = slice( other.indptr[irow], other.indptr[irow+1] )
      I[s] = self.indptr[irow] \
           + numpy.searchsorted( self.indices[self.indptr[irow]:self.indptr[irow+1]], other.indices[s] )
    assert all( self.indices[I] == other.indices )
    return I

  def __iadd__( self, other ):
    'in place addition'

    if other:
      self.data[self._indices_into(other)] += other.data
    return self

  def __isub__( self, other ):
    'in place addition'

    if other:
      self.data[self._indices_into(other)] -= other.data
    return self

  def __imul__( self, other ):
    'scalar multiplication'

    assert isinstance( other, (int,float) )
    self.data *= other
    return self

  def __idiv__( self, other ):
    'scalar multiplication'

    assert isinstance( other, (int,float) )
    self.data /= other
    return self

  @property
  def T( self ):
    'transpose'

    data = numpy.empty_like( self.data )
    indices = numpy.empty_like( self.indices )
    indptr = numpy.empty_like( self.indptr )
    _csr.csr_tocsc( self.shape[0], self.shape[1], self.indptr, self.indices, self.data, indptr, indices, data )
    return SparseMatrix( (data,indices,indptr), self.shape[0] )

  def toarray( self ):
    'convert to numpy array'

    array = numpy.zeros( self.shape )
    for irow in range( self.shape[0] ):
      a, b = self.indptr[irow:irow+2]
      array[irow,self.indices[a:b]] = self.data[a:b]
    return array

  def todense( self ):
    'convert to dense matrix'

    return DenseMatrix( self.toarray() )

  def rowsupp( self, tol=0 ):
    'return row indices with nonzero/non-small entries'

    supp = numpy.empty( self.shape[0], dtype=bool )
    for irow in range( self.shape[0] ):
      a, b = self.indptr[irow:irow+2]
      supp[irow] = a != b and ( tol == 0 or numpy.any( numpy.abs( self.data[a:b] ) > tol ) )
    return supp

  def get_splu( self, I, J, complete ):
    'register LU preconditioner'

    cij = tuple(numpy.where(~I)[0]), tuple(numpy.where(~J)[0]), complete
    precon = self.splu_cache.get( cij )
    if precon is None:
      log.info( 'building %s preconditioner' % ( 'SPLU' if complete else 'SPILU' ) )
      A = scipy.sparse.csr_matrix( (self.data,self.indices,self.indptr), shape=self.shape )[numpy.where(I)[0],:][:,numpy.where(J)[0]].tocsc()
      precon = scipy.sparse.linalg.splu( A ) if complete \
          else scipy.sparse.linalg.spilu( A, drop_tol=1e-5, fill_factor=None, drop_rule=None, permc_spec=None, diag_pivot_thresh=None, relax=None, panel_size=None, options=None )
      self.splu_cache[ cij ] = precon
    return precon

  def factor( self, constrain=None, lconstrain=None, rconstrain=None, complete=False ):
    'prepare preconditioner'

    x, I, J = parsecons( constrain, lconstrain, rconstrain, self.shape )
    return self.get_splu( I, J, complete )

  def solve( self, b=0, constrain=None, lconstrain=None, rconstrain=None, tol=0, x0=None, symmetric=False, maxiter=0, restart=999, title='solving system', callback=None, precon=None ):
    'solve'

    if tol == 0:
      return self.todense().solve( b=b, constrain=constrain, lconstrain=lconstrain, rconstrain=rconstrain, title=title, log=log )

    log.context( title )

    b = numpy.asarray( b, dtype=float )
    if b.ndim == 0:
      b = numeric.fastrepeat( b[_], self.shape[0] )
    else:
      assert b.ndim == 1, 'right-hand-side has shape %s, expected a vector' % (b.shape,)
      assert b.shape == self.shape[:1]

    if symmetric:
      restart = None

    x, I, J = parsecons( constrain, lconstrain, rconstrain, self.shape )
    if x0 is not None:
      x0 = x0[J]

    b = ( b - self.matvec(x) )[I]

    if precon == 'splu':

      precon = self.get_splu( I, J, complete=True )
      x[J] = precon.solve( b )

    else:

      tmpvec = numpy.zeros( self.shape[1] )
      def matvec( v ):
        tmpvec[J] = v
        return self.matvec(tmpvec)[I]

      if precon == 'spilu':
        precon = self.get_splu( I, J, complete=False ).solve
      elif precon:
        raise Exception( 'Unknown preconditioner %s' % precon )

      x[J] = krylov( matvec, b, x0=x0, tol=tol, maxiter=maxiter, restart=restart, callback=callback, precon=precon )

    return x

class DenseMatrix( Matrix ):
  'matrix wrapper class'

  def __init__( self, shape ):
    'constructor'

    if isinstance( shape, numpy.ndarray ):
      self.data = shape
    else:
      if isinstance( shape, int ):
        shape = shape, shape
      self.data = parallel.shzeros( shape )
    Matrix.__init__( self, self.data.shape )

  def __getitem__( self, (rows,cols) ):
    'get submatrix'

    if isinstance(rows,numpy.ndarray) and isinstance(cols,numpy.ndarray):
      rows = rows[:,_]
      cols = cols[_,:]
    return DenseSubMatrix( self.data, (rows,cols) )

  def __setitem__( self, item, value ):
    'set submatrix'

    assert self.data is value.data # apparently we are assigning ourselves

  def __iadd__( self, other ):
    'in place addition'

    assert self.shape == other.shape
    self.data += other.toarray()
    return self

  def __isub__( self, other ):
    'in place addition'

    assert isinstance( other, DenseMatrix )
    assert self.shape == other.shape
    self.data -= other.data
    return self

  def clone( self ):
    'clone matrix'

    return DenseMatrix( self.data.copy() )

  def addblock( self, rows, cols, vals ):
    'add matrix data'

    self.data[ rows[:,_], cols[:,_] ] += vals

  def toarray( self ):
    'convert to numpy array'

    return self.data

  def matvec( self, other ):
    'matrix-vector multiplication'

    assert other.shape == self.shape[1:]
    return numpy.dot( self.data, other )

  @property
  def T( self ):
    'transpose'

    return DenseMatrix( self.data.T )

  def solve( self, b=0, constrain=None, lconstrain=None, rconstrain=None, title='solving system', **dummy ):
    'solve'

    log.context( title + ' [direct]' )

    b = numpy.asarray( b, dtype=float )
    if b.ndim == 0:
      b = numeric.fastrepeat( b[_], self.shape[0] )
    else:
      assert b.ndim == 1, 'right-hand-side has shape %s, expected a vector' % (b.shape,)
      assert b.shape == self.shape[:1]

    if constrain is lconstrain is rconstrain is None:
      return numpy.linalg.solve( self.data, b )

    x, I, J = parsecons( constrain, lconstrain, rconstrain, self.shape )
    data = self.data[I]

    x[J] = numpy.linalg.solve( data[:,J], b[I] - numpy.dot( data[:,~J], x[~J] ) )
    log.info( 'done' )

    return x

# vim:shiftwidth=2:foldmethod=indent:foldnestmax=2
