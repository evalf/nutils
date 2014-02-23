from . import util, log, numeric, parallel, _
import scipy.sparse
from scipy.sparse.sparsetools.csr import _csr
from scipy.sparse.linalg.isolve import _iterative
import time

@log.title
def krylov( matvec, b, x0=None, tol=1e-5, restart=None, maxiter=0, precon=None, callback=None ):
  '''solve linear system iteratively

  restart=None: CG
  restart=integer: GMRES'''

  assert numeric.isarray( b ) and b.dtype == numeric.float64 and b.ndim == 1
  n = b.size
  if x0 is None:
    x0 = numeric.zeros( n, dtype=numeric.float64 )
  else:
    assert numeric.isarray( x0 ) and x0.dtype == numeric.float64 and x0.ndim == 1 and x0.size == n
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
    log.debug( 'method: CG' )
    work = numeric.zeros( 4*n, dtype=numeric.float64 )
    ijob_matvecx = 3
    revcom = lambda x, iiter, res, info, ndx1, ndx2, ijob: \
      _iterative.dcgrevcom( b, x, work, iiter, res, info, ndx1, ndx2, ijob )
  else:
    if restart > n:
      restart = n
    log.debug( 'method: GMRES%d' % restart )
    work = numeric.zeros( (6+restart)*n, dtype=numeric.float64 )
    work2 = numeric.zeros( (restart+1)*(2*restart+2), dtype=numeric.float64 )
    ijob_matvecx = 1
    revcom = lambda x, iiter, res, info, ndx1, ndx2, ijob: \
      _iterative.dgmresrevcom( b, x, restart, work, work2, iiter, res, info, ndx1, ndx2, ijob )
  stoptest = lambda vec1, bnrm2, info: \
    _iterative.dstoptest2( vec1, b, bnrm2, tol, info )

  x = x0
  clock = util.Clock()
  t0 = time.time()
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
    if clock.check():
      log.progress( 'residual %.2e (%.0f%%)' % ( res, numeric.log(res) * 100. / numeric.log(tol) ) )
  dt = time.time() - t0

  assert info == 0
  log.info( 'converged in %.1f seconds, %d iterations' % ( dt, iiter ) )
  return x

def parsecons( constrain, lconstrain, rconstrain, shape ):
  'parse constraints'

  I = numeric.ones( shape[0], dtype=bool )
  x = numeric.empty( shape[1] )
  x[:] = numeric.nan
  if constrain is not None:
    assert lconstrain is None
    assert rconstrain is None
    assert numeric.isarray( constrain )
    I[:] = numeric.isnan( constrain )
    x[:] = constrain
  if lconstrain is not None:
    assert numeric.isarray( lconstrain )
    x[:] = lconstrain
  if rconstrain is not None:
    assert numeric.isarray( rconstrain )
    I[:] = rconstrain
  J = numeric.isnan(x)
  assert numeric.sum(I) == numeric.sum(J), 'constrained matrix is not square: %dx%d' % ( numeric.sum(I), numeric.sum(J) )
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
    matrix = self.toarray()[I][:,J]
    return numeric.cond( matrix )

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
    if numeric.isarray( indices ):
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
      assert self.indices.dtype == numeric.intc
      assert self.indptr.dtype == numeric.intc
      assert len(self.indices) == len(self.data) == self.indptr[-1]
      nrows = len(self.indptr) - 1
    else:
      nrows = len(graph)
      nzrow = map(len,graph)
      count = sum( nzrow )
      assert numeric.sum( nzrow ) <= numeric.iinfo( numeric.intc ).max, 'matrix overflow: length %d > max intc %d' % ( numeric.sum( nzrow ), numeric.iinfo( numeric.intc ).max )
      self.data = parallel.shzeros( count, dtype=float )
      self.indptr = numeric.cumsum( [0] + nzrow, dtype=numeric.intc )
      self.indices = numeric.empty( count, dtype=numeric.intc )
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
      indptr = numeric.concatenate([ indptr, numeric.fastrepeat( indptr[-1:], nrows-self.shape[1] ) ])
    return self.__class__( (self.data,self.indices,indptr), ncols )

  def clone( self ):
    'clone matrix'

    return self.__class__( (self.data.copy(),self.indices,self.indptr), self.shape[1] )

  def matvec( self, other ):
    'matrix-vector multiplication'

    assert other.shape == self.shape[1:]
    result = numeric.zeros( self.shape[0] )
    _csr.csr_matvec( self.shape[0], self.shape[1], self.indptr, self.indices, self.data, other, result )
    return result

  def __getitem__( self, (rows,cols) ):
    'get submatrix'

    if isinstance(cols,numeric.ndarray) and cols.dtype == bool:
      assert len(cols) == self.shape[1]
      cols, = cols.nonzero()
    elif isinstance(cols,numeric.ndarray) and cols.dtype == int:
      pass
    else:
      raise Exception, 'invalid column argument'
    ncols = len(cols)

    if isinstance(rows,numeric.ndarray) and rows.dtype == bool:
      assert len(rows) == self.shape[0]
      rows, = rows.nonzero()
    elif isinstance(rows,numeric.ndarray) and rows.dtype == int:
      pass
    else:
      raise Exception, 'invalid row argument'
    nrows = len(rows)

    indptr = numeric.empty( nrows+1, dtype=int )
    I = numeric.empty( numeric.minimum( self.indptr[rows+1] - self.indptr[rows], ncols ).sum(), dtype=int ) # allocate for worst case
    indptr[0] = 0
    for n, irow in enumerate( rows ):
      a, b = self.indptr[irow:irow+2]
      where = a + numeric.findsorted( self.indices[a:b], cols )
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
    indptr = numeric.empty( self.shape[0]+1, dtype=numeric.intc )
    indices = numeric.empty( maxcount, dtype=numeric.intc )
    data = numeric.empty( maxcount, dtype=float )
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
    if self.indptr == other.indptr and self.indices == other.indices:
      return slice(None)
    I = numeric.empty( other.data.shape, dtype=int )
    for irow in range( self.shape[0] ):
      s = slice( other.indptr[irow], other.indptr[irow+1] )
      I[s] = self.indptr[irow] \
           + numeric.findsorted( self.indices[self.indptr[irow]:self.indptr[irow+1]], other.indices[s] )
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

    data = numeric.empty_like( self.data )
    indices = numeric.empty_like( self.indices )
    indptr = numeric.empty_like( self.indptr )
    _csr.csr_tocsc( self.shape[0], self.shape[1], self.indptr, self.indices, self.data, indptr, indices, data )
    return SparseMatrix( (data,indices,indptr), self.shape[0] )

  def toarray( self ):
    'convert to array'

    array = numeric.zeros( self.shape )
    for irow in range( self.shape[0] ):
      a, b = self.indptr[irow:irow+2]
      array[irow,self.indices[a:b]] = self.data[a:b]
    return array

  def todense( self ):
    'convert to dense matrix'

    return DenseMatrix( self.toarray() )

  def rowsupp( self, tol=0 ):
    'return row indices with nonzero/non-small entries'

    supp = numeric.empty( self.shape[0], dtype=bool )
    for irow in range( self.shape[0] ):
      a, b = self.indptr[irow:irow+2]
      supp[irow] = a != b and ( tol == 0 or numeric.greater( numeric.abs( self.data[a:b] ), tol ).any() )
    return supp

  def get_splu( self, I, J, complete ):
    'register LU preconditioner'

    cij = tuple(numeric.find(~I)), tuple(numeric.find(~J)), complete
    precon = self.splu_cache.get( cij )
    if precon is None:
      log.info( 'building %s preconditioner' % ( 'SPLU' if complete else 'SPILU' ) )
      A = scipy.sparse.csr_matrix( (self.data,self.indices,self.indptr), shape=self.shape )[numeric.find(I),:][:,numeric.find(J)].tocsc()
      precon = scipy.sparse.linalg.splu( A ) if complete \
          else scipy.sparse.linalg.spilu( A, drop_tol=1e-5, fill_factor=None, drop_rule=None, permc_spec=None, diag_pivot_thresh=None, relax=None, panel_size=None, options=None )
      self.splu_cache[ cij ] = precon
    return precon

  def factor( self, constrain=None, lconstrain=None, rconstrain=None, complete=False ):
    'prepare preconditioner'

    x, I, J = parsecons( constrain, lconstrain, rconstrain, self.shape )
    return self.get_splu( I, J, complete )

  @log.title
  def solve( self, b=0, constrain=None, lconstrain=None, rconstrain=None, tol=0, x0=None, symmetric=False, maxiter=0, restart=999, callback=None, precon=None ):
    'solve'

    if tol == 0:
      return self.todense().solve( b=b, constrain=constrain, lconstrain=lconstrain, rconstrain=rconstrain, log=log )

    b = numeric.asarray( b, dtype=float )
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

      tmpvec = numeric.zeros( self.shape[1] )
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

    if numeric.isarray( shape ):
      self.data = shape
    else:
      if isinstance( shape, int ):
        shape = shape, shape
      self.data = parallel.shzeros( shape )
    Matrix.__init__( self, self.data.shape )

  def __getitem__( self, (rows,cols) ):
    'get submatrix'

    if numeric.isarray( rows ) and numeric.isarray( cols ):
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
    'convert to array'

    return self.data

  def matvec( self, other ):
    'matrix-vector multiplication'

    assert other.shape == self.shape[1:]
    return numeric.dot( self.data, other )

  @property
  def T( self ):
    'transpose'

    return DenseMatrix( self.data.T )

  @log.title
  def solve( self, b=0, constrain=None, lconstrain=None, rconstrain=None, **dummy ):
    'solve'

    b = numeric.asarray( b, dtype=float )
    if b.ndim == 0:
      b = numeric.fastrepeat( b[_], self.shape[0] )
    else:
      assert b.ndim == 1, 'right-hand-side has shape %s, expected a vector' % (b.shape,)
      assert b.shape == self.shape[:1]
  
    if constrain is lconstrain is rconstrain is None:
      return numeric.solve( self.data, b )

    x, I, J = parsecons( constrain, lconstrain, rconstrain, self.shape )
    data = self.data[I]

    x[J] = numeric.solve( data[:,J], b[I] - numeric.dot( data[:,~J], x[~J] ) )
    log.info( 'done' )

    return x

# vim:shiftwidth=2:foldmethod=indent:foldnestmax=2
