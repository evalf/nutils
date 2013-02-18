from . import util, numpy, log, numeric, _
from scipy.sparse.sparsetools.csr import _csr
from scipy.sparse.linalg.isolve import _iterative

def krylov( matvec, b, x0=None, tol=1e-5, restart=None, maxiter=0, precon=None, log=log ):
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
    out = log.debug( 'CG' )
    work = numpy.zeros( 4*n, dtype=numpy.float64 )
    ijob_matvecx = 3
    revcom = lambda x, iiter, res, info, ndx1, ndx2, ijob: \
      _iterative.dcgrevcom( b, x, work, iiter, res, info, ndx1, ndx2, ijob )
  else:
    if restart > n:
      restart = n
    out = log.debug( 'GMRES%d' % restart )
    work = numpy.zeros( (6+restart)*n, dtype=numpy.float64 )
    work2 = numpy.zeros( (restart+1)*(2*restart+2), dtype=numpy.float64 )
    ijob_matvecx = 1
    revcom = lambda x, iiter, res, info, ndx1, ndx2, ijob: \
      _iterative.dgmresrevcom( b, x, restart, work, work2, iiter, res, info, ndx1, ndx2, ijob )
  stoptest = lambda vec1, bnrm2, info: \
    _iterative.dstoptest2( vec1, b, bnrm2, tol, info )

  x = x0
  it = out.iter( 'residual', target=numpy.log(tol) )
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
      it.update( numpy.log(res) )
    else:
      assert ijob == -1
      break
    ijob = 2

  assert info == 0
  out.info( 'converged in %d iterations' % iiter )
  return x

def parsecons( constrain, lconstrain, rconstrain, shape ):
  'parse constraints'

  J = numpy.ones( shape[0], dtype=bool )
  x = numpy.empty( shape[1] )
  x[:] = numpy.nan
  if constrain is not None:
    J[:constrain.size] = numpy.isnan( constrain )
    x[:constrain.size] = constrain
  if lconstrain is not None:
    x[:lconstrain.size] = lconstrain
  if rconstrain is not None:
    assert isinstance( rconstrain, numpy.ndarray ) and rconstrain.dtype == bool
    J[:rconstrain.size] = rconstrain
  I = numpy.isnan(x)
  assert numpy.sum(I) == numpy.sum(J)
  x[I] = 0
  return x, I, J

class Matrix( object ):
  'matrix base class'

  def __init__( self, (nrows,ncols) ):
    'constructor'

    assert type(nrows) is int # need exact type because of _csr funcs
    assert type(ncols) is int
    self.shape = nrows, ncols
    self.size = nrows * ncols

  def __add__( self, other ):
    'add'

    A = self.clone()
    A += other
    return A

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
      nrows = len(self.indptr) - 1
    else:
      nrows = len(graph)
      nzrow = map(len,graph)
      count = sum( nzrow )
      self.data = numpy.zeros( count, dtype=float )
      self.indptr = numpy.cumsum( [0] + nzrow, dtype=numpy.intc )
      self.indices = numpy.empty( count, dtype=numpy.intc )
      for irow, icols in enumerate( graph ):
        a, b = self.indptr[irow:irow+2]
        self.indices[a:b] = icols
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
      raise Excaption, 'invalid column argument'
    ncols = len(cols)

    if isinstance(rows,numpy.ndarray) and rows.dtype == bool:
      assert len(rows) == self.shape[0]
      rows, = rows.nonzero()
    elif isinstance(rows,numpy.ndarray) and rows.dtype == int:
      pass
    else:
      raise Excaption, 'invalid row argument'
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

  def __iadd__( self, other ):
    'in place addition'

    if not other:
      return self

    assert isinstance( other, self.__class__ ) and self.shape == other.shape
    if numpy.all( self.indptr == other.indptr ) and numpy.all( self.indices == other.indices ):
      I = slice(None)
    else:
      I = numpy.empty( other.data.shape, dtype=int )
      for irow in range( self.shape[0] ):
        s = slice( other.indptr[irow], other.indptr[irow+1] )
        I[s] = self.indptr[irow] \
             + numpy.searchsorted( self.indices[self.indptr[irow]:self.indptr[irow+1]], other.indices[s] )
      assert all( self.indices[I] == other.indices )
    self.data[I] += other.data
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
      supp[irow] = a == b or tol != 0 and numpy.all( numpy.abs( self.data[a:b] ) < tol )
    return supp

  def solve( self, b=0, constrain=None, lconstrain=None, rconstrain=None, tol=0, x0=None, symmetric=False, maxiter=0, restart=999, title='solving system', log=log ):
    'solve'

    if tol == 0:
      return self.todense().solve( b=b, constrain=constrain, lconstrain=lconstrain, rconstrain=rconstrain, title=title, log=log )

    out = log.debug( title )
  
    b = numpy.asarray( b, dtype=float )
    if b.ndim == 0:
      b = numeric.fastrepeat( b[_], self.shape[0] )
    else:
      assert b.ndim == 1, 'right-hand-side has shape %s, expected a vector' % (b.shape,)
      assert b.shape == self.shape[:1]
  
    if symmetric:
      restart = None

    if constrain is lconstrain is rconstrain is None:
      return krylov( self.matvec, b, tol=tol, maxiter=maxiter, restart=restart, log=out )

    x, I, J = parsecons( constrain, lconstrain, rconstrain, self.shape )
    if x0 is not None:
      x0 = x0[J]
    b = ( b - self.matvec(x) )[I]
    tmpvec = numpy.zeros( self.shape[1] )
    def matvec( v ):
      tmpvec[J] = v
      return self.matvec(tmpvec)[I]
    x[J] = krylov( matvec, b, x0=x0, tol=tol, maxiter=maxiter, restart=restart, log=out )

    ##ALTERNATIVE
    #from scipy.sparse import linalg
    #xJ, info = linalg.gmres( linalg.LinearOperator(b.shape*2,matvec,dtype=float), b, tol=tol, maxiter=maxiter )
    #assert info == 0
    #x[J] = xJ
    ##END

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
      self.data = numpy.zeros( shape )
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

  def solve( self, b=0, constrain=None, lconstrain=None, rconstrain=None, title='solving system', log=log, **dummy ):
    'solve'

    out = log.debug( title )

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

    out = out.debug( 'direct' )
    x[J] = numpy.linalg.solve( data[:,J], b[I] - numpy.dot( data[:,~J], x[~J] ) )
    out.info( 'done' )

    return x

# vim:shiftwidth=2:foldmethod=indent:foldnestmax=2
