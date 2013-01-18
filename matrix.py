from . import util, numpy, log, numeric, _
from scipy.sparse.sparsetools.csr import _csr
from scipy.sparse.linalg.isolve import _iterative

def cg( matvec, b, x0=None, tol=1e-5, maxiter=None, precon=None, title='solving system' ):
  'CG solver'

  n = len(b)

  pbar = log.ProgressBar( numpy.log(tol), title=title )
  pbar.add( '[CG:%d]' % n )
  clock = pbar.out and util.Clock( .1 )

  if x0 is None:
    x = numpy.zeros( n )
    res0 = numpy.linalg.norm( b )
  else:
    assert len(x0) == n
    x = x0
    res0 = numpy.linalg.norm( b - matvec(x) )

  if maxiter is None:
    maxiter = n*10

  assert x.dtype == float
  revcom = _iterative.dcgrevcom
  stoptest = _iterative.dstoptest2

  resid = tol
  ndx1 = 1
  ndx2 = -1
  work = numpy.zeros(4*n)
  ijob = 1
  info = 0
  ftflag = True
  bnrm2 = -1.0
  iter_ = maxiter
  while True:
    x, iter_, resid, info, ndx1, ndx2, sclr1, sclr2, ijob = revcom( b, x, work, iter_, resid, info, ndx1, ndx2, ijob )
    if clock:
      pbar.update( numpy.log( numpy.linalg.norm( b - matvec(x) ) / res0 ) )
    vec1 = work[ndx1-1:ndx1-1+n]
    vec2 = work[ndx2-1:ndx2-1+n]
    if ijob == 1:
      vec2 *= sclr2
      vec2 += sclr1 * matvec(vec1)
    elif ijob == 2:
      vec1[:] = precon(vec2) if precon else vec2
    elif ijob == 3:
      vec2 *= sclr2
      vec2 += sclr1 * matvec(x)
    elif ijob == 4:
      if ftflag:
        info = -1
        ftflag = False
      bnrm2, resid, info = stoptest(vec1, b, bnrm2, tol, info)
    else:
      assert ijob == -1
      break
    ijob = 2

  if info > 0 and iter_ == maxiter and resid > tol:
    #info isn't set appropriately otherwise
    info = iter_

  pbar.close()
  assert info == 0
  return x

def gmres( matvec, b, x0=None, tol=1e-5, restrt=None, maxiter=None, precon=None, title='solving system' ):
  'GMRES solver'

  n = len(b)

  pbar = log.ProgressBar( numpy.log(tol), title=title )
  pbar.add( '[GMRES:%d]' % n )
  clock = pbar.out and util.Clock( .1 )

  if x0 is None:
    x = numpy.zeros( n )
  else:
    assert len(x0) == n
    x = x0

  if maxiter is None:
    maxiter = n*10

  if restrt is None:
    restrt = 20
  restrt = min(restrt, n)

  assert x.dtype == float
  revcom = _iterative.dgmresrevcom
  stoptest = _iterative.dstoptest2

  resid = tol
  ndx1 = 1
  ndx2 = -1
  work  = numpy.zeros((6+restrt)*n,dtype=x.dtype)
  work2 = numpy.zeros((restrt+1)*(2*restrt+2),dtype=x.dtype)
  ijob = 1
  info = 0
  ftflag = True
  bnrm2 = -1.0
  iter_ = maxiter
  old_ijob = ijob
  first_pass = True
  resid_ready = False
  while True:
    x, iter_, resid, info, ndx1, ndx2, sclr1, sclr2, ijob = revcom(b, x, restrt, work, work2, iter_, resid, info, ndx1, ndx2, ijob)
    slice1 = slice(ndx1-1, ndx1-1+n)
    slice2 = slice(ndx2-1, ndx2-1+n)
    if (ijob == 1):
      work[slice2] *= sclr2
      work[slice2] += sclr1*matvec(x)
    elif (ijob == 2):
      work[slice1] = precon(work[slice2]) if precon else work[slice2]
      if not first_pass and old_ijob==3:
        resid_ready = True
      first_pass = False
    elif (ijob == 3):
      work[slice2] *= sclr2
      work[slice2] += sclr1*matvec(work[slice1])
      if resid_ready and clock:
        pbar.update( numpy.log(resid) )
        resid_ready = False
    elif (ijob == 4):
      if ftflag:
        info = -1
        ftflag = False
      bnrm2, resid, info = stoptest(work[slice1], b, bnrm2, tol, info)
    else:
      assert ijob == -1
      break
    old_ijob = ijob
    ijob = 2

  if info >= 0 and resid > tol:
    #info isn't set appropriately otherwise
    info = maxiter

  pbar.close()
  assert info == 0
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

    self.shape = nrows, ncols
    self.size = nrows * ncols

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

  def solve( self, b=0, constrain=None, lconstrain=None, rconstrain=None, tol=0, x0=None, symmetric=False, maxiter=99999, title='solving system' ):
    'solve'

    if tol == 0:
      return self.todense().solve( b=b, constrain=constrain, lconstrain=lconstrain, rconstrain=rconstrain, title=title )
  
    b = numpy.asarray( b, dtype=float )
    if b.ndim == 0:
      b = numeric.fastrepeat( b[_], self.shape[0] )
    else:
      assert b.ndim == 1, 'right-hand-side has shape %s, expected a vector' % (b.shape,)
      if b.shape[0] < self.shape[0]: # quick resize hack, temporary!
        b = b.copy()
        b.resize( self.shape[0] )
      assert b.shape == self.shape[:1]
  
    solver = cg if symmetric else gmres

    if constrain is lconstrain is rconstrain is None:
      return solver( self.matvec, b, tol=tol, maxiter=maxiter, title=title )

    x, I, J = parsecons( constrain, lconstrain, rconstrain, self.shape )
    if x0 is not None:
      x0 = x0[J]
    b = ( b - self.matvec(x) )[I]
    tmpvec = numpy.zeros( self.shape[1] )
    def matvec( v ):
      tmpvec[J] = v
      return self.matvec(tmpvec)[I]
    x[J] = solver( matvec, b, x0=x0, tol=tol, maxiter=maxiter, title=title )

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

  def __mul__( self, other ):
    'matrix-vector multiplication'

    return numpy.dot( self.data, other )

  def solve( self, b=0, constrain=None, lconstrain=None, rconstrain=None, title='solving system', **dummy ):
    'solve'

    b = numpy.asarray( b, dtype=float )
    if b.ndim == 0:
      b = numeric.fastrepeat( b[_], self.shape[0] )
    else:
      if b.shape[0] < self.shape[0]: # quick resize hack, temporary!
        b = b.copy()
        b.resize( self.shape[0] )
      assert b.shape == self.shape[:1]
  
    if constrain is lconstrain is rconstrain is None:
      return numpy.linalg.solve( self.data, b )

    x, I, J = parsecons( constrain, lconstrain, rconstrain, self.shape )
    data = self.data[I]

    pbar = log.ProgressBar( None, title=title )
    pbar.add( '[direct:%d]' % data.shape[0] )
    x[J] = numpy.linalg.solve( data[:,J], b[I] - numpy.dot( data[:,~J], x[~J] ) )
    pbar.close()

    return x

# vim:shiftwidth=2:foldmethod=indent:foldnestmax=2
