from . import util, numpy, _
from scipy.sparse.sparsetools.csr import _csr

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

def iterative_solve( matmul, rhs, title=True, symmetric=False, tol=0, maxiter=999999 ):
  'solve linear system iteratively'

  from scipy.sparse import linalg

  if title is True:
    title = 'solve system'

  shape = rhs.shape * 2
  matrix = linalg.LinearOperator( matvec=matmul, shape=shape, dtype=float )

  progress = None
  callback = None

  if title:
    progress = util.progressbar( n=numpy.log(tol), title='%s [%s:%d]' % ( title, symmetric and 'CG' or 'GMRES', matrix.shape[0] ) )
    clock = util.Clock( .1 )
    callback = lambda vec: clock and progress.update(
      numpy.log( vec if not symmetric else numpy.linalg.norm( rhs - matrix * vec ) ) )

  solver = linalg.cg if symmetric else linalg.gmres
  lhs, status = solver( matrix, rhs, callback=callback, tol=tol, maxiter=maxiter )
  assert status == 0, 'solution failed to converge (status=%d)' % status

  if progress:
    progress.finish()

  return lhs

class SparseBlock( object ):
  'sparse data block that supports in-place addition'

  def __init__( self, data, index ):
    self.data = data
    self.index = index

  def __iadd__( self, data ):
    self.data[ self.index ] += data

  def __str__( self ):
    return str( self.data[ self.index ] )

class SparseMatrix( object ):
  'matrix wrapper class'

  def __init__( self, graph, ncols=None ):
    'constructor'

    if isinstance( graph, tuple ):
      self.data, self.indices, self.indptr = graph
      nrows = len(self.indptr) - 1
    else:
      nrows = len(graph)
      nzrow = map(len,graph)
      count = sum( nzrow )
      self.data = numpy.zeros( count, dtype=float )
      self.indptr = numpy.cumsum( [0] + nzrow, dtype=int )
      self.indices = numpy.empty( count, dtype=int )
      for irow, icols in enumerate( graph ):
        a, b = self.indptr[irow:irow+2]
        self.indices[a:b] = icols

    self.shape = nrows, ncols or nrows

  def reshape( self, (nrows,ncols) ):
    'reshape matrix'

    assert nrows >= self.shape[0] and ncols >= self.shape[1]
    indptr = self.indptr
    if nrows > self.shape[1]:
      indptr = numpy.concatenate([ indptr, util.fastrepeat( indptr[-1:], nrows-self.shape[1] ) ])
    return self.__class__( (self.data,self.indices,indptr), ncols )

  def clone( self ):
    'clone matrix'

    return self.__class__( (self.data.copy(),self.indices,self.indptr), self.shape[1] )

  def __add__( self, other ):
    'addition'

    raise NotImplementedError
    assert isinstance( other, SparseMatrix )
    import scipy.sparse

    if other.shape != self.shape: # quick resize hack, temporary!
      indptr = numpy.concatenate([ other.matrix.indptr, [other.matrix.indptr[-1]] * (self.shape[0]-other.shape[0]) ])
      dii = other.matrix.data, other.matrix.indices, indptr
      othermat = scipy.sparse.csr_matrix( dii, shape=self.shape )
    else:
      othermat = other.matrix

    return SparseMatrix( self.matrix + othermat )

  def matvec( self, other ):
    'matrix-vector multiplication'

    assert other.shape == self.shape[1:]
    result = numpy.zeros( self.shape[0] )
    _csr.csr_matvec( self.shape[0], self.shape[1], self.indptr, self.indices, self.data, other, result )
    return result

  def __getitem__( self, (rows,cols) ):
    'isolate block (for addition)'

    import scipy.sparse

    J = numpy.empty( [len(rows),len(cols)], dtype=int )
    if cols.dtype == bool:
      assert len(cols) == self.shape[1]
      select = cols
      ncols = select.sum()
    else:
      ncols = cols.size
      select = numpy.zeros( self.shape[1], dtype=bool )
      select[cols] = True

    if rows.dtype == bool:
      rows, = rows.nonzero()
    nrows = rows.size

    indptr = numpy.empty( len(rows)+1, dtype=int )
    maxentries = numpy.minimum( self.indptr[rows+1] - self.indptr[rows], ncols ).sum()
    indices = numpy.empty( maxentries, dtype=int ) # allocate for worst case

    indptr[0] = 0
    for n, irow in enumerate( rows ):
      a, b = self.indptr[irow:irow+2]
      where, = select[ self.indices[a:b] ].nonzero()
      c = indptr[n]
      d = c + where.size
      indices[c:d] = a + where
      indptr[n+1] = d

    # TODO return proper CSR matrix
    # matrix = scipy.sparse.csr_matrix( (self.matrix.data,indices,indptr), shape=(nrows,ncols) )

    assert indptr[nrows] == nrows * ncols
    return SparseBlock( self.data, indices.reshape(nrows,ncols) )

  def __iadd__( self, other ):
    'in place addition'

    assert isinstance( other, self.__class__ )
    assert self.shape == other.shape
    if numpy.all( self.indptr == other.indptr ) and numpy.all( self.indices == other.indices ):
      self.data += other.data
      return self
    for irow in range( self.shape[0] ):
      c, d = other.indptr[irow:irow+2]
      if c == d:
        continue
      a, b = self.indptr[irow:irow+2]
      I = self.indices[a:b]
      J = other.indices[c:d]
      n = a + I.searchsorted( J )
      assert numpy.all( self.indices[n] == J )
      self.data[n] += other.data[c:d]
    return self

  def __setitem__( self, item, value ):
    'sucks'

    pass

  @property
  def T( self ):
    'transpose'

    raise NotImplementedError
    return SparseMatrix( self.matrix.T.tocsr() )

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

  def solve( self, b=0, constrain=None, lconstrain=None, rconstrain=None, tol=0, **kwargs ):
    'solve'

    if tol == 0:
      return self.todense().solve( b=b, constrain=constrain, lconstrain=lconstrain, rconstrain=rconstrain, tol=tol, **kwargs )
  
    b = numpy.asarray( b, dtype=float )
    if b.ndim == 0:
      b = util.fastrepeat( b[_], self.shape[0] )
    else:
      if b.shape[0] < self.shape[0]: # quick resize hack, temporary!
        b = b.copy()
        b.resize( self.shape[0] )
      assert b.shape == self.shape[:1]
  
    if constrain is lconstrain is rconstrain is None:
      return iterative_solve( self.matvec, b, tol=tol, **kwargs )

    x, I, J = parsecons( constrain, lconstrain, rconstrain, self.shape )
    b = ( b - self.matvec(x) )[I]
    tmpvec = numpy.zeros( self.shape[1] )
    def matvec( v ):
      tmpvec[J] = v
      return self.matvec(tmpvec)[I]
    x[J] = iterative_solve( matvec, b, tol=tol, **kwargs )
    return x

class DenseMatrix( object ):
  'matrix wrapper class'

  def __init__( self, shape ):
    'constructor'

    if isinstance( shape, numpy.ndarray ):
      self.data = shape
      self.shape = self.data.shape
    else:
      if isinstance( shape, int ):
        self.shape = shape, shape
      else:
        assert len(shape) == 2
        self.shape = tuple(shape)
      self.data = numpy.zeros( self.shape )

  def addblock( self, rows, cols, vals ):
    'add matrix data'

    self.data[ rows[:,_], cols[:,_] ] += vals

  def toarray( self ):
    'convert to numpy array'

    return self.data

  @property
  def T( self ):
    'transpose'

    return DenseMatrix( self.data.T )

  def __mul__( self, other ):
    'matrix-vector multiplication'

    return numpy.dot( self.data, other )

  def __getitem__( self, rows, cols ):

    return self.data[ rows[:,_], cols[:,_] ]

  def solve( self, b=0, constrain=None, lconstrain=None, rconstrain=None, tol=0, **kwargs ):
    'solve'

    b = numpy.asarray( b, dtype=float )
    if b.ndim == 0:
      b = util.fastrepeat( b[_], self.shape[0] )
    else:
      assert b.shape == self.shape[:1]
  
    if constrain is lconstrain is rconstrain is None:
      return numpy.linalg.solve( self.data, b )

    x, I, J = parsecons( constrain, lconstrain, rconstrain, self.shape )
    data = self.data[I]
    x[J] = numpy.linalg.solve( data[:,J], b[I] - numpy.dot( data[:,~J], x[~J] ) )
    return x

# vim:shiftwidth=2:foldmethod=indent:foldnestmax=2
