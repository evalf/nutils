from . import util, numpy, _

class ElemEval( object ):
  '''Element evaluation.

  Combines a specific Element instance with a shared LocalPoints
  instance, typically a set of integration points for this specific
  element type.
  
  Members:
   * elem: Element instance
   * points: LocalPoints instance
   * weights: integration weights (optional)'''

  def __init__( self, elem, points, transform=1 ):
    'constructor'

    self.elem = elem
    self.points = points
    self.weights = points.weights
    self.transform = transform

  def offset( self, offset ):
    'shift points'

    shifted = LocalPoints( self.points.coords + offset )
    return ElemEval( self.elem, shifted, self.transform )

  @util.cacheprop
  def next( self ):
    'get parent'

    if self.elem.parent is None:
      raise AttributeError, 'next'

    elem, transform = self.elem.parent
    points = transform.eval( self.points )
    return ElemEval( elem, points, transform.transform )

class Element( object ):
  '''Element base class.

  Represents the topological shape.'''

  def eval( self, where ):
    'get points'

    if isinstance( where, str ):
      points = self.getischeme( self.ndims, where )
    else:
      where = numpy.asarray( where )
      points = LocalPoints( where )
    return ElemEval( self, points )

  def zoom( self, elemset, points ):
    'zoom points'

    elem = self
    totaltransform = 1
    while elem not in elemset:
      elem, transform = self.parent
      points = transform( points )
      totaltransform = numpy.dot( transform.transform, totaltransform )
    return elem, points, totaltransform

class AffineTransformation( object ):
  'affine transformation'

  def __init__( self, offset, transform ):
    'constructor'

    self.offset = numpy.asarray( offset )
    self.transform = numpy.asarray( transform )

  @util.cachefunc
  def eval( self, points ):
    'apply transformation'

    if self.transform.ndim == 0:
      coords = self.offset[:,_] + self.transform * points.coords
    elif self.transform.shape[1] == 0:
      assert points.coords.shape == (0,1)
      coords = self.offset[:,_]
    else:
      coords = self.offset[:,_] + numpy.dot( self.transform, points.coords )
    return LocalPoints( coords, points.weights )

class QuadElement( Element ):
  'quadrilateral element'

  def __init__( self, ndims, parent=None ):
    'constructor'

    self.ndims = ndims
    self.parent = parent
    Element.__init__( self )

  @util.classcache
  def edgetransform( cls, ndims ):
    'edge transforms'

    transforms = []
    for idim in range( ndims ):
      for iside in range( 2 ):
        offset = numpy.zeros( ndims )
        offset[idim:] = 1-iside
        offset[:idim+1] = iside
        transform = numpy.zeros(( ndims, ndims-1 ))
        transform.flat[ :(ndims-1)*idim :ndims] = 1 - 2 * iside
        transform.flat[ndims*(idim+1)-1::ndims] = 2 * iside - 1
        transforms.append( AffineTransformation( offset=offset, transform=transform ) )
    return transforms

  def edge( self, iedge ):
    'edge'

    transform = self.edgetransform( self.ndims )[ iedge ]
    return QuadElement( self.ndims-1, parent=(self,transform) )

  @util.classcache
  def refinedtransform( cls, ndims, n ):
    'refined transform'

    transforms = []
    transform = 1. / n
    for i in range( n**ndims ):
      offset = numpy.zeros( ndims )
      for idim in range( ndims ):
        offset[ ndims-1-idim ] = transform * ( i % n )
        i //= n
      transforms.append( AffineTransformation( offset=offset, transform=transform ) )
    return transforms

  def refined( self, n ):
    'refine'

    return [ QuadElement( self.ndims, parent=(self,transform) ) for transform in self.refinedtransform( self.ndims, n ) ]

  @util.classcache
  def getischeme( cls, ndims, where ):
    'get integration scheme'

    if ndims == 0:
      return LocalPoints( numpy.zeros([0,1]), 1. )

    x = w = None
    if where.startswith( 'gauss' ):
      N = int( where[5:] )
      k = numpy.arange( 1, N )
      d = k / numpy.sqrt( 4*k**2-1 )
      x, w = numpy.linalg.eigh( numpy.diagflat(d,-1) ) # eigh operates (by default) on lower triangle
      w = w[0]**2
      x = ( x + 1 ) * .5
    elif where.startswith( 'uniform' ):
      N = int( where[7:] )
      x = numpy.arange( .5, N ) / N
      w = util.appendaxes( 1./N, N )
    elif where.startswith( 'subdivision' ):
      N = int( where[11:] ) + 1
      x = numpy.linspace( 0, 1, N )
      w = None
    elif where.startswith( 'contour' ):
      N = int( where[7:] )
      p = numpy.linspace( 0, 1, N )
      if ndims == 1:
        coords = p[_]
      elif ndims == 2:
        coords = numpy.array([ p[ range(N) + [N-1]*(N-2) + range(N)[::-1] + [0]*(N-2) ],
                               p[ [0]*(N-1) + range(N) + [N-1]*(N-2) + range(1,N)[::-1] ] ])
      else:
        raise Exception, 'contour not supported for ndims=%d' % ndims
    else:
      raise Exception, 'invalid element evaluation %r' % where
    if x is not None:
      coords = reduce( lambda coords, i:
        numpy.concatenate(( x[:,_].repeat( N**i, 1 ).reshape( 1, -1 ),
                       coords[:,_].repeat( N,    1 ).reshape( i, -1 ) )), range( 1, ndims ), x[_] )
    if w is not None:
      weights = reduce( lambda weights, i: ( weights * w[:,_] ).ravel(), range( 1, ndims ), w )
    else:
      weights = None
    return LocalPoints( coords, weights )

  def __repr__( self ):
    'string representation'

    return '%s#%x<ndims=%d>' % ( self.__class__.__name__, id(self), self.ndims )

class TriangularElement( Element ):
  'triangular element'

  ndims = 2
  edgetransform = (
    AffineTransformation( offset=[0,0], transform=[[ 1],[ 0]] ),
    AffineTransformation( offset=[1,0], transform=[[-1],[ 1]] ),
    AffineTransformation( offset=[0,1], transform=[[ 0],[-1]] ) )

  def __init__( self, parent=None ):
    'constructor'

    self.parent = parent
    Element.__init__( self )

  def edge( self, iedge ):
    'edge'

    transform = self.edgetransform[ iedge ]
    return QuadElement( ndims=1, parent=(self,transform) )

  @util.classcache
  def refinedtransform( cls, n ):
    'refined transform'

    offset = numpy.array( [.5,.5] )
    ROT = numpy.array( [[-.5,.5],[-.5,-.5]] )
    ROT = ROT, ROT.T
    if n == 2:
      transforms = [ AffineTransformation( offset=offset, transform=rot ) for rot in ROT ]
    else:
      assert n > 2
      transforms = [ AffineTransformation( offset = transform.offset + numpy.dot( transform.transform, offset ),
                                           transform = numpy.dot( transform.transform, rot ) )
                       for transform in cls.refinedtransform( n-1 ) for rot in ROT ]
    return transforms

  def refined( self, n ):
    'refine'

    return [ TriangularElement( parent=(self,transform) ) for transform in self.refinedtransform( n ) ]

  @util.classcache
  def getischeme( cls, ndims, where ):
    'get integration scheme'

    assert ndims == 2
    if where.startswith( 'contour' ):
      n = int( where[7:] or 0 )
      p = numpy.arange( n+1, dtype=float ) / (n+1)
      z = numpy.zeros_like( p )
      coords = numpy.hstack(( [1-p,p], [z,1-p], [p,z] ))
      weights = None
    elif where == 'gauss1':
      coords = numpy.array( [[1],[1]] ) / 3.
      weights = numpy.array( [1] ) / 2.
    elif where == 'gauss2':
      coords = numpy.array( [[4,1,1],[1,4,1]] ) / 6.
      weights = numpy.array( [1,1,1] ) / 6.
    elif where == 'gauss3':
      coords = numpy.array( [[5,9,3,3],[5,3,9,3]] ) / 15.
      weights = numpy.array( [-27,25,25,25] ) / 96.
    elif where == 'gauss4':
      A = 0.091576213509771; B = 0.445948490915965; W = 0.329855230965966
      coords = numpy.array( [[1-2*A,A,A,1-2*B,B,B],[A,1-2*A,A,B,1-2*B,B]] )
      weights = numpy.array( [W,W,W,1-W,1-W,1-W] ) / 6.
    else:
      raise Exception, 'invalid element evaluation: %r' % where
    return LocalPoints( coords, weights )

  def __repr__( self ):
    'string representation'

    return '%s#%x' % ( self.__class__.__name__, id(self) )

class LocalPoints( object ):
  'local point coordinates'

  def __init__( self, coords, weights=None ):
    'constructor'

    self.coords = coords
    self.weights = weights
    self.ndims, self.npoints = coords.shape

class StdElem( object ):
  'stdelem base class'

  pass

class PolyQuad( StdElem ):
  'poly quod'

  @util.classcache
  def __new__( cls, degree ):
    'constructor'

    self = object.__new__( cls )
    self.degree = degree
    self.ndims = len( degree )
    return self

  @util.classcache
  def comb( cls, p ):
    'comb'

    comb = numpy.empty( p, dtype=int )
    comb[0] = 1
    for j in range( 1, p ):
      comb[j] = ( comb[j-1] * (p-j) ) / j
    assert comb[-1] == 1
    return comb

  def __repr__( self ):
    'string representation'

    return 'PolyQuad#%x<degree=%s>' % ( id(self), ','.join( map( str, self.degree ) ) )

  @util.cachefunc
  def eval( self, points, grad=0 ):
    'evaluate'

    polydata = [ ( x, p, self.comb(p) ) for ( x, p ) in zip( points.coords, self.degree ) ]
    nshapes = numpy.prod( self.degree )

    F0 = [ numpy.array( [ [1.] ] if p == 1
                   else [ comb[i] * (1-x)**(p-1-i) * x**i for i in range(p) ]
                      ) for x, p, comb in polydata ]
    if grad == 0:
      return reduce( lambda f, fi: ( f[:,_] * fi ).reshape( -1, points.npoints ), F0 )

    F1 = [ numpy.array( [ [0.] ] if p < 2
                   else [ [-1.],[1.] ] if p == 2
                   else [ (1-p) * (1-x)**(p-2) ]
                      + [ comb[i] * (1-x)**(p-i-2) * x**(i-1) * (i-(p-1)*x) for i in range(1,p-1) ]
                      + [ (p-1) * x**(p-2) ]
                      ) for x, p, comb in polydata ]
    if grad == 1:
      data = numpy.empty(( nshapes, points.ndims, points.npoints ))
      for n in range( points.ndims ):
        Gi = [( F1 if m == n else F0 )[m] for m in range( points.ndims ) ]
        data[:,n] = reduce( lambda g, gi: ( g[:,_] * gi ).reshape( g.shape[0] * gi.shape[0], -1 ), Gi )
      return data

    F2 = [ numpy.array( [ [0.] ] * p if p < 3
                   else [ [2.],[-4.],[2.] ] if p == 3
                   else [ (p-1) * (p-2) * (1-x)**(p-3), (p-1) * (p-2) * (1-x)**(p-4) * ((p-1)*x-2) ]
                      + [ comb[i] * (1-x)**(p-i-3) * x**(i-2) * (x*(2*i-(p-1)*x)*(2-p)+i*(i-1)) for i in range(2,p-2) ]
                      + [ (p-1) * (p-2) * x**(p-4) * ((p-1)*(1-x)-2), (p-1) * (p-2) * x**(p-3) ]
                        ) for x, p, comb in polydata ]
    if grad == 2:
      data = numpy.empty(( nshapes, points.ndims, points.ndims, points.npoints ))
      for ni in range( points.ndims ):
        for nj in range( ni, points.ndims ):
          Di = [( F2 if m == ni == nj else F1 if m == ni or m == nj else F0 )[m] for m in range( points.ndims ) ]
          data[:,nj,ni] = data[:,ni,nj] = reduce( lambda d, di: ( d[:,_] * di ).reshape( d.shape[0] * di.shape[0], -1 ), Di )
      return data

    raise Exception

class PolyTriangle( StdElem ):
  'poly triangle'

  @util.classcache
  def __new__( cls, order ):
    'constructor'

    assert order == 1
    self = object.__new__( cls )
    return self

  @util.cachefunc
  def eval( self, points, grad=0 ):
    'eval'

    if grad == 0:
      x, y = points.coords
      data = numpy.array( [ x, y, 1-x-y ] )
    elif grad == 1:
      data = numpy.array( [[[1],[0]],[[0],[1]],[[-1],[-1]]], dtype=float )
    else:
      data = numpy.array( 0 ).reshape( (1,) * (grad+1+points.ndim) )
    return data

  def __repr__( self ):
    'string representation'

    return '%s#%x' % ( self.__class__.__name__, id(self) )

class ExtractionWrapper( object ):
  'extraction wrapper'

  def __init__( self, stdelem, extraction ):
    'constructor'

    self.stdelem = stdelem
    self.extraction = extraction

  @util.cachefunc
  def eval( self, points, grad=0 ):
    'call'

    return numpy.dot( self.stdelem.eval( points, grad ).T, self.extraction.T ).T

  def __repr__( self ):
    'string representation'

    return '%s#%x:%s' % ( self.__class__.__name__, id(self), self.stdelem )

# vim:shiftwidth=2:foldmethod=indent:foldnestmax=2
