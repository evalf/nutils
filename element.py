from . import util, numpy, core, numeric, function, _
import weakref

class TrimmedIScheme( object ):
  'integration scheme for truncated elements'

  def __init__( self, levelset, ischeme, maxrefine, finestscheme='uniform1', degree=3, retain=None ):
    'constructor'

    self.levelset = levelset
    self.ischeme = ischeme
    self.maxrefine = maxrefine
    self.finestscheme = finestscheme
    self.bezierscheme = 'bezier%d' % degree
    self.retain = retain
    self.cache = {}

  def __getitem__( self, elem ):
    'get ischeme for elem'

    ischeme = self.cache.get( elem, False )
    if ischeme is False:
      ischeme = self.generate_ischeme( elem, self.maxrefine )
      if ischeme is True:
        ischeme = elem.eval( self.ischeme )
      self.cache[elem] = ischeme
    return ischeme

  def generate_ischeme( self, elem, maxrefine ):
    'generate integration scheme'

    if self.retain:
      parents = [elem]
      for i in range(maxrefine):
        allchildren = []
        while parents:
          allchildren += parents.pop().children
        parents = allchildren  

      if not any(self.retain[child] for child in parents):
        return None

    if maxrefine <= 0:
      ipoints, iweights = elem.eval( self.finestscheme )
      inside = self.levelset( elem, ipoints ) > 0
      if inside.all():
        return True
      if not inside.any():
        return None
      return ipoints[inside], iweights[inside]

    try:
      inside = self.levelset( elem, self.bezierscheme ) > 0
    except function.EvaluationError:
      pass
    else:
      if inside.all():
        return True
      if not inside.any():
        return None

    ischemes = [ self.generate_ischeme( child, maxrefine-1 ) for child in elem.children ]
    if all( ischeme is True for ischeme in ischemes ):
      return True
    if all( ischeme is None for ischeme in ischemes ):
      return None

    points = []
    weights = []
    for child, ischeme in zip( elem.children, ischemes ):
      if ischeme is None:
        continue
      if ischeme is True:
        ischeme = child.eval( self.ischeme )
      pelem, transform = child.parent
      assert pelem is elem
      ipoints, iweights = ischeme
      points.append( transform.eval(ipoints) )
      print 'DET', transform.det
      weights.append( iweights * transform.det )

    coords = numpy.concatenate( coords, axis=0 )
    weights = numpy.concatenate( weights, axis=0 )
    return coords, weights

class AffineTransformation( object ):
  'affine transformation'

  def __init__( self, offset, transform ):
    'constructor'

    self.offset = numpy.asarray( offset )
    assert self.offset.ndim == 1
    self.transform = numpy.asarray( transform )
    assert self.transform.ndim == 2
    self.todim, self.fromdim = self.transform.shape
    assert self.offset.shape[0] == self.todim

  @core.cacheprop
  def invtrans( self ):
    'inverse transformation'

    assert self.todim == self.fromdim
    return numpy.linalg.inv( self.transform )

  @core.cacheprop
  def det( self ):
    'determinant'

    assert self.todim == self.fromdim
    return numpy.linalg.det( self.transform )

  def nest( self, other ):
    'merge transformations'

    offset = other.offset + numeric.dot( other.transform, self.offset )
    transform = numeric.dot( other.transform, self.transform )
    return AffineTransformation( offset, transform )

  def get_transform( self ):
    'get transformation copy'

    return self.transform.copy()

  def transform_to( self, A, axis=-1 ):
    'contract with axis 0'

    return numeric.dot( A, self.transform, axis )

  def transform_from( self, A, axis=-1 ):
    'contract with axis 1'

    return numeric.dot( A, self.transform.T, axis )

  def invapply( self, coords ):
    'apply inverse transformation'

    return numeric.dot( coords - self.offset, self.invtrans.T )

  @core.cachefunc
  def eval( self, points ):
    'apply transformation'

    assert isinstance( points, numpy.ndarray )
    return util.ImmutableArray( self.offset + numeric.dot( points, self.transform.T ) )

class Element( object ):
  '''Element base class.

  Represents the topological shape.'''

  def __init__( self, ndims, id, index=None, parent=None, context=None ):
    'constructor'

    self.ndims = ndims
    self.id = id
    assert index is None or parent is None
    self.index = index
    self.parent = parent
    self.context = context

  def eval( self, where ):
    'get points'

    if isinstance( where, str ):
      points, weights = self.getischeme( self.ndims, where )
    else:
      points = util.ImmutableArray( where )
      weights = None
    return points, weights

  def zoom( self, elemset, points ):
    'zoom points'

    elem = self
    totaltransform = 1
    while elem not in elemset:
      elem, transform = self.parent
      points = transform( points )
      totaltransform = numpy.dot( transform.transform, totaltransform )
    return elem, points, totaltransform

  def __repr__( self ):
    'string representation'

    return self.id

  def __str__( self ):
    'string representation'

    return self.id

  def __hash__( self ):
    'hash'

    return hash(self.id)

  def __eq__( self, other ):
    'hash'

    return self is other or self.id == other.id

  def intersected( self, levelset, lscheme, evalrefine=0 ):
    '''check levelset intersection:
      +1 for levelset > 0 everywhere
      -1 for levelset < 0 everywhere
       0 for intersected element'''

    elems = iter( [self] )
    for irefine in range(evalrefine):
      elems = ( child for elem in elems for child in elem.children )
    inside = levelset( elems.next(), lscheme ) > 0
    if inside.all():
      for elem in elems:
        inside = levelset( elem, lscheme ) > 0
        if not inside.all():
          return 0
      return 1
    elif not inside.any():
      for elem in elems:
        inside = levelset( elem, lscheme ) > 0
        if inside.any():
          return 0
      return -1
    return 0

  def trim( self, levelset, maxrefine, lscheme, finestscheme, evalrefine ):
    'trim element along levelset'

    intersected = self.intersected( levelset, lscheme, evalrefine )

    if intersected > 0:
      return self

    if intersected < 0:
      return None

    parent = self, AffineTransformation( numpy.zeros(self.ndims), numpy.eye(self.ndims) )
    return TrimmedElement( elem=self, levelset=levelset, maxrefine=maxrefine, lscheme=lscheme, finestscheme=finestscheme, evalrefine=evalrefine, parent=parent, id=self.id+'.trim' )

  def get_simplices ( self, **kwargs ):
    'divide in simple elements'

    return self,

class TrimmedElement( Element ):
  'trimmed element'

  def __init__( self, elem, levelset, maxrefine, lscheme, finestscheme, evalrefine, parent, id ):
    'constructor'

    assert not isinstance( elem, TrimmedElement )
    self.elem = elem
    self.levelset = levelset
    self.maxrefine = maxrefine
    self.lscheme = lscheme
    self.finestscheme = finestscheme
    self.evalrefine = evalrefine

    Element.__init__( self, ndims=elem.ndims, id=id, parent=parent )

  @core.cachefunc
  def eval( self, ischeme ):
    'get integration scheme'

    assert isinstance( ischeme, str )

    if ischeme[:7] == 'contour':
      n = int(ischeme[7:] or 0)
      points, weights = self.elem.eval( 'contour{}'.format(n) )
      inside = self.levelset( self.elem, points ) >= 0
      return points[inside], None

    if self.maxrefine <= 0:
      if self.finestscheme is None:

        points  = []
        weights = []

        for simplex in self.get_simplices( 0 ):
          spoints, sweights = simplex.eval( ischeme )
          pelem, transform = simplex.parent

          assert pelem is self 

          points.append( transform.eval( spoints ) )
          weights.append( sweights * transform.det )

        points  = util.ImmutableArray(numpy.concatenate(points,axis=0))
        weights = util.ImmutableArray(numpy.concatenate(weights))

        return points, weights

      else:
        
        points, weights = self.elem.eval( self.finestscheme )
        inside = self.levelset( self.elem, points ) > 0
        return points[inside], weights[inside] if weights is not None else None
        

    allcoords = []
    allweights = []
    for child in self.children:
      if child is None:
        continue
      points, weights = child.eval( ischeme )
      pelem, transform = child.parent
      assert pelem == self
      allcoords.append( transform.eval(points) )
      allweights.append( weights * transform.det )

    coords = util.ImmutableArray( numpy.concatenate( allcoords, axis=0 ) )
    weights = util.ImmutableArray( numpy.concatenate( allweights, axis=0 ) )
    return coords, weights

  @core.cacheprop
  def children( self ):
    'all 1x refined elements'

    children = []
    for ielem, child in enumerate( self.elem.children ):
      isect = child.intersected( self.levelset, self.lscheme, self.evalrefine-1 )
      pelem, transform = child.parent
      parent = self, transform
      if isect < 0:
        child = None
      elif isect > 0:
        child = QuadElement( id=self.id+'.child({})'.format(ielem), ndims=self.ndims, parent=parent )
      else:
        child = TrimmedElement( id=self.id+'.trimmedchild({})'.format(ielem), elem=child, levelset=self.levelset, maxrefine=self.maxrefine-1, lscheme=self.lscheme, finestscheme=self.finestscheme, evalrefine=self.evalrefine-1, parent=parent )
      children.append( child )
    return children

  def edge( self, iedge ):
    'edge'

    # TODO fix trimming of edges once refine/edge operations commute
    transform = self.elem.edgetransform( self.ndims )[ iedge ]
    return QuadElement( id=self.id+'.edge({})'.format(iedge), ndims=self.ndims-1, context=(self,transform) )

  def get_simplices ( self, maxrefine=3, **kwargs ):
    'divide in simple elements'

    if maxrefine > 0 or self.evalrefine > 0:
      return [ simplex for child in filter(None,self.children) for simplex in child.get_simplices( maxrefine=maxrefine-1 ) ]

    ischeme = self.elem.getischeme( self.elem.ndims, 'bezier2' )
    where   = self.levelset( self.elem, ischeme ) > 0
    points  = ischeme[0][where]

    if not where.any():
      return []

    if where.all():
	    lines = []
    else:		
    	lines = self.elem.ribbons  

    for line in lines:
      
      ischeme = line.getischeme( line.ndims, 'bezier2' )
      vals    = self.levelset( line, ischeme )
      pts     = ischeme[0]
      where   = vals > 0

      if  where[0] != where[1]:

        xi = vals[0] / ( vals[0] - vals[1] )

        assert xi > 0 and xi < 1, 'Illegal local coordinate'
 
        elem, transform = line.context

        pts = transform.eval( pts )

        newpoint = pts[0] + xi * ( pts[1] - pts[0] )

        points   = numpy.append( points, newpoint[_], axis=0 ) 

    try:
      submesh = util.delaunay( points )
    except RuntimeError:
      return []

    simplices = []
    Element   = TriangularElement if self.ndims == 2 else TetrahedronElement

    for i, tri in enumerate(submesh.vertices):

      for i in range(2): #Flip two points in case of negative determinant
        offset = points[ tri[0] ]
        affine = numpy.array( [ points[ tri[i+1] ] - offset for i in range(self.ndims) ] ).T

        transform = AffineTransformation( offset, affine )

        if transform.det > 0.:
          break

        tri[-2:] = tri[:-3:-1]
      else:
        if abs(transform.det) < numpy.spacing(1):
          continue
        raise Exception('Negative determinant with value %12.10e could not be resolved' % transform.det )

      simplices.append( Element( self.id + '.simplex(%d)' % i, parent=(self,transform) ) )

    return simplices
  
class QuadElement( Element ):
  'quadrilateral element'

  @property
  def children( self ):
    'all 1x refined elements'

    return ( QuadElement( id=self.id+'.child({})'.format(ielem), ndims=self.ndims, parent=(self,transform) )
      for ielem, transform in enumerate( self.refinedtransform( self.ndims, 2 ) ) )

  @core.classcache
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

  @property
  def ribbons( self ):
    'ribbons'

    if self.ndims == 2:
      transforms = self.edgetransform( self.ndims )
    elif self.ndims == 3:
      transforms = (
        AffineTransformation( offset=[0,0,0], transform=[[1],[0],[0]] ),
        AffineTransformation( offset=[0,0,0], transform=[[0],[1],[0]] ),
        AffineTransformation( offset=[0,0,0], transform=[[0],[0],[1]] ),
        AffineTransformation( offset=[1,1,1], transform=[[-1],[0],[0]] ),
        AffineTransformation( offset=[1,1,1], transform=[[0],[-1],[0]] ),
        AffineTransformation( offset=[1,1,1], transform=[[0],[0],[-1]] ),
        AffineTransformation( offset=[1,0,0], transform=[[0],[1],[0]] ),
        AffineTransformation( offset=[1,0,0], transform=[[0],[0],[1]] ),
        AffineTransformation( offset=[0,1,0], transform=[[1],[0],[0]] ),
        AffineTransformation( offset=[0,1,0], transform=[[0],[0],[1]] ),
        AffineTransformation( offset=[0,0,1], transform=[[1],[0],[0]] ),
        AffineTransformation( offset=[0,0,1], transform=[[0],[1],[0]] ) )
    else:
      raise NotImplementedError('Ribbons not implemented for ndims=%d'%self.ndims)

    return [ QuadElement( id=self.id+'.ribbon({})'.format(i), ndims=1, context=(self,transform) ) for i, transform in enumerate( transforms )]

  def edge( self, iedge ):
    'edge'
    transform = self.edgetransform( self.ndims )[ iedge ]
    return QuadElement( id=self.id+'.edge({})'.format(iedge), ndims=self.ndims-1, context=(self,transform) )

  @core.classcache
  def refinedtransform( cls, ndims, n ):
    'refined transform'

    transforms = []
    transform = 1. / n
    for i in range( n**ndims ):
      offset = numpy.zeros( ndims )
      for idim in range( ndims ):
        offset[ ndims-1-idim ] = transform * ( i % n )
        i //= n
      transforms.append( AffineTransformation( offset=offset, transform=numpy.diag([transform]*ndims) ) )
    return transforms

  @core.cachefunc
  def refined( self, n ):
    'refine'

    return [ QuadElement( self.ndims, parent=(self,transform) ) for transform in self.refinedtransform( self.ndims, n ) ]

  @core.classcache
  def getischeme( cls, ndims, where ):
    'get integration scheme'

    if ndims == 0:
      return numpy.zeros([1,0]), numpy.array([1.])

    x = w = None
    if where.startswith( 'gauss' ):
      N = int( where[5:] ) # //2+1 <= FUTURE!
      k = numpy.arange( 1, N )
      d = k / numpy.sqrt( 4*k**2-1 )
      x, w = numpy.linalg.eigh( numpy.diagflat(d,-1) ) # eigh operates (by default) on lower triangle
      w = w[0]**2
      x = ( x + 1 ) * .5
    elif where.startswith( 'uniform' ):
      N = int( where[7:] )
      x = numpy.arange( .5, N ) / N
      w = numeric.appendaxes( 1./N, N )
    elif where.startswith( 'bezier' ):
      N = int( where[6:] )
      x = numpy.linspace( 0, 1, N )
      w = numeric.appendaxes( 1./N, N )
    elif where.startswith( 'subdivision' ):
      N = int( where[11:] ) + 1
      x = numpy.linspace( 0, 1, N )
      w = None
    elif where.startswith( 'vtk' ):
      if ndims == 1:
        coords = numpy.array([[0,0]])
      elif ndims == 2:
        coords = numpy.array([[0,0],[1,0],[1,1],[0,1]]).T
      elif ndims == 3:
        coords = numpy.array([ [0,0,0], [1,0,0], [0,1,0], [1,1,0], [0,0,1], [1,0,1], [0,1,1], [1,1,1] ]).T
      else:
        raise Exception, 'contour not supported for ndims=%d' % ndims
    elif where.startswith( 'contour' ):
      N = int( where[7:] )
      p = numpy.linspace( 0, 1, N )
      if ndims == 1:
        coords = p[_]
      elif ndims == 2:
        coords = numpy.array([ p[ range(N) + [N-1]*(N-2) + range(N)[::-1] + [0]*(N-2) ],
                               p[ [0]*(N-1) + range(N) + [N-1]*(N-2) + range(1,N)[::-1] ] ])
      elif ndims == 3:
        assert N == 0
        coords = numpy.array([ [0,0,0], [1,0,0], [0,1,0], [1,1,0], [0,0,1], [1,0,1], [0,1,1], [1,1,1] ]).T
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
    return util.ImmutableArray( coords.T ), util.ImmutableArray( weights )

  def select_contained( self, points, eps=0 ):
    'select points contained in element'

    selection = numpy.ones( points.shape[0], dtype=bool )
    for idim in range( self.ndims ):
      newsel = ( points[:,idim] >= -eps ) & ( points[:,idim] <= 1+eps )
      selection[selection] &= newsel
      points = points[newsel]
      if not points.size:
        return None, None
    return points, selection

class TriangularElement( Element ):
  'triangular element'

  ndims = 2
  edgetransform = (
    AffineTransformation( offset=[0,0], transform=[[ 1],[ 0]] ),
    AffineTransformation( offset=[1,0], transform=[[-1],[ 1]] ),
    AffineTransformation( offset=[0,1], transform=[[ 0],[-1]] ) )

  def __init__( self, id, index=None, parent=None, context=None ):
    'constructor'

    Element.__init__( self, ndims=2, id=id, index=index, parent=parent, context=context )

  @property
  def children( self ):
    'all 1x refined elements'

    transforms = self.refinedtransform( 2 )
    refs = self.__dict__.get('children')
    if refs:
      for ichild, transform in enumerate( transforms ):
        elem = refs[ ichild ]()
        if not elem:
          elem = TriangularElement( id=self.id+'.child({})'.format(ichild), parent=(self,transform) )
          refs[ ichild ] = weakref.ref(elem)
        yield elem
    else:
      refs = []
      for ichild, transform in enumerate( transforms ):
        elem = TriangularElement( id=self.id+'.child({})'.format(ichild), parent=(self,transform) )
        refs.append( weakref.ref(elem) )
        yield elem
      self.__dict__['children'] = refs
      
  def edge( self, iedge ):
    'edge'

    transform = self.edgetransform[ iedge ]
    return QuadElement( id=self.id+'.edge({})'.format(iedge), ndims=1, context=(self,transform) )

  @core.classcache
  def refinedtransform( cls, n ):
    'refined transform'

    transforms = []
    trans = numpy.diag( [1./n]*2 )
    for i in range( n ):
      transforms.extend( AffineTransformation( offset=numpy.array( [i,j], dtype=float ) / n, transform=trans ) for j in range(0,n-i) )
      transforms.extend( AffineTransformation( offset=numpy.array( [n-j,n-i], dtype=float ) / n, transform=-trans ) for j in range(n-i,n) )
    return transforms

  def refined( self, n ):
    'refine'

    assert n == 2
    if n == 1:
      return self
    return [ TriangularElement( id=self.id+'.child({})'.format(ichild), parent=(self,transform) ) for ichild, transform in enumerate( self.refinedtransform( n ) ) ]

  @core.classcache
  def getischeme( cls, ndims, where ):
    '''get integration scheme
    gaussian quadrature: http://www.cs.rpi.edu/~flaherje/pdf/fea6.pdf
    '''

    assert ndims == 2
    if where.startswith( 'contour' ):
      n = int( where[7:] or 0 )
      p = numpy.arange( n+1, dtype=float ) / (n+1)
      z = numpy.zeros_like( p )
      coords = numpy.hstack(( [1-p,p], [z,1-p], [p,z] ))
      weights = None
    elif where.startswith( 'vtk' ):
      coords = numpy.array([[0,0],[1,0],[0,1]]).T
      weights = None
    elif where == 'gauss1':
      coords = numpy.array( [[1],[1]] ) / 3.
      weights = numpy.array( [1] ) / 2.
    elif where in 'gauss2':
      coords = numpy.array( [[4,1,1],[1,4,1]] ) / 6.
      weights = numpy.array( [1,1,1] ) / 6.
    elif where == 'gauss3':
      coords = numpy.array( [[5,9,3,3],[5,3,9,3]] ) / 15.
      weights = numpy.array( [-27,25,25,25] ) / 96.
    elif where == 'gauss4':
      A = 0.091576213509771; B = 0.445948490915965; W = 0.109951743655322
      coords = numpy.array( [[1-2*A,A,A,1-2*B,B,B],[A,1-2*A,A,B,1-2*B,B]] )
      weights = numpy.array( [W,W,W,1/3.-W,1/3.-W,1/3.-W] ) / 2.
    elif where == 'gauss5':
      A = 0.101286507323456; B = 0.470142064105115; V = 0.125939180544827; W = 0.132394152788506
      coords = numpy.array( [[1./3,1-2*A,A,A,1-2*B,B,B],[1./3,A,1-2*A,A,B,1-2*B,B]] )
      weights = numpy.array( [1-3*V-3*W,V,V,V,W,W,W] ) / 2.
    elif where == 'gauss6':
      A = 0.063089014491502; B = 0.249286745170910; C = 0.310352451033785; D = 0.053145049844816; V = 0.050844906370207; W = 0.116786275726379
      VW = 1/6. - (V+W) / 2.
      coords = numpy.array( [[1-2*A,A,A,1-2*B,B,B,1-C-D,1-C-D,C,C,D,D],[A,1-2*A,A,B,1-2*B,B,C,D,1-C-D,D,1-C-D,C]] )
      weights = numpy.array( [V,V,V,W,W,W,VW,VW,VW,VW,VW,VW] ) / 2.
    elif where == 'gauss7':
      A = 0.260345966079038; B = 0.065130102902216; C = 0.312865496004875; D = 0.048690315425316; U = 0.175615257433204; V = 0.053347235608839; W = 0.077113760890257
      coords = numpy.array( [[1./3,1-2*A,A,A,1-2*B,B,B,1-C-D,1-C-D,C,C,D,D],[1./3,A,1-2*A,A,B,1-2*B,B,C,D,1-C-D,D,1-C-D,C]] )
      weights = numpy.array( [1-3*U-3*V-6*W,U,U,U,V,V,V,W,W,W,W,W,W] ) / 2.
    elif where[:7] == 'uniform' or where[:6] == 'bezier':
      if where[:7] == 'uniform':
        N = int( where[7:] )
        points = ( numpy.arange( N ) + 1./3 ) / N
      else:
        N = int( where[6:] )
        points = numpy.linspace( 0, 1, N )
      NN = N**2
      C = numpy.empty( [2,N,N] )
      C[0] = points[:,_]
      C[1] = points[_,:]
      coords = C.reshape( 2, NN )
      flip = coords[0] + coords[1] > 1
      coords[:,flip] = 1 - coords[::-1,flip]
      weights = numeric.appendaxes( .5/NN, NN )
    else:
      raise Exception, 'invalid element evaluation: %r' % where
    return util.ImmutableArray( coords.T ), util.ImmutableArray( weights )

  def select_contained( self, points, eps=0 ):
    'select points contained in element'

    selection = numpy.ones( points.shape[0], dtype=bool )
    for idim in 0, 1, 2:
      points_i = points[:,idim] if idim < 2 else 1-points.sum(1)
      newsel = ( points_i >= -eps )
      selection[selection] &= newsel
      points = points[newsel]
      if not points.size:
        return None, None

    return points, selection

class TetrahedronElement( Element ):
  'triangular element'

  ndims = 3
  edgetransform = (
    AffineTransformation( offset=[0,0,0], transform=[[ 1, 0],[0,1],[0,0]] ),
    AffineTransformation( offset=[0,0,0], transform=[[ 0, 1],[0,0],[1,0]] ),
    AffineTransformation( offset=[0,0,0], transform=[[ 0, 0],[1,0],[0,1]] ),
    AffineTransformation( offset=[1,0,0], transform=[[-1,-1],[1,0],[0,1]] ) )

  def __init__( self, id, index=None, parent=None, context=None ):
    'constructor'

    Element.__init__( self, ndims=3, id=id, index=index, parent=parent, context=context )

  @property
  def children( self ):
    'all 1x refined elements'
    raise NotImplementedError( 'Children of tetrahedron' )  
      
  def edge( self, iedge ):
    'edge'

    transform = self.edgetransform[ iedge ]
    return TriangularElement( id=self.id+'.edge({})'.format(iedge), ndims=2, context=(self,transform) )

  @core.classcache
  def refinedtransform( cls, n ):
    'refined transform'
    raise NotImplementedError( 'Transformations for refined tetrahedrons' )  

  def refined( self, n ):
    'refine'
    raise NotImplementedError( 'Refinement tetrahedrons' )  

  @core.classcache
  def getischeme( cls, ndims, where ):
    '''get integration scheme
       http://people.sc.fsu.edu/~jburkardt/datasets/quadrature_rules_tet/quadrature_rules_tet.html'''

    assert ndims == 3
    if where.startswith( 'vtk' ):
      coords = numpy.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]]).T
      weights = None
    elif where == 'gauss1':
      coords = numpy.array( [[1],[1],[1]] ) / 4.
      weights = numpy.array( [1] ) / 6.
    elif where == 'gauss2':
      coords = numpy.array([[0.5854101966249685,0.1381966011250105,0.1381966011250105],
                            [0.1381966011250105,0.1381966011250105,0.1381966011250105],
                            [0.1381966011250105,0.1381966011250105,0.5854101966249685],
                            [0.1381966011250105,0.5854101966249685,0.1381966011250105]]).T
      weights = numpy.array([1,1,1,1]) / 24.
    elif where == 'gauss3':
      coords = numpy.array([[0.2500000000000000,0.2500000000000000,0.2500000000000000],
                            [0.5000000000000000,0.1666666666666667,0.1666666666666667],
                            [0.1666666666666667,0.1666666666666667,0.1666666666666667],
                            [0.1666666666666667,0.1666666666666667,0.5000000000000000],
                            [0.1666666666666667,0.5000000000000000,0.1666666666666667]]).T
      weights = numpy.array([-0.8000000000000000,0.4500000000000000,0.4500000000000000,0.4500000000000000,0.4500000000000000]) / 6.
    elif where == 'gauss4':
      coords = numpy.array([[0.2500000000000000,0.2500000000000000,0.2500000000000000],
                            [0.7857142857142857,0.0714285714285714,0.0714285714285714],
                            [0.0714285714285714,0.0714285714285714,0.0714285714285714],
                            [0.0714285714285714,0.0714285714285714,0.7857142857142857],
                            [0.0714285714285714,0.7857142857142857,0.0714285714285714],
                            [0.1005964238332008,0.3994035761667992,0.3994035761667992],
                            [0.3994035761667992,0.1005964238332008,0.3994035761667992],
                            [0.3994035761667992,0.3994035761667992,0.1005964238332008],
                            [0.3994035761667992,0.1005964238332008,0.1005964238332008],
                            [0.1005964238332008,0.3994035761667992,0.1005964238332008],
                            [0.1005964238332008,0.1005964238332008,0.3994035761667992]]).T
      weights = numpy.array([-0.0789333333333333,0.0457333333333333,0.0457333333333333,0.0457333333333333,0.0457333333333333,0.1493333333333333,0.1493333333333333,0.1493333333333333,0.1493333333333333,0.1493333333333333,0.1493333333333333]) / 6.
    elif where == 'gauss5':
      coords = numpy.array([[0.2500000000000000,0.2500000000000000,0.2500000000000000],
                            [0.0000000000000000,0.3333333333333333,0.3333333333333333],
                            [0.3333333333333333,0.3333333333333333,0.3333333333333333],
                            [0.3333333333333333,0.3333333333333333,0.0000000000000000],
                            [0.3333333333333333,0.0000000000000000,0.3333333333333333],
                            [0.7272727272727273,0.0909090909090909,0.0909090909090909],
                            [0.0909090909090909,0.0909090909090909,0.0909090909090909],
                            [0.0909090909090909,0.0909090909090909,0.7272727272727273],
                            [0.0909090909090909,0.7272727272727273,0.0909090909090909],
                            [0.4334498464263357,0.0665501535736643,0.0665501535736643],
                            [0.0665501535736643,0.4334498464263357,0.0665501535736643],
                            [0.0665501535736643,0.0665501535736643,0.4334498464263357],
                            [0.0665501535736643,0.4334498464263357,0.4334498464263357],
                            [0.4334498464263357,0.0665501535736643,0.4334498464263357],
                            [0.4334498464263357,0.4334498464263357,0.0665501535736643]]).T
      weights = numpy.array([0.1817020685825351,0.0361607142857143,0.0361607142857143,0.0361607142857143,0.0361607142857143,0.0698714945161738,0.0698714945161738,0.0698714945161738,0.0698714945161738,0.0656948493683187,0.0656948493683187,0.0656948493683187,0.0656948493683187,0.0656948493683187,0.0656948493683187]) / 6.
    elif where == 'gauss6':
      coords = numpy.array([[0.3561913862225449,0.2146028712591517,0.2146028712591517],
                            [0.2146028712591517,0.2146028712591517,0.2146028712591517],
                            [0.2146028712591517,0.2146028712591517,0.3561913862225449],
                            [0.2146028712591517,0.3561913862225449,0.2146028712591517],
                            [0.8779781243961660,0.0406739585346113,0.0406739585346113],
                            [0.0406739585346113,0.0406739585346113,0.0406739585346113],
                            [0.0406739585346113,0.0406739585346113,0.8779781243961660],
                            [0.0406739585346113,0.8779781243961660,0.0406739585346113],
                            [0.0329863295731731,0.3223378901422757,0.3223378901422757],
                            [0.3223378901422757,0.3223378901422757,0.3223378901422757],
                            [0.3223378901422757,0.3223378901422757,0.0329863295731731],
                            [0.3223378901422757,0.0329863295731731,0.3223378901422757],
                            [0.2696723314583159,0.0636610018750175,0.0636610018750175],
                            [0.0636610018750175,0.2696723314583159,0.0636610018750175],
                            [0.0636610018750175,0.0636610018750175,0.2696723314583159],
                            [0.6030056647916491,0.0636610018750175,0.0636610018750175],
                            [0.0636610018750175,0.6030056647916491,0.0636610018750175],
                            [0.0636610018750175,0.0636610018750175,0.6030056647916491],
                            [0.0636610018750175,0.2696723314583159,0.6030056647916491],
                            [0.2696723314583159,0.6030056647916491,0.0636610018750175],
                            [0.6030056647916491,0.0636610018750175,0.2696723314583159],
                            [0.0636610018750175,0.6030056647916491,0.2696723314583159],
                            [0.2696723314583159,0.0636610018750175,0.6030056647916491],
                            [0.6030056647916491,0.2696723314583159,0.0636610018750175]]).T
      weights = numpy.array([0.0399227502581679,0.0399227502581679,0.0399227502581679,0.0399227502581679,0.0100772110553207,0.0100772110553207,0.0100772110553207,0.0100772110553207,0.0553571815436544,0.0553571815436544,0.0553571815436544,0.0553571815436544,0.0482142857142857,0.0482142857142857,0.0482142857142857,0.0482142857142857,0.0482142857142857,0.0482142857142857,0.0482142857142857,0.0482142857142857,0.0482142857142857,0.0482142857142857,0.0482142857142857,0.0482142857142857]) / 6.
    elif where == 'gauss7':
      coords = numpy.array([[0.2500000000000000,0.2500000000000000,0.2500000000000000],
                            [0.7653604230090441,0.0782131923303186,0.0782131923303186],
                            [0.0782131923303186,0.0782131923303186,0.0782131923303186],
                            [0.0782131923303186,0.0782131923303186,0.7653604230090441],
                            [0.0782131923303186,0.7653604230090441,0.0782131923303186],
                            [0.6344703500082868,0.1218432166639044,0.1218432166639044],
                            [0.1218432166639044,0.1218432166639044,0.1218432166639044],
                            [0.1218432166639044,0.1218432166639044,0.6344703500082868],
                            [0.1218432166639044,0.6344703500082868,0.1218432166639044],
                            [0.0023825066607383,0.3325391644464206,0.3325391644464206],
                            [0.3325391644464206,0.3325391644464206,0.3325391644464206],
                            [0.3325391644464206,0.3325391644464206,0.0023825066607383],
                            [0.3325391644464206,0.0023825066607383,0.3325391644464206],
                            [0.0000000000000000,0.5000000000000000,0.5000000000000000],
                            [0.5000000000000000,0.0000000000000000,0.5000000000000000],
                            [0.5000000000000000,0.5000000000000000,0.0000000000000000],
                            [0.5000000000000000,0.0000000000000000,0.0000000000000000],
                            [0.0000000000000000,0.5000000000000000,0.0000000000000000],
                            [0.0000000000000000,0.0000000000000000,0.5000000000000000],
                            [0.2000000000000000,0.1000000000000000,0.1000000000000000],
                            [0.1000000000000000,0.2000000000000000,0.1000000000000000],
                            [0.1000000000000000,0.1000000000000000,0.2000000000000000],
                            [0.6000000000000000,0.1000000000000000,0.1000000000000000],
                            [0.1000000000000000,0.6000000000000000,0.1000000000000000],
                            [0.1000000000000000,0.1000000000000000,0.6000000000000000],
                            [0.1000000000000000,0.2000000000000000,0.6000000000000000],
                            [0.2000000000000000,0.6000000000000000,0.1000000000000000],
                            [0.6000000000000000,0.1000000000000000,0.2000000000000000],
                            [0.1000000000000000,0.6000000000000000,0.2000000000000000],
                            [0.2000000000000000,0.1000000000000000,0.6000000000000000],
                            [0.6000000000000000,0.2000000000000000,0.1000000000000000]]).T
      weights = numpy.array([0.1095853407966528,0.0635996491464850,0.0635996491464850,0.0635996491464850,0.0635996491464850,-0.3751064406859797,-0.3751064406859797,-0.3751064406859797,-0.3751064406859797,0.0293485515784412,0.0293485515784412,0.0293485515784412,0.0293485515784412,0.0058201058201058,0.0058201058201058,0.0058201058201058,0.0058201058201058,0.0058201058201058,0.0058201058201058,0.1653439153439105,0.1653439153439105,0.1653439153439105,0.1653439153439105,0.1653439153439105,0.1653439153439105,0.1653439153439105,0.1653439153439105,0.1653439153439105,0.1653439153439105,0.1653439153439105,0.1653439153439105]) / 6.
    elif where == 'gauss8':
      coords = numpy.array([[0.2500000000000000,0.2500000000000000,0.2500000000000000],
                            [0.6175871903000830,0.1274709365666390,0.1274709365666390],
                            [0.1274709365666390,0.1274709365666390,0.1274709365666390],
                            [0.1274709365666390,0.1274709365666390,0.6175871903000830],
                            [0.1274709365666390,0.6175871903000830,0.1274709365666390],
                            [0.9037635088221031,0.0320788303926323,0.0320788303926323],
                            [0.0320788303926323,0.0320788303926323,0.0320788303926323],
                            [0.0320788303926323,0.0320788303926323,0.9037635088221031],
                            [0.0320788303926323,0.9037635088221031,0.0320788303926323],
                            [0.4502229043567190,0.0497770956432810,0.0497770956432810],
                            [0.0497770956432810,0.4502229043567190,0.0497770956432810],
                            [0.0497770956432810,0.0497770956432810,0.4502229043567190],
                            [0.0497770956432810,0.4502229043567190,0.4502229043567190],
                            [0.4502229043567190,0.0497770956432810,0.4502229043567190],
                            [0.4502229043567190,0.4502229043567190,0.0497770956432810],
                            [0.3162695526014501,0.1837304473985499,0.1837304473985499],
                            [0.1837304473985499,0.3162695526014501,0.1837304473985499],
                            [0.1837304473985499,0.1837304473985499,0.3162695526014501],
                            [0.1837304473985499,0.3162695526014501,0.3162695526014501],
                            [0.3162695526014501,0.1837304473985499,0.3162695526014501],
                            [0.3162695526014501,0.3162695526014501,0.1837304473985499],
                            [0.0229177878448171,0.2319010893971509,0.2319010893971509],
                            [0.2319010893971509,0.0229177878448171,0.2319010893971509],
                            [0.2319010893971509,0.2319010893971509,0.0229177878448171],
                            [0.5132800333608811,0.2319010893971509,0.2319010893971509],
                            [0.2319010893971509,0.5132800333608811,0.2319010893971509],
                            [0.2319010893971509,0.2319010893971509,0.5132800333608811],
                            [0.2319010893971509,0.0229177878448171,0.5132800333608811],
                            [0.0229177878448171,0.5132800333608811,0.2319010893971509],
                            [0.5132800333608811,0.2319010893971509,0.0229177878448171],
                            [0.2319010893971509,0.5132800333608811,0.0229177878448171],
                            [0.0229177878448171,0.2319010893971509,0.5132800333608811],
                            [0.5132800333608811,0.0229177878448171,0.2319010893971509],
                            [0.7303134278075384,0.0379700484718286,0.0379700484718286],
                            [0.0379700484718286,0.7303134278075384,0.0379700484718286],
                            [0.0379700484718286,0.0379700484718286,0.7303134278075384],
                            [0.1937464752488044,0.0379700484718286,0.0379700484718286],
                            [0.0379700484718286,0.1937464752488044,0.0379700484718286],
                            [0.0379700484718286,0.0379700484718286,0.1937464752488044],
                            [0.0379700484718286,0.7303134278075384,0.1937464752488044],
                            [0.7303134278075384,0.1937464752488044,0.0379700484718286],
                            [0.1937464752488044,0.0379700484718286,0.7303134278075384],
                            [0.0379700484718286,0.1937464752488044,0.7303134278075384],
                            [0.7303134278075384,0.0379700484718286,0.1937464752488044],
                            [0.1937464752488044,0.7303134278075384,0.0379700484718286]]).T
      weights = numpy.array([-0.2359620398477557,0.0244878963560562,0.0244878963560562,0.0244878963560562,0.0244878963560562,0.0039485206398261,0.0039485206398261,0.0039485206398261,0.0039485206398261,0.0263055529507371,0.0263055529507371,0.0263055529507371,0.0263055529507371,0.0263055529507371,0.0263055529507371,0.0829803830550589,0.0829803830550589,0.0829803830550589,0.0829803830550589,0.0829803830550589,0.0829803830550589,0.0254426245481023,0.0254426245481023,0.0254426245481023,0.0254426245481023,0.0254426245481023,0.0254426245481023,0.0254426245481023,0.0254426245481023,0.0254426245481023,0.0254426245481023,0.0254426245481023,0.0254426245481023,0.0134324384376852,0.0134324384376852,0.0134324384376852,0.0134324384376852,0.0134324384376852,0.0134324384376852,0.0134324384376852,0.0134324384376852,0.0134324384376852,0.0134324384376852,0.0134324384376852,0.0134324384376852]) / 6.
    else:
      raise Exception, 'invalid element evaluation: %r' % where
    return util.ImmutableArray( coords.T ), util.ImmutableArray( weights )

  def select_contained( self, points, eps=0 ):
    'select points contained in element'
    raise NotImplementedError( 'Determine whether a point resides in the tetrahedron' )  

class StdElem( object ):
  'stdelem base class'

  def __mul__( self, other ):
    'multiply elements'

    return PolyProduct( self, other )

  def __pow__( self, n ):
    'repeated multiplication'

    assert n >= 1
    return self if n == 1 else self * self**(n-1)

  def extract( self, extraction ):
    'apply extraction matrix'

    return ExtractionWrapper( self, extraction )

class PolyProduct( StdElem ):
  'multiply standard elements'

  @core.classcache
  def __new__( cls, std1, std2 ):
    'constructor'

    self = object.__new__( cls )
    self.std1 = std1
    self.std2 = std2
    self.ndims = std1.ndims + std2.ndims
    self.nshapes = std1.nshapes * std2.nshapes
    return self

  @core.cachefunc
  def eval( self, points, grad=0 ):
    'evaluate'

    assert isinstance( grad, int ) and grad >= 0

    npoints, ndims = points.shape
    assert ndims == self.ndims

    s1 = slice(0,self.std1.ndims)
    p1 = points[:,s1]
    s2 = slice(self.std1.ndims,None)
    p2 = points[:,s2]

    S = slice(None),
    N = numpy.newaxis,

    G12 = [ numeric.reshape( self.std1.eval( p1, grad=i )[S+S+N+S*i+N*j]
                        * self.std2.eval( p2, grad=j )[S+N+S+N*i+S*j], 1, 2 )
            for i,j in zip( range(grad,-1,-1), range(grad+1) ) ]

    data = numpy.empty( [ npoints, self.std1.nshapes * self.std2.nshapes ] + [ ndims ] * grad )

    s12 = numpy.array([s1,s2])
    R = numpy.arange(grad)
    for n in range(2**grad):
      index = n>>R&1
      data[S*2+tuple(s12[index])] = G12[index.sum()].transpose(0,1,*2+index.argsort())

    return data

  def __str__( self ):
    'string representation'

    return '%s*%s' % ( self.std1, self.std2 )

class PolyLine( StdElem ):
  'polynomial on a line'

  @classmethod
  def bernstein_poly( cls, degree ):
    'bernstein polynomial coefficients'

    # magic bernstein triangle
    n = degree - 1
    poly = numpy.zeros( [n+1,n+1], dtype=int )
    root = (-1)**n
    for k in range(n//2+1):
      poly[k,k] = root
      for i in range(k+1,n+1-k):
        root = poly[i,k] = poly[k,i] = ( root * (k+i-n-1) ) / i
      root = ( poly[k,k+1] * (k*2-n+1) ) / (k+1)
    return poly

  @classmethod
  def spline_poly( cls, p, n ):
    'spline polynomial coefficients'

    assert p >= 1, 'invalid polynomial degree %d' % p
    if p == 1:
      assert n == -1
      return numpy.array( [[[1.]]] )

    assert 1 <= n < 2*(p-1)
    extractions = numpy.empty(( n, p, p ))
    extractions[0] = numpy.eye( p )
    for i in range( 1, n ):
      extractions[i] = numpy.eye( p )
      for j in range( 2, p ):
        for k in reversed( range( j, p ) ):
          alpha = 1. / min( 2+k-j, n-i+1 )
          extractions[i-1,:,k] = alpha * extractions[i-1,:,k] + (1-alpha) * extractions[i-1,:,k-1]
        extractions[i,-j-1:-1,-j-1] = extractions[i-1,-j:,-1]

    poly = cls.bernstein_poly( p )
    return numeric.contract( extractions[:,_,:,:], poly[_,:,_,:], axis=-1 )

  @core.classcache
  def spline_elems( cls, p, n ):
    'spline elements, minimum amount (just for caching)'

    return map( cls, cls.spline_poly(p,n) )

  @core.classcache
  def spline_elems_neumann( cls, p, n ):
    'spline elements, neumann endings (just for caching)'

    polys = cls.spline_poly(p,n)
    poly_0 = polys[0].copy()
    poly_0[:,1] += poly_0[:,0]
    poly_e = polys[-1].copy()
    poly_e[:,-2] += poly_e[:,-1]
    return cls(poly_0), cls(poly_e)

  @core.classcache
  def spline_elems_curvature( cls ):
    'spline elements, curve free endings (just for caching)'

    polys = cls.spline_poly(2,1)
    poly_0 = polys[0].copy()
    poly_0[:,0] += 0.5*(polys[0][:,0]+polys[0][:,1])
    poly_0[:,1] -= 0.5*(polys[0][:,0]+polys[0][:,1])

    poly_e = polys[-1].copy()
    poly_e[:,-2] -= 0.5*(polys[-1][:,-1]+polys[-1][:,-2])
    poly_e[:,-1] += 0.5*(polys[-1][:,-1]+polys[-1][:,-2])

    return cls(poly_0), cls(poly_e)

  @classmethod
  def spline( cls, degree, nelems, periodic=False, neumann=0, curvature=False ):
    'spline elements, any amount'

    p = degree
    n = 2*(p-1)-1
    if periodic:
      assert not neumann, 'periodic domains have no boundary'
      assert not curvature, 'curvature free option not possible for periodic domains'
      elems = cls.spline_elems( p, n )[p-2:p-1] * nelems
    else:
      elems = cls.spline_elems( p, min(nelems,n) )
      if len(elems) < nelems:
        elems = elems[:p-2] + elems[p-2:p-1] * (nelems-2*(p-2)) + elems[p-1:]
      if neumann:
        elem_0, elem_e = cls.spline_elems_neumann( p, min(nelems,n) )
        if neumann & 1:
          elems[0] = elem_0
        if neumann & 2:
          elems[-1] = elem_e
      if curvature:
        assert neumann==0, 'Curvature free not allowed in combindation with Neumann'
        assert degree==3, 'Curvature free only allowed for quadratic splines'  
        elem_0, elem_e = cls.spline_elems_curvature()
        elems[0] = elem_0
        elems[-1] = elem_e

        
    return numpy.array( elems )

  def __init__( self, poly ):
    'constructor'

    self.ndims = 1
    self.poly = numpy.asarray( poly, dtype=float )
    self.degree, self.nshapes = self.poly.shape

  @core.cachefunc
  def eval( self, points, grad=0 ):
    'evaluate'

    if grad >= self.degree:
      return numeric.appendaxes( 0., (points.shape[0],self.nshapes)+(1,)*grad )

    poly = self.poly
    for n in range(grad):
      poly = poly[:-1] * numpy.arange( poly.shape[0]-1, 0, -1 )[:,_]

    x, = points.T
    polyval = poly[0,_,:].repeat( x.size, axis=0 )
    for p in poly[1:]:
      polyval *= x[:,_]
      polyval += p[_,:]

    return polyval[(Ellipsis,)+(_,)*grad]

  def extract( self, extraction ):
    'apply extraction'

    return PolyLine( numpy.dot( self.poly, extraction ) )

  def __repr__( self ):
    'string representation'

    return 'PolyLine#%x' % id(self)

class PolyTriangle( StdElem ):
  'poly triangle'

  @core.classcache
  def __new__( cls, order ):
    'constructor'

    assert order == 1
    self = object.__new__( cls )
    return self

  @core.cachefunc
  def eval( self, points, grad=0 ):
    'eval'

    npoints, ndim = points.shape
    if grad == 0:
      x, y = points.T
      data = numpy.array( [ x, y, 1-x-y ] ).T
    elif grad == 1:
      data = numpy.array( [[[1,0],[0,1],[-1,-1]]], dtype=float )
    else:
      data = numpy.array( 0 ).reshape( (1,) * (grad+1+ndim) )
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

  @core.cachefunc
  def eval( self, points, grad=0 ):
    'call'

    return numeric.dot( self.stdelem.eval( points, grad ), self.extraction, axis=1 )

  def __repr__( self ):
    'string representation'

    return '%s#%x:%s' % ( self.__class__.__name__, id(self), self.stdelem )

# vim:shiftwidth=2:foldmethod=indent:foldnestmax=2
