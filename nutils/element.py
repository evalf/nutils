from . import log, util, numpy, core, numeric, function, _
import warnings

class TrimmedIScheme( object ):
  'integration scheme for truncated elements'

  __slots__ = 'levelset', 'ischeme', 'maxrefine', 'finestscheme', 'bezierscheme', 'retain'

  def __init__( self, levelset, ischeme, maxrefine, finestscheme='uniform1', degree=3, retain=None ):
    'constructor'

    self.levelset = levelset
    self.ischeme = ischeme
    self.maxrefine = maxrefine
    self.finestscheme = finestscheme
    self.bezierscheme = 'bezier%d' % degree
    self.retain = retain

  @core.cache
  def __getitem__( self, elem ):
    'get ischeme for elem'

    ischeme = self.generate_ischeme( elem, self.maxrefine )
    if ischeme is True:
      ischeme = elem.eval( self.ischeme )
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
      inside = numeric.greater( self.levelset( elem, ipoints ), 0 )
      if inside.all():
        return True
      if not inside.any():
        return None
      return ipoints[inside], iweights[inside]

    try:
      inside = numeric.greater( self.levelset( elem, self.bezierscheme ), 0 )
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
      weights.append( iweights * transform.det )

    coords = numeric.concatenate( coords, axis=0 )
    weights = numeric.concatenate( weights, axis=0 )
    return coords, weights

class Transformation( object ):
  'transform points'

  __slots__ = 'fromdim', 'todim'

  def __init__( self, fromdim, todim ):
    'constructor'

    self.fromdim = fromdim
    self.todim = todim

  def __str__( self ):
    'string representation'

    return '%s(%d->%d)' % ( self.__class__.__name__, self.fromdim, self.todim )

  def eval( self, points ):
    'evaluate'

    if points is None:
      return None

    return self._eval( points )

class SliceTransformation( Transformation ):
  'take slice'

  __slots__ = 'slice',

  def __init__( self, fromdim, start=None, stop=None, step=None ):
    'constructor'

    self.slice = slice( start, stop, step )
    Transformation.__init__( self, fromdim, todim=len(range(fromdim)[self.slice]) )
  
  @core.weakcache
  def _eval( self, points ):
    'apply transformation'

    assert points.shape[1] == self.fromdim
    return points[:,self.slice]

class AffineTransformation( Transformation ):
  'affine transformation'

  __slots__ = 'offset', 'transform'

  def __init__( self, offset, transform ):
    'constructor'

    self.offset = numeric.asarray( offset )
    assert self.offset.ndim == 1
    self.transform = numeric.asarray( transform )
    assert self.transform.ndim == 2
    assert self.transform.shape[0] == self.offset.shape[0]
    Transformation.__init__( self, fromdim=self.transform.shape[1], todim=self.transform.shape[0] )

  @property
  def invtrans( self ):
    'inverse transformation'

    assert self.todim == self.fromdim
    return numeric.inv( self.transform )

  @property
  def det( self ):
    'determinant'

    assert self.todim == self.fromdim
    return numeric.det( self.transform )

  def nest( self, other ):
    'merge transformations'

    offset = other.offset + numeric.dot( other.transform, self.offset )
    transform = numeric.dot( other.transform, self.transform )
    return AffineTransformation( offset, transform )

  def get_transform( self ):
    'get transformation copy'

    return self.transform.copy()

  def invapply( self, coords ):
    'apply inverse transformation'

    return numeric.dot( coords - self.offset, self.invtrans.T )

  @core.weakcache
  def _eval( self, points ):
    'apply transformation'

    transpoints = numeric.dot( points, self.transform.T )
    transpoints += self.offset
    transpoints.flags.writeable = False
    return transpoints

def IdentityTransformation( ndims ):
  return AffineTransformation( numeric.zeros(ndims), numeric.eye(ndims) )

PrimaryVertex = str
HalfVertex = lambda vertex1, vertex2, xi=.5: '%s<%.3f>%s' % ( (vertex1,xi,vertex2) if vertex1 < vertex2 else (vertex2,1-xi,vertex1) )
ProductVertex = lambda *vertices: ','.join( vertices )

class Element( object ):
  '''Element base class.

  Represents the topological shape.'''

  __slots__ = 'vertices', 'ndims', 'index', 'parent', 'context', 'interface', 'root_transform', 'inv_root_transform', 'root_det'

  def __init__( self, ndims, vertices, index=None, parent=None, context=None, interface=None ):
    'constructor'

    #assert all( isinstance(vertex,Vertex) for vertex in vertices )
    self.vertices = tuple(vertices)
    self.ndims = ndims
    assert index is None or parent is None
    self.index = index
    self.parent = parent
    self.context = context
    self.interface = interface

    if parent:
      pelem, trans = parent
      self.root_transform = numeric.dot( pelem.root_transform, trans.transform )
      self.inv_root_transform = numeric.dot( trans.invtrans, pelem.inv_root_transform )
      self.root_det = pelem.root_det * trans.det
    else:
      self.inv_root_transform = self.root_transform = numeric.eye( self.ndims )
      self.root_det = 1.

  def __mul__( self, other ):
    'multiply elements'

    return ProductElement( self, other )

  def neighbor( self, other ):
    'level of neighborhood; 0=self'

    if self == other:
      return 0
    ncommon = len( set(self.vertices) & set(other.vertices) )
    return self.neighbormap[ ncommon ]

  def eval( self, where ):
    'get points'

    if isinstance( where, str ):
      points, weights = self.getischeme( self.ndims, where )
    else:
      points = where
      weights = None
    return points, weights

  def zoom( self, elemset, points ):
    'zoom points'

    elem = self
    totaltransform = 1
    while elem not in elemset:
      elem, transform = self.parent
      points = transform( points )
      totaltransform = numeric.dot( transform.transform, totaltransform )
    return elem, points, totaltransform

  def __str__( self ):
    'string representation'

    return '%s(%s)' % ( self.__class__.__name__, self.vertices )

  def __hash__( self ):
    'hash'

    return hash(str(self))

  def __eq__( self, other ):
    'hash'

    return self is other or ( self.__class__ == other.__class__
                          and self.vertices == other.vertices
                          and self.parent == other.parent
                          and self.context == other.context
                          and self.interface == other.interface )

  def intersected( self, levelset, lscheme, evalrefine=0 ):
    '''check levelset intersection:
      +1 for levelset > 0 everywhere
      -1 for levelset < 0 everywhere
       0 for intersected element'''

    elems = iter( [self] )
    for irefine in range(evalrefine):
      elems = ( child for elem in elems for child in elem.children )

    inside = numeric.greater( levelset( elems.next(), lscheme ), 0 )
    if inside.all():
      for elem in elems:
        inside = numeric.greater( levelset( elem, lscheme ), 0 )
        if not inside.all():
          return 0
      return 1
    elif not inside.any():
      for elem in elems:
        inside = numeric.greater( levelset( elem, lscheme ), 0 )
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

    parent = self, AffineTransformation( numeric.zeros(self.ndims), numeric.eye(self.ndims) )
    return TrimmedElement( elem=self, vertices=self.vertices, levelset=levelset, maxrefine=maxrefine, lscheme=lscheme, finestscheme=finestscheme, evalrefine=evalrefine, parent=parent )

  def get_simplices ( self, maxrefine ):
    'divide in simple elements'

    return self,

  def get_trimmededges ( self, maxrefine ):
    return []

class ProductElement( Element ):
  'element product'

  __slots__ = 'elem1', 'elem2', 'root_det'

  @staticmethod
  @core.cache
  def getslicetransforms( ndims1, ndims2 ):
    ndims = ndims1 + ndims2
    slice1 = SliceTransformation( fromdim=ndims, stop=ndims1 )
    slice2 = SliceTransformation( fromdim=ndims, start=ndims1 )
    return slice1, slice2

  def __init__( self, elem1, elem2 ):
    'constructor'

    self.elem1 = elem1
    self.elem2 = elem2
    slice1, slice2 = self.getslicetransforms( elem1.ndims, elem2.ndims )
    iface1 = elem1, slice1
    iface2 = elem2, slice2
    vertices = [] # TODO [ ProductVertex(vertex1,vertex2) for vertex1 in elem1.vertices for vertex2 in elem2.vertices ]
    Element.__init__( self, ndims=elem1.ndims+elem2.ndims, vertices=vertices, interface=(iface1,iface2) )

    self.root_det = elem1.root_det * elem2.root_det # HACK. TODO via constructor

  @staticmethod
  @core.cache
  def get_tri_bem_ischeme( ischeme, neighborhood ):
    'Some cached quantities for the singularity quadrature scheme.'
    points, weights = QuadElement.getischeme( ndims=4, where=ischeme )
    eta1, eta2, eta3, xi = points.T
    if neighborhood == 0:
      temp = xi*eta1*eta2*eta3
      pts0 = xi*eta1*(1 - eta2)
      pts1 = xi - pts0
      pts2 = xi - temp
      pts3 = xi*(1 - eta1)
      pts4 = pts0 + temp
      pts5 = xi*(1 - eta1*eta2)
      pts6 = xi*eta1 - temp
      points = numeric.asarray(
        [[1-xi,   1-pts2, 1-xi,   1-pts5, 1-pts2, 1-xi  ],
         [pts1, pts3, pts4, pts0, pts6, pts0],
         [1-pts2, 1-xi,   1-pts5, 1-xi,   1-xi,   1-pts2],
         [pts3, pts1, pts0, pts4, pts0, pts6]]).reshape( 4, -1 ).T
      points = numeric.asarray( points * [-1,1,-1,1] + [1,0,1,0] ) # flipping in x -GJ
      weights = numeric.concatenate( 6*[xi**3*eta1**2*eta2*weights] )
    elif neighborhood == 1:
      A = xi*eta1
      B = A*eta2
      C = A*eta3
      D = B*eta3
      E = xi - B
      F = A - B
      G = xi - D
      H = B - D
      I = A - D
      points = numeric.asarray(
        [[1-xi, 1-xi, 1-E,  1-G,  1-G ],
         [C,  G,  F,  H,  I ],
         [1-E,  1-G,  1-xi, 1-xi, 1-xi],
         [F,  H,  D,  A,  B ]] ).reshape( 4, -1 ).T
      temp = xi*A
      weights = numeric.concatenate( [A*temp*weights] + 4*[B*temp*weights] )
    elif neighborhood == 2:
      A = xi*eta2
      B = A*eta3
      C = xi*eta1
      points = numeric.asarray(
        [[1-xi, 1-A ],
         [C,  B ],
         [1-A,  1-xi],
         [B,  C ]] ).reshape( 4, -1 ).T
      weights = numeric.concatenate( 2*[xi**2*A*weights] )
    else:
      assert neighborhood == -1, 'invalid neighborhood %r' % neighborhood
      points = numeric.asarray([ eta1*eta2, 1-eta2, eta3*xi, 1-xi ]).T
      weights = eta2*xi*weights
    return points, weights
  
  @staticmethod
  @core.cache
  def get_quad_bem_ischeme( ischeme, neighborhood ):
    'Some cached quantities for the singularity quadrature scheme.'
    points, weights = QuadElement.getischeme( ndims=4, where=ischeme )
    eta1, eta2, eta3, xi = points.T
    if neighborhood == 0:
      xe = xi*eta1
      A = (1 - xi)*eta3
      B = (1 - xe)*eta2
      C = xi + A
      D = xe + B
      points = numeric.asarray(
        [[A, B, A, D, B, C, C, D],
         [B, A, D, A, C, B, D, C],
         [C, D, C, B, D, A, A, B],
         [D, C, B, C, A, D, B, A]]).reshape( 4, -1 ).T
      weights = numeric.concatenate( 8*[xi*(1-xi)*(1-xe)*weights] )
    elif neighborhood == 1:
      ox = 1 - xi
      A = xi*eta1
      B = xi*eta2
      C = ox*eta3
      D = C + xi
      E = 1 - A
      F = E*eta3
      G = A + F
      points = numeric.asarray(
        [[D,  C,  G,  G,  F,  F ],
         [B,  B,  B,  xi, B,  xi],
         [C,  D,  F,  F,  G,  G ],
         [A,  A,  xi, B,  xi, B ]]).reshape( 4, -1 ).T
      weights = numeric.concatenate( 2*[xi**2*ox*weights] + 4*[xi**2*E*weights] )
    elif neighborhood == 2:
      A = xi*eta1
      B = xi*eta2
      C = xi*eta3
      points = numeric.asarray(
        [[xi, A,  A,  A ], 
         [A,  xi, B,  B ],
         [B,  B,  xi, C ], 
         [C,  C,  C,  xi]]).reshape( 4, -1 ).T
      weights = numeric.concatenate( 4*[xi**3*weights] )
    else:
      assert neighborhood == -1, 'invalid neighborhood %r' % neighborhood
    return points, weights

  @staticmethod
  @core.cache
  def concat( ischeme1, ischeme2 ):
    coords1, weights1 = ischeme1
    coords2, weights2 = ischeme2
    if weights1 is not None:
      assert weights2 is not None
      weights = numeric.asarray( ( weights1[:,_] * weights2[_,:] ).ravel() )
    else:
      assert weights2 is None
      weights = None
    npoints1,ndims1 = coords1.shape  
    npoints2,ndims2 = coords2.shape 
    coords = numeric.empty( [ coords1.shape[0], coords2.shape[0], ndims1+ndims2 ] )
    coords[:,:,:ndims1] = coords1[:,_,:]
    coords[:,:,ndims1:] = coords2[_,:,:]
    coords = numeric.asarray( coords.reshape(-1,ndims1+ndims2) )
    return coords, weights
  
  @property
  def orientation( self ):
    '''Neighborhood of elem1 and elem2 and transformations to get mutual overlap in right location
    O: neighborhood,  as given by Element.neighbor(),
       transf1,       required rotation of elem1 map: {0:0, 1:pi/2, 2:pi, 3:3*pi/2},
       transf2,       required rotation of elem2 map (is indep of transf1 in UnstructuredTopology.'''
    neighborhood = self.elem1.neighbor( self.elem2 )
    common_vertices = list( set(self.elem1.vertices) & set(self.elem2.vertices) )
    vertices1 = [self.elem1.vertices.index( ni ) for ni in common_vertices]
    vertices2 = [self.elem2.vertices.index( ni ) for ni in common_vertices]
    if neighborhood == 0:
      # test for strange topological features
      assert self.elem1 == self.elem2, 'Topological feature not supported: try refining here, possibly periodicity causes elems to touch on both sides.'
      transf1 = transf2 = 0
    elif neighborhood == -1:
      transf1 = transf2 = 0
    elif isinstance( self.elem1, QuadElement ):
      # define local map rotations
      if neighborhood==1:
        transform = [0,2], [2,3], [3,1], [1,0], [2,0], [3,2], [1,3], [0,1]
      elif neighborhood==2:
        transform = [0], [2], [3], [1]
      else:
        raise ValueError( 'Unknown neighbor type %i' % neighborhood )
      transf1 = transform.index( vertices1 )
      transf2 = transform.index( vertices2 )
    elif isinstance( self.elem1, TriangularElement ):
      raise NotImplementedError( 'Pending completed implementation and verification.' )
      # define local map rotations
      if neighborhood==1:
        transform = [0,1], [1,2], [0,2]
      elif neighborhood==2:
        transform = [0], [1], [2]
      else:
        raise ValueError( 'Unknown neighbor type %i' % neighborhood )
      transf1 = transform.index( vertices1 )
      transf2 = transform.index( vertices2 )
    else:
      raise NotImplementedError( 'Reorientation not implemented for element of class %s' % type(self.elem1) )
    return neighborhood, transf1, transf2

  @staticmethod
  @core.cache
  def singular_ischeme_tri( orientation, ischeme ):
    neighborhood, transf1, transf2 = orientation
    points, weights = ProductElement.get_tri_bem_ischeme( ischeme, neighborhood )
    transfpoints = points#numeric.empty( points.shape )
    #   transfpoints[:,0] = points[:,0] if transf1 == 0 else \
    #                       points[:,1] if transf1 == 1 else \
    #                     1-points[:,0] if transf1 == 2 else \
    #                     1-points[:,1]
    #   transfpoints[:,1] = points[:,1] if transf1 == 0 else \
    #                     1-points[:,0] if transf1 == 1 else \
    #                     1-points[:,1] if transf1 == 2 else \
    #                       points[:,0]
    #   transfpoints[:,2] = points[:,2] if transf2 == 0 else \
    #                       points[:,3] if transf2 == 1 else \
    #                     1-points[:,2] if transf2 == 2 else \
    #                     1-points[:,3]
    #   transfpoints[:,3] = points[:,3] if transf2 == 0 else \
    #                     1-points[:,2] if transf2 == 1 else \
    #                     1-points[:,3] if transf2 == 2 else \
    #                       points[:,2]
    return numeric.asarray( transfpoints ), numeric.asarray( weights )
    
  @staticmethod
  @core.cache
  def singular_ischeme_quad( orientation, ischeme ):
    neighborhood, transf1, transf2 = orientation
    points, weights = ProductElement.get_quad_bem_ischeme( ischeme, neighborhood )
    transfpoints = numeric.empty( points.shape )
    def transform( points, orientation ):
      x, y = points[:,0], points[:,1]
      tx = x if orientation in (0,1,6,7) else 1-x
      ty = y if orientation in (0,3,4,7) else 1-y
      return function.stack( (ty, tx) if orientation%2 else (tx, ty), axis=1 )
    transfpoints[:,:2] = transform( points[:,:2], transf1 )
    transfpoints[:,2:] = transform( points[:,2:], transf2 )
    return numeric.asarray( transfpoints ), numeric.asarray( weights )
    
  def eval( self, where ):
    'get integration scheme'
    
    if where.startswith( 'singular' ):
      assert type(self.elem1) == type(self.elem2), 'mixed element-types case not implemented'
      assert self.elem1.ndims == 2 and self.elem2.ndims == 2, 'singular quadrature only for bivariate surfaces'
      gauss = 'gauss%d'% (int(where[8:])*2-2)
      if isinstance( self.elem1, QuadElement ):
        xw = ProductElement.singular_ischeme_quad( self.orientation, gauss )
      elif isinstance( self.elem1, TriangularElement ):
        if self.elem1 == self.elem2:
          xw = self.get_tri_bem_ischeme( gauss, neighborhood=0 )
        else:
          xw = self.concat( self.elem1.eval(gauss), self.elem2.eval(gauss) )
      else:
        raise Exception, 'invalid element type %r' % type(self.elem1)
    else:
      where1, where2 = where.split( '*' ) if '*' in where else ( where, where )
      xw = self.concat( self.elem1.eval(where1), self.elem2.eval(where2) )
    return xw

class TrimmedElement( Element ):
  'trimmed element'

  __slots__ = 'elem', 'levelset', 'maxrefine', 'lscheme', 'finestscheme', 'evalrefine'

  def __init__( self, elem, levelset, maxrefine, lscheme, finestscheme, evalrefine, parent, vertices ):
    'constructor'

    assert not isinstance( elem, TrimmedElement )
    self.elem = elem
    self.levelset = levelset
    self.maxrefine = maxrefine
    self.lscheme = lscheme
    self.finestscheme = finestscheme if finestscheme != None else 'simplex1'
    self.evalrefine = evalrefine

    Element.__init__( self, ndims=elem.ndims, vertices=vertices, parent=parent )

  @core.cache
  def eval( self, ischeme ):
    'get integration scheme'

    assert isinstance( ischeme, str )

    if ischeme[:7] == 'contour':
      n = int(ischeme[7:] or 0)
      points, weights = self.elem.eval( 'contour{}'.format(n) )
      inside = numeric.greater_equal( self.levelset( self.elem, points ), 0 )
      return points[inside], None

    if self.maxrefine <= 0:
      if self.finestscheme.startswith('simplex'):

        points  = []
        weights = []

        for simplex in self.get_simplices( 0 ):

          spoints, sweights = simplex.eval( ischeme )
          pelem, transform = simplex.parent

          assert pelem is self 

          points.append( transform.eval( spoints ) )
          weights.append( sweights * transform.det )

        if len(points) == 0:
          return numeric.zeros((0,self.ndims)), numeric.zeros((0,))

        points  = numeric.concatenate( points, axis=0 )
        weights = numeric.concatenate( weights )

        return points, weights

      else:
        
        if self.finestscheme.endswith( '.all' ):
          points, weights = self.elem.eval( self.finestscheme[:-4] )
        elif self.finestscheme.endswith( '.none' ):
          points, weights = self.elem.eval( self.finestscheme[:-5] )
          return points[numeric.zeros_like(weights,dtype=bool)], weights[numeric.zeros_like(weights,dtype=bool)] if weights is not None else None
        else:  
          points, weights = self.elem.eval( self.finestscheme )
          inside = numeric.greater( self.levelset( self.elem, points ), 0 )
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

    coords = numeric.concatenate( allcoords, axis=0 )
    weights = numeric.concatenate( allweights, axis=0 )
    return coords, weights

  @property
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
        child = QuadElement( vertices=child.vertices, ndims=self.ndims, parent=parent )
      else:
        child = TrimmedElement( vertices=child.vertices, elem=child, levelset=self.levelset, maxrefine=self.maxrefine-1, lscheme=self.lscheme, finestscheme=self.finestscheme, evalrefine=self.evalrefine-1, parent=parent )
      children.append( child )
    return tuple( children )

  def edge( self, iedge ):
    'edge'

    # TODO fix trimming of edges once refine/edge operations commute
    edge = self.elem.edge( iedge )
    pelem, transform = edge.context

    # transform = self.elem.edgetransform( self.ndims )[ iedge ]
    return QuadElement( vertices=edge.vertices, ndims=self.ndims-1, context=(self,transform) )

  def get_simplices ( self, maxrefine ):
    'divide in simple elements'

    if maxrefine > 0 or self.evalrefine > 0:
      return [ simplex for child in filter(None,self.children) for simplex in child.get_simplices( maxrefine=maxrefine-1 ) ]

    simplices, trimmededges = self.triangulate()

    return simplices

  def get_trimmededges ( self, maxrefine ):

    if maxrefine > 0 or self.evalrefine > 0:
      return [ trimmededge for child in filter(None,self.children) for trimmededge in child.get_trimmededges( maxrefine=maxrefine-1 ) ]

    simplices, trimmededges = self.triangulate()

    return trimmededges

  #@core.cache
  def triangulate ( self ):

    assert self.finestscheme.startswith('simplex'), 'Expected simplex scheme'
    order = int(self.finestscheme[7:])

    lvltol  = numeric.spacing(100)
    xistol  = numeric.spacing(100)
    ischeme = self.elem.getischeme( self.elem.ndims, 'bezier2' )
    where   = numeric.greater( self.levelset( self.elem, ischeme ), -lvltol )
    points  = ischeme[0][where]
    vertices   = numeric.array(self.vertices)[where].tolist()
    norig   = sum(where)

    if not where.any():
      return []

    if where.all():
	    lines = []
    else:		
    	lines = self.elem.ribbons

    for line in lines:
      
      ischeme = line.getischeme( line.ndims, 'bezier'+str(order+1) )
      vals    = self.levelset( line, ischeme )
      pts     = ischeme[0]
      where   = numeric.greater( vals, -lvltol )
      zeros   = numeric.less(vals, lvltol) & where

      if order == 1:

        if where[0] == where[1] or zeros.any():
          continue
        xi = vals[0]/(vals[0]-vals[1])

      elif order == 2:

        #Check whether the levelset is linear
        if abs(2*vals[1]-vals[0]-vals[2]) < lvltol:

          if where[0] == where[2] or zeros[0] or zeros[2]:
            continue
          xi = vals[0]/(vals[0]-vals[2])
          
        else:

          disc = vals[0]**2+(-4*vals[1]+vals[2])**2-2*vals[0]*(4*vals[1]+vals[2])

          if disc < -lvltol or abs(disc) < lvltol:
            #No intersections or minimum at zero
            continue
          else:
            #Two intersections
            num2 = numeric.sqrt( disc )
            num1 = 3*vals[0]-4*vals[1]+vals[2]
            denr = 1./(4*(vals[0]-2*vals[1]+vals[2]))

            xis = [(num1-num2)*denr,\
                   (num1+num2)*denr ]

            intersects = [(xi > xistol and xi < 1-xistol) for xi in xis]

            if sum(intersects) == 0:
              continue
            elif sum(intersects) == 1:
              xi = xis[intersects[0] == False]
            else:
              raise Exception('Found multiple ribbon intersections. MAXREFINE should be increased.')

      else:
        #TODO General order scheme based on bisection
        raise NotImplementedError('Simplex generation only implemented for order 1 and 2')

      assert ( xi > xistol and xi < 1.-xistol ), 'Illegal local coordinate'
 
      elem, transform = line.context

      pts = transform.eval( pts )

      newpoint = pts[0] + xi * ( pts[-1] - pts[0] )

      points   = numpy.append( points, newpoint[_], axis=0 ) 
      v1, v2 = line.vertices
      vertices.append( HalfVertex( v1, v2, xi=xi ) )

    try:
      submesh = util.delaunay( points )
    except RuntimeError:
      return [], []

    Simplex = TriangularElement if self.ndims == 2 else TetrahedronElement

    convex_hull = [ [vertices[iv] for iv in tri] for tri in submesh.convex_hull if numeric.greater_equal(tri,norig).all() ]

    ##########################################
    # Extract the simplices from the submesh #
    ##########################################

    simplices = []
    degensim  = []
    for tri in submesh.vertices:

      for j in range(2): #Flip two points in case of negative determinant
        if self.ndims == 3:
          # TODO renumber tetrahedron vertices to match triangle, removes the following:
          offset = points[ tri[0] ]
          affine = numeric.array( [ points[ tri[ii+1] ] - offset for ii in range(self.ndims) ] ).T
        else:
          offset = points[ tri[-1] ]
          affine = numeric.array( [ points[ tri[ii] ] - offset for ii in range(self.ndims) ] ).T

        transform = AffineTransformation( offset, affine )

        if transform.det > numeric.spacing(100):
          break

        tri[-2:] = tri[:-3:-1]

      else:
        if abs(transform.det) < numeric.spacing(100):
          degensim.append( [ vertices[ii] for ii in tri ] )
          continue

        raise Exception('Negative determinant with value %12.10e could not be resolved by flipping two vertices' % transform.det )

      simplices.append( Simplex( vertices=[ vertices[ii] for ii in tri ], parent=(self,transform) ) )

    assert len(simplices)+len(degensim)==submesh.vertices.shape[0], 'Simplices should be stored in either of the two containers'

    #############################################################
    # Loop over the edges of the simplex and check whether they #
    # reside in the part of the convex hull on the levelset     #
    #############################################################
      
    trimmededges = []  
              
    import itertools          
    for simplex in simplices:
      for iedge in range(self.ndims+1):

        #The edge potentially to be added to the trimmededges
        sedge = simplex.edge(iedge) 

        #Create lists to store edges which are to be checked on residence in the
        #convex hull, or which have been checked
        checkedges = [ sedge.vertices ]
        visitedges = []

        while checkedges:
          #Edge to be check on residence in the convex hull
          checkedge = checkedges.pop(0)
          visitedges.append( checkedge )

          #Check whether this edge is in the convex hull
          for hull_edge in convex_hull:
            #The checkedge is found in the convex hull. Append trimmededge and
            #terminate loop
            if all(checkvertex in hull_edge for checkvertex in checkedge):
              trimmededges.append( sedge )
              checkedges = []
              break
          else:
            #Check whether the checkedge is in a degenerate simplex
            for sim in degensim:
              if all(checkvertex in sim for checkvertex in checkedge):
                #Append all the edges to the checkedges pool
                for jedge in itertools.combinations(sim,self.ndims):
                  dedge = list(jedge)
                  for cedge in visitedges:
                    #The dedge is already in visitedges
                    if all(dvertex in cedge for dvertex in dedge):
                      break
                  else:
                    #The dedge is appended to to pool
                    checkedges.append( dedge )


    return simplices, trimmededges
  
class QuadElement( Element ):
  'quadrilateral element'

  __slots__ = ()

  def __init__( self, ndims, vertices, index=None, parent=None, context=None, interface=None ):
    'constructor'

    assert len(vertices) == 2**ndims
    Element.__init__( self, ndims, vertices, index=index, parent=parent, context=context, interface=interface )

  @property
  @core.cache
  def neighbormap( self ):
    'maps # matching vertices --> codim of interface: {0: -1, 1: 2, 2: 1, 4: 0}'
    return dict( [ (0,-1) ] + [ (2**(self.ndims-i),i) for i in range(self.ndims+1) ] )

  def children_by( self, N ):
    'divide element by n'

    assert len(N) == self.ndims
    vertices = numeric.empty( [ ni+1 for ni in N ], dtype=object )
    vertices[ tuple( slice(None,None,ni) for ni in N ) ] = numpy.reshape( self.vertices, [2]*self.ndims )
    for idim in range(self.ndims):
      s1 = tuple( slice(None) for ni in N[:idim] )
      s2 = tuple( slice(None,None,ni) for ni in N[idim+1:] )
      for i in range( 1, N[idim] ):
        vertices[s1+(i,)+s2] = numeric.objmap( HalfVertex, vertices[s1+(0,)+s2], vertices[s1+(2,)+s2], float(i)/N[idim] )

    elemvertices = [ vertices[ tuple( slice(i,i+2) for i in index ) ].ravel() for index in numpy.ndindex(*N) ]
    return tuple( QuadElement( vertices=elemvertices[ielem], ndims=self.ndims, parent=(self,transform) )
      for ielem, transform in enumerate( self.refinedtransform(N) ) )

  @property
  def children( self ):
    'all 1x refined elements'

    return self.children_by( (2,)*self.ndims )

  @staticmethod
  @core.cache
  def edgetransform( ndims ):
    'edge transforms'

    transforms = []
    for idim in range( ndims ):
#     for iside in range( 2 ):
#       offset = numeric.zeros( ndims )
#       offset[idim:] = 1-iside
#       offset[:idim+1] = iside
#       transform = numeric.zeros(( ndims, ndims-1 ))
#       transform.flat[ :(ndims-1)*idim :ndims] = 1 - 2 * iside
#       transform.flat[ndims*(idim+1)-1::ndims] = 2 * iside - 1
#       transforms.append( AffineTransformation( offset=offset, transform=transform ) )
      offset = numeric.zeros( ndims )
      offset[:idim] = 1
      transform = numeric.zeros(( ndims, ndims-1 ))
      transform.flat[ :(ndims-1)*idim :ndims] = -1
      transform.flat[ndims*(idim+1)-1::ndims] = 1
      # flip normal:
      offset[idim-1] = 1 - offset[idim-1]
      transform[idim-1] *= -1
      transforms.append( AffineTransformation( offset=offset.copy(), transform=transform.copy() ) )
      # other side:
      offset[idim] = 1
      # flip normal back:
      offset[idim-1] = 1 - offset[idim-1]
      transform[idim-1] *= -1
      transforms.append( AffineTransformation( offset=offset, transform=transform ) )
    return transforms

  @property
  def ribbons( self ):
    'ribbons'

    if self.ndims == 2:
      return [ self.edge(iedge) for iedge in range(4) ]

    if self.ndims != 3:
      raise NotImplementedError('Ribbons not implemented for ndims=%d'%self.ndims)

    ndvertices = numpy.reshape( self.vertices, [2]*self.ndims )
    ribbons = []
    for i1, i2 in numeric.array([[[0,0,0],[1,0,0]],
                                 [[0,0,0],[0,1,0]],
                                 [[0,0,0],[0,0,1]],
                                 [[1,1,1],[0,1,1]],
                                 [[1,1,1],[1,0,1]],
                                 [[1,1,1],[1,1,0]],
                                 [[1,0,0],[1,1,0]],
                                 [[1,0,0],[1,0,1]],
                                 [[0,1,0],[1,1,0]],
                                 [[0,1,0],[0,1,1]],
                                 [[0,0,1],[1,0,1]],
                                 [[0,0,1],[0,1,1]]] ):
      transform = AffineTransformation( offset=i1, transform=(i2-i1)[:,_] )
      vertices = ndvertices[tuple(i1)], ndvertices[tuple(i2)]
      ribbons.append( QuadElement( vertices=vertices, ndims=1, context=(self,transform) ) )

    return ribbons

  @property
  def edges( self ):
    return [ self.edge(iedge) for iedge in range(2**self.ndims) ]

  def edge( self, iedge ):
    'edge'
    transform = self.edgetransform( self.ndims )[ iedge ]
    idim = iedge // 2
    iside = iedge % 2
    s = (slice(None,None, 1 if iside else -1),) * idim + (iside,) \
      + (slice(None,None,-1 if iside else  1),) * (self.ndims-idim-1)
    vertices = numeric.asarray( numpy.reshape( self.vertices, (2,)*self.ndims )[s] ).ravel() # TODO check
    return QuadElement( vertices=vertices, ndims=self.ndims-1, context=(self,transform) )

  @staticmethod
  @core.cache
  def refinedtransform( N ):
    'refined transform'

    Nf = numeric.asarray( N, dtype=float )
    assert Nf.ndim == 1
    return [ AffineTransformation( offset=I/Nf, transform=numeric.diag(1/Nf) ) for I in numpy.ndindex(*N) ]

  def refine( self, n ):
    'refine n times'

    elems = [ self ]
    for i in range(n):
      elems = [ child for elem in elems for child in elem.children ]
    return elems

  @staticmethod
  @core.cache
  def getgauss( degree ):
    'compute gauss points and weights'

    assert isinstance( degree, int ) and degree >= 0
    k = numeric.arange( 1, degree // 2 + 1 )
    d = k / numeric.sqrt( 4*k**2-1 )
    x, w = numeric.eigh( numeric.diagflat(d,-1) ) # eigh operates (by default) on lower triangle
    return (x+1) * .5, w[0]**2

  @classmethod
  @core.cache
  def getischeme( cls, ndims, where ):
    'get integration scheme'

    x = w = None
    if ndims == 0:
      coords = numeric.zeros([1,0])
      weights = numeric.array([1.])
    elif where.startswith( 'gauss' ):
      N = eval( where[5:] )
      if isinstance( N, tuple ):
        assert len(N) == ndims
      else:
        N = [N]*ndims
      x, w = zip( *map( cls.getgauss, N ) )
    elif where.startswith( 'uniform' ):
      N = eval( where[7:] )
      if isinstance( N, tuple ):
        assert len(N) == ndims
      else:
        N = [N]*ndims
      x = [ numeric.arange( .5, n ) / n for n in N ]
      w = [ numeric.appendaxes( 1./n, n ) for n in N ]
    elif where.startswith( 'bezier' ):
      N = int( where[6:] )
      x = [ numeric.linspace( 0, 1, N ) ] * ndims
      w = [ numeric.appendaxes( 1./N, N ) ] * ndims
    elif where.startswith( 'subdivision' ):
      N = int( where[11:] ) + 1
      x = [ numeric.linspace( 0, 1, N ) ] * ndims
      w = None
    elif where.startswith( 'vtk' ):
      if ndims == 1:
        coords = numeric.array([[0,1]]).T
      elif ndims == 2:
        eps = 0 if not len(where[3:]) else float(where[3:]) # subdivision fix (avoid extraordinary point)
        coords = numeric.array([[eps,eps],[1-eps,eps],[1-eps,1-eps],[eps,1-eps]])
      elif ndims == 3:
        coords = numeric.array([ [0,0,0], [1,0,0], [0,1,0], [1,1,0], [0,0,1], [1,0,1], [0,1,1], [1,1,1] ])
      else:
        raise Exception, 'contour not supported for ndims=%d' % ndims
    elif where.startswith( 'contour' ):
      N = int( where[7:] )
      p = numeric.linspace( 0, 1, N )
      if ndims == 1:
        coords = p[_].T
      elif ndims == 2:
        coords = numeric.array([ p[ range(N) + [N-1]*(N-2) + range(N)[::-1] + [0]*(N-1) ],
                                 p[ [0]*(N-1) + range(N) + [N-1]*(N-2) + range(0,N)[::-1] ] ]).T
      elif ndims == 3:
        assert N == 0
        coords = numeric.array([ [0,0,0], [1,0,0], [0,1,0], [1,1,0], [0,0,1], [1,0,1], [0,1,1], [1,1,1] ])
      else:
        raise Exception, 'contour not supported for ndims=%d' % ndims
    else:
      raise Exception, 'invalid element evaluation %r' % where
    if x is not None:
      coords = numeric.empty( map( len, x ) + [ ndims ] )
      for i, xi in enumerate( x ):
        coords[...,i] = xi[ (slice(None),) + (_,)*(ndims-i-1) ]
      coords = coords.reshape( -1, ndims )
    if w is not None:
      weights = reduce( lambda weights, wi: ( weights * wi[:,_] ).ravel(), w )
    else:
      weights = None
    coords.flags.writeable = False
    if weights is not None:
      weights.flags.writeable = False
    return coords, weights

  def select_contained( self, points, eps=0 ):
    'select points contained in element'

    selection = numeric.ones( points.shape[0], dtype=bool )
    for idim in range( self.ndims ):
      newsel = numeric.greater_equal( points[:,idim], -eps ) & numeric.less_equal( points[:,idim], 1+eps )
      selection[selection] &= newsel
      points = points[newsel]
      if not points.size:
        return None, None
    return points, selection

class TriangularElement( Element ):
  '''triangular element
     conventions: reference elem:   unit simplex {(x,y) | x>0, y>0, x+y<1}
                  vertex numbering: {(1,0):0, (0,1):1, (0,0):2}
                  edge numbering:   {bottom:0, slanted:1, left:2}
                  edge local coords run counter-clockwise.'''

  __slots__ = ()

  neighbormap = -1, 2, 1, 0
  edgetransform = (
    AffineTransformation( offset=[0,0], transform=[[ 1],[ 0]] ),
    AffineTransformation( offset=[1,0], transform=[[-1],[ 1]] ),
    AffineTransformation( offset=[0,1], transform=[[ 0],[-1]] ) )

  def __init__( self, vertices, index=None, parent=None, context=None ):
    'constructor'

    assert len(vertices) == 3
    Element.__init__( self, ndims=2, vertices=vertices, index=index, parent=parent, context=context )

  @property
  def children( self ):
    'all 1x refined elements'

    t1, t2, t3, t4 = self.refinedtransform( 2 )
    v1, v2, v3 = self.vertices
    h1, h2, h3 = HalfVertex(v1,v2), HalfVertex(v2,v3), HalfVertex(v3,v1)
    return tuple([ # TODO check!
      TriangularElement( vertices=[v1,h1,h3], parent=(self,t1) ),
      TriangularElement( vertices=[h1,v2,h2], parent=(self,t2) ),
      TriangularElement( vertices=[h3,h2,v3], parent=(self,t3) ),
      TriangularElement( vertices=[h2,h3,h1], parent=(self,t4) ) ])
      
  @property
  def edges( self ):
    return [ self.edge(iedge) for iedge in range(3) ]

  def edge( self, iedge ):
    'edge'

    transform = self.edgetransform[ iedge ]
    vertices = [ self.vertices[::-2], self.vertices[:2], self.vertices[1:] ][iedge]
    return QuadElement( vertices=vertices, ndims=1, context=(self,transform) )

  @staticmethod
  @core.cache
  def refinedtransform( n ):
    'refined transform'

    transforms = []
    trans = numeric.diag( [1./n]*2 )
    for i in range( n ):
      transforms.extend( AffineTransformation( offset=numeric.array( [i,j], dtype=float ) / n, transform=trans ) for j in range(0,n-i) )
      transforms.extend( AffineTransformation( offset=numeric.array( [n-j,n-i], dtype=float ) / n, transform=-trans ) for j in range(n-i,n) )
    return transforms

  def refined( self, n ):
    'refine'

    assert n == 2
    if n == 1:
      return self
    return [ TriangularElement( id=self.id+'.child({})'.format(ichild), parent=(self,transform) ) for ichild, transform in enumerate( self.refinedtransform( n ) ) ]

  @staticmethod
  @core.cache
  def getischeme( ndims, where ):
    '''get integration scheme
    gaussian quadrature: http://www.cs.rpi.edu/~flaherje/pdf/fea6.pdf
    '''

    assert ndims == 2
    if where.startswith( 'contour' ):
      n = int( where[7:] or 0 )
      p = numeric.arange( n+1, dtype=float ) / (n+1)
      z = numeric.zeros_like( p )
      coords = numeric.hstack(( [1-p,p], [z,1-p], [p,z] ))
      weights = None
    elif where.startswith( 'vtk' ):
      coords = numeric.array([[0,0],[1,0],[0,1]]).T
      weights = None
    elif where == 'gauss1':
      coords = numeric.array( [[1],[1]] ) / 3.
      weights = numeric.array( [1] ) / 2.
    elif where in 'gauss2':
      coords = numeric.array( [[4,1,1],[1,4,1]] ) / 6.
      weights = numeric.array( [1,1,1] ) / 6.
    elif where == 'gauss3':
      coords = numeric.array( [[5,9,3,3],[5,3,9,3]] ) / 15.
      weights = numeric.array( [-27,25,25,25] ) / 96.
    elif where == 'gauss4':
      A = 0.091576213509771; B = 0.445948490915965; W = 0.109951743655322
      coords = numeric.array( [[1-2*A,A,A,1-2*B,B,B],[A,1-2*A,A,B,1-2*B,B]] )
      weights = numeric.array( [W,W,W,1/3.-W,1/3.-W,1/3.-W] ) / 2.
    elif where == 'gauss5':
      A = 0.101286507323456; B = 0.470142064105115; V = 0.125939180544827; W = 0.132394152788506
      coords = numeric.array( [[1./3,1-2*A,A,A,1-2*B,B,B],[1./3,A,1-2*A,A,B,1-2*B,B]] )
      weights = numeric.array( [1-3*V-3*W,V,V,V,W,W,W] ) / 2.
    elif where == 'gauss6':
      A = 0.063089014491502; B = 0.249286745170910; C = 0.310352451033785; D = 0.053145049844816; V = 0.050844906370207; W = 0.116786275726379
      VW = 1/6. - (V+W) / 2.
      coords = numeric.array( [[1-2*A,A,A,1-2*B,B,B,1-C-D,1-C-D,C,C,D,D],[A,1-2*A,A,B,1-2*B,B,C,D,1-C-D,D,1-C-D,C]] )
      weights = numeric.array( [V,V,V,W,W,W,VW,VW,VW,VW,VW,VW] ) / 2.
    elif where == 'gauss7':
      A = 0.260345966079038; B = 0.065130102902216; C = 0.312865496004875; D = 0.048690315425316; U = 0.175615257433204; V = 0.053347235608839; W = 0.077113760890257
      coords = numeric.array( [[1./3,1-2*A,A,A,1-2*B,B,B,1-C-D,1-C-D,C,C,D,D],[1./3,A,1-2*A,A,B,1-2*B,B,C,D,1-C-D,D,1-C-D,C]] )
      weights = numeric.array( [1-3*U-3*V-6*W,U,U,U,V,V,V,W,W,W,W,W,W] ) / 2.
    elif where[:7] == 'uniform':
      N = int( where[7:] )
      points = ( numeric.arange( N ) + 1./3 ) / N
      NN = N**2
      C = numeric.empty( [2,N,N] )
      C[0] = points[:,_]
      C[1] = points[_,:]
      coords = C.reshape( 2, NN )
      flip = coords[0] + numeric.greater( coords[1], 1 )
      coords[:,flip] = 1 - coords[::-1,flip]
      weights = numeric.appendaxes( .5/NN, NN )
    elif where[:6] == 'bezier':
      N = int( where[6:] )
      points = numeric.linspace( 0, 1, N )
      coords = numeric.array([ [x,y] for i, y in enumerate(points) for x in points[:N-i] ]).T
      weights = None
    else:
      raise Exception, 'invalid element evaluation: %r' % where
    return coords.T, weights

  def select_contained( self, points, eps=0 ):
    'select points contained in element'

    selection = numeric.ones( points.shape[0], dtype=bool )
    for idim in 0, 1, 2:
      points_i = points[:,idim] if idim < 2 else 1-points.sum(1)
      newsel = numeric.greater_equal( points_i, -eps )
      selection[selection] &= newsel
      points = points[newsel]
      if not points.size:
        return None, None

    return points, selection

class TetrahedronElement( Element ):
  'tetrahedron element'

  __slots__ = ()

  neighbormap = -1, 3, 2, 1, 0
  #Defined to create outward pointing normal vectors for all edges (i.c. triangular faces)
  edgetransform = (
    AffineTransformation( offset=[0,0,0], transform=[[ 0, 1],[1,0],[0,0]] ),
    AffineTransformation( offset=[0,0,0], transform=[[ 1, 0],[0,0],[0,1]] ),
    AffineTransformation( offset=[0,0,0], transform=[[ 0, 0],[0,1],[1,0]] ),
    AffineTransformation( offset=[1,0,0], transform=[[-1,-1],[1,0],[0,1]] ) )

  def __init__( self, vertices, index=None, parent=None, context=None ):
    'constructor'

    assert len(vertices) == 4
    Element.__init__( self, ndims=3, vertices=vertices, index=index, parent=parent, context=context )

  @property
  def children( self ):
    'all 1x refined elements'
    raise NotImplementedError( 'Children of tetrahedron' )  
      
  @property
  def edges( self ):
    return [ self.edge(iedge) for iedge in range(4) ]

  def edge( self, iedge ):
    'edge'

    transform = self.edgetransform[ iedge ]
    v1, v2, v3, v4 = self.vertices
    vertices = [ [v1,v3,v2], [v1,v2,v4], [v1,v4,v3], [v2,v3,v4] ][ iedge ] # TODO check!
    return TriangularElement( vertices=vertices, context=(self,transform) )

  @staticmethod
  @core.cache
  def refinedtransform( n ):
    'refined transform'
    raise NotImplementedError( 'Transformations for refined tetrahedrons' )  

  def refined( self, n ):
    'refine'
    raise NotImplementedError( 'Refinement tetrahedrons' )  

  @staticmethod
  @core.cache
  def getischeme( ndims, where ):
    '''get integration scheme
       http://people.sc.fsu.edu/~jburkardt/datasets/quadrature_rules_tet/quadrature_rules_tet.html'''

    assert ndims == 3
    if where.startswith( 'vtk' ):
      coords = numeric.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]]).T
      weights = None
    elif where == 'gauss1':
      coords = numeric.array( [[1],[1],[1]] ) / 4.
      weights = numeric.array( [1] ) / 6.
    elif where == 'gauss2':
      coords = numeric.array([[0.5854101966249685,0.1381966011250105,0.1381966011250105],
                              [0.1381966011250105,0.1381966011250105,0.1381966011250105],
                              [0.1381966011250105,0.1381966011250105,0.5854101966249685],
                              [0.1381966011250105,0.5854101966249685,0.1381966011250105]]).T
      weights = numeric.array([1,1,1,1]) / 24.
    elif where == 'gauss3':
      coords = numeric.array([[0.2500000000000000,0.2500000000000000,0.2500000000000000],
                              [0.5000000000000000,0.1666666666666667,0.1666666666666667],
                              [0.1666666666666667,0.1666666666666667,0.1666666666666667],
                              [0.1666666666666667,0.1666666666666667,0.5000000000000000],
                              [0.1666666666666667,0.5000000000000000,0.1666666666666667]]).T
      weights = numeric.array([-0.8000000000000000,0.4500000000000000,0.4500000000000000,0.4500000000000000,0.4500000000000000]) / 6.
    elif where == 'gauss4':
      coords = numeric.array([[0.2500000000000000,0.2500000000000000,0.2500000000000000],
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
      weights = numeric.array([-0.0789333333333333,0.0457333333333333,0.0457333333333333,0.0457333333333333,0.0457333333333333,0.1493333333333333,0.1493333333333333,0.1493333333333333,0.1493333333333333,0.1493333333333333,0.1493333333333333]) / 6.
    elif where == 'gauss5':
      coords = numeric.array([[0.2500000000000000,0.2500000000000000,0.2500000000000000],
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
      weights = numeric.array([0.1817020685825351,0.0361607142857143,0.0361607142857143,0.0361607142857143,0.0361607142857143,0.0698714945161738,0.0698714945161738,0.0698714945161738,0.0698714945161738,0.0656948493683187,0.0656948493683187,0.0656948493683187,0.0656948493683187,0.0656948493683187,0.0656948493683187]) / 6.
    elif where == 'gauss6':
      coords = numeric.array([[0.3561913862225449,0.2146028712591517,0.2146028712591517],
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
      weights = numeric.array([0.0399227502581679,0.0399227502581679,0.0399227502581679,0.0399227502581679,0.0100772110553207,0.0100772110553207,0.0100772110553207,0.0100772110553207,0.0553571815436544,0.0553571815436544,0.0553571815436544,0.0553571815436544,0.0482142857142857,0.0482142857142857,0.0482142857142857,0.0482142857142857,0.0482142857142857,0.0482142857142857,0.0482142857142857,0.0482142857142857,0.0482142857142857,0.0482142857142857,0.0482142857142857,0.0482142857142857]) / 6.
    elif where == 'gauss7':
      coords = numeric.array([[0.2500000000000000,0.2500000000000000,0.2500000000000000],
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
      weights = numeric.array([0.1095853407966528,0.0635996491464850,0.0635996491464850,0.0635996491464850,0.0635996491464850,-0.3751064406859797,-0.3751064406859797,-0.3751064406859797,-0.3751064406859797,0.0293485515784412,0.0293485515784412,0.0293485515784412,0.0293485515784412,0.0058201058201058,0.0058201058201058,0.0058201058201058,0.0058201058201058,0.0058201058201058,0.0058201058201058,0.1653439153439105,0.1653439153439105,0.1653439153439105,0.1653439153439105,0.1653439153439105,0.1653439153439105,0.1653439153439105,0.1653439153439105,0.1653439153439105,0.1653439153439105,0.1653439153439105,0.1653439153439105]) / 6.
    elif where == 'gauss8':
      coords = numeric.array([[0.2500000000000000,0.2500000000000000,0.2500000000000000],
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
      weights = numeric.array([-0.2359620398477557,0.0244878963560562,0.0244878963560562,0.0244878963560562,0.0244878963560562,0.0039485206398261,0.0039485206398261,0.0039485206398261,0.0039485206398261,0.0263055529507371,0.0263055529507371,0.0263055529507371,0.0263055529507371,0.0263055529507371,0.0263055529507371,0.0829803830550589,0.0829803830550589,0.0829803830550589,0.0829803830550589,0.0829803830550589,0.0829803830550589,0.0254426245481023,0.0254426245481023,0.0254426245481023,0.0254426245481023,0.0254426245481023,0.0254426245481023,0.0254426245481023,0.0254426245481023,0.0254426245481023,0.0254426245481023,0.0254426245481023,0.0254426245481023,0.0134324384376852,0.0134324384376852,0.0134324384376852,0.0134324384376852,0.0134324384376852,0.0134324384376852,0.0134324384376852,0.0134324384376852,0.0134324384376852,0.0134324384376852,0.0134324384376852,0.0134324384376852]) / 6.
    else:
      raise Exception, 'invalid element evaluation: %r' % where
    return coords.T, weights

  def select_contained( self, points, eps=0 ):
    'select points contained in element'
    raise NotImplementedError( 'Determine whether a point resides in the tetrahedron' )  

class StdElem( object ):
  'stdelem base class'

  __slots__ = 'ndims', 'nshapes'

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

  __slots__ = 'std1', 'std2'

  @core.cache
  def __new__( cls, std1, std2 ):
    'constructor'

    self = object.__new__( cls )
    self.std1 = std1
    self.std2 = std2
    self.ndims = std1.ndims + std2.ndims
    self.nshapes = std1.nshapes * std2.nshapes
    return self

  @core.cache
  def eval( self, points, grad=0 ):
    'evaluate'
    # log.debug( '@ PolyProduct.eval: ', id(self), id(points), id(grad) )

    assert isinstance( grad, int ) and grad >= 0

    assert points.shape[-1] == self.ndims

    s1 = slice(0,self.std1.ndims)
    p1 = points[...,s1]
    s2 = slice(self.std1.ndims,None)
    p2 = points[...,s2]

    E = Ellipsis,
    S = slice(None),
    N = _,

    shape = points.shape[:-1] + (self.std1.nshapes * self.std2.nshapes,)
    G12 = [ ( self.std1.eval( p1, grad=i )[E+S+N+S*i+N*j]
            * self.std2.eval( p2, grad=j )[E+N+S+N*i+S*j] ).reshape( shape + (self.std1.ndims,) * i + (self.std2.ndims,) * j )
            for i,j in zip( range(grad,-1,-1), range(grad+1) ) ]

    data = numeric.empty( shape + (self.ndims,) * grad )

    s = (s1,)*grad + (s2,)*grad
    R = numeric.arange(grad)
    for n in range(2**grad):
      index = n>>R&1
      n = index.argsort() # index[s] = [0,...,1]
      shuffle = range(points.ndim) + list( points.ndim + n )
      iprod = index.sum()
      data.transpose(shuffle)[E+s[iprod:iprod+grad]] = G12[iprod]

    return data

  def __str__( self ):
    'string representation'

    return '%s*%s' % ( self.std1, self.std2 )

class PolyLine( StdElem ):
  'polynomial on a line'

  __slots__ = 'degree', 'poly'

  @classmethod
  def bernstein_poly( cls, degree ):
    'bernstein polynomial coefficients'

    # magic bernstein triangle
    revpoly = numeric.zeros( [degree+1,degree+1], dtype=int )
    for k in range(degree//2+1):
      revpoly[k,k] = root = (-1)**degree if k == 0 else ( revpoly[k-1,k] * (k*2-1-degree) ) / k
      for i in range(k+1,degree+1-k):
        revpoly[i,k] = revpoly[k,i] = root = ( root * (k+i-degree-1) ) / i
    return revpoly[::-1]

  @classmethod
  def spline_poly( cls, p, n ):
    'spline polynomial coefficients'

    assert p >= 0, 'invalid polynomial degree %d' % p
    if p == 0:
      assert n == -1
      return numeric.array( [[[1.]]] )

    assert 1 <= n < 2*p
    extractions = numeric.empty(( n, p+1, p+1 ))
    extractions[0] = numeric.eye( p+1 )
    for i in range( 1, n ):
      extractions[i] = numeric.eye( p+1 )
      for j in range( 2, p+1 ):
        for k in reversed( range( j, p+1 ) ):
          alpha = 1. / min( 2+k-j, n-i+1 )
          extractions[i-1,:,k] = alpha * extractions[i-1,:,k] + (1-alpha) * extractions[i-1,:,k-1]
        extractions[i,-j-1:-1,-j-1] = extractions[i-1,-j:,-1]

    poly = cls.bernstein_poly( p )
    return numeric.contract( extractions[:,_,:,:], poly[_,:,_,:], axis=-1 )

  @classmethod
  @core.cache
  def spline_elems( cls, p, n ):
    'spline elements, minimum amount (just for caching)'

    return map( cls, cls.spline_poly(p,n) )

  @classmethod
  @core.cache
  def spline_elems_neumann( cls, p, n ):
    'spline elements, neumann endings (just for caching)'

    polys = cls.spline_poly(p,n)
    poly_0 = polys[0].copy()
    poly_0[:,1] += poly_0[:,0]
    poly_e = polys[-1].copy()
    poly_e[:,-2] += poly_e[:,-1]
    return cls(poly_0), cls(poly_e)

  @classmethod
  @core.cache
  def spline_elems_curvature( cls ):
    'spline elements, curve free endings (just for caching)'

    polys = cls.spline_poly(1,1)
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
    n = 2*p-1
    if periodic:
      assert not neumann, 'periodic domains have no boundary'
      assert not curvature, 'curvature free option not possible for periodic domains'
      if nelems == 1: # periodicity on one element can only mean a constant
        elems = cls.spline_elems( 0, n )
      else:
        elems = cls.spline_elems( p, n )[p-1:p] * nelems
    else:
      elems = cls.spline_elems( p, min(nelems,n) )
      if len(elems) < nelems:
        elems = elems[:p-1] + elems[p-1:p] * (nelems-2*(p-1)) + elems[p:]
      if neumann:
        elem_0, elem_e = cls.spline_elems_neumann( p, min(nelems,n) )
        if neumann & 1:
          elems[0] = elem_0
        if neumann & 2:
          elems[-1] = elem_e
      if curvature:
        assert neumann==0, 'Curvature free not allowed in combindation with Neumann'
        assert degree==2, 'Curvature free only allowed for quadratic splines'  
        elem_0, elem_e = cls.spline_elems_curvature()
        elems[0] = elem_0
        elems[-1] = elem_e

    return numeric.array( elems )

  def __init__( self, poly ):
    '''Create polynomial from order x nfuncs array of coefficients 'poly'.
       Evaluates to sum_i poly[i,:] x**i.'''

    self.ndims = 1
    self.poly = numeric.asarray( poly, dtype=float )
    order, self.nshapes = self.poly.shape
    self.degree = order - 1

  @core.cache
  def eval( self, points, grad=0 ):
    'evaluate'

    assert points.shape[-1] == 1
    x = points[...,0]

    if grad > self.degree:
      return numeric.appendaxes( 0., x.shape+(self.nshapes,)+(1,)*grad )

    poly = self.poly
    for n in range(grad):
      poly = poly[1:] * numeric.arange( 1, poly.shape[0] )[:,_]

    polyval = numeric.empty( x.shape+(self.nshapes,) )
    polyval[:] = poly[-1]
    for p in poly[-2::-1]:
      polyval *= x[...,_]
      polyval += p

    return polyval[(Ellipsis,)+(_,)*grad]

  def extract( self, extraction ):
    'apply extraction'

    return PolyLine( numeric.dot( self.poly, extraction ) )

  def __repr__( self ):
    'string representation'

    return 'PolyLine#%x' % id(self)

class PolyTriangle( StdElem ):
  '''poly triangle (linear for now)
     conventions: dof numbering as vertices, see TriangularElement docstring.'''

  __slots__ = ()

  @core.cache
  def __new__( cls, order ):
    'constructor'

    assert order == 1
    self = object.__new__( cls )
    return self

  @core.cache
  def eval( self, points, grad=0 ):
    'eval'

    npoints, ndim = points.shape
    if grad == 0:
      x, y = points.T
      data = numeric.array( [ x, y, 1-x-y ] ).T
    elif grad == 1:
      data = numeric.array( [[1,0],[0,1],[-1,-1]], dtype=float )
    else:
      data = numeric.array( 0 ).reshape( (1,) * (grad+ndim) )
    return data

  def __repr__( self ):
    'string representation'

    return '%s#%x' % ( self.__class__.__name__, id(self) )

class BubbleTriangle( StdElem ):
  '''linear triangle + bubble function
     conventions: dof numbering as vertices (see TriangularElement docstring), then barycenter.'''

  __slots__ = ()

  @core.cache
  def __new__( cls, order ):
    'constructor'

    assert order == 1
    self = object.__new__( cls )
    return self

  @core.cache
  def eval( self, points, grad=0 ):
    'eval'

    npoints, ndim = points.shape
    if grad == 0:
      x, y = points.T
      data = numeric.array( [ x, y, 1-x-y, 27*x*y*(1-x-y) ] ).T
    elif grad == 1:
      x, y = points.T
      const_block = numeric.array( [1,0,0,1,-1,-1]*npoints, dtype=float ).reshape( npoints,3,2 )
      grad1_bubble = 27*numeric.array( [y*(1-2*x-y),x*(1-x-2*y)] ).T.reshape( npoints,1,2 )
      data = numeric.concatenate( [const_block, grad1_bubble], axis=1 )
    elif grad == 2:
      x, y = points.T
      zero_block = numeric.zeros( (npoints,3,2,2) )
      grad2_bubble = 27*numeric.array( [-2*y,1-2*x-2*y, 1-2*x-2*y,-2*x] ).T.reshape( npoints,1,2,2 )
      data = numeric.concatenate( [zero_block, grad2_bubble], axis=1 )
    elif grad == 3:
      zero_block = numeric.zeros( (3,2,2,2) )
      grad3_bubble = 27*numeric.array( [0,-2,-2,-2,-2,-2,-2,0], dtype=float ).reshape( 1,2,2,2 )
      data = numeric.concatenate( [zero_block, grad3_bubble], axis=0 )
    else:
      assert ndim==2, 'Triangle takes 2D coordinates' # otherwise tested by unpacking points.T
      data = numeric.array( 0 ).reshape( (1,) * (grad+2) )
    return data

  def __repr__( self ):
    'string representation'

    return '%s#%x' % ( self.__class__.__name__, id(self) )

class ExtractionWrapper( object ):
  'extraction wrapper'

  __slots__ = 'stdelem', 'extraction'

  def __init__( self, stdelem, extraction ):
    'constructor'

    self.stdelem = stdelem
    self.extraction = extraction

  @core.cache
  def eval( self, points, grad=0 ):
    'call'

    return numeric.dot( self.stdelem.eval( points, grad ), self.extraction, axis=1 )

  def __repr__( self ):
    'string representation'

    return '%s#%x:%s' % ( self.__class__.__name__, id(self), self.stdelem )

# vim:shiftwidth=2:foldmethod=indent:foldnestmax=2
