from . import log, util, cache, numeric, transform, function, _
import warnings


class Element( cache.WeakCacheObject ):
  '''Element base class.

  Represents the topological shape.'''

  __slots__ = 'vertices', 'simplex', 'parent', 'context', 'interface', 'root_transform'

  def __init__( self, simplex, vertices=(), parent=None, context=None, interface=None ):
    'constructor'

    assert isinstance( vertices, tuple )
    self.vertices = vertices
    self.simplex = simplex
    self.parent = parent
    self.context = context
    self.interface = interface
    if parent:
      pelem, trans = parent
      self.root_transform = pelem.root_transform * trans
    else:
      self.root_transform = transform.Identity( simplex.ndims )

  @property
  def simplices( self ):
    if isinstance( self.simplex, Simplex ):
      return self,
    simplices = []
    for child in self.children:
      simplices.extend( child.simplices )
    return simplices

  @property
  def ndims( self ):
    return self.simplex.ndims

  @property
  def edges( self ):
    return [ Element( simplex=simplex, context=(self,trans), vertices=mknewvertices(self.vertices,iverts) )
      for simplex, trans, iverts in self.simplex.edges ]

  def edge( self, iedge ):
    simplex, trans, iverts = self.simplex.edges[iedge]
    return Element( simplex=simplex, context=(self,trans), vertices=mknewvertices(self.vertices,iverts) )

  @property
  def children( self ):
    return [ Element( simplex=simplex, parent=(self,trans), vertices=mknewvertices(self.vertices,iverts) )
      for simplex, trans, iverts in self.simplex.children ]

  def children_by( self, N ):
    return [ Element( simplex=simplex, parent=(self,trans), vertices=mknewvertices(self.vertices,iverts) )
      for simplex, trans, iverts in self.simplex.children_by(N) ]
    
  def trim( self, levelset, maxrefine=0, minrefine=0 ):
    assert maxrefine >= minrefine >= 0
    if minrefine == 0:
      values = levelset.eval( self, 'bezier%d' % (2**maxrefine+1) )
      mosaics = self.simplex.mosaic( values )
    else:
      # refine to evaluate levelset, then assemble the pieces
      allpieces = [], []
      for child in self.children:
        self_, ptrans = child.parent
        for pieces, elem in zip( allpieces, child.trim( levelset, maxrefine-1, minrefine-1 ) ):
          if elem:
            if isinstance( elem.simplex, Mosaic ):
              pieces.extend( (simplex,ptrans*trans,iverts) for simplex, trans, iverts in elem.simplex.children )
            else:
              pieces.append( (elem.simplex,ptrans,()) )
      mosaics = [ Mosaic(pieces) if pieces else None for pieces in allpieces ]
    if not mosaics[1]:
      return self, None
    if not mosaics[0]:
      return None, self
    identity = transform.Identity(self.ndims)
    return [ Element( simplex=mosaic, parent=(self,identity), vertices=self.vertices ) for mosaic in mosaics ]

# def __mul__( self, other ):
#   'multiply elements'

#   return ProductElement( self, other )

# def neighbor( self, other ):
#   'level of neighborhood; 0=self'

#   if self == other:
#     return 0
#   ncommon = len( set(self.vertices) & set(other.vertices) )
#   return self.neighbormap[ ncommon ]

# def intersected( self, levelset, lscheme, evalrefine=0 ):
#   '''check levelset intersection:
#     +1 for levelset > 0 everywhere
#     -1 for levelset < 0 everywhere
#      0 for intersected element'''

#   levelset = function.ascompiled( levelset )
#   elems = iter( [self] )
#   for irefine in range(evalrefine):
#     elems = ( child for elem in elems for child in elem.children )

#   inside = numeric.greater( levelset.eval( elems.next(), lscheme ), 0 )
#   if inside.all():
#     for elem in elems:
#       inside = numeric.greater( levelset.eval( elem, lscheme ), 0 )
#       if not inside.all():
#         return 0
#     return 1
#   elif not inside.any():
#     for elem in elems:
#       inside = numeric.greater( levelset.eval( elem, lscheme ), 0 )
#       if inside.any():
#         return 0
#     return -1
#   return 0

# def get_trimmededges ( self, maxrefine ):
#   return []


## SIMPLICES


class Simplex( cache.WeakCacheObject ):

  def __init__( self, ndims ):
    self.ndims = ndims

class Quad( Simplex ):
  'quadrilateral element'

  __slots__ = ()

  def __init__( self, ndims ):
    'constructor'

    Simplex.__init__( self, ndims=ndims )

  def getischeme( self, where ):
    'get integration scheme'

    x = w = None
    if self.ndims == 0:
      coords = numeric.zeros([1,0])
      weights = numeric.array([1.])
    elif where.startswith( 'gauss' ):
      N = eval( where[5:] )
      if isinstance( N, tuple ):
        assert len(N) == self.ndims
      else:
        N = [N]*self.ndims
      x, w = zip( *map( getgauss, N ) )
    elif where.startswith( 'uniform' ):
      N = eval( where[7:] )
      if isinstance( N, tuple ):
        assert len(N) == self.ndims
      else:
        N = [N]*self.ndims
      x = [ numeric.arange( .5, n ) / n for n in N ]
      w = [ numeric.appendaxes( 1./n, n ) for n in N ]
    elif where.startswith( 'bezier' ):
      N = int( where[6:] )
      x = [ numeric.linspace( 0, 1, N ) ] * self.ndims
      w = [ numeric.appendaxes( 1./N, N ) ] * self.ndims
    elif where.startswith( 'subdivision' ):
      N = int( where[11:] ) + 1
      x = [ numeric.linspace( 0, 1, N ) ] * self.ndims
      w = None
    elif where.startswith( 'vtk' ):
      if self.ndims == 1:
        coords = numeric.array([[0,1]]).T
      elif self.ndims == 2:
        eps = 0 if not len(where[3:]) else float(where[3:]) # subdivision fix (avoid extraordinary point)
        coords = numeric.array([[eps,eps],[1-eps,eps],[1-eps,1-eps],[eps,1-eps]])
      elif self.ndims == 3:
        coords = numeric.array([ [0,0,0], [1,0,0], [0,1,0], [1,1,0], [0,0,1], [1,0,1], [0,1,1], [1,1,1] ])
      else:
        raise Exception, 'contour not supported for ndims=%d' % self.ndims
    elif where.startswith( 'contour' ):
      N = int( where[7:] )
      p = numeric.linspace( 0, 1, N )
      if self.ndims == 1:
        coords = p[_].T
      elif self.ndims == 2:
        coords = numeric.array([ p[ range(N) + [N-1]*(N-2) + range(N)[::-1] + [0]*(N-1) ],
                                 p[ [0]*(N-1) + range(N) + [N-1]*(N-2) + range(0,N)[::-1] ] ]).T
      elif self.ndims == 3:
        assert N == 0
        coords = numeric.array([ [0,0,0], [1,0,0], [0,1,0], [1,1,0], [0,0,1], [1,0,1], [0,1,1], [1,1,1] ])
      else:
        raise Exception, 'contour not supported for ndims=%d' % self.ndims
    else:
      raise Exception, 'invalid element evaluation %r' % where
    if x is not None:
      coords = numeric.empty( map( len, x ) + [ self.ndims ] )
      for i, xi in enumerate( x ):
        coords[...,i] = xi[ (slice(None),) + (_,)*(self.ndims-i-1) ]
      coords = coords.reshape( -1, self.ndims )
    if w is not None:
      weights = reduce( lambda weights, wi: ( weights * wi[:,_] ).ravel(), w )
    else:
      weights = None
    return numeric.concatenate( [coords,weights[:,_]], axis=-1 ) if weights is not None else coords

  def stdfunc( self, degree ):
    return PolyLine.bernstein_poly( degree )**self.ndims

  @cache.property
  def edges( self ):
    transforms = []
    simplex = Quad( self.ndims-1 )
    iverts = numeric.arange( 2**self.ndims ).reshape( (2,)*self.ndims )
    for idim in range( self.ndims ):
      offset = numeric.zeros( self.ndims )
      offset[:idim] = 1
      matrix = numeric.zeros(( self.ndims, self.ndims-1 ))
      matrix.flat[ :(self.ndims-1)*idim :self.ndims] = -1
      matrix.flat[self.ndims*(idim+1)-1::self.ndims] = 1
      for side in 0, 1:
        offset[idim] = side
        # flip normal:
        offset[idim-1] = 1 - offset[idim-1]
        matrix[idim-1] *= -1
        iedgeverts = numeric.getitem(iverts,idim,side) # TODO fix!
        transforms.append(( simplex, transform.Linear(matrix.copy()) + offset, iedgeverts ))
    return transforms

  def children_by( self, N ):
    'divide element by n'

    Nrcp = numeric.reciprocal( N, dtype=float )
    scale = transform.Scale(Nrcp)
    refinedtransform = [ scale + i*Nrcp for i in numeric.ndindex(*N) ]

    assert len(N) == self.ndims
    vertices = numeric.empty( [ ni+1 for ni in N ], dtype=object )
    vertices[ tuple( slice(None,None,ni) for ni in N ) ] = numeric.arange( 2*self.ndims ).reshape( [2]*self.ndims )
    for idim in range(self.ndims):
      s1 = tuple( slice(None) for ni in N[:idim] )
      s2 = tuple( slice(None,None,ni) for ni in N[idim+1:] )
      for i in range( 1, N[idim] ):
        vertices[s1+(i,)+s2] = zip( vertices[s1+(0,)+s2], vertices[s1+(2,)+s2] ) # TODO fix fraction

    elemvertices = [ vertices[ tuple( slice(i,i+2) for i in index ) ].ravel() for index in numeric.ndindex(*N) ]
    return [ ( self, trans, elemvertices[ielem] ) for ielem, trans in enumerate( refinedtransform ) ]

  @cache.property
  def children( self ):
    'all 1x refined elements'

    return self.children_by( (2,)*self.ndims )

  def _triangulate( self, values ):
    assert values.shape == (2,)*self.ndims
    pos = []
    neg = []
    coords = numeric.zeros( (2,)*self.ndims + (self.ndims,) )
    for i in range(self.ndims):
      numeric.getitem(coords,i,1)[...,i] = 1
    for coord, value in zip( coords.reshape(-1,self.ndims), values.ravel() ):
      ( pos if value > 0 else neg ).append( coord )
    for idim in range( self.ndims ):
      v0 = numeric.getitem( values, idim, 0 ).ravel()
      dv = numeric.getitem( values, idim, 1 ).ravel() - v0
      c0 = numeric.getitem( coords, idim, 0 ).reshape(-1,self.ndims)
      dc = numeric.getitem( coords, idim, 1 ).reshape(-1,self.ndims) - c0
      x = -v0 / dv # v0 + x dv = 0
      for i in numeric.find( numeric.greater_equal(x,0) & numeric.less_equal(x,1) ):
        coord = c0[i] + x[i] * dc[i]
        pos.append( coord )
        neg.append( coord )
    triangle = Triangle()
    return [ (triangle,trans,()) for trans in transform.delaunay(pos) ], \
           [ (triangle,trans,()) for trans in transform.delaunay(neg) ]

  def _mosaic( self, values, eps ):
    if numeric.greater( values, -eps ).all():
      return [(self,transform.Identity(self.ndims),())], []
    if numeric.less( values, +eps ).all():
      return [], [(self,transform.Identity(self.ndims),())]
    n = values.shape[0]
    assert values.shape == (n,)*self.ndims
    if n == 2:
      return self._triangulate( values )
    slices = slice(0,n//2+1), slice(n//2,n+1)
    allpos = []
    allneg = []
    for i, (dummy,trans,dummy) in enumerate( self.children ):
      s = tuple( slices[n] for n in ( i >> numeric.arange(self.ndims-1,-1,-1) ) & 1 )
      pos, neg = self._mosaic( values[s], eps )
      for simplex, trans2, iverts in pos:
        allpos.append(( simplex, trans * trans2, iverts ))
      for simplex, trans2, iverts in neg:
        allneg.append(( simplex, trans * trans2, iverts ))
    return allpos, allneg

  def mosaic( self, values, eps=1e-10 ):
    assert values.ndim == 1
    n = 2
    while n**self.ndims < values.size:
      n = n*2 - 1
    assert n**self.ndims == values.size, 'cannot reshape values to appropriate shape'
    values = values.reshape( (n,)*self.ndims )
    pos, neg = self._mosaic( values, eps )
    if not neg:
      return self, None
    if not pos:
      return None, self
    return Mosaic(pos), Mosaic(neg)

# @cache.property
# def neighbormap( self ):
#   'maps # matching vertices --> codim of interface: {0: -1, 1: 2, 2: 1, 4: 0}'
#   return dict( [ (0,-1) ] + [ (2**(self.ndims-i),i) for i in range(self.ndims+1) ] )

# def select_contained( self, points, eps=0 ):
#   'select points contained in element'

#   selection = numeric.ones( points.shape[0], dtype=bool )
#   for idim in range( self.ndims ):
#     newsel = numeric.greater_equal( points[:,idim], -eps ) & numeric.less_equal( points[:,idim], 1+eps )
#     selection[selection] &= newsel
#     points = points[newsel]
#     if not points.size:
#       return None, None
#   return points, selection

class Triangle( Simplex ):
  '''triangular element
     conventions: reference elem:   unit simplex {(x,y) | x>0, y>0, x+y<1}
                  vertex numbering: {(1,0):0, (0,1):1, (0,0):2}
                  edge numbering:   {bottom:0, slanted:1, left:2}
                  edge local coords run counter-clockwise.'''

  __slots__ = ()

  neighbormap = -1, 2, 1, 0
  edges = (
    ( Quad(ndims=1), transform.Linear( numeric.array([[ 1],[ 0]]) ),         (2,0) ),
    ( Quad(ndims=1), transform.Linear( numeric.array([[-1],[ 1]]) ) + [1,0], (0,1) ),
    ( Quad(ndims=1), transform.Linear( numeric.array([[ 0],[-1]]) ) + [0,1], (1,2) ),
  )
  #  1
  # 1-2  0-1
  #  2   0-2   0
  @property
  def children( self ):
    return (
    ( self, transform.Scale(  numeric.asarray([.5,.5]) ),           ((0,2),(1,2),2) ),
    ( self, transform.Scale(  numeric.asarray([.5,.5]) ) + [.5, 0], (0,(0,1),(0,2)) ),
    ( self, transform.Scale(  numeric.asarray([.5,.5]) ) + [ 0,.5], ((0,1),1,(1,2)) ),
    ( self, transform.Scale( -numeric.asarray([.5,.5]) ) + [.5,.5], ((1,2),(0,2),(0,1)) ),
  )

  def __init__( self ):
    'constructor'

    Simplex.__init__( self, ndims=2 )

  def stdfunc( self, degree ):
    return PolyTriangle( degree )

  def getischeme( self, where ):
    '''get integration scheme
    gaussian quadrature: http://www.cs.rpi.edu/~flaherje/pdf/fea6.pdf
    '''

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
    return numeric.concatenate([coords.T,weights[...,_]],axis=-1) if weights is not None else coords.T


# @property
# def children( self ):
#   'all 1x refined elements'

#   t1, t2, t3, t4 = self.refinedtransform( 2 )
#   v1, v2, v3 = self.vertices
#   h1, h2, h3 = HalfVertex(v1,v2), HalfVertex(v2,v3), HalfVertex(v3,v1)
#   return tuple([ # TODO check!
#     TriangularElement( vertices=[v1,h1,h3], parent=(self,t1) ),
#     TriangularElement( vertices=[h1,v2,h2], parent=(self,t2) ),
#     TriangularElement( vertices=[h3,h2,v3], parent=(self,t3) ),
#     TriangularElement( vertices=[h2,h3,h1], parent=(self,t4) ) ])
#     
# @staticmethod
# def refinedtransform( n ):
#   'refined transform'

#   transforms = []
#   scale = transform.Scale( numeric.asarray([1./n,1./n]) )
#   negscale = transform.Scale( numeric.asarray([-1./n,-1./n]) )
#   for i in range( n ):
#     transforms.extend( scale + numeric.array( [i,j], dtype=float ) / n for j in range(0,n-i) )
#     transforms.extend( negscale + numeric.array( [n-j,n-i], dtype=float ) / n for j in range(n-i,n) )
#   return transforms

# def refined( self, n ):
#   'refine'

#   assert n == 2
#   if n == 1:
#     return self
#   return [ TriangularElement( id=self.id+'.child({})'.format(ichild), parent=(self,trans) ) for ichild, trans in enumerate( self.refinedtransform( n ) ) ]

# def select_contained( self, points, eps=0 ):
#   'select points contained in element'

#   selection = numeric.ones( points.shape[0], dtype=bool )
#   for idim in 0, 1, 2:
#     points_i = points[:,idim] if idim < 2 else 1-points.sum(1)
#     newsel = numeric.greater_equal( points_i, -eps )
#     selection[selection] &= newsel
#     points = points[newsel]
#     if not points.size:
#       return None, None

#   return points, selection

class Tetrahedron( Simplex ):
  'tetrahedron element'

  __slots__ = ()

  neighbormap = -1, 3, 2, 1, 0
  #Defined to create outward pointing normal vectors for all edges (i.c. triangular faces)
  edgetransform = (
    transform.Linear( numeric.array([[ 0, 1],[1,0],[0,0]]) ),
    transform.Linear( numeric.array([[ 1, 0],[0,0],[0,1]]) ),
    transform.Linear( numeric.array([[ 0, 0],[0,1],[1,0]]) ),
    transform.Linear( numeric.array([[-1,-1],[1,0],[0,1]]) ) + [1,0,0] )

  def __init__( self, vertices, parent=None, context=None ):
    'constructor'

    assert len(vertices) == 4
    Element.__init__( self, ndims=3, vertices=vertices, parent=parent, context=context )

  @property
  def children( self ):
    'all 1x refined elements'
    raise NotImplementedError( 'Children of tetrahedron' )  
      
  @property
  def edges( self ):
    return [ self.edge(iedge) for iedge in range(4) ]

  def edge( self, iedge ):
    'edge'

    trans = self.edgetransform[ iedge ]
    v1, v2, v3, v4 = self.vertices
    vertices = [ [v1,v3,v2], [v1,v2,v4], [v1,v4,v3], [v2,v3,v4] ][ iedge ] # TODO check!
    return TriangularElement( vertices=vertices, context=(self,trans) )

  @staticmethod
  def refinedtransform( n ):
    'refined transform'
    raise NotImplementedError( 'Transformations for refined tetrahedrons' )  

  def refined( self, n ):
    'refine'
    raise NotImplementedError( 'Refinement tetrahedrons' )  

  @staticmethod
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

class Product( Simplex ):
  'element product'

  __slots__ = 'elem1', 'elem2'

  @staticmethod
  def getslicetransforms( ndims1, ndims2 ):
    ndims = ndims1 + ndims2
    slice1 = transform.Slice( ndims, 0, ndims1 )
    slice2 = transform.Slice( ndims, ndims1, ndims )
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
    self.root_transform = transform.Scale( numeric.array([elem1.root_transform.det * elem2.root_transform.det]) ) # HACK

  @staticmethod
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
        trans = [0,2], [2,3], [3,1], [1,0], [2,0], [3,2], [1,3], [0,1]
      elif neighborhood==2:
        trans = [0], [2], [3], [1]
      else:
        raise ValueError( 'Unknown neighbor type %i' % neighborhood )
      transf1 = trans.index( vertices1 )
      transf2 = trans.index( vertices2 )
    elif isinstance( self.elem1, TriangularElement ):
      raise NotImplementedError( 'Pending completed implementation and verification.' )
      # define local map rotations
      if neighborhood==1:
        trans = [0,1], [1,2], [0,2]
      elif neighborhood==2:
        trans = [0], [1], [2]
      else:
        raise ValueError( 'Unknown neighbor type %i' % neighborhood )
      transf1 = trans.index( vertices1 )
      transf2 = trans.index( vertices2 )
    else:
      raise NotImplementedError( 'Reorientation not implemented for element of class %s' % type(self.elem1) )
    return neighborhood, transf1, transf2

  @staticmethod
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
  def singular_ischeme_quad( orientation, ischeme ):
    neighborhood, transf1, transf2 = orientation
    points, weights = ProductElement.get_quad_bem_ischeme( ischeme, neighborhood )
    transfpoints = numeric.empty( points.shape )
    def flipxy( points, orientation ):
      x, y = points[:,0], points[:,1]
      tx = x if orientation in (0,1,6,7) else 1-x
      ty = y if orientation in (0,3,4,7) else 1-y
      return function.stack( (ty, tx) if orientation%2 else (tx, ty), axis=1 )
    transfpoints[:,:2] = flipxy( points[:,:2], transf1 )
    transfpoints[:,2:] = flipxy( points[:,2:], transf2 )
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


## MOSAIC


class Mosaic( object ):
  'trimmed simplex'

  def __init__( self, children ):
    self.children = children
    self.ndims, = util.filterrepeat( simplex.ndims for simplex, trans, iverts in children )

  def getischeme( self, ischeme ):
    pw_all = []
    for simplex, trans, iverts in self.children:
      pw = simplex.getischeme( ischeme )
      pw_trans = trans.apply( pw[...,:self.ndims] ), trans.det * pw[...,self.ndims:]
      pw_concat = numeric.concatenate( pw_trans, axis=-1 )
      pw_all.append( pw_concat )
    return numeric.concatenate( pw_all, axis=0 )


## STDELEMS


class StdElem( cache.WeakCacheObject ):
  'stdelem base class'

  __slots__ = 'ndims', 'nshapes'

  def __init__( self, ndims, nshapes ):
    self.ndims = ndims
    self.nshapes = nshapes

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

  __slots__ = 'std',

  def __init__( self, *std ):
    'constructor'

    std1, std2 = self.std = std
    StdElem.__init__( self, ndims=std1.ndims+std2.ndims, nshapes=std1.nshapes*std2.nshapes )

  def eval( self, points, grad=0 ):
    'evaluate'
    # log.debug( '@ PolyProduct.eval: ', id(self), id(points), id(grad) )

    assert isinstance( grad, int ) and grad >= 0

    assert points.shape[-1] == self.ndims
    std1, std2 = self.std

    s1 = slice(0,std1.ndims)
    p1 = points[...,s1]
    s2 = slice(std1.ndims,None)
    p2 = points[...,s2]

    E = Ellipsis,
    S = slice(None),
    N = _,

    shape = points.shape[:-1] + (std1.nshapes * std2.nshapes,)
    G12 = [ ( std1.eval( p1, grad=i )[E+S+N+S*i+N*j]
            * std2.eval( p2, grad=j )[E+N+S+N*i+S*j] ).reshape( shape + (std1.ndims,) * i + (std2.ndims,) * j )
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

    return '%s*%s' % self.std

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
  def spline_elems( cls, p, n ):
    'spline elements, minimum amount (just for caching)'

    return map( cls, cls.spline_poly(p,n) )

  @classmethod
  def spline_elems_neumann( cls, p, n ):
    'spline elements, neumann endings (just for caching)'

    polys = cls.spline_poly(p,n)
    poly_0 = polys[0].copy()
    poly_0[:,1] += poly_0[:,0]
    poly_e = polys[-1].copy()
    poly_e[:,-2] += poly_e[:,-1]
    return cls(poly_0), cls(poly_e)

  @classmethod
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

    self.poly = numeric.asarray( poly, dtype=float )
    order, nshapes = self.poly.shape
    self.degree = order - 1
    StdElem.__init__( self, ndims=1, nshapes=nshapes )

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

  def __init__( self, degree ):
    'constructor'

    assert degree == 1, 'only linear implemented on triangles for now'
    StdElem.__init__( self, ndims=2, nshapes=3 )

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

  def __init__( self, order ):
    'constructor'

    assert order == 1
    StdElem.__init__( self, ndims=2, nshapes=4 )

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

class ExtractionWrapper( StdElem ):
  'extraction wrapper'

  __slots__ = 'stdelem', 'extraction'

  def __init__( self, stdelem, extraction ):
    'constructor'

    self.stdelem = stdelem
    assert extraction.shape[0] == stdelem.nshapes
    self.extraction = extraction
    StdElem.__init__( self, ndims=stdelem.ndims, nshapes=extraction.shape[1] )

  def eval( self, points, grad=0 ):
    'call'

    return numeric.dot( self.stdelem.eval( points, grad ), self.extraction, axis=1 )

  def __repr__( self ):
    'string representation'

    return '%s#%x:%s' % ( self.__class__.__name__, id(self), self.stdelem )


PrimaryVertex = str
HalfVertex = lambda vertex1, vertex2, xi=.5: '%s<%.3f>%s' % ( (vertex1,xi,vertex2) if vertex1 < vertex2 else (vertex2,1-xi,vertex1) )
ProductVertex = lambda *vertices: ','.join( vertices )


## UTILITY FUNCTIONS


def mknewvertices( verts, iverts ):
  return tuple( verts[ivert] if isinstance( ivert, int ) \
    else '(%s)' % ','.join( sorted( mknewvertices( verts, ivert ) ) )
      for ivert in iverts )

def getgauss( degree ):
  'compute gauss points and weights'

  assert isinstance( degree, int ) and degree >= 0
  k = numeric.arange( 1, degree // 2 + 1 )
  d = k / numeric.sqrt( 4*k**2-1 )
  x, w = numeric.eigh( numeric.diagflat(d,-1) ) # eigh operates (by default) on lower triangle
  return (x+1) * .5, w[0]**2


# vim:shiftwidth=2:foldmethod=indent:foldnestmax=2
