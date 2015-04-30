# -*- coding: utf8 -*-
#
# Module ELEMENT
#
# Part of Nutils: open source numerical utilities for Python. Jointly developed
# by HvZ Computational Engineering, TU/e Multiscale Engineering Fluid Dynamics,
# and others. More info at http://nutils.org <info@nutils.org>. (c) 2014

"""
The element module defines reference elements such as the :class:`QuadElement`
and :class:`TriangularElement`, but also more exotic objects like the
:class:`TrimmedElement`. A set of (interconnected) elements together form a
:mod:`nutils.topology`. Elements have edges and children (for refinement), which
are in turn elements and map onto self by an affine transformation. They also
have a well defined reference coordinate system, and provide pointsets for
purposes of integration and sampling.
"""

from __future__ import print_function, division
from . import log, util, numpy, core, numeric, function, cache, transform, rational, _
import re, warnings


## ELEMENT

class Element( object ):
  'element class'

  __slots__ = 'transform', 'reference', '__opposite'

  def __init__( self, reference, trans, opposite=None ):
    assert isinstance( reference, Reference )
    assert trans.fromdims == reference.ndims
    assert trans.todims == None
    self.reference = reference
    self.transform = trans.canonical
    if opposite is not None:
      opposite = opposite.canonical
      if opposite == self.transform:
        opposite = None
    self.__opposite = opposite

  @property
  def opposite( self ):
    return self.__opposite or self.transform

  def __hash__( self ):
    return object.__hash__( self )

  def __eq__( self, other ):
    return self is other or isinstance(other,Element) \
      and self.reference == other.reference \
      and self.transform == other.transform \
      and self.__opposite == other.__opposite

  @property
  def vertices( self ):
    return self.transform.apply( self.reference.vertices )

  @property
  def ndims( self ):
    return self.reference.ndims

  @property
  def nverts( self ):
    return self.reference.nverts

  @property
  def nedges( self ):
    return self.reference.nedges

  @property
  def edges( self ):
    return [ self.edge(i) for i in range(self.nedges) ]
    
  def edge( self, iedge ):
    trans, edge = self.reference.edges[iedge]
    return Element( edge, self.transform << trans, self.__opposite and self.__opposite << trans ) if edge else None

  def getedge( self, trans ):
    iedge = self.reference.edge_transforms.index(trans)
    edge = self.reference.edge_refs[iedge]
    return edge and Element( edge, self.transform << trans, self.__opposite and self.__opposite << trans )

  @property
  def children( self ):
    return [ Element( child, self.transform << trans, self.__opposite and self.__opposite << trans )
      for trans, child in self.reference.children if child ]

  def trim( self, levelset, maxrefine, denom, check, fcache ):
    'trim element along levelset'

    assert not self.__opposite
    return self.reference.trim( (self.transform,levelset), maxrefine, denom, check, fcache )

  @property
  def flipped( self ):
    return Element( self.reference, self.__opposite, self.transform ) if self.__opposite else self

  @property
  def simplices( self ):
    return [ Element( reference, self.transform << trans, self.__opposite and self.__opposite << trans )
      for trans, reference in self.reference.simplices ]

  def __str__( self ):
    return 'Element({})'.format( self.vertices )


## REFERENCE ELEMENTS

class Reference( cache.Immutable ):
  'reference element'

  def __init__( self, vertices ):
    self.vertices = numpy.asarray( vertices )
    assert self.vertices.dtype == int
    self.nverts, self.ndims = self.vertices.shape

  def __or__( self, other ):
    if other is None:
      return self
    if other == self:
      return self
    return NotImplemented

  def __ror__( self, other ):
    return self.__or__( other )

  def __and__( self, other ):
    if other is None:
      return None
    if other == self:
      return self
    return NotImplemented

  def __rand__( self, other ):
    return self.__and__( other )

  def __xor__( self, other ):
    if other is None:
      return self
    if other == self:
      return None
    return NotImplemented

  def __rxor__( self, other ):
    return self.__xor__( other )

  def __sub__( self, other ):
    if other == self:
      return None
    if other is None:
      return self
    return NotImplemented

  def __rsub__( self, other ):
    if other == self:
      return None
    if other is None:
      raise TypeError
    return NotImplemented

  def __mul__( self, other ):
    assert isinstance( other, Reference )
    return other if self.ndims == 0 \
      else self if other.ndims == 0 \
      else TensorReference( self, other )

  def __pow__( self, n ):
    assert numeric.isint( n ) and n >= 0
    return PointReference() if n == 0 \
      else self if n == 1 \
      else self * self**(n-1)

  @property
  def nedges( self ):
    return len( self.edge_transforms )

  @property
  def nchildren( self ):
    return len( self.child_transforms )

  @property
  def edges( self ):
    return list( zip( self.edge_transforms, self.edge_refs ) )

  @property
  def children( self ):
    return list( zip( self.child_transforms, self.child_refs ) )

  @property
  def simplices( self ):
    return [ (transform.identity,self) ]

  @cache.property
  def childedgemap( self ):
    vmap = {}
    childedgemap = tuple( [None] * child.nedges for child in self.child_refs )
    for iedge, (etrans,edge) in enumerate(self.edges):
      for ichild, (ctrans,child) in enumerate(edge.children):
        v = tuple( sorted( (etrans<<ctrans).apply(child.vertices).totuple() ) )
        vmap[v] = ichild, iedge, True
    for ichild, (ctrans,child) in enumerate(self.children):
      for iedge, (etrans,edge) in enumerate(child.edges):
        v = tuple( sorted( (ctrans<<etrans).apply(edge.vertices).totuple() ) )
        try:
          jchild, jedge, isouter = childedgemap[ichild][iedge] = vmap.pop(v)
        except KeyError:
          vmap[v] = ichild, iedge, False
        else:
          trans = ( self.child_transforms[ichild] << self.child_refs[ichild].edge_transforms[iedge] ).flat
          if isouter:
            assert trans == ( self.edge_transforms[jedge] << self.edge_refs[jedge].child_transforms[jchild] ).flat
          else:
            assert trans == ( self.child_transforms[jchild] << self.child_refs[jchild].edge_transforms[jedge] ).flat.flipped
            childedgemap[jchild][jedge] = ichild, iedge, False
    assert not vmap
    return childedgemap

  @cache.property
  def edge2children( self ):
    edge2children = [ [ None ] * edge.nchildren for edge in self.edge_refs ]
    for ichild, row in enumerate(self.childedgemap):
      for iedge, (jchild,jedge,isouter) in enumerate( row ):
        if isouter:
          edge2children[jedge][jchild] = ichild, iedge
    assert all( all(items) for items in edge2children )
    return tuple( edge2children )

  @cache.property
  def edge2vertex( self ):
    edge2vertex = []
    for trans, edge in self.edges:
      where = numpy.zeros( self.nverts, dtype=bool )
      if edge:
        for v in trans.apply( edge.vertices ):
          where |= rational.equal( self.vertices, v ).all( axis=1 )
      edge2vertex.append( where )
    return numpy.array( edge2vertex )

  def getischeme( self, ischeme ):
    if self.ndims == 0:
      return numpy.zeros([1,0]), numpy.array([1.])
    match = re.match( '([a-zA-Z]+)(.*)', ischeme )
    assert match, 'cannot parse integration scheme %r' % ischeme
    ptype, args = match.groups()
    get = getattr( self, 'getischeme_'+ptype )
    return get( eval(args) ) if args else get()

  @classmethod
  def register( cls, ptype, func ):
    setattr( cls, 'getischeme_%s' % ptype, func )

  def with_children( self, child_refs, interfaces=[], check=False ):
    child_refs = tuple(child_refs)
    if not any( child_refs ):
      return None
    if child_refs == self.child_refs:
      return self
    return WithChildrenReference( self, child_refs, interfaces, check )

  def trim( self, levels, maxrefine, denom, check, fcache ):
    'trim element along levelset'

    assert maxrefine >= 0
    assert numeric.isint( denom )

    evaluated_levels = isinstance( levels, numpy.ndarray )
    if not evaluated_levels: # levelset is not evaluated
      trans, levelfun = levels
      try:
        levels = levelfun.eval( Element(self,trans), 'vertex%d' % maxrefine, fcache )
      except function.EvaluationError:
        pass
      else:
        evaluated_levels = True

    if evaluated_levels and ( levels > 0 ).all():
      posneg = self, None
      intrafaces = ()

    elif evaluated_levels and ( levels < 0 ).all():
      posneg = None, self
      intrafaces = ()

    elif maxrefine > 0:
      poselems = []
      negelems = []
      interfaces = set()
      intrafaces = set()
      for ichild, (ctrans,child) in enumerate( self.children ):
        if not child:
          poselems.append( None )
          negelems.append( None )
          continue
        if evaluated_levels:
          N, I = fcache( self.subvertex, ichild, maxrefine )
          assert len(levels) == N
          childlevels = levels[I]
        else:
          trans, levelfun = levels
          childlevels = trans << ctrans, levelfun
        (cposelem,cnegelem), cintrafaces = child.trim( childlevels, maxrefine-1, denom, check, fcache )
        poselems.append( cposelem )
        negelems.append( cnegelem )
        for iedge in cintrafaces:
          jchild, jedge, isouter = self.childedgemap[ichild][iedge]
          if isouter:
            intrafaces.add(jedge)
          else:
            interfaces.add((ichild,iedge,jchild,jedge) if ichild < jchild else (jchild,jedge,ichild,iedge))
      posneg = self.with_children( poselems, interfaces, check ), self.with_children( negelems, interfaces, check )

    else:
      assert evaluated_levels, 'failed to evaluate levelset up to level maxrefine'
      newverts, isectribs, oniface = mknewvtx( self.vertices, levels, self.ribbon2vertices, denom )
      edge2newvert = self.edge2vertex[:,self.ribbon2vertices[isectribs]].all( axis=2 )
      if ( levels[~oniface] >= 0 ).all():
        posneg = self, None
      elif ( levels[~oniface] <= 0 ).all():
        posneg = None, self
      else:
        vertices = rational.concatenate( [ self.vertices, newverts ] )
        newlevels = numpy.zeros( len(vertices) )
        newlevels[:self.nverts][~oniface] = levels[~oniface]
        triangulation, ispos = signed_triangulate( vertices, newlevels )
        edge2vertex = numpy.concatenate( [ self.edge2vertex, edge2newvert ], axis=1 )
        posneg = MultiSimplexReference( self, vertices, triangulation,  ispos, edge2vertex, check ), \
                 MultiSimplexReference( self, vertices, triangulation, ~ispos, edge2vertex, check )
      intrafaces, = numpy.where( self.edge2vertex[:,oniface].sum(1) + edge2newvert.sum(1) >= self.ndims )

    return posneg, tuple(intrafaces)

  def check_edges( self, decimal=10 ):
    x, w = self.getischeme( 'gauss1' )
    volume = w.sum()
    assert volume > 0
    check_volume = 0
    check_zero = 0
    for trans, edge in self.edges:
      if not edge:
        continue
      xe, we = edge.getischeme( 'gauss1' )
      w_normal = we[:,_] * rational.ext( trans.linear ).astype( float )
      if trans.isflipped:
        w_normal = -w_normal
      check_zero += w_normal.sum(0)
      check_volume += numeric.contract( trans.apply(xe), w_normal, axis=0 )
    numpy.testing.assert_almost_equal( check_zero, 0, decimal, '%s fails divergence test' % self )
    numpy.testing.assert_almost_equal( check_volume, volume, decimal, '%s fails divergence test' % self )

class SimplexReference( Reference ):
  'simplex reference'

  def __init__( self, ndims ):
    assert ndims >= 0
    vertices = numpy.concatenate( [ numpy.zeros(ndims,dtype=int)[_,:], numpy.eye(ndims,dtype=int) ], axis=0 )
    if ndims:
      self.edge_refs = (getsimplex(ndims-1),) * (ndims+1)
    Reference.__init__( self, vertices )

  @cache.property
  def ribbon2vertices( self ):
    return numpy.array([ (i,j) for i in range( self.ndims+1 ) for j in range( i+1, self.ndims+1 ) ])

  @property
  def edge2vertex( self ):
    return ~numpy.eye( self.nverts, dtype=bool )

  @property
  def child_refs( self ):
    return (self,) * len(self.child_transforms)

  def getischeme_vertex( self, n=0 ):
    if n == 0:
      return self.vertices.astype(float), None
    return self.getischeme_bezier( 2**n+1 )

  def __str__( self ):
    return self.__class__.__name__

  __repr__ = __str__

class PointReference( SimplexReference ):
  '0D simplex'

  def __init__( self ):
    SimplexReference.__init__( self, 0 )
    self.child_transforms = transform.identity,

  def getischeme( self, ischeme ):
    return numpy.zeros((1,0)), numpy.ones(1)

class LineReference( SimplexReference ):
  '1D simplex'

  def __init__( self ):
    self._bernsteincache = [] # TEMPORARY
    SimplexReference.__init__( self, 1 )
    self.edge_transforms = transform.simplex( self.vertices[1:], isflipped=False ), transform.simplex( self.vertices[:1], isflipped=True )
    self.child_transforms = transform.affine(1,[0],2), transform.affine(1,[1],2)
    self.check_edges()

  def stdfunc( self, degree ):
    if len(self._bernsteincache) <= degree or self._bernsteincache[degree] is None:
      self._bernsteincache += [None] * (degree-len(self._bernsteincache))
      self._bernsteincache.append( PolyLine( PolyLine.bernstein_poly(degree) ) )
    return self._bernsteincache[degree]

  def getischeme_gauss( self, degree ):
    assert isinstance( degree, int ) and degree >= 0
    x, w = gauss( degree )
    return x[:,_], w

  def getischeme_uniform( self, n ):
    return numpy.arange( .5, n )[:,_] / n, numeric.appendaxes( 1./n, n )

  def getischeme_bezier( self, np ):
    return numpy.linspace( 0, 1, np )[:,_], None

  def subvertex( self, ichild, i ):
    if i == 0:
      assert ichild == 0
      return self.nverts, numpy.arange(self.nverts)
    assert 0 <= ichild < 2
    n = 2**i+1
    return n, numpy.arange(n//2+1) if ichild == 0 else numpy.arange(n//2,n)

class TriangleReference( SimplexReference ):
  '2D simplex'

  def __init__( self ):
    SimplexReference.__init__( self, 2 )
    self.edge_transforms = transform.simplex( self.vertices[1:], isflipped=False ), transform.simplex( self.vertices[::-2], isflipped=False ), transform.simplex( self.vertices[:-1], isflipped=False )
    self.child_transforms = transform.affine(1,[0,0],2), transform.affine(1,[0,1],2), transform.affine(1,[1,0],2), transform.affine([[0,-1],[-1,0]],[1,1],2,isflipped=True )
    self.check_edges()

  def stdfunc( self, degree ):
    return PolyTriangle(degree)

  def getischeme_contour( self, n ):
    p = numpy.arange( n+1, dtype=float ) / (n+1)
    z = numpy.zeros_like( p )
    return numpy.hstack(( [1-p,p], [z,1-p], [p,z] )).T, None

  def getischeme_vtk( self ):
    return self.vertices.astype(float), None

  def getischeme_gauss( self, degree ):
    '''get integration scheme
    http://www.cs.rpi.edu/~flaherje/pdf/fea6.pdf'''
    if isinstance( degree, tuple ):
      assert len(degree) == self.ndims
      degree = sum(degree)
    assert isinstance( degree, int ) and degree >= 0

    I = [0,0],
    J = [1,1],[0,1],[1,0]
    K = [1,2],[2,0],[0,1],[2,1],[1,0],[0,2]

    icw = [
      ( I, [1/3], 1 )
    ] if degree == 1 else [
      ( J, [2/3,1/6], 1/3 )
    ] if degree == 2 else [
      ( I, [1/3], -9/16 ),
      ( J, [3/5,1/5], 25/48 ),
    ] if degree == 3 else [
      ( J, [0.816847572980458,0.091576213509771], 0.109951743655322 ),
      ( J, [0.108103018168070,0.445948490915965], 0.223381589678011 ),
    ] if degree == 4 else [
      ( I, [1/3], 0.225 ),
      ( J, [0.797426985353088,0.101286507323456], 0.125939180544827 ),
      ( J, [0.059715871789770,0.470142064105115], 0.132394152788506 ),
    ] if degree == 5 else [
      ( J, [0.873821971016996,0.063089014491502], 0.050844906370207 ),
      ( J, [0.501426509658180,0.249286745170910], 0.116786275726379 ),
      ( K, [0.636502499121399,0.310352451033785,0.053145049844816], 0.082851075618374 ),
    ] if degree == 6 else [
      ( I, [1/3.], -0.149570044467671 ),
      ( J, [0.479308067841924,0.260345966079038], 0.175615257433204 ),
      ( J, [0.869739794195568,0.065130102902216], 0.053347235608839 ),
      ( K, [0.638444188569809,0.312865496004875,0.048690315425316], 0.077113760890257 ),
    ]

    if degree > 7:
      warnings.warn( 'inexact integration for polynomial of degree %i'.format(degree) )

    return numpy.concatenate( [ numpy.take(c,i) for i, c, w in icw ], axis=0 ), \
           numpy.concatenate( [ [w/2] * len(i) for i, c, w in icw ] )

  def getischeme_uniform( self, n ):
    points = numpy.arange( 1./3, n ) / n
    nn = n**2
    C = numpy.empty( [2,n,n] )
    C[0] = points[:,_]
    C[1] = points[_,:]
    coords = C.reshape( 2, nn )
    flip = coords.sum(0) > 1
    coords[:,flip] = 1 - coords[::-1,flip]
    weights = numeric.appendaxes( .5/nn, nn )
    return coords.T, weights

  def getischeme_bezier( self, np ):
    points = numpy.linspace( 0, 1, np )
    return numpy.array([ [x,y] for i, y in enumerate(points) for x in points[:np-i] ]), None

  def subvertex( self, ichild, irefine ):
    if irefine == 0:
      assert ichild == 0
      return self.nverts, numpy.arange(self.nverts)

    N = 1 + 2**irefine # points along parent edge
    n = 1 + 2**(irefine-1) # points along child edge

    flatten_parent = lambda i, j: j + i*N - (i*(i-1))//2

    if ichild == 0: # lower left
      flatten_child = lambda i, j: flatten_parent( i, j )
    elif ichild == 1: # upper left
      flatten_child = lambda i, j: flatten_parent( n-1+i, j )
    elif ichild == 2: # lower right
      flatten_child = lambda i, j: flatten_parent( i, n-1+j )
    elif ichild == 3: # inverted
      flatten_child = lambda i, j: flatten_parent( n-1-j, n-1-i )
    else:
      raise Exception( 'invalid ichild: {}'.format( ichild ) )

    return ((N+1)*N)//2, numpy.concatenate([ flatten_child(i,numpy.arange(n-i)) for i in range(n) ])

class TetrahedronReference( SimplexReference ):
  '3D simplex'

  def __init__( self ):
    SimplexReference.__init__( self, 3 )
    self.edge_transforms = tuple( transform.simplex( self.vertices[I], isflipped=False ) for I in [[1,2,3],[0,3,2],[3,0,1],[2,1,0]] )
    self.check_edges()

  def getischeme_vtk( self ):
    return self.vertices.astype(float), None

  def getischeme_gauss( self, degree ):
    '''get integration scheme
    http://www.cs.rpi.edu/~flaherje/pdf/fea6.pdf'''
    if isinstance( degree, tuple ):
      assert len(degree) == 3
      degree = sum(degree)
    assert isinstance( degree, int ) and degree >= 0

    I = [0,0,0],
    J = [1,1,1],[0,1,1],[1,1,0],[1,0,1]
    K = [0,1,1],[1,0,1],[1,1,0],[1,0,0],[0,1,0],[0,0,1]
    L = [0,1,1],[1,0,1],[1,1,0],[2,1,1],[1,2,1],[1,1,2],[1,0,2],[0,2,1],[2,1,0],[1,2,0],[0,1,2],[2,0,1]

    icw = [
      ( I, [1/4], 1 ),
    ] if degree == 1 else [
      ( J, [0.5854101966249685,0.1381966011250105], 1/4 ),
    ] if degree == 2 else [
      ( I, [.25], -.8 ),
      ( J, [.5,1/6], .45 ),
    ] if degree == 3 else [
      ( I, [.25], -.2368/3 ),
      ( J, [0.7857142857142857,0.0714285714285714], .1372/3 ),
      ( K, [0.1005964238332008,0.3994035761667992], .448/3 ),
    ] if degree == 4 else [
      ( I, [.25], 0.1817020685825351 ),
      ( J, [0,1/3.], 0.0361607142857143 ),
      ( J, [8/11.,1/11.], 0.0698714945161738 ),
      ( K, [0.4334498464263357,0.0665501535736643], 0.0656948493683187 ),
    ] if degree == 5 else [
      ( J, [0.3561913862225449,0.2146028712591517], 0.0399227502581679 ),
      ( J, [0.8779781243961660,0.0406739585346113], 0.0100772110553207 ),
      ( J, [0.0329863295731731,0.3223378901422757], 0.0553571815436544 ),
      ( L, [0.2696723314583159,0.0636610018750175,0.6030056647916491], 0.0482142857142857 ),
    ] if degree == 6 else [
      ( I, [.25], 0.1095853407966528 ),
      ( J, [0.7653604230090441,0.0782131923303186],  0.0635996491464850 ),
      ( J, [0.6344703500082868,0.1218432166639044], -0.3751064406859797 ),
      ( J, [0.0023825066607383,0.3325391644464206],  0.0293485515784412 ),
      ( K, [0,.5], 0.0058201058201058 ),
      ( L, [.2,.1,.6], 0.1653439153439105 )
    ] if degree == 7 else [
      ( I, [.25], -0.2359620398477557),
      ( J, [0.6175871903000830,0.1274709365666390], 0.0244878963560562),
      ( J, [0.9037635088221031,0.0320788303926323], 0.0039485206398261),
      ( K, [0.4502229043567190,0.0497770956432810], 0.0263055529507371),
      ( K, [0.3162695526014501,0.1837304473985499], 0.0829803830550589),
      ( L, [0.0229177878448171,0.2319010893971509,0.5132800333608811], 0.0254426245481023),
      ( L, [0.7303134278075384,0.0379700484718286,0.1937464752488044], 0.0134324384376852),
    ]

    if degree > 8:
      warnings.warn( 'inexact integration for polynomial of degree %i'.format(degree) )

    return numpy.concatenate( [ numpy.take(c,i) for i, c, w in icw ], axis=0 ), \
           numpy.concatenate( [ [w/6] * len(i) for i, c, w in icw ] )

class TensorReference( Reference ):
  'tensor reference'

  _re_ischeme = re.compile( '([a-zA-Z]+)(.*)' )

  def __init__( self, ref1, ref2 ):
    self.ref1 = ref1
    self.ref2 = ref2
    ndims = ref1.ndims + ref2.ndims
    vertices = numpy.empty( ( ref1.nverts, ref2.nverts, ndims ), dtype=int )
    vertices[:,:,:ref1.ndims] = ref1.vertices[:,_]
    vertices[:,:,ref1.ndims:] = ref2.vertices[_,:]
    Reference.__init__( self, vertices.reshape(-1,ndims) )
    self.check_edges()

  def subvertex( self, ichild, i ):
    ichild1, ichild2 = divmod( ichild, len(self.ref2.child_transforms) )
    N1, I1 = self.ref1.subvertex( ichild1, i )
    N2, I2 = self.ref2.subvertex( ichild2, i )
    return N1 * N2, ( N2 * I1[:,_] + I2[_,:] ).ravel()

  def stdfunc( self, degree, *n ):
    if n:
      degree = (degree,)+n
      assert len(degree) == self.ndims
      degree1 = degree[:self.ref1.ndims]
      degree2 = degree[self.ref1.ndims:]
    else:
      degree1 = degree2 = degree
    return self.ref1.stdfunc( degree1 ) \
         * self.ref2.stdfunc( degree2 )

  @cache.property
  def ribbon2vertices( self ):
    r2v1 = self.ref1.ribbon2vertices[:,_,:] + self.ref1.nverts * numpy.arange(self.ref2.nverts)[_,:,_]
    r2v2 = self.ref2.ribbon2vertices[:,_,:] * self.ref1.nverts + numpy.arange(self.ref1.nverts)[_,:,_]
    return numpy.concatenate([ r2v1.reshape(-1,2), r2v2.reshape(-1,2), ], axis=0 )

  def __str__( self ):
    return '%s*%s' % ( self.ref1, self.ref2 )

  def stdfunc( self, degree ):
    return self.ref1.stdfunc(degree) * self.ref2.stdfunc(degree)

  def getischeme_vtk( self ):
    if self == LineReference()**2:
      points = [[0,0],[1,0],[1,1],[0,1]]
    elif self == LineReference()**3:
      points = [[0,0,0],[1,0,0],[0,1,0],[1,1,0],[0,0,1],[1,0,1],[0,1,1],[1,1,1]]
    else:
      raise NotImplementedError
    return numpy.array(points,dtype=float), numpy.ones(self.nverts,dtype=float)

  def getischeme_contour( self, n ):
    assert self == LineReference()**2
    p = numpy.arange( n+1, dtype=float ) / (n+1)
    z = numpy.zeros_like( p )
    return numpy.hstack(( [p,z], [1-z,p], [1-p,1-z], [z,1-p] )).T, None

  def getischeme( self, ischeme ):
    if '*' in ischeme:
      ischeme1, ischeme2 = ischeme.split( '*', 1 )
    else:
      match = self._re_ischeme.match( ischeme )
      assert match, 'cannot parse integration scheme %r' % ischeme
      ptype, args = match.groups()
      get = getattr( self, 'getischeme_'+ptype, None )
      if get:
        return get( eval(args) ) if args else get()
      if args and ',' in args:
        args = eval(args)
        assert len(args) == self.ndims
        ischeme1 = ptype+','.join( str(n) for n in args[:self.ref1.ndims] )
        ischeme2 = ptype+','.join( str(n) for n in args[self.ref1.ndims:] )
      else:
        ischeme1 = ischeme2 = ischeme
    ipoints1, iweights1 = self.ref1.getischeme( ischeme1 )
    ipoints2, iweights2 = self.ref2.getischeme( ischeme2 )
    ipoints = numpy.empty( (ipoints1.shape[0],ipoints2.shape[0],self.ndims) )
    ipoints[:,:,0:self.ref1.ndims] = ipoints1[:,_,:self.ref1.ndims]
    ipoints[:,:,self.ref1.ndims:self.ndims] = ipoints2[_,:,:self.ref2.ndims]
    iweights = ( iweights1[:,_] * iweights2[_,:] ).ravel() if iweights1 is not None and iweights2 is not None else None
    return ipoints.reshape( -1, self.ndims ), iweights

  @cache.property
  def edge_transforms( self ):
    return tuple(
      [ transform.affine(
        rational.blockdiag([ trans1.linear, rational.eye(self.ref2.ndims) ]),
        rational.concatenate([ trans1.offset, rational.zeros(self.ref2.ndims) ]),
        isflipped=trans1.isflipped )
          for trans1 in self.ref1.edge_transforms ]
   + [ transform.affine(
        rational.blockdiag([ rational.eye(self.ref1.ndims), trans2.linear ]),
        rational.concatenate([ rational.zeros(self.ref1.ndims), trans2.offset ]),
        isflipped=trans2.isflipped if self.ref1.ndims%2==0 else not trans2.isflipped )
          for trans2 in self.ref2.edge_transforms ])

  @property
  def edge_refs( self ):
    return tuple([ edge1 * self.ref2 for edge1 in self.ref1.edge_refs ]
               + [ self.ref1 * edge2 for edge2 in self.ref2.edge_refs ])

  @cache.property
  def child_transforms( self ):
    return [ transform.affine( trans1.linear if trans1.linear.ndim == 0 and trans1.linear == trans2.linear
                          else rational.blockdiag([ trans1.linear, trans2.linear ]),
                               rational.concatenate([ trans1.offset, trans2.offset ]) )
            for trans1 in self.ref1.child_transforms
              for trans2 in self.ref2.child_transforms ]

  @property
  def child_refs( self ):
    return tuple( child1 * child2 for child1 in self.ref1.child_refs for child2 in self.ref2.child_refs )

class NeighborhoodTensorReference( TensorReference ):
  'product reference element'

  def __init__( self, ref1, ref2, neighborhood, transf ):
    '''Neighborhood of elem1 and elem2 and transformations to get mutual
    overlap in right location. Returns 3-element tuple:
    * neighborhood, as given by Element.neighbor(),
    * transf1, required rotation of elem1 map: {0:0, 1:pi/2, 2:pi, 3:3*pi/2},
    * transf2, required rotation of elem2 map (is indep of transf1 in Topology.'''

    TensorReference.__init__( self, ref1, ref2 )
    self.neighborhood = neighborhood
    self.transf = transf

  def singular_ischeme_quad( self, points ):
    transfpoints = numpy.empty( points.shape )
    def transform( points, transf ):
      x, y = points[:,0], points[:,1]
      tx = x if transf in (0,1,6,7) else 1-x
      ty = y if transf in (0,3,4,7) else 1-y
      return function.stack( (ty, tx) if transf%2 else (tx, ty), axis=1 )
    transfpoints[:,:2] = transform( points[:,:2], self.transf[0] )
    transfpoints[:,2:] = transform( points[:,2:], self.transf[1] )
    return transfpoints

  def get_tri_bem_ischeme( self, ischeme ):
    'Some cached quantities for the singularity quadrature scheme.'
    points, weights = (LineReference()**4).getischeme( ischeme )
    eta1, eta2, eta3, xi = points.T
    if self.neighborhood == 0:
      temp = xi*eta1*eta2*eta3
      pts0 = xi*eta1*(1 - eta2)
      pts1 = xi - pts0
      pts2 = xi - temp
      pts3 = xi*(1 - eta1)
      pts4 = pts0 + temp
      pts5 = xi*(1 - eta1*eta2)
      pts6 = xi*eta1 - temp
      points = numpy.array(
        [[1-xi,   1-pts2, 1-xi,   1-pts5, 1-pts2, 1-xi  ],
         [pts1, pts3, pts4, pts0, pts6, pts0],
         [1-pts2, 1-xi,   1-pts5, 1-xi,   1-xi,   1-pts2],
         [pts3, pts1, pts0, pts4, pts0, pts6]]).reshape( 4, -1 ).T
      points = points * [-1,1,-1,1] + [1,0,1,0] # flipping in x -GJ
      weights = numpy.concatenate( 6*[xi**3*eta1**2*eta2*weights] )
    elif self.neighborhood == 1:
      A = xi*eta1
      B = A*eta2
      C = A*eta3
      D = B*eta3
      E = xi - B
      F = A - B
      G = xi - D
      H = B - D
      I = A - D
      points = numpy.array(
        [[1-xi, 1-xi, 1-E,  1-G,  1-G ],
         [C,  G,  F,  H,  I ],
         [1-E,  1-G,  1-xi, 1-xi, 1-xi],
         [F,  H,  D,  A,  B ]] ).reshape( 4, -1 ).T
      temp = xi*A
      weights = numpy.concatenate( [A*temp*weights] + 4*[B*temp*weights] )
    elif self.neighborhood == 2:
      A = xi*eta2
      B = A*eta3
      C = xi*eta1
      points = numpy.array(
        [[1-xi, 1-A ],
         [C,  B ],
         [1-A,  1-xi],
         [B,  C ]] ).reshape( 4, -1 ).T
      weights = numpy.concatenate( 2*[xi**2*A*weights] )
    else:
      assert self.neighborhood == -1, 'invalid neighborhood %r' % self.neighborhood
      points = numpy.array([ eta1*eta2, 1-eta2, eta3*xi, 1-xi ]).T
      weights = eta2*xi*weights
    return points, weights

  def get_quad_bem_ischeme( self, ischeme ):
    'Some cached quantities for the singularity quadrature scheme.'
    quad = LineReference()**4
    points, weights = quad.getischeme( ischeme )
    eta1, eta2, eta3, xi = points.T
    if self.neighborhood == 0:
      xe = xi*eta1
      A = (1 - xi)*eta3
      B = (1 - xe)*eta2
      C = xi + A
      D = xe + B
      points = numpy.array(
        [[A, B, A, D, B, C, C, D],
         [B, A, D, A, C, B, D, C],
         [C, D, C, B, D, A, A, B],
         [D, C, B, C, A, D, B, A]]).reshape( 4, -1 ).T
      weights = numpy.concatenate( 8*[xi*(1-xi)*(1-xe)*weights] )
    elif self.neighborhood == 1:
      ox = 1 - xi
      A = xi*eta1
      B = xi*eta2
      C = ox*eta3
      D = C + xi
      E = 1 - A
      F = E*eta3
      G = A + F
      points = numpy.array(
        [[D,  C,  G,  G,  F,  F ],
         [B,  B,  B,  xi, B,  xi],
         [C,  D,  F,  F,  G,  G ],
         [A,  A,  xi, B,  xi, B ]]).reshape( 4, -1 ).T
      weights = numpy.concatenate( 2*[xi**2*ox*weights] + 4*[xi**2*E*weights] )
    elif self.neighborhood == 2:
      A = xi*eta1
      B = xi*eta2
      C = xi*eta3
      points = numpy.array(
        [[xi, A,  A,  A ], 
         [A,  xi, B,  B ],
         [B,  B,  xi, C ], 
         [C,  C,  C,  xi]]).reshape( 4, -1 ).T
      weights = numpy.concatenate( 4*[xi**3*weights] )
    else:
      assert self.neighborhood == -1, 'invalid neighborhood %r' % self.neighborhood
    return points, weights

  def getischeme_singular( self, n ):
    'get integration scheme'
    
    gauss = 'gauss%d'% (n*2-2)
    assert self.ref1 == self.ref2 == LineReference()**2
    points, weights = self.get_quad_bem_ischeme( gauss )
    return self.singular_ischeme_quad( points ), weights

class WrappedReference( Reference ):
  'derive properties from baseref'

  def __init__( self, baseref ):
    assert isinstance( baseref, Reference )
    self.baseref = baseref
    Reference.__init__( self, baseref.vertices )

  def getischeme( self, ischeme ):
    return self.baseref.getischeme( ischeme )

  def stdfunc( self, degree ):
    return self.baseref.stdfunc( degree )

  @property
  def subvertex( self ):
    return self.baseref.subvertex

  def __or__( self, other ):
    return self if other is None \
      else self.baseref if other == self.baseref \
      else self._logical( other, lambda ref1, ref2: ref1 | ref2 )

  def __and__( self, other ):
    return None if other is None \
      else self if other == self.baseref \
      else self._logical( other, lambda ref1, ref2: ref1 & ref2 )

  def __xor__( self, other ):
    return self if other is None \
      else ~self if other == self.baseref \
      else self._logical( other, lambda ref1, ref2: ref1 ^ ref2 )

  def __sub__( self, other ):
    if other.__class__ == self.__class__ and other.baseref == self:
      return ~other
    return Reference.__sub__( self, other )

  def __rsub__( self, other ):
    if other != self.baseref:
      return NotImplemented
    return ~self

class ForwardReference( WrappedReference ):
  'forward self as child'

  def __init__( self, baseref ):
    self.child_refs = baseref,
    self.child_transforms = transform.identity,
    WrappedReference.__init__( self, baseref )

  @property
  def childedgemap( self ):
    return self.baseref.childedgemap

  @property
  def edge2children( self ):
    return self.baseref.edge2children

class WithChildrenReference( WrappedReference ):
  'base reference with explicit children'

  def __init__( self, baseref, child_refs, interfaces=[], check=False ):
    assert isinstance( child_refs, tuple ) and len(child_refs) == baseref.nchildren and any(child_refs) and child_refs != baseref.child_refs
    self.child_transforms = baseref.child_transforms
    self.child_refs = child_refs
    WrappedReference.__init__( self, baseref )
    self.mkinterfaces( interfaces )
    if check:
      self.check_edges()

  def mkinterfaces( self, interfaces ):
    self.__interfaces = []
    for ichild, iedge, jchild, jedge in interfaces:
      refi = self.child_refs[ichild] and self.child_refs[ichild].edge_refs[iedge]
      refj = self.child_refs[jchild] and self.child_refs[jchild].edge_refs[jedge]
      ref = refi^refj if refi or refj else None
      if not ref:
        continue

      transi = self.baseref.child_transforms[ichild] << self.baseref.child_refs[ichild].edge_transforms[iedge]
      transj = self.baseref.child_transforms[jchild] << self.baseref.child_refs[jchild].edge_transforms[jedge]

      if ref == (ref & refi):
        assert not (ref & refj)
        self.__interfaces.append(( ichild, iedge, transi, transj, ref ))
      elif ref == (ref & refj):
        assert not (ref & refi)
        self.__interfaces.append(( jchild, jedge, transj, transi, ref ))
      else:
        raise NotImplementedError

  def __invert__( self ):
    return self._logical( self.baseref, lambda ref1, ref2: ref2 - ref1 )

  def _logical( self, other, op ):
    return self.baseref.with_children( op(child1,child2) if child1 or child2 else None for child1, child2 in zip( self.child_refs, other.child_refs ) )

  def getischeme( self, ischeme ):
    'get integration scheme'
    
    if ischeme.startswith('vertex'):
      return self.baseref.getischeme( ischeme )

    allcoords = []
    allweights = []
    for trans, simplex in self.children:
      if simplex:
        points, weights = simplex.getischeme( ischeme )
        allcoords.append( trans.apply(points) )
        if weights is not None:
          allweights.append( weights * abs(float(trans.det)) )

    coords = numpy.concatenate( allcoords, axis=0 )
    weights = numpy.concatenate( allweights, axis=0 ) \
      if len(allweights) == len(allcoords) else None

    return coords, weights

  @property
  def simplices( self ):
    return [ (trans2<<trans1, simplex) for trans2, child in self.children for trans1, simplex in (child.simplices if child else []) ]

  @cache.property
  def edge_transforms( self ):
    edge_transforms = list( self.baseref.edge_transforms )
    for trans, mychild, basechild in zip( self.child_transforms, self.child_refs, self.baseref.child_refs ):
      if mychild:
        edge_transforms.extend( trans << etrans for etrans in mychild.edge_transforms[basechild.nedges:] )
    edge_transforms.extend( trans for ichild, iedge, trans, opptrans, ref in self.__interfaces )
    return tuple(edge_transforms)

  @cache.property
  def edge_refs( self ):
    # to avoid circular references we cannot mention 'self' inside getedgeref
    def getedgeref( iedge, baseref=self.baseref, child_refs=self.child_refs, edge2children=self.edge2children ):
      baseedge = baseref.edge_refs[iedge]
      return baseedge and baseedge.with_children( child_refs[jchild] and child_refs[jchild].edge_refs[jedge] for jchild, jedge in edge2children[iedge] )
    items = [ cache.Tuple.unknown ] * self.baseref.nedges
    for mychild, basechild in zip( self.child_refs, self.baseref.child_refs ):
      if mychild:
        items.extend( ForwardReference(edge) for edge in mychild.edge_refs[basechild.nedges:] )
    items.extend( ForwardReference(ref) for ichild, iedge, trans, opptrans, ref in self.__interfaces )
    return cache.Tuple( items, getedgeref )

  @property
  def childedgemap( self ):
    childedgemap = tuple( list(row) for row in self.baseref.childedgemap )
    for ichild, mychild in enumerate( self.child_refs ):
      if mychild:
        basechild = self.baseref.child_refs[ichild]
        childedgemap[ichild].extend( (ichild,iedge,False) for iedge in range(basechild.nedges,mychild.nedges) )
    return childedgemap

  @cache.property
  def edge2children( self ):
    edge2children = list( self.baseref.edge2children )
    for ichild, mychild in enumerate( self.child_refs ):
      if mychild:
        basechild = self.baseref.child_refs[ichild]
        edge2children.extend( [(ichild,iedge)] for iedge in range(basechild.nedges,mychild.nedges) )
    edge2children.extend( [(ichild,iedge)] for ichild, iedge, trans, opptrans, ref in self.__interfaces )
    return tuple( edge2children )

class MultiSimplexReference( WrappedReference ):
  'triangulation'

  def __init__( self, baseref, coords, triangulation, ismine, edge2coords=None, check=False ):
    assert coords.shape[1] == baseref.ndims
    self.coords = coords
    assert isinstance( triangulation, numpy.ndarray ) and triangulation.shape[1] == baseref.ndims+1
    self.triangulation = triangulation
    assert isinstance( ismine, numpy.ndarray ) and len(ismine) == len(triangulation) and ismine.any()
    self.ismine = ismine
    self.__transforms = tuple( transform.simplex(self.coords[tri]) for tri in triangulation[ismine] )
    self.__areas = numpy.array([ float(trans.det) for trans in self.__transforms ])
    assert numpy.all( self.__areas > 0 )
    assert edge2coords is None or edge2coords.shape == ( baseref.nedges, len(coords) )
    self.__edge2coords = edge2coords
    WrappedReference.__init__( self, baseref )
    if check:
      self.check_edges()

  @cache.property
  def interfaces( self ):
    edgemasks = numeric.overlapping( numpy.arange(1,2*self.ndims+1)%(self.ndims+1), n=self.ndims )
    interfaces = {}
    for itri, tri in enumerate( self.triangulation[self.ismine] ):
      etris = tri[edgemasks]
      iedges, = numpy.where( ~(self.__edge2coords[:,etris].all(axis=2).any(axis=0)) )
      for iedge in iedges:
        key = tuple(sorted(etris[iedge]))
        try:
          interfaces.pop( key )
        except KeyError:
          interfaces[key] = itri, iedge
    return tuple( interfaces.values() )

  @cache.property
  def edge_transforms( self ):
    simplex = getsimplex( self.ndims )
    interfaces = tuple( self.__transforms[itri] << simplex.edge_transforms[iedge] for itri, iedge in self.interfaces )
    return self.baseref.edge_transforms + interfaces

  @cache.property
  def edge_refs( self ):
    # to avoid circular references we cannot mention 'self' inside getedgeref
    def getedgeref( iedge,
        baseref = self.baseref,
        coords = self.coords,
        triangulation = self.triangulation,
        ismine = self.ismine,
        edge2vertex = self.__edge2coords ):
      onedge = edge2vertex[iedge]
      used_triangles = numpy.array([ onedge[tri].sum() == baseref.ndims for tri in triangulation ])
      ismineused = ismine[used_triangles]
      if not ismineused.any():
        return None
      baseedge = baseref.edge_refs[iedge]
      if ismineused.all():
        return baseedge
      etrans = baseref.edge_transforms[iedge]
      used_coords = numpy.zeros( len(coords), dtype=bool )
      etri = []
      for tri in triangulation[used_triangles]:
        w = onedge[tri]
        t = tri[w]
        if w[int(etrans.isflipped)::2].all(): # even/odd point is removed
          t[-2:] = t[-1], t[-2] # flip simplex
        used_coords[t] = True
        etri.append( t )
      ecoords = etrans.solve( coords[used_coords] )
      triangulation = normtri( numpy.take( used_coords.cumsum()-1, etri ) )
      renumber = arglexsort( triangulation )
      return MultiSimplexReference( baseedge, ecoords, triangulation[renumber], ismineused[renumber] )
    items = [ cache.Tuple.unknown ] * self.baseref.nedges + [ getsimplex(self.ndims-1) ] * len(self.interfaces)
    return cache.Tuple( items, getedgeref )

  def __invert__( self ):
    return MultiSimplexReference( self.baseref, self.coords, self.triangulation, ~self.ismine, self.__edge2coords )

  def _logical( self, other, op ):
    if not isinstance( other, MultiSimplexReference ) or other.baseref != self.baseref \
         or len(other.coords) != len(self.coords) or ( other.coords != self.coords ).any() \
         or len(other.triangulation) != len(self.triangulation) or ( other.triangulation != self.triangulation ).any():
      return NotImplemented
    ismine = op( self.ismine, other.ismine )
    return None if not ismine.any() \
      else self.baseref if ismine.all() \
      else MultiSimplexReference( self.baseref, self.coords, self.triangulation, ismine, self.__edge2coords )

  @property
  def simplices( self ):
    simplex = getsimplex(self.ndims)
    return [ (trans,simplex) for trans in self.__transforms ]

  def getischeme( self, ischeme ):
    'get integration scheme'
    
    if ischeme.startswith('vertex'):
      return self.baseref.getischeme( ischeme )

    points, weights = getsimplex(self.ndims).getischeme( ischeme )
    allpoints = numpy.array([ trans.apply(points) for trans in self.__transforms ]).reshape(-1,self.ndims)
    allweights = ( self.__areas[:,_] * weights ).ravel() if weights is not None else None
    return allpoints, allweights

  @cache.property
  def edge2coords( self ):
    ifaceedge2vertex = numpy.zeros( (len(self.interfaces),len(self.coords)), dtype=bool )
    arange = numpy.arange(self.ndims+1)
    mytriangulation = self.triangulation[self.ismine]
    for mask, (itri,iedge) in zip( ifaceedge2vertex, self.interfaces ):
      mask[mytriangulation[itri][arange!=iedge]] = True
    return numpy.concatenate( [ self.__edge2coords, ifaceedge2vertex ], axis=0 )

  def trim( self, levels, maxrefine, denom, check, fcache ):
    if not isinstance( levels, numpy.ndarray ):
      trans, levelfun = levels
      levels = levelfun.eval( Element(self,trans), 'vertex%d' % maxrefine, fcache )

    mytriangulation = self.triangulation[self.ismine]
    used = numpy.zeros( len(self.coords), dtype=bool )
    used[mytriangulation] = True
    mytriangulation = (used.cumsum()-1)[mytriangulation]
    edge2coords = self.edge2coords[:,used]
    coords = self.coords[used]
    coordlevels = self.stdfunc(1).eval( coords.astype(float) ).dot( levels )
    simplexribbons = numpy.array([ [i,j] for i in range(self.ndims) for j in range(i+1,self.ndims+1) ])
    allribbons = [ tuple(ribbon) for tri in mytriangulation for ribbon in numpy.sort( tri[simplexribbons], axis=1 ) ]
    uniqueribbons = []
    tri2ribs = numpy.array([ index_or_append(uniqueribbons,item) for item in allribbons ]).reshape( len(mytriangulation), len(simplexribbons) )
    newverts, isectribs, oniface = mknewvtx( coords, coordlevels, uniqueribbons, denom )
    edge2newcoord = edge2coords[:,numpy.array(uniqueribbons)[isectribs]].all( axis=2 )
    if ( coordlevels[~oniface] >= 0 ).all():
      posneg = self, None
    elif ( coordlevels[~oniface] <= 0 ).all():
      posneg = None, self
    else:
      vertices = rational.concatenate( [ coords, newverts ], axis=0 )
      newlevels = numpy.zeros( len(vertices) )
      newlevels[:len(coords)][~oniface] = coordlevels[~oniface]
      triangulation = []
      allispos = []
      for itri, tri in enumerate( mytriangulation ):
        ribs = tri2ribs[itri]
        trinewverts = [ len(coords) + inewvert for inewvert, irib in enumerate(isectribs) if irib in ribs ]
        if trinewverts:
          pts = numpy.concatenate([ tri, trinewverts ])
          newtri, ispos = signed_triangulate( vertices[pts], newlevels[pts] )
          triangulation.extend( normtri(pts[newtri]) )
          allispos.extend( ispos )
        else:
          ispos = ( newlevels[tri] >= 0 ).all()
          isneg = ( newlevels[tri] <= 0 ).all()
          assert ispos != isneg
          triangulation.append( tri )
          allispos.append( ispos )
      renumber = arglexsort( triangulation )
      triangulation = numpy.array(triangulation)[renumber]
      allispos = numpy.array(allispos)[renumber]
      thisedge2vertex = numpy.concatenate( [ edge2coords, edge2newcoord ], axis=1 )
      posneg = MultiSimplexReference( self, vertices, triangulation,  allispos, thisedge2vertex, check ), \
               MultiSimplexReference( self, vertices, triangulation, ~allispos, thisedge2vertex, check )
    intrafaces, = numpy.where( edge2coords[:,oniface].sum(1) + edge2newcoord.sum(1) >= self.ndims )
    return posneg, intrafaces

  @property
  def childedgemap( self ):
    raise NotImplementedError

  @property
  def edge2children( self ):
    raise NotImplementedError


# SHAPE FUNCTIONS

class StdElem( cache.Immutable ):
  'stdelem base class'

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

  __slots__ = 'std1', 'std2'

  def __init__( self, std1, std2 ):
    'constructor'

    self.std1 = std1
    self.std2 = std2
    StdElem.__init__( self, std1.ndims+std2.ndims, std1.nshapes*std2.nshapes )

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
    N = numpy.newaxis,

    shape = points.shape[:-1] + (self.std1.nshapes * self.std2.nshapes,)
    G12 = [ ( self.std1.eval( p1, grad=i )[E+S+N+S*i+N*j]
            * self.std2.eval( p2, grad=j )[E+N+S+N*i+S*j] ).reshape( shape + (self.std1.ndims,) * i + (self.std2.ndims,) * j )
            for i,j in zip( range(grad,-1,-1), range(grad+1) ) ]

    data = numpy.empty( shape + (self.ndims,) * grad )

    s = (s1,)*grad + (s2,)*grad
    R = numpy.arange(grad)
    for n in range(2**grad):
      index = n>>R&1
      n = index.argsort() # index[s] = [0,...,1]
      shuffle = list( range(points.ndim) ) + list( points.ndim + n )
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
    revpoly = numpy.zeros( [degree+1,degree+1], dtype=int )
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
      return numpy.array( [[[1.]]] )

    assert 1 <= n < 2*p
    extractions = numpy.empty(( n, p+1, p+1 ))
    extractions[0] = numpy.eye( p+1 )
    for i in range( 1, n ):
      extractions[i] = numpy.eye( p+1 )
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

    return [ cls(c) for c in cls.spline_poly(p,n) ]

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

    return numpy.array( elems )

  def __init__( self, poly ):
    '''Create polynomial from order x nfuncs array of coefficients 'poly'.
       Evaluates to sum_i poly[i,:] x**i.'''

    self.poly = numpy.asarray( poly, dtype=float )
    order, nshapes = self.poly.shape
    self.degree = order - 1
    StdElem.__init__( self, ndims=1, nshapes=nshapes )

  def eval( self, points, grad=0 ):
    'evaluate'

    assert points.shape[-1] == 1
    assert points.dtype == float
    x = points[...,0]

    if grad > self.degree:
      return numeric.appendaxes( 0., x.shape+(self.nshapes,)+(1,)*grad )

    poly = self.poly
    for n in range(grad):
      poly = poly[1:] * numpy.arange( 1, poly.shape[0] )[:,_]

    polyval = numpy.empty( x.shape+(self.nshapes,) )
    polyval[:] = poly[-1]
    for p in poly[-2::-1]:
      polyval *= x[...,_]
      polyval += p

    return polyval[(Ellipsis,)+(_,)*grad]

  def extract( self, extraction ):
    'apply extraction'

    return PolyLine( numpy.dot( self.poly, extraction ) )

  def __repr__( self ):
    'string representation'

    return 'PolyLine#%x' % id(self)

class PolyTriangle( StdElem ):
  '''poly triangle (linear for now)
     conventions: dof numbering as vertices, see TriangularElement docstring.'''

  __slots__ = ()

  def __init__( self, order ):
    'constructor'

    assert order == 1
    StdElem.__init__( self, ndims=2, nshapes=3 )

  def eval( self, points, grad=0 ):
    'eval'

    npoints, ndim = points.shape
    if grad == 0:
      x, y = points.T
      data = numpy.array( [ 1-x-y, x, y ] ).T
    elif grad == 1:
      data = numpy.array( [[[-1,-1],[1,0],[0,1]]], dtype=float )
    else:
      data = numpy.zeros( (1,3)+(2,)*grad )
    return data

  def __repr__( self ):
    'string representation'

    return '%s#%x' % ( self.__class__.__name__, id(self) )

class BubbleTriangle( StdElem ):
  '''linear triangle + bubble function
     conventions: dof numbering as vertices (see TriangularElement docstring), then barycenter.'''

  __slots__ = ()

  def __init__( self ):
    StdElem.__init__( self, ndims=2, nshapes=4 )

  def eval( self, points, grad=0 ):
    'eval'

    npoints, ndims = points.shape
    assert ndims == 2, 'Triangle takes 2D coordinates'
    x, y = points.T
    if grad == 0:
      data = numpy.array( [ 1-x-y, x, y, 27*x*y*(1-x-y) ] ).T
    elif grad == 1:
      data = numpy.empty( (npoints,4,2) )
      data[:,:3] = numpy.array( [[[-1,-1], [1,0], [0,1]]] )
      data[:,3] = numpy.array( [27*y*(1-2*x-y),27*x*(1-x-2*y)] ).T
    elif grad == 2:
      data = numpy.zeros( (npoints,4,2,2) )
      data[:,3] = [-27*2*y,27*(1-2*x-2*y)], [27*(1-2*x-2*y),-27*2*x]
    elif grad == 3:
      data = numpy.zeros( (1,4,2,2,2) )
      data[:,3] = -2
      data[:,3,0,0,0] = 0
      data[:,3,1,1,1] = 0
    else:
      data = numpy.zeros( (1,4)+(2,)*grad )
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

  def eval( self, points, grad=0 ):
    'call'

    return numeric.dot( self.stdelem.eval( points, grad ), self.extraction, axis=1 )

  def __repr__( self ):
    'string representation'

    return '%s#%x:%s' % ( self.__class__.__name__, id(self), self.stdelem )


# UTILITY FUNCTIONS

_gauss = []
def gauss( degree ):
  n = degree // 2
  while len(_gauss) <= n:
    _gauss.append( None )
  gaussn = _gauss[n]
  if gaussn is None:
    k = numpy.arange(n) + 1
    d = k / numpy.sqrt( 4*k**2-1 )
    x, w = numpy.linalg.eigh( numpy.diagflat(d,-1) ) # eigh operates (by default) on lower triangle
    _gauss[n] = gaussn = (x+1) * .5, w[0]**2
  return gaussn

def signed_triangulate( points, vsigns ):
  assert len(points) == len(vsigns)
  npoints, ndims = points.shape
  fpoints = points.astype(float)
  triangulation = util.delaunay( fpoints )
  esigns = numpy.array([ +1 if min(s) >= 0 else -1 if max(s) <= 0 else 0 for s in vsigns[triangulation] ])
  if not esigns.all():
    ambiguous, = numpy.where( esigns == 0 )
    vmap = list(range(len(vsigns)))
    W = numpy.array([ .01/ndims, .99 ])
    for tri in triangulation[ambiguous]:
      I, = numpy.where( vsigns[tri] == 0 )
      fpoints = numpy.vstack([ fpoints, numpy.dot( W[(numpy.arange(ndims+1)==I[:,_]).astype(int)], fpoints[tri] ) ])
      vmap.extend( tri[I] )
    triangulation = numpy.array([ tri for tri in numpy.take(vmap,util.delaunay(fpoints)) if len(set(tri)) == len(tri) ])
    esigns = numpy.array([ +1 if min(s) >= 0 else -1 if max(s) <= 0 else 0 for s in vsigns[triangulation] ])
    assert esigns.all()
  selection = []
  for tri in triangulation:
    area = rational.det( points[tri[1:]] - points[tri[0]] )
    if area < 0:
      tri[-2:] = tri[-1], tri[-2]
    selection.append( area != 0 )
  selection = numpy.array( selection )
  if selection.any():
    triangulation = triangulation[selection]
    esigns = esigns[selection]
  triangulation = normtri( triangulation )
  renumber = arglexsort( triangulation )
  return triangulation[renumber], esigns[renumber] > 0

def getsimplex( ndims ):
  constructors = PointReference, LineReference, TriangleReference, TetrahedronReference
  return constructors[ndims]()

def mknewvtx( vertices, levels, ribbons, denom ):
  numer = []
  oniface = levels == 0
  isectribs = []
  if rational.isrational( vertices ):
    superdenom = denom * vertices.denom
    vertices = vertices.numer
  else:
    superdenom = denom
  for iribbon, (i,j) in enumerate(ribbons):
    a, b = levels[[i,j]]
    if a * b < 0:
      x = int( denom * a / float(a-b) + .5 ) # round to [0,1,..,denom]
      if x == 0:
        oniface[i] = True
      elif x == denom:
        oniface[j] = True
      else: # add new vertex with level zero
        numer.append( tuple( numpy.dot( (denom-x,x), vertices[[i,j]] ) ) )
        isectribs.append( iribbon )
  if not numer:
    return numpy.zeros( (0,vertices.shape[1]), dtype=int ), numpy.zeros( (0,), dtype=int ), oniface
  numer, isectribs = zip( *sorted( zip( numer, isectribs ) ) ) # canonical ordering (for element equivalence)
  return rational.frac(numer,superdenom), numpy.array(isectribs), oniface

def index_or_append( items, item ):
  try:
    index = items.index( item )
  except ValueError:
    index = len(items)
    items.append( item )
  return index

def normtri( triangulation ):
  triangulation = numpy.asarray(triangulation)
  if triangulation.shape[1] <= 2:
    return triangulation
  I = numpy.argsort( triangulation, axis=1 ).T
  oddperm = sum( (I[n+1:] < i).sum(axis=0) for n, i in enumerate(I[:-1]) ) % 2 == 1
  I[-2:,oddperm] = I[:-3:-1,oddperm]
  return numpy.array([ tuple(tri[i]) for tri, i in zip( triangulation, I.T ) ])

def arglexsort( triangulation ):
  return numpy.argsort( numeric.asobjvector( tuple(tri) for tri in triangulation ) )


# vim:shiftwidth=2:foldmethod=indent:foldnestmax=2
