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
from . import log, util, numpy, core, numeric, function, cache, transform, _
import re, warnings, math


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

  def trim( self, levelset, maxrefine, denom, fcache ):
    'trim element along levelset'

    assert not self.__opposite
    return self.reference.trim( (self.transform,levelset), maxrefine, denom, fcache )

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
    self.nverts, self.ndims = self.vertices.shape

  #__or__ = __ror__ = lambda self, other: other if other.contains(self) else self if self.contains(other) else NotImplemented
  #__and__ = __rand__ = lambda self, other: self if other.contains(self) else other if self.contains(other) else NotImplemented
  #__xor__ = __rxor__ = lambda self, other: other - self if other.contains(self) else self - other if self.contains(other) else NotImplemented

  __sub__ = __rsub__ = lambda self, other: self.empty if self == other else NotImplemented
  __bool__ = __nonzero__ = lambda self: bool(self.volume)

  @property
  def empty( self ):
    return EmptyReference( self )

  def __mul__( self, other ):
    assert isinstance( other, Reference )
    return other if self.ndims == 0 \
      else self if other.ndims == 0 \
      else TensorReference( self, other )

  def __pow__( self, n ):
    assert numeric.isint( n ) and n >= 0
    return getsimplex(0) if n == 0 \
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
    # ichild, iedge -> jchild, jedge, isouter
    # isouter=True: coinciding interface
    # isouter=False: corresponding edge
    vmap = {}
    childedgemap = tuple( [None] * child.nedges for child in self.child_refs )
    for iedge, (etrans,edge) in enumerate(self.edges):
      for ichild, (ctrans,child) in enumerate(edge.children):
        v = tuple( sorted( tuple(vi) for vi in (etrans<<ctrans).apply(child.vertices) ) )
        vmap[v] = ichild, iedge, True
    for ichild, (ctrans,child) in enumerate(self.children):
      for iedge, (etrans,edge) in enumerate(child.edges):
        v = tuple( sorted( tuple(vi) for vi in (ctrans<<etrans).apply(edge.vertices) ) )
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
  def interfaces( self ):
    return [ ((ichild,iedge),(jchild,jedge))
      for ichild, tmp in enumerate( self.childedgemap )
        for iedge, (jchild,jedge,isouter) in enumerate( tmp )
          if not isouter and ichild < jchild ]

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
          where |= numpy.all( self.vertices == v, axis=1 )
      edge2vertex.append( where )
    return numpy.array( edge2vertex )

  @cache.property
  def edgevertexmap( self ):
    return [ numpy.array( map( self.vertices.tolist().index, etrans.apply(edge.vertices).tolist() ), dtype=int ) for etrans, edge in self.edges ]

  def getischeme( self, ischeme ):
    match = re.match( '([a-zA-Z]+)(.*)', ischeme )
    assert match, 'cannot parse integration scheme %r' % ischeme
    ptype, args = match.groups()
    get = getattr( self, 'getischeme_'+ptype )
    return get( eval(args) ) if args else get()

  @classmethod
  def register( cls, ptype, func ):
    setattr( cls, 'getischeme_%s' % ptype, func )

  def with_children( self, child_refs, interfaces=[] ):
    child_refs = tuple(child_refs)
    if not any( child_refs ):
      return self.empty
    if child_refs == self.child_refs:
      return self
    return WithChildrenReference( self, child_refs, interfaces )

  def trim( self, levels, maxrefine, denom, fcache ):
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
      trimmed = self
    elif evaluated_levels and ( levels < 0 ).all():
      trimmed = self.empty
    elif maxrefine > 0:
      childrefs = []
      for ichild, (ctrans,child) in enumerate( self.children ):
        if not child:
          cref = child
        else:
          if evaluated_levels:
            N, I = fcache[self.subvertex]( ichild, maxrefine )
            assert len(levels) == N
            childlevels = levels[I]
          else:
            trans, levelfun = levels
            childlevels = trans << ctrans, levelfun
          cref = child.trim( childlevels, maxrefine-1, denom, fcache )
        childrefs.append( cref )
      trimmed = self.with_children( childrefs )
    else:
      assert evaluated_levels, 'failed to evaluate levelset up to level maxrefine'
      trimmed = self.slice( levels, denom )

    return trimmed

  def slice( self, levels, denom ):
    assert len(levels) == self.nverts
    if numpy.all( levels >= 0 ):
      return self
    if numpy.all( levels <= 0 ):
      return self.empty
    refs = [ edgeref.slice( levels[self.edgevertexmap[iedge]], denom ) for iedge, edgeref in enumerate( self.edge_refs ) ]
    if refs == list(self.edge_refs):
      return self
    if not any( refs ):
      return self.empty
    midpoint = numpy.mean( [ self.vertices[v1] + ( self.vertices[v2]-self.vertices[v1] ) * (levels[v1]/(levels[v1]-levels[v2]))
      for v1, v2 in self.ribbon2vertices if levels[v1] * levels[v2] <= 0 and levels[v1] != levels[v2] ], axis=0 )
    midpoint = numpy.round( midpoint * (1024*denom) ) / (1024*denom)
    return MosaicReference( self, refs, midpoint )

  def cone( self, trans, tip ):
    assert trans.fromdims == self.ndims
    assert trans.todims == len(tip)
    return Cone( self, trans, tip )

  def check_edges( self, tol=1e-10 ):
    if not self:
      return
    x, w = self.getischeme( 'gauss1' )
    volume = w.sum()
    assert abs( volume - self.volume ) < tol
    assert volume > 0
    check_volume = 0
    check_zero = 0
    for trans, edge in self.edges:
      if not edge:
        continue
      xe, we = edge.getischeme( 'gauss1' )
      w_normal = we[:,_] * trans.ext
      check_zero += w_normal.sum(0)
      check_volume += numeric.contract( trans.apply(xe), w_normal, axis=0 )
    zero_ok = numpy.all( abs(check_zero) < tol )
    volume_ok = numpy.all( abs(check_volume-volume) < tol )
    if zero_ok and volume_ok:
      return
    s = [ 'divergence check failed: ' + ', '.join( name for (name,ok) in (('zero',zero_ok),('volume',volume_ok)) if not ok ) ]
    try:
      s.append( 'Volume:' )
      s.extend( '* {} {} -> {}'.format( ref, numeric.fhex(trans.apply(ref.vertices)), numeric.fhex(trans.det*ref.volume) ) for trans, ref in self.simplices )
      s.append( 'Edges:' )
      s.extend( '* {} {} -> {}'.format( subref, numeric.fhex((etrans<<subtrans).apply(subref.vertices)), numeric.fhex((etrans<<subtrans).ext*subref.volume) ) for etrans, eref in self.edges for subtrans, subref in eref.simplices )
    except Exception as e:
      s.extend( 'processing failed: {}'.format(e) )
    raise MyException( '\n'.join(s) )

  def __str__( self ):
    return self.__class__.__name__

  __repr__ = __str__


class MyException( Exception ):
  def __repr__( self ):
    return str(self)


class EmptyReference( Reference ):
  'inverse reference element'

  volume = 0

  def __init__( self, baseref ):
    self.baseref = baseref
    Reference.__init__( self, numpy.zeros((0,baseref.ndims)) )

  @property
  def edge_transforms( self ):
    return self.baseref.edge_transforms

  @cache.property
  def edge_refs( self ):
    return [ edge_ref.empty for edge_ref in self.baseref.edge_refs ]

# __and__ = __rand__ = lambda self, other: self if self.baseref == other.rootref else NotImplemented
# __or__ = __ror__ = __xor__ = __rxor__ = __rsub__ = lambda self, other: other if self.baseref == other.rootref else NotImplemented

  __rsub__ = lambda self, other: other if other.ndims == self.ndims else NotImplementedError
  __bool__ = __nonzero__ = lambda self: False


class SimplexReference( Reference ):
  'simplex reference'

  def __init__( self, vertices ):
    nverts, ndims = vertices.shape
    assert nverts == ndims+1
    if ndims:
      self.edge_refs = (getsimplex(ndims-1),) * (ndims+1)
      self.x0 = vertices[0]
      self.dx = vertices[1:] - self.x0
      self.volume = numpy.linalg.det(self.dx) / math.factorial(ndims)
      #assert self.volume != 0
      self.inverted = self.volume < 0
      if self.inverted:
        self.volume = -self.volume
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

  def getischeme_vtk( self ):
    return self.vertices, None

  def getischeme_vertex( self, n=0 ):
    if n == 0:
      return self.vertices, None
    return self.getischeme_bezier( 2**n+1 )

  def cone( self, trans, tip ):
    assert trans.fromdims == self.ndims
    assert trans.todims == len(tip)
    return Simplex_by_dim[self.ndims+1]( numpy.vstack([ [tip], trans.apply( self.vertices ) ]) )

class PointReference( SimplexReference ):
  '0D simplex'

  volume = 1

  def __init__( self, vertices ):
    SimplexReference.__init__( self, vertices )
    self.child_transforms = transform.identity,

  def getischeme( self, ischeme ):
    return numpy.zeros((1,0)), numpy.ones(1)

  def slice( self, (level,), denom ):
    return self if level > 0 else self.empty

class LineReference( SimplexReference ):
  '1D simplex'

  def __init__( self, vertices ):
    self._bernsteincache = [] # TEMPORARY
    SimplexReference.__init__( self, vertices )
    self.edge_transforms = transform.simplex( vertices[1:], isflipped=self.inverted ), transform.simplex( vertices[:1], isflipped=not self.inverted )
    self.child_transforms = transform.affine(1,[0],2), transform.affine(1,[1],2)
    if core.getprop( 'selfcheck', False ):
      self.check_edges()

  def stdfunc( self, degree ):
    if len(self._bernsteincache) <= degree or self._bernsteincache[degree] is None:
      self._bernsteincache += [None] * (degree-len(self._bernsteincache))
      self._bernsteincache.append( PolyLine( PolyLine.bernstein_poly(degree) ) )
    return self._bernsteincache[degree]

  def getischeme_gauss( self, degree ):
    assert isinstance( degree, int ) and degree >= 0
    x, w = gauss( degree )
    return self.x0 + x[:,_] * self.dx, w * self.volume

  def getischeme_uniform( self, n ):
    return self.x0 + ( numpy.arange(.5,n) / n )[:,_] * self.dx, numeric.appendaxes( self.volume/n, n )

  def getischeme_bezier( self, np ):
    return self.x0 + numpy.linspace( 0, 1, np )[:,_] * self.dx, None

  def subvertex( self, ichild, i ):
    if i == 0:
      assert ichild == 0
      return self.nverts, numpy.arange(self.nverts)
    assert 0 <= ichild < 2
    n = 2**i+1
    return n, numpy.arange(n//2+1) if ichild == 0 else numpy.arange(n//2,n)

class TriangleReference( SimplexReference ):
  '2D simplex'

  def __init__( self, vertices ):
    SimplexReference.__init__( self, vertices )
    self.edge_transforms = tuple( transform.simplex( vertices[I], isflipped=self.inverted ) for I in [slice(1,None),slice(None,None,-2),slice(2)] )
    self.child_transforms = transform.affine(1,[0,0],2), transform.affine(1,[0,1],2), transform.affine(1,[1,0],2), transform.affine([[0,-1],[-1,0]],[1,1],2,isflipped=True )
    if core.getprop( 'selfcheck', False ):
      self.check_edges()

  def stdfunc( self, degree ):
    return PolyTriangle(degree)

  def getischeme_contour( self, n ):
    p = numpy.arange( n+1, dtype=float ) / (n+1)
    z = numpy.zeros_like( p )
    return self.x0 + numpy.dot( numpy.hstack(( [1-p,p], [z,1-p], [p,z] )).T, self.dx ), None

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

    return self.x0 + numpy.dot( numpy.concatenate( [ numpy.take(c,i) for i, c, w in icw ], axis=0 ), self.dx ), \
           numpy.concatenate( [ [w*self.volume] * len(i) for i, c, w in icw ] )

  def getischeme_uniform( self, n ):
    points = numpy.arange( 1./3, n ) / n
    nn = n**2
    C = numpy.empty( [2,n,n] )
    C[0] = points[:,_]
    C[1] = points[_,:]
    coords = C.reshape( 2, nn )
    flip = coords.sum(0) > 1
    coords[:,flip] = 1 - coords[::-1,flip]
    weights = numeric.appendaxes( self.volume/nn, nn )
    return self.x0 + numpy.dot( coords.T, self.dx ), weights

  def getischeme_bezier( self, np ):
    points = numpy.linspace( 0, 1, np )
    return self.x0 + numpy.dot( numpy.array([ [x,y] for i, y in enumerate(points) for x in points[:np-i] ]), self.dx ), None

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

  def __init__( self, vertices ):
    SimplexReference.__init__( self, vertices )
    self.edge_transforms = tuple( transform.simplex( vertices[I], isflipped=self.inverted ) for I in [[1,2,3],[0,3,2],[3,0,1],[2,1,0]] )
    if core.getprop( 'selfcheck', False ):
      self.check_edges()

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

    return self.x0 + numpy.dot( numpy.concatenate( [ numpy.take(c,i) for i, c, w in icw ], axis=0 ), self.dx ), \
           numpy.concatenate( [ [w*self.volume] * len(i) for i, c, w in icw ] )

Simplex_by_dim = PointReference, LineReference, TriangleReference, TetrahedronReference

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
    if core.getprop( 'selfcheck', False ):
      self.check_edges()

  @property
  def volume( self ):
    return self.ref1.volume * self.ref2.volume

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
    if self == getsimplex(1)**2:
      points = [[0,0],[1,0],[1,1],[0,1]]
    elif self == getsimplex(1)**3:
      points = [[0,0,0],[1,0,0],[0,1,0],[1,1,0],[0,0,1],[1,0,1],[0,1,1],[1,1,1]]
    else:
      raise NotImplementedError
    return numpy.array(points,dtype=float), numpy.ones(self.nverts,dtype=float)

  def getischeme_contour( self, n ):
    assert self == getsimplex(1)**2
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
        numeric.blockdiag([ trans1.linear, numpy.eye(self.ref2.ndims) ]),
        numpy.concatenate([ trans1.offset, numpy.zeros(self.ref2.ndims) ]),
        isflipped=trans1.isflipped )
          for trans1 in self.ref1.edge_transforms ]
   + [ transform.affine(
        numeric.blockdiag([ numpy.eye(self.ref1.ndims), trans2.linear ]),
        numpy.concatenate([ numpy.zeros(self.ref1.ndims), trans2.offset ]),
        isflipped=trans2.isflipped if self.ref1.ndims%2==0 else not trans2.isflipped )
          for trans2 in self.ref2.edge_transforms ])

  @property
  def edge_refs( self ):
    return tuple([ edge1 * self.ref2 for edge1 in self.ref1.edge_refs ]
               + [ self.ref1 * edge2 for edge2 in self.ref2.edge_refs ])

  @cache.property
  def child_transforms( self ):
    return [ transform.affine( trans1.linear if trans1.linear.ndim == 0 and trans1.linear == trans2.linear
                          else numeric.blockdiag([ trans1.linear, trans2.linear ]),
                               numpy.concatenate([ trans1.offset, trans2.offset ]) )
            for trans1 in self.ref1.child_transforms
              for trans2 in self.ref2.child_transforms ]

  @property
  def child_refs( self ):
    return tuple( child1 * child2 for child1 in self.ref1.child_refs for child2 in self.ref2.child_refs )

class Cone( Reference ):
  'cone'

  def __init__( self, edgeref, etrans, tip ):
    vertices = numpy.vstack([ [tip], etrans.apply( edgeref.vertices ) ])
    Reference.__init__( self, vertices )
    self.edgeref = edgeref
    self.etrans = etrans
    self.axisref = getsimplex(1)
    self.tip = tip
    ext = numeric.ext( etrans.linear )
    self.extnorm = numpy.linalg.norm( ext )
    self.height = abs( numpy.dot( tip - etrans.offset, ext ) / self.extnorm )
    if core.getprop( 'selfcheck', False ):
      self.check_edges()

  @property
  def volume( self ):
    return self.edgeref.volume * self.extnorm * self.height / self.ndims

  @cache.property
  def edge_transforms( self ):
    edge_transforms = [ self.etrans ]
    for trans, edge in self.edgeref.edges:
      if edge:
        b = self.etrans.apply( trans.offset )
        A = numpy.hstack([ numpy.dot( self.etrans.linear, trans.linear ), (self.tip-b)[:,_] ])
        newtrans = transform.affine( A, b, isflipped=self.etrans.isflipped^trans.isflipped )
        edge_transforms.append( newtrans )
    return edge_transforms

  @cache.property
  def edge_refs( self ):
    extrudetrans = transform.affine( numpy.eye(self.ndims-1)[:,:-1], numpy.zeros(self.ndims-1), isflipped=False )
    tip = numpy.array( [0]*(self.ndims-2)+[1], dtype=float )
    return [ self.edgeref ] + [ edge.cone( extrudetrans, tip ) for trans, edge in self.edgeref.edges if edge ]

  def getischeme( self, ischeme ):
    if ischeme == 'vtk':
      return self.getischeme_vtk()
    epoints, eweights = self.edgeref.getischeme( ischeme )
    tpoints, tweights = self.axisref.getischeme( ischeme )
    s = 1/self.ndims
    tx, = ( tpoints.T )**s
    points = ( tx[:,_,_] * (self.etrans.apply(epoints)-self.tip)[_,:,:] + self.tip ).reshape( -1, self.ndims )
    if tweights is None:
      weights = None
    else:
      wx = tweights * s * self.extnorm * self.height
      weights = ( eweights[_,:] * wx[:,_] ).ravel()
    return points, weights

  def getischeme_vtk( self ):
    assert self.ndims == 3
    if self.nverts == 4:
      I = slice(None)
    elif self.nverts == 5:
      I = numpy.array([1,2,4,3,0])
    else:
      raise Exception( 'invalid number of points: {}'.format(self.nverts) )
    return self.vertices[I], None

  @property
  def simplices( self ):
    return [ ( trans, ref.cone(self.etrans,self.tip) ) for trans, ref in self.edgeref.simplices ]

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
    points, weights = (getsimplex(1)**4).getischeme( ischeme )
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
    quad = getsimplex(1)**4
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
    assert self.ref1 == self.ref2 == getsimplex(1)**2
    points, weights = self.get_quad_bem_ischeme( gauss )
    return self.singular_ischeme_quad( points ), weights

class WrappedReference( Reference ):
  'derive properties from baseref'

  def __init__( self, baseref ):
    self.baseref = baseref
    Reference.__init__( self, baseref.vertices )

# def getischeme( self, ischeme ):
#   return self.baseref.getischeme( ischeme )

  def stdfunc( self, degree ):
    return self.baseref.stdfunc( degree )

  @property
  def subvertex( self ):
    return self.baseref.subvertex

#  def __or__( self, other ):
#    return self if other is None or other == self \
#      else self.baseref if other == self.baseref or other == ~self \
#      else self._logical( other, lambda ref1, ref2: ref1 | ref2 )
#
#  def __and__( self, other ):
#    return None if other is None or other == ~self \
#      else self if other == self.baseref or other == self \
#      else self._logical( other, lambda ref1, ref2: ref1 & ref2 )
#
#  def __xor__( self, other ):
#    return self if other is None \
#      else None if other == self \
#      else self.baseref if other == ~self \
#      else ~self if other == self.baseref \
#      else self._logical( other, lambda ref1, ref2: ref1 ^ ref2 )
#
#  def __sub__( self, other ):
#    if other.__class__ == self.__class__ and other.baseref == self:
#      return ~other
#    return Reference.__sub__( self, other )
#
#  def __rsub__( self, other ):
#    if other != self.baseref:
#      return NotImplemented
#    return ~self

class OwnChildReference( WrappedReference ):
  'forward self as child'

  def __init__( self, baseref ):
    self.child_refs = baseref,
    self.child_transforms = transform.identity,
    WrappedReference.__init__( self, baseref )

  @property
  def volume( self ):
    return self.baseref.volume

  def getischeme( self, ischeme ):
    return self.baseref.getischeme( ischeme )

  @property
  def childedgemap( self ):
    return self.baseref.childedgemap

  @property
  def edge2children( self ):
    return self.baseref.edge2children

class WithChildrenReference( WrappedReference ):
  'base reference with explicit children'

  def __init__( self, baseref, child_refs, interfaces=[] ):
    assert isinstance( child_refs, tuple ) and len(child_refs) == baseref.nchildren and any(child_refs) and child_refs != baseref.child_refs
    assert all( isinstance(child_ref,Reference) for child_ref in child_refs )
    assert all( child_ref.ndims == baseref.ndims for child_ref in child_refs )
    self.child_transforms = baseref.child_transforms
    self.child_refs = child_refs
    WrappedReference.__init__( self, baseref )
    self.__interfaces_arg = interfaces
    if core.getprop( 'selfcheck', False ):
      self.check_edges()

  @property
  def volume( self ):
    return sum( abs(trans.det) * ref.volume for trans, ref in self.children )

  __rsub__ = lambda self, other: self.baseref.with_children( other_child-self_child for self_child, other_child in zip( self.child_refs, other.child_refs ) ) if self.baseref == other or isinstance( other, WithChildrenReference ) and self.baseref == other.baseref else NotImplementedError

  @cache.property
  def __interfaces( self ):
    interfaces = []
    for (ichild,iedge), (jchild,jedge) in self.baseref.interfaces:
      edge1 = self.child_refs[ichild].edge_refs[iedge]
      edge2 = self.child_refs[jchild].edge_refs[jedge]
      if edge2 and not edge1:
        interfaces.append(( jchild, jedge, self.child_transforms[jchild] << self.child_refs[jchild].edge_transforms[jedge], self.child_refs[jchild].edge_refs[jedge] ))
      elif edge1 and not edge2:
        interfaces.append(( ichild, iedge, self.child_transforms[ichild] << self.child_refs[ichild].edge_transforms[iedge], self.child_refs[ichild].edge_refs[iedge] ))
      elif edge1 != edge2:
        raise Exception
    return interfaces

  def _logical( self, other, op ):
    return self.baseref.with_children( op(child1,child2) if child1 or child2 else None for child1, child2 in zip( self.child_refs, other.child_refs ) )

  def getischeme( self, ischeme ):
    'get integration scheme'
    
    if ischeme.startswith('vertex'):
      return self.baseref.getischeme( ischeme )

    allcoords = []
    allweights = []
    for trans, ref in self.children:
      if ref:
        points, weights = ref.getischeme( ischeme )
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
    edge_transforms.extend( trans for ichild, iedge, trans, ref in self.__interfaces )
    return tuple(edge_transforms)

  @cache.property
  def edge_refs( self ):
    # to avoid circular references we cannot mention 'self' inside getedgeref
    def getedgeref( iedge, baseref=self.baseref, child_refs=self.child_refs, edge2children=self.edge2children ):
      baseedge = baseref.edge_refs[iedge]
      return baseedge and baseedge.with_children( child_refs[jchild].edge_refs[jedge] for jchild, jedge in edge2children[iedge] )
    items = [ cache.Tuple.unknown ] * self.baseref.nedges
    for mychild, basechild in zip( self.child_refs, self.baseref.child_refs ):
      if mychild:
        items.extend( OwnChildReference(edge) for edge in mychild.edge_refs[basechild.nedges:] )
    items.extend( OwnChildReference(ref) for ichild, iedge, trans, ref in self.__interfaces )
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
    edge2children.extend( [(ichild,iedge)] for ichild, iedge, trans, ref in self.__interfaces )
    return tuple( edge2children )

class MosaicReference( Reference ):
  'triangulation'

  def __init__( self, baseref, edge_refs, midpoint ):
    assert len(edge_refs) == baseref.nedges
    self.baseref = baseref
    self._edge_refs = tuple( edge_refs )
    self._midpoint = midpoint

    self.subrefs = [ ref.cone(trans,midpoint) for trans, ref in zip( baseref.edge_transforms, edge_refs ) if ref ]

    keep = {}
    for sub in self.subrefs:
      for trans2, edge2 in sub.edges[1:]:
        vertices = tuple( sorted( tuple(v) for v in trans2.apply(edge2.vertices) ) )
        try:
          keep.pop( vertices )
        except KeyError:
          keep[vertices] = trans2, edge2

    self.edge_transforms = tuple(baseref.edge_transforms) + tuple( trans for trans, ref in keep.values() )
    self.edge_refs = tuple(edge_refs) + tuple( ref for trans, ref in keep.values() )

    vertices = []
    edgevertexmap = []
    for etrans, eref in self.edges:
      indices = []
      for vertex in etrans.apply( eref.vertices ).tolist():
        try:
          index = vertices.index( vertex )
        except ValueError:
          index = len(vertices)
          vertices.append( vertex )
        indices.append( index )
      edgevertexmap.append( numpy.array(indices,dtype=int) )
    self._edgevertexmap = edgevertexmap

    Reference.__init__( self, vertices )

    if core.getprop( 'selfcheck', False ):
      self.check_edges()

  @property
  def edgevertexmap( self ):
    return self._edgevertexmap

  def stdfunc( self, degree ):
    return self.baseref.stdfunc( degree )

  @property
  def volume( self ):
    return sum( subref.volume for subref in self.subrefs )

  @property
  def simplices( self ):
    return [ simplex for subvol in self.subrefs for simplex in subvol.simplices ]

  def getischeme( self, ischeme ):
    'get integration scheme'
    
    if ischeme.startswith('vertex'):
      return self.baseref.getischeme( ischeme )

    allpoints, allweights = zip( *[ subvol.getischeme(ischeme) for subvol in self.subrefs ] )
    points = numpy.concatenate( allpoints, axis=0 )
    if allweights[0] is None:
      assert not any( allweights )
      weights = None
    else:
      weights = numpy.concatenate( allweights, axis=0 )
    return points, weights

  def __rsub__( self, other ):
    assert other == self.baseref
    inv_edge_refs = [ baseedge - edge for baseedge, edge in zip( self.baseref.edge_refs, self._edge_refs ) ]
    return MosaicReference( self.baseref, inv_edge_refs, self._midpoint )


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

class ExtractionWrapper( StdElem ):
  'extraction wrapper'

  __slots__ = 'stdelem', 'extraction'

  def __init__( self, stdelem, extraction ):
    'constructor'

    assert extraction.ndim == 2
    assert stdelem.nshapes == extraction.shape[0]
    self.stdelem = stdelem
    self.extraction = extraction
    StdElem.__init__( self, stdelem.ndims, extraction.shape[1] )

  def extract( self, extraction ):
    return ExtractionWrapper( self.stdelem, numpy.dot( self.extraction, extraction ) )

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

def getsimplex( ndims ):
  vertices = numpy.concatenate( [ numpy.zeros(ndims,dtype=int)[_,:], numpy.eye(ndims,dtype=int) ], axis=0 )
  return Simplex_by_dim[ndims]( vertices )

def mknewvtx( vertices, levels, ribbons, denom ):
  numer = []
  oniface = levels == 0
  isectribs = []
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
  return numer/denom, numpy.array(isectribs), oniface

def index_or_append( items, item ):
  try:
    index = items.index( item )
  except ValueError:
    index = len(items)
    items.append( item )
  return index

def arglexsort( triangulation ):
  return numpy.argsort( numeric.asobjvector( tuple(tri) for tri in triangulation ) )


# vim:shiftwidth=2:softtabstop=2:expandtab:foldmethod=indent:foldnestmax=2
