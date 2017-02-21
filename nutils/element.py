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

from . import log, util, numpy, core, numeric, function, cache, transform, _
import re, warnings, math


## ELEMENT

class Element( object ):
  'element class'

  __slots__ = 'reference', 'transform', 'opposite'

  def __init__( self, reference, trans, opptrans=None, oriented=False ):
    assert isinstance( reference, Reference )
    assert isinstance( trans, transform.TransformChain ) and trans.fromdims == reference.ndims and trans.todims == None
    trans = trans.canonical
    if opptrans is not None:
      assert isinstance( opptrans, transform.TransformChain ) and opptrans.fromdims == reference.ndims and opptrans.todims == None
      opptrans = opptrans.canonical
      if not oriented:
        vtx1 = trans.apply( reference.vertices )
        vtx2 = opptrans.apply( reference.vertices )
        if vtx1 != vtx2:
          assert reference.ndims == 1 # TODO leverage rotation information from element
          assert vtx1 == vtx2[::-1]
          opptrans <<= transform.affine( -1, [1] )
    self.reference = reference
    self.transform = trans
    self.opposite = opptrans

  def withopposite( self, opp, oriented=False ):
    if isinstance( opp, transform.TransformChain ):
      return Element( self.reference, self.transform, opp, oriented )
    assert isinstance( opp, Element ) and opp.reference == self.reference
    return Element( self.reference, self.transform, opp.transform, oriented or opp.opposite==self.transform )

  def __mul__( self, other ):
    if self.opposite or other.opposite:
      assert not self.opposite or not other.opposite, 'cannot multiply two interface elements'
      opposite = transform.stack( self.opposite or self.transform, other.opposite or other.transform )
    else:
      opposite = None
    return Element( self.reference * other.reference, transform.stack( self.transform, other.transform ), opposite, oriented=True )

  def __getnewargs__( self ):
    return self.reference, self.transform, self.opposite, True

  def __hash__( self ):
    return hash(( self.reference, self.transform, self.opposite ))

  def __eq__( self, other ):
    return self is other or isinstance(other,Element) \
      and self.reference == other.reference \
      and self.transform == other.transform \
      and self.opposite == other.opposite

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
    return Element( edge, self.transform << trans, self.opposite and self.opposite << trans, oriented=True ) if edge else None

  @property
  def children( self ):
    return [ Element( child, self.transform << trans, self.opposite and self.opposite << trans, oriented=True )
      for trans, child in self.reference.children if child ]

  @property
  def flipped( self ):
    assert self.opposite, 'element does not define an opposite'
    return Element( self.reference, self.opposite, self.transform, oriented=True )

  @property
  def simplices( self ):
    return [ Element( reference, self.transform << trans, self.opposite and self.opposite << trans, oriented=True )
      for trans, reference in self.reference.simplices ]

  def __str__( self ):
    return 'Element({})'.format( self.vertices )


## REFERENCE ELEMENTS

class Reference( cache.Immutable ):
  'reference element'

  def __init__( self, vertices ):
    self.vertices = numpy.asarray( vertices )
    self.nverts, self.ndims = self.vertices.shape

  __and__ = lambda self, other: self if self == other else NotImplemented
  __or__ = lambda self, other: self if self == other else NotImplemented
  __rand__ = lambda self, other: self.__and__( other )
  __ror__ = lambda self, other: self.__or__( other )
  __sub__ = __rsub__ = lambda self, other: self.empty if self == other else NotImplemented
  __bool__ = __nonzero__ = lambda self: bool(self.volume)

  @property
  def empty( self ):
    return EmptyReference( self.ndims )

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

  @cache.property
  def childedgemap( self ):
    # ichild>iedge --> jchild>jedge, isouter=False (corresponding edge)
    # ichild>iedge --> jchild<jedge, isouter=True (coinciding interface)

    vmap = {}
    childedgemap = tuple( [None] * child.nedges for child in self.child_refs )

    for ichild, (ctrans,child) in enumerate(self.children):
      for iedge, (etrans,edge) in enumerate(child.edges):
        v = (ctrans<<etrans).flat
        try:
          jchild, jedge = vmap.pop(v.flipped)
        except KeyError:
          vmap[v] = ichild, iedge
        else:
          childedgemap[jchild][jedge] = ichild, iedge, False
          childedgemap[ichild][iedge] = jchild, jedge, False

    for iedge, (etrans,edge) in enumerate(self.edges):
      for ichild, (ctrans,child) in enumerate(edge.children):
        jchild, jedge = vmap.pop( (etrans<<ctrans).flat )
        childedgemap[jchild][jedge] = ichild, iedge, True

    assert not vmap, 'boundaries and edges do not commute'
    return childedgemap

  @cache.property
  def ribbons( self ):
    # tuples of (iedge1,jedge1), (iedge2,jedge2) pairs
    assert self.ndims >= 2
    ribbons = tuple( self._ribbons )
    if core.getprop( 'selfcheck', False ):
      for (iedge1,iedge2), (jedge1,jedge2) in ribbons:
        itrans = self.edge_transforms[iedge1] << self.edge_refs[iedge1].edge_transforms[iedge2]
        jtrans = self.edge_transforms[jedge1] << self.edge_refs[jedge1].edge_transforms[jedge2]
        assert ( itrans.linear == jtrans.linear ).all() and ( itrans.offset == jtrans.offset ).all()
        iref = self.edge_refs[iedge1].edge_refs[iedge2]
        jref = self.edge_refs[jedge1].edge_refs[jedge2]
        assert iref == jref
    return ribbons

  @property
  def _ribbons( self ):
    # tuples of (iedge1,jedge1), (iedge2,jedge2) pairs
    assert self.ndims >= 2
    transforms = {}
    ribbons = []
    for iedge1, (etrans1,edge1) in enumerate( self.edges ):
      if edge1:
        for iedge2, (etrans2,edge2) in enumerate( edge1.edges ):
          if edge2:
            key = tuple( sorted( tuple(p) for p in (etrans1 << etrans2).apply( edge2.vertices ) ) )
            try:
              jedge1, jedge2 = transforms.pop(key)
            except KeyError:
              transforms[key] = iedge1, iedge2
            else:
              assert self.edge_refs[jedge1].edge_refs[jedge2] == edge2
              ribbons.append(( (iedge1,iedge2), (jedge1,jedge2) ))
    assert not transforms
    return tuple( ribbons )

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
  def edgevertexmap( self ):
    return [ numpy.array( list( map( self.vertices.tolist().index, etrans.apply(edge.vertices).tolist() ) ), dtype=int ) for etrans, edge in self.edges ]

  def getischeme( self, ischeme ):
    match = re.match( '([a-zA-Z]+)(.*)', ischeme )
    assert match, 'cannot parse integration scheme %r' % ischeme
    ptype, args = match.groups()
    get = getattr( self, 'getischeme_'+ptype )
    return get( eval(args) ) if args else get()

  @classmethod
  def register( cls, ptype, func ):
    setattr( cls, 'getischeme_%s' % ptype, func )

  def with_children( self, child_refs ):
    child_refs = tuple(child_refs)
    if not any( child_refs ):
      return self.empty
    if child_refs == self.child_refs:
      return self
    return WithChildrenReference( self, child_refs )

  def trim( self, levels, maxrefine, ndivisions ):
    'trim element along levelset'

    assert len(levels) == self.nvertices_by_level(maxrefine)
    return self if not self or ( levels >= 0 ).all() \
      else self.empty if ( levels <= 0 ).all() \
      else self.with_children( cref.trim( clevels, maxrefine-1, ndivisions )
            for cref, clevels in zip( self.child_refs, self.child_divide(levels,maxrefine) ) ) if maxrefine > 0 \
      else self.slice( numeric.dot( self.stdfunc(1).eval(self.vertices), levels ), ndivisions )

  def slice( self, levels, ndivisions ):
    # slice along levelset by recursing over dimensions

    assert numeric.isint( ndivisions )
    assert len(levels) == self.nverts
    if numpy.all( levels >= 0 ):
      return self
    if numpy.all( levels <= 0 ):
      return self.empty

    nbins = 2**ndivisions

    if self.ndims == 1:

      l0, l1 = levels
      xi = numpy.round( l0/(l0-l1) * nbins )
      if xi in (0,nbins):
        return self.empty if xi == 0 and l1 < 0 or xi == nbins and l0 < 0 else self
      v0, v1 = self.vertices
      midpoint = v0 + (xi/nbins) * (v1-v0)
      refs = [ edgeref if levels[self.edgevertexmap[iedge]] > 0 else edgeref.empty for iedge, edgeref in enumerate( self.edge_refs ) ]

    else:

      refs = [ edgeref.slice( levels[self.edgevertexmap[iedge]], ndivisions ) for iedge, edgeref in enumerate( self.edge_refs ) ]
      if sum( ref != baseref for ref, baseref in zip( refs, self.edge_refs ) ) <= 1:
        return self
      if sum( bool(ref) for ref in refs ) <= 1:
        return self.empty

      #TODO: Optimization possible for case in which no EmptyReferences are present
      keep = {}
      for trans, ref in zip(self.edge_transforms,refs):
        if ref:
          vertices = trans.apply( ref.vertices )
          for (trans2, edge2), indices in zip(ref.edges,ref.edgevertexmap):
            if edge2:
              key = tuple(sorted(tuple(v) for v in vertices[indices]))
              try:
                keep.pop( key )
              except KeyError:
                keep[key] = trans, trans2, edge2

      length = 0.
      midpoint = 0.
      for  trans, trans2, edge2 in keep.values():
        points, weights = edge2.getischeme( 'gauss1' )
        wtot = weights.sum()*numpy.linalg.norm(trans2.ext)*numpy.linalg.norm(trans.ext)
        length += wtot
        midpoint += wtot * (trans<<trans2).apply( numeric.dot( weights, points )/weights.sum() )
      midpoint /= length

    mosaic = MosaicReference( self, refs, midpoint )
    return self.empty if mosaic.volume == 0 else mosaic if mosaic.volume < self.volume else self

  def cone( self, trans, tip ):
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
      s.append( 'Vertices:' )
      s.extend( '{} {}'.format( chr(ord('A')+i), numeric.fhex(v) ) for i, v in enumerate( self.vertices ) )
      index = self.vertices.tolist().index
      vtx2abc = lambda vertices: '[' + ','.join( chr(ord('A')+index(v)) for v in vertices.tolist() ) + ']'
      s.append( 'Volume:' )
      s.extend( '* {} {} -> {}'.format( ref, vtx2abc(trans.apply(ref.vertices)), trans.det*ref.volume ) for trans, ref in self.simplices )
      s.append( 'Edges:' )
      s.extend( '* {} {} -> {}'.format( subref, vtx2abc((etrans<<subtrans).apply(subref.vertices)), numeric.fhex((etrans<<subtrans).ext*subref.volume) ) for etrans, eref in self.edges for subtrans, subref in eref.simplices )
    except Exception as e:
      s.append( 'processing failed: {}'.format(e) )
    ## useful code for the debugging of failed selfchecks
    #if self.ndims == 2:
    #  from . import plot
    #  with plot.PyPlot( 'selfcheckfailed' ) as plt:
    #    plt.segments( trans.apply(edge.getischeme('bezier2')[0]) for trans, edge in self.edges if edge )
    raise MyException( '\n'.join(s) )

  def vertex_cover( self, ctransforms, maxrefine ):
    if maxrefine < 0:
      raise Exception( 'maxrefine is too low' )
    npoints = self.nvertices_by_level(maxrefine)
    allindices = numpy.arange(npoints)
    if len(ctransforms) == 1:
      assert ctransforms[0] == transform.identity
      return ( transform.identity, self.getischeme('vertex{}'.format(maxrefine))[0], allindices ),
    if maxrefine == 0:
      raise Exception( 'maxrefine is too low' )
    cbins = [ [] for ichild in range(self.nchildren) ]
    for ctrans in ctransforms:
      ichild = self.child_transforms.index( ctrans[:1] )
      cbins[ichild].append( ctrans[1:] )
    if not all( cbins ):
      raise Exception( 'transformations to not form an element cover' )
    fcache = cache.WrapperCache()
    return tuple( ( ctrans << trans, points, cindices[indices] )
      for ctrans, cref, cbin, cindices in zip( self.child_transforms, self.child_refs, cbins, self.child_divide(allindices,maxrefine) )
        for trans, points, indices in fcache[cref.vertex_cover]( sorted(cbin), maxrefine-1 ) )

  def __str__( self ):
    return self.__class__.__name__

  __repr__ = __str__

class MyException( Exception ):
  def __repr__( self ):
    return str(self)

class EmptyReference( Reference ):
  'inverse reference element'

  volume = 0

  edge_transforms = ()
  edge_refs = ()

  def __init__( self, ndims ):
    Reference.__init__( self, numpy.zeros((0,ndims)) )

  __and__ = __sub__ = lambda self, other: self if other.ndims == self.ndims else NotImplemented
  __or__ = lambda self, other: other if other.ndims == self.ndims else NotImplemented
  __rsub__ = lambda self, other: other if other.ndims == self.ndims else NotImplemented

class RevolutionReference( Reference ):
  'modify gauss integration to always return a single point'

  def __init__( self ):
    self.volume = 2 * numpy.pi
    Reference.__init__( self, numpy.zeros((1,1)) )

  @property
  def edge_transforms( self ): # only used in check_edges
    return transform.affine( numpy.zeros((1,0)), [-numpy.pi], isflipped=True ), transform.affine( numpy.zeros((1,0)), [+numpy.pi], isflipped=False )

  @property
  def edge_refs( self ): # idem edge_transforms
    return PointReference(), PointReference()

  @property
  def simplices( self ):
    return [ (transform.identity,self) ]

  def getischeme( self, ischeme ):
    return numpy.zeros([1,1]), numpy.array([ self.volume ])

class SimplexReference( Reference ):
  'simplex reference'

  def __init__( self, ndims ):
    Reference.__init__( self, numpy.concatenate( [ numpy.zeros(ndims,dtype=int)[_,:], numpy.eye(ndims,dtype=int) ], axis=0 ) )
    self.volume = 1. / math.factorial(ndims)
    if self.ndims > 0 and core.getprop( 'selfcheck', False ):
      self.check_edges()

  @cache.property
  def edge_refs( self ):
    return (getsimplex(self.ndims-1),) * (self.ndims+1)

  @cache.property
  def edge_transforms( self ):
    assert self.ndims > 0
    return tuple( transform.simplex( self.vertices[list(range(i))+list(range(i+1,self.ndims+1))], isflipped=i%2==1 ) for i in range(self.ndims+1) )

  @property
  def _ribbons( self ):
    return [ ((iedge1,iedge2),(iedge2+1,iedge1)) for iedge1 in range(self.ndims+1) for iedge2 in range(iedge1,self.ndims) ]

  def getischeme_vtk( self ):
    return self.vertices, None

  def getischeme_vertex( self, n=0 ):
    if n == 0:
      return self.vertices, None
    return self.getischeme_bezier( 2**n+1 )

  @property
  def simplices( self ):
    return [ (transform.identity,self) ]

class PointReference( SimplexReference ):
  '0D simplex'

  volume = 1

  def __init__( self ):
    SimplexReference.__init__( self, ndims=0 )

  @property
  def child_transforms( self ):
    return transform.identity,

  @property
  def child_refs( self ):
    return self,

  def getischeme( self, ischeme ):
    return numpy.zeros((1,0)), numpy.ones(1)

class LineReference( SimplexReference ):
  '1D simplex'

  def __init__( self ):
    self._bernsteincache = [] # TEMPORARY
    SimplexReference.__init__( self, ndims=1 )

  @cache.property
  def child_transforms( self ):
    return transform.affine(1,[0],2), transform.affine(1,[1],2)

  @property
  def child_refs( self ):
    return self, self

  def stdfunc( self, degree ):
    if len(self._bernsteincache) <= degree or self._bernsteincache[degree] is None:
      self._bernsteincache += [None] * (degree-len(self._bernsteincache))
      self._bernsteincache.append( PolyLine( PolyLine.bernstein_poly(degree) ) )
    return self._bernsteincache[degree]

  def getischeme_gauss( self, degree ):
    assert isinstance( degree, int ) and degree >= 0
    x, w = gauss( degree )
    return x[:,_], w * self.volume

  def getischeme_uniform( self, n ):
    return ( numpy.arange(.5,n) / n )[:,_], numeric.appendaxes( self.volume/n, n )

  def getischeme_bezier( self, np ):
    return numpy.linspace( 0, 1, np )[:,_], None

  def nvertices_by_level( self, n ):
    return 2**n + 1

  def child_divide( self, vals, n ):
    assert n > 0
    assert len(vals) == self.nvertices_by_level(n)
    m = (len(vals)+1) // 2
    return vals[:m], vals[m-1:]

  def inside( self, points, eps=0 ):
    points = numpy.asarray( points, dtype=float )
    assert points.ndim == 2 and points.shape[1] == self.ndims
    vmin, vmax = sorted( self.vertices[:,0] )
    return ( (points >= vmin-eps) & (points <= vmax+eps) ).all( axis=1 )

class TriangleReference( SimplexReference ):
  '2D simplex'

  def __init__( self ):
    SimplexReference.__init__( self, ndims=2 )

  @cache.property
  def child_transforms( self ):
    return transform.affine(1,[0,0],2), transform.affine(1,[0,1],2), transform.affine(1,[1,0],2), transform.affine([[-1,0],[1,1]],[1,0],2)

  @property
  def child_refs( self ):
    return self, self, self, self

  def stdfunc( self, degree ):
    return PolyTriangle(degree)

  def getischeme_contour( self, n ):
    p = numpy.arange( n+1, dtype=float ) / (n+1)
    z = numpy.zeros_like( p )
    return numpy.hstack(( [1-p,p], [z,1-p], [p,z] )).T, None

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
    ] if degree <= 1 else [
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
    return coords.T, weights

  def getischeme_bezier( self, np ):
    points = numpy.linspace( 0, 1, np )
    return numpy.array([ [x,y] for i, y in enumerate(points) for x in points[:np-i] ]), None

  def nvertices_by_level( self, n ):
    m = 2**n + 1
    return ((m+1)*m) // 2

  def child_divide( self, vals, n ):
    assert len(vals) == self.nvertices_by_level(n)
    np = 1 + 2**n # points along parent edge
    mp = 1 + 2**(n-1) # points along child edge
    cvals = []
    for i in range(mp):
      j = numpy.arange(mp-i)
      cvals.append( [ vals[b+a*np-(a*(a-1))//2] for a, b in [(i,j),(mp-1+i,j),(i,mp-1+j),(i+j,mp-1-j)] ] )
    return numpy.concatenate( cvals, axis=1 )

  def inside( self, points, eps=0 ):
    points = numpy.asarray( points, dtype=float )
    assert points.ndim == 2 and points.shape[1] == self.ndims
    baricentric = numpy.linalg.solve( self.vertices[1:]-self.vertices[0], (points-self.vertices[0]).T ).T
    return (baricentric >= -eps).all() and (baricentric <= 1+eps).all() and (baricentric.sum(1) <= 1+eps).all()

class TetrahedronReference( SimplexReference ):
  '3D simplex'

  def __init__( self ):
    SimplexReference.__init__( self, ndims=3 )

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
           numpy.concatenate( [ [w*self.volume] * len(i) for i, c, w in icw ] )

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

  def nvertices_by_level( self, n ):
    return self.ref1.nvertices_by_level(n) * self.ref2.nvertices_by_level(n)

  def child_divide( self, vals, n ):
    np1 = self.ref1.nvertices_by_level(n)
    np2 = self.ref2.nvertices_by_level(n)
    return [ v2.swapaxes(0,1).reshape((-1,)+vals.shape[1:])
      for v1 in self.ref1.child_divide( vals.reshape((np1,np2)+vals.shape[1:]), n )
        for v2 in self.ref2.child_divide( v1.swapaxes(0,1), n ) ]

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

  def __str__( self ):
    return '%s*%s' % ( self.ref1, self.ref2 )

  def stdfunc( self, degree ):
    return self.ref1.stdfunc(degree) * self.ref2.stdfunc(degree)

  def getischeme_vtk( self ):
    if self.ref1.ndims == self.ref2.ndims == 1:
      points = numpy.empty([ 2, 2, 2 ])
      points[...,:1] = self.ref1.vertices[:,_]
      points[0,:,1:] = self.ref2.vertices
      points[1,:,1:] = self.ref2.vertices[::-1]
    elif self.ref1.ndims == 1 and self.ref2.ndims == 2:
      points = numpy.empty([ 2, self.ref2.nverts, 3 ])
      points[...,:1] = self.ref1.vertices[:,_]
      points[...,1:] = self.ref2.vertices[_,:]
    elif self.ref1.ndims == 2 and self.ref2.ndims == 1:
      points = numpy.empty([ 2, self.ref1.nverts, 3 ])
      points[...,:2] = self.ref1.vertices[_,:]
      points[...,2:] = self.ref2.vertices[:,_]
    else:
      raise NotImplementedError
    return points.reshape( self.nverts, self.ndims ), None

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

  @property
  def _ribbons( self ):
    ribbons = []
    for iedge1 in range( self.ref1.nedges ):
      #iedge = self.ref1.edge_refs[iedge] * self.ref2
      for iedge2 in range( self.ref2.nedges ):
        #jedge = self.ref1 * self.ref2.edge_refs[jedge]
        jedge1 = self.ref1.nedges + iedge2
        jedge2 = iedge1
        if self.ref1.ndims > 1:
          iedge2 += self.ref1.edge_refs[iedge1].nedges
        ribbons.append( ((iedge1,iedge2),(jedge1,jedge2)) )
    if self.ref1.ndims >= 2:
      ribbons.extend( self.ref1.ribbons )
    if self.ref2.ndims >= 2:
      ribbons.extend( ((iedge1+self.ref1.nedges,iedge2+self.ref1.nedges),
                       (jedge1+self.ref1.nedges,jedge2+self.ref1.nedges)) for (iedge1,iedge2), (jedge1,jedge2) in self.ref2.ribbons )
    return ribbons

  @cache.property
  def child_transforms( self ):
    return [ transform.tensor(trans1,trans2) for trans1 in self.ref1.child_transforms for trans2 in self.ref2.child_transforms ]

  @property
  def child_refs( self ):
    return tuple( child1 * child2 for child1 in self.ref1.child_refs for child2 in self.ref2.child_refs )

  def inside( self, points, eps=0 ):
    points = numpy.asarray( points, dtype=float )
    assert points.ndim == 2 and points.shape[1] == self.ndims
    return self.ref1.inside(points[:,:self.ref1.ndims],eps) & self.ref2.inside(points[:,self.ref1.ndims:],eps)

  @property
  def simplices( self ):
    return [ ( transform.tensor(trans1,trans2), TensorReference( simplex1, simplex2 ) ) for trans1, simplex1 in self.ref1.simplices for trans2, simplex2 in self.ref2.simplices ]

class Cone( Reference ):
  'cone'

  def __init__( self, edgeref, etrans, tip ):
    assert etrans.fromdims == edgeref.ndims
    assert etrans.todims == len(tip)
    vertices = numpy.vstack([ [tip], etrans.apply( edgeref.vertices ) ])
    Reference.__init__( self, vertices )
    self.edgeref = edgeref
    self.etrans = etrans
    self.tip = tip
    ext = etrans.ext
    self.extnorm = numpy.linalg.norm( ext )
    self.height = numpy.dot( etrans.offset - tip, ext ) / self.extnorm
    assert self.height >= 0, 'tip is positioned at the negative side of edge'
    if core.getprop( 'selfcheck', False ):
      self.check_edges()

  @property
  def volume( self ):
    return self.edgeref.volume * self.extnorm * self.height / self.ndims

  @cache.property
  def edge_transforms( self ):
    edge_transforms = [ self.etrans ]
    if self.edgeref.ndims > 0:
      for trans, edge in self.edgeref.edges:
        if edge:
          b = self.etrans.apply( trans.offset )
          A = numpy.hstack([ numpy.dot( self.etrans.linear, trans.linear ), (self.tip-b)[:,_] ])
          newtrans = transform.affine( A, b, isflipped=self.etrans.isflipped^trans.isflipped^(self.ndims%2==1) ) # isflipped logic tested up to 3D
          edge_transforms.append( newtrans )
    else:
      edge_transforms.append( transform.affine( numpy.zeros((1,0)), self.tip, isflipped=not self.etrans.isflipped ) )
    return edge_transforms

  @cache.property
  def edge_refs( self ):
    edge_refs = [ self.edgeref ]
    if self.edgeref.ndims > 0:
      extrudetrans = transform.affine( numpy.eye(self.ndims-1)[:,:-1], numpy.zeros(self.ndims-1), isflipped=self.ndims%2==0 )
      tip = numpy.array( [0]*(self.ndims-2)+[1], dtype=float )
      edge_refs.extend( edge.cone( extrudetrans, tip ) for edge in self.edgeref.edge_refs if edge )
    else:
      edge_refs.append( getsimplex(0) )
    return edge_refs

  def getischeme_gauss( self, degree ):
    if self.nverts == self.ndims+1: # simplex
      spoints, sweights = getsimplex(self.ndims).getischeme_gauss( degree )
      offset = self.vertices[0,:]
      linear = self.vertices[1:,:] - offset
      points = numpy.dot( spoints, linear ) + offset
      weights = sweights * abs(numpy.linalg.det(linear))
    else:
      epoints, eweights = self.edgeref.getischeme( 'gauss{}'.format(degree) )
      tpoints, tweights = getsimplex(1).getischeme_gauss( degree + self.ndims - 1 )
      tx, = tpoints.T
      points = ( tx[:,_,_] * (self.etrans.apply(epoints)-self.tip)[_,:,:] + self.tip ).reshape( -1, self.ndims )
      wx = tx**(self.ndims-1) * tweights * self.extnorm * self.height
      weights = ( eweights[_,:] * wx[:,_] ).ravel()
    return points, weights

  def getischeme_bezier( self, degree ):
    assert self.nverts == self.ndims+1
    spoints, none = getsimplex(self.ndims).getischeme_bezier( degree )
    offset = self.vertices[0,:]
    linear = self.vertices[1:,:] - offset
    return numpy.dot( spoints, linear ) + offset, None

  def getischeme_vtk( self ):
    if self.nverts == 4 and self.ndims==3: # tetrahedron
      I = slice(None)
    elif self.nverts == 5 and self.ndims==3: # pyramid
      I = numpy.array([1,2,4,3,0])
    elif self.nverts == 3 and self.ndims==2: # triangle
      I = slice(None)
    else:
      raise Exception( 'invalid number of points: {}'.format(self.nverts) )
    return self.vertices[I], None

  @property
  def simplices( self ):
    if self.nverts == self.ndims+1 or self.edgeref.ndims == 2 and self.edgeref.nverts == 4: # simplices and square-based pyramids are ok
      return [ ( transform.identity, self ) ]
    return [ ( transform.identity, ref.cone(self.etrans<<trans,self.tip) ) for trans, ref in self.edgeref.simplices ]

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

class OwnChildReference( Reference ):
  'forward self as child'

  def __init__( self, baseref ):
    self.baseref = baseref
    self.child_refs = baseref,
    self.child_transforms = transform.identity,
    self.interfaces = ()
    Reference.__init__( self, baseref.vertices )

  @property
  def edge_transforms( self ):
    return self.baseref.edge_transforms

  @property
  def edge_refs( self ):
    return [ OwnChildReference(edge) for edge in self.baseref.edge_refs ]

  def stdfunc( self, degree ):
    return self.baseref.stdfunc( degree )

  @property
  def volume( self ):
    return self.baseref.volume

  def getischeme( self, ischeme ):
    return self.baseref.getischeme( ischeme )

  @property
  def simplices( self ):
    return self.baseref.simplices

class WithChildrenReference( Reference ):
  'base reference with explicit children'

  def __init__( self, baseref, child_refs ):
    assert isinstance( child_refs, tuple ) and len(child_refs) == baseref.nchildren and any(child_refs) and child_refs != baseref.child_refs
    assert all( isinstance(child_ref,Reference) for child_ref in child_refs )
    assert all( child_ref.ndims == baseref.ndims for child_ref in child_refs )
    self.baseref = baseref
    self.child_transforms = baseref.child_transforms
    self.child_refs = child_refs
    Reference.__init__( self, baseref.vertices )
    if core.getprop( 'selfcheck', False ):
      self.check_edges()

  def stdfunc( self, degree ):
    return self.baseref.stdfunc( degree )

  @property
  def interfaces( self ):
    return self.baseref.interfaces

  @property
  def volume( self ):
    return sum( abs(trans.det) * ref.volume for trans, ref in self.children )

  def nvertices_by_level( self, n ):
    return self.baseref.nvertices_by_level(n)

  def child_divide( self, vals, n ):
    return self.baseref.child_divide( vals, n )

  __sub__ = lambda self, other: self.empty if other in (self,self.baseref) else self.baseref.with_children( self_child-other_child for self_child, other_child in zip( self.child_refs, other.child_refs ) ) if isinstance( other, WithChildrenReference ) and other.baseref in (self,self.baseref) else NotImplemented
  __rsub__ = lambda self, other: self.baseref.with_children( other_child - self_child for self_child, other_child in zip( self.child_refs, other.child_refs ) ) if other == self.baseref else NotImplemented
  __and__ = lambda self, other: self if other == self.baseref else other if isinstance(other,WithChildrenReference) and self == other.baseref else self.baseref.with_children( self_child & other_child for self_child, other_child in zip( self.child_refs, other.child_refs ) ) if isinstance( other, WithChildrenReference ) and other.baseref == self.baseref else NotImplemented
  __or__ = lambda self, other: other if other == self.baseref else self.baseref.with_children( self_child | other_child for self_child, other_child in zip( self.child_refs, other.child_refs ) ) if isinstance( other, WithChildrenReference ) and other.baseref == self.baseref else NotImplemented

  @cache.property
  def __extra_edges( self ):
    interfaces = []
    for (ichild,iedge), (jchild,jedge) in self.interfaces:
      child1 = self.child_refs[ichild]
      child2 = self.child_refs[jchild]
      edge1 = child1.edge_refs[iedge] if child1 else EmptyReference(self.ndims-1)
      edge2 = child2.edge_refs[jedge] if child2 else EmptyReference(self.ndims-1)
      if edge1 - edge2:
        trans = self.child_transforms[ichild] << child1.edge_transforms[iedge]
        if trans not in self.baseref.edge_transforms:
          interfaces.append(( ichild, iedge, trans, edge1-edge2 ))
      if edge2 - edge1:
        trans = self.child_transforms[jchild] << child2.edge_transforms[jedge]
        if trans not in self.baseref.edge_transforms:
          interfaces.append(( jchild, jedge, trans, edge2-edge1 ))
    return interfaces

  def subvertex( self, ichild, i ):
    assert 0<=ichild<self.nchildren
    npoints = 0
    for childindex, child in enumerate(self.child_refs):
      if child:
        points, weights = child.getischeme( 'vertex%d' % (i-1) )
        assert weights is None
        if childindex == ichild:
          rng = numpy.arange( npoints, npoints+len(points) )
        npoints += len(points)
      elif ichild==childindex:
        rng = numpy.array([],dtype=int)
    return npoints, rng

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
    edge_transforms.extend( trans for ichild, iedge, trans, ref in self.__extra_edges )
    return tuple(edge_transforms)

  @cache.property
  def edge_refs( self ):
    # to avoid circular references we cannot mention 'self' inside getedgeref
    def getedgeref( iedge, baseref=self.baseref, child_refs=self.child_refs, edge2children=self.edge2children ):
      baseedge = baseref.edge_refs[iedge]
      return baseedge and baseedge.with_children( child_refs[jchild].edge_refs[jedge] if child_refs[jchild] else EmptyReference(baseref.ndims-1) for jchild, jedge in edge2children[iedge] )
    items = [ cache.Tuple.unknown ] * self.baseref.nedges
    for mychild, basechild in zip( self.child_refs, self.baseref.child_refs ):
      if mychild:
        items.extend( OwnChildReference(edge) for edge in mychild.edge_refs[basechild.nedges:] )
    items.extend( OwnChildReference(ref) for ichild, iedge, trans, ref in self.__extra_edges )
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
    edge2children.extend( [(ichild,iedge)] for ichild, iedge, trans, ref in self.__extra_edges )
    return tuple( edge2children )

class MosaicReference( Reference ):
  'triangulation'

  def __init__( self, baseref, edge_refs, midpoint ):
    assert len(edge_refs) == baseref.nedges
    self.baseref = baseref
    self._edge_refs = tuple( edge_refs )
    self._midpoint = midpoint

    self.edge_refs = list( edge_refs )
    self.edge_transforms = list( baseref.edge_transforms )

    if baseref.ndims == 1:

      nz = [ i for i, edge in enumerate(edge_refs) if edge ]
      if len(nz) == 1:
        self.edge_refs.append( getsimplex(0) )
        self.edge_transforms.append( transform.affine( linear=numpy.zeros((1,0)), offset=midpoint, isflipped=not baseref.edge_transforms[nz[0]].isflipped ) )
      else:
        assert len(nz) == 2

    else:

      newedges = [ ( etrans1, etrans2, edge ) for (etrans1,orig), new in zip( baseref.edges, edge_refs ) for etrans2, edge in new.edges[orig.nedges:] ]
      for (iedge1,iedge2), (jedge1,jedge2) in baseref.ribbons:
        Ei = edge_refs[iedge1]
        ei = Ei.edge_refs[iedge2] if Ei else EmptyReference(Ei.ndims-1)
        Ej = edge_refs[jedge1]
        ej = Ej.edge_refs[jedge2] if Ej else EmptyReference(Ej.ndims-1)
        ejsubi = ej - ei
        if ejsubi:
          newedges.append(( self.edge_transforms[jedge1], Ej.edge_transforms[jedge2], ejsubi ))
        eisubj = ei - ej
        if eisubj:
          newedges.append(( self.edge_transforms[iedge1], Ei.edge_transforms[iedge2], eisubj ))

      extrudetrans = transform.affine( numpy.eye(baseref.ndims-1)[:,:-1], numpy.zeros(baseref.ndims-1), isflipped=baseref.ndims%2==0 )
      tip = numpy.array( [0]*(baseref.ndims-2)+[1], dtype=float )
      for etrans, trans, edge in newedges:
        b = etrans.apply( trans.offset )
        A = numpy.hstack([ numpy.dot( etrans.linear, trans.linear ), (midpoint-b)[:,_] ])
        newtrans = transform.affine( A, b, isflipped=etrans.isflipped^trans.isflipped^(baseref.ndims%2==1) ) # isflipped logic tested up to 3D
        self.edge_transforms.append( newtrans )
        self.edge_refs.append( edge.cone( extrudetrans, tip ) )

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

  def __and__( self, other ):
    if other in (self,self.baseref):
      return self
    if isinstance( other, MosaicReference ) and other.baseref == self:
      return other
    if isinstance( other, MosaicReference ) and self.baseref == other.baseref and numpy.all( other._midpoint == self._midpoint ):
      isect_edge_refs = [ selfedge & otheredge for selfedge, otheredge in zip( self._edge_refs, other._edge_refs ) ]
      if not any(isect_edge_refs):
        return self.empty
      return MosaicReference( self.baseref, isect_edge_refs, self._midpoint )
    return NotImplemented

  def __or__( self, other ):
    if other in (self,self.baseref):
      return other
    if isinstance( other, MosaicReference ) and self.baseref == other.baseref and numpy.all( other._midpoint == self._midpoint ):
      union_edge_refs = [ selfedge | otheredge for selfedge, otheredge in zip( self._edge_refs, other._edge_refs ) ]
      if tuple(union_edge_refs) == tuple(self.baseref.edge_refs):
        return self.baseref
      return MosaicReference( self.baseref, union_edge_refs, self._midpoint )
    return NotImplemented

  def __sub__( self, other ):
    if other in (self,self.baseref):
      return self.empty
    if isinstance( other, MosaicReference ) and other.baseref == self:
      inv_edge_refs = [ baseedge - edge for baseedge, edge in zip( self.edge_refs, other._edge_refs ) ]
      return MosaicReference( self, inv_edge_refs, other._midpoint )
    return NotImplemented

  def __rsub__( self, other ):
    if other == self.baseref:
      inv_edge_refs = [ baseedge - edge for baseedge, edge in zip( other.edge_refs, self._edge_refs ) ]
      return MosaicReference( other, inv_edge_refs, self._midpoint )
    return NotImplemented

  def nvertices_by_level( self, n ):
    return self.baseref.nvertices_by_level( n )

  @cache.property
  def subrefs( self ):
    return [ ref.cone(trans,self._midpoint) for trans, ref in zip( self.baseref.edge_transforms, self._edge_refs ) if ref ]

  def stdfunc( self, degree ):
    return self.baseref.stdfunc( degree )

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

  def __getnewargs__( self ):
    return self.std1, self.std2

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

  def __getnewargs__( self ):
    return self.poly,

  def eval( self, points, grad=0 ):
    'evaluate'

    assert points.shape[-1] == 1
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

    self.order = order
    assert order in (0,1)
    StdElem.__init__( self, ndims=2, nshapes=3 if order else 1 )

  def __getnewargs__( self ):
    return self.order,

  def eval( self, points, grad=0 ):
    'eval'

    if self.order == 0:
      data = numpy.ones( (1,1) ) if grad == 0 \
        else numpy.zeros( (1,1)+(2,)*grad )
    else:
      data = numpy.concatenate( [ 1-points.sum(1)[:,_], points ], axis=1 ) if grad == 0 \
        else numpy.array( [[[-1,-1],[1,0],[0,1]]], dtype=float ) if grad == 1 \
        else numpy.zeros( (1,3)+(2,)*grad )

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

  def __getnewargs__( self ):
    return ()

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

  def __getnewargs__( self ):
    return self.stdelem, self.extraction

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
  Simplex_by_dim = PointReference, LineReference, TriangleReference, TetrahedronReference
  return Simplex_by_dim[ndims]()

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
