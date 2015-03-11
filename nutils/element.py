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

  __slots__ = 'transform', 'reference', 'opposite'

  def __init__( self, reference, trans, opposite=None ):
    assert trans.fromdims == reference.ndims
    self.reference = reference
    self.transform = trans.canonical
    self.opposite = opposite.canonical if opposite is not None else self.transform

  def __hash__( self ):
    return object.__hash__( self )

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
  def edges( self ):
    return [ self.edge(i) for i in range(self.nverts) ]
    
  def edge( self, iedge ):
    trans, edge = self.reference.edges[iedge]
    return Element( edge, self.transform << trans, self.opposite << trans )

  def getedge( self, trans ):
    iedge = self.reference.edge_transforms.index(trans)
    edge = self.reference.edge_refs[iedge]
    return edge and Element( edge, self.transform << trans, self.opposite << trans )

  @property
  def children( self ):
    return [ Element( child, self.transform << trans, self.opposite << trans )
      for trans, child in self.reference.children ]

  def trim( self, levelset, maxrefine, denom, check, fcache ):
    'trim element along levelset'

    assert self.transform == self.opposite
    pos, neg, ifaces, posouter, negouter = self.reference.trim( (self.transform,levelset), maxrefine, denom, check, fcache )
    poselem = pos and Element( pos, self.transform )
    negelem = neg and Element( neg, self.transform )
    ifaces = [ Element( edge, self.transform<<postrans, self.transform<<negtrans ) for postrans, negtrans, edge in ifaces ]
    posouter = [ Element( edge, self.transform<<trans ) for trans, edge in zip( self.reference.edge_transforms, posouter ) if edge ]
    negouter = [ Element( edge, self.transform<<trans ) for trans, edge in zip( self.reference.edge_transforms, negouter ) if edge ]
    return poselem, negelem, ifaces, posouter, negouter

  @property
  def flipped( self ):
    return Element( self.reference, self.opposite, self.transform )

  def without_edge( self, edge ):
    iedge = self.reference.edge_transforms.index( edge.transform.fromtrans( self.transform ) )
    edge_refs = list( self.reference.edge_refs )
    edge_refs[iedge] -= edge.reference
    return Element( self.reference.with_edges( edge_refs ), self.transform, self.opposite )

  @property
  def simplices( self ):
    return [ Element( reference, self.transform << trans, self.opposite << trans )
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

  def __xor__( self, other ):
    if other is None:
      return self
    if other == self:
      return None
    return NotImplemented

  def __rxor__( self, other ):
    return self.__xor__( self, other )

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
    return zip( self.edge_transforms, self.edge_refs )

  @property
  def children( self ):
    return zip( self.child_transforms, self.child_refs )

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
  def edge2vertex( self ):
    edge2vertex = []
    for trans, edge in self.edges:
      where = numpy.zeros( self.nverts, dtype=bool )
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

  def with_edges( self, edge_refs ):
    edge_refs = tuple(edge_refs)
    if edge_refs == getattr(self,'edge_refs',None):
      return self
    return WithEdgesReference( self, edge_refs )

  def with_children( self, child_refs ):
    child_refs = tuple(child_refs)
    if not any( child_refs ):
      return None
    if child_refs == self.child_refs:
      return self
    return WithChildrenReference( self, child_refs )

  def triangulate( self, levels, denom, check ):
    'triangulate based on vertex levelset values'

    assert numeric.isint( denom )
    assert levels.shape == (self.nverts,)

    numer = list( self.vertices * denom )
    vertex2edge = list(self.edge2vertex.T)
    oniface = numpy.zeros( len(levels), dtype=bool )
    for i, j in self.ribbon2vertices:
      a, b = levels[[i,j]]
      if numpy.sign(a) != numpy.sign(b):
        x = int( denom * a / float(a-b) + .5 ) # round to [0,1,..,denom]
        if x == 0:
          oniface[i] = True
        elif x == denom:
          oniface[j] = True
        else:
          numer.append( numpy.dot( (denom-x,x), self.vertices[[i,j]] ) )
          vertex2edge.append( vertex2edge[i] & vertex2edge[j] )

    coords = rational.Rational( numer, denom )
    vertex2edge = numpy.array(vertex2edge)
    vsigns = numpy.zeros( len(coords), dtype=int )
    vsigns[~oniface] = numpy.sign(levels[~oniface])

    haspos = +1 in vsigns
    if not haspos or -1 not in vsigns:
      empty = (None,)*self.nedges
      if ( vsigns == 0 ).sum() < self.ndims:
        return (self,None,(),empty,empty) if haspos else (None,self,(),empty,empty)
      inner = []
      outer = list( empty )
      for iedge, onedge in enumerate( vertex2edge.T ):
        oniface = vsigns[onedge] == 0
        etrans, edge = self.edges[iedge]
        if oniface.all():
          outer[iedge] = edge
          inner.append( None )
        elif oniface.sum() >= self.ndims:
          ecoords = etrans.solve( coords[onedge] )
          innertrans = []
          outertrans = []
          for tri in util.delaunay( ecoords.astype(float) ):
            strans = transform.simplex( ecoords[tri] )
            ( outertrans if oniface[tri].all() else innertrans ).append( strans )
          assert innertrans and outertrans
          outer[iedge] = MultiSimplexReference(edge,outertrans,complement=innertrans)
          inner.append( MultiSimplexReference(edge,innertrans,complement=outertrans) )
        else:
          inner.append( edge )
      ref = self.with_edges( inner )
      if check:
        refcheck( ref, ref.edges + zip(self.edge_transforms,outer) )
      return (ref,None,(),outer,empty) if haspos else (None,ref,(),empty,outer)

    postri, negtri = signed_triangulate( coords.astype(float), vsigns )
    assert len(postri) and len(negtri)

    mosaics = ()
    posouter, negouter = [None]*self.nedges, [None]*self.nedges
    posifaces, negifaces = {}, {}
    for triangulation, ifaces, outer, sign in (postri,posifaces,posouter,+1), (negtri,negifaces,negouter,-1):
      sref = MultiSimplexReference( self, [ transform.simplex(coords[tri]) for tri in triangulation ] )
      edges = []
      for iedge, (etrans,baseedge) in enumerate( self.edges ):
        onedge = vertex2edge.T[iedge]
        signs = vsigns[onedge]
        if not any( signs == sign ):
          edge = None
        elif not any( signs == -sign ) and sum( signs == 0 ) < self.ndims:
          edge = baseedge
        else:
          transforms = []
          outertransforms = []
          for tri in triangulation:
            etri = tri[ onedge[tri] ]
            if len(etri) == self.ndims:
              trioffset = coords[etri[0]]
              trilinear = ( coords[etri[1:]] - trioffset ).T
              slinear, soffset = rational.solve( etrans.linear, trilinear, trioffset - etrans.offset )
              strans = transform.affine( slinear, soffset ) # etrans << strans == affine( trilinear, trioffset ) ~/flip
              if any( vsigns[etri] ):
                transforms.append( strans )
              else:
                ifaces[ tuple(sorted(etri)) ] = None
                outertransforms.append( strans )
          if outertransforms:
            outer[iedge] = MultiSimplexReference( baseedge, outertransforms, complement=transforms )
          edge = MultiSimplexReference( baseedge, transforms, complement=outertransforms ) if transforms else None
        edges.append( edge )
      mosaics += sref.with_edges( edges ),
      for tri in triangulation:
        where, = numpy.where( vsigns[tri] != 0 )
        if len(where) <= 1:
          for iedge in where if len(where) else range(self.ndims+1):
            key = tuple( sorted( tuple(tri[:iedge]) + tuple(tri[iedge+1:]) ) )
            if key in ifaces:
              del ifaces[key]
            else:
              ifaces[key] = tri, iedge

    ifaces = []
    for key, (tri,iedge) in posifaces.items():
      negifaces.pop( key )
      etrans, edge = getsimplex(self.ndims).edges[iedge]
      trans = ( transform.simplex( coords[tri] ) << etrans ).flat
      ifaces.append(( trans, trans.flipped, edge ))
    assert not negifaces

    if check:
      posmosaic, negmosaic = mosaics
      refcheck( posmosaic, posmosaic.edges + [ (postrans,edge) for postrans, negtrans, edge in ifaces ] + zip(self.edge_transforms,posouter) )
      refcheck( negmosaic, negmosaic.edges + [ (negtrans,edge) for postrans, negtrans, edge in ifaces ] + zip(self.edge_transforms,negouter) )

    return mosaics + (ifaces,posouter,negouter)

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

    allpos = all( levels > 0 )
    if allpos or all( levels < 0 ):
      return (self,None,[],[None]*self.nedges,[None]*self.nedges) if allpos \
        else (None,self,[],[None]*self.nedges,[None]*self.nedges)

    if not maxrefine:
      assert evaluated_levels, 'failed to evaluate levelset up to level maxrefine'
      return self.triangulate( levels, denom, check )

    poselems, negelems = [], []
    posinner, neginner = {}, {}
    posouter = [ [None]*edge.nchildren for edge in self.edge_refs ]
    negouter = [ [None]*edge.nchildren for edge in self.edge_refs ]
    ifaces = []
    for ichild, (ctrans,child) in enumerate( self.children ):
      if evaluated_levels:
        N, I = fcache( self.subvertex, ichild, maxrefine )
        assert len(levels) == N
        childlevels = levels[I]
      else:
        trans, levelfun = levels
        childlevels = trans << ctrans, levelfun
      cposelem, cnegelem, cifaces, cposouter, cnegouter = child.trim( childlevels, maxrefine-1, denom, check, fcache )
      poselems.append( cposelem )
      negelems.append( cnegelem )
      ifaces.extend( ( (ctrans<<postrans).flat, (ctrans<<negtrans).flat, edge ) for postrans, negtrans, edge in cifaces )
      for childouter, outer, inner in (cposouter,posouter,posinner), (cnegouter,negouter,neginner):
        for ichildedge, edge in enumerate(childouter):
          if not edge:
            continue
          jchild, jedge, isouter = self.childedgemap[ichild][ichildedge]
          if isouter:
            outer[jedge][jchild] |= edge
          else:
            ichild2, ichildedge2, edge2 = inner.pop( (jchild,jedge), (ichild,ichildedge,None) )
            assert (ichild2,ichildedge2) == (ichild,ichildedge)
            remaining = edge^edge2
            if remaining:
              inner[ichild,ichildedge] = jchild, jedge, remaining

    posouter = [ edge.with_children( child_refs ) for edge, child_refs in zip( self.edge_refs, posouter ) ]
    negouter = [ edge.with_children( child_refs ) for edge, child_refs in zip( self.edge_refs, negouter ) ]

    if not any(negelems):

      assert not ifaces and not any(negouter)
      assert not posinner and not neginner
      posmosaic = self.with_edges( edge - rmedge for edge, rmedge in zip( self.edge_refs, posouter ) )
      negmosaic = None

    elif not any(poselems):

      assert not ifaces and not any(posouter)
      assert not posinner and not neginner
      posmosaic = None
      negmosaic = self.with_edges( edge - rmedge for edge, rmedge in zip( self.edge_refs, negouter ) )

    else:

      for (ichild,iedge), (jchild,jedge,edge) in posinner.items():
        ctrans, child = self.children[ichild]
        trans = (ctrans << child.edge_transforms[iedge]).flat
        try:
          ichild2, iedge2, edge2 = neginner.pop( (jchild,jedge) )
        except KeyError:
          elemtrans = self.child_transforms[jchild]
          ref = negelems[jchild]
          trans2, edge2 = ref.edges[jedge]
          assert (elemtrans<<trans2).flat == trans.flipped
          ifaces.append(( trans, trans.flipped, edge|edge2 ))
          edge_refs = list( ref.edge_refs )
          edge_refs[jedge] -= edge2
          negelems[jchild] = ref.with_edges( edge_refs )
        else:
          ifaces.append(( trans, trans.flipped, edge|edge2 ))

      for (ichild,iedge), (jchild,jedge,edge) in neginner.items():
        ctrans, child = self.children[ichild]
        trans = (ctrans << child.edge_transforms[iedge]).flat
        elemtrans = self.child_transforms[jchild]
        ref = poselems[jchild]
        trans2, edge2 = ref.edges[jedge]
        assert (elemtrans<<trans2).flat == trans.flipped
        ifaces.append(( trans.flipped, trans, edge|edge2 ))
        edge_refs = list( ref.edge_refs )
        edge_refs[jedge] -= edge2
        poselems[jchild] = ref.with_edges( edge_refs )

      posmosaic = self.with_children( poselems )
      negmosaic = self.with_children( negelems )

    if check:
      if posmosaic:
        refcheck( posmosaic, posmosaic.edges + [ (postrans,edge) for postrans, negtrans, edge in ifaces ] + list(zip(self.edge_transforms,posouter)) )
      if negmosaic:
        refcheck( negmosaic, negmosaic.edges + [ (negtrans,edge) for postrans, negtrans, edge in ifaces ] + list(zip(self.edge_transforms,negouter)) )

    return posmosaic, negmosaic, ifaces, posouter, negouter

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
  def edge2vertices( self ):
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
    refcheck( self )

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
    refcheck( self )

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
      raise Exception, 'invalid ichild: {}'.format( ichild )

    return ((N+1)*N)//2, numpy.concatenate([ flatten_child(i,numpy.arange(n-i)) for i in range(n) ])

class TetrahedronReference( SimplexReference ):
  '3D simplex'

  def __init__( self ):
    SimplexReference.__init__( self, 3 )
    self.edge_transforms = tuple( transform.simplex( self.vertices[I], isflipped=False ) for I in [[1,2,3],[0,3,2],[3,0,1],[2,1,0]] )
    refcheck( self )

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
    refcheck( self )

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

class WithChildrenReference( Reference ):
  'base reference with explicit children'

  def __init__( self, baseref, child_refs ):
    assert not isinstance( baseref, WithChildrenReference )
    assert isinstance( child_refs, tuple ) and len(child_refs) == baseref.nchildren and any(child_refs) and child_refs != baseref.child_refs
    self.baseref = baseref
    self.edge_transforms = baseref.edge_transforms
    self.child_transforms = baseref.child_transforms
    self.child_refs = child_refs
    Reference.__init__( self, baseref.vertices )

  def __or__( self, other ):
    if other is None:
      return self
    if other == self.baseref:
      return other
    if not isinstance( other, WithChildrenReference ) or other.baseref != self.baseref:
      raise TypeError
    child_refs = tuple( child1 | child2 if child1 or child2 else None for child1, child2 in zip( self.child_refs, other.child_refs ) )
    assert child_refs
    if child_refs == self.baseref.child_refs:
      return self.baseref
    return WithChildrenReference( self.baseref, child_refs )

  def __rsub__( self, baseref ):
    assert self.baseref == baseref
    return baseref.with_children( child - rmchild for child, rmchild in zip( baseref.child_refs, self.child_refs ) )

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
  def edge_refs( self ):
    all_child_refs = [ [None] * baseedge.nchildren for baseedge in self.baseref.edge_refs ]
    for child, row in zip( self.child_refs, self.baseref.childedgemap ):
      if child:
        for edge, (ichild,iedge,isouter) in zip( child.edge_refs, row ):
          if isouter:
            all_child_refs[iedge][ichild] = edge
    return tuple( baseedge.with_children( child_refs ) for baseedge, child_refs in zip( self.baseref.edge_refs, all_child_refs ) )

class WithEdgesReference( Reference ):
  'base reference with explicit edges'

  def __init__( self, baseref, edge_refs ):
    assert not isinstance( baseref, WithEdgesReference )
    assert isinstance( edge_refs, tuple ) and len(edge_refs) == baseref.nedges and any(edge_refs) and edge_refs != getattr(baseref,'edge_refs',None)
    self.baseref = baseref
    self.edge_transforms = baseref.edge_transforms
    self.edge_refs = edge_refs
    Reference.__init__( self, baseref.vertices )

  @property
  def simplices( self ):
    return self.baseref.simplices

  def getischeme( self, ischeme ):
    return self.baseref.getischeme( ischeme )

class MultiSimplexReference( Reference ):
  'triangulation'

  def __init__( self, baseref, transforms, complement=None ):
    self.baseref = baseref
    self.edge_transforms = baseref.edge_transforms
    self.__transforms = tuple(transforms)
    self.__complement = complement and tuple(complement)
    Reference.__init__( self, baseref.vertices )

  def __or__( self, other ):
    if other is None:
      return self
    if other == self.baseref:
      return other
    if not isinstance( other, MultiSimplexReference ) or other.baseref != self.baseref:
      raise TypeError
    assert self.__complement and other.__complement
    total = set(self.__transforms+self.__complement)
    assert set(other.__transforms+other.__complement) == total
    transforms = set(self.__transforms) | set(other.__transforms)
    complement = total - transforms
    if not complement:
      return self.baseref
    return MultiSimplexReference( self.baseref, tuple(transforms), tuple(complement) )

  def __rsub__( self, baseref ):
    assert self.baseref == baseref
    assert self.__complement
    return MultiSimplexReference( self.baseref, self.__complement, self.__transforms )

  @property
  def simplices( self ):
    simplex = getsimplex(self.ndims)
    return [ (trans,simplex) for trans in self.__transforms ]

  def getischeme( self, ischeme ):
    'get integration scheme'
    
    if ischeme.startswith('vertex'):
      return self.baseref.getischeme( ischeme )

    simplex = getsimplex(self.ndims)
    points, weights = simplex.getischeme( ischeme )
    allcoords = numpy.empty( (len(self.__transforms),)+points.shape, dtype=float )
    if weights is not None:
      allweights = numpy.empty( (len(self.__transforms),)+weights.shape, dtype=float )
    for i, trans in enumerate( self.__transforms ):
      allcoords[i] = trans.apply(points)
      if weights is not None:
        allweights[i] = weights * abs(float(trans.det))
    return allcoords.reshape(-1,self.ndims), allweights.ravel() if weights is not None else None


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

@cache.argdict
def gauss( degree ):
  k = numpy.arange( 1, degree // 2 + 1 )
  d = k / numpy.sqrt( 4*k**2-1 )
  x, w = numpy.linalg.eigh( numpy.diagflat(d,-1) ) # eigh operates (by default) on lower triangle
  return (x+1) * .5, w[0]**2

def signed_triangulate( points, vsigns ):
  points = numpy.asarray( points )
  npoints, ndims = points.shape
  triangulation = util.delaunay( points )
  esigns = numpy.array([ +1 if min(s) >= 0 else -1 if max(s) <= 0 else 0 for s in vsigns[triangulation] ])
  if not esigns.all():
    ambiguous, = numpy.where( esigns == 0 )
    extra = []
    for tri in triangulation[ambiguous]:
      I = numpy.where( vsigns[tri] == 0 )[0]
      points = numpy.concatenate([ points, [ numpy.dot( [ 0.99 if i == j else .01/ndims for j in tri ], points[tri] ) for i in I ]], axis=0 )
      extra.extend( I )
    vmap = numpy.concatenate( [ numpy.arange(len(vsigns)), extra ] )
    triangulation = [ tri for tri in vmap[ util.delaunay( points ) ] if len(set(tri)) == len(tri) ]
    esigns = numpy.array([ +1 if min(s) >= 0 else -1 if max(s) <= 0 else 0 for s in vsigns[triangulation] ])
    assert esigns.all()
  for tri in triangulation:
    if numpy.linalg.det( points[tri[1:]] - points[tri[0]] ) < 0:
      tri[-2:] = tri[-1], tri[-2]
  return triangulation[ esigns > 0 ], triangulation[ esigns < 0 ]

def refcheck( ref, edges=None, decimal=10 ):
  if edges is None:
    edges = ref.edges
  x, w = ref.getischeme( 'gauss1' )
  volume = w.sum()
  assert volume > 0
  check_volume = 0
  check_zero = 0
  L = []
  for trans, edge in edges:
    if not edge:
      continue
    xe, we = edge.getischeme( 'gauss1' )
    w_normal = we[:,_] * rational.ext( trans.linear ).astype( float )
    if trans.isflipped:
      w_normal = -w_normal
    L.append(( trans, edge, trans.apply(xe), numpy.linalg.norm(w_normal.sum(0)), w_normal ))
    check_zero += w_normal.sum(0)
    check_volume += numeric.contract( trans.apply(xe), w_normal, axis=0 )
  numpy.testing.assert_almost_equal( check_zero, 0, decimal, '%s fails divergence test' % ref )
  numpy.testing.assert_almost_equal( check_volume, volume, decimal, '%s fails divergence test' % ref )

def getsimplex( ndims ):
  constructors = PointReference, LineReference, TriangleReference, TetrahedronReference
  return constructors[ndims]()


# vim:shiftwidth=2:foldmethod=indent:foldnestmax=2
