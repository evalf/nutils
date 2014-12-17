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
    self.transform = transform.canonical( trans )
    self.opposite = transform.canonical( opposite ) if opposite is not None else self.transform

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

  def findedge( self, trans ):
    edge = self.reference.findedge( trans )
    if not edge:
      return None
    return Element( edge, self.transform << trans, self.opposite << trans )

  @property
  def children( self ):
    return [ Element( child, self.transform << trans, self.opposite << trans )
      for trans, child in self.reference.children ]

  def trim( self, levelset, maxrefine, numer ):
    'trim element along levelset'

    pos, neg = self.reference.trim( self.transform, levelset, maxrefine, numer )
    poselem = negelem = None
    if pos:
      posref, postrim = pos
      poselem = Element( posref, self.transform, self.opposite ), postrim
    if neg:
      negref, negtrim = neg
      negelem = Element( negref, self.transform, self.opposite ), negtrim
    return poselem, negelem

  @property
  def simplices( self ):
    return [ Element( reference, self.transform << trans, self.opposite << trans )
      for trans, reference in self.reference.simplices ]

  def __str__( self ):
    return 'Element(%s)' % self.vertices


## REFERENCE ELEMENTS

class Reference( cache.Immutable ):
  'reference element'

  def __init__( self, vertices ):
    self.vertices = numpy.asarray( vertices )
    assert self.vertices.dtype == int
    self.nverts, self.ndims = self.vertices.shape

  def selfcheck( self, decimal=10 ):
    if self.ndims == 0:
      return
    x, w = self.getischeme( 'gauss1' )
    volume = w.sum()
    assert volume > 0
    check_volume = 0
    check_zero = 0
    L = []
    for trans, edge in self.edges:
      xe, we = edge.getischeme( 'gauss1' )
      w_normal = we[:,_] * rational.ext( trans.linear ).astype( float )
      if trans.isflipped:
        w_normal = -w_normal
      L.append(( trans, edge, trans.apply(xe), numpy.linalg.norm(w_normal.sum(0)), w_normal ))
      check_zero += w_normal.sum(0)
      check_volume += numeric.contract( trans.apply(xe), w_normal, axis=0 )
    numpy.testing.assert_almost_equal( check_zero, 0, decimal, '%s fails divergence test' % self )
    numpy.testing.assert_almost_equal( check_volume, volume, decimal, '%s fails divergence test' % self )

  def findedge( self, trans ):
    return dict( self.edges )[ trans ]

  @property
  def simplices( self ):
    return [ (transform.TransformChain(),self) ]

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

  def __mul__( self, other ):
    assert isinstance( other, Reference )
    return other if self.ndims == 0 \
      else self if other.ndims == 0 \
      else TensorReference( self, other )

  def __pow__( self, n ):
    assert numeric.isint( n ) and n >= 0
    return SimplexReference(0) if n == 0 \
      else self if n == 1 \
      else self * self**(n-1)

  def _trim_triangulate( self, levelset, numer ):
    'triangulate based on vertex levelset values'

    assert numeric.isint( numer )
    assert levelset.shape == (self.nverts,)

    coords = list( self.vertices * numer )
    vmap = list( numpy.arange(self.nverts) )
    for ivert in self.ribbon2vertices:
      a, b = levelset[ivert]
      if numpy.sign(a) != numpy.sign(b):
        x = int( numer * a / float(a-b) + .5 ) # round to [0,1,..,numer]
        if x == 0:
          vmap.append( ivert[0] )
          x = 1e-5
        elif x == numer:
          vmap.append( ivert[1] )
          x = 1 - 1e-5
        else:
          vmap.append( len(vmap) )
        coords.append( numpy.dot( (numer-x,x), self.vertices[ivert] ) )

    triangulation = util.delaunay( coords, True )

    sides = ([],{}), ([],{}) # elems, belems
    ifaces = {}, {}
    simplex = SimplexReference( self.ndims )
    for tri in triangulation:
      onvertex = tri < self.nverts
      if not onvertex.any():
        isneg = False # if all points have level 0, add element to positive side for now
      else:
        minlevel, maxlevel = util.minmax( levelset[tri[onvertex]] )
        assert minlevel * maxlevel >= 0, 'element did not separate in a positive and negative part'
        isneg = minlevel < 0
      vtri = numpy.take( vmap, tri )
      offset = coords[vtri[0]]
      matrix = numpy.array([ coords[i] - offset for i in vtri[1:] ]).T
      strans = transform.affine( matrix, offset, numer )
      elems, belems = sides[isneg]
      if onvertex.sum() <= 1 and util.allunique( vtri[~onvertex] ):
        iface = ifaces[isneg]
        for iedge in numpy.arange( simplex.nverts )[ onvertex if onvertex.any() else slice(None) ]:
          key = tuple( sorted( ivert for i, ivert in enumerate(vtri) if i != iedge ) )
          if key in iface:
            del belems[iface[key]]
            del iface[key]
          else:
            etrans, edge = simplex.edges[ iedge ]
            mtrans = ( strans << etrans ).flat
            iface[key] = mtrans
            belems[mtrans] = edge
      volume = numpy.linalg.det( matrix.astype(float) )
      assert volume >= 0
      if volume:
        elems.append(( strans, simplex ))

    poskeys, negkeys = map( set, ifaces )
    if poskeys != negkeys:
      log.warning( 'leftover interfaces:',
        ', '.join( str(ifaces[0][key]) for key in poskeys-negkeys ), 'and',
        ', '.join( str(ifaces[1][key]) for key in negkeys-poskeys ) )

    return sides

  def trim( self, trans, levelset, maxrefine, numer ):
    'trim element along levelset'

    assert maxrefine >= 0
    assert numeric.isint( numer )

    if trans: # levelset is not evaluated
      try:
        levelset = levelset.eval( Element(self,trans), 'vertex%d' % maxrefine )
      except:
        pass
      else:
        trans = False

    if not trans: # levelset is evaluated
      allpos = numpy.greater( levelset, 0 ).all() # strict
      if allpos or numpy.less( levelset, 0 ).all(): # strict
        retval = self, ()
        return (retval,None) if allpos else (None,retval)

    if not maxrefine and isinstance( self, MosaicReference ):
      maxrefine += 1 # for trimming of trimmed elements, hack until MosaicReference.trim is implemented

    if not maxrefine:
      assert not trans, 'failed to evaluate levelset up to level maxrefine'
      sides = self._trim_triangulate( levelset, numer )
    else:
      sides = ([],{}), ([],{}) # elems, belems
      for ichild, (ctrans,child) in enumerate( self.children ):
        if trans:
          trims = child.trim( trans << ctrans, levelset, maxrefine-1, numer )
        else:
          N, I = self.subvertex(ichild,maxrefine)
          assert len(levelset) == N
          trims = child.trim( False, levelset[I], maxrefine-1, numer )
        for isneg in [ i for i, trim in enumerate(trims) if trim ]:
          elems, belems = sides[isneg]
          ref, iface = trims[isneg]
          elems.append( (ctrans,ref) )
          belems.update({ (ctrans << etrans).flat: ForwardReference( ref.findedge(etrans) ) for etrans in iface })

    (poselems,posbelems), (negelems,negbelems) = sides

    identity = (transform.identity,self),
    if not poselems:
      posmosaic = None
      negmosaic = MosaicReference( self.ndims, identity, negbelems ) if negbelems else self, negbelems.keys()
    elif not negelems:
      posmosaic = MosaicReference( self.ndims, identity, posbelems ) if posbelems else self, posbelems.keys()
      negmosaic = None
    else:
      posmosaic = MosaicReference( self.ndims, poselems, posbelems ), posbelems.keys()
      negmosaic = MosaicReference( self.ndims, negelems, negbelems ), negbelems.keys()

    return posmosaic, negmosaic

class SimplexReference( Reference ):
  'simplex reference'

  def __init__( self, ndims ):
    assert ndims >= 0
    vertices = numpy.concatenate( [ numpy.zeros(ndims,dtype=int)[_,:],
                                    numpy.eye(ndims,dtype=int) ], axis=0 )
    Reference.__init__( self, vertices )
    self.selfcheck()
    self._bernsteincache = [] # TEMPORARY

  def stdfunc( self, degree ):
    if self.ndims == 1:
      if len(self._bernsteincache) <= degree or self._bernsteincache[degree] is None:
        self._bernsteincache += [None] * (degree-len(self._bernsteincache))
        self._bernsteincache.append( PolyLine( PolyLine.bernstein_poly(degree) ) )
      return self._bernsteincache[degree]
    if self.ndims == 2:
      return PolyTriangle(degree)
    raise NotImplementedError

  @cache.property
  def ribbon2vertices( self ):
    return numpy.array([ (i,j) for i in range( self.ndims+1 ) for j in range( i+1, self.ndims+1 ) ])

  @property
  def edge2vertices( self ):
    return ~numpy.eye( self.nverts, dtype=bool )

  def getischeme_contour( self, n ):
    assert self.ndims == 2
    p = numpy.arange( n+1, dtype=float ) / (n+1)
    z = numpy.zeros_like( p )
    return numpy.hstack(( [1-p,p], [z,1-p], [p,z] )).T, None

  def getischeme_vtk( self ):
    assert self.ndims in (2,3)
    return self.vertices.astype(float), None

  def getischeme_gauss( self, degree ):
    '''get integration scheme
    http://www.cs.rpi.edu/~flaherje/pdf/fea6.pdf
    '''
    if isinstance( degree, tuple ):
      assert len(degree) == self.ndims
      degree = sum(degree)

    assert isinstance( degree, int ) and degree >= 0
    if self.ndims == 0: # point
      return numpy.zeros((1,0)), numpy.ones(1)
    if self.ndims == 1: # line
      x, w = gauss( degree )
      return x[:,_], w
    if self.ndims == 2: # triangle: http://www.cs.rpi.edu/~flaherje/pdf/fea6.pdf
      if degree == 1:
        coords = numpy.array( [[1],[1]] ) / 3.
        weights = numpy.array( [1] ) / 2.
      elif degree == 2:
        coords = numpy.array( [[4,1,1],[1,4,1]] ) / 6.
        weights = numpy.array( [1,1,1] ) / 6.
      elif degree == 3:
        coords = numpy.array( [[5,9,3,3],[5,3,9,3]] ) / 15.
        weights = numpy.array( [-27,25,25,25] ) / 96.
      elif degree == 4:
        A = 0.091576213509771; B = 0.445948490915965; W = 0.109951743655322
        coords = numpy.array( [[1-2*A,A,A,1-2*B,B,B],[A,1-2*A,A,B,1-2*B,B]] )
        weights = numpy.array( [W,W,W,1/3.-W,1/3.-W,1/3.-W] ) / 2.
      elif degree == 5:
        A = 0.101286507323456; B = 0.470142064105115; V = 0.125939180544827; W = 0.132394152788506
        coords = numpy.array( [[1./3,1-2*A,A,A,1-2*B,B,B],[1./3,A,1-2*A,A,B,1-2*B,B]] )
        weights = numpy.array( [1-3*V-3*W,V,V,V,W,W,W] ) / 2.
      elif degree == 6:
        A = 0.063089014491502; B = 0.249286745170910; C = 0.310352451033785; D = 0.053145049844816; V = 0.050844906370207; W = 0.116786275726379
        VW = 1/6. - (V+W) / 2.
        coords = numpy.array( [[1-2*A,A,A,1-2*B,B,B,1-C-D,1-C-D,C,C,D,D],[A,1-2*A,A,B,1-2*B,B,C,D,1-C-D,D,1-C-D,C]] )
        weights = numpy.array( [V,V,V,W,W,W,VW,VW,VW,VW,VW,VW] ) / 2.
      else:
        A = 0.260345966079038; B = 0.065130102902216; C = 0.312865496004875; D = 0.048690315425316; U = 0.175615257433204; V = 0.053347235608839; W = 0.077113760890257
        coords = numpy.array( [[1./3,1-2*A,A,A,1-2*B,B,B,1-C-D,1-C-D,C,C,D,D],[1./3,A,1-2*A,A,B,1-2*B,B,C,D,1-C-D,D,1-C-D,C]] )
        weights = numpy.array( [1-3*U-3*V-6*W,U,U,U,V,V,V,W,W,W,W,W,W] ) / 2.
        if degree > 7:
          warnings.warn('Inexact integration for polynomial of degree %i'%degree)
    elif self.ndims == 3: # tetrahedron: http://people.sc.fsu.edu/~jburkardt/datasets/quadrature_rules_tet/quadrature_rules_tet.html'''
      if degree == 1:
        coords = numpy.array( [[1],[1],[1]] ) / 4.
        weights = numpy.array( [1] ) / 6.
      elif degree == 2:
        coords = numpy.array([[0.5854101966249685,0.1381966011250105,0.1381966011250105],
                              [0.1381966011250105,0.1381966011250105,0.1381966011250105],
                              [0.1381966011250105,0.1381966011250105,0.5854101966249685],
                              [0.1381966011250105,0.5854101966249685,0.1381966011250105]]).T
        weights = numpy.array([1,1,1,1]) / 24.
      elif degree == 3:
        coords = numpy.array([[0.2500000000000000,0.2500000000000000,0.2500000000000000],
                              [0.5000000000000000,0.1666666666666667,0.1666666666666667],
                              [0.1666666666666667,0.1666666666666667,0.1666666666666667],
                              [0.1666666666666667,0.1666666666666667,0.5000000000000000],
                              [0.1666666666666667,0.5000000000000000,0.1666666666666667]]).T
        weights = numpy.array([-0.8000000000000000,0.4500000000000000,0.4500000000000000,0.4500000000000000,0.4500000000000000]) / 6.
      elif degree == 4:
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
      elif degree == 5:
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
      elif degree == 6:
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
      elif degree == 7:
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
      else: #degree=8 exact otherwise not exact
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
        if degree > 8:
          warnings.warn('Inexact integration for polynomial of degree %i'%degree)
    else:
      raise NotImplementedError
    return coords.T, weights

  def getischeme_uniform( self, n ):
    if self.ndims == 1:
      return numpy.arange( .5, n )[:,_] / n, numeric.appendaxes( 1./n, n )
    elif self.ndims == 2:
      points = numpy.arange( 1./3, n ) / n
      nn = n**2
      C = numpy.empty( [2,n,n] )
      C[0] = points[:,_]
      C[1] = points[_,:]
      coords = C.reshape( 2, nn )
      flip = coords.sum(0) > 1
      coords[:,flip] = 1 - coords[::-1,flip]
      weights = numeric.appendaxes( .5/nn, nn )
    else:
      raise NotImplementedError
    return coords.T, weights

  def getischeme_bezier( self, np ):
    points = numpy.linspace( 0, 1, np )
    if self.ndims == 1:
      return points[:,_], None
    if self.ndims == 2:
      return numpy.array([ [x,y] for i, y in enumerate(points) for x in points[:np-i] ]), None
    raise NotImplementedError

  def getischeme_vertex( self, n ):
    if n == 0:
      return self.vertices.astype(float), None
    return self.getischeme_bezier( 2**n+1 )

  @cache.property
  def child_transforms( self ):
    if self.ndims == 0:
      return [
        transform.affine( 1, [], 2 ),
      ]
    if self.ndims == 1:
      return [
        transform.affine( 1, [0], 2 ),
        transform.affine( 1, [1], 2 ) ]
    if self.ndims == 2:
      return [
        transform.affine(  1, [0,0], 2 ),
        transform.affine(  1, [0,1], 2 ),
        transform.affine(  1, [1,0], 2 ),
        transform.affine( -1, [1,1], 2 ) ]
    raise NotImplementedError

  @property
  def children( self ):
    return [ (ctrans,self) for ctrans in self.child_transforms ]

  def subvertex( self, ichild, i ):
    n = 2**(i-1)
    if self.ndims == 1:
      assert 0 <= ichild < 2
      return 2*n+1, numpy.arange(n+1) if ichild == 0 else numpy.arange(n,2*n+1)
    if self.ndims == 2:
      assert 0 <= ichild < 4
      return ((2*n+2)*(2*n+1))//2, numpy.concatenate(
             [ (((4*n+3-i)*i)//2) + numpy.arange(n+1-i) for i in range(n+1) ] if ichild == 0
        else [ ((3*(n+1)*n)//2) + numpy.arange(((n+2)*(n+1))//2) ] if ichild == 1
        else [ (((4*n+3-i)*i)//2+n) + numpy.arange(n+1-i) for i in range(n+1) ] if ichild == 2
        else [ (((3*n+3+i)*(n-i))//2) + numpy.arange(n,i-1,-1) for i in range(n+1) ] )
    raise NotImplementedError( 'ndims=%d' % self.ndims )

  @cache.property
  def edges( self ):
    edge = SimplexReference( self.ndims-1 )
    eye = numpy.eye( self.ndims, dtype=int )
    return [ ( transform.affine( (eye[1:]-eye[0]).T, eye[0] ), edge ) ] \
         + [ ( transform.affine( eye[ list(range(i))+list(range(i+1,self.ndims)) ].T, isflipped=(i%2==0) ), edge )
                  for i in range( self.ndims ) ]

  def __str__( self ):
    return 'SimplexReference(%d)' % self.ndims

  __repr__ = __str__

class TensorReference( Reference ):
  'tensor reference'

  def __init__( self, ref1, ref2 ):
    self.ref1 = ref1
    self.ref2 = ref2
    ndims = ref1.ndims + ref2.ndims
    vertices = numpy.empty( ( ref1.nverts, ref2.nverts, ndims ), dtype=int )
    vertices[:,:,:ref1.ndims] = ref1.vertices[:,_]
    vertices[:,:,ref1.ndims:] = ref2.vertices[_,:]
    Reference.__init__( self, vertices.reshape(-1,ndims) )
    self.selfcheck()

  def subvertex( self, ichild, i ):
    ichild1, ichild2 = divmod( ichild, len(self.ref2.children) )
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
    if self == SimplexReference(1)**2:
      points = [[0,0],[1,0],[1,1],[0,1]]
    elif self == SimplexReference(1)**3:
      points = [[0,0,0],[1,0,0],[0,1,0],[1,1,0],[0,0,1],[1,0,1],[0,1,1],[1,1,1]]
    else:
      raise NotImplementedError
    return numpy.array(points,dtype=float), numpy.ones(self.nverts,dtype=float)

  def getischeme_contour( self, n ):
    assert self == SimplexReference(1)**2
    p = numpy.arange( n+1, dtype=float ) / (n+1)
    z = numpy.zeros_like( p )
    return numpy.hstack(( [p,z], [1-z,p], [1-p,1-z], [z,1-p] )).T, None

  def getischeme( self, ischeme ):
    match = re.match( '([a-zA-Z]+)(.*)', ischeme )
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
  def edges( self ):
    return [ ( transform.affine(
                rational.blockdiag([ trans1.linear, rational.eye(self.ref2.ndims) ]),
                rational.concatenate([ trans1.offset, rational.zeros(self.ref2.ndims) ]),
                isflipped=trans1.isflipped ), edge1 * self.ref2 )
                  for trans1, edge1 in self.ref1.edges ] \
         + [ ( transform.affine(
                rational.blockdiag([ rational.eye(self.ref1.ndims), trans2.linear ]),
                rational.concatenate([ rational.zeros(self.ref1.ndims), trans2.offset ]),
                isflipped=trans2.isflipped if self.ref1.ndims%2==0 else not trans2.isflipped ), self.ref1 * edge2 )
                  for trans2, edge2 in self.ref2.edges ]

  @cache.property
  def children( self ):
    children = []
    for trans1, child1 in self.ref1.children:
      for trans2, child2 in self.ref2.children:
        offset = rational.concatenate([ trans1.offset, trans2.offset ])
        if trans1.linear.ndim == 0 and trans1.linear == trans2.linear:
          trans = transform.affine( trans1.linear, offset )
        else:
          assert trans1.ndim == trans2.ndim == 2
          trans = transform.affine( rational.blockdiag([ trans1.linear, trans2.linear ]), offset ) # TODO isflipped?
        children.append(( trans, child1 * child2 ))
    return children

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
    points, weights = (SimplexReference(1)**4).getischeme( ischeme )
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
    quad = SimplexReference(1)**4
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
    assert self.ref1 == self.ref2 == SimplexReference(1)**2
    points, weights = self.get_quad_bem_ischeme( gauss )
    return self.singular_ischeme_quad( points ), weights

class MosaicReference( Reference ):
  'mosaic reference element'

  def __init__( self, ndims, children, edgedict={} ):
    self.children = tuple(children)
    self.edgedict = dict(edgedict)
    vertices = numpy.zeros( (0,ndims), dtype=int )
    Reference.__init__( self, vertices )

  def findedge( self, etrans ):
    edge = self.edgedict.get( etrans )
    if not edge:
      echildren = []
      if any( isinstance( child, MosaicReference ) for ctrans, child in self.children ):
        for ctrans, child in self.children:
          try:
            x = transform.solve( etrans, ctrans << etrans )
          except:
            pass
          else:
            echild = child.findedge( etrans )
            if echild:
              echildren.append(( x, echild ))
      else:
        for ctrans, child in self.children:
          for cetrans, echild in child.edges:
            try:
              x = transform.solve( etrans, ctrans << cetrans )
            except:
              pass
            else:
              echildren.append(( x, echild ))
      edge = MosaicReference( self.ndims-1, echildren ) if echildren else None
      self.edgedict[ etrans ] = edge
    return edge

  def getischeme( self, ischeme ):
    'get integration scheme'
    
    assert not ischeme.startswith('vertex')
    allcoords = []
    allweights = []
    for trans, child in self.children:
      points, weights = child.getischeme( ischeme )
      allcoords.append( trans.apply(points) )
      if weights is not None:
        allweights.append( weights * abs(float(trans.det)) )

    coords = numpy.concatenate( allcoords, axis=0 )
    weights = numpy.concatenate( allweights, axis=0 ) \
      if len(allweights) == len(allcoords) else None

    return coords, weights

  @cache.property
  def simplices( self ):
    return [ ( trans2 << trans1, simplex ) for trans2, child in self.children for trans1, simplex in child.simplices ]

class ForwardReference( Reference ):
  'necessary to count refinement levels of trimmed boundaries'

  def __init__( self, child ):
    self.child = child
    Reference.__init__( self, child.vertices )

  def getischeme( self, ischeme ):
    return self.child.getischeme( ischeme )

  @property
  def simplices( self ):
    return self.child.simplices

  @property
  def edges( self ):
    return [ ( etrans, ForwardReference(edge) ) for etrans, edge in self.child.edges ]

  @property
  def children( self ):
    return ( transform.TransformChain(), self.child ),


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

def gauss( degree ):
  k = numpy.arange( 1, degree // 2 + 1 )
  d = k / numpy.sqrt( 4*k**2-1 )
  x, w = numpy.linalg.eigh( numpy.diagflat(d,-1) ) # eigh operates (by default) on lower triangle
  return (x+1) * .5, w[0]**2


# vim:shiftwidth=2:foldmethod=indent:foldnestmax=2
