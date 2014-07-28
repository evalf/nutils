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

from . import log, util, numpy, core, numeric, function, cache, transform, rational, _
import warnings

PrimaryVertex = str
HalfVertex = lambda vertex1, vertex2, xi=.5: '%s<%.3f>%s' % ( (vertex1,xi,vertex2) if vertex1 < vertex2 else (vertex2,1-xi,vertex1) )
ProductVertex = lambda *vertices: ','.join( vertices )

class Reference( cache.Immutable ):
  'reference element'

  def __init__( self, ndims, nverts ):
    self.ndims = ndims
    self.nverts = nverts

  @property
  def simplices( self ):
    return [ (transform.identity(self.ndims),self) ]

class QuadReference( Reference ):
  'quadrilateral reference element'

  def __init__( self, ndims ):
    Reference.__init__( self, ndims=ndims, nverts=2**ndims )

  @cache.property
  def vertices( self ):
    return numpy.array( list(numpy.ndindex((2,)*self.ndims)), dtype=int )

  @cache.property
  def ribbon2vertices( self ):
    nums = numpy.arange( 2**self.ndims ).reshape( (2,)*self.ndims )
    return numpy.vstack( nums.transpose( [i for i in range(self.ndims) if i != idim] + [idim] ).reshape(-1,2)
      for idim in range( self.ndims ) )

  @staticmethod
  def getgauss( degree ):
    'compute gauss points and weights'

    assert isinstance( degree, int ) and degree >= 0
    k = numpy.arange( 1, degree // 2 + 1 )
    d = k / numpy.sqrt( 4*k**2-1 )
    x, w = numpy.linalg.eigh( numpy.diagflat(d,-1) ) # eigh operates (by default) on lower triangle
    return (x+1) * .5, w[0]**2

  def getischeme( self, where ):
    'get integration scheme'

    if self.ndims == 0:
      return numpy.zeros([1,0]), numpy.array([1.])

    x = w = None
    if where.startswith( 'gauss' ):
      N = eval( where[5:] )
      if isinstance( N, tuple ):
        assert len(N) == self.ndims
      else:
        N = [N]*self.ndims
      x, w = zip( *map( self.getgauss, N ) )
    elif where.startswith( 'uniform' ):
      N = eval( where[7:] )
      if isinstance( N, tuple ):
        assert len(N) == self.ndims
      else:
        N = [N]*self.ndims
      x = [ numpy.arange( .5, n ) / n for n in N ]
      w = [ numeric.appendaxes( 1./n, n ) for n in N ]
    elif where.startswith( 'bezier' ):
      N = int( where[6:] )
      x = [ numpy.linspace( 0, 1, N ) ] * self.ndims
      w = [ numeric.appendaxes( 1./N, N ) ] * self.ndims
    elif where.startswith( 'subdivision' ):
      N = int( where[11:] ) + 1
      x = [ numpy.linspace( 0, 1, N ) ] * self.ndims
      w = None
    elif where.startswith( 'vtk' ):
      if self.ndims == 1:
        coords = numpy.array([[0,1]]).T
      elif self.ndims == 2:
        eps = 0 if not len(where[3:]) else float(where[3:]) # subdivision fix (avoid extraordinary point)
        coords = numpy.array([[eps,eps],[1-eps,eps],[1-eps,1-eps],[eps,1-eps]])
      elif self.ndims == 3:
        coords = numpy.array([ [0,0,0], [1,0,0], [0,1,0], [1,1,0], [0,0,1], [1,0,1], [0,1,1], [1,1,1] ])
      else:
        raise Exception, 'contour not supported for ndims=%d' % self.ndims
    elif where.startswith( 'contour' ):
      N = int( where[7:] )
      p = numpy.linspace( 0, 1, N )
      if self.ndims == 1:
        coords = p[_].T
      elif self.ndims == 2:
        coords = numpy.array([ p[ range(N) + [N-1]*(N-2) + range(N)[::-1] + [0]*(N-1) ],
                               p[ [0]*(N-1) + range(N) + [N-1]*(N-2) + range(0,N)[::-1] ] ]).T
      elif self.ndims == 3:
        assert N == 0
        coords = numpy.array([ [0,0,0], [1,0,0], [0,1,0], [1,1,0], [0,0,1], [1,0,1], [0,1,1], [1,1,1] ])
      else:
        raise Exception, 'contour not supported for ndims=%d' % self.ndims
    else:
      raise Exception, 'invalid element evaluation %r' % where
    if x is not None:
      coords = numpy.empty( map( len, x ) + [ self.ndims ] )
      for i, xi in enumerate( x ):
        coords[...,i] = xi[ (slice(None),) + (_,)*(self.ndims-i-1) ]
      coords = coords.reshape( -1, self.ndims )
    if w is not None:
      weights = reduce( lambda weights, wi: ( weights * wi[:,_] ).ravel(), w )
    else:
      weights = None
    return coords, weights

  @cache.property
  def children( self ):
    'refined transform'

    scale = numpy.diag((.5,)*self.ndims)
    # warning: cache property make cyclic reference
    return [ ( transform.shift(index) >> transform.half(self.ndims), self ) for index in numpy.ndindex(*(2,)*self.ndims) ]

  def get_child_vertices( self, vertices ):
    child_vertices = numpy.empty( [3]*self.ndims, dtype=object )
    child_vertices[ (slice(None,None,2),)*self.ndims ] = numpy.reshape( vertices, [2]*self.ndims )
    for idim in range(self.ndims):
      s1 = (slice(None),)*idim
      s2 = (slice(None,None,2),)*(self.ndims-idim-1)
      child_vertices[s1+(1,)+s2] = util.objmap( HalfVertex, child_vertices[s1+(0,)+s2], child_vertices[s1+(2,)+s2], .5 )
    return [ child_vertices[ tuple( slice(i,i+2) for i in index ) ].ravel() for index in numpy.ndindex(*(2,)*self.ndims) ]

  @cache.property
  def edges( self ):
    'edge transforms'

    transforms = []
    edgeref = QuadReference( self.ndims-1 )
    side = -1
    for idim in range( self.ndims ):
      offset = numpy.zeros( self.ndims, dtype=int )
      matrix = numpy.zeros( ( self.ndims, self.ndims-1 ), dtype=int )
      matrix.flat[ :(self.ndims-1)*idim :self.ndims] = 1
      matrix.flat[self.ndims*(idim+1)-1::self.ndims] = 1
      linear = transform.linear(matrix)
      transforms.append(( linear, side, edgeref ))
      offset[idim] = 1
      side = -side
      transforms.append(( linear >> transform.shift(offset), side, edgeref ))
    return transforms

  def get_edge_vertices( self, vertices ):
    vertices = numpy.reshape( vertices, (2,)*self.ndims )
    edge_vertices = []
    for idim in range(self.ndims):
      for iside in (0,1):
        s = (slice(None),) * idim + (iside,) + (slice(None),) * (self.ndims-idim-1)
        edge_vertices.append( vertices[s].ravel() )
    return edge_vertices

class TriangularReference( Reference ):
  '''triangular reference element

  Conventions:
  * reference elem: unit simplex {(x,y) | x>0, y>0, x+y<1}
  * vertex numbering: {(1,0):0, (0,1):1, (0,0):2}
  * edge numbering: {bottom:0, slanted:1, left:2}
  * edge local coords run counter-clockwise.'''

  def __init__( self ):
    Reference.__init__( self, ndims=2, nverts=3 )

  def getischeme( self, where ):
    '''get integration scheme
    gaussian quadrature: http://www.cs.rpi.edu/~flaherje/pdf/fea6.pdf
    '''

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
    elif where[:7] == 'uniform':
      N = int( where[7:] )
      points = ( numpy.arange( N ) + 1./3 ) / N
      NN = N**2
      C = numpy.empty( [2,N,N] )
      C[0] = points[:,_]
      C[1] = points[_,:]
      coords = C.reshape( 2, NN )
      flip = coords[0] + coords[1] > 1
      coords[:,flip] = 1 - coords[::-1,flip]
      weights = numeric.appendaxes( .5/NN, NN )
    elif where[:6] == 'bezier':
      N = int( where[6:] )
      points = numpy.linspace( 0, 1, N )
      coords = numpy.array([ [x,y] for i, y in enumerate(points) for x in points[:N-i] ]).T
      weights = None
    else:
      raise Exception, 'invalid element evaluation: %r' % where
    return coords.T, weights

  @cache.property
  def children( self ):
    transforms = [
      transform.half(2),
      transform.shift([0,1]) >> transform.half(2),
      transform.shift([1,0]) >> transform.half(2),
      transform.shift([-1,-1]) >> transform.linear([[-1,0],[0,-1]],2)
    ]
    # warning: cache property make cyclic reference
    return [ (trans,self) for trans in transforms ]

  def get_child_vertices( self, vertices ):
    v1, v2, v3 = vertices
    h1, h2, h3 = HalfVertex(v1,v2), HalfVertex(v2,v3), HalfVertex(v3,v1)
    return [ [v1,h1,h3], [h1,v2,h2], [h3,h2,v3], [h2,h3,h1] ]

  @cache.property
  def edges( self ):
    edge = QuadReference(1)
    return (
      ( transform.linear([[ 1],[ 0]]), 1, edge ),
      ( transform.linear([[-1],[ 1]]) >> transform.shift([1,0]), 1, edge ),
      ( transform.linear([[ 0],[-1]]) >> transform.shift([0,1]), 1, edge ) )

  def get_edge_vertices( self, vertices ):
    return [ vertices[::-2], vertices[:2], vertices[1:] ]

class TetrahedronReference( Reference ):
  'tetrahedron reference element'

  def __init__( self ):
    Reference.__init__( self, ndims=3, nverts=4 )

  def getischeme( self, where ):
    '''get integration scheme
       http://people.sc.fsu.edu/~jburkardt/datasets/quadrature_rules_tet/quadrature_rules_tet.html'''

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
    return coords.T, weights

  @property
  def edges( self ):
    assert len(vertices) == 4
    edge = TriangularReference()
    return [
      ( transfor,linear([[ 0, 1],[1,0],[0,0]]), 1, edge ),
      ( transfor,linear([[ 1, 0],[0,0],[0,1]]), 1, edge ),
      ( transfor,linear([[ 0, 0],[0,1],[1,0]]), 1, edge ),
      ( transfor,linear([[-1,-1],[1,0],[0,1]]) >> transform.shift([1,0,0]), 1, edge ) ]

  def get_edgevertices( self, vertices ):
    v1, v2, v3, v4 = vertices
    return [ [v1,v3,v2], [v1,v2,v4], [v1,v4,v3], [v2,v3,v4] ] # TODO check!

class MosaicReference( Reference ):
  'mosaic reference element'

  def __init__( self, ndims, children ):
    self.children = children
    Reference.__init__( self, ndims=ndims, nverts=0 )

  def getischeme( self, ischeme ):
    'get integration scheme'

    allcoords = []
    allweights = []
    for trans, child in self.children:
      points, weights = child.getischeme( ischeme )
      allcoords.append( trans.apply(points) )
      if weights is not None:
        allweights.append( weights * float(trans.det) )

    coords = numpy.concatenate( allcoords, axis=0 )
    weights = numpy.concatenate( allweights, axis=0 ) \
      if len(allweights) == len(allcoords) else None

    return coords, weights

  @cache.property
  def simplices( self ):
    return [ ( trans1 >> trans2, simplex ) for trans2, child in self.children for trans1, simplex in child.simplices ]

class ProductReference( Reference ):
  'product reference element'

  def __init__( self, ref1, ref2, neighborhood, transf ):
    '''Neighborhood of elem1 and elem2 and transformations to get mutual
    overlap in right location. Returns 3-element tuple:
    * neighborhood, as given by Element.neighbor(),
    * transf1, required rotation of elem1 map: {0:0, 1:pi/2, 2:pi, 3:3*pi/2},
    * transf2, required rotation of elem2 map (is indep of transf1 in UnstructuredTopology.'''

    self.ref1 = ref1
    self.ref2 = ref2
    self.neighborhood = neighborhood
    self.transf = transf
    Reference.__init__( self, ndims=ref1.ndims+ref2.ndims, nverts=ref1.nverts*ref2.nverts )

  @staticmethod
  def singular_ischeme_quad( points, transf ):
    transfpoints = numpy.empty( points.shape )
    def transform( points, transf ):
      x, y = points[:,0], points[:,1]
      tx = x if transf in (0,1,6,7) else 1-x
      ty = y if transf in (0,3,4,7) else 1-y
      return function.stack( (ty, tx) if transf%2 else (tx, ty), axis=1 )
    transfpoints[:,:2] = transform( points[:,:2], transf[0] )
    transfpoints[:,2:] = transform( points[:,2:], transf[1] )
    return transfpoints

  @staticmethod
  def get_tri_bem_ischeme( self, ischeme, neighborhood ):
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
      points = numpy.array(
        [[1-xi,   1-pts2, 1-xi,   1-pts5, 1-pts2, 1-xi  ],
         [pts1, pts3, pts4, pts0, pts6, pts0],
         [1-pts2, 1-xi,   1-pts5, 1-xi,   1-xi,   1-pts2],
         [pts3, pts1, pts0, pts4, pts0, pts6]]).reshape( 4, -1 ).T
      points = points * [-1,1,-1,1] + [1,0,1,0] # flipping in x -GJ
      weights = numpy.concatenate( 6*[xi**3*eta1**2*eta2*weights] )
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
      points = numpy.array(
        [[1-xi, 1-xi, 1-E,  1-G,  1-G ],
         [C,  G,  F,  H,  I ],
         [1-E,  1-G,  1-xi, 1-xi, 1-xi],
         [F,  H,  D,  A,  B ]] ).reshape( 4, -1 ).T
      temp = xi*A
      weights = numpy.concatenate( [A*temp*weights] + 4*[B*temp*weights] )
    elif neighborhood == 2:
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
      assert neighborhood == -1, 'invalid neighborhood %r' % neighborhood
      points = numpy.array([ eta1*eta2, 1-eta2, eta3*xi, 1-xi ]).T
      weights = eta2*xi*weights
    return points, weights

  @staticmethod
  def get_quad_bem_ischeme( ischeme, neighborhood ):
    'Some cached quantities for the singularity quadrature scheme.'
    quad = QuadReference(4)
    points, weights = quad.getischeme( ischeme )
    eta1, eta2, eta3, xi = points.T
    if neighborhood == 0:
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
    elif neighborhood == 1:
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
    elif neighborhood == 2:
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
      assert neighborhood == -1, 'invalid neighborhood %r' % neighborhood
    return points, weights

  @staticmethod
  def concat( ischeme1, ischeme2 ):
    coords1, weights1 = ischeme1
    coords2, weights2 = ischeme2
    if weights1 is not None:
      assert weights2 is not None
      weights = ( weights1[:,_] * weights2[_,:] ).ravel()
    else:
      assert weights2 is None
      weights = None
    npoints1,ndims1 = coords1.shape  
    npoints2,ndims2 = coords2.shape 
    coords = numpy.empty( [ coords1.shape[0], coords2.shape[0], ndims1+ndims2 ] )
    coords[:,:,:ndims1] = coords1[:,_,:]
    coords[:,:,ndims1:] = coords2[_,:,:]
    coords = coords.reshape(-1,ndims1+ndims2)
    return coords, weights
    
  def getischeme( self, where ):
    'get integration scheme'
    
    if where.startswith( 'singular' ):
      gauss = 'gauss%d'% (int(where[8:])*2-2)
      if self.ref1 == self.ref2 == QuadReference(2):
        points, weights = self.get_quad_bem_ischeme( gauss, self.neighborhood )
        mod_points = self.singular_ischeme_quad( points, self.transf )
        xw = mod_points, weights
      elif self.ref1 == self.ref2 == TriangularReference():
        xw = self.get_tri_bem_ischeme( gauss, self.neighborhood )
      else:
        raise Exception, 'invalid element type %r' % type(self.ref1)
    else:
      where1, where2 = where.split( '*' ) if '*' in where else ( where, where )
      xw = self.concat( self.ref1.getischeme(where1), self.ref2.getischeme(where2) )
    return xw

class Element( object ):
  '''Element base class.

  Represents the topological shape.'''

  __slots__ = 'reference', 'vertices', 'parent', 'parents', 'sign'

  def __init__( self, reference, vertices, parent=None, parents=None, sign=1 ):
    'constructor'

    assert isinstance( reference, Reference )
    assert len(vertices) == reference.nverts
    #assert all( isinstance(vertex,Vertex) for vertex in vertices )
    self.reference = reference
    self.vertices = tuple(vertices)
    if parent:
      assert not parents
      self.parent = parent
      self.parents = parent, parent
    elif parents:
      self.parent = None
      self.parents = parents
    else:
      self.parent = None
      self.parents = None
    self.sign = sign

  @property
  def ndims( self ):
    return self.reference.ndims

  def __mul__( self, other ):
    'multiply elements'

    return ProductElement( self, other )

  @property
  def simplices( self ):
    return [ Element( reference, vertices=[None]*reference.nverts, parent=(self,trans), sign=self.sign ) for trans, reference in self.reference.simplices ]

  @property
  def children( self ):
    return tuple( Element( reference=reference, vertices=vertices, parent=(self,transform), sign=self.sign )
      for (transform,reference), vertices in zip(self.reference.children,self.reference.get_child_vertices(self.vertices)) )

  @property
  def edges( self ):
    return tuple( Element( reference=reference, vertices=vertices, parent=(self,transform), sign=self.sign*sign )
      for (transform,sign,reference), vertices in zip(self.reference.edges,self.reference.get_edge_vertices(self.vertices)) )

  def edge( self, iedge ):
    return self.edges[iedge]

  def __str__( self ):
    'string representation'

    return '%s(%s)' % ( self.__class__.__name__, self.vertices )

  def __hash__( self ):
    'hash'

    return hash(str(self))

  def __eq__( self, other ):
    'hash'

    return self is other or ( isinstance( other, Element )
      and self.reference == other.reference
      and self.vertices == other.vertices
      and self.parent == other.parent
      and self.parents == other.parents )

  def trim( self, levelset, maxrefine, numer ):
    'trim element along levelset'

    assert rational.isrational( numer )
    children = []
    if maxrefine <= 0:
      repeat = True
      levels = levelset.eval( self, 'bezier2' )
      while repeat: # set almost-zero points to zero if cutoff within eps
        repeat = False
        assert levels.shape == (self.reference.nverts,)
        if numpy.greater_equal( levels, 0 ).all():
          return self
        if numpy.less_equal( levels, 0 ).all():
          return None
        isects = []
        for ribbon in self.reference.ribbon2vertices:
          a, b = levels[ribbon]
          if a * b < 0: # strict sign change
            x = int( numer * a / float(a-b) + .5 ) # round to [0,1,..,numer]
            if 0 < x < numer:
              isects.append(( x, ribbon ))
            else: # near intersection of vertex
              v = ribbon[ (0,numer).index(x) ]
              log.debug( 'rounding vertex #%d from %f to 0' % ( v, levels[v] ) )
              levels[v] = 0
              repeat = True
      coords = self.reference.vertices
      if isects:
        coords = numpy.vstack([
          self.reference.vertices * int(numer),
          [ numpy.dot( (int(numer)-x,x), self.reference.vertices[ribbon] ) for x, ribbon in isects ]
        ])
      assert coords.dtype == int
      if self.ndims == 2:
        simplex = TriangularReference()
      elif self.ndims == 3:
        simplex = TetrahedronReference()
      else:
        raise NotImplementedError
      for tri in util.delaunay( coords ):
        ispos = isneg = False
        for ivert in tri:
          if ivert < self.reference.nverts:
            sign = levels[ivert]
            ispos = ispos or sign > 0
            isneg = isneg or sign < 0
        assert ispos is not isneg, 'domains do not separate in two convex parts'
        if isneg:
          continue
        offset = coords[tri[0]]
        matrix = ( coords[tri[1:]] - offset ).T
        if numpy.linalg.det( matrix.astype(float) ) < 0:
          tri[-2:] = tri[-1], tri[-2]
          matrix = ( coords[tri[1:]] - offset ).T
        trans = transform.linear(matrix,numer) >> transform.shift(offset,numer)
        children.append(( trans, simplex ))
    else:
      complete = True
      for child in self.children:
        _self, trans = child.parent
        assert _self is self
        trimmed = child.trim( levelset, maxrefine-1, numer )
        complete = complete and trimmed == child
        if trimmed != None:
          children.append( (trans,trimmed.reference) )
      if complete:
        return self
    if not children:
      return None
    reference = MosaicReference( self.ndims, tuple(children) )
    return Element( reference=reference, vertices=[], parent=(self,transform.identity(self.ndims)), sign=self.sign )

def ProductElement( elem1, elem2 ):
  eye = numpy.eye( elem1.ndims + elem2.ndims, dtype=int )
  iface1 = elem1, transform.linear( eye[:elem1.ndims] )
  iface2 = elem2, transform.linear( eye[elem1.ndims:] )
  #vertices = [] # TODO [ ProductVertex(vertex1,vertex2) for vertex1 in elem1.vertices for vertex2 in elem2.vertices ]
  vertices = [ ProductVertex(vertex1,vertex2) for vertex1 in elem1.vertices for vertex2 in elem2.vertices ]

  transf = 0, 0
  neighborhood = -1
  if elem1.reference == elem2.reference == QuadReference(2):
    common_vertices = list( set(elem1.vertices) & set(elem2.vertices) )
    vertices1 = [elem1.vertices.index( ni ) for ni in common_vertices]
    vertices2 = [elem2.vertices.index( ni ) for ni in common_vertices]
    neighborhood = neighbor( elem1, elem2 )
    if neighborhood == 0:
      # test for strange topological features
      assert elem1 == elem2, 'Topological feature not supported: try refining here, possibly periodicity causes elems to touch on both sides.'
    elif neighborhood != -1:
      # define local map rotations
      if neighborhood==1:
        vertex = [0,2], [2,3], [3,1], [1,0], [2,0], [3,2], [1,3], [0,1]
      elif neighborhood==2:
        vertex = [0], [2], [3], [1]
      else:
        raise ValueError( 'Unknown neighbor type %i' % neighborhood )
      transf = vertex.index( vertices1 ), vertex.index( vertices2 )
  reference = ProductReference( elem1.reference, elem2.reference, neighborhood, transf )

  return Element( reference=reference, vertices=vertices, parents=(iface1,iface2), sign=1 )
  
def QuadElement( ndims, vertices, parent=None, parents=None ):
  reference = QuadReference( ndims )
  return Element( reference, vertices, parent=parent, parents=parents, sign=1 )

def TriangularElement( vertices, parent=None ):
  reference = TriangularReference()
  return Element( reference=reference, vertices=vertices, parent=parent, sign=1 )

def TetrahedronElement( vertices, parent=None ):
  reference = TetrahedronReference()
  return Element( reference=reference, vertices=vertices, parent=parent, sign=1 )

class StdElem( cache.Immutable ):
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
      data = numpy.array( [ x, y, 1-x-y ] ).T
    elif grad == 1:
      data = numpy.array( [[1,0],[0,1],[-1,-1]], dtype=float )
    else:
      data = numpy.array( 0 ).reshape( (1,) * (grad+ndim) )
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

  def eval( self, points, grad=0 ):
    'eval'

    npoints, ndim = points.shape
    if grad == 0:
      x, y = points.T
      data = numpy.array( [ x, y, 1-x-y, 27*x*y*(1-x-y) ] ).T
    elif grad == 1:
      x, y = points.T
      const_block = numpy.array( [1,0,0,1,-1,-1]*npoints, dtype=float ).reshape( npoints,3,2 )
      grad1_bubble = 27*numpy.array( [y*(1-2*x-y),x*(1-x-2*y)] ).T.reshape( npoints,1,2 )
      data = numpy.concatenate( [const_block, grad1_bubble], axis=1 )
    elif grad == 2:
      x, y = points.T
      zero_block = numpy.zeros( (npoints,3,2,2) )
      grad2_bubble = 27*numpy.array( [-2*y,1-2*x-2*y, 1-2*x-2*y,-2*x] ).T.reshape( npoints,1,2,2 )
      data = numpy.concatenate( [zero_block, grad2_bubble], axis=1 )
    elif grad == 3:
      zero_block = numpy.zeros( (3,2,2,2) )
      grad3_bubble = 27*numpy.array( [0,-2,-2,-2,-2,-2,-2,0], dtype=float ).reshape( 1,2,2,2 )
      data = numpy.concatenate( [zero_block, grad3_bubble], axis=0 )
    else:
      assert ndim==2, 'Triangle takes 2D coordinates' # otherwise tested by unpacking points.T
      data = numpy.array( 0 ).reshape( (1,) * (grad+2) )
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

def neighbor( elem1, elem2 ):
  ncommon = len( set(elem1.vertices) & set(elem2.vertices) )
  if elem1.reference == elem2.reference == QuadReference(1):
    neighbormap = { 0:-1, 1:1, 2:0 }
  elif elem1.reference == elem2.reference == QuadReference(2):
    neighbormap = { 0:-1, 1:2, 2:1, 4:0 }
  elif elem1.reference == elem2.reference == QuadReference(3):
    neighbormap = { 0:-1, 1:3, 2:2, 4:1, 8:0 }
  else:
    raise NotImplementedError, 'neighbor for %s and %s' % ( elem1, elem2 )
  return neighbormap[ncommon]

# vim:shiftwidth=2:foldmethod=indent:foldnestmax=2
