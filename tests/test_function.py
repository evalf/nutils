from nutils import *


class FuncTest( object ):

  haslocalgradient = True

  def __init__( self ):
    domainelem = element.Element( ndims=2, vertices=[] )
    r, theta = function.ElemFunc( domainelem ) # corners at (0,0), (0,1), (1,1), (1,0)
    geom = r * function.stack([ function.cos(theta), function.sin(theta) ])
    trans = element.AffineTransformation( offset=[1,0], transform=[[2,1],[-1,3]] )
    avertices = [element.PrimaryVertex('A(%d)'%i) for i in range(4)]
    interelem = element.QuadElement( ndims=2, vertices=avertices, parent=(domainelem,trans) ) # corners at (1,0), (3,-1), (4,2), (2,3)
    trans = element.AffineTransformation( offset=[1,0], transform=[[0,1],[-1,0]] )
    bvertices = [element.PrimaryVertex('B(%d)'%i) for i in range(4)]
    elem = element.QuadElement( ndims=2, vertices=bvertices, parent=(interelem,trans) ) # corners at (3,-1), (2,-4), (4,-5), (5,-2)

    #trans1 = element.AffineTransformation( offset=[1,0], transform=[[1,0]] )
    #trans2 = element.AffineTransformation( offset=[0,0], transform=[[1,0]] )
    #iface = element.QuadElement( ndims=1, id='test.quad.quad.iface', interface=((elem,trans1),(elem,trans2)) )
    cvertices = [element.PrimaryVertex('C(%d)'%i) for i in range(2)]
    iface = element.QuadElement( ndims=1, vertices=cvertices, interface=(elem.edge(1).context,elem.edge(0).context) )
    ifpoints, ifweights = iface.eval('uniform2')

    fmap = { elem: element.PolyLine( element.PolyLine.bernstein_poly(1) )
                 * element.PolyLine( element.PolyLine.bernstein_poly(2) ) }
    nmap = { elem: numpy.arange(6) }
    funcsp = function.function( fmap, nmap, ndofs=6, ndims=2 )
    numpy.random.seed(0)
    args = [ ( numpy.random.uniform( size=shape+(funcsp.shape[0],) ) * funcsp ).sum() for shape in self.shapes ]
    points, weights = elem.eval('uniform2')

    self.args = function.Tuple( args )
    self.elem = elem
    self.points = points
    self.geom = geom
    self.iface = iface
    self.ifpoints = ifpoints

  def find( self, target, xi0 ):
    ndim, = self.geom.shape
    J = function.localgradient( self.geom, ndim )
    Jinv = function.inverse( J )
    countdown = 5
    iiter = 0
    xi = xi0
    while countdown:
      err = target - self.geom(self.elem,xi)
      if numpy.all( numpy.abs(err) < 1e-12 ):
        countdown -= 1
      dxi_root = ( Jinv(self.elem,xi) * err[...,_,:] ).sum(-1)
      xi = xi + numpy.dot( dxi_root, self.elem.inv_root_transform.T )
      iiter += 1
      assert iiter < 100, 'failed to converge in 100 iterations'
    return xi

  def test_eval( self ):
    numpy.testing.assert_array_almost_equal(
      self.n_op( *self.args(self.elem,self.points) ),
        self.op( *self.args )(self.elem,self.points), decimal=15 )

  def test_getitem( self ):
    shape = self.op( *self.args ).shape
    for idim in range( len(shape) ):
      s = (Ellipsis,) + (slice(None),)*idim + (shape[idim]//2,) + (slice(None),)*(len(shape)-idim-1)
      print s
      numpy.testing.assert_array_almost_equal(
        self.n_op( *self.args(self.elem,self.points) )[s],
          self.op( *self.args )[s](self.elem,self.points), decimal=15 )

  def test_getslice( self ):
    shape = self.op( *self.args ).shape
    for idim in range( len(shape) ):
      if shape[idim] == 1:
        continue
      s = (Ellipsis,) + (slice(None),)*idim + (slice(0,shape[idim]-1),) + (slice(None),)*(len(shape)-idim-1)
      print s
      numpy.testing.assert_array_almost_equal(
        self.n_op( *self.args(self.elem,self.points) )[s],
          self.op( *self.args )[s](self.elem,self.points), decimal=15 )

  def test_localgradient( self ):
    if not self.haslocalgradient:
      return
    eps = 1e-6
    D = numpy.array([-.5*eps,.5*eps])[:,_,_] * self.elem.inv_root_transform.T[_,:,:]
    fdpoints = self.points[_,_,:,:] + D[:,:,_,:]
    F = self.n_op( *self.args(self.elem,fdpoints) )
    fdgrad = ((F[1]-F[0])/eps).transpose( numpy.roll(numpy.arange(F.ndim-1),-1) )
    G = function.localgradient( self.op( *self.args ), ndims=self.elem.ndims )
    numpy.testing.assert_array_almost_equal( fdgrad, G(self.elem,self.points), decimal=5 )

  def test_gradient( self ):
    if not self.haslocalgradient:
      return
    eps = 1e-6
    D = numpy.array([-.5*eps,.5*eps])[:,_,_] * numpy.eye(2)
    fdpoints = self.find( self.geom(self.elem,self.points)[_,_,:,:] + D[:,:,_,:], self.points[_,_,:,:] )
    F = self.n_op( *self.args(self.elem,fdpoints) )
    fdgrad = ((F[1]-F[0])/eps).transpose( numpy.roll(numpy.arange(F.ndim-1),-1) )
    G = self.op( *self.args ).grad(self.geom)
    numpy.testing.assert_array_almost_equal( fdgrad, G(self.elem,self.points), decimal=5 )

  def test_doublegradient( self ):
    if not self.haslocalgradient:
      return
    eps = 1e-5
    D = numpy.array([-.5*eps,.5*eps])[:,_,_] * numpy.eye(2)
    DD = D[:,_,:,_,:] + D[_,:,_,:,:]
    fdpoints = self.find( self.geom(self.elem,self.points)[_,_,_,_,:,:] + DD[:,:,:,:,_,:], self.points[_,_,_,_,:,:] )
    F = self.n_op( *self.args(self.elem,fdpoints) )
    fddgrad = (((F[1,1]-F[1,0])-(F[0,1]-F[0,0]))/(eps**2)).transpose( numpy.roll(numpy.arange(F.ndim-2),-2) )
    G = self.op( *self.args ).grad(self.geom).grad(self.geom)
    numpy.testing.assert_array_almost_equal( fddgrad, G(self.elem,self.points), decimal=2 )

  def test_opposite( self ):
    opposite_args = function.Tuple([ function.opposite(arg) for arg in self.args ])
    numpy.testing.assert_array_almost_equal(
      self.n_op( *opposite_args(self.iface,self.ifpoints) ),
        function.opposite( self.op( *self.args ) )(self.iface,self.ifpoints), decimal=15 )

# UNARY POINTWISE OPERATIONS

class TestSin( FuncTest ):
  shapes = [(3,)]
  op = staticmethod( function.sin )
  n_op = staticmethod( numpy.sin )

class TestCos( FuncTest ):
  shapes = [(3,)]
  op = staticmethod(function.cos)
  n_op = staticmethod(numpy.cos)

class TestTan( FuncTest ):
  shapes = [(3,)]
  op = staticmethod(function.tan)
  n_op = staticmethod(numpy.tan)

class TestSqrt( FuncTest ):
  shapes = [(3,)]
  op = staticmethod(function.sqrt)
  n_op = staticmethod(numpy.sqrt)

class TestLog( FuncTest ):
  shapes = [(3,)]
  op = staticmethod(function.ln)
  n_op = staticmethod(numpy.log)

class TestLog2( FuncTest ):
  shapes = [(3,)]
  op = staticmethod(function.log2)
  n_op = staticmethod(numpy.log2)

class TestLog10( FuncTest ):
  shapes = [(3,)]
  op = staticmethod(function.log10)
  n_op = staticmethod(numpy.log10)

class TestExp( FuncTest ):
  shapes = [(3,)]
  op = staticmethod(function.exp)
  n_op = staticmethod(numpy.exp)

class TestArctanh( FuncTest ):
  shapes = [(3,)]
  op = staticmethod(function.arctanh)
  n_op = staticmethod(numpy.arctanh)

class TestTanh( FuncTest ):
  shapes = [(3,)]
  op = staticmethod(function.tanh)
  n_op = staticmethod(numpy.tanh)

class TestCosh( FuncTest ):
  shapes = [(3,)]
  op = staticmethod(function.cosh)
  n_op = staticmethod(numpy.cosh)

class TestSinh( FuncTest ):
  shapes = [(3,)]
  op = staticmethod(function.sinh)
  n_op = staticmethod(numpy.sinh)

class TestAbs( FuncTest ):
  shapes = [(3,)]
  op = staticmethod(function.abs)
  n_op = staticmethod(numpy.abs)

class TestSign( FuncTest ):
  shapes = [(3,)]
  op = staticmethod(function.sign)
  n_op = staticmethod(numpy.sign)

class TestPower( FuncTest ):
  shapes = [(3,)]
  op = staticmethod(lambda a: function.power(a,1.5))
  n_op = staticmethod(lambda a: numpy.power(a,1.5))

class TestNegative( FuncTest ):
  shapes = [(3,)]
  op = staticmethod(function.negative)
  n_op = staticmethod(numpy.negative)

class TestReciprocal( FuncTest ):
  shapes = [(3,)]
  op = staticmethod(function.reciprocal)
  n_op = staticmethod(numpy.reciprocal)

class TestArcsin( FuncTest ):
  shapes = [(3,)]
  op = staticmethod( function.arcsin )
  n_op = staticmethod( numpy.arcsin )


# UNARY ARRAY OPERATIONS

class TestProduct( FuncTest ):
  shapes = [(2,3,2)]
  op = staticmethod(lambda a: function.product(a,1))
  n_op = staticmethod(lambda a: numpy.product(a,-2))

class TestNorm2( FuncTest ):
  shapes = [(2,3,2)]
  op = staticmethod(lambda a: function.norm2(a,1))
  n_op = staticmethod(lambda a: (a**2).sum(-2)**.5)

class TestSum( FuncTest ):
  shapes = [(2,3,2)]
  op = staticmethod(lambda a: function.sum(a,1))
  n_op = staticmethod(lambda a: a.sum(-2))

class TestAlign( FuncTest ):
  shapes = [(2,3)]
  op = staticmethod(lambda a: function.align(a,[0,2],3))
  n_op = staticmethod(lambda a: a[...,:,_,:])

class TestGet( FuncTest ):
  shapes = [(2,3,2)]
  op = staticmethod(lambda a: function.get(a,1,1))
  n_op = staticmethod(lambda a: a[...,1,:])

class TestTakediag1x1( FuncTest ):
  shapes = [(1,2,1)]
  op = staticmethod(lambda a: function.takediag(a,0,2))
  n_op = staticmethod(lambda a: numeric.takediag(a.swapaxes(-3,-2)))

class TestTakediag2x2( FuncTest ):
  shapes = [(2,3,2)]
  op = staticmethod(lambda a: function.takediag(a,0,2))
  n_op = staticmethod(lambda a: numeric.takediag(a.swapaxes(-3,-2)))

class TestTakediag3x3( FuncTest ):
  shapes = [(3,2,3)]
  op = staticmethod(lambda a: function.takediag(a,0,2))
  n_op = staticmethod(lambda a: numeric.takediag(a.swapaxes(-3,-2)))

class TestDet1x1( FuncTest ):
  shapes = [(1,3,1)]
  op = staticmethod(lambda a: function.determinant(a,(0,2)))
  n_op = staticmethod(lambda a: numeric.determinant(a.swapaxes(-3,-2)))

class TestDet2x2( FuncTest ):
  shapes = [(2,3,2)]
  op = staticmethod(lambda a: function.determinant(a,(0,2)))
  n_op = staticmethod(lambda a: numeric.determinant(a.swapaxes(-3,-2)))

class TestDet3x3( FuncTest ):
  shapes = [(3,2,3)]
  op = staticmethod(lambda a: function.determinant(a,(0,2)))
  n_op = staticmethod(lambda a: numeric.determinant(a.swapaxes(-3,-2)))

class TestInv1x1( FuncTest ):
  shapes = [(1,3,1)]
  op = staticmethod(lambda a: function.inverse(a,(0,2)))
  n_op = staticmethod(lambda a: numeric.inverse(a.swapaxes(-3,-2)).swapaxes(-3,-2))

class TestInv2x2( FuncTest ):
  shapes = [(2,3,2)]
  op = staticmethod(lambda a: function.inverse(a,(0,2)))
  n_op = staticmethod(lambda a: numeric.inverse(a.swapaxes(-3,-2)).swapaxes(-3,-2))

class TestInv3x3( FuncTest ):
  shapes = [(3,2,3)]
  op = staticmethod(lambda a: function.inverse(a,(0,2)))
  n_op = staticmethod(lambda a: numeric.inverse(a.swapaxes(-3,-2)).swapaxes(-3,-2))

class TestRepeat( FuncTest ):
  shapes = [(2,1,2)]
  op = staticmethod(lambda a: function.repeat(a,3,1))
  n_op = staticmethod(lambda a: numpy.repeat(a,3,-2))

class TestDiagonalize( FuncTest ):
  shapes = [(2,1,2)]
  op = staticmethod(function.diagonalize)
  n_op = staticmethod(numeric.diagonalize)


# BINARY OPERATIONS

class TestMul( FuncTest ):
  shapes = [(3,1),(1,3)]
  op = staticmethod(lambda a,b: a * b)
  n_op = staticmethod(numpy.multiply)

class TestDiv( FuncTest ):
  shapes = [(3,1),(1,3)]
  op = staticmethod(lambda a,b: a / b)
  n_op = staticmethod(numpy.divide)

class TestAdd( FuncTest ):
  shapes = [(3,1),(1,3)]
  op = staticmethod(lambda a,b: a + b)
  n_op = staticmethod(numpy.add)

class TestSub( FuncTest ):
  shapes = [(3,1),(1,3)]
  op = staticmethod(lambda a,b: a - b)
  n_op = staticmethod(numpy.subtract)

class TestDot( FuncTest ):
  shapes = [(2,3,1),(1,3,2)]
  op = staticmethod(lambda a,b: (a*b).sum(-2))
  n_op = staticmethod(lambda a,b: (a*b).sum(-2))

class TestCross( FuncTest ):
  shapes = [(2,3,1),(1,3,2)]
  op = staticmethod(lambda a,b: function.cross(a,b,-2))
  n_op = staticmethod(lambda a,b: numpy.cross(a,b,axis=-2))

class TestMin( FuncTest ):
  shapes = [(3,1),(1,3)]
  op = staticmethod(lambda a,b: function.min(a,b))
  n_op = staticmethod(numpy.minimum)

class TestMax( FuncTest ):
  shapes = [(3,1),(1,3)]
  op = staticmethod(lambda a,b: function.max(a,b))
  n_op = staticmethod(numpy.maximum)

class TestGreater( FuncTest ):
  shapes = [(3,1),(1,3)]
  op = staticmethod(lambda a,b: function.greater(a,b))
  n_op = staticmethod(numpy.greater)

class TestLess( FuncTest ):
  shapes = [(3,1),(1,3)]
  op = staticmethod(lambda a,b: function.less(a,b))
  n_op = staticmethod(numpy.less)

class TestArctan2( FuncTest ):
  shapes = [(3,1),(1,3)]
  op = staticmethod(function.arctan2)
  n_op = staticmethod(numpy.arctan2)

class TestStack( FuncTest ):
  shapes = [(3,),(3,)]
  op = staticmethod(lambda a,b: function.stack([a,b]))
  n_op = staticmethod(lambda a,b: numpy.concatenate( [a[...,_,:],b[...,_,:]], axis=-2))

# EVAL ONLY

class TestEig( FuncTest ):
  shapes = [(3,3)]
  haslocalgradient = False
  op = staticmethod(lambda a: function.eig(a,symmetric=False)[1] )
  n_op = staticmethod(lambda a: numpy.array([ numpy.linalg.eig(ai)[1] for ai in a ]) )
  

# vim:shiftwidth=2:foldmethod=indent:foldnestmax=2
