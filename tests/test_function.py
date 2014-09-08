from nutils import *


class FuncTest( object ):

  def __init__( self, op, n_op, *shapes ):

    roottrans = transform.affine( [[0,1],[-1,0]], [1,0] ) >> transform.affine( [[2,1],[-1,3]], [1,0] )
    elem = element.Element( element.SimplexReference(1)**2, roottrans >> transform.roottrans( 'test', (0,0) ) )
    iface = element.Element( elem.edge(0).reference, elem.edge(0).transform, elem.edge(1).transform )
    ifpoints, ifweights = iface.reference.getischeme('uniform2')

    r, theta = function.ElemFunc( 2 ) # corners at (0,0), (0,1), (1,1), (1,0)
    geom = r * function.stack([ function.cos(theta), function.sin(theta) ])

    funcsp = topology.StructuredTopology([[ elem ]]).splinefunc( (1,2) )

    numpy.random.seed(0)
    args = [ ( numpy.random.uniform( size=shape+(funcsp.shape[0],) ) * funcsp ).sum() for shape in shapes ]
    points, weights = elem.reference.getischeme('uniform2')

    self.op = op
    self.n_op = n_op
    self.args = args
    self.argsfun = function.Tuple( args ).compiled()
    self.elem = elem
    self.points = points
    self.geom = geom
    self.geomfun = geom.compiled()
    self.iface = iface
    self.ifpoints = ifpoints
    self.invroottransmatrix = roottrans.invlinear.astype( float )

  def find( self, target, xi0 ):
    ndim, = self.geom.shape
    J = function.localgradient( self.geom, ndim )
    Jinv = function.inverse( J ).compiled()
    countdown = 5
    iiter = 0
    xi = xi0
    while countdown:
      err = target - self.geomfun.eval(self.elem,xi)
      if numpy.all( numpy.abs(err) < 1e-12 ):
        countdown -= 1
      dxi_root = ( Jinv.eval(self.elem,xi) * err[...,_,:] ).sum(-1)
      #xi = xi + numpy.dot( dxi_root, self.elem.inv_root_transform.T )
      xi = xi + numpy.dot( dxi_root, self.invroottransmatrix.T )
      iiter += 1
      assert iiter < 100, 'failed to converge in 100 iterations'
    return xi

  def test_eval( self ):
    numpy.testing.assert_array_almost_equal(
      self.n_op( *self.argsfun.eval(self.elem,self.points) ),
        self.op( *self.args ).compiled().eval(self.elem,self.points), decimal=15 )

  def test_getitem( self ):
    shape = self.op( *self.args ).shape
    for idim in range( len(shape) ):
      s = (Ellipsis,) + (slice(None),)*idim + (shape[idim]//2,) + (slice(None),)*(len(shape)-idim-1)
      numpy.testing.assert_array_almost_equal(
        self.n_op( *self.argsfun.eval(self.elem,self.points) )[s],
          self.op( *self.args )[s].compiled().eval(self.elem,self.points), decimal=15 )

  def test_getslice( self ):
    shape = self.op( *self.args ).shape
    for idim in range( len(shape) ):
      if shape[idim] == 1:
        continue
      s = (Ellipsis,) + (slice(None),)*idim + (slice(0,shape[idim]-1),) + (slice(None),)*(len(shape)-idim-1)
      numpy.testing.assert_array_almost_equal(
        self.n_op( *self.argsfun.eval(self.elem,self.points) )[s],
          self.op( *self.args )[s].compiled().eval(self.elem,self.points), decimal=15 )

class FuncTestGrad( FuncTest ):

  def test_localgradient( self ):
    eps = 1e-6
    D = numpy.array([-.5*eps,.5*eps])[:,_,_] * self.invroottransmatrix.T[_,:,:]
    fdpoints = self.points[_,_,:,:] + D[:,:,_,:]
    F = self.n_op( *self.argsfun.eval(self.elem,fdpoints) )
    fdgrad = ((F[1]-F[0])/eps).transpose( numpy.roll(numpy.arange(F.ndim-1),-1) )
    G = function.localgradient( self.op( *self.args ), ndims=self.elem.ndims ).compiled()
    numpy.testing.assert_array_almost_equal( fdgrad, G.eval(self.elem,self.points), decimal=5 )

  def test_gradient( self ):
    eps = 1e-6
    D = numpy.array([-.5*eps,.5*eps])[:,_,_] * numpy.eye(2)
    fdpoints = self.find( self.geomfun.eval(self.elem,self.points)[_,_,:,:] + D[:,:,_,:], self.points[_,_,:,:] )
    F = self.n_op( *self.argsfun.eval(self.elem,fdpoints) )
    fdgrad = ((F[1]-F[0])/eps).transpose( numpy.roll(numpy.arange(F.ndim-1),-1) )
    G = self.op( *self.args ).grad(self.geom).compiled()
    numpy.testing.assert_array_almost_equal( fdgrad, G.eval(self.elem,self.points), decimal=5 )

  def test_doublegradient( self ):
    eps = 1e-5
    D = numpy.array([-.5*eps,.5*eps])[:,_,_] * numpy.eye(2)
    DD = D[:,_,:,_,:] + D[_,:,_,:,:]
    fdpoints = self.find( self.geomfun.eval(self.elem,self.points)[_,_,_,_,:,:] + DD[:,:,:,:,_,:], self.points[_,_,_,_,:,:] )
    F = self.n_op( *self.argsfun.eval(self.elem,fdpoints) )
    fddgrad = (((F[1,1]-F[1,0])-(F[0,1]-F[0,0]))/(eps**2)).transpose( numpy.roll(numpy.arange(F.ndim-2),-2) )
    G = self.op( *self.args ).grad(self.geom).grad(self.geom).compiled()
    numpy.testing.assert_array_almost_equal( fddgrad, G.eval(self.elem,self.points), decimal=2 )

  def test_opposite( self ):
    opposite_args = function.Tuple([ function.opposite(arg) for arg in self.args ]).compiled()
    numpy.testing.assert_array_almost_equal(
      self.n_op( *opposite_args.eval(self.iface,self.ifpoints) ),
        function.opposite( self.op( *self.args ) ).compiled().eval(self.iface,self.ifpoints), decimal=15 )

# UNARY POINTWISE OPERATIONS

class TestSin( FuncTestGrad ):
  def __init__( self ):
    FuncTestGrad.__init__( self, function.sin, numpy.sin, (3,) )

class TestCos( FuncTestGrad ):
  def __init__( self ):
    FuncTestGrad.__init__( self, function.cos, numpy.cos, (3,) )

class TestTan( FuncTestGrad ):
  def __init__( self ):
    FuncTestGrad.__init__( self, function.tan, numpy.tan, (3,) )

class TestSqrt( FuncTestGrad ):
  def __init__( self ):
    FuncTestGrad.__init__( self, function.sqrt, numpy.sqrt, (3,) )

class TestLog( FuncTestGrad ):
  def __init__( self ):
    FuncTestGrad.__init__( self, function.ln, numpy.log, (3,) )

class TestLog2( FuncTestGrad ):
  def __init__( self ):
    FuncTestGrad.__init__( self, function.log2, numpy.log2, (3,) )

class TestLog10( FuncTestGrad ):
  def __init__( self ):
    FuncTestGrad.__init__( self, function.log10, numpy.log10, (3,) )

class TestExp( FuncTestGrad ):
  def __init__( self ):
    FuncTestGrad.__init__( self, function.exp, numpy.exp, (3,) )

class TestArctanh( FuncTestGrad ):
  def __init__( self ):
    FuncTestGrad.__init__( self, function.arctanh, numpy.arctanh, (3,) )

class TestTanh( FuncTestGrad ):
  def __init__( self ):
    FuncTestGrad.__init__( self, function.tanh, numpy.tanh, (3,) )

class TestCosh( FuncTestGrad ):
  def __init__( self ):
    FuncTestGrad.__init__( self, function.cosh, numpy.cosh, (3,) )

class TestSinh( FuncTestGrad ):
  def __init__( self ):
    FuncTestGrad.__init__( self, function.sinh, numpy.sinh, (3,) )

class TestAbs( FuncTestGrad ):
  def __init__( self ):
    FuncTestGrad.__init__( self, function.abs, numpy.abs, (3,) )

class TestSign( FuncTestGrad ):
  def __init__( self ):
    FuncTestGrad.__init__( self, function.sign, numpy.sign, (3,) )

class TestPower( FuncTestGrad ):
  def __init__( self ):
    FuncTestGrad.__init__( self, lambda a: function.power(a,1.5), lambda a: numpy.power(a,1.5), (3,) )

class TestNegative( FuncTestGrad ):
  def __init__( self ):
    FuncTestGrad.__init__( self, function.negative, numpy.negative, (3,) )

class TestReciprocal( FuncTestGrad ):
  def __init__( self ):
    FuncTestGrad.__init__( self, function.reciprocal, numpy.reciprocal, (3,) )

class TestArcsin( FuncTestGrad ):
  def __init__( self ):
    FuncTestGrad.__init__( self, function.arcsin, numpy.arcsin, (3,) )


# UNARY ARRAY OPERATIONS

class TestProduct( FuncTestGrad ):
  def __init__( self ):
    FuncTestGrad.__init__( self, lambda a: function.product(a,1), lambda a: numpy.product(a,-2), (2,3,2) )

class TestNorm2( FuncTestGrad ):
  def __init__( self ):
    FuncTestGrad.__init__( self, lambda a: function.norm2(a,1), lambda a: (a**2).sum(-2)**.5, (2,3,2) )

class TestSum( FuncTestGrad ):
  def __init__( self ):
    FuncTestGrad.__init__( self, lambda a: function.sum(a,1), lambda a: a.sum(-2), (2,3,2) )

class TestAlign( FuncTestGrad ):
  def __init__( self ):
    FuncTestGrad.__init__( self, lambda a: function.align(a,[0,2],3), lambda a: a[...,:,_,:], (2,3) )

class TestGet( FuncTestGrad ):
  def __init__( self ):
    FuncTestGrad.__init__( self, lambda a: function.get(a,1,1), lambda a: a[...,1,:], (2,3,2) )

class TestTakediag1x1( FuncTestGrad ):
  def __init__( self ):
    FuncTestGrad.__init__( self, lambda a: function.takediag(a,0,2), lambda a: numeric.takediag(a.swapaxes(-3,-2)), (1,2,1) )

class TestTakediag2x2( FuncTestGrad ):
  def __init__( self ):
    FuncTestGrad.__init__( self, lambda a: function.takediag(a,0,2), lambda a: numeric.takediag(a.swapaxes(-3,-2)), (2,3,2) )

class TestTakediag3x3( FuncTestGrad ):
  def __init__( self ):
    FuncTestGrad.__init__( self, lambda a: function.takediag(a,0,2), lambda a: numeric.takediag(a.swapaxes(-3,-2)), (3,2,3) )

class TestDet1x1( FuncTestGrad ):
  def __init__( self ):
    FuncTestGrad.__init__( self, lambda a: function.determinant(a,(0,2)), lambda a: numpy.linalg.det(a.swapaxes(-3,-2)), (1,3,1) )

class TestDet2x2( FuncTestGrad ):
  def __init__( self ):
    FuncTestGrad.__init__( self, lambda a: function.determinant(a,(0,2)), lambda a: numpy.linalg.det(a.swapaxes(-3,-2)), (2,3,2) )

class TestDet3x3( FuncTestGrad ):
  def __init__( self ):
    FuncTestGrad.__init__( self, lambda a: function.determinant(a,(0,2)), lambda a: numpy.linalg.det(a.swapaxes(-3,-2)), (3,2,3) )

class TestInv1x1( FuncTestGrad ):
  def __init__( self ):
    FuncTestGrad.__init__( self, lambda a: function.inverse(a,(0,2)), lambda a: numpy.linalg.inv(a.swapaxes(-3,-2)).swapaxes(-3,-2), (1,3,1) )

class TestInv2x2( FuncTestGrad ):
  def __init__( self ):
    FuncTestGrad.__init__( self, lambda a: function.inverse(a,(0,2)), lambda a: numpy.linalg.inv(a.swapaxes(-3,-2)).swapaxes(-3,-2), (2,3,2) )

class TestInv3x3( FuncTestGrad ):
  def __init__( self ):
    FuncTestGrad.__init__( self, lambda a: function.inverse(a,(0,2)), lambda a: numpy.linalg.inv(a.swapaxes(-3,-2)).swapaxes(-3,-2), (3,2,3) )

class TestRepeat( FuncTestGrad ):
  def __init__( self ):
    FuncTestGrad.__init__( self, lambda a: function.repeat(a,3,1), lambda a: numpy.repeat(a,3,-2), (2,1,2) )

class TestDiagonalize( FuncTestGrad ):
  def __init__( self ):
    FuncTestGrad.__init__( self, function.diagonalize, numeric.diagonalize, (2,1,2) )


# BINARY OPERATIONS

class TestMul( FuncTestGrad ):
  def __init__( self ):
    FuncTestGrad.__init__( self, lambda a,b: a * b, numpy.multiply, (3,1), (1,3) )

class TestDiv( FuncTestGrad ):
  def __init__( self ):
    FuncTestGrad.__init__( self, lambda a,b: a / b, numpy.divide, (3,1), (1,3) )

class TestAdd( FuncTestGrad ):
  def __init__( self ):
    FuncTestGrad.__init__( self, lambda a,b: a + b, numpy.add, (3,1), (1,3) )

class TestSub( FuncTestGrad ):
  def __init__( self ):
    FuncTestGrad.__init__( self, lambda a,b: a - b, numpy.subtract, (3,1), (1,3) )

class TestDot( FuncTestGrad ):
  def __init__( self ):
    FuncTestGrad.__init__( self, lambda a,b: (a*b).sum(-2), lambda a,b: (a*b).sum(-2), (2,3,1), (1,3,2) )

class TestCross( FuncTestGrad ):
  def __init__( self ):
    FuncTestGrad.__init__( self, lambda a,b: function.cross(a,b,-2), lambda a,b: numpy.cross(a,b,axis=-2), (2,3,1), (1,3,2) )

class TestMin( FuncTestGrad ):
  def __init__( self ):
    FuncTestGrad.__init__( self, lambda a,b: function.min(a,b), numpy.minimum, (3,1), (1,3) )

class TestMax( FuncTestGrad ):
  def __init__( self ):
    FuncTestGrad.__init__( self, lambda a,b: function.max(a,b), numpy.maximum, (3,1), (1,3) )

class TestGreater( FuncTestGrad ):
  def __init__( self ):
    FuncTestGrad.__init__( self, lambda a,b: function.greater(a,b), numpy.greater, (3,1), (1,3) )

class TestLess( FuncTestGrad ):
  def __init__( self ):
    FuncTestGrad.__init__( self, lambda a,b: function.less(a,b), numpy.less, (3,1), (1,3) )

class TestArctan2( FuncTestGrad ):
  def __init__( self ):
    FuncTestGrad.__init__( self, function.arctan2, numpy.arctan2, (3,1), (1,3) )

class TestStack( FuncTestGrad ):
  def __init__( self ):
    FuncTestGrad.__init__( self, lambda a,b: function.stack([a,b]), lambda a,b: numpy.concatenate( [a[...,_,:],b[...,_,:]], axis=-2), (3,), (3,) )

# EVAL ONLY

class TestEig( FuncTest ):
  def __init__( self ):
    FuncTest.__init__( self, lambda a: function.eig(a,symmetric=False)[1], lambda a: numpy.array([ numpy.linalg.eig(ai)[1] for ai in a ]), (3,3) )
  

# vim:shiftwidth=2:foldmethod=indent:foldnestmax=2
