from nutils import *
from . import register, unittest


@register( 'sin', function.sin, numpy.sin, [(3,)] )
@register( 'cos', function.cos, numpy.cos, [(3,)] )
@register( 'tan', function.tan, numpy.tan, [(3,)] )
@register( 'sqrt', function.sqrt, numpy.sqrt, [(3,)] )
@register( 'log', function.ln, numpy.log, [(3,)] )
@register( 'log2', function.log2, numpy.log2, [(3,)] )
@register( 'log10', function.log10, numpy.log10, [(3,)] )
@register( 'exp', function.exp, numpy.exp, [(3,)] )
@register( 'arctanh', function.arctanh, numpy.arctanh, [(3,)] )
@register( 'tanh', function.tanh, numpy.tanh, [(3,)] )
@register( 'cosh', function.cosh, numpy.cosh, [(3,)] )
@register( 'sinh', function.sinh, numpy.sinh, [(3,)] )
@register( 'abs', function.abs, numpy.abs, [(3,)] )
@register( 'sign', function.sign, numpy.sign, [(3,)] )
@register( 'power', lambda a: function.power(a,1.5), lambda a: numpy.power(a,1.5), [(3,)] )
@register( 'negative', function.negative, numpy.negative, [(3,)] )
@register( 'reciprocal', function.reciprocal, numpy.reciprocal, [(3,)] )
@register( 'arcsin', function.arcsin, numpy.arcsin, [(3,)] )
@register( 'ln', function.ln, numpy.log, [(3,)] )
@register( 'product', lambda a: function.product(a,1), lambda a: numpy.product(a,-2), [(2,3,2)] )
@register( 'norm2', lambda a: function.norm2(a,1), lambda a: (a**2).sum(-2)**.5, [(2,3,2)] )
@register( 'sum', lambda a: function.sum(a,1), lambda a: a.sum(-2), [(2,3,2)] )
@register( 'align', lambda a: function.align(a,[0,2],3), lambda a: a[...,:,_,:], [(2,3)] )
@register( 'get', lambda a: function.get(a,1,1), lambda a: a[...,1,:], [(2,3,2)] )
@register( 'takediag121', lambda a: function.takediag(a,0,2), lambda a: numeric.takediag(a.swapaxes(-3,-2)), [(1,2,1)] )
@register( 'takediag232', lambda a: function.takediag(a,0,2), lambda a: numeric.takediag(a.swapaxes(-3,-2)), [(2,3,2)] )
@register( 'takediag323', lambda a: function.takediag(a,0,2), lambda a: numeric.takediag(a.swapaxes(-3,-2)), [(3,2,3)] )
@register( 'determinant131', lambda a: function.determinant(a,(0,2)), lambda a: numpy.linalg.det(a.swapaxes(-3,-2)), [(1,3,1)] )
@register( 'determinant232', lambda a: function.determinant(a,(0,2)), lambda a: numpy.linalg.det(a.swapaxes(-3,-2)), [(2,3,2)] )
@register( 'determinant323', lambda a: function.determinant(a,(0,2)), lambda a: numpy.linalg.det(a.swapaxes(-3,-2)), [(3,2,3)] )
@register( 'inverse131', lambda a: function.inverse(a,(0,2)), lambda a: numpy.linalg.inv(a.swapaxes(-3,-2)).swapaxes(-3,-2), [(1,3,1)] )
@register( 'inverse232', lambda a: function.inverse(a,(0,2)), lambda a: numpy.linalg.inv(a.swapaxes(-3,-2)).swapaxes(-3,-2), [(2,3,2)] )
@register( 'inverse323', lambda a: function.inverse(a,(0,2)), lambda a: numpy.linalg.inv(a.swapaxes(-3,-2)).swapaxes(-3,-2), [(3,2,3)] )
@register( 'repeat', lambda a: function.repeat(a,3,1), lambda a: numpy.repeat(a,3,-2), [(2,1,2)] )
@register( 'diagonalize', function.diagonalize, numeric.diagonalize, [(2,1,2)] )
@register( 'multiply', function.multiply, numpy.multiply, [(3,1),(1,3)] )
@register( 'divide', function.divide, numpy.divide, [(3,1),(1,3)] )
@register( 'divide2', lambda a: function.asarray(a)/2, lambda a: a/2, [(3,1)] )
@register( 'add', function.add, numpy.add, [(3,1),(1,3)] )
@register( 'subtract', function.subtract, numpy.subtract, [(3,1),(1,3)] )
@register( 'product2', lambda a,b: function.multiply(a,b).sum(-2), lambda a,b: (a*b).sum(-2), [(2,3,1),(1,3,2)] )
@register( 'cross', lambda a,b: function.cross(a,b,-2), lambda a,b: numpy.cross(a,b,axis=-2), [(2,3,1),(1,3,2)] )
@register( 'min', lambda a,b: function.min(a,b), numpy.minimum, [(3,1),(1,3)] )
@register( 'max', lambda a,b: function.max(a,b), numpy.maximum, [(3,1),(1,3)] )
@register( 'equal', lambda a,b: function.equal(a,b), numpy.equal, [(3,1),(1,3)] )
@register( 'greater', lambda a,b: function.greater(a,b), numpy.greater, [(3,1),(1,3)] )
@register( 'less', lambda a,b: function.less(a,b), numpy.less, [(3,1),(1,3)] )
@register( 'arctan2', function.arctan2, numpy.arctan2, [(3,1),(1,3)] )
@register( 'stack', lambda a,b: function.stack([a,b]), lambda a,b: numpy.concatenate( [a[...,_,:],b[...,_,:]], axis=-2), [(3,),(3,)] )
@register( 'concatenate', lambda a,b: function.concatenate([a,b],axis=0), lambda a,b: numpy.concatenate( [a,b], axis=-2), [(3,2),(2,2)] )
@register( 'eig', lambda a: function.eig(a,symmetric=False)[1], lambda a: numpy.linalg.eig(a)[1], [(3,3)], hasgrad=False )
@register( 'trignormal', lambda a: function.trignormal(a), lambda a: numpy.array([ numpy.cos(a), numpy.sin(a) ]).T, [()] )
@register( 'trigtangent', lambda a: function.trigtangent(a), lambda a: numpy.array([ -numpy.sin(a), numpy.cos(a) ]).T, [()] )
@register( 'mod', lambda a,b: function.mod(a,b), lambda a,b: numpy.mod(a,b), [(3,),(3,)], hasgrad=False )
@register( 'kronecker', lambda f: function.kronecker(f,axis=-2,length=3,pos=1), lambda a: numeric.kronecker(a,axis=-2,length=3,pos=1), [(2,3,)] )
def check( op, n_op, shapes, hasgrad=True ):

  anchor = transform.roottrans( 'test', (0,0) )
  roottrans = transform.affine( [[0,1],[-1,0]], [1,0] ) >> transform.affine( [[2,1],[-1,3]], [1,0] )
  axes = topology.DimAxis(0,1,False), topology.DimAxis(0,1,False)
  domain = topology.StructuredTopology( root=anchor<<roottrans, axes=axes )
  elem, = domain
  iface = element.Element( elem.edge(0).reference, elem.edge(0).transform, elem.edge(1).transform )
  ifpoints, ifweights = iface.reference.getischeme('uniform2')

  r, theta = function.localcoords( 2 ) # corners at (0,0), (0,1), (1,1), (1,0)
  geom = r * function.stack([ function.cos(theta), function.sin(theta) ])

  basis = domain.basis( 'spline', degree=(1,2) )

  numpy.random.seed(0)
  args = [ ( numpy.random.uniform( size=shape+basis.shape ) * basis ).sum(-1) for shape in shapes ]
  points, weights = elem.reference.getischeme('uniform2')

  argsfun = function.Tuple( args )
  invroottransmatrix = roottrans.invlinear.astype( float )

  @unittest
  def evalconst():
    constargs = [ numpy.random.uniform( size=shape ) for shape in shapes ]
    numpy.all( n_op( *constargs ) == op( *constargs ).eval(elem,points) )

  @unittest
  def eval():
    numpy.testing.assert_array_almost_equal(
      n_op( *argsfun.eval(elem,points) ),
        op( *args ).eval(elem,points), decimal=15 )

  shape = op( *args ).shape
  shapearg = numpy.random.uniform( size=shape )

  @unittest
  def getitem():
    for idim in range( len(shape) ):
      s = (Ellipsis,) + (slice(None),)*idim + (shape[idim]//2,) + (slice(None),)*(len(shape)-idim-1)
      numpy.testing.assert_array_almost_equal(
        n_op( *argsfun.eval(elem,points) )[s],
          op( *args )[s].eval(elem,points), decimal=15 )

  @unittest
  def align():
    ndim = len(shape)+1
    axes = (numpy.arange(len(shape))+2) % ndim
    numpy.testing.assert_array_almost_equal(
      numeric.align( n_op( *argsfun.eval(elem,points) ), [0]+list(axes+1), ndim+1 ),
     function.align( op( *args ), axes, ndim ).eval(elem,points), decimal=15 )

  count = {}
  for i, sh in enumerate(shape):
    count.setdefault(sh,[]).append(i)
  pairs = [ sorted(axes[:2]) for axes in count.values() if len(axes) > 1 ] # axis pairs with same length
  if pairs:
    @unittest
    def takediag():
      for ax1, ax2 in pairs:
        alignaxes = list(range(ax1+1)) + [-2] + list(range(ax1+1,ax2)) + [-1] + list(range(ax2,len(shape)-1))
        numpy.testing.assert_array_almost_equal(
          numeric.takediag( numeric.align( n_op( *argsfun.eval(elem,points) ), alignaxes, len(shape)+1 ) ),
         function.takediag( op( *args ), ax1, ax2 ).eval(elem,points), decimal=15 )

  @unittest
  def concatenate():
    for idim in range(len(shape)):
      numpy.testing.assert_array_almost_equal(
        numpy.concatenate( [ n_op( *argsfun.eval(elem,points) ), shapearg[_].repeat(len(points),0) ], axis=idim+1 ),
       function.concatenate( [ op( *args ), shapearg ], axis=idim ).eval(elem,points), decimal=15 )


  @unittest
  def getslice():
    for idim in range( len(shape) ):
      if shape[idim] == 1:
        continue
      s = (Ellipsis,) + (slice(None),)*idim + (slice(0,shape[idim]-1),) + (slice(None),)*(len(shape)-idim-1)
      numpy.testing.assert_array_almost_equal(
        n_op( *argsfun.eval(elem,points) )[s],
          op( *args )[s].eval(elem,points), decimal=15 )

  @unittest
  def sumaxis():
    for idim in range( len(shape) ):
      numpy.testing.assert_array_almost_equal(
        n_op( *argsfun.eval(elem,points) ).sum(1+idim),
          op( *args ).sum(idim).eval(elem,points), decimal=15 )

  @unittest
  def add():
    numpy.testing.assert_array_almost_equal(
      n_op( *argsfun.eval(elem,points) ) + shapearg,
      ( op( *args ) + shapearg ).eval(elem,points), decimal=15 )

  @unittest
  def multiply():
    numpy.testing.assert_array_almost_equal(
      n_op( *argsfun.eval(elem,points) ) * shapearg,
      ( op( *args ) * shapearg ).eval(elem,points), decimal=15 )

  @unittest
  def power():
    numpy.testing.assert_array_almost_equal(
      n_op( *argsfun.eval(elem,points) )**3,
      ( op( *args )**3 ).eval(elem,points), decimal=14 )

  @unittest
  def edit():
    identity = lambda arg: function.edit( arg, identity )
    func = op( *args )
    assert identity(func) == func

  if not hasgrad:
    return

  def find( target, xi0 ):
    ndim, = geom.shape
    J = function.localgradient( geom, ndim )
    Jinv = function.inverse( J )
    countdown = 5
    iiter = 0
    assert target.shape[-1:] == geom.shape
    if xi0.shape != target.shape:
      tmp = numpy.empty_like( target )
      tmp[...] = xi0
      xi0 = tmp
    target = target.reshape( -1, target.shape[-1] )
    xi = xi0.reshape( -1, xi0.shape[-1] )
    while countdown:
      err = target - geom.eval(elem,xi)
      if numpy.all( numpy.abs(err) < 1e-12 ):
        countdown -= 1
      dxi_root = ( Jinv.eval(elem,xi) * err[...,_,:] ).sum(-1)
      #xi = xi + numpy.dot( dxi_root, elem.inv_root_transform.T )
      xi = xi + numpy.dot( dxi_root, invroottransmatrix.T )
      iiter += 1
      assert iiter < 100, 'failed to converge in 100 iterations'
    return xi.reshape( xi0.shape )

  @unittest
  def localgradient():
    eps = 1e-6
    D = numpy.array([-.5*eps,.5*eps])[:,_,_] * invroottransmatrix.T[_,:,:]
    fdpoints = points[_,_,:,:] + D[:,:,_,:]
    tmp = n_op( *argsfun.eval(elem,fdpoints.reshape(-1,fdpoints.shape[-1])) )
    F = tmp.reshape( fdpoints.shape[:-1] + tmp.shape[1:] )
    fdgrad = ((F[1]-F[0])/eps).transpose( numpy.roll(numpy.arange(F.ndim-1),-1) )
    G = function.localgradient( op( *args ), ndims=elem.ndims )
    exact = numpy.empty_like( fdgrad )
    exact[...] = G.eval(elem,points)
    numpy.testing.assert_array_almost_equal( fdgrad, exact, decimal=5 )

  @unittest
  def jacobian():
    eps = 1e-8
    numpy.random.seed(0)
    for iarg in range(len(args)):
      def f(x):
        myargs = list(args)
        myargs[iarg] = ( x * basis ).sum(-1)
        return op( *myargs )
      J = function.partial_derivative( f, 0 )
      x0 = numpy.random.uniform( size=shapes[iarg]+basis.shape )
      dx = numpy.random.normal( size=x0.shape ) * eps
      fx0, fx1, Jx0 = domain.elem_eval( [ f(x0), f(x0+dx), J(x0) ], ischeme='gauss1' )
      fx1approx = (fx0 + numeric.contract_fast( Jx0, dx, naxes=dx.ndim ))
      numpy.testing.assert_array_almost_equal( fx1approx, fx1, decimal=12 )

  @unittest
  def gradient( ):
    eps = 1e-6
    D = numpy.array([-.5*eps,.5*eps])[:,_,_] * numpy.eye(2)
    fdpoints = find( geom.eval(elem,points)[_,_,:,:] + D[:,:,_,:], points[_,_,:,:] )
    tmp = n_op( *argsfun.eval(elem,fdpoints.reshape(-1,fdpoints.shape[-1])) )
    F = tmp.reshape( fdpoints.shape[:-1] + tmp.shape[1:] )
    fdgrad = ((F[1]-F[0])/eps).transpose( numpy.roll(numpy.arange(F.ndim-1),-1) )
    G = op( *args ).grad(geom)
    exact = numpy.empty_like( fdgrad )
    exact[...] = G.eval(elem,points)
    numpy.testing.assert_array_almost_equal( fdgrad, exact, decimal=5 )

  @unittest
  def doublegradient():
    eps = .000022
    D = numpy.array([-.5*eps,.5*eps])[:,_,_] * numpy.eye(2)
    DD = D[:,_,:,_,:] + D[_,:,_,:,:]
    fdpoints = find( geom.eval(elem,points)[_,_,_,_,:,:] + DD[:,:,:,:,_,:], points[_,_,_,_,:,:] )
    tmp = n_op( *argsfun.eval(elem,fdpoints.reshape(-1,fdpoints.shape[-1])) )
    F = tmp.reshape( fdpoints.shape[:-1] + tmp.shape[1:] )
    fddgrad = (((F[1,1]-F[1,0])-(F[0,1]-F[0,0]))/(eps**2)).transpose( numpy.roll(numpy.arange(F.ndim-2),-2) )
    G = op( *args ).grad(geom).grad(geom)
    exact = numpy.empty_like( fddgrad )
    exact[...] = G.eval(elem,points)
    numpy.testing.assert_array_almost_equal( fddgrad, exact, decimal=2 )

  @unittest
  def opposite():
    opposite_args = function.Tuple([ function.opposite(arg) for arg in args ])
    numpy.testing.assert_array_almost_equal(
      n_op( *opposite_args.eval(iface,ifpoints) ),
        function.opposite( op( *args ) ).eval(iface,ifpoints), decimal=15 )


@register
def commutativity():

  numpy.random.seed(0)
  A = function.asarray( numpy.random.uniform( size=[2,3] ) )
  B = function.asarray( numpy.random.uniform( size=[2,3] ) )

  @unittest
  def add():
    assert function.add( A, B ) == function.add( B, A )

  @unittest
  def multiply():
    assert function.multiply( A, B ) == function.multiply( B, A )

  @unittest
  def dot():
    assert function.dot( A, B, axes=[0] ) == function.dot( B, A, axes=[0] )

  @unittest
  def combined():
    assert function.add( A, B ) * function.dot( A, B, axes=[0] ) == function.dot( B, A, axes=[0] ) * function.add( B, A )


@register
def sampled():

  domain, geom = mesh.demo()
  basis = domain.basis( 'std', degree=1 )
  numpy.random.seed(0)
  f = basis.dot( numpy.random.uniform(size=len(basis)) )
  f_sampled = domain.elem_eval( f, ischeme='gauss2', asfunction=True )

  @unittest
  def isarray():
    assert function.isarray( f_sampled )

  @unittest
  def values():
    diff = domain.integrate( f - f_sampled, ischeme='gauss2' )
    assert diff == 0

  @unittest( raises=function.EvaluationError )
  def pointset():
    domain.integrate( f_sampled, ischeme='uniform2' )
