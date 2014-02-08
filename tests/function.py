from test import unittest
from nutils import *
import numpy

domainelem = element.Element( element.Dummy(2) )
r, theta = function.ElemFunc( domainelem )
geom = r * function.stack([ function.cos(theta), function.sin(theta) ])

line = element.Line()
line_points = line.getischeme( 'uniform2' )[:,:1]

quad = line**2
quad_points = quad.getischeme( 'uniform2' )[:,:2]

interelem = element.Element( simplex=quad, # corners at (1,0), (3,-1), (4,2), (2,3)
  vertices=tuple( 'A(%d)'%i for i in range(4) ),
  parent=( domainelem, transform.Linear( numeric.array([[2,1],[-1,3]]) ) + [1,0]),
)
elem = element.Element( simplex=quad, # corners at (3,-1), (2,-4), (4,-5), (5,-2)
  vertices=tuple( 'B(%d)'%i for i in range(4) ),
  parent=( interelem, transform.Linear( numeric.array([[0,1],[-1,0]]) ) + [1,0] ),
)
iface = element.Element( line,
  vertices=tuple( 'C(%d)'%i for i in range(2) ),
  interface=(elem.edge(1).context,elem.edge(0).context),
)
funcsp = function.function( ndims=2, ndofs=6,
  fmap={ elem: line.stdfunc(1) * line.stdfunc(2) },
  nmap={ elem: numeric.arange(6) },
)
geom_compiled = geom.compiled()

def find( target, xi0 ):
  ndim, = geom.shape
  J = function.localgradient( geom, ndim )
  Jinv = function.inverse( J ).compiled()
  countdown = 5
  iiter = 0
  xi = xi0
  while countdown:
    err = target - geom_compiled.eval(elem,xi)
    if numeric.less( numpy.abs(err), 1e-12 ).all():
      countdown -= 1
    dxi_root = ( Jinv.eval(elem,xi) * err[...,_,:] ).sum(-1)
    xi = xi + numeric.dot( dxi_root, elem.root_transform.inv.matrix.T )
    iiter += 1
    assert iiter < 100, 'failed to converge in 100 iterations'
  return xi

def checkfunc( name, op, n_op, *shapes ):

  print name,

  numpy.random.seed(0)
  args = [ ( numpy.random.uniform( size=shape+(funcsp.shape[0],) ) * funcsp ).sum() for shape in shapes ]
  argsfunc = function.Tuple( args ).compiled()
  op_args = op( *args )

  @unittest
  def eval():
    op_fun = op_args.compiled()
    numpy.testing.assert_array_almost_equal(
      n_op( *argsfunc.eval(elem,quad_points) ), op_fun.eval(elem,quad_points), decimal=15 )

  @unittest
  def getitem():
    shape = op_args.shape
    for idim in range( len(shape) ):
      s = (Ellipsis,) + (slice(None),)*idim + (shape[idim]//2,) + (slice(None),)*(len(shape)-idim-1)
      op_fun = op_args[s].compiled()
      numpy.testing.assert_array_almost_equal(
        n_op( *argsfunc.eval(elem,quad_points) )[s], op_fun.eval(elem,quad_points), decimal=15 )

  @unittest
  def getslice():
    shape = op_args.shape
    for idim in range( len(shape) ):
      if shape[idim] == 1:
        continue
      s = (Ellipsis,) + (slice(None),)*idim + (slice(0,shape[idim]-1),) + (slice(None),)*(len(shape)-idim-1)
      op_fun = op_args[s].compiled()
      numpy.testing.assert_array_almost_equal(
        n_op( *argsfunc.eval(elem,quad_points) )[s], op_fun.eval(elem,quad_points), decimal=15 )

  @unittest
  def localgradient():
    eps = 1e-6
    D = numeric.array([-.5*eps,.5*eps])[:,_,_] * elem.root_transform.inv.matrix.T[_,:,:]
    fdpoints = quad_points[_,_,:,:] + D[:,:,_,:]
    F = n_op( *argsfunc.eval(elem,fdpoints) )
    fdgrad = ((F[1]-F[0])/eps).transpose( numeric.roll(numeric.arange(F.ndim-1),-1) )
    G = function.localgradient( op_args, ndims=elem.ndims ).compiled()
    numpy.testing.assert_array_almost_equal( fdgrad, G.eval(elem,quad_points), decimal=5 )

  @unittest
  def gradient():
    eps = 1e-6
    D = numeric.array([-.5*eps,.5*eps])[:,_,_] * numeric.eye(2)
    fdpoints = find( geom_compiled.eval(elem,quad_points)[_,_,:,:] + D[:,:,_,:], quad_points[_,_,:,:] )
    F = n_op( *argsfunc.eval(elem,fdpoints) )
    fdgrad = ((F[1]-F[0])/eps).transpose( numeric.roll(numeric.arange(F.ndim-1),-1) )
    G = op_args.grad(geom).compiled()
    numpy.testing.assert_array_almost_equal( fdgrad, G.eval(elem,quad_points), decimal=5 )

  @unittest
  def doublegradient():
    eps = 1e-5
    D = numeric.array([-.5*eps,.5*eps])[:,_,_] * numeric.eye(2)
    DD = D[:,_,:,_,:] + D[_,:,_,:,:]
    fdpoints = find( geom_compiled.eval(elem,quad_points)[_,_,_,_,:,:] + DD[:,:,:,:,_,:], quad_points[_,_,_,_,:,:] )
    F = n_op( *argsfunc.eval(elem,fdpoints) )
    fddgrad = (((F[1,1]-F[1,0])-(F[0,1]-F[0,0]))/(eps**2)).transpose( numeric.roll(numeric.arange(F.ndim-2),-2) )
    G = op_args.grad(geom).grad(geom).compiled()
    numpy.testing.assert_array_almost_equal( fddgrad, G.eval(elem,quad_points), decimal=2 )

  @unittest
  def opposite():
    opposite_args = function.Tuple([ function.opposite(arg) for arg in args ]).compiled()
    opposite_func = function.opposite( op_args ).compiled()
    numpy.testing.assert_array_almost_equal(
      n_op( *opposite_args.eval(iface,line_points) ),
        opposite_func.eval(iface,line_points), decimal=15 )


## UNARY POINTWISE

checkfunc( 'sin', function.sin, numeric.sin, (3,) )
checkfunc( 'cos', function.cos, numeric.cos, (3,) )
checkfunc( 'tan', function.tan, numeric.tan, (3,) )
checkfunc( 'sqrt', function.sqrt, numeric.sqrt, (3,) )
checkfunc( 'ln', function.ln, numeric.log, (3,) )
checkfunc( 'log2', function.log2, numeric.log2, (3,) )
checkfunc( 'log10', function.log10, numeric.log10, (3,) )
checkfunc( 'exp', function.exp, numeric.exp, (3,) )
checkfunc( 'arctanh', function.arctanh, numeric.arctanh, (3,) )
checkfunc( 'tanh', function.tanh, numeric.tanh, (3,) )
checkfunc( 'cosh', function.cosh, numeric.cosh, (3,) )
checkfunc( 'sinh', function.sinh, numeric.sinh, (3,) )
checkfunc( 'abs', function.abs, numeric.abs, (3,) )
checkfunc( 'sign', function.sign, numeric.sign, (3,) )
checkfunc( 'sign', function.sign, numeric.sign, (3,) )
checkfunc( 'power', lambda a: function.power(a,1.5), lambda a: numeric.power(a,1.5), (3,) )
checkfunc( 'negative', function.negative, numeric.negative, (3,) )
checkfunc( 'reciprocal', function.reciprocal, numeric.reciprocal, (3,) )
checkfunc( 'arcsin', function.arcsin, numeric.arcsin, (3,) )
checkfunc( 'arccos', function.arccos, numeric.arccos, (3,) )

## UNARY ARRAY

checkfunc( 'product', lambda a: function.product(a,1), lambda a: numeric.product(a,-2), (2,3,2) )
checkfunc( 'norm2', lambda a: function.norm2(a,1), lambda a: (a**2).sum(-2)**.5, (2,3,2) )
checkfunc( 'sum', lambda a: function.sum(a,1), lambda a: a.sum(-2), (2,3,2) )
checkfunc( 'align', lambda a: function.align(a,[0,2],3), lambda a: a[...,:,_,:], (2,3) )
checkfunc( 'get', lambda a: function.get(a,1,1), lambda a: a[...,1,:], (2,3,2) )
checkfunc( 'takediag1x1', lambda a: function.takediag(a,0,2), lambda a: numeric.takediag(a.swapaxes(-3,-2)), (1,2,1) )
checkfunc( 'takediag2x2', lambda a: function.takediag(a,0,2), lambda a: numeric.takediag(a.swapaxes(-3,-2)), (2,3,2) )
checkfunc( 'takediag3x3', lambda a: function.takediag(a,0,2), lambda a: numeric.takediag(a.swapaxes(-3,-2)), (3,2,3) )
checkfunc( 'det1x1', lambda a: function.determinant(a,(0,2)), lambda a: numeric.det(a.swapaxes(-3,-2)), (1,3,1) )
checkfunc( 'det2x2', lambda a: function.determinant(a,(0,2)), lambda a: numeric.det(a.swapaxes(-3,-2)), (2,3,2) )
checkfunc( 'det3x3', lambda a: function.determinant(a,(0,2)), lambda a: numeric.det(a.swapaxes(-3,-2)), (3,2,3) )
checkfunc( 'inv1x1', lambda a: function.inverse(a,(0,2)), lambda a: numeric.inv(a.swapaxes(-3,-2)).swapaxes(-3,-2), (1,3,1) )
checkfunc( 'inv2x2', lambda a: function.inverse(a,(0,2)), lambda a: numeric.inv(a.swapaxes(-3,-2)).swapaxes(-3,-2), (2,3,2) )
checkfunc( 'inv3x3', lambda a: function.inverse(a,(0,2)), lambda a: numeric.inv(a.swapaxes(-3,-2)).swapaxes(-3,-2), (3,2,3) )
checkfunc( 'repeat', lambda a: function.repeat(a,3,1), lambda a: numeric.repeat(a,3,-2), (2,1,2) )
checkfunc( 'diagonalize', function.diagonalize, numeric.diagonalize, (2,1,2) )

## BINARY

checkfunc( 'mul', lambda a,b: a * b, numeric.multiply, (3,1), (1,3) )
checkfunc( 'div', lambda a,b: a / b, numeric.divide, (3,1), (1,3) )
checkfunc( 'add', lambda a,b: a + b, numeric.add, (3,1), (1,3) )
checkfunc( 'sub', lambda a,b: a - b, numeric.subtract, (3,1), (1,3) )
checkfunc( 'dot', lambda a,b: (a*b).sum(-2), lambda a,b: (a*b).sum(-2), (2,3,1), (1,3,2) )
checkfunc( 'cross', lambda a,b: function.cross(a,b,-2), lambda a,b: numeric.cross(a,b,axis=-2), (2,3,1), (1,3,2) )
checkfunc( 'min', lambda a,b: function.min(a,b), numeric.minimum, (3,1), (1,3) )
checkfunc( 'max', lambda a,b: function.max(a,b), numeric.maximum, (3,1), (1,3) )
checkfunc( 'greater', lambda a,b: function.greater(a,b), numeric.greater, (3,1), (1,3) )
checkfunc( 'less', lambda a,b: function.less(a,b), numeric.less, (3,1), (1,3) )
checkfunc( 'arctan2', function.arctan2, numeric.arctan2, (3,1), (1,3) )
checkfunc( 'stack', lambda a,b: function.stack([a,b]), lambda a,b: numeric.concatenate( [a[...,_,:],b[...,_,:]], axis=-2), (3,), (3,) )

# vim:shiftwidth=2:foldmethod=indent:foldnestmax=2
