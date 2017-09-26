from nutils import *
from . import *


@parametrize
class check(TestCase):

  def setUp(self):
    super().setUp()
    anchor = transform.roottrans('test', (0,0))
    roottrans = transform.affine([[0,1],[-1,0]], [1,0]) >> transform.affine([[2,1],[-1,3]], [1,0])
    axes = topology.DimAxis(0,1,False), topology.DimAxis(0,1,False)
    self.domain = topology.StructuredTopology(root=anchor<<roottrans, axes=axes)
    self.elem, = self.domain
    self.iface = element.Element(self.elem.edge(0).reference, self.elem.edge(0).transform, self.elem.edge(1).transform, oriented=True)
    self.ifpoints, ifweights = self.iface.reference.getischeme('uniform2')

    r, theta = function.rootcoords(2) # corners at (0,0), (0,1), (1,1), (1,0)
    self.geom = r * function.stack([function.cos(theta), function.sin(theta)])

    self.basis = self.domain.basis('spline', degree=(1,2))

    numpy.random.seed(0)
    self.args = [(numpy.random.uniform(size=shape+self.basis.shape) * self.basis).sum(-1) for shape in self.shapes]
    self.points, weights = self.elem.reference.getischeme('uniform2')

    self.argsfun = function.Tuple(self.args)

    self.shape = self.op(*self.args).shape
    self.shapearg = numpy.random.uniform(size=self.shape)

  def test_evalconst(self):
    constargs = [numpy.random.uniform(size=shape) for shape in self.shapes]
    numpy.testing.assert_array_almost_equal(
      self.n_op(*[arg[_] for arg in constargs]),
      self.op(*constargs).eval(self.elem,self.points), decimal=15)

  def test_eval(self):
    numpy.testing.assert_array_almost_equal(
      self.n_op(*self.argsfun.eval(self.elem,self.points)),
        self.op(*self.args).eval(self.elem,self.points), decimal=15)

  def test_getitem(self):
    for idim in range(len(self.shape)):
      s = (Ellipsis,) + (slice(None),)*idim + (self.shape[idim]//2,) + (slice(None),)*(len(self.shape)-idim-1)
      numpy.testing.assert_array_almost_equal(
        self.n_op(*self.argsfun.eval(self.elem,self.points))[s],
          self.op(*self.args)[s].eval(self.elem,self.points), decimal=15)

  def test_transpose(self):
    ndim = len(self.shape)
    trans = numpy.arange(ndim,0,-1) % ndim
    numpy.testing.assert_array_almost_equal(
        numpy.transpose(self.n_op(*self.argsfun.eval(self.elem,self.points)), [0]+list(trans+1)),
     function.transpose(self.op(*self.args), trans).eval(self.elem,self.points), decimal=15)

  def test_expand_dims(self):
    axis = (len(self.shape)+1) // 2
    numpy.testing.assert_array_almost_equal(
        numpy.expand_dims(self.n_op(*self.argsfun.eval(self.elem,self.points)), axis+1),
     function.expand_dims(self.op(*self.args), axis).eval(self.elem,self.points), decimal=15)

  def test_takediag(self):
    count = {}
    for i, sh in enumerate(self.shape):
      count.setdefault(sh,[]).append(i)
    pairs = [sorted(axes[:2]) for axes in count.values() if len(axes) > 1] # axis pairs with same length
    if pairs:
      for ax1, ax2 in pairs:
        numpy.testing.assert_array_almost_equal(
          numeric.takediag(self.n_op(*self.argsfun.eval(self.elem,self.points)), ax1+1, ax2+1),
         function.takediag(self.op(*self.args), ax1, ax2).eval(self.elem,self.points), decimal=15)

  def test_eig(self):
    count = {}
    for i, sh in enumerate(self.shape):
      count.setdefault(sh,[]).append(i)
    pairs = [sorted(axes[:2]) for axes in count.values() if len(axes) > 1] # axis pairs with same length
    if pairs and self.op(*self.args).dtype == float:
      for ax1, ax2 in pairs:
        A = self.op(*self.args).eval(self.elem,self.points)
        L, V = function.eig(self.op(*self.args), axes=(ax1,ax2)).eval(self.elem,self.points)
        M1 = (numpy.expand_dims(A,ax2+1) * numpy.expand_dims(V,ax2+2).swapaxes(ax1+1,ax2+2)).sum(ax2+2)
        M2 = (numpy.expand_dims(V,ax2+1) * numpy.expand_dims(L,ax2+2).swapaxes(ax1+1,ax2+2)).sum(ax2+2)
        numpy.testing.assert_array_almost_equal(M1, M2, decimal=12)

  def test_take(self):
    indices = [0,-1]
    for iax, sh in enumerate(self.shape):
      if sh >= 2:
        numpy.testing.assert_array_almost_equal(
          numpy.take(self.n_op(*self.argsfun.eval(self.elem,self.points)), indices, axis=iax+1),
       function.take(self.op(*self.args), indices, axis=iax).eval(self.elem,self.points), decimal=15)

  def test_diagonalize(self):
    for axis in range(len(self.shape)):
      for newaxis in range(axis+1, len(self.shape)+1):
        numpy.testing.assert_array_almost_equal(
          numeric.diagonalize(self.n_op(*self.argsfun.eval(self.elem,self.points)), axis+1, newaxis+1),
         function.diagonalize(self.op(*self.args), axis, newaxis).eval(self.elem,self.points), decimal=15)

  def test_product(self):
    for iax in range(len(self.shape)):
      numpy.testing.assert_array_almost_equal(
          numpy.product(self.n_op(*self.argsfun.eval(self.elem,self.points)), axis=iax+1),
       function.product(self.op(*self.args), axis=iax).eval(self.elem,self.points), decimal=15)

  def test_power(self):
    numpy.testing.assert_array_almost_equal(
        numpy.power(self.n_op(*self.argsfun.eval(self.elem,self.points)), 3),
     function.power(self.op(*self.args), 3).eval(self.elem,self.points), decimal=13)

  def test_concatenate(self):
    for idim in range(len(self.shape)):
      numpy.testing.assert_array_almost_equal(
        numpy.concatenate([self.n_op(*self.argsfun.eval(self.elem,self.points)), self.shapearg[_].repeat(len(self.points),0)], axis=idim+1),
       function.concatenate([self.op(*self.args), self.shapearg], axis=idim).eval(self.elem,self.points), decimal=15)

  def test_getslice(self):
    for idim in range(len(self.shape)):
      if self.shape[idim] == 1:
        continue
      s = (Ellipsis,) + (slice(None),)*idim + (slice(0,self.shape[idim]-1),) + (slice(None),)*(len(self.shape)-idim-1)
      numpy.testing.assert_array_almost_equal(
        self.n_op(*self.argsfun.eval(self.elem,self.points))[s],
          self.op(*self.args)[s].eval(self.elem,self.points), decimal=15)

  def test_sumaxis(self):
    for idim in range(len(self.shape)):
      numpy.testing.assert_array_almost_equal(
        self.n_op(*self.argsfun.eval(self.elem,self.points)).sum(1+idim),
          self.op(*self.args).sum(idim).eval(self.elem,self.points), decimal=15)

  def test_add(self):
    numpy.testing.assert_array_almost_equal(
      self.n_op(*self.argsfun.eval(self.elem,self.points)) + self.shapearg,
      (self.op(*self.args) + self.shapearg).eval(self.elem,self.points), decimal=15)

  def test_multiply(self):
    numpy.testing.assert_array_almost_equal(
      self.n_op(*self.argsfun.eval(self.elem,self.points)) * self.shapearg,
      (self.op(*self.args) * self.shapearg).eval(self.elem,self.points), decimal=15)

  def test_dot(self):
    for iax in range(len(self.shape)):
      numpy.testing.assert_array_almost_equal(
        numeric.contract(self.n_op(*self.argsfun.eval(self.elem,self.points)), self.shapearg, axis=iax+1),
        function.dot(self.op(*self.args), self.shapearg, axes=iax).eval(self.elem,self.points), decimal=15)

  def test_pointwise(self):
    numpy.testing.assert_array_almost_equal(
        numpy.sin(self.n_op(*self.argsfun.eval(self.elem,self.points))).astype(float), # "astype" necessary for boolean operations (float16->float64)
     function.sin(self.op(*self.args)).eval(self.elem,self.points), decimal=15)

  def test_cross(self):
    triaxes = [iax for iax, sh in enumerate(self.shape) if sh == 3]
    if triaxes:
      for iax in triaxes:
        numpy.testing.assert_array_almost_equal(
          numeric.cross(self.n_op(*self.argsfun.eval(self.elem,self.points)), self.shapearg, axis=iax+1),
          function.cross(self.op(*self.args), self.shapearg, axis=iax).eval(self.elem,self.points), decimal=15)

  def test_power(self):
    numpy.testing.assert_array_almost_equal(
      self.n_op(*self.argsfun.eval(self.elem,self.points))**3,
      (self.op(*self.args)**3).eval(self.elem,self.points), decimal=14)

  def test_mask(self):
    for idim in range(len(self.shape)):
      if self.shape[idim] <= 1:
        continue
      mask = numpy.ones(self.shape[idim], dtype=bool)
      mask[0] = False
      if self.shape[idim] > 2:
        mask[-1] = False
      numpy.testing.assert_array_almost_equal(
        self.n_op(*self.argsfun.eval(self.elem,self.points))[(slice(None,),)*(idim+1)+(mask,)],
        function.mask(self.op(*self.args), mask, axis=idim).eval(self.elem,self.points), decimal=15)

  def test_ravel(self):
    for idim in range(len(self.shape)-1):
      A = self.n_op(*self.argsfun.eval(self.elem,self.points))
      numpy.testing.assert_array_almost_equal(
        A.reshape(A.shape[:idim+1]+(-1,)+A.shape[idim+3:]),
        function.ravel(self.op(*self.args), axis=idim).eval(self.elem,self.points), decimal=15)

  def test_unravel(self):
    for idim in range(len(self.shape)):
      A = self.n_op(*self.argsfun.eval(self.elem,self.points))
      length = A.shape[idim+1]
      unravelshape = (length//3,3) if (length%3==0) else (length//2,2) if (length%2==0) else (length,1)
      numpy.testing.assert_array_almost_equal(
        A.reshape(A.shape[:idim+1]+unravelshape+A.shape[idim+2:]),
        function.unravel(self.op(*self.args), axis=idim, shape=unravelshape).eval(self.elem,self.points), decimal=15)

  def test_edit(self):
    def check_identity(arg):
      if function.isevaluable(arg):
        newarg = arg.edit(check_identity)
        self.assertEqual(arg, newarg)
      return arg
    check_identity(self.op(*self.args))

  def find(self, target, xi0):
    ndim, = self.geom.shape
    J = function.localgradient(self.geom, ndim)
    Jinv = function.inverse(J)
    countdown = 5
    iiter = 0
    self.assertEqual(target.shape[-1:], self.geom.shape)
    if xi0.shape != target.shape:
      tmp = numpy.empty_like(target)
      tmp[...] = xi0
      xi0 = tmp
    target = target.reshape(-1, target.shape[-1])
    xi = xi0.reshape(-1, xi0.shape[-1])
    while countdown:
      err = target - self.geom.eval(self.elem,xi)
      if numpy.less(numpy.abs(err), 1e-12).all():
        countdown -= 1
      dxi_root = (Jinv.eval(self.elem,xi) * err[...,_,:]).sum(-1)
      #xi = xi + numpy.dot(dxi_root, self.elem.inv_root_transform.T)
      xi = xi + dxi_root
      iiter += 1
      self.assertLess(iiter, 100, 'failed to converge in 100 iterations')
    return xi.reshape(xi0.shape)

  @parametrize.enable_if(lambda hasgrad, **kwargs: hasgrad)
  def test_localgradient(self):
    eps = 1e-6
    D = numpy.array([-.5*eps,.5*eps])[:,_,_] * numpy.eye(self.elem.ndims)
    fdpoints = self.points[_,_,:,:] + D[:,:,_,:]
    tmp = self.n_op(*self.argsfun.eval(self.elem,fdpoints.reshape(-1,fdpoints.shape[-1])))
    F = tmp.reshape(fdpoints.shape[:-1] + tmp.shape[1:])
    fdgrad = numpy.zeros(F.shape[1:], bool) if F.dtype.kind == 'b' else (F[1]-F[0])/eps
    fdgrad = fdgrad.transpose(numpy.roll(numpy.arange(F.ndim-1),-1))
    G = function.localgradient(self.op(*self.args), ndims=self.elem.ndims)
    exact = numpy.empty_like(fdgrad)
    exact[...] = G.eval(self.elem,self.points)
    numpy.testing.assert_array_almost_equal(fdgrad, exact, decimal=5)

  @parametrize.enable_if(lambda hasgrad, **kwargs: hasgrad)
  def test_jacobian(self):
    eps = 1e-8
    numpy.random.seed(0)
    for iarg in range(len(self.args)):
      x0 = numpy.random.uniform(size=self.shapes[iarg]+self.basis.shape)
      dx = numpy.random.normal(size=x0.shape) * eps
      x = function.Argument('x', x0.shape)
      f = self.op(*(*self.args[:iarg], (x*self.basis).sum(-1), *self.args[iarg+1:]))
      fx0, fx1, Jx0 = self.domain.elem_eval([f, function.replace_arguments(f, dict(x=x+dx)),function.derivative(f, x)], ischeme='gauss1', arguments=dict(x=x0))
      fx1approx = fx0 + numeric.contract_fast(Jx0, dx, naxes=dx.ndim)
      numpy.testing.assert_array_almost_equal(fx1approx, fx1, decimal=12)

  @parametrize.enable_if(lambda hasgrad, **kwargs: hasgrad)
  def test_gradient(self):
    eps = 1e-7
    D = numpy.array([-.5*eps,.5*eps])[:,_,_] * numpy.eye(2)
    fdpoints = self.find(self.geom.eval(self.elem,self.points)[_,_,:,:] + D[:,:,_,:], self.points[_,_,:,:])
    tmp = self.n_op(*self.argsfun.eval(self.elem,fdpoints.reshape(-1,fdpoints.shape[-1])))
    F = tmp.reshape(fdpoints.shape[:-1] + tmp.shape[1:])
    fdgrad = numpy.zeros(F.shape[1:], bool) if F.dtype.kind == 'b' else (F[1]-F[0])/eps
    fdgrad = fdgrad.transpose(numpy.roll(numpy.arange(F.ndim-1),-1))
    G = self.op(*self.args).grad(self.geom)
    exact = numpy.empty_like(fdgrad)
    exact[...] = G.eval(self.elem,self.points)
    numpy.testing.assert_array_almost_equal(fdgrad, exact, decimal=5)

  @parametrize.enable_if(lambda hasgrad, **kwargs: hasgrad)
  def test_doublegradient(self):
    eps = 1e-4
    D = numpy.array([-.5*eps,.5*eps])[:,_,_] * numpy.eye(2)
    DD = D[:,_,:,_,:] + D[_,:,_,:,:]
    fdpoints = self.find(self.geom.eval(self.elem,self.points)[_,_,_,_,:,:] + DD[:,:,:,:,_,:], self.points[_,_,_,_,:,:])
    tmp = self.n_op(*self.argsfun.eval(self.elem,fdpoints.reshape(-1,fdpoints.shape[-1])))
    F = tmp.reshape(fdpoints.shape[:-1] + tmp.shape[1:])
    fddgrad = numpy.zeros(F.shape[2:], bool) if F.dtype.kind == 'b' else ((F[1,1]-F[1,0])-(F[0,1]-F[0,0]))/(eps**2)
    fddgrad = fddgrad.transpose(numpy.roll(numpy.arange(F.ndim-2),-2))
    G = self.op(*self.args).grad(self.geom).grad(self.geom)
    exact = numpy.empty_like(fddgrad)
    exact[...] = G.eval(self.elem,self.points)
    numpy.testing.assert_allclose(fddgrad, exact, rtol=2e-4)

  @parametrize.enable_if(lambda hasgrad, **kwargs: hasgrad)
  def test_opposite(self):
    opposite_args = function.Tuple([function.opposite(arg) for arg in self.args])
    numpy.testing.assert_array_almost_equal(
      self.n_op(*opposite_args.eval(self.iface,self.ifpoints)),
        function.opposite(self.op(*self.args)).eval(self.iface,self.ifpoints), decimal=15)

_check = lambda name, op, n_op, shapes, hasgrad=True: check(name, op=op, n_op=n_op, shapes=shapes, hasgrad=hasgrad)
_check('sin', function.sin, numpy.sin, [(3,)])
_check('cos', function.cos, numpy.cos, [(3,)])
_check('tan', function.tan, numpy.tan, [(3,)])
_check('sqrt', function.sqrt, numpy.sqrt, [(3,)])
_check('log', function.ln, numpy.log, [(3,)])
_check('log2', function.log2, numpy.log2, [(3,)])
_check('log10', function.log10, numpy.log10, [(3,)])
_check('exp', function.exp, numpy.exp, [(3,)])
_check('arctanh', function.arctanh, numpy.arctanh, [(3,)])
_check('tanh', function.tanh, numpy.tanh, [(3,)])
_check('cosh', function.cosh, numpy.cosh, [(3,)])
_check('sinh', function.sinh, numpy.sinh, [(3,)])
_check('abs', function.abs, numpy.abs, [(3,)])
_check('sign', function.sign, numpy.sign, [(3,)])
_check('power', lambda a: function.power(a,1.5), lambda a: numpy.power(a,1.5), [(3,)])
_check('negative', function.negative, numpy.negative, [(3,)])
_check('reciprocal', function.reciprocal, numpy.reciprocal, [(3,)])
_check('arcsin', function.arcsin, numpy.arcsin, [(3,)])
_check('ln', function.ln, numpy.log, [(3,)])
_check('product', lambda a: function.product(a,1), lambda a: numpy.product(a,-2), [(2,3,2)])
_check('norm2', lambda a: function.norm2(a,1), lambda a: (a**2).sum(-2)**.5, [(2,3,2)])
_check('sum', lambda a: function.sum(a,1), lambda a: a.sum(-2), [(2,3,2)])
_check('transpose', lambda a: function.transpose(a,[0,2,1]), lambda a: a.transpose([0,1,3,2]), [(2,3,2)])
_check('expand_dims', lambda a: function.expand_dims(a,1), lambda a: numpy.expand_dims(a,2), [(2,3)])
_check('get', lambda a: function.get(a,1,1), lambda a: a[...,1,:], [(2,3,2)])
_check('takediag121', lambda a: function.takediag(a,0,2), lambda a: numeric.takediag(a,1,3), [(1,2,1)])
_check('takediag232', lambda a: function.takediag(a,0,2), lambda a: numeric.takediag(a,1,3), [(2,3,2)])
_check('takediag323', lambda a: function.takediag(a,0,2), lambda a: numeric.takediag(a,1,3), [(3,2,3)])
_check('determinant131', lambda a: function.determinant(a,(0,2)), lambda a: numpy.linalg.det(a.swapaxes(-3,-2)), [(1,3,1)])
_check('determinant232', lambda a: function.determinant(a,(0,2)), lambda a: numpy.linalg.det(a.swapaxes(-3,-2)), [(2,3,2)])
_check('determinant323', lambda a: function.determinant(a,(0,2)), lambda a: numpy.linalg.det(a.swapaxes(-3,-2)), [(3,2,3)])
_check('inverse131', lambda a: function.inverse(a,(0,2)), lambda a: numpy.linalg.inv(a.swapaxes(-3,-2)).swapaxes(-3,-2), [(1,3,1)])
_check('inverse232', lambda a: function.inverse(a,(0,2)), lambda a: numpy.linalg.inv(a.swapaxes(-3,-2)).swapaxes(-3,-2), [(2,3,2)])
_check('inverse323', lambda a: function.inverse(a,(0,2)), lambda a: numpy.linalg.inv(a.swapaxes(-3,-2)).swapaxes(-3,-2), [(3,2,3)])
_check('repeat', lambda a: function.repeat(a,3,1), lambda a: numpy.repeat(a,3,-2), [(2,1,2)])
_check('diagonalize', lambda a: function.diagonalize(a,0,2), lambda a: numeric.diagonalize(a,1,3), [(2,1,2)])
_check('multiply', function.multiply, numpy.multiply, [(3,1),(1,3)])
_check('divide', function.divide, numpy.divide, [(3,1),(1,3)])
_check('divide2', lambda a: function.asarray(a)/2, lambda a: a/2, [(3,1)])
_check('add', function.add, numpy.add, [(3,1),(1,3)])
_check('subtract', function.subtract, numpy.subtract, [(3,1),(1,3)])
_check('product2', lambda a,b: function.multiply(a,b).sum(-2), lambda a,b: (a*b).sum(-2), [(2,3,1),(1,3,2)])
_check('cross', lambda a,b: function.cross(a,b,-2), lambda a,b: numpy.cross(a,b,axis=-2), [(2,3,1),(1,3,2)])
_check('min', lambda a,b: function.min(a,b), numpy.minimum, [(3,1),(1,3)])
_check('max', lambda a,b: function.max(a,b), numpy.maximum, [(3,1),(1,3)])
_check('equal', lambda a,b: function.equal(a,b), numpy.equal, [(3,1),(1,3)])
_check('greater', lambda a,b: function.greater(a,b), numpy.greater, [(3,1),(1,3)])
_check('less', lambda a,b: function.less(a,b), numpy.less, [(3,1),(1,3)])
_check('arctan2', function.arctan2, numpy.arctan2, [(3,1),(1,3)])
_check('stack', lambda a,b: function.stack([a,b]), lambda a,b: numpy.concatenate([a[...,_,:],b[...,_,:]], axis=-2), [(3,),(3,)])
_check('concatenate1', lambda a,b: function.concatenate([a,b],axis=0), lambda a,b: numpy.concatenate([a,b], axis=-2), [(3,2),(2,2)])
_check('concatenate2', lambda a,b: function.concatenate([a,b],axis=1), lambda a,b: numpy.concatenate([a,b], axis=-1), [(3,2),(3,1)])
_check('eig', lambda a: function.eig(a,symmetric=False)[1], lambda a: numpy.linalg.eig(a)[1], [(3,3)], hasgrad=False)
_check('trignormal', lambda a: function.trignormal(a), lambda a: numpy.array([numpy.cos(a), numpy.sin(a)]).T, [()])
_check('trigtangent', lambda a: function.trigtangent(a), lambda a: numpy.array([-numpy.sin(a), numpy.cos(a)]).T, [()])
_check('mod', lambda a,b: function.mod(a,b), lambda a,b: numpy.mod(a,b), [(3,),(3,)], hasgrad=False)
_check('kronecker', lambda f: function.kronecker(f,axis=-2,length=3,pos=1), lambda a: numeric.kronecker(a,axis=-2,length=3,pos=1), [(2,3,)])
_check('mask', lambda f: function.mask(f,numpy.array([True,False,True]),axis=1), lambda a: a[:,:,[True,False,True]], [(2,3,4)])
_check('ravel', lambda f: function.ravel(f,axis=1), lambda a: a.reshape(-1,2,6), [(2,3,2)])
_check('unravel', lambda f: function.unravel(f,axis=0,shape=[2,3]), lambda a: a.reshape(-1,2,3,2), [(6,2)])


class commutativity(TestCase):

  def setUp(self):
    super().setUp()
    numpy.random.seed(0)
    self.A = function.asarray(numpy.random.uniform(size=[2,3]))
    self.B = function.asarray(numpy.random.uniform(size=[2,3]))

  def test_add(self):
    self.assertEqual(function.add(self.A, self.B), function.add(self.B, self.A))

  def test_multiply(self):
    self.assertEqual(function.multiply(self.A, self.B), function.multiply(self.B, self.A))

  def test_dot(self):
    self.assertEqual(function.dot(self.A, self.B, axes=[0]), function.dot(self.B, self.A, axes=[0]))

  def test_combined(self):
    self.assertEqual(function.add(self.A, self.B) * function.dot(self.A, self.B, axes=[0]), function.dot(self.B, self.A, axes=[0]) * function.add(self.B, self.A))


class sampled(TestCase):

  def setUp(self):
    super().setUp()
    self.domain, geom = mesh.demo()
    basis = self.domain.basis('std', degree=1)
    numpy.random.seed(0)
    self.f = basis.dot(numpy.random.uniform(size=len(basis)))
    self.f_sampled = self.domain.elem_eval(self.f, ischeme='gauss2', asfunction=True)

  def test_isarray(self):
    self.assertTrue(function.isarray(self.f_sampled))

  def test_values(self):
    diff = self.domain.integrate(self.f - self.f_sampled, ischeme='gauss2')
    self.assertEqual(diff, 0)

  def test_pointset(self):
    with self.assertRaises(function.EvaluationError):
      self.domain.integrate(self.f_sampled, ischeme='uniform2')


class namespace(TestCase):

  def test_set_scalar(self):
    ns = function.Namespace()
    ns.scalar = 1

  def test_set_array(self):
    ns = function.Namespace()
    ns.array = function.zeros([2,3])

  def test_set_scalar_expression(self):
    ns = function.Namespace()
    ns.scalar = '1'

  def test_set_array_expression(self):
    ns = function.Namespace()
    ns.foo = function.zeros([3,3])
    ns.array_ij = 'foo_ij + foo_ji'

  def test_set_readonly(self):
    ns = function.Namespace()
    with self.assertRaises(AttributeError):
      ns._foo = None

  def test_set_readonly_internal(self):
    ns = function.Namespace()
    with self.assertRaises(AttributeError):
      ns._attributes = None

  def test_del_existing(self):
    ns = function.Namespace()
    ns.foo = function.zeros([2,3])
    del ns.foo

  def test_del_readonly_internal(self):
    ns = function.Namespace()
    with self.assertRaises(AttributeError):
      del ns._attributes

  def test_del_nonexisting(self):
    ns = function.Namespace()
    with self.assertRaises(AttributeError):
      del ns.foo

  def test_get_nonexisting(self):
    ns = function.Namespace()
    with self.assertRaises(AttributeError):
      ns.foo

  def test_invalid_default_geometry_no_str(self):
    with self.assertRaises(ValueError):
      function.Namespace(default_geometry_name=None)

  def test_invalid_default_geometry_no_variable(self):
    with self.assertRaises(ValueError):
      function.Namespace(default_geometry_name='foo_bar')

  def test_default_geometry_property(self):
    ns = function.Namespace()
    ns.x = 1
    self.assertEqual(ns.default_geometry, ns.x)
    ns = function.Namespace(default_geometry_name='y')
    ns.y = 2
    self.assertEqual(ns.default_geometry, ns.y)

  def test_copy(self):
    ns = function.Namespace()
    ns.foo = function.zeros([2,3])
    ns = ns.copy_()
    self.assertTrue(hasattr(ns, 'foo'))

  def test_copy_change_geom(self):
    ns1 = function.Namespace()
    domain, ns1.y = mesh.rectilinear([2,2])
    ns1.basis = domain.basis('spline', degree=2)
    ns2 = ns1.copy_(default_geometry_name='y')
    self.assertEqual(ns2.default_geometry_name, 'y')
    self.assertEqual(ns2.eval_ni('basis_n,i'), ns2.basis.grad(ns2.y))

  def test_copy_preserve_geom(self):
    ns1 = function.Namespace(default_geometry_name='y')
    domain, ns1.y = mesh.rectilinear([2,2])
    ns1.basis = domain.basis('spline', degree=2)
    ns2 = ns1.copy_()
    self.assertEqual(ns2.default_geometry_name, 'y')
    self.assertEqual(ns2.eval_ni('basis_n,i'), ns2.basis.grad(ns2.y))

  def test_eval(self):
    ns = function.Namespace()
    ns.foo = function.zeros([3,3])
    ns.eval_ij('foo_ij + foo_ji')

  def test_matmul_0d(self):
    ns = function.Namespace()
    ns.foo = 2
    self.assertEqual('foo' @ ns, ns.foo)

  def test_matmul_1d(self):
    ns = function.Namespace()
    ns.foo = function.zeros([2])
    self.assertEqual('foo_i' @ ns, ns.foo)

  def test_matmul_2d(self):
    ns = function.Namespace()
    ns.foo = function.zeros([2, 3])
    with self.assertRaises(ValueError):
      'foo_ij' @ ns

  def test_matmul_nostr(self):
    ns = function.Namespace()
    with self.assertRaises(TypeError):
      1 @ ns

  def test_replace(self):
    ns = function.Namespace(default_geometry_name='y')
    ns.foo = function.Argument('arg', [2,3])
    ns.bar_ij = 'sin(foo_ij) + cos(2 foo_ij)'
    ns = ns | dict(arg=function.zeros([2,3]))
    self.assertEqual(ns.foo, function.zeros([2,3]))
    self.assertEqual(ns.default_geometry_name, 'y')

  def test_replace_no_mapping(self):
    ns = function.Namespace()
    ns.foo = function.Argument('arg', [2,3])
    ns.bar_ij = 'sin(foo_ij) + cos(2 foo_ij)'
    with self.assertRaises(TypeError):
      ns | 2


class eval_ast(TestCase):

  def setUp(self):
    super().setUp()
    domain, x = mesh.rectilinear([2,2])
    self.ns = function.Namespace()
    self.ns.x = x
    self.ns.altgeom_i = '<x_i, 0>_i'
    self.ns.basis = domain.basis('spline', degree=2)
    self.ns.a = 2
    self.ns.a2 = numpy.array([1,2])
    self.ns.a3 = numpy.array([1,2,3])
    self.ns.a22 = numpy.array([[1,2],[3,4]])
    self.ns.a32 = numpy.array([[1,2],[3,4],[5,6]])
    self.x = function.Argument('x',())

  def assertIdentical(self, s, f):
    self.assertEqual((s @ self.ns).simplified, f.simplified)

  def test_group(self): self.assertIdentical('(a)', self.ns.a)
  def test_arg(self): self.assertIdentical('a2_i ?x_i', function.dot(self.ns.a2, function.Argument('x', [2]), axes=[0]))
  def test_substitute(self): self.assertIdentical('?x_i^2 | ?x_i = a2_i', self.ns.a2**2)
  def test_call(self): self.assertIdentical('sin(a)', function.sin(self.ns.a))
  def test_eye(self): self.assertIdentical('δ_ij a2_i', function.dot(function.eye(2), self.ns.a2, axes=[0]))
  def test_normal(self): self.assertIdentical('n_i', self.ns.x.normal())
  def test_getitem(self): self.assertIdentical('a2_0', self.ns.a2[0])
  def test_trace(self): self.assertIdentical('a22_ii', function.trace(self.ns.a22, 0, 1))
  def test_sum(self): self.assertIdentical('a2_i a2_i', function.sum(self.ns.a2 * self.ns.a2, axis=0))
  def test_concatenate(self): self.assertIdentical('<a, a2_i>_i', function.concatenate([self.ns.a[None],self.ns.a2], axis=0))
  def test_grad(self): self.assertIdentical('basis_n,0', self.ns.basis.grad(self.ns.x)[:,0])
  def test_surfgrad(self): self.assertIdentical('basis_n;altgeom_0', function.grad(self.ns.basis, self.ns.altgeom, len(self.ns.altgeom)-1)[:,0])
  def test_derivative(self): self.assertIdentical('exp(?x)_,?x', function.derivative(function.exp(self.x), self.x))
  def test_append_axis(self): self.assertIdentical('a a2_i', self.ns.a[None]*self.ns.a2)
  def test_transpose(self): self.assertIdentical('a22_ij a22_ji', function.dot(self.ns.a22, self.ns.a22.T, axes=[0,1]))
  def test_jump(self): self.assertIdentical('[a]', function.jump(self.ns.a))
  def test_mean(self): self.assertIdentical('{a}', function.mean(self.ns.a))
  def test_neg(self): self.assertIdentical('-a', -self.ns.a)
  def test_add(self): self.assertIdentical('a + ?x', self.ns.a + self.x)
  def test_sub(self): self.assertIdentical('a - ?x', self.ns.a - self.x)
  def test_mul(self): self.assertIdentical('a ?x', self.ns.a * self.x)
  def test_truediv(self): self.assertIdentical('a / ?x', self.ns.a / self.x)
  def test_pow(self): self.assertIdentical('a^2', self.ns.a**2)

  def test_unknown_opcode(self):
    with self.assertRaises(ValueError):
      function._eval_ast(('invalid-opcode',), {})
