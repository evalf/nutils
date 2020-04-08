import itertools, pickle, warnings as _builtin_warnings
from nutils import *
from nutils.testing import *


@parametrize
class check(TestCase):

  def setUp(self):
    super().setUp()
    domain, param = mesh.rectilinear([1]*self.ndim, periodic=[0])
    if self.ndim == 1:
      self.geom = param**2
      poly = numpy.array([[1,-2,1],[0,2,-2],[0,0,1]]) # 2nd order bernstein
    elif self.ndim == 2:
      r, theta = param
      self.geom = r * function.stack([function.cos(theta), function.sin(theta)])
      poly = numeric.poly_outer_product([[1,-2,1],[0,2,-2],[0,0,1]], [[1,-1],[0,1]]) # 2nd x 1st order bernstein
    else:
      raise Exception('invalid ndim {!r}'.format(self.ndim))
    numpy.random.seed(0)
    self.args = [function.Guard(function.Polyval(numeric.dot(numpy.random.uniform(size=shape+poly.shape[:1], low=self.low, high=self.high), poly), function.rootcoords(self.ndim))) for shape in self.shapes]
    if self.pass_geom:
        self.args += [self.geom]
    self.sample = domain.sample('uniform', 2)
    self.ifacesmp = domain.interfaces.sample('uniform', 2)
    self.n_op_argsfun = self.n_op(*self.sample.eval(self.args))
    self.op_args = self.op(*self.args)
    self.shapearg = numpy.random.uniform(size=self.op_args.shape, low=self.low, high=self.high)
    self.pairs = [(i, j) for i in range(self.op_args.ndim-1) for j in range(i+1, self.op_args.ndim) if self.op_args.shape[i] == self.op_args.shape[j]]

  def setUpContext(self, stack):
    super().setUpContext(stack)
    stack.enter_context(_builtin_warnings.catch_warnings())
    _builtin_warnings.simplefilter('ignore', category=function.ExpensiveEvaluationWarning)

  def assertArrayAlmostEqual(self, actual, desired, decimal):
    if actual.shape[1:] != desired.shape[1:] or actual.shape[0] != 1 and desired.shape[0] != 1 and actual.shape[0] != desired.shape[0]:
      self.fail('shapes of actual {} and desired {} are incompatible.'.format(actual.shape, desired.shape))
    error = actual - desired if not actual.dtype.kind == desired.dtype.kind == 'b' else actual ^ desired
    approx = error.dtype.kind in 'fc'
    indices = (numpy.greater_equal(abs(error), 1.5 * 10**-decimal) if approx else error).nonzero()
    if not len(indices[0]):
      return
    lines = ['arrays are not equal']
    if approx:
      lines.append(' up to {} decimals'.format(decimal))
    lines.append(' in {}/{} entries:'.format(len(indices[0]), error.size))
    lines.extend('\n  {} actual={} desired={} difference={}'.format(index, actual[index], desired[index], error[index]) for index in zip(*indices))
    self.fail(''.join(lines))

  def assertFunctionAlmostEqual(self, actual, desired, decimal):
    evalargs = dict(_transforms=[trans[0] for trans in self.sample.transforms], _points=self.sample.points[0].coords)
    with self.subTest('vanilla'):
      self.assertArrayAlmostEqual(actual.eval(**evalargs), desired, decimal)
    with self.subTest('simplified'):
      self.assertArrayAlmostEqual(actual.simplified.eval(**evalargs), desired, decimal)
    with self.subTest('optimized'):
      self.assertArrayAlmostEqual(actual.simplified.optimized_for_numpy.eval(**evalargs), desired, decimal)
    with self.subTest('sample'):
      self.assertArrayAlmostEqual(self.sample.eval(actual), desired, decimal)

  def test_evalconst(self):
    constargs = [numpy.random.uniform(size=shape) for shape in self.shapes]
    if self.pass_geom:
      constargs += [numpy.random.uniform(size=self.geom.shape)]
    self.assertFunctionAlmostEqual(decimal=15,
      desired=self.n_op(*[constarg[_] for constarg in constargs]),
      actual=self.op(*constargs))

  def test_eval(self):
    self.assertFunctionAlmostEqual(decimal=15,
      actual=self.op_args,
      desired=self.n_op_argsfun)

  def test_getitem(self):
    for idim in range(self.op_args.ndim):
      for item in range(self.op_args.shape[idim]):
        s = (Ellipsis,) + (slice(None),)*idim + (item,) + (slice(None),)*(self.op_args.ndim-idim-1)
        self.assertFunctionAlmostEqual(decimal=15,
          desired=self.n_op_argsfun[s],
          actual=self.op_args[s])

  def test_transpose(self):
    trans = numpy.arange(self.op_args.ndim,0,-1) % self.op_args.ndim
    self.assertFunctionAlmostEqual(decimal=15,
      desired=numpy.transpose(self.n_op_argsfun, [0]+list(trans+1)),
      actual=function.transpose(self.op_args, trans))

  def test_insertaxis(self):
    for axis in range(self.op_args.ndim+1):
      with self.subTest(axis=axis):
        self.assertFunctionAlmostEqual(decimal=15,
          desired=numpy.repeat(numpy.expand_dims(self.n_op_argsfun, axis+1), 2, axis+1),
          actual=function.InsertAxis(self.op_args, axis=axis, length=2))

  def test_takediag(self):
    for ax1, ax2 in self.pairs:
      self.assertFunctionAlmostEqual(decimal=15,
        desired=numeric.takediag(self.n_op_argsfun, ax1+1, ax2+1),
        actual=function.takediag(self.op_args, ax1, ax2))

  def test_eig(self):
    if self.op_args.dtype == float:
      for ax1, ax2 in self.pairs:
        A = self.sample.eval(self.op_args)
        L, V = self.sample.eval(list(function.eig(self.op_args, axes=(ax1,ax2))))
        self.assertArrayAlmostEqual(decimal=11,
          actual=(numpy.expand_dims(V,ax2+1) * numpy.expand_dims(L,ax2+2).swapaxes(ax1+1,ax2+2)).sum(ax2+2),
          desired=(numpy.expand_dims(A,ax2+1) * numpy.expand_dims(V,ax2+2).swapaxes(ax1+1,ax2+2)).sum(ax2+2))

  def test_inv(self):
    for ax1, ax2 in self.pairs:
      trans = [i for i in range(self.n_op_argsfun.ndim) if i not in (ax1+1,ax2+1)] + [ax1+1,ax2+1]
      invtrans = list(map(trans.index, range(len(trans))))
      self.assertFunctionAlmostEqual(decimal=14,
        desired=numeric.inv(self.n_op_argsfun.transpose(trans)).transpose(invtrans),
        actual=function.inverse(self.op_args, axes=(ax1,ax2)))

  def test_determinant(self):
    for ax1, ax2 in self.pairs:
      self.assertFunctionAlmostEqual(decimal=11,
        desired=numpy.linalg.det(self.n_op_argsfun.transpose([i for i in range(self.n_op_argsfun.ndim) if i not in (ax1+1,ax2+1)] + [ax1+1,ax2+1])),
        actual=function.determinant(self.op_args, axes=(ax1,ax2)))

  def test_take(self):
    indices = [0,-1]
    for iax, sh in enumerate(self.op_args.shape):
      if sh >= 2:
        self.assertFunctionAlmostEqual(decimal=15,
          desired=numpy.take(self.n_op_argsfun, indices, axis=iax+1),
          actual=function.take(self.op_args, indices, axis=iax))

  def test_take_nomask(self):
    for iax, sh in enumerate(self.op_args.shape):
      if sh >= 2:
        indices = [0,sh-1]
        self.assertFunctionAlmostEqual(decimal=15,
          desired=numpy.take(self.n_op_argsfun, indices, axis=iax+1),
          actual=function.take(self.op_args, function.Guard(function.asarray(indices)), axis=iax))

  def test_take_reversed(self):
    indices = [-1,0]
    for iax, sh in enumerate(self.op_args.shape):
      if sh >= 2:
        self.assertFunctionAlmostEqual(decimal=15,
          desired=numpy.take(self.n_op_argsfun, indices, axis=iax+1),
          actual=function.take(self.op_args, indices, axis=iax))

  def test_inflate(self):
    for iax, sh in enumerate(self.op_args.shape):
      dofmap = function.Constant(numpy.arange(sh) * 2)
      desired = numpy.zeros(self.n_op_argsfun.shape[:iax+1] + (sh*2-1,) + self.n_op_argsfun.shape[iax+2:], dtype=self.n_op_argsfun.dtype)
      desired[(slice(None),)*(iax+1)+(slice(None,None,2),)] = self.n_op_argsfun
      self.assertFunctionAlmostEqual(decimal=15,
        desired=desired,
        actual=function.Inflate(self.op_args, dofmap=dofmap, length=sh*2-1, axis=iax))

  def test_diagonalize(self):
    for axis in range(self.op_args.ndim):
      for newaxis in range(axis+1, self.op_args.ndim+1):
        self.assertFunctionAlmostEqual(decimal=15,
          desired=numeric.diagonalize(self.n_op_argsfun, axis+1, newaxis+1),
          actual=function.diagonalize(self.op_args, axis, newaxis))

  def test_product(self):
    for iax in range(self.op_args.ndim):
      self.assertFunctionAlmostEqual(decimal=15,
        desired=numpy.product(self.n_op_argsfun, axis=iax+1),
        actual=function.product(self.op_args, axis=iax))

  def test_concatenate(self):
    for idim in range(self.op_args.ndim):
      self.assertFunctionAlmostEqual(decimal=15,
        desired=numpy.concatenate([self.n_op_argsfun, self.shapearg[_].repeat(len(self.n_op_argsfun),0)], axis=idim+1),
        actual=function.concatenate([self.op_args, self.shapearg], axis=idim))

  def test_getslice(self):
    for idim in range(self.op_args.ndim):
      if self.op_args.shape[idim] == 1:
        continue
      s = (Ellipsis,) + (slice(None),)*idim + (slice(0,self.op_args.shape[idim]-1),) + (slice(None),)*(self.op_args.ndim-idim-1)
      self.assertFunctionAlmostEqual(decimal=15,
        desired=self.n_op_argsfun[s],
        actual=self.op_args[s])

  def test_sumaxis(self):
    for idim in range(self.op_args.ndim):
      self.assertFunctionAlmostEqual(decimal=15,
        desired=self.n_op_argsfun.sum(1+idim),
        actual=self.op_args.sum(idim))

  def test_add(self):
    self.assertFunctionAlmostEqual(decimal=15,
      desired=self.n_op_argsfun + self.shapearg,
      actual=(self.op_args + self.shapearg))

  def test_multiply(self):
    self.assertFunctionAlmostEqual(decimal=15,
      desired=self.n_op_argsfun * self.shapearg,
      actual=(self.op_args * self.shapearg))

  def test_dot(self):
    for iax in range(self.op_args.ndim):
      self.assertFunctionAlmostEqual(decimal=14,
        desired=numeric.contract(self.n_op_argsfun, self.shapearg, axis=iax+1),
        actual=function.dot(self.op_args, self.shapearg, axes=iax))

  def test_pointwise(self):
    self.assertFunctionAlmostEqual(decimal=15,
      desired=numpy.sin(self.n_op_argsfun).astype(float), # "astype" necessary for boolean operations (float16->float64)
      actual=function.sin(self.op_args))

  def test_cross(self):
    triaxes = [iax for iax, sh in enumerate(self.op_args.shape) if sh == 3]
    if triaxes:
      for iax in triaxes:
        self.assertFunctionAlmostEqual(decimal=15,
          desired=numpy.cross(self.n_op_argsfun, self.shapearg[_], axis=iax+1),
          actual=function.cross(self.op_args, self.shapearg, axis=iax))

  def test_power(self):
    self.assertFunctionAlmostEqual(decimal=13,
      desired=self.n_op_argsfun**3,
      actual=(self.op_args**3))

  def test_sign(self):
    if self.n_op_argsfun.dtype.kind != 'b':
      self.assertFunctionAlmostEqual(decimal=15,
        desired=numpy.sign(self.n_op_argsfun),
        actual=function.sign(self.op_args))

  def test_mask(self):
    for idim in range(self.op_args.ndim):
      if self.op_args.shape[idim] <= 1:
        continue
      mask = numpy.ones(self.op_args.shape[idim], dtype=bool)
      mask[0] = False
      if self.op_args.shape[idim] > 2:
        mask[-1] = False
      self.assertFunctionAlmostEqual(decimal=15,
        desired=self.n_op_argsfun[(slice(None,),)*(idim+1)+(mask,)],
        actual=function.mask(self.op_args, mask, axis=idim))

  def test_ravel(self):
    for idim in range(self.op_args.ndim-1):
      self.assertFunctionAlmostEqual(decimal=15,
        desired=self.n_op_argsfun.reshape(self.n_op_argsfun.shape[:idim+1]+(-1,)+self.n_op_argsfun.shape[idim+3:]),
        actual=function.ravel(self.op_args, axis=idim))

  def test_unravel(self):
    for idim in range(self.op_args.ndim):
      length = self.n_op_argsfun.shape[idim+1]
      unravelshape = (length//3,3) if (length%3==0) else (length//2,2) if (length%2==0) else (length,1)
      self.assertFunctionAlmostEqual(decimal=15,
        desired=self.n_op_argsfun.reshape(self.n_op_argsfun.shape[:idim+1]+unravelshape+self.n_op_argsfun.shape[idim+2:]),
        actual=function.unravel(self.op_args, axis=idim, shape=unravelshape))

  def test_kronecker(self):
    for idim in range(self.op_args.ndim+1):
      desired = numpy.zeros(self.n_op_argsfun.shape[:idim+1]+(3,)+self.n_op_argsfun.shape[idim+1:], dtype=self.n_op_argsfun.dtype)
      desired[(slice(None),)*(idim+1)+(1,)] = self.n_op_argsfun
      self.assertFunctionAlmostEqual(decimal=15,
        desired=desired,
        actual=function.kronecker(self.op_args, axis=idim, pos=1, length=3))

  def test_edit(self):
    def check_identity(arg):
      if function.isevaluable(arg):
        newarg = arg.edit(check_identity)
        self.assertEqual(arg, newarg)
      return arg
    check_identity(self.op_args)

  def test_opposite(self):
    self.assertArrayAlmostEqual(decimal=14,
      desired=self.n_op(*self.ifacesmp.eval([function.opposite(arg) for arg in self.args])),
      actual=self.ifacesmp.eval(function.opposite(self.op_args)))

  def find(self, target, xi0):
    elemtrans, = self.sample.transforms[0]
    ndim, = self.geom.shape
    J = function.localgradient(self.geom, ndim)
    Jinv = function.inverse(J).prepare_eval()
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
      err = target - self.geom.prepare_eval().eval(_transforms=[elemtrans], _points=xi)
      if numpy.less(numpy.abs(err), 1e-12).all():
        countdown -= 1
      dxi_root = (Jinv.eval(_transforms=[elemtrans], _points=xi) * err[...,_,:]).sum(-1)
      #xi = xi + numpy.dot(dxi_root, self.elem.inv_root_transform.T)
      xi = xi + dxi_root
      iiter += 1
      self.assertLess(iiter, 100, 'failed to converge in 100 iterations')
    return xi.reshape(xi0.shape)

  @parametrize.enable_if(lambda hasgrad, **kwargs: hasgrad)
  def test_localgradient(self):
    elemtrans, = self.sample.transforms[0]
    points = self.sample.points[0].coords
    argsfun = function.Tuple(self.args).prepare_eval()
    exact = self.sample.eval(function.localgradient(self.op_args, ndims=self.ndim))
    D = numpy.array([-.5,.5])[:,_,_] * numpy.eye(self.ndim)
    good = False
    eps = 1e-5
    while not numpy.all(good):
      fdpoints = points[_,_,:,:] + D[:,:,_,:] * eps
      tmp = self.n_op(*argsfun.eval(_transforms=[elemtrans], _points=fdpoints.reshape(-1,fdpoints.shape[-1])))
      if len(tmp) == 1 or tmp.dtype.kind in 'bi' or self.zerograd:
        error = exact
      else:
        fdgrad, = numpy.diff(tmp.reshape(fdpoints.shape[:-1] + tmp.shape[1:]), axis=0) / eps
        error = exact - fdgrad.transpose(numpy.roll(numpy.arange(fdgrad.ndim),-1))
      good |= numpy.less(abs(error / exact), 1e-8)
      good |= numpy.less(abs(error), 1e-14)
      eps *= .8
      if eps < 1e-10:
        self.fail('local gradient failed to reach tolerance ({}/{})'.format((~good).sum(), good.size))

  @parametrize.enable_if(lambda hasgrad, **kwargs: hasgrad)
  def test_jacobian(self):
    eps = 1e-8
    for iarg, shape in enumerate(self.shapes):
      x0 = numpy.random.uniform(size=shape, low=self.low, high=self.high)
      dx = numpy.random.normal(size=shape) * eps
      x = function.Argument('x', shape)
      f = self.op(*self.args[:iarg]+[x]+self.args[iarg+1:])
      fx0, Jx0 = self.sample.eval([f, function.derivative(f, x)], x=x0)
      fx1 = self.sample.eval(f, x=x0+dx)
      fx1approx = fx0 + numeric.contract(Jx0, dx, range(fx0.ndim, Jx0.ndim))
      self.assertArrayAlmostEqual(fx1approx, fx1, decimal=12)

  @parametrize.enable_if(lambda hasgrad, **kwargs: hasgrad)
  def test_gradient(self):
    elemtrans, = self.sample.transforms[0]
    points = self.sample.points[0].coords
    argsfun = function.Tuple(self.args).prepare_eval()
    exact = self.sample.eval(self.op_args.grad(self.geom))
    fddeltas = numpy.array([1,2,3])
    fdfactors = numpy.linalg.solve(2*fddeltas**numpy.arange(1,1+2*len(fddeltas),2)[:,None], [1]+[0]*(len(fddeltas)-1))
    fdfactors = numpy.concatenate([-fdfactors[::-1], fdfactors])
    D = numpy.concatenate([-fddeltas[::-1], fddeltas])[:,_,_] * numpy.eye(self.geom.shape[-1])
    good = False
    eps = 1e-4
    while not numpy.all(good):
      fdpoints = self.find(self.sample.eval(self.geom)[_,_,:,:] + D[:,:,_,:] * eps, points[_,_,:,:])
      tmp = self.n_op(*argsfun.eval(_transforms=[elemtrans], _points=fdpoints.reshape(-1,fdpoints.shape[-1])))
      if len(tmp) == 1 or tmp.dtype.kind in 'bi' or self.zerograd:
        error = exact
      else:
        fdgrad = numpy.tensordot(fdfactors, tmp.reshape(fdpoints.shape[:-1] + tmp.shape[1:]), [[0]]*2) / eps
        error = exact - fdgrad.transpose(numpy.roll(numpy.arange(fdgrad.ndim),-1))
      good |= numpy.less(abs(error / exact), 1e-9)
      good |= numpy.less(abs(error), 1e-14)
      eps *= .8
      if eps < 1e-10:
        self.fail('gradient failed to reach tolerance ({}/{})'.format((~good).sum(), good.size))

  @parametrize.enable_if(lambda hasgrad, **kwargs: hasgrad)
  def test_doublegradient(self):
    elemtrans, = self.sample.transforms[0]
    points = self.sample.points[0].coords
    argsfun = function.Tuple(self.args).prepare_eval()
    exact = self.sample.eval(self.op_args.grad(self.geom).grad(self.geom))
    fddeltas = numpy.array([1,2,3])
    fdfactors = numpy.linalg.solve(2*fddeltas**numpy.arange(1,1+2*len(fddeltas),2)[:,None], [1]+[0]*(len(fddeltas)-1))
    fdfactors = numpy.concatenate([-fdfactors[::-1], fdfactors])
    D = numpy.concatenate([-fddeltas[::-1], fddeltas])[:,_,_] * numpy.eye(self.geom.shape[-1])
    DD = D[:,_,:,_,:] + D[_,:,_,:,:]
    good = False
    eps = 1e-4
    while not numpy.all(good):
      fdpoints = self.find(self.sample.eval(self.geom)[_,_,_,_,:,:] + DD[:,:,:,:,_,:] * eps, points[_,_,_,_,:,:])
      tmp = self.n_op(*argsfun.eval(_transforms=[elemtrans], _points=fdpoints.reshape(-1,fdpoints.shape[-1])))
      if len(tmp) == 1 or tmp.dtype.kind in 'bi' or self.zerograd:
        error = exact
      else:
        fddgrad = numpy.tensordot(numpy.outer(fdfactors, fdfactors), tmp.reshape(fdpoints.shape[:-1] + tmp.shape[1:]), [[0,1]]*2) / eps**2
        error = exact - fddgrad.transpose(numpy.roll(numpy.arange(fddgrad.ndim),-2))
      good |= numpy.less(abs(error / exact), 1e-4)
      good |= numpy.less(abs(error), 1e-14)
      eps *= .8
      if eps < 1e-10:
        self.fail('double gradient failed to reach tolerance ({}/{})'.format((~good).sum(), good.size))

def _check(name, op, n_op, shapes, hasgrad=True, zerograd=False, pass_geom=False, ndim=2, low=-1, high=1):
  check(name, op=op, n_op=n_op, shapes=shapes, hasgrad=hasgrad, zerograd=zerograd, pass_geom=pass_geom, ndim=ndim, low=low, high=high)

_check('identity', lambda f: function.asarray(f), lambda a: a, [(2,4,2)])
_check('const', lambda f: function.asarray(numpy.arange(16, dtype=float).reshape(2,4,2)), lambda a: numpy.arange(16, dtype=float).reshape(1,2,4,2), [(2,4,2)])
_check('zeros', lambda f: function.zeros([4,3,4]), lambda a: numpy.zeros([1,4,3,4]), [(4,3,4)])
_check('ones', lambda f: function.ones([4,3,4]), lambda a: numpy.ones([1,4,3,4]), [(4,3,4)])
_check('range', lambda f: function.Range(4, offset=2), lambda a: numpy.arange(2,6)[numpy.newaxis], [(4,)])
_check('sin', function.sin, numpy.sin, [(4,)])
_check('cos', function.cos, numpy.cos, [(4,)])
_check('tan', function.tan, numpy.tan, [(4,)])
_check('sqrt', function.sqrt, numpy.sqrt, [(4,)], low=0)
_check('log', function.ln, numpy.log, [(4,)], low=0)
_check('log2', function.log2, numpy.log2, [(4,)], low=0)
_check('log10', function.log10, numpy.log10, [(4,)], low=0)
_check('exp', function.exp, numpy.exp, [(4,)])
_check('arctanh', function.arctanh, numpy.arctanh, [(4,)])
_check('tanh', function.tanh, numpy.tanh, [(4,)])
_check('cosh', function.cosh, numpy.cosh, [(4,)])
_check('sinh', function.sinh, numpy.sinh, [(4,)])
_check('abs', function.abs, numpy.abs, [(4,)])
_check('sign', function.sign, numpy.sign, [(4,4)], zerograd=True)
_check('power', function.power, numpy.power, [(4,1),(1,4)], low=0)
_check('negative', function.negative, numpy.negative, [(4,)])
_check('reciprocal', function.reciprocal, numpy.reciprocal, [(4,)], low=-2, high=-1)
_check('arcsin', function.arcsin, numpy.arcsin, [(4,)])
_check('arccos', function.arccos, numpy.arccos, [(4,)])
_check('arctan', function.arctan, numpy.arctan, [(4,)])
_check('ln', function.ln, numpy.log, [(4,)], low=0)
_check('product', lambda a: function.product(a,1), lambda a: numpy.product(a,-2), [(4,3,4)])
_check('norm2', lambda a: function.norm2(a,1), lambda a: (a**2).sum(-2)**.5, [(4,3,4)])
_check('norm2scalar', lambda a: function.norm2(a,1), lambda a: abs(a.sum(-2)), [(4,1,4)])
_check('sum', lambda a: function.sum(a,1), lambda a: a.sum(-2), [(4,3,4)])
_check('transpose1', lambda a: function.transpose(a,[0,2,1]), lambda a: a.transpose([0,1,3,2]), [(4,4,4)], low=0, high=20)
_check('transpose2', lambda a: function.transpose(a,[1,2,0]), lambda a: a.transpose([0,2,3,1]), [(4,4,4)])
_check('expand_dims', lambda a: function.expand_dims(a,1), lambda a: numpy.expand_dims(a,2), [(2,4)])
_check('get', lambda a: function.get(a,1,1), lambda a: a[...,1,:], [(4,3,4)])
_check('getvar', lambda a: function.get(function.Constant([[1,2],[3,4]]),1,function.Int(a)%2), lambda a: numpy.array([[1,2],[3,4]])[:,a.astype(int)%2].T, [()])
_check('takediag141', lambda a: function.takediag(a,0,2), lambda a: numeric.takediag(a,1,3), [(1,4,1)])
_check('takediag434', lambda a: function.takediag(a,0,2), lambda a: numeric.takediag(a,1,3), [(4,3,4)])
_check('takediag343', lambda a: function.takediag(a,0,2), lambda a: numeric.takediag(a,1,3), [(3,4,3)])
_check('determinant141', lambda a: function.determinant(a,(0,2)), lambda a: numpy.linalg.det(a.swapaxes(-3,-2)), [(1,4,1)])
_check('determinant434', lambda a: function.determinant(a,(0,2)), lambda a: numpy.linalg.det(a.swapaxes(-3,-2)), [(4,3,4)])
_check('determinant4433', lambda a: function.determinant(a,(2,3)), lambda a: numpy.linalg.det(a), [(4,4,3,3)])
_check('determinant200', lambda a: function.determinant(a,(1,2)), lambda a: numpy.linalg.det(a) if a.shape[-1] else numpy.ones(a.shape[:-2], float), [(2,0,0)], zerograd=True)
_check('inverse141', lambda a: function.inverse(a+function.eye(1)[:,None],(0,2)), lambda a: numpy.linalg.inv(a.swapaxes(-3,-2)+numpy.eye(1)).swapaxes(-3,-2), [(1,4,1)])
_check('inverse434', lambda a: function.inverse(a+function.eye(4)[:,None],(0,2)), lambda a: numpy.linalg.inv(a.swapaxes(-3,-2)+numpy.eye(4)).swapaxes(-3,-2), [(4,3,4)])
_check('inverse4422', lambda a: function.inverse(a+function.eye(2)), lambda a: numpy.linalg.inv(a+numpy.eye(2)), [(4,4,2,2)])
_check('repeat', lambda a: function.repeat(a,3,1), lambda a: numpy.repeat(a,3,-2), [(4,1,4)])
_check('diagonalize', lambda a: function.diagonalize(a,1,3), lambda a: numeric.diagonalize(a,2,4), [(4,4,4,4,4)])
_check('multiply', function.multiply, numpy.multiply, [(4,1),(4,4)])
_check('dot', lambda a,b: function.dot(a,b,axes=1), lambda a,b: (a*b).sum(2), [(4,2,4),(4,2,4)])
_check('divide', function.divide, lambda a, b: a * b**-1, [(4,4),(1,4)], low=-2, high=-1)
_check('divide2', lambda a: function.asarray(a)/2, lambda a: a/2, [(4,1)])
_check('add', function.add, numpy.add, [(4,1),(1,4)])
_check('subtract', function.subtract, numpy.subtract, [(4,1),(1,4)])
_check('dot2', lambda a,b: function.multiply(a,b).sum(-2), lambda a,b: (a*b).sum(-2), [(4,2,4),(1,2,4)])
_check('cross', lambda a,b: function.cross(a,b,-2), lambda a,b: numpy.cross(a,b,axis=-2), [(4,3,1),(4,3,4)])
_check('min', lambda a,b: function.min(a,b), numpy.minimum, [(4,1),(1,4)])
_check('max', lambda a,b: function.max(a,b), numpy.maximum, [(4,1),(1,4)])
_check('equal', lambda a,b: function.equal(a,b), numpy.equal, [(4,1),(1,4)])
_check('greater', lambda a,b: function.greater(a,b), numpy.greater, [(4,1),(1,4)])
_check('less', lambda a,b: function.less(a,b), numpy.less, [(4,1),(1,4)])
_check('arctan2', function.arctan2, numpy.arctan2, [(4,1),(1,4)])
_check('stack', lambda a,b: function.stack([a,b]), lambda a,b: numpy.concatenate([a[...,_,:],b[...,_,:]], axis=-2), [(4,),(4,)])
_check('concatenate1', lambda a,b: function.concatenate([a,b],axis=0), lambda a,b: numpy.concatenate([a,b], axis=-2), [(4,6),(2,6)])
_check('concatenate2', lambda a,b: function.concatenate([a,b],axis=1), lambda a,b: numpy.concatenate([a,b], axis=-1), [(4,3),(4,1)])
_check('eig', lambda a: function.eig(a+a.T,symmetric=True)[1], lambda a: numpy.linalg.eigh(a+a.swapaxes(1,2))[1], [(4,4)], hasgrad=False)
_check('trignormal', lambda a: function.trignormal(a), lambda a: numpy.array([numpy.cos(a), numpy.sin(a)]).T, [()])
_check('trigtangent', lambda a: function.trigtangent(a), lambda a: numpy.array([-numpy.sin(a), numpy.cos(a)]).T, [()])
_check('mod', lambda a,b: function.mod(a,b), lambda a,b: numpy.mod(a,b), [(4,),(4,)], hasgrad=False)
_check('kronecker', lambda f: function.kronecker(f,axis=2,length=4,pos=1), lambda a: numeric.kronecker(a,axis=3,length=4,pos=1), [(4,4,4,4)])
_check('mask', lambda f: function.mask(f,numpy.array([True,False,True,False,True,False,True]),axis=1), lambda a: a[:,:,::2], [(4,7,4)])
_check('ravel', lambda f: function.ravel(f,axis=1), lambda a: a.reshape(-1,4,4,4,4), [(4,2,2,4,4)])
_check('unravel', lambda f: function.unravel(f,axis=1,shape=[2,2]), lambda a: a.reshape(-1,4,2,2,4,4), [(4,4,4,4)])
_check('inflate', lambda f: function.Inflate(f,dofmap=function.Guard([0,3]),length=4,axis=1), lambda a: numpy.concatenate([a[:,:,:1], numpy.zeros_like(a), a[:,:,1:]], axis=2), [(4,2,4)])
_check('inflate-constant', lambda f: function.Inflate(f,dofmap=[0,3],length=4,axis=1), lambda a: numpy.concatenate([a[:,:,:1], numpy.zeros_like(a), a[:,:,1:]], axis=2), [(4,2,4)])
_check('vectorize', lambda a,b: function.vectorize([a, b]), lambda a,b: numpy.concatenate([numpy.stack([a, numpy.zeros_like(a)], axis=2), numpy.stack([numpy.zeros_like(b), b], axis=2)], axis=1), [(3,),(5,)])

_polyval_mask = lambda shape, ndim: 1 if ndim == 0 else numpy.array([sum(i[-ndim:]) < shape[-1] for i in numpy.ndindex(shape)], dtype=int).reshape(shape)
_polyval_desired = lambda c, x: sum(c[(...,*i)]*(x[(slice(None),*[None]*(c.ndim-1-x.shape[1]))]**i).prod(-1) for i in itertools.product(*[range(c.shape[-1])]*x.shape[1]) if sum(i) < c.shape[-1])
_check('polyval_1d_p0', lambda c, x: function.Polyval(c*_polyval_mask(c.shape,2), function.asarray(x)), _polyval_desired, [(1,)], pass_geom=True, ndim=1)
_check('polyval_1d_p1', lambda c, x: function.Polyval(c*_polyval_mask(c.shape,2), function.asarray(x)), _polyval_desired, [(2,)], pass_geom=True, ndim=1)
_check('polyval_1d_p2', lambda c, x: function.Polyval(c*_polyval_mask(c.shape,2), function.asarray(x)), _polyval_desired, [(3,)], pass_geom=True, ndim=1)
_check('polyval_2d_p0', lambda c, x: function.Polyval(c*_polyval_mask(c.shape,2), function.asarray(x)), _polyval_desired, [(1,1)], pass_geom=True, ndim=2)
_check('polyval_2d_p1', lambda c, x: function.Polyval(c*_polyval_mask(c.shape,2), function.asarray(x)), _polyval_desired, [(2,2)], pass_geom=True, ndim=2)
_check('polyval_2d_p2', lambda c, x: function.Polyval(c*_polyval_mask(c.shape,2), function.asarray(x)), _polyval_desired, [(3,3)], pass_geom=True, ndim=2)
_check('polyval_2d_p1_23', lambda c, x: function.Polyval(c*_polyval_mask(c.shape,2), function.asarray(x)), _polyval_desired, [(2,3,2,2)], pass_geom=True, ndim=2)


class blocks(TestCase):

  def test_multiply_equal(self):
    ((i,), f), = function.multiply(function.Inflate([1,2], dofmap=[0,2], length=3, axis=0), function.Inflate([3,4], dofmap=[0,2], length=3, axis=0)).blocks
    self.assertEqual(i, function.asarray([0,2]))
    self.assertAllEqual(f.eval(), [1*3,2*4])

  def test_multiply_embedded(self):
    ((i,), f), = function.multiply([1,2,3], function.Inflate([4,5], dofmap=[0,2], length=3, axis=0)).blocks
    self.assertEqual(i, function.asarray([0,2]))
    self.assertAllEqual(f.eval(), [1*4,3*5])

  def test_multiply_overlapping(self):
    ((i,), f), = function.multiply(function.Inflate([1,2], dofmap=[0,1], length=3, axis=0), function.Inflate([3,4], dofmap=[1,2], length=3, axis=0)).blocks
    self.assertEqual(i, function.asarray([1]))
    self.assertAllEqual(f.eval(), 2*3)

  def test_multiply_disjoint(self):
    blocks = function.multiply(function.Inflate([1,2], dofmap=[0,2], length=4, axis=0), function.Inflate([3,4], dofmap=[1,3], length=4, axis=0)).blocks
    self.assertEqual(blocks, ())

  def test_multiply_fallback(self):
    ((i,), f), = function.multiply(function.Inflate([1,2], dofmap=function.Guard([0,1]), length=3, axis=0), function.Inflate([3,4], dofmap=function.Guard([1,2]), length=3, axis=0)).blocks
    self.assertEqual(i, function.Range(3))
    self.assertAllEqual(f.eval(), [0,2*3,0])

  def test_takediag(self):
    ((i,), f), = function.takediag([[1,2,3],[4,5,6],[7,8,9]]).blocks
    self.assertEqual(i, function.Range(3))
    self.assertAllEqual(f.eval(), [1,5,9])

  def test_takediag_embedded_axis(self):
    ((i,), f), = function.takediag(function.Inflate([[1,2,3],[4,5,6]], dofmap=[0,2], length=3, axis=0)).blocks
    self.assertEqual(i, function.asarray([0,2]))
    self.assertAllEqual(f.eval(), [1,6])

  def test_takediag_embedded_rmaxis(self):
    ((i,), f), = function.takediag(function.Inflate([[1,2],[3,4],[5,6]], dofmap=[0,2], length=3, axis=1)).blocks
    self.assertEqual(i, function.asarray([0,2]))
    self.assertAllEqual(f.eval(), [1,6])

  def test_takediag_overlapping(self):
    ((i,), f), = function.takediag(function.Inflate(function.Inflate([[1,2],[3,4]], dofmap=[0,1], length=3, axis=0), dofmap=[1,2], length=3, axis=1)).blocks
    self.assertEqual(i, function.asarray([1]))
    self.assertAllEqual(f.eval(), 3)

  def test_takediag_disjoint(self):
    blocks = function.takediag(function.Inflate(function.Inflate([[1,2],[3,4]], dofmap=[0,2], length=4, axis=0), dofmap=[1,3], length=4, axis=1)).blocks
    self.assertEqual(blocks, ())

  def test_takediag_fallback(self):
    ((i,), f), = function.takediag(function.Inflate(function.Inflate([[1,2],[3,4]], dofmap=function.Guard([0,1]), length=3, axis=0), dofmap=function.Guard([1,2]), length=3, axis=1)).blocks
    self.assertEqual(i, function.Range(3))
    self.assertAllEqual(f.eval(), [0,3,0])


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


@parametrize
class sampled(TestCase):

  def setUp(self):
    super().setUp()
    self.domain, geom = mesh.unitsquare(4, self.etype)
    basis = self.domain.basis('std', degree=1)
    numpy.random.seed(0)
    self.f = basis.dot(numpy.random.uniform(size=len(basis)))
    sample = self.domain.sample('gauss', 2)
    self.f_sampled = sample.asfunction(sample.eval(self.f))

  def test_isarray(self):
    self.assertTrue(function.isarray(self.f_sampled))

  def test_values(self):
    diff = self.domain.integrate(self.f - self.f_sampled, ischeme='gauss2')
    self.assertEqual(diff, 0)

  def test_pointset(self):
    with self.assertRaises(function.EvaluationError):
      self.domain.integrate(self.f_sampled, ischeme='uniform2')

for etype in 'square', 'triangle', 'mixed':
  sampled(etype=etype)


@parametrize
class piecewise(TestCase):

  def setUp(self):
    self.domain, self.geom = mesh.rectilinear([1])
    x, = self.geom
    if self.partition:
      left, mid, right = function.partition(x, .2, .8)
      self.f = left + function.sin(x) * mid + x**2 * right
    else:
      self.f = function.piecewise(x, [.2,.8], 1, function.sin(x), x**2)

  def test_evalf(self):
    f_ = self.domain.sample('uniform', 4).eval(self.f) # x=.125, .375, .625, .875
    assert numpy.equal(f_, [1, numpy.sin(.375), numpy.sin(.625), .875**2]).all()

  def test_deriv(self):
    g_ = self.domain.sample('uniform', 4).eval(function.grad(self.f, self.geom)) # x=.125, .375, .625, .875
    assert numpy.equal(g_, [[0], [numpy.cos(.375)], [numpy.cos(.625)], [2*.875]]).all()

piecewise(partition=False)
piecewise(partition=True)


class elemwise(TestCase):

  def setUp(self):
    super().setUp()
    self.domain, geom = mesh.rectilinear([5])
    self.index = function.TransformsIndexWithTail(self.domain.transforms, function.TRANS).index
    self.data = tuple(map(types.frozenarray, (
      numpy.arange(1, dtype=float).reshape(1,1),
      numpy.arange(2, dtype=float).reshape(1,2),
      numpy.arange(3, dtype=float).reshape(3,1),
      numpy.arange(4, dtype=float).reshape(2,2),
      numpy.arange(6, dtype=float).reshape(3,2),
    )))
    self.func = function.Elemwise(self.data, self.index, float)

  def test_evalf(self):
    for i, trans in enumerate(self.domain.transforms):
      with self.subTest(i=i):
        numpy.testing.assert_array_almost_equal(self.func.prepare_eval().eval(_transforms=(trans,)), self.data[i][_])

  def test_shape(self):
    for i, trans in enumerate(self.domain.transforms):
      with self.subTest(i=i):
        self.assertEqual(self.func.size.prepare_eval().eval(_transforms=(trans,))[0], self.data[i].size)

  def test_derivative(self):
    self.assertTrue(function.iszero(function.localgradient(self.func, self.domain.ndims)))

  def test_shape_derivative(self):
    self.assertEqual(function.localgradient(self.func, self.domain.ndims).shape, self.func.shape+(self.domain.ndims,))


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

  def test_copy_fixed_lengths(self):
    ns = function.Namespace(length_i=2)
    ns = ns.copy_()
    self.assertEqual(ns.eval_ij('δ_ij').shape, (2,2))

  def test_copy_fallback_length(self):
    ns = function.Namespace(fallback_length=2)
    ns = ns.copy_()
    self.assertEqual(ns.eval_ij('δ_ij').shape, (2,2))

  def test_eval(self):
    ns = function.Namespace()
    ns.foo = function.zeros([3,3])
    ns.eval_ij('foo_ij + foo_ji')

  def test_eval_fixed_lengths(self):
    ns = function.Namespace(length_i=2)
    self.assertEqual(ns.eval_ij('δ_ij').shape, (2,2))

  def test_eval_fixed_lengths_multiple(self):
    ns = function.Namespace(length_jk=2)
    self.assertEqual(ns.eval_ij('δ_ij').shape, (2,2))
    self.assertEqual(ns.eval_ik('δ_ik').shape, (2,2))

  def test_eval_fallback_length(self):
    ns = function.Namespace(fallback_length=2)
    self.assertEqual(ns.eval_ij('δ_ij').shape, (2,2))

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

  def test_matmul_fixed_lengths(self):
    ns = function.Namespace(length_i=2)
    self.assertEqual(('1_i δ_ij' @ ns).shape, (2,))

  def test_matmul_fallback_length(self):
    ns = function.Namespace(fallback_length=2)
    self.assertEqual(('1_i δ_ij' @ ns).shape, (2,))

  def test_replace(self):
    ns = function.Namespace(default_geometry_name='y')
    ns.foo = function.Argument('arg', [2,3])
    ns.bar_ij = 'sin(foo_ij) + cos(2 foo_ij)'
    ns = ns(arg=function.zeros([2,3]))
    self.assertEqual(ns.foo, function.zeros([2,3]))
    self.assertEqual(ns.default_geometry_name, 'y')

  def test_pickle(self):
    orig = function.Namespace()
    domain, geom = mesh.unitsquare(2, 'square')
    orig.x = geom
    orig.v = domain.basis('std', 1)
    orig.u = 'v_n ?lhs_n'
    orig.f = 'cosh(x_0)'
    pickled = pickle.loads(pickle.dumps(orig))
    for attr in ('x', 'v', 'u', 'f'):
      self.assertEqual(getattr(pickled, attr), getattr(orig, attr))
    self.assertEqual(pickled.arg_shapes['lhs'], orig.arg_shapes['lhs'])

  def test_pickle_default_geometry_name(self):
    orig = function.Namespace(default_geometry_name='g')
    pickled = pickle.loads(pickle.dumps(orig))
    self.assertEqual(pickled.default_geometry_name, orig.default_geometry_name)

  def test_pickle_fixed_lengths(self):
    orig = function.Namespace(length_i=2)
    pickled = pickle.loads(pickle.dumps(orig))
    self.assertEqual(pickled.eval_ij('δ_ij').shape, (2,2))

  def test_pickle_fallback_length(self):
    orig = function.Namespace(fallback_length=2)
    pickled = pickle.loads(pickle.dumps(orig))
    self.assertEqual(pickled.eval_ij('δ_ij').shape, (2,2))

  def test_duplicate_fixed_lengths(self):
    with self.assertRaisesRegex(ValueError, '^length of index i specified more than once$'):
      function.Namespace(length_ii=2)

  def test_unexpected_keyword_argument(self):
    with self.assertRaisesRegex(TypeError, r"^__init__\(\) got an unexpected keyword argument 'test'$"):
      function.Namespace(test=2)

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
  def test_substitute(self): self.assertIdentical('(?x_i^2)(x_i=a2_i)', self.ns.a2**2)
  def test_multisubstitute(self): self.assertIdentical('(a2_i + ?x_i + ?y_i)(x_i=?y_i, y_i=?x_i)', self.ns.a2 + function.Argument('y', [2]) + function.Argument('x', [2]))
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


@parametrize
class jacobian(TestCase):

  def setUp(self):
    self.domain, self.geom = mesh.unitsquare(1, 'square')
    self.basis = self.domain.basis('std', degree=1)
    arg = function.Argument('dofs', [4])
    self.v = self.basis.dot(arg)
    self.X = (self.geom[numpy.newaxis,:] * [[0,1],[-self.v,0]]).sum(-1) # X_i = <x_1, -2 x_0>_i
    self.J = function.J(self.X, None if self.delayed else 2)
    self.dJ = function.derivative(self.J, arg)

  def test_shape(self):
    self.assertEqual(self.J.shape, ())
    self.assertEqual(self.dJ.shape, (4,))

  def test_value(self):
    values = self.domain.sample('uniform', 2).eval(self.J, dofs=[2]*4)
    numpy.testing.assert_almost_equal(values, [2]*4)
    values1, values2 = self.domain.sample('uniform', 2).eval([self.J,
      self.v + self.v.grad(self.geom)[0] * self.geom[0]], dofs=[1,2,3,10])
    numpy.testing.assert_almost_equal(values1, values2)

  def test_derivative(self):
    values1, values2 = self.domain.sample('uniform', 2).eval([self.dJ,
      self.basis + self.basis.grad(self.geom)[:,0] * self.geom[0]], dofs=[1,2,3,10])
    numpy.testing.assert_almost_equal(values1, values2)

  def test_zeroderivative(self):
    otherarg = function.Argument('otherdofs', (10,))
    values = self.domain.sample('uniform', 2).eval(function.derivative(self.dJ, otherarg))
    self.assertEqual(values.shape[1:], self.dJ.shape + otherarg.shape)
    self.assertAllEqual(values, 0)

jacobian(delayed=True)
jacobian(delayed=False)

@parametrize
class basis(TestCase):

  def setUp(self):
    super().setUp()
    self.checknelems = len(self.checkcoeffs)
    self.checksupp = [[] for i in range(self.checkndofs)]
    for ielem, dofs in enumerate(self.checkdofs):
      for dof in dofs:
        self.checksupp[dof].append(ielem)
    assert len(self.checkcoeffs) == len(self.checkdofs)
    assert all(len(c) == len(d) for c, d in zip(self.checkcoeffs, self.checkdofs))

  def test_shape(self):
    self.assertEqual(self.basis.shape, (self.checkndofs,))

  def test_get_coefficients_pos(self):
    for ielem in range(self.checknelems):
      self.assertEqual(self.basis.get_coefficients(ielem).tolist(), self.checkcoeffs[ielem])

  def test_get_coefficients_neg(self):
    for ielem in range(-self.checknelems, 0):
      self.assertEqual(self.basis.get_coefficients(ielem).tolist(), self.checkcoeffs[ielem])

  def test_get_coefficients_outofbounds(self):
    with self.assertRaises(IndexError):
      self.basis.get_coefficients(-self.checknelems-1)
    with self.assertRaises(IndexError):
      self.basis.get_coefficients(self.checknelems)

  def test_get_dofs_scalar_pos(self):
    for ielem in range(self.checknelems):
      self.assertEqual(self.basis.get_dofs(ielem).tolist(), self.checkdofs[ielem])

  def test_get_dofs_scalar_neg(self):
    for ielem in range(-self.checknelems, 0):
      self.assertEqual(self.basis.get_dofs(ielem).tolist(), self.checkdofs[ielem])

  def test_get_dofs_scalar_outofbounds(self):
    with self.assertRaises(IndexError):
      self.basis.get_dofs(-self.checknelems-1)
    with self.assertRaises(IndexError):
      self.basis.get_dofs(self.checknelems)

  def test_dofs_array(self):
    for mask in itertools.product(*[[False, True]]*self.checknelems):
      mask = numpy.array(mask, dtype=bool)
      indices, = numpy.where(mask)
      for value in mask, indices:
        with self.subTest(tuple(value)):
          self.assertEqual(self.basis.get_dofs(value).tolist(), list(sorted(set(itertools.chain.from_iterable(self.checkdofs[i] for i in indices)))))

  def test_dofs_intarray_outofbounds(self):
    for i in [-1, self.checknelems]:
      with self.assertRaises(IndexError):
        self.basis.get_dofs(numpy.array([i], dtype=int))

  def test_dofs_intarray_invalidndim(self):
    with self.assertRaises(IndexError):
      self.basis.get_dofs(numpy.array([[0]], dtype=int))

  def test_dofs_boolarray_invalidshape(self):
    with self.assertRaises(IndexError):
      self.basis.get_dofs(numpy.array([True]*(self.checknelems+1), dtype=bool))
    with self.assertRaises(IndexError):
      self.basis.get_dofs(numpy.array([[True]*self.checknelems], dtype=bool))

  def test_get_support_scalar_pos(self):
    for dof in range(self.checkndofs):
      self.assertEqual(self.basis.get_support(dof).tolist(), self.checksupp[dof])

  def test_get_support_scalar_neg(self):
    for dof in range(-self.checkndofs, 0):
      self.assertEqual(self.basis.get_support(dof).tolist(), self.checksupp[dof])

  def test_get_support_scalar_outofbounds(self):
    with self.assertRaises(IndexError):
      self.basis.get_support(-self.checkndofs-1)
    with self.assertRaises(IndexError):
      self.basis.get_support(self.checkndofs)

  def test_get_support_array(self):
    for mask in itertools.product(*[[False, True]]*self.checkndofs):
      mask = numpy.array(mask, dtype=bool)
      indices, = numpy.where(mask)
      for value in mask, indices:
        with self.subTest(tuple(value)):
          self.assertEqual(self.basis.get_support(value).tolist(), list(sorted(set(itertools.chain.from_iterable(self.checksupp[i] for i in indices)))))

  def test_get_support_intarray_outofbounds(self):
    for i in [-1, self.checkndofs]:
      with self.assertRaises(IndexError):
        self.basis.get_support(numpy.array([i], dtype=int))

  def test_get_support_intarray_invalidndim(self):
    with self.assertRaises(IndexError):
      self.basis.get_support(numpy.array([[0]], dtype=int))

  def test_get_support_boolarray(self):
    for mask in itertools.product(*[[False, True]]*self.checkndofs):
      mask = numpy.array(mask, dtype=bool)
      indices, = numpy.where(mask)
      with self.subTest(tuple(indices)):
        self.assertEqual(self.basis.get_support(mask).tolist(), list(sorted(set(itertools.chain.from_iterable(self.checksupp[i] for i in indices)))))

  def test_get_support_boolarray_invalidshape(self):
    with self.assertRaises(IndexError):
      self.basis.get_support(numpy.array([True]*(self.checkndofs+1), dtype=bool))
    with self.assertRaises(IndexError):
      self.basis.get_support(numpy.array([[True]*self.checkndofs], dtype=bool))

  def test_getitem_array(self):
    for mask in itertools.product(*[[False, True]]*self.checkndofs):
      mask = numpy.array(mask, dtype=bool)
      indices, = numpy.where(mask)
      for value in mask, indices:
        with self.subTest(tuple(value)):
          maskedbasis = self.basis[value]
          self.assertIsInstance(maskedbasis, function.Basis)
          for ielem in range(self.checknelems):
            m = numpy.asarray(numeric.sorted_contains(indices, self.checkdofs[ielem]))
            self.assertEqual(maskedbasis.get_dofs(ielem).tolist(), numeric.sorted_index(indices, numpy.compress(m, self.checkdofs[ielem], axis=0)).tolist())
            self.assertEqual(maskedbasis.get_coefficients(ielem).tolist(), numpy.compress(m, self.checkcoeffs[ielem], axis=0).tolist())

  def checkeval(self, ielem, points):
    result = numpy.zeros((points.shape[0], self.checkndofs,), dtype=float)
    numpy.add.at(result, (slice(None),numpy.array(self.checkdofs[ielem], dtype=int)), numeric.poly_eval(numpy.array(self.checkcoeffs[ielem], dtype=float)[None], points))
    return result.tolist()

  def test_evalf(self):
    ref = element.PointReference() if self.basis.transforms.fromdims == 0 else element.LineReference()**self.basis.transforms.fromdims
    points = ref.getpoints('bezier', 4).coords
    with self.assertWarnsRegex(function.ExpensiveEvaluationWarning, 'using explicit basis evaluation.*'):
      for ielem in range(self.checknelems):
        self.assertEqual(self.basis.evalf([ielem], points).tolist(), self.checkeval(ielem, points))

  def test_simplified(self):
    ref = element.PointReference() if self.basis.transforms.fromdims == 0 else element.LineReference()**self.basis.transforms.fromdims
    points = ref.getpoints('bezier', 4).coords
    simplified = self.basis.simplified
    with _builtin_warnings.catch_warnings():
      _builtin_warnings.simplefilter('ignore', category=function.ExpensiveEvaluationWarning)
      for ielem in range(self.checknelems):
        value = simplified.prepare_eval().eval(_transforms=(self.basis.transforms[ielem],), _points=points)
        if value.shape[0] == 1:
          value = numpy.tile(value, (points.shape[0], 1))
        self.assertEqual(value.tolist(), self.checkeval(ielem, points))

basis(
  'PlainBasis',
  basis=function.PlainBasis([[1],[2,3],[4,5],[6]], [[0],[2,3],[1,3],[2]], 4, transformseq.PlainTransforms([(transform.Identifier(0,k),) for k in 'abcd'], 0)),
  checkcoeffs=[[1],[2,3],[4,5],[6]],
  checkdofs=[[0],[2,3],[1,3],[2]],
  checkndofs=4)
basis(
  'DiscontBasis',
  basis=function.DiscontBasis([[1],[2,3],[4,5],[6]], transformseq.PlainTransforms([(transform.Identifier(0,k),) for k in 'abcd'], 0)),
  checkcoeffs=[[1],[2,3],[4,5],[6]],
  checkdofs=[[0],[1,2],[3,4],[5]],
  checkndofs=6)
basis(
  'MaskedBasis',
  basis=function.MaskedBasis(function.PlainBasis([[1],[2,3],[4,5],[6]], [[0],[2,3],[1,3],[2]], 4, transformseq.PlainTransforms([(transform.Identifier(0,k),) for k in 'abcd'], 0)), [0,2]),
  checkcoeffs=[[1],[2],[],[6]],
  checkdofs=[[0],[1],[],[1]],
  checkndofs=2)
basis(
  'PrunedBasis',
  basis=function.PrunedBasis(function.PlainBasis([[1],[2,3],[4,5],[6]], [[0],[2,3],[1,3],[2]], 4, transformseq.PlainTransforms([(transform.Identifier(0,k),) for k in 'abcd'], 0)), [0,2]),
  checkcoeffs=[[1],[4,5]],
  checkdofs=[[0],[1,2]],
  checkndofs=3)

structtrans4 = transformseq.StructuredTransforms(transform.Identifier(1, 'test'), [transformseq.DimAxis(0,4,False)], 0)
structtrans4p = transformseq.StructuredTransforms(transform.Identifier(1, 'test'), [transformseq.DimAxis(0,4,True)], 0)
structtrans22 = transformseq.StructuredTransforms(transform.Identifier(2, 'test'), [transformseq.DimAxis(0,2,False),transformseq.DimAxis(0,2,False)], 0)
basis(
  'StructuredBasis1D',
  basis=function.StructuredBasis([[[[1],[2]],[[3],[4]],[[5],[6]],[[7],[8]]]], [[0,1,2,3]], [[2,3,4,5]], [5], structtrans4, [4]),
  checkcoeffs=[[[1],[2]],[[3],[4]],[[5],[6]],[[7],[8]]],
  checkdofs=[[0,1],[1,2],[2,3],[3,4]],
  checkndofs=5)
basis(
  'StructuredBasis1DPeriodic',
  basis=function.StructuredBasis([[[[1],[2]],[[3],[4]],[[5],[6]],[[7],[8]]]], [[0,1,2,3]], [[2,3,4,5]], [4], structtrans4p, [4]),
  checkcoeffs=[[[1],[2]],[[3],[4]],[[5],[6]],[[7],[8]]],
  checkdofs=[[0,1],[1,2],[2,3],[3,0]],
  checkndofs=4)
basis(
  'StructuredBasis2D',
  basis=function.StructuredBasis([[[[1],[2]],[[3],[4]]],[[[5],[6]],[[7],[8]]]], [[0,1],[0,1]], [[2,3],[2,3]], [3,3], structtrans22, [2,2]),
  checkcoeffs=[[[[5]],[[6]],[[10]],[[12]]],[[[7]],[[8]],[[14]],[[16]]],[[[15]],[[18]],[[20]],[[24]]],[[[21]],[[24]],[[28]],[[32]]]],
  checkdofs=[[0,1,3,4],[1,2,4,5],[3,4,6,7],[4,5,7,8]],
  checkndofs=9)
