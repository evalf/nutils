import numpy, itertools, pickle, warnings as _builtin_warnings
from nutils import *
from nutils.testing import *
_ = numpy.newaxis


@parametrize
class check(TestCase):

  def setUp(self):
    super().setUp()
    domain, param = mesh.rectilinear([1]*self.ndim, periodic=[0])
    param = param.prepare_eval()
    rootcoords = evaluable.ApplyTransforms(evaluable.PopHead(self.ndim, evaluable.SelectChain(0)), evaluable.Points())
    if self.ndim == 1:
      self.geom = param**2
      poly = numpy.array([[1,-2,1],[0,2,-2],[0,0,1]]) # 2nd order bernstein
    elif self.ndim == 2:
      r, theta = param
      self.geom = r * evaluable.stack([evaluable.cos(theta), evaluable.sin(theta)])
      poly = numeric.poly_outer_product([[1,-2,1],[0,2,-2],[0,0,1]], [[1,-1],[0,1]]) # 2nd x 1st order bernstein
    else:
      raise Exception('invalid ndim {!r}'.format(self.ndim))
    numpy.random.seed(0)
    self.args = [evaluable.Guard(evaluable.Polyval(numeric.dot(numpy.random.uniform(size=shape+poly.shape[:1], low=self.low, high=self.high), poly), rootcoords)) for shape in self.shapes]
    if self.pass_geom:
        self.args += [self.geom]
    self.sample = domain.sample('uniform', 2)
    self.ifacesmp = domain.interfaces.sample('uniform', 2)
    self.n_op_argsfun = self.n_op(*self.sample.eval_evaluable(self.args))
    self.op_args = self.op(*self.args)
    self.shapearg = numpy.random.uniform(size=self.op_args.shape, low=self.low, high=self.high)
    self.pairs = [(i, j) for i in range(self.op_args.ndim-1) for j in range(i+1, self.op_args.ndim) if self.op_args.shape[i] == self.op_args.shape[j]]
    _builtin_warnings.simplefilter('ignore', evaluable.ExpensiveEvaluationWarning)

  def assertArrayAlmostEqual(self, actual, desired, decimal):
    if actual.shape[1:] != desired.shape[1:] or actual.shape[0] != 1 and desired.shape[0] != 1 and actual.shape[0] != desired.shape[0]:
      self.fail('shapes of actual {} and desired {} are incompatible.'.format(actual.shape, desired.shape))
    error = actual - desired if not actual.dtype.kind == desired.dtype.kind == 'b' else actual ^ desired
    approx = error.dtype.kind in 'fc'
    indices = tuple(zip(*(numpy.greater_equal(abs(error), 1.5 * 10**-decimal) if approx else error).nonzero()))
    if not indices:
      return
    lines = ['arrays are not equal']
    if approx:
      lines.append(' up to {} decimals'.format(decimal))
    lines.append(' in {}/{} entries:'.format(len(indices), error.size))
    n = 5
    lines.extend('\n  {} actual={} desired={} difference={}'.format(index, actual[index], desired[index], error[index]) for index in indices[:n])
    if len(indices) > 2*n:
      lines.append('\n  ...')
      n = -n
    lines.extend('\n  {} actual={} desired={} difference={}'.format(index, actual[index], desired[index], error[index]) for index in indices[n:])
    self.fail(''.join(lines))

  def assertFunctionAlmostEqual(self, actual, desired, decimal):
    evalargs = dict(_transforms=[trans[0] for trans in self.sample.transforms], _points=self.sample.points[0].coords)
    with self.subTest('vanilla'):
      self.assertArrayAlmostEqual(actual.eval(**evalargs), desired, decimal)
    with self.subTest('simplified'):
      self.assertArrayAlmostEqual(actual.simplified.eval(**evalargs), desired, decimal)
    with self.subTest('optimized'):
      self.assertArrayAlmostEqual(actual.optimized_for_numpy.eval(**evalargs), desired, decimal)
    with self.subTest('sample'):
      self.assertArrayAlmostEqual(self.sample.eval_evaluable(actual), desired, decimal)

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
      actual=evaluable.transpose(self.op_args, trans))

  def test_insertaxis(self):
    for axis in range(self.op_args.ndim+1):
      with self.subTest(axis=axis):
        self.assertFunctionAlmostEqual(decimal=15,
          desired=numpy.repeat(numpy.expand_dims(self.n_op_argsfun, axis+1), 2, axis+1),
          actual=evaluable.insertaxis(self.op_args, axis, 2))

  def test_takediag(self):
    for ax1, ax2 in self.pairs:
      self.assertFunctionAlmostEqual(decimal=15,
        desired=numeric.takediag(self.n_op_argsfun, ax1+1, ax2+1),
        actual=evaluable.takediag(self.op_args, ax1, ax2))

  def test_eig(self):
    if self.op_args.dtype == float:
      for ax1, ax2 in self.pairs:
        A = self.sample.eval_evaluable(self.op_args)
        L, V = self.sample.eval_evaluable(list(evaluable.eig(self.op_args, axes=(ax1,ax2))))
        self.assertArrayAlmostEqual(decimal=11,
          actual=(numpy.expand_dims(V,ax2+1) * numpy.expand_dims(L,ax2+2).swapaxes(ax1+1,ax2+2)).sum(ax2+2),
          desired=(numpy.expand_dims(A,ax2+1) * numpy.expand_dims(V,ax2+2).swapaxes(ax1+1,ax2+2)).sum(ax2+2))

  def test_inv(self):
    for ax1, ax2 in self.pairs:
      trans = [i for i in range(self.n_op_argsfun.ndim) if i not in (ax1+1,ax2+1)] + [ax1+1,ax2+1]
      invtrans = list(map(trans.index, range(len(trans))))
      self.assertFunctionAlmostEqual(decimal=10,
        desired=numeric.inv(self.n_op_argsfun.transpose(trans)).transpose(invtrans),
        actual=evaluable.inverse(self.op_args, axes=(ax1,ax2)))

  def test_determinant(self):
    for ax1, ax2 in self.pairs:
      self.assertFunctionAlmostEqual(decimal=11,
        desired=numpy.linalg.det(self.n_op_argsfun.transpose([i for i in range(self.n_op_argsfun.ndim) if i not in (ax1+1,ax2+1)] + [ax1+1,ax2+1])),
        actual=evaluable.determinant(self.op_args, axes=(ax1,ax2)))

  def test_take(self):
    indices = [0,-1]
    for iax, sh in enumerate(self.op_args.shape):
      if sh >= 2:
        self.assertFunctionAlmostEqual(decimal=15,
          desired=numpy.take(self.n_op_argsfun, indices, axis=iax+1),
          actual=evaluable.take(self.op_args, indices, axis=iax))

  def test_take_nomask(self):
    for iax, sh in enumerate(self.op_args.shape):
      if sh >= 2:
        indices = [0,sh-1]
        self.assertFunctionAlmostEqual(decimal=15,
          desired=numpy.take(self.n_op_argsfun, indices, axis=iax+1),
          actual=evaluable.take(self.op_args, evaluable.Guard(evaluable.asarray(indices)), axis=iax))

  def test_take_reversed(self):
    indices = [-1,0]
    for iax, sh in enumerate(self.op_args.shape):
      if sh >= 2:
        self.assertFunctionAlmostEqual(decimal=15,
          desired=numpy.take(self.n_op_argsfun, indices, axis=iax+1),
          actual=evaluable.take(self.op_args, indices, axis=iax))

  def test_inflate(self):
    for iax, sh in enumerate(self.op_args.shape):
      dofmap = evaluable.Constant(numpy.arange(sh) * 2)
      desired = numpy.zeros(self.n_op_argsfun.shape[:iax+1] + (sh*2-1,) + self.n_op_argsfun.shape[iax+2:], dtype=self.n_op_argsfun.dtype)
      desired[(slice(None),)*(iax+1)+(slice(None,None,2),)] = self.n_op_argsfun
      self.assertFunctionAlmostEqual(decimal=15,
        desired=desired,
        actual=evaluable._inflate(self.op_args, dofmap=dofmap, length=sh*2-1, axis=iax))

  def test_diagonalize(self):
    for axis in range(self.op_args.ndim):
      for newaxis in range(axis+1, self.op_args.ndim+1):
        self.assertFunctionAlmostEqual(decimal=15,
          desired=numeric.diagonalize(self.n_op_argsfun, axis+1, newaxis+1),
          actual=evaluable.diagonalize(self.op_args, axis, newaxis))

  def test_product(self):
    for iax in range(self.op_args.ndim):
      self.assertFunctionAlmostEqual(decimal=15,
        desired=numpy.product(self.n_op_argsfun, axis=iax+1),
        actual=evaluable.product(self.op_args, axis=iax))

  def test_concatenate(self):
    for idim in range(self.op_args.ndim):
      self.assertFunctionAlmostEqual(decimal=15,
        desired=numpy.concatenate([self.n_op_argsfun, self.shapearg[_].repeat(len(self.n_op_argsfun),0)], axis=idim+1),
        actual=evaluable.concatenate([self.op_args, self.shapearg], axis=idim))

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
        actual=evaluable.dot(self.op_args, self.shapearg, axes=iax))

  def test_pointwise(self):
    self.assertFunctionAlmostEqual(decimal=15,
      desired=numpy.sin(self.n_op_argsfun).astype(float), # "astype" necessary for boolean operations (float16->float64)
      actual=evaluable.sin(self.op_args))

  def test_cross(self):
    triaxes = [iax for iax, sh in enumerate(self.op_args.shape) if sh == 3]
    if triaxes:
      for iax in triaxes:
        self.assertFunctionAlmostEqual(decimal=15,
          desired=numpy.cross(self.n_op_argsfun, self.shapearg[_], axis=iax+1),
          actual=evaluable.cross(self.op_args, self.shapearg, axis=iax))

  def test_power(self):
    self.assertFunctionAlmostEqual(decimal=13,
      desired=self.n_op_argsfun**3,
      actual=(self.op_args**3))

  def test_power0(self):
    power = (numpy.arange(self.op_args.size) % 2).reshape(self.op_args.shape)
    self.assertFunctionAlmostEqual(decimal=13,
      desired=self.n_op_argsfun**power,
      actual=self.op_args**power)

  def test_sign(self):
    if self.n_op_argsfun.dtype.kind != 'b':
      self.assertFunctionAlmostEqual(decimal=15,
        desired=numpy.sign(self.n_op_argsfun),
        actual=evaluable.sign(self.op_args))

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
        actual=evaluable.mask(self.op_args, mask, axis=idim))

  def test_ravel(self):
    for idim in range(self.op_args.ndim-1):
      self.assertFunctionAlmostEqual(decimal=15,
        desired=self.n_op_argsfun.reshape(self.n_op_argsfun.shape[:idim+1]+(-1,)+self.n_op_argsfun.shape[idim+3:]),
        actual=evaluable.ravel(self.op_args, axis=idim))

  def test_unravel(self):
    for idim in range(self.op_args.ndim):
      length = self.n_op_argsfun.shape[idim+1]
      unravelshape = (length//3,3) if (length%3==0) else (length//2,2) if (length%2==0) else (length,1)
      self.assertFunctionAlmostEqual(decimal=15,
        desired=self.n_op_argsfun.reshape(self.n_op_argsfun.shape[:idim+1]+unravelshape+self.n_op_argsfun.shape[idim+2:]),
        actual=evaluable.unravel(self.op_args, axis=idim, shape=unravelshape))

  def test_kronecker(self):
    for idim in range(self.op_args.ndim+1):
      desired = numpy.zeros(self.n_op_argsfun.shape[:idim+1]+(3,)+self.n_op_argsfun.shape[idim+1:], dtype=self.n_op_argsfun.dtype)
      desired[(slice(None),)*(idim+1)+(1,)] = self.n_op_argsfun
      self.assertFunctionAlmostEqual(decimal=15,
        desired=desired,
        actual=evaluable.kronecker(self.op_args, axis=idim, pos=1, length=3))

  def test_desparsify(self):
    args = []
    for arg in self.args:
      for i in range(arg.ndim):
        arg = evaluable._inflate(arg, evaluable.Guard(numpy.arange(arg.shape[i])), arg.shape[i], i)
      args.append(arg)
    op_args = self.op(*args).simplified
    _transforms = [trans[0] for trans in self.sample.transforms]
    for axis, prop in enumerate(op_args._axes):
      if isinstance(prop, evaluable.Sparse):
        actual = numpy.zeros_like(self.n_op_argsfun)
        for ind, f in op_args._desparsify(axis):
          _ind, = ind.eval(_transforms=_transforms)
          actual[(slice(None),)*(axis+1)+(_ind,)] += f.eval(_transforms=_transforms, _points=self.sample.points[0].coords)
        self.assertArrayAlmostEqual(actual, self.n_op_argsfun, decimal=15)

  def find(self, target, xi0):
    elemtrans, = self.sample.transforms[0]
    ndim, = self.geom.shape
    J = evaluable.localgradient(self.geom, ndim)
    Jinv = evaluable.inverse(J)
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
      err = target - self.geom.eval(_transforms=[elemtrans], _points=xi)
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
    argsfun = evaluable.Tuple(self.args)
    exact = self.sample.eval_evaluable(evaluable.localgradient(self.op_args, ndims=self.ndim))
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
      x = evaluable.Argument('x', shape)
      f = self.op(*self.args[:iarg]+[x]+self.args[iarg+1:])
      fx0, Jx0 = self.sample.eval_evaluable([f, evaluable.derivative(f, x)], x=x0)
      fx1 = self.sample.eval_evaluable(f, x=x0+dx)
      fx1approx = fx0 + numeric.contract(Jx0, dx, range(fx0.ndim, Jx0.ndim))
      self.assertArrayAlmostEqual(fx1approx, fx1, decimal=12)

  @parametrize.enable_if(lambda hasgrad, **kwargs: hasgrad)
  def test_gradient(self):
    elemtrans, = self.sample.transforms[0]
    points = self.sample.points[0].coords
    argsfun = evaluable.Tuple(self.args)
    exact = self.sample.eval_evaluable(self.op_args.grad(self.geom))
    fddeltas = numpy.array([1,2,3])
    fdfactors = numpy.linalg.solve(2*fddeltas**numpy.arange(1,1+2*len(fddeltas),2)[:,None], [1]+[0]*(len(fddeltas)-1))
    D = numpy.array([-fddeltas, fddeltas])[:,:,_,_] * numpy.eye(self.geom.shape[-1])
    good = False
    eps = 1e-4
    while not numpy.all(good):
      fdpoints = self.find(self.sample.eval_evaluable(self.geom)[_,_,_,:,:] + D[:,:,:,_,:] * eps, points[_,_,_,:,:])
      tmp = self.n_op(*argsfun.eval(_transforms=[elemtrans], _points=fdpoints.reshape(-1,fdpoints.shape[-1])))
      if len(tmp) == 1 or tmp.dtype.kind in 'bi' or self.zerograd:
        error = exact
      else:
        a, b = tmp.reshape(fdpoints.shape[:-1] + tmp.shape[1:])
        diff = b - a
        fdgrad = (diff.T @ fdfactors).T / eps
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
    argsfun = evaluable.Tuple(self.args)
    exact = self.sample.eval_evaluable(self.op_args.grad(self.geom).grad(self.geom))
    fddeltas = numpy.array([1,2,3])
    fdfactors = numpy.linalg.solve(2*fddeltas**numpy.arange(1,1+2*len(fddeltas),2)[:,None], [1]+[0]*(len(fddeltas)-1))
    D = numpy.array([-fddeltas, fddeltas])[:,:,_,_] * numpy.eye(self.geom.shape[-1])
    DD = D[:,_,:,_,:,_,:] + D[_,:,_,:,_,:,:]
    good = False
    eps = 1e-4
    while not numpy.all(good):
      fdpoints = self.find(self.sample.eval_evaluable(self.geom)[_,_,_,_,_,_,:,:] + DD[:,:,:,:,:,:,_,:] * eps, points[_,_,_,_,_,_,:,:])
      tmp = self.n_op(*argsfun.eval(_transforms=[elemtrans], _points=fdpoints.reshape(-1,fdpoints.shape[-1])))
      if len(tmp) == 1 or tmp.dtype.kind in 'bi' or self.zerograd:
        error = exact
      else:
        (a,b), (c,d) = tmp.reshape(fdpoints.shape[:-1] + tmp.shape[1:])
        diff = (d-c) - (b-a)
        fddgrad = (diff.T @ fdfactors @ fdfactors).T / eps**2
        error = exact - fddgrad.transpose(numpy.roll(numpy.arange(fddgrad.ndim),-2))
      good |= numpy.less(abs(error / exact), 1e-4)
      good |= numpy.less(abs(error), 1e-14)
      eps *= .8
      if eps < 1e-10:
        self.fail('double gradient failed to reach tolerance ({}/{})'.format((~good).sum(), good.size))

def _check(name, op, n_op, shapes, hasgrad=True, zerograd=False, pass_geom=False, ndim=2, low=-1, high=1):
  check(name, op=op, n_op=n_op, shapes=shapes, hasgrad=hasgrad, zerograd=zerograd, pass_geom=pass_geom, ndim=ndim, low=low, high=high)

_check('identity', lambda f: evaluable.asarray(f), lambda a: a, [(2,4,2)])
_check('const', lambda f: evaluable.asarray(numpy.arange(16, dtype=float).reshape(2,4,2)), lambda a: numpy.arange(16, dtype=float).reshape(1,2,4,2), [(2,4,2)])
_check('zeros', lambda f: evaluable.zeros([4,3,4]), lambda a: numpy.zeros([1,4,3,4]), [(4,3,4)])
_check('ones', lambda f: evaluable.ones([4,3,4]), lambda a: numpy.ones([1,4,3,4]), [(4,3,4)])
_check('range', lambda f: evaluable.Range(4, offset=2), lambda a: numpy.arange(2,6)[numpy.newaxis], [(4,)])
_check('sin', evaluable.sin, numpy.sin, [(4,)])
_check('cos', evaluable.cos, numpy.cos, [(4,)])
_check('tan', evaluable.tan, numpy.tan, [(4,)])
_check('sqrt', evaluable.sqrt, numpy.sqrt, [(4,)], low=0)
_check('log', evaluable.ln, numpy.log, [(4,)], low=0)
_check('log2', evaluable.log2, numpy.log2, [(4,)], low=0)
_check('log10', evaluable.log10, numpy.log10, [(4,)], low=0)
_check('exp', evaluable.exp, numpy.exp, [(4,)])
_check('arctanh', evaluable.arctanh, numpy.arctanh, [(4,)])
_check('tanh', evaluable.tanh, numpy.tanh, [(4,)])
_check('cosh', evaluable.cosh, numpy.cosh, [(4,)])
_check('sinh', evaluable.sinh, numpy.sinh, [(4,)])
_check('abs', evaluable.abs, numpy.abs, [(4,)])
_check('sign', evaluable.sign, numpy.sign, [(4,4)], zerograd=True)
_check('power', evaluable.power, numpy.power, [(4,1),(1,4)], low=0)
_check('negative', evaluable.negative, numpy.negative, [(4,)])
_check('reciprocal', evaluable.reciprocal, numpy.reciprocal, [(4,)], low=-2, high=-1)
_check('arcsin', evaluable.arcsin, numpy.arcsin, [(4,)])
_check('arccos', evaluable.arccos, numpy.arccos, [(4,)])
_check('arctan', evaluable.arctan, numpy.arctan, [(4,)])
_check('ln', evaluable.ln, numpy.log, [(4,)], low=0)
_check('product', lambda a: evaluable.product(a,1), lambda a: numpy.product(a,-2), [(4,3,4)])
_check('norm2', lambda a: evaluable.norm2(a,1), lambda a: (a**2).sum(-2)**.5, [(4,3,4)])
_check('norm2scalar', lambda a: evaluable.norm2(a,1), lambda a: abs(a.sum(-2)), [(4,1,4)])
_check('sum', lambda a: evaluable.sum(a,1), lambda a: a.sum(-2), [(4,3,4)])
_check('transpose1', lambda a: evaluable.transpose(a,[0,2,1]), lambda a: a.transpose([0,1,3,2]), [(4,4,4)], low=0, high=20)
_check('transpose2', lambda a: evaluable.transpose(a,[1,2,0]), lambda a: a.transpose([0,2,3,1]), [(4,4,4)])
_check('expand_dims', lambda a: evaluable.expand_dims(a,1), lambda a: numpy.expand_dims(a,2), [(2,4)])
_check('get', lambda a: evaluable.get(a,1,1), lambda a: a[...,1,:], [(4,3,4)])
_check('takediag141', lambda a: evaluable.takediag(a,0,2), lambda a: numeric.takediag(a,1,3), [(1,4,1)])
_check('takediag434', lambda a: evaluable.takediag(a,0,2), lambda a: numeric.takediag(a,1,3), [(4,3,4)])
_check('takediag343', lambda a: evaluable.takediag(a,0,2), lambda a: numeric.takediag(a,1,3), [(3,4,3)])
_check('determinant141', lambda a: evaluable.determinant(a,(0,2)), lambda a: numpy.linalg.det(a.swapaxes(-3,-2)), [(1,4,1)])
_check('determinant434', lambda a: evaluable.determinant(a,(0,2)), lambda a: numpy.linalg.det(a.swapaxes(-3,-2)), [(4,3,4)])
_check('determinant4433', lambda a: evaluable.determinant(a,(2,3)), lambda a: numpy.linalg.det(a), [(4,4,3,3)])
_check('determinant200', lambda a: evaluable.determinant(a,(1,2)), lambda a: numpy.linalg.det(a) if a.shape[-1] else numpy.ones(a.shape[:-2], float), [(2,0,0)], zerograd=True)
_check('inverse141', lambda a: evaluable.inverse(a+evaluable.eye(1)[:,None],(0,2)), lambda a: numpy.linalg.inv(a.swapaxes(-3,-2)+numpy.eye(1)).swapaxes(-3,-2), [(1,4,1)])
_check('inverse434', lambda a: evaluable.inverse(a+evaluable.eye(4)[:,None],(0,2)), lambda a: numpy.linalg.inv(a.swapaxes(-3,-2)+numpy.eye(4)).swapaxes(-3,-2), [(4,3,4)])
_check('inverse4422', lambda a: evaluable.inverse(a+evaluable.eye(2)), lambda a: numpy.linalg.inv(a+numpy.eye(2)), [(4,4,2,2)])
_check('repeat', lambda a: evaluable.repeat(a,3,1), lambda a: numpy.repeat(a,3,-2), [(4,1,4)])
_check('diagonalize', lambda a: evaluable.diagonalize(a,1,3), lambda a: numeric.diagonalize(a,2,4), [(4,4,4,4,4)])
_check('multiply', evaluable.multiply, numpy.multiply, [(4,1),(4,4)])
_check('dot', lambda a,b: evaluable.dot(a,b,axes=1), lambda a,b: (a*b).sum(2), [(4,2,4),(4,2,4)])
_check('divide', evaluable.divide, lambda a, b: a * b**-1, [(4,4),(1,4)], low=-2, high=-1)
_check('divide2', lambda a: evaluable.asarray(a)/2, lambda a: a/2, [(4,1)])
_check('add', evaluable.add, numpy.add, [(4,1),(1,4)])
_check('subtract', evaluable.subtract, numpy.subtract, [(4,1),(1,4)])
_check('dot2', lambda a,b: evaluable.multiply(a,b).sum(-2), lambda a,b: (a*b).sum(-2), [(4,2,4),(1,2,4)])
_check('cross', lambda a,b: evaluable.cross(a,b,-2), lambda a,b: numpy.cross(a,b,axis=-2), [(4,3,1),(4,3,4)])
_check('min', lambda a,b: evaluable.min(a,b), numpy.minimum, [(4,1),(1,4)])
_check('max', lambda a,b: evaluable.max(a,b), numpy.maximum, [(4,1),(1,4)])
_check('equal', lambda a,b: evaluable.equal(a,b), numpy.equal, [(4,1),(1,4)])
_check('greater', lambda a,b: evaluable.greater(a,b), numpy.greater, [(4,1),(1,4)])
_check('less', lambda a,b: evaluable.less(a,b), numpy.less, [(4,1),(1,4)])
_check('arctan2', evaluable.arctan2, numpy.arctan2, [(4,1),(1,4)])
_check('stack', lambda a,b: evaluable.stack([a,b]), lambda a,b: numpy.concatenate([a[...,_,:],b[...,_,:]], axis=-2), [(4,),(4,)])
_check('concatenate1', lambda a,b: evaluable.concatenate([a,b],axis=0), lambda a,b: numpy.concatenate([a,b], axis=-2), [(4,6),(2,6)])
_check('concatenate2', lambda a,b: evaluable.concatenate([a,b],axis=1), lambda a,b: numpy.concatenate([a,b], axis=-1), [(4,3),(4,1)])
_check('eig', lambda a: evaluable.eig(a+a.T,symmetric=True)[1], lambda a: numpy.linalg.eigh(a+a.swapaxes(1,2))[1], [(4,4)], hasgrad=False)
_check('trignormal', lambda a: evaluable.trignormal(a), lambda a: numpy.array([numpy.cos(a), numpy.sin(a)]).T, [()])
_check('trigtangent', lambda a: evaluable.trigtangent(a), lambda a: numpy.array([-numpy.sin(a), numpy.cos(a)]).T, [()])
_check('mod', lambda a,b: evaluable.mod(a,b), lambda a,b: numpy.mod(a,b), [(4,),(4,)], hasgrad=False)
_check('kronecker', lambda f: evaluable.kronecker(f,axis=2,length=4,pos=1), lambda a: numeric.kronecker(a,axis=3,length=4,pos=1), [(4,4,4,4)])
_check('mask', lambda f: evaluable.mask(f,numpy.array([True,False,True,False,True,False,True]),axis=1), lambda a: a[:,:,::2], [(4,7,4)])
_check('ravel', lambda f: evaluable.ravel(f,axis=1), lambda a: a.reshape(-1,4,4,4,4), [(4,2,2,4,4)])
_check('unravel', lambda f: evaluable.unravel(f,axis=1,shape=[2,2]), lambda a: a.reshape(-1,4,2,2,4,4), [(4,4,4,4)])
_check('inflate', lambda f: evaluable._inflate(f,dofmap=evaluable.Guard([0,3]),length=4,axis=1), lambda a: numpy.concatenate([a[:,:,:1], numpy.zeros_like(a), a[:,:,1:]], axis=2), [(4,2,4)])
_check('inflate-constant', lambda f: evaluable._inflate(f,dofmap=[0,3],length=4,axis=1), lambda a: numpy.concatenate([a[:,:,:1], numpy.zeros_like(a), a[:,:,1:]], axis=2), [(4,2,4)])
_check('vectorize', lambda a,b: evaluable.vectorize([a, b]), lambda a,b: numpy.concatenate([numpy.stack([a, numpy.zeros_like(a)], axis=2), numpy.stack([numpy.zeros_like(b), b], axis=2)], axis=1), [(3,),(5,)])
_check('choose', lambda a, b, c: evaluable.Choose(evaluable.Int(a)%2, [b,c]), lambda a, b, c: numpy.stack([b,c], axis=1)[numpy.arange(len(a)), a.astype(int)%2], [(), (3,3), (3,3)])
_check('slice', lambda a: evaluable.asarray(a)[::2], lambda a: a[:,::2], [(5,3)])


class blocks(TestCase):

  def setUp(self):
    super().setUp()
    _builtin_warnings.simplefilter('ignore', evaluable.ExpensiveEvaluationWarning)

  def test_multiply_equal(self):
    ((i,), f), = evaluable.multiply(evaluable._inflate([1,2], dofmap=[0,2], length=3, axis=0), evaluable._inflate([3,4], dofmap=[0,2], length=3, axis=0)).blocks
    self.assertAllEqual(i.eval(), [0,2])
    self.assertAllEqual(f.eval(), [1*3,2*4])

  def test_multiply_embedded(self):
    ((i,), f), = evaluable.multiply([1,2,3], evaluable._inflate([4,5], dofmap=[0,2], length=3, axis=0)).blocks
    self.assertAllEqual(i.eval(), [0,2])
    self.assertAllEqual(f.eval(), [1*4,3*5])

  def test_multiply_overlapping(self):
    ((i,), f), = evaluable.multiply(evaluable._inflate([1,2], dofmap=[0,1], length=3, axis=0), evaluable._inflate([3,4], dofmap=[1,2], length=3, axis=0)).blocks
    self.assertAllEqual(i.eval(), [1])
    self.assertAllEqual(f.eval(), [2*3])

  def test_multiply_disjoint(self):
    blocks = evaluable.multiply(evaluable._inflate([1,2], dofmap=[0,2], length=4, axis=0), evaluable._inflate([3,4], dofmap=[1,3], length=4, axis=0)).blocks
    self.assertEqual(blocks, ())

  def test_multiply_overlap(self):
    ((i,), f), = evaluable.multiply(evaluable._inflate([1,2], dofmap=evaluable.Guard([0,1]), length=3, axis=0), evaluable._inflate([3,4], dofmap=evaluable.Guard([1,2]), length=3, axis=0)).blocks
    self.assertAllEqual(i.eval(), [1])
    self.assertAllEqual(f.eval(), [2*3])

  def test_takediag(self):
    ((i,), f), = evaluable.takediag([[1,2,3],[4,5,6],[7,8,9]]).blocks
    self.assertAllEqual(i.eval(), [0,1,2])
    self.assertAllEqual(f.eval(), [1,5,9])

  def test_takediag_embedded_axis(self):
    ((i,), f), = evaluable.takediag(evaluable._inflate([[1,2,3],[4,5,6]], dofmap=[0,2], length=3, axis=0)).blocks
    self.assertAllEqual(i.eval(), [0,2])
    self.assertAllEqual(f.eval(), [1,6])

  def test_takediag_embedded_rmaxis(self):
    ((i,), f), = evaluable.takediag(evaluable._inflate([[1,2],[3,4],[5,6]], dofmap=[0,2], length=3, axis=1)).blocks
    self.assertAllEqual(i.eval(), [0,2])
    self.assertAllEqual(f.eval(), [1,6])

  def test_takediag_overlapping(self):
    ((i,), f), = evaluable.takediag(evaluable._inflate(evaluable._inflate([[1,2],[3,4]], dofmap=[0,1], length=3, axis=0), dofmap=[1,2], length=3, axis=1)).blocks
    self.assertAllEqual(i.eval(), [1])
    self.assertAllEqual(f.eval(), [3])

  def test_takediag_disjoint(self):
    blocks = evaluable.takediag(evaluable._inflate(evaluable._inflate([[1,2],[3,4]], dofmap=[0,2], length=4, axis=0), dofmap=[1,3], length=4, axis=1)).blocks
    self.assertEqual(blocks, ())

  def test_takediag_overlap(self):
    ((i,), f), = evaluable.takediag(evaluable._inflate(evaluable._inflate([[1,2],[3,4]], dofmap=evaluable.Guard([0,1]), length=3, axis=0), dofmap=evaluable.Guard([1,2]), length=3, axis=1)).blocks
    self.assertAllEqual(i.eval(), [1])
    self.assertAllEqual(f.eval(), [3])


class commutativity(TestCase):

  def setUp(self):
    super().setUp()
    numpy.random.seed(0)
    self.A = evaluable.asarray(numpy.random.uniform(size=[2,3]))
    self.B = evaluable.asarray(numpy.random.uniform(size=[2,3]))

  def test_add(self):
    self.assertEqual(evaluable.add(self.A, self.B), evaluable.add(self.B, self.A))

  def test_multiply(self):
    self.assertEqual(evaluable.multiply(self.A, self.B), evaluable.multiply(self.B, self.A))

  def test_dot(self):
    self.assertEqual(evaluable.dot(self.A, self.B, axes=[0]), evaluable.dot(self.B, self.A, axes=[0]))

  def test_combined(self):
    self.assertEqual(evaluable.add(self.A, self.B) * evaluable.dot(self.A, self.B, axes=[0]), evaluable.dot(self.B, self.A, axes=[0]) * evaluable.add(self.B, self.A))


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
    with self.assertRaises(evaluable.EvaluationError):
      self.domain.integrate(self.f_sampled, ischeme='uniform2')

for etype in 'square', 'triangle', 'mixed':
  sampled(etype=etype)


@parametrize
class piecewise(TestCase):

  def setUp(self):
    super().setUp()
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
    self.index = self.domain.f_index.prepare_eval(ndims=self.domain.ndims)
    self.data = tuple(map(types.frozenarray, (
      numpy.arange(1, dtype=float).reshape(1,1),
      numpy.arange(2, dtype=float).reshape(1,2),
      numpy.arange(3, dtype=float).reshape(3,1),
      numpy.arange(4, dtype=float).reshape(2,2),
      numpy.arange(6, dtype=float).reshape(3,2),
    )))
    self.func = evaluable.Elemwise(self.data, self.index, float)

  def test_evalf(self):
    for i, (trans, points) in enumerate(zip(self.domain.transforms, self.domain.sample('gauss', 1).points)):
      with self.subTest(i=i):
        numpy.testing.assert_array_almost_equal(self.func.eval(_transforms=(trans,), _points=points), self.data[i][_])

  def test_shape(self):
    for i, (trans, points) in enumerate(zip(self.domain.transforms, self.domain.sample('gauss', 1).points)):
      with self.subTest(i=i):
        self.assertEqual(self.func.size.eval(_transforms=(trans,), _points=points), self.data[i].size)

  def test_derivative(self):
    self.assertTrue(evaluable.iszero(evaluable.localgradient(self.func, self.domain.ndims).simplified))

  def test_shape_derivative(self):
    self.assertEqual(evaluable.localgradient(self.func, self.domain.ndims).shape, self.func.shape+(self.domain.ndims,))

@parametrize
class jacobian(TestCase):

  def setUp(self):
    super().setUp()
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

class grad(TestCase):

  def assertEvalAlmostEqual(self, topo, factual, fdesired):
    actual, desired = topo.sample('uniform', 2).eval([function.asarray(factual), function.asarray(fdesired)])
    self.assertAllAlmostEqual(actual, desired)

  def test_0d(self):
    domain, (x,) = mesh.rectilinear([1])
    self.assertEvalAlmostEqual(domain, function.grad(x**2, x), 2*x)

  def test_1d(self):
    domain, x = mesh.rectilinear([1]*2)
    self.assertEvalAlmostEqual(domain, function.grad([x[0]**2, x[1]**2], x), [[2*x[0], 0], [0, 2*x[1]]])

  def test_2d(self):
    domain, x = mesh.rectilinear([1]*4)
    x = function.unravel(x, 0, (2, 2))
    self.assertEvalAlmostEqual(domain, function.grad(x, x), numpy.eye(4, 4).reshape(2, 2, 2, 2))

  def test_3d(self):
    domain, x = mesh.rectilinear([1]*4)
    x = function.unravel(function.unravel(x, 0, (2, 2)), 0, (2, 1))
    self.assertEvalAlmostEqual(domain, function.grad(x, x), numpy.eye(4, 4).reshape(2, 1, 2, 2, 1, 2))

class normal(TestCase):

  def assertEvalAlmostEqual(self, topo, factual, fdesired):
    actual, desired = topo.sample('uniform', 2).eval([function.asarray(factual), function.asarray(fdesired)])
    self.assertAllAlmostEqual(actual, desired)

  def test_0d(self):
    domain, (x,) = mesh.rectilinear([1])
    self.assertEvalAlmostEqual(domain.boundary['right'], function.normal(x), 1)
    self.assertEvalAlmostEqual(domain.boundary['left'], function.normal(x), -1)

  def test_1d(self):
    domain, x = mesh.rectilinear([1]*2)
    for bnd, n in ('right', [1, 0]), ('left', [-1, 0]), ('top', [0, 1]), ('bottom', [0, -1]):
      self.assertEvalAlmostEqual(domain.boundary[bnd], function.normal(x), n)

  def test_2d(self):
    domain, x = mesh.rectilinear([1]*2)
    x = function.unravel(x, 0, [2, 1])
    for bnd, n in ('right', [1, 0]), ('left', [-1, 0]), ('top', [0, 1]), ('bottom', [0, -1]):
      self.assertEvalAlmostEqual(domain.boundary[bnd], function.normal(x), numpy.array(n)[:,_])

  def test_3d(self):
    domain, x = mesh.rectilinear([1]*2)
    x = function.unravel(function.unravel(x, 0, [2, 1]), 0, [1, 2])
    for bnd, n in ('right', [1, 0]), ('left', [-1, 0]), ('top', [0, 1]), ('bottom', [0, -1]):
      self.assertEvalAlmostEqual(domain.boundary[bnd], function.normal(x), numpy.array(n)[_,:,_])
