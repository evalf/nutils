import numpy, itertools, pickle, weakref, gc, warnings as _builtin_warnings, collections, sys, unittest
from nutils import *
from nutils.testing import *
_ = numpy.newaxis


@parametrize
class check(TestCase):

  def setUp(self):
    super().setUp()
    numpy.random.seed(0)
    self.arg_names = tuple(map('arg{}'.format, range(len(self.arg_values))))
    self.args = tuple(evaluable.Argument(name, value.shape, value.dtype) for name, value in zip(self.arg_names, self.arg_values))
    self.actual = self.op(*self.args)
    self.desired = self.n_op(*self.arg_values)
    assert numpy.isfinite(self.desired).all(), 'something is wrong with the design of this unit test'
    self.other = numpy.random.normal(size=self.desired.shape)
    self.pairs = [(i, j) for i in range(self.actual.ndim-1) for j in range(i+1, self.actual.ndim) if self.actual.shape[i] == self.actual.shape[j]]
    _builtin_warnings.simplefilter('ignore', evaluable.ExpensiveEvaluationWarning)

  def test_dtype(self):
    self.assertEqual(self.desired.dtype, self.actual.dtype)

  def test_shapes(self):
    self.assertEqual(self.desired.shape, tuple(n.__index__() for n in self.actual.shape))

  def assertArrayAlmostEqual(self, actual, desired, decimal):
    if actual.shape != desired.shape:
      self.fail('shapes of actual {} and desired {} are incompatible.'.format(actual.shape, desired.shape))
    error = actual - desired if not actual.dtype.kind == desired.dtype.kind == 'b' else actual ^ desired
    approx = error.dtype.kind in 'fc'
    indices = tuple(map(tuple, numpy.argwhere(numpy.greater_equal(abs(error), 1.5 * 10**-decimal) if approx else error)))
    if not len(indices):
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
    evalargs = dict(zip(self.arg_names, self.arg_values))
    with self.subTest('vanilla'):
      self.assertArrayAlmostEqual(actual.eval(**evalargs), desired, decimal)
    with self.subTest('simplified'):
      self.assertArrayAlmostEqual(actual.simplified.eval(**evalargs), desired, decimal)
    with self.subTest('optimized'):
      self.assertArrayAlmostEqual(actual.optimized_for_numpy.eval(**evalargs), desired, decimal)
    with self.subTest('sparse'):
      indices, values, shape = sparse.extract(actual.assparse.eval(**evalargs))
      self.assertEqual(tuple(map(int, shape)), desired.shape)
      if not indices:
        dense = values.sum()
      else:
        dense = numpy.zeros(desired.shape, values.dtype)
        numpy.add.at(dense, indices, values)
      self.assertArrayAlmostEqual(dense, desired, decimal)

  def test_str(self):
    a = evaluable.Array((), shape=(2, 3), dtype=float)
    self.assertEqual(str(a),'nutils.evaluable.Array<f:2,3>')

  def test_evalconst(self):
    self.assertFunctionAlmostEqual(decimal=14,
      desired=self.n_op(*self.arg_values),
      actual=self.op(*self.arg_values))

  def test_evalzero(self):
    for iarg, arg_value in enumerate(self.arg_values):
      if 0 in arg_value.flat:
        args = (*self.arg_values[:iarg], numpy.zeros_like(arg_value), *self.arg_values[iarg+1:])
        self.assertFunctionAlmostEqual(decimal=14,
          desired=self.n_op(*args),
          actual=self.op(*args))

  def test_eval(self):
    self.assertFunctionAlmostEqual(decimal=14,
      actual=self.actual,
      desired=self.desired)

  @unittest.skipIf(sys.version_info < (3,7), 'time.perf_counter_ns is not available')
  def test_eval_withtimes(self):
    evalargs = dict(zip(self.arg_names, self.arg_values))
    without_times = self.actual.eval(**evalargs)
    stats = collections.defaultdict(evaluable._Stats)
    with_times = self.actual.eval_withtimes(stats, **evalargs)
    self.assertArrayAlmostEqual(with_times, without_times, 15)
    self.assertIn(self.actual, stats)

  def test_getitem(self):
    for idim in range(self.actual.ndim):
      for item in range(self.desired.shape[idim]):
        s = (Ellipsis,) + (slice(None),)*idim + (item,) + (slice(None),)*(self.actual.ndim-idim-1)
        self.assertFunctionAlmostEqual(decimal=14,
          desired=self.desired[s],
          actual=self.actual[s])

  def test_transpose(self):
    trans = numpy.arange(self.actual.ndim,0,-1) % self.actual.ndim
    self.assertFunctionAlmostEqual(decimal=14,
      desired=numpy.transpose(self.desired, trans),
      actual=evaluable.transpose(self.actual, trans))

  def test_insertaxis(self):
    for axis in range(self.actual.ndim+1):
      with self.subTest(axis=axis):
        self.assertFunctionAlmostEqual(decimal=14,
          desired=numpy.repeat(numpy.expand_dims(self.desired, axis), 2, axis),
          actual=evaluable.insertaxis(self.actual, axis, 2))

  def test_takediag(self):
    for ax1, ax2 in self.pairs:
      self.assertFunctionAlmostEqual(decimal=14,
        desired=numeric.takediag(self.desired, ax1, ax2),
        actual=evaluable.takediag(self.actual, ax1, ax2))

  def test_eig(self):
    if self.actual.dtype == float:
      for ax1, ax2 in self.pairs:
        A, L, V = evaluable.Tuple((self.actual, *evaluable.eig(self.actual, axes=(ax1,ax2)))).eval(**dict(zip(self.arg_names, self.arg_values)))
        self.assertArrayAlmostEqual(decimal=11,
          actual=(numpy.expand_dims(V,ax2) * numpy.expand_dims(L,ax2+1).swapaxes(ax1,ax2+1)).sum(ax2+1),
          desired=(numpy.expand_dims(A,ax2) * numpy.expand_dims(V,ax2+1).swapaxes(ax1,ax2+1)).sum(ax2+1))

  def test_inv(self):
    for ax1, ax2 in self.pairs:
      trans = [i for i in range(self.desired.ndim) if i not in (ax1,ax2)] + [ax1,ax2]
      invtrans = list(map(trans.index, range(len(trans))))
      self.assertFunctionAlmostEqual(decimal=10,
        desired=numeric.inv(self.desired.transpose(trans)).transpose(invtrans),
        actual=evaluable.inverse(self.actual, axes=(ax1,ax2)))

  def test_determinant(self):
    for ax1, ax2 in self.pairs:
      self.assertFunctionAlmostEqual(decimal=11,
        desired=numpy.linalg.det(self.desired.transpose([i for i in range(self.desired.ndim) if i not in (ax1,ax2)] + [ax1,ax2])),
        actual=evaluable.determinant(self.actual, axes=(ax1,ax2)))

  def test_take(self):
    indices = [0,-1]
    for iax, sh in enumerate(self.desired.shape):
      if sh >= 2:
        self.assertFunctionAlmostEqual(decimal=14,
          desired=numpy.take(self.desired, indices, axis=iax),
          actual=evaluable.take(self.actual, indices, axis=iax))

  def test_take_block(self):
    for iax, sh in enumerate(self.desired.shape):
      if sh >= 2:
        indices = [[0,sh-1],[sh-1,0]]
        self.assertFunctionAlmostEqual(decimal=14,
          desired=numpy.take(self.desired, indices, axis=iax),
          actual=evaluable._take(self.actual, indices, axis=iax))

  def test_take_nomask(self):
    for iax, sh in enumerate(self.desired.shape):
      if sh >= 2:
        indices = [0,sh-1]
        self.assertFunctionAlmostEqual(decimal=14,
          desired=numpy.take(self.desired, indices, axis=iax),
          actual=evaluable.take(self.actual, evaluable.Guard(evaluable.asarray(indices)), axis=iax))

  def test_take_reversed(self):
    indices = [-1,0]
    for iax, sh in enumerate(self.desired.shape):
      if sh >= 2:
        self.assertFunctionAlmostEqual(decimal=14,
          desired=numpy.take(self.desired, indices, axis=iax),
          actual=evaluable.take(self.actual, indices, axis=iax))

  def test_take_duplicate_indices(self):
    for iax, sh in enumerate(self.desired.shape):
      if sh >= 2:
        indices = [0,sh-1,0,0]
        self.assertFunctionAlmostEqual(decimal=14,
          desired=numpy.take(self.desired, indices, axis=iax),
          actual=evaluable.take(self.actual, evaluable.Guard(evaluable.asarray(indices)), axis=iax))

  def test_inflate(self):
    for iax, sh in enumerate(self.desired.shape):
      dofmap = evaluable.Constant(numpy.arange(int(sh)) * 2)
      desired = numpy.zeros(self.desired.shape[:iax] + (int(sh)*2-1,) + self.desired.shape[iax+1:], dtype=self.desired.dtype)
      desired[(slice(None),)*iax+(slice(None,None,2),)] = self.desired
      self.assertFunctionAlmostEqual(decimal=14,
        desired=desired,
        actual=evaluable._inflate(self.actual, dofmap=dofmap, length=sh*2-1, axis=iax))

  def test_inflate_duplicate_indices(self):
    for iax, sh in enumerate(self.desired.shape):
      dofmap = numpy.arange(sh) % 2
      desired = numpy.zeros(self.desired.shape[:iax] + (2,) + self.desired.shape[iax+1:], dtype=self.desired.dtype)
      numpy.add.at(desired, (slice(None),)*iax+(dofmap,), self.desired)
      self.assertFunctionAlmostEqual(decimal=14,
        desired=desired,
        actual=evaluable._inflate(self.actual, dofmap=dofmap, length=2, axis=iax))

  def test_diagonalize(self):
    for axis in range(self.actual.ndim):
      for newaxis in range(axis+1, self.actual.ndim+1):
        self.assertFunctionAlmostEqual(decimal=14,
          desired=numeric.diagonalize(self.desired, axis, newaxis),
          actual=evaluable.diagonalize(self.actual, axis, newaxis))

  def test_product(self):
    if self.desired.dtype == bool:
      return
    for iax in range(self.actual.ndim):
      self.assertFunctionAlmostEqual(decimal=14,
        desired=numpy.product(self.desired, axis=iax),
        actual=evaluable.product(self.actual, axis=iax))

  def test_getslice(self):
    for idim in range(self.actual.ndim):
      if self.desired.shape[idim] == 1:
        continue
      s = (Ellipsis,) + (slice(None),)*idim + (slice(0,int(self.desired.shape[idim])-1),) + (slice(None),)*(self.actual.ndim-idim-1)
      self.assertFunctionAlmostEqual(decimal=14,
        desired=self.desired[s],
        actual=self.actual[s])

  def test_sumaxis(self):
    if self.desired.dtype == bool:
      return
    for idim in range(self.actual.ndim):
      self.assertFunctionAlmostEqual(decimal=14,
        desired=self.desired.sum(idim),
        actual=self.actual.sum(idim))

  def test_add(self):
    self.assertFunctionAlmostEqual(decimal=14,
      desired=self.desired + self.other,
      actual=(self.actual + self.other))

  def test_multiply(self):
    self.assertFunctionAlmostEqual(decimal=14,
      desired=self.desired * self.other,
      actual=(self.actual * self.other))

  def test_dot(self):
    for iax in range(self.actual.ndim):
      self.assertFunctionAlmostEqual(decimal=13,
        desired=numeric.contract(self.desired, self.other, axis=iax),
        actual=evaluable.dot(self.actual, self.other, axes=iax))

  def test_pointwise(self):
    self.assertFunctionAlmostEqual(decimal=14,
      desired=numpy.sin(self.desired).astype(float), # "astype" necessary for boolean operations (float16->float64)
      actual=evaluable.sin(self.actual))

  def test_power(self):
    self.assertFunctionAlmostEqual(decimal=13,
      desired=self.desired**3,
      actual=(self.actual**3))

  def test_power0(self):
    power = (numpy.arange(self.desired.size) % 2).reshape(self.desired.shape)
    self.assertFunctionAlmostEqual(decimal=13,
      desired=self.desired**power,
      actual=self.actual**power)

  def test_sign(self):
    if self.desired.dtype.kind != 'b':
      self.assertFunctionAlmostEqual(decimal=14,
        desired=numpy.sign(self.desired),
        actual=evaluable.sign(self.actual))

  def test_mask(self):
    for idim in range(self.actual.ndim):
      if self.desired.shape[idim] <= 1:
        continue
      mask = numpy.ones(self.desired.shape[idim], dtype=bool)
      mask[0] = False
      if self.desired.shape[idim] > 2:
        mask[-1] = False
      self.assertFunctionAlmostEqual(decimal=14,
        desired=self.desired[(slice(None,),)*idim+(mask,)],
        actual=evaluable.mask(self.actual, mask, axis=idim))

  def test_ravel(self):
    for idim in range(self.actual.ndim-1):
      self.assertFunctionAlmostEqual(decimal=14,
        desired=self.desired.reshape(self.desired.shape[:idim]+(-1,)+self.desired.shape[idim+2:]),
        actual=evaluable.ravel(self.actual, axis=idim))

  def test_unravel(self):
    for idim in range(self.actual.ndim):
      length = self.desired.shape[idim]
      unravelshape = (length//3,3) if (length%3==0) else (length//2,2) if (length%2==0) else (length,1)
      self.assertFunctionAlmostEqual(decimal=14,
        desired=self.desired.reshape(self.desired.shape[:idim]+unravelshape+self.desired.shape[idim+1:]),
        actual=evaluable.unravel(self.actual, axis=idim, shape=unravelshape))

  def test_loopsum(self):
    if self.desired.dtype == bool:
      return
    length = 3
    index = evaluable.loop_index('_testindex', length)
    for iarg, arg_value in enumerate(self.arg_values):
      testvalue = numpy.repeat(arg_value[numpy.newaxis], length, axis=0)
      numpy.random.shuffle(testvalue.ravel())
      desired = functools.reduce(operator.add, (self.n_op(*self.arg_values[:iarg], v, *self.arg_values[iarg+1:]) for v in testvalue))
      args = (*self.args[:iarg], evaluable.Guard(evaluable.get(evaluable.asarray(testvalue), 0, index)), *self.args[iarg+1:])
      self.assertFunctionAlmostEqual(decimal=14,
        actual=evaluable.loop_sum(self.op(*args), index),
        desired=desired)

  def test_loopconcatenate(self):
    length = 3
    index = evaluable.loop_index('_testindex', length)
    for iarg, arg_value in enumerate(self.arg_values):
      testvalue = numpy.repeat(arg_value[numpy.newaxis], length, axis=0)
      numpy.random.shuffle(testvalue.ravel())
      desired = numpy.concatenate([self.n_op(*self.arg_values[:iarg], v, *self.arg_values[iarg+1:]) for v in testvalue], axis=-1)
      args = (*self.args[:iarg], evaluable.Guard(evaluable.get(evaluable.asarray(testvalue), 0, index)), *self.args[iarg+1:])
      self.assertFunctionAlmostEqual(decimal=14,
        actual=evaluable.loop_concatenate(self.op(*args), index),
        desired=desired)

  @parametrize.enable_if(lambda hasgrad, **kwargs: hasgrad)
  def test_derivative(self):
    eps = 1e-4
    fddeltas = numpy.array([1,2,3])
    fdfactors = numpy.linalg.solve(2*fddeltas**numpy.arange(1,1+2*len(fddeltas),2)[:,None], [1]+[0]*(len(fddeltas)-1))
    for arg, arg_name, x0 in zip(self.args, self.arg_names, self.arg_values):
      if arg.dtype != float:
        continue
      with self.subTest(arg_name):
        dx = numpy.random.normal(size=x0.shape)
        evalargs = dict(zip(self.arg_names, self.arg_values))
        f0 = evaluable.derivative(self.actual, arg).eval(**evalargs)
        exact = numeric.contract(f0, dx, range(self.actual.ndim, self.actual.ndim+dx.ndim))
        if exact.dtype.kind in 'bi' or self.zerograd:
          approx = numpy.zeros_like(exact)
          scale = 1
        else:
          fdvals = numpy.stack([self.actual.eval(**collections.ChainMap({arg_name: numpy.asarray(x0+eps*n*dx)}, evalargs)) for n in (*-fddeltas, *fddeltas)], axis=0)
          if fdvals.dtype.kind == 'i':
            fdvals = fdvals.astype(float)
          fdvals = fdvals.reshape(2, len(fddeltas), *fdvals.shape[1:])
          approx = ((fdvals[1] - fdvals[0]).T @ fdfactors).T / eps
          scale = numpy.linalg.norm(f0.ravel()) or 1
        self.assertArrayAlmostEqual(exact / scale, approx / scale, decimal=10)

  @unittest.skipIf(sys.version_info < (3,7), 'time.perf_counter_ns is not available')
  def test_node(self):
    # This tests only whether `Evaluable._node` returns without exception.
    cache = {}
    times = collections.defaultdict(evaluable._Stats)
    with self.subTest('new'):
      node = self.actual._node(cache, None, times)
      if node:
        self.assertIn(self.actual, cache)
        self.assertEqual(cache[self.actual], node)
    with self.subTest('from-cache'):
      if node:
        self.assertEqual(self.actual._node(cache, None, times), node)
    with self.subTest('with-times'):
      times = collections.defaultdict(evaluable._Stats)
      self.actual.eval_withtimes(times, **dict(zip(self.arg_names, self.arg_values)))
      self.actual._node(cache, None, times)

def generate(*shape, real, zero, negative):
  'generate array values that cover certain numerical classes'
  size = numpy.prod(shape, dtype=int)
  a = numpy.arange(size)
  if negative:
    iz = size // 2
    a -= iz
  else:
    iz = 0
  assert a[iz] == 0
  if not zero:
    a[iz:] += 1
  if not a[-1]: # no positive numbers
    raise Exception('shape is too small to test at least one of all selected number categories')
  if real:
    a = numpy.tanh(2 * a / a[-1]) # map to (-1,1)
  return a.reshape(shape)

INT = functools.partial(generate, real=False, zero=True, negative=False)
ANY = functools.partial(generate, real=True, zero=True, negative=True)
NZ  = functools.partial(generate, real=True, zero=False, negative=True)
POS = functools.partial(generate, real=True, zero=False, negative=False)
NN  = functools.partial(generate, real=True, zero=True, negative=False)

def _check(name, op, n_op, *arg_values, hasgrad=True, zerograd=False, ndim=2):
  check(name, op=op, n_op=n_op, arg_values=arg_values, hasgrad=hasgrad, zerograd=zerograd, ndim=ndim)

_check('identity', lambda f: evaluable.asarray(f), lambda a: a, ANY(2,4,2))
_check('int', lambda f: evaluable.astype(f, int), lambda a: a.astype(int), INT(2,4,2))
_check('float', lambda f: evaluable.astype(f, float), lambda a: a.astype(float), INT(2,4,2))
_check('complex', lambda f: evaluable.astype(f, complex), lambda a: a.astype(complex), ANY(2,4,2))
_check('const', lambda f: evaluable.asarray(numpy.arange(16, dtype=float).reshape(2,4,2)), lambda a: numpy.arange(16, dtype=float).reshape(2,4,2), ANY(2,4,2))
_check('zeros', lambda f: evaluable.zeros([1,4,3,4]), lambda a: numpy.zeros([1,4,3,4]), ANY(4,3,4))
_check('ones', lambda f: evaluable.ones([1,4,3,4]), lambda a: numpy.ones([1,4,3,4]), ANY(4,3,4))
_check('range', lambda f: evaluable.Range(4) + 2, lambda a: numpy.arange(2,6), ANY(4))
_check('sin', evaluable.sin, numpy.sin, ANY(4,4))
_check('cos', evaluable.cos, numpy.cos, ANY(4,4))
_check('tan', evaluable.tan, numpy.tan, ANY(4,4))
_check('sqrt', evaluable.sqrt, numpy.sqrt, NN(4,4))
_check('log', evaluable.ln, numpy.log, POS(2,2))
_check('log2', evaluable.log2, numpy.log2, POS(2,2))
_check('log10', evaluable.log10, numpy.log10, POS(2,2))
_check('exp', evaluable.exp, numpy.exp, ANY(4,4))
_check('arctanh', evaluable.arctanh, numpy.arctanh, ANY(3,3))
_check('tanh', evaluable.tanh, numpy.tanh, ANY(4,4))
_check('cosh', evaluable.cosh, numpy.cosh, ANY(4,4))
_check('sinh', evaluable.sinh, numpy.sinh, ANY(4,4))
_check('abs', evaluable.abs, numpy.abs, ANY(4,4))
_check('sign', evaluable.sign, numpy.sign, ANY(4,4), zerograd=True)
_check('power', evaluable.power, numpy.power, POS(4,4), ANY(4,4))
_check('negative', evaluable.negative, numpy.negative, ANY(4,4))
_check('reciprocal', evaluable.reciprocal, numpy.reciprocal, NZ(4,4))
_check('arcsin', evaluable.arcsin, numpy.arcsin, ANY(4,4))
_check('arccos', evaluable.arccos, numpy.arccos, ANY(4,4))
_check('arctan', evaluable.arctan, numpy.arctan, ANY(4,4))
_check('ln', evaluable.ln, numpy.log, POS(4,4))
_check('product', lambda a: evaluable.product(a,2), lambda a: numpy.product(a,2), ANY(4,3,4))
_check('sum', lambda a: evaluable.sum(a,2), lambda a: a.sum(2), ANY(4,3,4))
_check('transpose1', lambda a: evaluable.transpose(a,[0,1,3,2]), lambda a: a.transpose([0,1,3,2]), ANY(2,3,4,5))
_check('transpose2', lambda a: evaluable.transpose(a,[0,2,3,1]), lambda a: a.transpose([0,2,3,1]), ANY(2,3,4,5))
_check('insertaxis', lambda a: evaluable.insertaxis(a,1,3), lambda a: numpy.repeat(a[:,None], 3, 1), ANY(2,4))
_check('get', lambda a: evaluable.get(a,2,1), lambda a: a[:,:,1], ANY(4,3,4))
_check('takediag141', lambda a: evaluable.takediag(a,0,2), lambda a: numeric.takediag(a,0,2), ANY(1,4,1))
_check('takediag434', lambda a: evaluable.takediag(a,0,2), lambda a: numeric.takediag(a,0,2), ANY(4,3,4))
_check('takediag343', lambda a: evaluable.takediag(a,0,2), lambda a: numeric.takediag(a,0,2), ANY(3,4,3))
_check('determinant141', lambda a: evaluable.determinant(a,(0,2)), lambda a: numpy.linalg.det(a.swapaxes(0,1)), ANY(1,4,1))
_check('determinant434', lambda a: evaluable.determinant(a,(0,2)), lambda a: numpy.linalg.det(a.swapaxes(0,1)), ANY(4,3,4))
_check('determinant4433', lambda a: evaluable.determinant(a,(2,3)), lambda a: numpy.linalg.det(a), ANY(4,4,3,3))
_check('determinant200', lambda a: evaluable.determinant(a,(1,2)), lambda a: numpy.linalg.det(a) if a.shape[-1] else numpy.ones(a.shape[:-2], float), numpy.empty((2,0,0)), zerograd=True)
_check('inverse141', lambda a: evaluable.inverse(a,(0,2)), lambda a: numpy.linalg.inv(a.swapaxes(0,1)).swapaxes(0,1), NZ(1,4,1))
_check('inverse434', lambda a: evaluable.inverse(a,(0,2)), lambda a: numpy.linalg.inv(a.swapaxes(0,1)).swapaxes(0,1), POS(4,3,4)+numpy.eye(4,4)[:,numpy.newaxis,:])
_check('inverse4422', lambda a: evaluable.inverse(a), lambda a: numpy.linalg.inv(a), POS(4,4,2,2)+numpy.eye(2))
_check('repeat', lambda a: evaluable.repeat(a,3,1), lambda a: numpy.repeat(a,3,1), ANY(4,1,4))
_check('diagonalize', lambda a: evaluable.diagonalize(a,1,3), lambda a: numeric.diagonalize(a,1,3), ANY(4,4,4,4))
_check('multiply', evaluable.multiply, numpy.multiply, ANY(4,4), ANY(4,4))
_check('dot', lambda a,b: evaluable.dot(a,b,axes=1), lambda a,b: (a*b).sum(1), ANY(4,2,4), ANY(4,2,4))
_check('divide', evaluable.divide, lambda a, b: a * b**-1, ANY(4,4), NZ(4,4))
_check('divide2', lambda a: evaluable.asarray(a)/2, lambda a: a/2, ANY(4,1))
_check('add', evaluable.add, numpy.add, ANY(4,4), ANY(4,4))
_check('subtract', evaluable.subtract, numpy.subtract, ANY(4,4), ANY(4,4))
_check('dot2', lambda a,b: evaluable.multiply(a,b).sum(-2), lambda a,b: (a*b).sum(-2), ANY(4,2,4), ANY(4,2,4))
_check('min', lambda a,b: evaluable.Minimum(a,b), numpy.minimum, ANY(4,4), ANY(4,4))
_check('max', lambda a,b: evaluable.Maximum(a,b), numpy.maximum, ANY(4,4), ANY(4,4))
_check('equal', evaluable.Equal, numpy.equal, ANY(4,4), ANY(4,4), zerograd=True)
_check('greater', evaluable.Greater, numpy.greater, ANY(4,4), ANY(4,4), zerograd=True)
_check('less', evaluable.Less, numpy.less, ANY(4,4), ANY(4,4), zerograd=True)
_check('arctan2', evaluable.arctan2, numpy.arctan2, ANY(4,4), ANY(4,4))
_check('stack', lambda a,b: evaluable.stack([a,b], 0), lambda a,b: numpy.concatenate([a[_,:],b[_,:]], axis=0), ANY(4), ANY(4))
_check('eig', lambda a: evaluable.eig(a+a.swapaxes(0,1),symmetric=True)[1], lambda a: numpy.linalg.eigh(a+a.swapaxes(0,1))[1], ANY(4,4), hasgrad=False)
_check('mod', lambda a,b: evaluable.mod(a,b), lambda a,b: numpy.mod(a,b), ANY(4), NZ(4), hasgrad=False)
_check('mask', lambda f: evaluable.mask(f,numpy.array([True,False,True,False,True,False,True]),axis=1), lambda a: a[:,::2], ANY(4,7,4))
_check('ravel', lambda f: evaluable.ravel(f,axis=1), lambda a: a.reshape(4,4,4,4), ANY(4,2,2,4,4))
_check('unravel', lambda f: evaluable.unravel(f,axis=1,shape=[2,2]), lambda a: a.reshape(4,2,2,4,4), ANY(4,4,4,4))
_check('ravelindex', lambda a, b: evaluable.RavelIndex(a, b, 12, 20), lambda a, b: a[...,_,_] * 20 + b, INT(3,4), INT(4,5))
_check('inflate', lambda f: evaluable._inflate(f,dofmap=evaluable.Guard([0,3]),length=4,axis=1), lambda a: numpy.concatenate([a[:,:1], numpy.zeros_like(a), a[:,1:]], axis=1), ANY(4,2,4))
_check('inflate-constant', lambda f: evaluable._inflate(f,dofmap=[0,3],length=4,axis=1), lambda a: numpy.concatenate([a[:,:1], numpy.zeros_like(a), a[:,1:]], axis=1), ANY(4,2,4))
_check('inflate-duplicate', lambda f: evaluable.Inflate(f,dofmap=[0,1,0,3],length=4), lambda a: numpy.stack([a[:,0]+a[:,2], a[:,1], numpy.zeros_like(a[:,0]), a[:,3]], axis=1), ANY(2,4))
_check('inflate-block', lambda f: evaluable.Inflate(f,dofmap=[[5,4,3],[2,1,0]],length=6), lambda a: a.ravel()[::-1], ANY(2,3))
_check('inflate-scalar', lambda f: evaluable.Inflate(f,dofmap=1,length=3), lambda a: numpy.array([0,a,0]), numpy.array(.5))
_check('inflate-diagonal', lambda f: evaluable.Inflate(evaluable.Inflate(f,1,3),1,3), lambda a: numpy.diag(numpy.array([0,a,0])), numpy.array(.5))
_check('inflate-one', lambda f: evaluable.Inflate(f,0,1), lambda a: numpy.array([a]), numpy.array(.5))
_check('inflate-range', lambda f: evaluable.Inflate(f,evaluable.Range(3),3), lambda a: a, ANY(3))
_check('take', lambda f: evaluable.Take(f, [0,3,2]), lambda a: a[:,[0,3,2]], ANY(2,4))
_check('take-duplicate', lambda f: evaluable.Take(f, [0,3,0]), lambda a: a[:,[0,3,0]], ANY(2,4))
_check('choose', lambda a, b, c: evaluable.Choose(a%2, [b,c]), lambda a, b, c: numpy.choose(a%2, [b,c]), INT(3,3), ANY(3,3), ANY(3,3))
_check('slice', lambda a: evaluable.asarray(a)[::2], lambda a: a[::2], ANY(5,3))
_check('normal1d', lambda a: evaluable.Normal(a), lambda a: numpy.sign(a[...,0]), NZ(3,1,1))
_check('normal2d', lambda a: evaluable.Normal(a), lambda a: numpy.stack([Q[:,-1]*numpy.sign(R[-1,-1]) for ai in a for Q, R in [numpy.linalg.qr(ai, mode='complete')]], axis=0), POS(1,2,2)+numpy.eye(2))
_check('normal3d', lambda a: evaluable.Normal(a), lambda a: numpy.stack([Q[:,-1]*numpy.sign(R[-1,-1]) for ai in a for Q, R in [numpy.linalg.qr(ai, mode='complete')]], axis=0), POS(2,3,3)+numpy.eye(3))
_check('loopsum1', lambda: evaluable.loop_sum(evaluable.loop_index('index', 3), evaluable.loop_index('index', 3)), lambda: numpy.array(3))
_check('loopsum2', lambda a: evaluable.loop_sum(a, evaluable.loop_index('index', 2)), lambda a: 2*a, ANY(3,4,2,4))
_check('loopsum3', lambda a: evaluable.loop_sum(evaluable.get(a, 0, evaluable.loop_index('index', 3)), evaluable.loop_index('index', 3)), lambda a: numpy.sum(a, 0), ANY(3,4,2,4))
_check('loopsum4', lambda: evaluable.loop_sum(evaluable.Inflate(evaluable.loop_index('index', 3), 0, 2), evaluable.loop_index('index', 3)), lambda: numpy.array([3, 0]))
_check('loopsum5', lambda: evaluable.loop_sum(evaluable.loop_index('index', 1), evaluable.loop_index('index', 1)), lambda: numpy.array(0))
_check('loopconcatenate1', lambda a: evaluable.loop_concatenate(a+evaluable.prependaxes(evaluable.loop_index('index', 3), a.shape), evaluable.loop_index('index', 3)), lambda a: a+numpy.arange(3)[None], ANY(3,1))
_check('loopconcatenate2', lambda: evaluable.loop_concatenate(evaluable.Elemwise([numpy.arange(48).reshape(4,4,3)[:,:,a:b] for a, b in util.pairwise([0,2,3])], evaluable.loop_index('index', 2), int), evaluable.loop_index('index', 2)), lambda: numpy.arange(48).reshape(4,4,3))
_check('loopconcatenatecombined', lambda a: evaluable.loop_concatenate_combined([a+evaluable.prependaxes(evaluable.loop_index('index', 3), a.shape)], evaluable.loop_index('index', 3))[0], lambda a: a+numpy.arange(3)[None], ANY(3,1), hasgrad=False)
_check('legendre', lambda a: evaluable.Legendre(evaluable.asarray(a), 5), lambda a: numpy.moveaxis(numpy.polynomial.legendre.legval(a, numpy.eye(6)), 0, -1), ANY(3,4,3))

_polyval_mask = lambda shape, ndim: 1 if ndim == 0 else numpy.array([sum(i[-ndim:]) < int(shape[-1]) for i in numpy.ndindex(shape)], dtype=int).reshape(shape)
_polyval_desired = lambda c, x: sum(c[(...,*i)]*(x[(slice(None),*[None]*(c.ndim-x.shape[1]))]**i).prod(-1) for i in itertools.product(*[range(c.shape[-1])]*x.shape[1]) if sum(i) < c.shape[-1])
_check('polyval_1d_p0', lambda c, x: evaluable.Polyval(c*_polyval_mask(c.shape,x.shape[1]), x), _polyval_desired, POS(1), ANY(4,1), ndim=1)
_check('polyval_1d_p1', lambda c, x: evaluable.Polyval(c*_polyval_mask(c.shape,x.shape[1]), x), _polyval_desired, NZ(2), ANY(4,1), ndim=1)
_check('polyval_1d_p2', lambda c, x: evaluable.Polyval(c*_polyval_mask(c.shape,x.shape[1]), x), _polyval_desired, ANY(3), ANY(4,1), ndim=1)
_check('polyval_2d_p0', lambda c, x: evaluable.Polyval(c*_polyval_mask(c.shape,x.shape[1]), x), _polyval_desired, POS(1,1), ANY(4,2), ndim=2)
_check('polyval_2d_p1', lambda c, x: evaluable.Polyval(c*_polyval_mask(c.shape,x.shape[1]), x), _polyval_desired, ANY(2,2), ANY(4,2), ndim=2)
_check('polyval_2d_p2', lambda c, x: evaluable.Polyval(c*_polyval_mask(c.shape,x.shape[1]), x), _polyval_desired, ANY(3,3), ANY(4,2), ndim=2)
_check('polyval_2d_p1_23', lambda c, x: evaluable.Polyval(c*_polyval_mask(c.shape,x.shape[1]), x), _polyval_desired, ANY(2,3,2,2), ANY(4,2), ndim=2)

class intbounds(TestCase):

  @staticmethod
  def R(start, shape):
    # A range of numbers starting at `start` with the given `shape`.
    if isinstance(shape, int):
      size = shape
      shape = shape,
    else:
      size = util.product(shape)
    return evaluable.Constant(numpy.arange(start, start+size).reshape(*shape))

  class S(evaluable.Array):
    # An evaluable scalar argument with given bounds.
    def __init__(self, argname, lower, upper):
      self._argname = argname
      self._lower = lower
      self._upper = upper
      super().__init__(args=(evaluable.EVALARGS,), shape=(), dtype=int)
    def evalf(self, evalargs):
      value = numpy.array(evalargs[self._argname])
      assert self._lower <= value <= self._upper
      return numpy.array(value)
    @property
    def _intbounds(self):
      return self._lower, self._upper

  def assertBounds(self, func, *, tight_lower=True, tight_upper=True, **evalargs):
    lower, upper = func._intbounds
    value = func.eval(**evalargs)
    (self.assertEqual if tight_lower else self.assertLessEqual)(lower, value.min())
    (self.assertEqual if tight_upper else self.assertGreaterEqual)(upper, value.max())

  def test_default(self):
    class Test(evaluable.Array):
      def __init__(self):
        super().__init__(args=(evaluable.Argument('dummy', (), int),), shape=(), dtype=int)
      def evalf(self):
        raise NotImplementedError
    self.assertEqual(Test()._intbounds, (float('-inf'), float('inf')))

  def test_constant(self):
    self.assertEqual(self.R(-4,[2,3,4])._intbounds, (-4, 19))

  def test_constant_empty(self):
    self.assertEqual(self.R(0,[0])._intbounds, (float('-inf'), float('inf')))

  def test_insertaxis(self):
    arg = self.R(-4,[2,3,4])
    self.assertEqual(evaluable.InsertAxis(arg, 2)._intbounds, arg._intbounds)

  def test_transpose(self):
    arg = self.R(-4,[2,3,4])
    self.assertEqual(evaluable.Transpose(arg, (2,0,1))._intbounds, arg._intbounds)

  def test_multiply(self):
    args = tuple(self.R(low, [high+1-low]) for low, high in ((-13, -5), (-2, 7), (3, 11)))
    for arg1 in args:
      for arg2 in args:
        self.assertBounds(evaluable.Multiply((evaluable.insertaxis(arg1, 1, arg2.shape[0]), evaluable.insertaxis(arg2, 0, arg1.shape[0]))))

  def test_add(self):
    self.assertBounds(evaluable.Add((evaluable.insertaxis(self.R(-5,[8]), 1, 5), evaluable.insertaxis(self.R(2,[5]), 0, 8))))

  def test_sum_zero_axis(self):
    self.assertEqual(evaluable.Sum(self.R(0,[0]))._intbounds, (0, 0))

  def test_sum_variable_axis_including_zero(self):
    self.assertEqual(evaluable.Sum(evaluable.Argument('test', (self.S('n', 0, 4),), int))._intbounds, (float('-inf'), float('inf')))

  def test_sum_zero_size(self):
    self.assertEqual(evaluable.Sum(self.R(0,[2,3,0]))._intbounds, (0, 0))

  def test_sum_nonzero(self):
    self.assertBounds(evaluable.Sum(self.R(-3,[9,1])))

  def test_sum_unknown(self):
    func = lambda l, h: evaluable.Sum(evaluable.InsertAxis(self.R(l,[h+1-l]), self.S('n',2,5)))
    self.assertBounds(func(-3, 5), n=5)
    self.assertBounds(func(-3, 5), n=5, tight_lower=False, tight_upper=False)
    self.assertBounds(func(3, 5), n=5, tight_lower=False)
    self.assertBounds(func(3, 5), n=2, tight_upper=False)
    self.assertBounds(func(-3, -2), n=5, tight_upper=False)
    self.assertBounds(func(-3, -2), n=2, tight_lower=False)

  def test_takediag(self):
    arg = self.R(-4,[2,3,3])
    self.assertEqual(evaluable.TakeDiag(arg)._intbounds, arg._intbounds)

  def test_take(self):
    arg = self.R(-4,[2,3,4])
    idx = self.R(0,[1])
    self.assertEqual(evaluable.Take(arg, idx)._intbounds, arg._intbounds)

  def test_negative(self):
    self.assertBounds(evaluable.Negative(self.R(-4,[2,3,4])))

  def test_absolute_negative(self):
    self.assertBounds(evaluable.Absolute(self.R(-4,[3])))

  def test_absolute_positive(self):
    self.assertBounds(evaluable.Absolute(self.R(1,[3])))

  def test_absolute_full(self):
    self.assertBounds(evaluable.Absolute(self.R(-3,[7])))

  def test_mod_nowrap(self):
    self.assertBounds(evaluable.Mod(evaluable.insertaxis(self.R(1,[4]), 1, 3), evaluable.insertaxis(self.R(5,[3]), 0, 4)))

  def test_mod_wrap_negative(self):
    self.assertBounds(evaluable.Mod(evaluable.insertaxis(self.R(-3,[7]), 1, 3), evaluable.insertaxis(self.R(5,[3]), 0, 7)))

  def test_mod_wrap_positive(self):
    self.assertBounds(evaluable.Mod(evaluable.insertaxis(self.R(3,[7]), 1, 3), evaluable.insertaxis(self.R(5,[3]), 0, 7)))

  def test_mod_negative_divisor(self):
    self.assertEqual(evaluable.Mod(evaluable.Argument('d', (2,), int), self.R(-3,[2]))._intbounds, (float('-inf'), float('inf')))

  def test_sign(self):
    for i in range(-2, 3):
      for j in range(i, 3):
        self.assertBounds(evaluable.Sign(self.R(i,[j-i+1])))

  def test_zeros(self):
    self.assertEqual(evaluable.Zeros((2,3), int)._intbounds, (0, 0))

  def test_range(self):
    self.assertEqual(evaluable.Range(self.S('n', 0, 0))._intbounds, (0, 0))
    self.assertBounds(evaluable.Range(self.S('n', 1, 3)), n=3)

  def test_inrange_loose(self):
    self.assertEqual(evaluable.InRange(self.S('n', 3, 5), evaluable.Constant(6))._intbounds, (3, 5))

  def test_inrange_strict(self):
    self.assertEqual(evaluable.InRange(self.S('n', float('-inf'), float('inf')), self.S('m', 2, 4))._intbounds, (0, 3))

  def test_inrange_empty(self):
    self.assertEqual(evaluable.InRange(self.S('n', float('-inf'), float('inf')), evaluable.Constant(0))._intbounds, (0, 0))

  def test_npoints(self):
    self.assertEqual(evaluable.NPoints()._intbounds, (0, float('inf')))

  def test_bool_to_int(self):
    self.assertEqual(evaluable.BoolToInt(evaluable.Constant(numpy.array([False, True], dtype=bool)))._intbounds, (0, 1))

  def test_array_from_tuple(self):
    self.assertEqual(evaluable.ArrayFromTuple(evaluable.Tuple((evaluable.Argument('n', (3,), int),)), 0, (3,), int, _lower=-2, _upper=3)._intbounds, (-2, 3))

  def test_inflate(self):
    self.assertEqual(evaluable.Inflate(self.R(4, (2,3)), evaluable.Constant(numpy.arange(6).reshape(2,3)), 7)._intbounds, (0, 9))

  def test_normdim_positive(self):
    self.assertEqual(evaluable.NormDim(self.S('l', 2, 4), self.S('i', 1, 3))._intbounds, (1, 3))

  def test_normdim_negative(self):
    self.assertEqual(evaluable.NormDim(self.S('l', 4, 4), self.S('i', -3, -1))._intbounds, (1, 3))

  def test_normdim_mixed(self):
    self.assertEqual(evaluable.NormDim(self.S('l', 4, 5), self.S('i', -3, 2))._intbounds, (0, 4))

  def test_minimum(self):
    self.assertEqual(evaluable.Minimum(self.S('a', 0, 4), self.S('b', 1, 3))._intbounds, (0, 3))

  def test_maximum(self):
    self.assertEqual(evaluable.Maximum(self.S('a', 0, 4), self.S('b', 1, 3))._intbounds, (1, 4))

class simplifications(TestCase):

  def test_minimum_maximum_bounds(self):

    class R(evaluable.Array):
      # An evaluable scalar argument with given bounds.
      def __init__(self, lower, upper):
        self._lower = lower
        self._upper = upper
        super().__init__(args=(evaluable.EVALARGS,), shape=(), dtype=int)
      def evalf(self, evalargs):
        raise NotImplementedError
      @property
      def _intbounds(self):
        return self._lower, self._upper

    a = R(0, 2)
    b = R(2, 4)

    with self.subTest('min-left'):
      self.assertEqual(evaluable.Minimum(a, b).simplified, a)
    with self.subTest('min-right'):
      self.assertEqual(evaluable.Minimum(b, a).simplified, a)
    with self.subTest('max-left'):
      self.assertEqual(evaluable.Maximum(b, a).simplified, b)
    with self.subTest('max-right'):
      self.assertEqual(evaluable.Maximum(a, b).simplified, b)


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
    self.assertEqual(evaluable.add(self.A, self.B) * evaluable.insertaxis(evaluable.dot(self.A, self.B, axes=[0]), 0, 2),
                     evaluable.insertaxis(evaluable.dot(self.B, self.A, axes=[0]), 0, 2) * evaluable.add(self.B, self.A))


class sampled(TestCase):

  def test_match(self):
    f = evaluable.Sampled(numpy.array([[1,2], [3,4]]), numpy.array([[1,2], [3,4]]))
    self.assertAllEqual(f.eval(), numpy.eye(2))

  def test_no_match(self):
    f = evaluable.Sampled(numpy.array([[1,2], [3,4]]), numpy.array([[3,4], [1,2]]))
    with self.assertRaises(Exception):
      f.eval()


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

  def assertElemwise(self, items):
    items = tuple(map(numpy.array, items))
    index = evaluable.Argument('index', (), int)
    elemwise = evaluable.Elemwise(items, index, int)
    for i, item in enumerate(items):
      self.assertEqual(elemwise.eval(index=i).tolist(), item.tolist())

  def test_const_values(self):
    self.assertElemwise((numpy.arange(2*3*4).reshape(2,3,4),)*3)

  def test_const_shape(self):
    self.assertElemwise(numpy.arange(4*2*3*4).reshape(4,2,3,4))

  def test_mixed_shape(self):
    self.assertElemwise(numpy.arange(4*i*j*3).reshape(4,i,j,3) for i, j in ((1,2),(2,4)))

  def test_var_shape(self):
    self.assertElemwise(numpy.arange(i*j).reshape(i,j) for i, j in ((1,2),(2,4)))

class derivative(TestCase):

  def test_int(self):
    arg = evaluable.Argument('arg', (2,), int)
    self.assertEqual(evaluable.derivative(evaluable.insertaxis(arg, 0, 1), arg), evaluable.Zeros((1,2,2), int))

class asciitree(TestCase):

  @unittest.skipIf(sys.version_info < (3, 6), 'test requires dicts maintaining insertion order')
  def test_asciitree(self):
    f = evaluable.Sin((evaluable.Zeros((), int))**evaluable.Diagonalize(evaluable.Argument('arg', (2,))))
    self.assertEqual(f.asciitree(richoutput=True),
                     '%0 = Sin; f:2,2\n'
                     '└ %1 = Power; f:2,2\n'
                     '  ├ %2 = InsertAxis; f:2,2\n'
                     '  │ ├ %3 = InsertAxis; f:2\n'
                     '  │ │ ├ %4 = IntToFloat; f:\n'
                     '  │ │ │ └ 0\n'
                     '  │ │ └ 2\n'
                     '  │ └ 2\n'
                     '  └ %5 = Diagonalize; f:2,2\n'
                     '    └ Argument; arg; f:2\n')

  @unittest.skipIf(sys.version_info < (3, 6), 'test requires dicts maintaining insertion order')
  def test_loop_sum(self):
    i = evaluable.loop_index('i', 2)
    f = evaluable.loop_sum(i, i)
    self.assertEqual(f.asciitree(richoutput=True),
                     'SUBGRAPHS\n'
                     'A\n'
                     '└ B = Loop\n'
                     'NODES\n'
                     '%B0 = LoopSum\n'
                     '└ func = %B1 = LoopIndex\n'
                     '  └ length = 2\n')

  @unittest.skipIf(sys.version_info < (3, 6), 'test requires dicts maintaining insertion order')
  def test_loop_concatenate(self):
    i = evaluable.loop_index('i', 2)
    f = evaluable.loop_concatenate(evaluable.InsertAxis(i, 1), i)
    self.assertEqual(f.asciitree(richoutput=True),
                     'SUBGRAPHS\n'
                     'A\n'
                     '└ B = Loop\n'
                     'NODES\n'
                     '%B0 = LoopConcatenate\n'
                     '├ shape[0] = %A1 = Take; i:; [2,2]\n'
                     '│ ├ %A2 = _SizesToOffsets; i:3; [0,2]\n'
                     '│ │ └ %A3 = InsertAxis; i:2; [1,1]\n'
                     '│ │   ├ 1\n'
                     '│ │   └ 2\n'
                     '│ └ 2\n'
                     '├ start = %B4 = Take; i:; [0,2]\n'
                     '│ ├ %A2\n'
                     '│ └ %B5 = LoopIndex\n'
                     '│   └ length = 2\n'
                     '├ stop = %B6 = Take; i:; [0,2]\n'
                     '│ ├ %A2\n'
                     '│ └ %B7 = Add; i:; [1,2]\n'
                     '│   ├ %B5\n'
                     '│   └ 1\n'
                     '└ func = %B8 = InsertAxis; i:1; [0,1]\n'
                     '  ├ %B5\n'
                     '  └ 1\n')

  @unittest.skipIf(sys.version_info < (3, 6), 'test requires dicts maintaining insertion order')
  def test_loop_concatenatecombined(self):
    i = evaluable.loop_index('i', 2)
    f, = evaluable.loop_concatenate_combined([evaluable.InsertAxis(i, 1)], i)
    self.assertEqual(f.asciitree(richoutput=True),
                     'SUBGRAPHS\n'
                     'A\n'
                     '└ B = Loop\n'
                     'NODES\n'
                     '%B0 = LoopConcatenate\n'
                     '├ shape[0] = %A1 = Take; i:; [2,2]\n'
                     '│ ├ %A2 = _SizesToOffsets; i:3; [0,2]\n'
                     '│ │ └ %A3 = InsertAxis; i:2; [1,1]\n'
                     '│ │   ├ 1\n'
                     '│ │   └ 2\n'
                     '│ └ 2\n'
                     '├ start = %B4 = Take; i:; [0,2]\n'
                     '│ ├ %A2\n'
                     '│ └ %B5 = LoopIndex\n'
                     '│   └ length = 2\n'
                     '├ stop = %B6 = Take; i:; [0,2]\n'
                     '│ ├ %A2\n'
                     '│ └ %B7 = Add; i:; [1,2]\n'
                     '│   ├ %B5\n'
                     '│   └ 1\n'
                     '└ func = %B8 = InsertAxis; i:1; [0,1]\n'
                     '  ├ %B5\n'
                     '  └ 1\n')

class simplify(TestCase):

  def test_multiply_transpose(self):
    dummy = evaluable.Argument('dummy', shape=[2,2,2], dtype=float)
    f = evaluable.multiply(dummy,
          evaluable.Transpose(evaluable.multiply(dummy,
            evaluable.Transpose(dummy, (2,0,1))), (2,0,1)))
    # The test below is not only to verify that no simplifications are
    # performed, but also to make sure that simplified does not get stuck in a
    # circular dependence. This used to be the case prior to adding the
    # isinstance(other_trans, Transpose) restriction in Transpose._multiply.
    self.assertEqual(f.simplified, f)

  def test_add_sparse(self):
    a = evaluable.Inflate(
      func=evaluable.Argument('a', shape=[2,3,2], dtype=float),
      dofmap=evaluable.Argument('dofmap', shape=[2], dtype=int),
      length=3)
    b = evaluable.Diagonalize(
      func=evaluable.Argument('b', shape=[2,3], dtype=float))
    c = evaluable.Argument('c', shape=[2,3,3], dtype=float)
    # Since a and b are both sparse, we expect (a+b)*c to be simplified to a*c+b*c.
    self.assertIsInstance(((a + b) * c).simplified, evaluable.Add)
    # If the sparsity of the terms is equal then sparsity propagates through the addition.
    self.assertIsInstance(((a + a) * c).simplified, evaluable.Inflate)
    self.assertIsInstance(((b + b) * c).simplified, evaluable.Diagonalize)
    # If either term in the addition is dense, the original structure remains.
    self.assertIsInstance(((a + c) * c).simplified, evaluable.Multiply)
    self.assertIsInstance(((c + b) * c).simplified, evaluable.Multiply)

class memory(TestCase):

  def assertCollected(self, ref):
    gc.collect()
    if ref() is not None:
      self.fail('object was not garbage collected')

  def test_general(self):
    # NOTE: The list of numbers must be unique in the entire test suite. If
    # not, a test leaking this specific array will cause this test to fail.
    A = evaluable.Constant([1,2,3,98,513])
    A = weakref.ref(A)
    self.assertCollected(A)

  def test_simplified(self):
    # NOTE: The list of numbers must be unique in the entire test suite. If
    # not, a test leaking this specific array will cause this test to fail.
    A = evaluable.Constant([1,2,3,99,514])
    A.simplified # constant simplified to itself, which should be handled as a special case to avoid circular references
    A = weakref.ref(A)
    self.assertCollected(A)

  def test_replace(self):
    class MyException(Exception):
      pass
    class A(evaluable.Array):
      def __init__(self):
        super().__init__(args=[], shape=(), dtype=float)
      def _simplified(self):
        raise MyException
    t = evaluable.Tuple([A()])
    with self.assertRaises(MyException):
      t.simplified
    with self.assertRaises(MyException): # make sure no placeholders remain in the replacement cache
      t.simplified

class combine_loop_concatenates(TestCase):

  def test_same_index(self):
    i = evaluable.loop_index('i', 3)
    A = evaluable.LoopConcatenate((evaluable.InsertAxis(i, 1), i, i+1, 3,), i._name, i.length)
    B = evaluable.LoopConcatenate((evaluable.InsertAxis(i, 2), i*2, i*2+2, 6,), i._name, i.length)
    actual = evaluable.Tuple((A, B))._combine_loop_concatenates(set())
    L = evaluable.LoopConcatenateCombined(((evaluable.InsertAxis(i, 1), i, i+1, 3), (evaluable.InsertAxis(i, 2), i*2, i*2+2, 6)), i._name, i.length)
    desired = evaluable.Tuple((evaluable.ArrayFromTuple(L, 0, (3,), int, **dict(zip(('_lower', '_upper'), A._intbounds))), evaluable.ArrayFromTuple(L, 1, (6,), int, **dict(zip(('_lower', '_upper'), B._intbounds)))))
    self.assertEqual(actual, desired)

  def test_different_index(self):
    i = evaluable.loop_index('i', 3)
    j = evaluable.loop_index('j', 3)
    A = evaluable.LoopConcatenate((evaluable.InsertAxis(i, 1), i, i+1, 3,), i._name, i.length)
    B = evaluable.LoopConcatenate((evaluable.InsertAxis(j, 1), j, j+1, 3,), j._name, j.length)
    actual = evaluable.Tuple((A, B))._combine_loop_concatenates(set())
    L1 = evaluable.LoopConcatenateCombined(((evaluable.InsertAxis(i, 1), i, i+1, 3),), i._name, i.length)
    L2 = evaluable.LoopConcatenateCombined(((evaluable.InsertAxis(j, 1), j, j+1, 3),), j._name, j.length)
    desired = evaluable.Tuple((evaluable.ArrayFromTuple(L1, 0, (3,), int, **dict(zip(('_lower', '_upper'), A._intbounds))), evaluable.ArrayFromTuple(L2, 0, (3,), int, **dict(zip(('_lower', '_upper'), B._intbounds)))))
    self.assertEqual(actual, desired)

  def test_nested_invariant(self):
    i = evaluable.loop_index('i', 3)
    A = evaluable.LoopConcatenate((evaluable.InsertAxis(i, 1), i, i+1, 3,), i._name, i.length)
    B = evaluable.LoopConcatenate((A, i*3, i*3+3, 9,), i._name, i.length)
    actual = evaluable.Tuple((A, B))._combine_loop_concatenates(set())
    L1 = evaluable.LoopConcatenateCombined(((evaluable.InsertAxis(i, 1), i, i+1, 3),), i._name, i.length)
    A_ = evaluable.ArrayFromTuple(L1, 0, (3,), int, **dict(zip(('_lower', '_upper'), A._intbounds)))
    L2 = evaluable.LoopConcatenateCombined(((A_, i*3, i*3+3, 9),), i._name, i.length)
    self.assertIn(A_, L2._Evaluable__args)
    desired = evaluable.Tuple((A_, evaluable.ArrayFromTuple(L2, 0, (9,), int, **dict(zip(('_lower', '_upper'), B._intbounds)))))
    self.assertEqual(actual, desired)

  def test_nested_variant(self):
    i = evaluable.loop_index('i', 3)
    j = evaluable.loop_index('j', 3)
    A = evaluable.LoopConcatenate((evaluable.InsertAxis(i+j, 1), i, i+1, 3,), i._name, i.length)
    B = evaluable.LoopConcatenate((A, j*3, j*3+3, 9,), j._name, j.length)
    actual = evaluable.Tuple((A, B))._combine_loop_concatenates(set())
    L1 = evaluable.LoopConcatenateCombined(((evaluable.InsertAxis(i+j, 1), i, i+1, 3),), i._name, i.length)
    A_ = evaluable.ArrayFromTuple(L1, 0, (3,), int, **dict(zip(('_lower', '_upper'), A._intbounds)))
    L2 = evaluable.LoopConcatenateCombined(((A_, j*3, j*3+3, 9),), j._name, j.length)
    self.assertNotIn(A_, L2._Evaluable__args)
    desired = evaluable.Tuple((A_, evaluable.ArrayFromTuple(L2, 0, (9,), int, **dict(zip(('_lower', '_upper'), B._intbounds)))))
    self.assertEqual(actual, desired)

class EvaluableConstant(TestCase):

  def test_evalf(self):
    self.assertEqual(evaluable.EvaluableConstant(1).evalf(), 1)
    self.assertEqual(evaluable.EvaluableConstant('1').evalf(), '1')

  def test_node_details(self):

    class Test:
      def __init__(self, s):
        self.s = s
      def __repr__(self):
        return self.s

    self.assertEqual(evaluable.EvaluableConstant(Test('some string'))._node_details, 'some string')
    self.assertEqual(evaluable.EvaluableConstant(Test('a very long string that should be abbreviated'))._node_details, 'a very long strin...')
    self.assertEqual(evaluable.EvaluableConstant(Test('a string with\nmultiple lines'))._node_details, 'a string with...')
    self.assertEqual(evaluable.EvaluableConstant(Test('a very long string with\nmultiple lines'))._node_details, 'a very long strin...')

class Einsum(TestCase):

  def test_swapaxes(self):
    arg = numpy.arange(6).reshape(2,3)
    ret = evaluable.einsum('ij->ji', evaluable.asarray(arg))
    self.assertAllEqual(ret.eval(), arg.T)

  def test_rollaxes(self):
    arg = numpy.arange(6).reshape(1,2,3)
    ret = evaluable.einsum('Ai->iA', evaluable.asarray(arg))
    self.assertAllEqual(ret.eval(), arg.transpose([2,0,1]))

  def test_swapgroups(self):
    arg = numpy.arange(24).reshape(1,2,3,4)
    ret = evaluable.einsum('AB->BA', evaluable.asarray(arg), B=2)
    self.assertAllEqual(ret.eval(), arg.transpose([2,3,0,1]))

  def test_matvec(self):
    arg1 = numpy.arange(6).reshape(2,3)
    arg2 = numpy.arange(6).reshape(3,2)
    ret = evaluable.einsum('ij,jk->ik', evaluable.asarray(arg1), evaluable.asarray(arg2))
    self.assertAllEqual(ret.eval(), arg1 @ arg2)

  def test_multidot(self):
    arg1 = numpy.arange(6).reshape(2,3)
    arg2 = numpy.arange(9).reshape(3,3)
    arg3 = numpy.arange(6).reshape(3,2)
    ret = evaluable.einsum('ij,jk,kl->il', evaluable.asarray(arg1), evaluable.asarray(arg2), evaluable.asarray(arg3))
    self.assertAllEqual(ret.eval(), arg1 @ arg2 @ arg3)

  def test_wrong_args(self):
    arg = numpy.arange(6).reshape(2,3)
    with self.assertRaisesRegex(ValueError, 'number of arguments does not match format string'):
      evaluable.einsum('ij,jk->ik', arg)

  def test_wrong_ellipse(self):
    arg = numpy.arange(6)
    with self.assertRaisesRegex(ValueError, 'argument dimensions are inconsistent with format string'):
      evaluable.einsum('iAj->jAi', arg)

  def test_wrong_dimension(self):
    arg = numpy.arange(9).reshape(3,3)
    with self.assertRaisesRegex(ValueError, 'argument dimensions are inconsistent with format string'):
      evaluable.einsum('ijk->kji', arg)

  def test_wrong_multi_ellipse(self):
    arg = numpy.arange(6)
    with self.assertRaisesRegex(ValueError, 'cannot establish length of variable groups A, B'):
      evaluable.einsum('AB->BA', arg)

  def test_wrong_indices(self):
    arg = numpy.arange(9).reshape(3,3)
    with self.assertRaisesRegex(ValueError, 'internal repetitions are not supported'):
      evaluable.einsum('kk->', arg)

  def test_wrong_shapes(self):
    arg1 = numpy.arange(6).reshape(2,3)
    arg2 = numpy.arange(6).reshape(3,2)
    with self.assertRaisesRegex(ValueError, 'shapes do not match for axis i0'):
      ret = evaluable.einsum('ij,ik->jk', evaluable.asarray(arg1), evaluable.asarray(arg2))

  def test_wrong_group_dimension(self):
    arg = numpy.arange(6)
    with self.assertRaisesRegex(ValueError, 'axis group dimensions cannot be negative'):
      evaluable.einsum('Aij->ijA', arg, A=-1)

@parametrize
class AsType(TestCase):

  def test_bool(self):
    self.assertEqual(evaluable.astype(True, self.dtype).dtype, self.dtype)

  def test_int(self):
    self.assertEqual(evaluable.astype(1, self.dtype).dtype, self.dtype)

  def test_float(self):
    if self.dtype in (float, complex):
      self.assertEqual(evaluable.astype(1., self.dtype).dtype, self.dtype)
    else:
      with self.assertRaises(TypeError):
        evaluable.astype(1., self.dtype)

  def test_complex(self):
    if self.dtype == complex:
      self.assertEqual(evaluable.astype(1j, self.dtype).dtype, self.dtype)
    else:
      with self.assertRaises(TypeError):
        evaluable.astype(1j, self.dtype)

AsType(dtype=int)
AsType(dtype=float)
AsType(dtype=complex)
