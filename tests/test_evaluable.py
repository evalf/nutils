import numpy, itertools, pickle, weakref, gc, warnings as _builtin_warnings, collections, sys, unittest
from nutils import *
from nutils.testing import *
_ = numpy.newaxis


@parametrize
class check(TestCase):

  def setUp(self):
    super().setUp()
    numpy.random.seed(0)
    self.arg_names = tuple(map('arg{}'.format, range(len(self.shapes))))
    self.args = tuple(map(evaluable.Argument, self.arg_names, self.shapes))
    self.arg_values = [numpy.random.uniform(size=shape, low=self.low, high=self.high) for shape in self.shapes]
    self.n_op_argsfun = self.n_op(*self.arg_values)
    self.op_args = self.op(*self.args)
    self.shapearg = numpy.random.uniform(size=self.n_op_argsfun.shape, low=self.low, high=self.high)
    self.pairs = [(i, j) for i in range(self.op_args.ndim-1) for j in range(i+1, self.op_args.ndim) if self.op_args.shape[i] == self.op_args.shape[j]]
    _builtin_warnings.simplefilter('ignore', evaluable.ExpensiveEvaluationWarning)

  def assertShapes(self):
    self.assertEqual(self.n_op_argsfun.shape, tuple(map(int, self.op_args.shape)))

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
    constargs = [numpy.random.uniform(size=shape) for shape in self.shapes]
    self.assertFunctionAlmostEqual(decimal=15,
      desired=self.n_op(*[constarg for constarg in constargs]),
      actual=self.op(*constargs))

  def test_eval(self):
    self.assertFunctionAlmostEqual(decimal=15,
      actual=self.op_args,
      desired=self.n_op_argsfun)

  @unittest.skipIf(sys.version_info < (3,7), 'time.perf_counter_ns is not available')
  def test_eval_withtimes(self):
    evalargs = dict(zip(self.arg_names, self.arg_values))
    without_times = self.op_args.eval(**evalargs)
    stats = collections.defaultdict(evaluable._Stats)
    with_times = self.op_args.eval_withtimes(stats, **evalargs)
    self.assertArrayAlmostEqual(with_times, without_times, 15)
    self.assertIn(self.op_args, stats)

  def test_getitem(self):
    for idim in range(self.op_args.ndim):
      for item in range(self.n_op_argsfun.shape[idim]):
        s = (Ellipsis,) + (slice(None),)*idim + (item,) + (slice(None),)*(self.op_args.ndim-idim-1)
        self.assertFunctionAlmostEqual(decimal=15,
          desired=self.n_op_argsfun[s],
          actual=self.op_args[s])

  def test_transpose(self):
    trans = numpy.arange(self.op_args.ndim,0,-1) % self.op_args.ndim
    self.assertFunctionAlmostEqual(decimal=15,
      desired=numpy.transpose(self.n_op_argsfun, trans),
      actual=evaluable.transpose(self.op_args, trans))

  def test_insertaxis(self):
    for axis in range(self.op_args.ndim+1):
      with self.subTest(axis=axis):
        self.assertFunctionAlmostEqual(decimal=15,
          desired=numpy.repeat(numpy.expand_dims(self.n_op_argsfun, axis), 2, axis),
          actual=evaluable.insertaxis(self.op_args, axis, 2))

  def test_takediag(self):
    for ax1, ax2 in self.pairs:
      self.assertFunctionAlmostEqual(decimal=15,
        desired=numeric.takediag(self.n_op_argsfun, ax1, ax2),
        actual=evaluable.takediag(self.op_args, ax1, ax2))

  def test_eig(self):
    if self.op_args.dtype == float:
      for ax1, ax2 in self.pairs:
        A, L, V = evaluable.Tuple((self.op_args, *evaluable.eig(self.op_args, axes=(ax1,ax2)))).eval(**dict(zip(self.arg_names, self.arg_values)))
        self.assertArrayAlmostEqual(decimal=11,
          actual=(numpy.expand_dims(V,ax2) * numpy.expand_dims(L,ax2+1).swapaxes(ax1,ax2+1)).sum(ax2+1),
          desired=(numpy.expand_dims(A,ax2) * numpy.expand_dims(V,ax2+1).swapaxes(ax1,ax2+1)).sum(ax2+1))

  def test_inv(self):
    for ax1, ax2 in self.pairs:
      trans = [i for i in range(self.n_op_argsfun.ndim) if i not in (ax1,ax2)] + [ax1,ax2]
      invtrans = list(map(trans.index, range(len(trans))))
      self.assertFunctionAlmostEqual(decimal=10,
        desired=numeric.inv(self.n_op_argsfun.transpose(trans)).transpose(invtrans),
        actual=evaluable.inverse(self.op_args, axes=(ax1,ax2)))

  def test_determinant(self):
    for ax1, ax2 in self.pairs:
      self.assertFunctionAlmostEqual(decimal=11,
        desired=numpy.linalg.det(self.n_op_argsfun.transpose([i for i in range(self.n_op_argsfun.ndim) if i not in (ax1,ax2)] + [ax1,ax2])),
        actual=evaluable.determinant(self.op_args, axes=(ax1,ax2)))

  def test_take(self):
    indices = [0,-1]
    for iax, sh in enumerate(self.n_op_argsfun.shape):
      if sh >= 2:
        self.assertFunctionAlmostEqual(decimal=15,
          desired=numpy.take(self.n_op_argsfun, indices, axis=iax),
          actual=evaluable.take(self.op_args, indices, axis=iax))

  def test_take_block(self):
    for iax, sh in enumerate(self.n_op_argsfun.shape):
      if sh >= 2:
        indices = [[0,sh-1],[sh-1,0]]
        self.assertFunctionAlmostEqual(decimal=15,
          desired=numpy.take(self.n_op_argsfun, indices, axis=iax),
          actual=evaluable._take(self.op_args, indices, axis=iax))

  def test_take_nomask(self):
    for iax, sh in enumerate(self.n_op_argsfun.shape):
      if sh >= 2:
        indices = [0,sh-1]
        self.assertFunctionAlmostEqual(decimal=15,
          desired=numpy.take(self.n_op_argsfun, indices, axis=iax),
          actual=evaluable.take(self.op_args, evaluable.Guard(evaluable.asarray(indices)), axis=iax))

  def test_take_reversed(self):
    indices = [-1,0]
    for iax, sh in enumerate(self.n_op_argsfun.shape):
      if sh >= 2:
        self.assertFunctionAlmostEqual(decimal=15,
          desired=numpy.take(self.n_op_argsfun, indices, axis=iax),
          actual=evaluable.take(self.op_args, indices, axis=iax))

  def test_take_duplicate_indices(self):
    for iax, sh in enumerate(self.n_op_argsfun.shape):
      if sh >= 2:
        indices = [0,sh-1,0,0]
        self.assertFunctionAlmostEqual(decimal=15,
          desired=numpy.take(self.n_op_argsfun, indices, axis=iax),
          actual=evaluable.take(self.op_args, evaluable.Guard(evaluable.asarray(indices)), axis=iax))

  def test_inflate(self):
    for iax, sh in enumerate(self.n_op_argsfun.shape):
      dofmap = evaluable.Constant(numpy.arange(int(sh)) * 2)
      desired = numpy.zeros(self.n_op_argsfun.shape[:iax] + (int(sh)*2-1,) + self.n_op_argsfun.shape[iax+1:], dtype=self.n_op_argsfun.dtype)
      desired[(slice(None),)*iax+(slice(None,None,2),)] = self.n_op_argsfun
      self.assertFunctionAlmostEqual(decimal=15,
        desired=desired,
        actual=evaluable._inflate(self.op_args, dofmap=dofmap, length=sh*2-1, axis=iax))

  def test_inflate_duplicate_indices(self):
    for iax, sh in enumerate(self.n_op_argsfun.shape):
      dofmap = numpy.arange(sh) % 2
      desired = numpy.zeros(self.n_op_argsfun.shape[:iax] + (2,) + self.n_op_argsfun.shape[iax+1:], dtype=self.n_op_argsfun.dtype)
      numpy.add.at(desired, (slice(None),)*iax+(dofmap,), self.n_op_argsfun)
      self.assertFunctionAlmostEqual(decimal=15,
        desired=desired,
        actual=evaluable._inflate(self.op_args, dofmap=dofmap, length=2, axis=iax))

  def test_diagonalize(self):
    for axis in range(self.op_args.ndim):
      for newaxis in range(axis+1, self.op_args.ndim+1):
        self.assertFunctionAlmostEqual(decimal=15,
          desired=numeric.diagonalize(self.n_op_argsfun, axis, newaxis),
          actual=evaluable.diagonalize(self.op_args, axis, newaxis))

  def test_product(self):
    for iax in range(self.op_args.ndim):
      self.assertFunctionAlmostEqual(decimal=15,
        desired=numpy.product(self.n_op_argsfun, axis=iax),
        actual=evaluable.product(self.op_args, axis=iax))

  def test_getslice(self):
    for idim in range(self.op_args.ndim):
      if self.n_op_argsfun.shape[idim] == 1:
        continue
      s = (Ellipsis,) + (slice(None),)*idim + (slice(0,int(self.n_op_argsfun.shape[idim])-1),) + (slice(None),)*(self.op_args.ndim-idim-1)
      self.assertFunctionAlmostEqual(decimal=15,
        desired=self.n_op_argsfun[s],
        actual=self.op_args[s])

  def test_sumaxis(self):
    op_args = evaluable.Int(self.op_args) if self.op_args.dtype == bool else self.op_args
    for idim in range(self.op_args.ndim):
      self.assertFunctionAlmostEqual(decimal=14,
        desired=self.n_op_argsfun.sum(idim),
        actual=op_args.sum(idim))

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
      self.assertFunctionAlmostEqual(decimal=13,
        desired=numeric.contract(self.n_op_argsfun, self.shapearg, axis=iax),
        actual=evaluable.dot(self.op_args, self.shapearg, axes=iax))

  def test_pointwise(self):
    self.assertFunctionAlmostEqual(decimal=15,
      desired=numpy.sin(self.n_op_argsfun).astype(float), # "astype" necessary for boolean operations (float16->float64)
      actual=evaluable.sin(self.op_args))

  def test_power(self):
    self.assertFunctionAlmostEqual(decimal=13,
      desired=self.n_op_argsfun**3,
      actual=(self.op_args**3))

  def test_power0(self):
    power = (numpy.arange(self.n_op_argsfun.size) % 2).reshape(self.n_op_argsfun.shape)
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
      if self.n_op_argsfun.shape[idim] <= 1:
        continue
      mask = numpy.ones(self.n_op_argsfun.shape[idim], dtype=bool)
      mask[0] = False
      if self.n_op_argsfun.shape[idim] > 2:
        mask[-1] = False
      self.assertFunctionAlmostEqual(decimal=15,
        desired=self.n_op_argsfun[(slice(None,),)*idim+(mask,)],
        actual=evaluable.mask(self.op_args, mask, axis=idim))

  def test_ravel(self):
    for idim in range(self.op_args.ndim-1):
      self.assertFunctionAlmostEqual(decimal=15,
        desired=self.n_op_argsfun.reshape(self.n_op_argsfun.shape[:idim]+(-1,)+self.n_op_argsfun.shape[idim+2:]),
        actual=evaluable.ravel(self.op_args, axis=idim))

  def test_unravel(self):
    for idim in range(self.op_args.ndim):
      length = self.n_op_argsfun.shape[idim]
      unravelshape = (length//3,3) if (length%3==0) else (length//2,2) if (length%2==0) else (length,1)
      self.assertFunctionAlmostEqual(decimal=15,
        desired=self.n_op_argsfun.reshape(self.n_op_argsfun.shape[:idim]+unravelshape+self.n_op_argsfun.shape[idim+1:]),
        actual=evaluable.unravel(self.op_args, axis=idim, shape=unravelshape))

  def test_loopsum(self):
    length = 3
    index = evaluable.loop_index('_testindex', length)
    for iarg, shape in enumerate(self.shapes):
      testvalue = numpy.random.uniform(size=(length, *shape), low=self.low, high=self.high)
      n_op_argsfun = functools.reduce(operator.add, (self.n_op(*self.arg_values[:iarg], v, *self.arg_values[iarg+1:]) for v in testvalue))
      args = (*self.args[:iarg], evaluable.Guard(evaluable.get(evaluable.asarray(testvalue), 0, index)), *self.args[iarg+1:])
      self.assertFunctionAlmostEqual(decimal=15,
        actual=evaluable.loop_sum(self.op(*args), index),
        desired=n_op_argsfun)

  def test_loopconcatenate(self):
    length = 3
    index = evaluable.loop_index('_testindex', length)
    for iarg, shape in enumerate(self.shapes):
      testvalue = numpy.random.uniform(size=(length, *shape), low=self.low, high=self.high)
      n_op_argsfun = numpy.concatenate([self.n_op(*self.arg_values[:iarg], v, *self.arg_values[iarg+1:]) for v in testvalue], axis=-1)
      args = (*self.args[:iarg], evaluable.Guard(evaluable.get(evaluable.asarray(testvalue), 0, index)), *self.args[iarg+1:])
      self.assertFunctionAlmostEqual(decimal=15,
        actual=evaluable.loop_concatenate(self.op(*args), index),
        desired=n_op_argsfun)

  def test_desparsify(self):
    args = []
    for arg in self.args:
      for i in range(arg.ndim):
        arg = evaluable._inflate(arg, evaluable.Guard(numpy.arange(int(arg.shape[i]))), arg.shape[i], i)
      args.append(arg)
    op_args = self.op(*args).simplified
    evalargs = dict(zip(self.arg_names, self.arg_values))
    for axis, prop in enumerate(op_args._axes):
      if isinstance(prop, evaluable.Sparse):
        actual = numpy.zeros_like(self.n_op_argsfun)
        for ind, f in op_args._desparsify(axis):
          _ind = ind.eval(**evalargs)
          numpy.add.at(actual, (slice(None),)*(axis)+(_ind,), f.eval(**evalargs))
        self.assertArrayAlmostEqual(actual, self.n_op_argsfun, decimal=15)

  @parametrize.enable_if(lambda hasgrad, **kwargs: hasgrad)
  def test_derivative(self):
    eps = 1e-4
    fddeltas = numpy.array([1,2,3])
    fdfactors = numpy.linalg.solve(2*fddeltas**numpy.arange(1,1+2*len(fddeltas),2)[:,None], [1]+[0]*(len(fddeltas)-1))
    for arg, arg_name, x0 in zip(self.args, self.arg_names, self.arg_values):
      with self.subTest(arg_name):
        dx = numpy.random.normal(size=x0.shape)
        evalargs = dict(zip(self.arg_names, self.arg_values))
        f0 = evaluable.derivative(self.op_args, arg).eval(**evalargs)
        exact = numeric.contract(f0, dx, range(self.op_args.ndim, self.op_args.ndim+dx.ndim))
        if exact.dtype.kind in 'bi' or self.zerograd:
          approx = numpy.zeros_like(exact)
          scale = 1
        else:
          fdvals = numpy.stack([self.op_args.eval(**collections.ChainMap({arg_name: numpy.asarray(x0+eps*n*dx)}, evalargs)) for n in (*-fddeltas, *fddeltas)], axis=0)
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
      node = self.op_args._node(cache, None, times)
      if node:
        self.assertIn(self.op_args, cache)
        self.assertEqual(cache[self.op_args], node)
    with self.subTest('from-cache'):
      if node:
        self.assertEqual(self.op_args._node(cache, None, times), node)
    with self.subTest('with-times'):
      times = collections.defaultdict(evaluable._Stats)
      self.op_args.eval_withtimes(times, **dict(zip(self.arg_names, self.arg_values)))
      self.op_args._node(cache, None, times)

def _check(name, op, n_op, shapes, hasgrad=True, zerograd=False, ndim=2, low=-1, high=1):
  check(name, op=op, n_op=n_op, shapes=shapes, hasgrad=hasgrad, zerograd=zerograd, ndim=ndim, low=low, high=high)

_check('identity', lambda f: evaluable.asarray(f), lambda a: a, [(2,4,2)])
_check('const', lambda f: evaluable.asarray(numpy.arange(16, dtype=float).reshape(2,4,2)), lambda a: numpy.arange(16, dtype=float).reshape(2,4,2), [(2,4,2)])
_check('zeros', lambda f: evaluable.zeros([1,4,3,4]), lambda a: numpy.zeros([1,4,3,4]), [(4,3,4)])
_check('ones', lambda f: evaluable.ones([1,4,3,4]), lambda a: numpy.ones([1,4,3,4]), [(4,3,4)])
_check('range', lambda f: evaluable.Range(4) + 2, lambda a: numpy.arange(2,6), [(4,)])
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
_check('power', evaluable.power, numpy.power, [(4,4),(4,4)], low=0)
_check('negative', evaluable.negative, numpy.negative, [(4,)])
_check('reciprocal', evaluable.reciprocal, numpy.reciprocal, [(4,)], low=-2, high=-1)
_check('arcsin', evaluable.arcsin, numpy.arcsin, [(4,)])
_check('arccos', evaluable.arccos, numpy.arccos, [(4,)])
_check('arctan', evaluable.arctan, numpy.arctan, [(4,)])
_check('ln', evaluable.ln, numpy.log, [(4,)], low=0)
_check('product', lambda a: evaluable.product(a,2), lambda a: numpy.product(a,2), [(4,3,4)])
_check('sum', lambda a: evaluable.sum(a,2), lambda a: a.sum(2), [(4,3,4)])
_check('transpose1', lambda a: evaluable.transpose(a,[0,1,3,2]), lambda a: a.transpose([0,1,3,2]), [(2,3,4,5)], low=0, high=20)
_check('transpose2', lambda a: evaluable.transpose(a,[0,2,3,1]), lambda a: a.transpose([0,2,3,1]), [(2,3,4,5)])
_check('insertaxis', lambda a: evaluable.insertaxis(a,1,3), lambda a: numpy.repeat(a[:,None], 3, 1), [(2,4)])
_check('get', lambda a: evaluable.get(a,2,1), lambda a: a[:,:,1], [(4,3,4)])
_check('takediag141', lambda a: evaluable.takediag(a,0,2), lambda a: numeric.takediag(a,0,2), [(1,4,1)])
_check('takediag434', lambda a: evaluable.takediag(a,0,2), lambda a: numeric.takediag(a,0,2), [(4,3,4)])
_check('takediag343', lambda a: evaluable.takediag(a,0,2), lambda a: numeric.takediag(a,0,2), [(3,4,3)])
_check('determinant141', lambda a: evaluable.determinant(a,(0,2)), lambda a: numpy.linalg.det(a.swapaxes(0,1)), [(1,4,1)])
_check('determinant434', lambda a: evaluable.determinant(a,(0,2)), lambda a: numpy.linalg.det(a.swapaxes(0,1)), [(4,3,4)])
_check('determinant4433', lambda a: evaluable.determinant(a,(2,3)), lambda a: numpy.linalg.det(a), [(4,4,3,3)])
_check('determinant200', lambda a: evaluable.determinant(a,(1,2)), lambda a: numpy.linalg.det(a) if a.shape[-1] else numpy.ones(a.shape[:-2], float), [(2,0,0)], zerograd=True)
_check('inverse141', lambda a: evaluable.inverse(a+1,(0,2)), lambda a: numpy.linalg.inv(a.swapaxes(0,1)+1).swapaxes(0,1), [(1,4,1)])
_check('inverse434', lambda a: evaluable.inverse(a+5*evaluable.insertaxis(evaluable.Diagonalize(evaluable.ones([4])), 1, 3),(0,2)), lambda a: numpy.linalg.inv(a.swapaxes(0,1)+5*numpy.eye(4)).swapaxes(0,1), [(4,3,4)])
_check('inverse4422', lambda a: evaluable.inverse(a+evaluable.prependaxes(evaluable.Diagonalize(evaluable.ones([2])), (4,4))), lambda a: numpy.linalg.inv(a+numpy.eye(2)), [(4,4,2,2)])
_check('repeat', lambda a: evaluable.repeat(a,3,1), lambda a: numpy.repeat(a,3,1), [(4,1,4)])
_check('diagonalize', lambda a: evaluable.diagonalize(a,1,3), lambda a: numeric.diagonalize(a,1,3), [(4,4,4,4)])
_check('multiply', evaluable.multiply, numpy.multiply, [(4,4),(4,4)])
_check('dot', lambda a,b: evaluable.dot(a,b,axes=1), lambda a,b: (a*b).sum(1), [(4,2,4),(4,2,4)])
_check('divide', evaluable.divide, lambda a, b: a * b**-1, [(4,4),(4,4)], low=-2, high=-1)
_check('divide2', lambda a: evaluable.asarray(a)/2, lambda a: a/2, [(4,1)])
_check('add', evaluable.add, numpy.add, [(4,4),(4,4)])
_check('subtract', evaluable.subtract, numpy.subtract, [(4,4),(4,4)])
_check('dot2', lambda a,b: evaluable.multiply(a,b).sum(-2), lambda a,b: (a*b).sum(-2), [(4,2,4),(4,2,4)])
_check('min', lambda a,b: evaluable.Minimum(a,b), numpy.minimum, [(4,4),(4,4)])
_check('max', lambda a,b: evaluable.Maximum(a,b), numpy.maximum, [(4,4),(4,4)])
_check('equal', evaluable.Equal, numpy.equal, [(4,4),(4,4)], zerograd=True)
_check('greater', evaluable.Greater, numpy.greater, [(4,4),(4,4)], zerograd=True)
_check('less', evaluable.Less, numpy.less, [(4,4),(4,4)], zerograd=True)
_check('arctan2', evaluable.arctan2, numpy.arctan2, [(4,4),(4,4)])
_check('stack', lambda a,b: evaluable.stack([a,b], 0), lambda a,b: numpy.concatenate([a[_,:],b[_,:]], axis=0), [(4,),(4,)])
_check('eig', lambda a: evaluable.eig(a+a.swapaxes(0,1),symmetric=True)[1], lambda a: numpy.linalg.eigh(a+a.swapaxes(0,1))[1], [(4,4)], hasgrad=False)
_check('trignormal', lambda a: evaluable.TrigNormal(a), lambda a: numpy.array([numpy.cos(a), numpy.sin(a)]), [()])
_check('trigtangent', lambda a: evaluable.TrigTangent(a), lambda a: numpy.array([-numpy.sin(a), numpy.cos(a)]), [()])
_check('mod', lambda a,b: evaluable.mod(a,b), lambda a,b: numpy.mod(a,b), [(4,),(4,)], hasgrad=False)
_check('mask', lambda f: evaluable.mask(f,numpy.array([True,False,True,False,True,False,True]),axis=1), lambda a: a[:,::2], [(4,7,4)])
_check('ravel', lambda f: evaluable.ravel(f,axis=1), lambda a: a.reshape(4,4,4,4), [(4,2,2,4,4)])
_check('unravel', lambda f: evaluable.unravel(f,axis=1,shape=[2,2]), lambda a: a.reshape(4,2,2,4,4), [(4,4,4,4)])
_check('inflate', lambda f: evaluable._inflate(f,dofmap=evaluable.Guard([0,3]),length=4,axis=1), lambda a: numpy.concatenate([a[:,:1], numpy.zeros_like(a), a[:,1:]], axis=1), [(4,2,4)])
_check('inflate-constant', lambda f: evaluable._inflate(f,dofmap=[0,3],length=4,axis=1), lambda a: numpy.concatenate([a[:,:1], numpy.zeros_like(a), a[:,1:]], axis=1), [(4,2,4)])
_check('inflate-duplicate', lambda f: evaluable.Inflate(f,dofmap=[0,1,0,3],length=4), lambda a: numpy.stack([a[:,0]+a[:,2], a[:,1], numpy.zeros_like(a[:,0]), a[:,3]], axis=1), [(2,4)])
_check('inflate-block', lambda f: evaluable.Inflate(f,dofmap=[[5,4,3],[2,1,0]],length=6), lambda a: a.ravel()[::-1], [(2,3)])
_check('inflate-scalar', lambda f: evaluable.Inflate(f,dofmap=1,length=3), lambda a: numpy.array([0,a,0]), [()])
_check('inflate-diagonal', lambda f: evaluable.Inflate(evaluable.Inflate(f,1,3),1,3), lambda a: numpy.diag(numpy.array([0,a,0])), [()])
_check('inflate-one', lambda f: evaluable.Inflate(f,0,1), lambda a: numpy.array([a]), [()])
_check('inflate-range', lambda f: evaluable.Inflate(f,evaluable.Range(3),3), lambda a: a, [(3,)])
_check('take', lambda f: evaluable.Take(f, [0,3,2]), lambda a: a[:,[0,3,2]], [(2,4)])
_check('take-duplicate', lambda f: evaluable.Take(f, [0,3,0]), lambda a: a[:,[0,3,0]], [(2,4)])
_check('choose', lambda a, b, c: evaluable.Choose(evaluable.appendaxes(evaluable.Int(a)%2, (3,3)), [b,c]), lambda a, b, c: numpy.choose(a[_,_].astype(int)%2, [b,c]), [(), (3,3), (3,3)])
_check('slice', lambda a: evaluable.asarray(a)[::2], lambda a: a[::2], [(5,3)])
_check('normal1d', lambda a: evaluable.Normal(a), lambda a: numpy.sign(a[...,0]), [(3,1,1)])
_check('normal2d', lambda a: evaluable.Normal(a), lambda a: numpy.stack([Q[:,-1]*numpy.sign(R[-1,-1]) for ai in a for Q, R in [numpy.linalg.qr(ai, mode='complete')]], axis=0), [(1,2,2)])
_check('normal3d', lambda a: evaluable.Normal(a), lambda a: numpy.stack([Q[:,-1]*numpy.sign(R[-1,-1]) for ai in a for Q, R in [numpy.linalg.qr(ai, mode='complete')]], axis=0), [(2,3,3)])
_check('loopsum1', lambda: evaluable.loop_sum(evaluable.loop_index('index', 3), evaluable.loop_index('index', 3)), lambda: numpy.array(3), [])
_check('loopsum2', lambda a: evaluable.loop_sum(a, evaluable.loop_index('index', 2)), lambda a: 2*a, [(3,4,2,4)])
_check('loopsum3', lambda a: evaluable.loop_sum(evaluable.get(a, 0, evaluable.loop_index('index', 3)), evaluable.loop_index('index', 3)), lambda a: numpy.sum(a, 0), [(3,4,2,4)])
_check('loopsum4', lambda: evaluable.loop_sum(evaluable.Inflate(evaluable.loop_index('index', 3), 0, 2), evaluable.loop_index('index', 3)), lambda: numpy.array([3, 0]), [])
_check('loopsum5', lambda: evaluable.loop_sum(evaluable.loop_index('index', 1), evaluable.loop_index('index', 1)), lambda: numpy.array(0), [])
_check('loopconcatenate1', lambda a: evaluable.loop_concatenate(a+evaluable.prependaxes(evaluable.loop_index('index', 3), a.shape), evaluable.loop_index('index', 3)), lambda a: a+numpy.arange(3)[None], [(2,1)])
_check('loopconcatenate2', lambda: evaluable.loop_concatenate(evaluable.Elemwise([numpy.arange(48).reshape(4,4,3)[:,:,a:b] for a, b in util.pairwise([0,2,3])], evaluable.loop_index('index', 2), int), evaluable.loop_index('index', 2)), lambda: numpy.arange(48).reshape(4,4,3), [])
_check('loopconcatenatecombined', lambda a: evaluable.loop_concatenate_combined([a+evaluable.prependaxes(evaluable.loop_index('index', 3), a.shape)], evaluable.loop_index('index', 3))[0], lambda a: a+numpy.arange(3)[None], [(2,1)], hasgrad=False)

_polyval_mask = lambda shape, ndim: 1 if ndim == 0 else numpy.array([sum(i[-ndim:]) < int(shape[-1]) for i in numpy.ndindex(shape)], dtype=int).reshape(shape)
_polyval_desired = lambda c, x: sum(c[(...,*i)]*(x[(slice(None),*[None]*(c.ndim-x.shape[1]))]**i).prod(-1) for i in itertools.product(*[range(c.shape[-1])]*x.shape[1]) if sum(i) < c.shape[-1])
_check('polyval_1d_p0', lambda c, x: evaluable.Polyval(c*_polyval_mask(c.shape,x.shape[1]), x), _polyval_desired, [(1,),(4,1)], ndim=1)
_check('polyval_1d_p1', lambda c, x: evaluable.Polyval(c*_polyval_mask(c.shape,x.shape[1]), x), _polyval_desired, [(2,),(4,1)], ndim=1)
_check('polyval_1d_p2', lambda c, x: evaluable.Polyval(c*_polyval_mask(c.shape,x.shape[1]), x), _polyval_desired, [(3,),(4,1)], ndim=1)
_check('polyval_2d_p0', lambda c, x: evaluable.Polyval(c*_polyval_mask(c.shape,x.shape[1]), x), _polyval_desired, [(1,1),(4,2)], ndim=2)
_check('polyval_2d_p1', lambda c, x: evaluable.Polyval(c*_polyval_mask(c.shape,x.shape[1]), x), _polyval_desired, [(2,2),(4,2)], ndim=2)
_check('polyval_2d_p2', lambda c, x: evaluable.Polyval(c*_polyval_mask(c.shape,x.shape[1]), x), _polyval_desired, [(3,3),(4,2)], ndim=2)
_check('polyval_2d_p1_23', lambda c, x: evaluable.Polyval(c*_polyval_mask(c.shape,x.shape[1]), x), _polyval_desired, [(2,3,2,2),(4,2)], ndim=2)

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

  def test_square_negative(self):
    self.assertBounds(evaluable.Square(self.R(-4,[4])))

  def test_square_positive(self):
    self.assertBounds(evaluable.Square(self.R(1,[4])))

  def test_square_full(self):
    self.assertBounds(evaluable.Square(self.R(-3,[7])))

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

  def test_int_bool(self):
    self.assertEqual(evaluable.Int(evaluable.Constant(numpy.array([False, True], dtype=bool)))._intbounds, (0, 1))

  def test_int_int(self):
    self.assertEqual(evaluable.Int(self.S('n', 3, 5))._intbounds, (3, 5))

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

class jacobian(TestCase):

  def setUp(self):
    super().setUp()
    self.domain, self.geom = mesh.unitsquare(1, 'square')
    self.basis = self.domain.basis('std', degree=1)
    arg = function.Argument('dofs', [4])
    self.v = self.basis.dot(arg)
    self.X = (self.geom[numpy.newaxis,:] * [[0,1],[-self.v,0]]).sum(-1) # X_i = <x_1, -2 x_0>_i
    self.J = function.J(self.X, 2)
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

class asciitree(TestCase):

  @unittest.skipIf(sys.version_info < (3, 6), 'test requires dicts maintaining insertion order')
  def test_asciitree(self):
    f = evaluable.Sin((evaluable.Zeros((), int))**evaluable.Diagonalize(evaluable.Argument('arg', (2,))))
    self.assertEqual(f.asciitree(richoutput=True),
                     '%0 = Sin; f:2,2\n'
                     '└ %1 = Power; f:2,2\n'
                     '  ├ %2 = InsertAxis; i:2,2\n'
                     '  │ ├ %3 = InsertAxis; i:2\n'
                     '  │ │ ├ 0\n'
                     '  │ │ └ 2\n'
                     '  │ └ 2\n'
                     '  └ %4 = Diagonalize; f:2,2\n'
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
                     '├ shape[0] = %A1 = Take; i:\n'
                     '│ ├ %A2 = _SizesToOffsets; i:3\n'
                     '│ │ └ %A3 = InsertAxis; i:2\n'
                     '│ │   ├ 1\n'
                     '│ │   └ 2\n'
                     '│ └ 2\n'
                     '├ start = %B4 = Take; i:\n'
                     '│ ├ %A2\n'
                     '│ └ %B5 = LoopIndex\n'
                     '│   └ length = 2\n'
                     '├ stop = %B6 = Take; i:\n'
                     '│ ├ %A2\n'
                     '│ └ %B7 = Add; i:\n'
                     '│   ├ %B5\n'
                     '│   └ 1\n'
                     '└ func = %B8 = InsertAxis; i:1\n'
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
                     '├ shape[0] = %A1 = Take; i:\n'
                     '│ ├ %A2 = _SizesToOffsets; i:3\n'
                     '│ │ └ %A3 = InsertAxis; i:2\n'
                     '│ │   ├ 1\n'
                     '│ │   └ 2\n'
                     '│ └ 2\n'
                     '├ start = %B4 = Take; i:\n'
                     '│ ├ %A2\n'
                     '│ └ %B5 = LoopIndex\n'
                     '│   └ length = 2\n'
                     '├ stop = %B6 = Take; i:\n'
                     '│ ├ %A2\n'
                     '│ └ %B7 = Add; i:\n'
                     '│   ├ %B5\n'
                     '│   └ 1\n'
                     '└ func = %B8 = InsertAxis; i:1\n'
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

class memory(TestCase):

  def assertCollected(self, ref):
    gc.collect()
    if ref() is not None:
      self.fail('object was not garbage collected')

  def test_general(self):
    A = evaluable.Constant([1,2,3])
    A = weakref.ref(A)
    self.assertCollected(A)

  def test_simplified(self):
    A = evaluable.Constant([1,2,3])
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
    desired = evaluable.Tuple((evaluable.ArrayFromTuple(L, 0, (3,), int), evaluable.ArrayFromTuple(L, 1, (6,), int)))
    self.assertEqual(actual, desired)

  def test_different_index(self):
    i = evaluable.loop_index('i', 3)
    j = evaluable.loop_index('j', 3)
    A = evaluable.LoopConcatenate((evaluable.InsertAxis(i, 1), i, i+1, 3,), i._name, i.length)
    B = evaluable.LoopConcatenate((evaluable.InsertAxis(j, 1), j, j+1, 3,), j._name, j.length)
    actual = evaluable.Tuple((A, B))._combine_loop_concatenates(set())
    L1 = evaluable.LoopConcatenateCombined(((evaluable.InsertAxis(i, 1), i, i+1, 3),), i._name, i.length)
    L2 = evaluable.LoopConcatenateCombined(((evaluable.InsertAxis(j, 1), j, j+1, 3),), j._name, j.length)
    desired = evaluable.Tuple((evaluable.ArrayFromTuple(L1, 0, (3,), int), evaluable.ArrayFromTuple(L2, 0, (3,), int)))
    self.assertEqual(actual, desired)

  def test_nested_invariant(self):
    i = evaluable.loop_index('i', 3)
    A = evaluable.LoopConcatenate((evaluable.InsertAxis(i, 1), i, i+1, 3,), i._name, i.length)
    B = evaluable.LoopConcatenate((A, i*3, i*3+3, 9,), i._name, i.length)
    actual = evaluable.Tuple((A, B))._combine_loop_concatenates(set())
    L1 = evaluable.LoopConcatenateCombined(((evaluable.InsertAxis(i, 1), i, i+1, 3),), i._name, i.length)
    A_ = evaluable.ArrayFromTuple(L1, 0, (3,), int)
    L2 = evaluable.LoopConcatenateCombined(((A_, i*3, i*3+3, 9),), i._name, i.length)
    self.assertIn(A_, L2._Evaluable__args)
    desired = evaluable.Tuple((A_, evaluable.ArrayFromTuple(L2, 0, (9,), int)))
    self.assertEqual(actual, desired)

  def test_nested_variant(self):
    i = evaluable.loop_index('i', 3)
    j = evaluable.loop_index('j', 3)
    A = evaluable.LoopConcatenate((evaluable.InsertAxis(i+j, 1), i, i+1, 3,), i._name, i.length)
    B = evaluable.LoopConcatenate((A, j*3, j*3+3, 9,), j._name, j.length)
    actual = evaluable.Tuple((A, B))._combine_loop_concatenates(set())
    L1 = evaluable.LoopConcatenateCombined(((evaluable.InsertAxis(i+j, 1), i, i+1, 3),), i._name, i.length)
    A_ = evaluable.ArrayFromTuple(L1, 0, (3,), int)
    L2 = evaluable.LoopConcatenateCombined(((A_, j*3, j*3+3, 9),), j._name, j.length)
    self.assertNotIn(A_, L2._Evaluable__args)
    desired = evaluable.Tuple((A_, evaluable.ArrayFromTuple(L2, 0, (9,), int)))
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
