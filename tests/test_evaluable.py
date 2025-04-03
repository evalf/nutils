from nutils import evaluable, sparse, numeric, _util as util, types, sample, matrix, _pyast
from nutils.testing import TestCase, parametrize
import nutils_poly as poly
import numpy
import itertools
import weakref
import gc
import warnings as _builtin_warnings
import collections
import sys
import unittest
import functools
import operator
import logging


@parametrize
class check(TestCase):

    def setUp(self):
        super().setUp()
        numpy.random.seed(0)
        self.arg_names = tuple(map('arg{}'.format, range(len(self.arg_values))))
        self.args = tuple(evaluable.Argument(name, tuple(evaluable.constant(n) for n in value.shape), evaluable.asdtype(value.dtype)) for name, value in zip(self.arg_names, self.arg_values))
        self.actual = self.op(*self.args)
        self.desired = self.n_op(*self.arg_values)
        assert numpy.isfinite(self.desired).all(), 'something is wrong with the design of this unit test'
        self.varshape = not all(n.isconstant for n in self.actual.shape)
        othershape = self.desired.shape if not self.varshape else ()
        if self.actual.dtype == bool:
            self.other = numpy.random.normal(size=othershape) < 0
        elif self.actual.dtype == int:
            self.other = numpy.random.randint(0, 100, size=othershape)
        elif self.actual.dtype == float:
            self.other = numpy.random.normal(size=othershape)
        elif self.actual.dtype == complex:
            self.other = numpy.random.normal(size=othershape) + 1j * numpy.random.normal(size=othershape)
        self.pairs = [(i, j) for i in range(self.actual.ndim-1) for j in range(i+1, self.actual.ndim) if self.actual.shape[i] == self.actual.shape[j]]
        _builtin_warnings.simplefilter('ignore', evaluable.ExpensiveEvaluationWarning)

    def test_dtype(self):
        self.assertEqual(self.desired.dtype, self.actual.dtype)

    def test_shapes(self):
        evalargs = dict(zip(self.arg_names, self.arg_values))
        self.assertEqual(self.desired.shape, evaluable.eval_once(self.actual.shape, arguments=evalargs))

    def assertArrayAlmostEqual(self, actual, desired, decimal):
        if actual.dtype != desired.dtype:
            self.fail('dtypes of actual {} and desired {} are different.'.format(actual.dtype, desired.dtype))
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
            self.assertArrayAlmostEqual(evaluable.eval_once(actual, _simplify=False, _optimize=False, arguments=evalargs), desired, decimal)
        with self.subTest('simplified'):
            self.assertArrayAlmostEqual(evaluable.eval_once(actual, _simplify=True, _optimize=False, arguments=evalargs), desired, decimal)
        with self.subTest('optimized'):
            self.assertArrayAlmostEqual(evaluable.eval_once(actual, _simplify=True, _optimize=True, arguments=evalargs), desired, decimal)
        with self.subTest('sparse'):
            values, indices, shape = evaluable.eval_once(actual.simplified.assparse, arguments=evalargs)
            self.assertEqual(shape, desired.shape)
            self.assertEqual(values.ndim, 1)
            self.assertEqual(len(indices), len(shape))
            if shape:
                flatindex = numpy.ravel_multi_index(indices, shape)
                self.assertTrue(numpy.all(flatindex[1:] > flatindex[:-1]))
            else:
                self.assertEqual(len(values), 1)
            self.assertArrayAlmostEqual(numeric.accumulate(values, indices, shape), desired, decimal)
        if actual.ndim == 2:
            with self.subTest('csr'):
                values, rowptr, colidx, ncols = evaluable.eval_once(evaluable.as_csr(actual), arguments=evalargs)
                shape = len(rowptr) - 1, ncols
                self.assertEqual(shape, desired.shape)
                actual = numpy.zeros(shape, dtype=values.dtype)
                actual[numpy.concatenate([numpy.full(n, i) for i, n in enumerate(numpy.diff(rowptr))]), colidx] = values
                self.assertArrayAlmostEqual(actual, desired, decimal)

    def test_evalconst(self):
        self.assertFunctionAlmostEqual(decimal=14,
                                       desired=self.n_op(*self.arg_values),
                                       actual=self.op(*map(evaluable.constant, self.arg_values)))

    def test_evalzero(self):
        for iarg, arg_value in enumerate(self.arg_values):
            if 0 in arg_value.flat:
                args = (*self.arg_values[:iarg], numpy.zeros_like(arg_value), *self.arg_values[iarg+1:])
                self.assertFunctionAlmostEqual(decimal=14,
                                               desired=self.n_op(*args),
                                               actual=self.op(*[evaluable.zeros_like(arg) if i == iarg else arg for i, arg in enumerate(map(evaluable.constant, args))]))

    def test_eval(self):
        self.assertFunctionAlmostEqual(decimal=14,
                                       actual=self.actual,
                                       desired=self.desired)

    def test_getitem(self):
        for idim in range(self.actual.ndim):
            for item in range(self.desired.shape[idim]):
                s = (Ellipsis,) + (slice(None),)*idim + (item,) + (slice(None),)*(self.actual.ndim-idim-1)
                self.assertFunctionAlmostEqual(decimal=14,
                                               desired=self.desired[s],
                                               actual=self.actual[s])

    def test_transpose(self):
        trans = numpy.arange(self.actual.ndim, 0, -1) % self.actual.ndim
        self.assertFunctionAlmostEqual(decimal=14,
                                       desired=numpy.transpose(self.desired, trans),
                                       actual=evaluable.transpose(self.actual, trans))

    def test_insertaxis(self):
        for axis in range(self.actual.ndim+1):
            with self.subTest(axis=axis):
                self.assertFunctionAlmostEqual(decimal=14,
                                               desired=numpy.repeat(numpy.expand_dims(self.desired, axis), 2, axis),
                                               actual=evaluable.insertaxis(self.actual, axis, evaluable.constant(2)))

    def test_takediag(self):
        for ax1, ax2 in self.pairs:
            self.assertFunctionAlmostEqual(decimal=14,
                                           desired=numeric.takediag(self.desired, ax1, ax2),
                                           actual=evaluable.takediag(self.actual, ax1, ax2))

    def test_eig(self):
        if self.actual.dtype == float:
            for ax1, ax2 in self.pairs:
                A, L, V = evaluable.eval_once((self.actual, *evaluable.eig(self.actual, axes=(ax1, ax2))), arguments=dict(zip(self.arg_names, self.arg_values)))
                self.assertArrayAlmostEqual(decimal=11,
                                            actual=(numpy.expand_dims(V, ax2) * numpy.expand_dims(L, ax2+1).swapaxes(ax1, ax2+1)).sum(ax2+1),
                                            desired=(numpy.expand_dims(A, ax2) * numpy.expand_dims(V, ax2+1).swapaxes(ax1, ax2+1)).sum(ax2+1))

    def test_inv(self):
        if self.actual.dtype == bool:
            return
        for ax1, ax2 in self.pairs:
            trans = [i for i in range(self.desired.ndim) if i not in (ax1, ax2)] + [ax1, ax2]
            invtrans = list(map(trans.index, range(len(trans))))
            self.assertFunctionAlmostEqual(decimal=10,
                                           desired=numeric.inv(self.desired.transpose(trans)).transpose(invtrans),
                                           actual=evaluable.inverse(self.actual, axes=(ax1, ax2)))

    def test_determinant(self):
        if self.actual.dtype == bool:
            return
        for ax1, ax2 in self.pairs:
            self.assertFunctionAlmostEqual(decimal=11,
                                           desired=numpy.linalg.det(self.desired.transpose([i for i in range(self.desired.ndim) if i not in (ax1, ax2)] + [ax1, ax2])),
                                           actual=evaluable.determinant(self.actual, axes=(ax1, ax2)))

    def test_take(self):
        indices = [0, -1]
        for iax, sh in enumerate(self.desired.shape):
            if sh >= 2 and self.actual.shape[iax].isconstant:
                self.assertFunctionAlmostEqual(decimal=14,
                                               desired=numpy.take(self.desired, indices, axis=iax),
                                               actual=evaluable.take(self.actual, evaluable.constant(indices), axis=iax))

    def test_take_block(self):
        for iax, sh in enumerate(self.desired.shape):
            if sh >= 2:
                indices = [[0, sh-1], [sh-1, 0]]
                self.assertFunctionAlmostEqual(decimal=14,
                                               desired=numpy.take(self.desired, indices, axis=iax),
                                               actual=evaluable._take(self.actual, evaluable.constant(indices), axis=iax))

    def test_take_nomask(self):
        for iax, sh in enumerate(self.desired.shape):
            if sh >= 2:
                indices = [0, sh-1]
                self.assertFunctionAlmostEqual(decimal=14,
                                               desired=numpy.take(self.desired, indices, axis=iax),
                                               actual=evaluable.take(self.actual, evaluable.InRange(evaluable.Guard(evaluable.asarray(indices)), self.actual.shape[iax]), axis=iax))

    def test_take_reversed(self):
        indices = [-1, 0]
        for iax, sh in enumerate(self.desired.shape):
            if sh >= 2 and self.actual.shape[iax].isconstant:
                self.assertFunctionAlmostEqual(decimal=14,
                                               desired=numpy.take(self.desired, indices, axis=iax),
                                               actual=evaluable.take(self.actual, evaluable.constant(indices), axis=iax))

    def test_take_duplicate_indices(self):
        for iax, sh in enumerate(self.desired.shape):
            if sh >= 2:
                indices = [0, sh-1, 0, 0]
                self.assertFunctionAlmostEqual(decimal=14,
                                               desired=numpy.take(self.desired, indices, axis=iax),
                                               actual=evaluable.take(self.actual, evaluable.InRange(evaluable.Guard(evaluable.asarray(indices)), self.actual.shape[iax]), axis=iax))

    def test_inflate(self):
        for iax, sh in enumerate(self.desired.shape):
            dofmap = evaluable.Range(self.actual.shape[iax]) * 2
            desired = numpy.zeros(self.desired.shape[:iax] + (int(sh)*2-1,) + self.desired.shape[iax+1:], dtype=self.desired.dtype)
            desired[(slice(None),)*iax+(slice(None, None, 2),)] = self.desired
            self.assertFunctionAlmostEqual(decimal=14,
                                           desired=desired,
                                           actual=evaluable._inflate(self.actual, dofmap=dofmap, length=evaluable.constant(sh*2-1), axis=iax))

    def test_inflate_duplicate_indices(self):
        for iax, sh in enumerate(self.desired.shape):
            dofmap = evaluable.Range(self.actual.shape[iax]) % 2
            desired = numpy.zeros(self.desired.shape[:iax] + (2,) + self.desired.shape[iax+1:], dtype=self.desired.dtype)
            numpy.add.at(desired, (slice(None),)*iax+(numpy.arange(sh) % 2,), self.desired)
            self.assertFunctionAlmostEqual(decimal=14,
                                           desired=desired,
                                           actual=evaluable._inflate(self.actual, dofmap=dofmap, length=evaluable.constant(2), axis=iax))

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
                                           desired=(numpy.all if self.actual.dtype == bool else numpy.prod)(self.desired, axis=iax),
                                           actual=evaluable.product(self.actual, axis=iax))

    def test_getslice(self):
        for idim in range(self.actual.ndim):
            if self.desired.shape[idim] == 1:
                continue
            s = (Ellipsis,) + (slice(None),)*idim + (slice(0, int(self.desired.shape[idim])-1),) + (slice(None),)*(self.actual.ndim-idim-1)
            self.assertFunctionAlmostEqual(decimal=14,
                                           desired=self.desired[s],
                                           actual=self.actual[s])

    def test_sumaxis(self):
        for idim in range(self.actual.ndim):
            self.assertFunctionAlmostEqual(decimal=14,
                                           desired=(numpy.any if self.actual.dtype == bool else numpy.sum)(self.desired, axis=idim),
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
        if self.actual.dtype == bool:
            return
        for iax in range(self.actual.ndim):
            self.assertFunctionAlmostEqual(decimal=13,
                                           desired=numeric.contract(self.desired, self.other, axis=iax),
                                           actual=evaluable.dot(self.actual, self.other, axes=iax))

    def test_pointwise(self):
        if self.actual.dtype == bool:
            return
        dtype = complex if self.actual.dtype == complex else float
        self.assertFunctionAlmostEqual(decimal=14,
                                       desired=numpy.sin(self.desired).astype(dtype),  # "astype" necessary for boolean operations (float16->float64)
                                       actual=evaluable.sin(evaluable.astype(self.actual, dtype)))

    def test_power(self):
        if self.actual.dtype == bool:
            return
        self.assertFunctionAlmostEqual(decimal=13,
                                       desired=self.desired**3,
                                       actual=(self.actual**self.actual.dtype(3)))

    def test_power0(self):
        if self.actual.dtype == bool:
            return
        power = (numpy.arange(self.desired.size) % 2).reshape(self.desired.shape).astype(self.actual.dtype) if not self.varshape else 3
        self.assertFunctionAlmostEqual(decimal=13,
                                       desired=self.desired**power,
                                       actual=self.actual**power)

    def test_sign(self):
        if self.desired.dtype.kind not in ('b', 'c'):
            self.assertFunctionAlmostEqual(decimal=14,
                                           desired=numpy.sign(self.desired),
                                           actual=evaluable.sign(self.actual))

    def test_real(self):
        self.assertFunctionAlmostEqual(decimal=14,
                                       desired=numpy.real(self.desired),
                                       actual=evaluable.real(self.actual))

    def test_imag(self):
        self.assertFunctionAlmostEqual(decimal=14,
                                       desired=numpy.imag(self.desired),
                                       actual=evaluable.imag(self.actual))

    def test_conjugate(self):
        self.assertFunctionAlmostEqual(decimal=14,
                                       desired=numpy.conjugate(self.desired).astype(self.desired.dtype),
                                       actual=evaluable.conjugate(self.actual))

    def test_ravel(self):
        for idim in range(self.actual.ndim-1):
            self.assertFunctionAlmostEqual(decimal=14,
                                           desired=self.desired.reshape(self.desired.shape[:idim]+(-1,)+self.desired.shape[idim+2:]),
                                           actual=evaluable.ravel(self.actual, axis=idim))

    def test_unravel(self):
        for idim in range(self.actual.ndim):
            if not self.actual.shape[idim].isconstant:
                continue
            length = self.desired.shape[idim]
            unravelshape = (length//3, 3) if (length % 3 == 0) else (length//2, 2) if (length % 2 == 0) else (length, 1)
            self.assertFunctionAlmostEqual(decimal=14,
                                           desired=self.desired.reshape(self.desired.shape[:idim]+unravelshape+self.desired.shape[idim+1:]),
                                           actual=evaluable.unravel(self.actual, axis=idim, shape=tuple(map(evaluable.constant, unravelshape))))

    def test_loopsum(self):
        if self.desired.dtype == bool or self.varshape:
            return
        length = 3
        index = evaluable.loop_index('_testindex', length)
        for iarg, arg_value in enumerate(self.arg_values):
            testvalue = numpy.repeat(arg_value[numpy.newaxis], length, axis=0)
            numpy.random.shuffle(testvalue.ravel())
            desired = functools.reduce(operator.add, (self.n_op(*self.arg_values[:iarg], v, *self.arg_values[iarg+1:]) for v in testvalue))
            args = (*self.args[:iarg], evaluable.Guard(evaluable.get(evaluable.asarray(testvalue), 0, index)), *self.args[iarg+1:])
            self.assertFunctionAlmostEqual(decimal=13,
                                           actual=evaluable.loop_sum(self.op(*args), index),
                                           desired=desired)

    def test_loopconcatenate(self):
        if not self.desired.ndim:
            return
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
        fddeltas = numpy.array([1, 2, 3])
        fdfactors = numpy.linalg.solve(2*fddeltas**numpy.arange(1, 1+2*len(fddeltas), 2)[:, None], [1]+[0]*(len(fddeltas)-1))
        actual = evaluable.compile(self.actual)
        for arg, arg_name, x0 in zip(self.args, self.arg_names, self.arg_values):
            if arg.dtype in (bool, int):
                continue
            with self.subTest(arg_name):
                dx = numpy.random.normal(size=x0.shape)
                if arg.dtype == bool:
                    dx = dx + 1j*numpy.random.normal(size=x0.shape)
                evalargs = dict(zip(self.arg_names, self.arg_values))
                f0 = evaluable.eval_once(evaluable.derivative(self.actual, arg), arguments=evalargs)
                exact = numeric.contract(f0, dx, range(self.actual.ndim, self.actual.ndim+dx.ndim))
                if exact.dtype.kind in 'bi' or self.zerograd:
                    approx = numpy.zeros_like(exact)
                    scale = 1
                else:
                    fdvals = numpy.stack([actual(collections.ChainMap({arg_name: numpy.asarray(x0+eps*n*dx)}, evalargs)) for n in (*-fddeltas, *fddeltas)], axis=0)
                    if fdvals.dtype.kind == 'i':
                        fdvals = fdvals.astype(float)
                    fdvals = fdvals.reshape(2, len(fddeltas), *fdvals.shape[1:])
                    approx = ((fdvals[1] - fdvals[0]).T @ fdfactors).T / eps
                    scale = numpy.linalg.norm(f0.ravel()) or 1
                self.assertArrayAlmostEqual(exact / scale, approx / scale, decimal=10)

    @unittest.skipIf(sys.version_info < (3, 7), 'time.perf_counter_ns is not available')
    def test_node(self):
        # This tests only whether `Evaluable._node` returns without exception.
        cache = {}
        times = collections.defaultdict(evaluable._Stats)
        with self.subTest('new'):
            node = self.actual._node(cache, None, times, False)
            if node:
                self.assertIn(self.actual, cache)
                self.assertEqual(cache[self.actual], node)
        with self.subTest('from-cache'):
            if node:
                self.assertEqual(self.actual._node(cache, None, times, False), node)


def generate(*shape, real, imag, zero, negative):
    'generate array values that cover certain numerical classes'
    size = numpy.prod(shape, dtype=int)
    a = numpy.arange(size)
    if negative and not (real and imag):
        iz = size // 2
        a -= iz
    else:
        iz = 0
    assert a[iz] == 0
    if not zero:
        a[iz:] += 1
    if not a[-1]:  # no positive numbers
        raise Exception('shape is too small to test at least one of all selected number categories')
    if real or imag:
        a = numpy.tanh(2 * a / a[-1])  # map to (-1,1)
        if negative:
            a[:iz] -= a[iz-1] / 2 # introduce asymmetry to reduce risk of singular matrices
        if real and imag:
            assert negative
            a = a * numpy.exp(1j * numpy.arange(size)**2)
        elif imag:
            a = a * 1j
    return a.reshape(shape)


INT = functools.partial(generate, real=False, imag=False, zero=True, negative=False)
ANY = functools.partial(generate, real=True, imag=False, zero=True, negative=True)
NZ = functools.partial(generate, real=True, imag=False, zero=False, negative=True)
POS = functools.partial(generate, real=True, imag=False, zero=False, negative=False)
NN = functools.partial(generate, real=True, imag=False, zero=True, negative=False)
IM = functools.partial(generate, real=False, imag=True, zero=True, negative=True)
ANC = functools.partial(generate, real=True, imag=True, zero=True, negative=True)
NZC = functools.partial(generate, real=True, imag=True, zero=False, negative=True)


def _check(name, op, n_op, *arg_values, hasgrad=True, zerograd=False, ndim=2):
    check(name, op=op, n_op=n_op, arg_values=arg_values, hasgrad=hasgrad, zerograd=zerograd, ndim=ndim)


_check('identity', lambda f: evaluable.asarray(f), lambda a: a, ANY(2, 4, 2))
_check('int', lambda f: evaluable.astype(f, int), lambda a: a.astype(int), INT(2, 4, 2))
_check('float', lambda f: evaluable.astype(f, float), lambda a: a.astype(float), INT(2, 4, 2))
_check('complex', lambda f: evaluable.astype(f, complex), lambda a: a.astype(complex), ANY(2, 4, 2))
_check('real', lambda f: evaluable.real(f), lambda a: a.real, ANY(2, 4, 2))
_check('real-complex', lambda f: evaluable.real(f), lambda a: a.real, ANC(2, 4, 2), hasgrad=False)
_check('imag', lambda f: evaluable.imag(f), lambda a: a.imag, ANY(2, 4, 2))
_check('imag-complex', lambda f: evaluable.imag(f), lambda a: a.imag, ANC(2, 4, 2), hasgrad=False)
_check('conjugate', lambda f: evaluable.conjugate(f), lambda a: a.conjugate(), ANY(2, 4, 2))
_check('conjugate-complex', lambda f: evaluable.conjugate(f), lambda a: a.conjugate(), ANC(2, 4, 2), hasgrad=False)
_check('const', lambda: evaluable.constant(ANY(2, 4, 2)), lambda: ANY(2, 4, 2))
_check('zeros', lambda: evaluable.zeros(tuple(map(evaluable.constant, [1, 4, 3, 4]))), lambda: numpy.zeros([1, 4, 3, 4]))
_check('zeros-bool', lambda: evaluable.zeros(tuple(map(evaluable.constant, [1, 4, 3, 4])), dtype=bool), lambda: numpy.zeros([1, 4, 3, 4], dtype=bool))
_check('ones', lambda: evaluable.ones(tuple(map(evaluable.constant, [1, 4, 3, 4]))), lambda: numpy.ones([1, 4, 3, 4]))
_check('ones-bool', lambda: evaluable.ones(tuple(map(evaluable.constant, [1, 4, 3, 4])), dtype=bool), lambda: numpy.ones([1, 4, 3, 4], dtype=bool))
_check('range', lambda: evaluable.Range(evaluable.constant(4)) + 2, lambda: numpy.arange(2, 6))
_check('sin', evaluable.sin, numpy.sin, ANY(4, 4))
_check('sin-complex', evaluable.sin, numpy.sin, ANC(4, 4))
_check('cos', evaluable.cos, numpy.cos, ANY(4, 4))
_check('cos-complex', evaluable.cos, numpy.cos, ANC(4, 4))
_check('tan', evaluable.tan, numpy.tan, ANY(4, 4))
_check('tan-complex', evaluable.tan, numpy.tan, ANC(4, 4))
_check('sinc', evaluable.sinc, lambda a: numpy.sinc(a/numpy.pi), ANY(4, 4))
_check('sinc-complex', evaluable.sinc, lambda a: numpy.sinc(a/numpy.pi), ANC(4, 4))
_check('sqrt', evaluable.sqrt, lambda a: numpy.power(a, .5), NN(4, 4)) # NOTE: not comparing against numpy.sqrt because of AVX-512 related accuracy issues, see #770
_check('sqrt-complex', evaluable.sqrt, numpy.sqrt, ANC(4, 4))
_check('log', evaluable.ln, numpy.log, POS(2, 2))
_check('log-complex', evaluable.ln, numpy.log, NZC(2, 2))
_check('log2', evaluable.log2, numpy.log2, POS(2, 2))
_check('log2-complex', evaluable.log2, lambda a: numpy.log(a) * numpy.power(numpy.log(2), -1), NZC(2, 2)) # NOTE: not comparing against numpy.log2 because of AVX-512 related accuracy issues, see #770
_check('log10', evaluable.log10, numpy.log10, POS(2, 2))
_check('log10-complex', evaluable.log10, numpy.log10, NZC(2, 2))
_check('exp', evaluable.exp, numpy.exp, ANY(4, 4))
_check('exp-complex', evaluable.exp, numpy.exp, ANC(4, 4))
_check('arctanh', evaluable.arctanh, numpy.arctanh, ANY(3, 3))
_check('arctanh-complex', evaluable.arctanh, numpy.arctanh, ANC(3, 3))
_check('tanh', evaluable.tanh, numpy.tanh, ANY(4, 4))
_check('tanh-complex', evaluable.tanh, numpy.tanh, ANC(4, 4))
_check('cosh', evaluable.cosh, numpy.cosh, ANY(4, 4))
_check('cosh-complex', evaluable.cosh, numpy.cosh, ANC(4, 4))
_check('sinh', evaluable.sinh, numpy.sinh, ANY(4, 4))
_check('sinh-complex', evaluable.sinh, numpy.sinh, ANC(4, 4))
_check('abs', evaluable.abs, numpy.abs, ANY(4, 4))
_check('abs-complex', evaluable.abs, numpy.abs, .2-.2j+ANC(4, 4), hasgrad=False)
_check('sign', evaluable.sign, numpy.sign, ANY(4, 4), zerograd=True)
_check('power', evaluable.power, numpy.power, POS(4, 4), ANY(4, 4))
_check('power-complex', evaluable.power, numpy.power, NZC(4, 4), (2-2j)*ANC(4, 4), hasgrad=False)
_check('negative', evaluable.negative, numpy.negative, ANY(4, 4))
_check('negative-complex', evaluable.negative, numpy.negative, ANC(4, 4))
_check('reciprocal', evaluable.reciprocal, numpy.reciprocal, NZ(4, 4))
_check('reciprocal-complex', evaluable.reciprocal, numpy.reciprocal, 10*NZC(4, 4))
_check('arcsin', evaluable.arcsin, numpy.arcsin, ANY(4, 4))
_check('arcsin-complex', evaluable.arcsin, numpy.arcsin, ANC(4, 4))
_check('arccos', evaluable.arccos, numpy.arccos, ANY(4, 4))
_check('arccos-complex', evaluable.arccos, numpy.arccos, ANC(4, 4))
_check('arctan', evaluable.arctan, numpy.arctan, ANY(4, 4))
_check('arctan-complex', evaluable.arctan, numpy.arctan, ANC(4, 4))
_check('ln', evaluable.ln, numpy.log, POS(4, 4))
_check('ln-complex', evaluable.ln, numpy.log, NZC(4, 4))
_check('product', lambda a: evaluable.product(a, 2), lambda a: numpy.prod(a, 2), ANY(4, 3, 4))
_check('product-complex', lambda a: evaluable.product(a, 2), lambda a: numpy.prod(a, 2), ANC(4, 3, 4))
_check('sum', lambda a: evaluable.sum(a, 2), lambda a: a.sum(2), ANY(4, 3, 4))
_check('sum-complex', lambda a: evaluable.sum(a, 2), lambda a: a.sum(2), ANC(4, 3, 4))
_check('transpose1', lambda a: evaluable.transpose(a, [0, 1, 3, 2]), lambda a: a.transpose([0, 1, 3, 2]), ANY(2, 3, 4, 5))
_check('transpose2', lambda a: evaluable.transpose(a, [0, 2, 3, 1]), lambda a: a.transpose([0, 2, 3, 1]), ANY(2, 3, 4, 5))
_check('insertaxis', lambda a: evaluable.insertaxis(a, 1, evaluable.constant(3)), lambda a: numpy.repeat(a[:, None], 3, 1), ANY(2, 4))
_check('get', lambda a: evaluable.get(a, 2, evaluable.constant(1)), lambda a: a[:, :, 1], ANY(4, 3, 4))
_check('takediag141', lambda a: evaluable.takediag(a, 0, 2), lambda a: numeric.takediag(a, 0, 2), ANY(1, 4, 1))
_check('takediag434', lambda a: evaluable.takediag(a, 0, 2), lambda a: numeric.takediag(a, 0, 2), ANY(4, 3, 4))
_check('takediag343', lambda a: evaluable.takediag(a, 0, 2), lambda a: numeric.takediag(a, 0, 2), ANY(3, 4, 3))
_check('determinant141', lambda a: evaluable.determinant(a, (0, 2)), lambda a: numpy.linalg.det(a.swapaxes(0, 1)), ANY(1, 4, 1))
_check('determinant141-complex', lambda a: evaluable.determinant(a, (0, 2)), lambda a: numpy.linalg.det(a.swapaxes(0, 1)), ANC(1, 4, 1))
_check('determinant434', lambda a: evaluable.determinant(a, (0, 2)), lambda a: numpy.linalg.det(a.swapaxes(0, 1)), ANY(4, 3, 4))
_check('determinant434-complex', lambda a: evaluable.determinant(a, (0, 2)), lambda a: numpy.linalg.det(a.swapaxes(0, 1)), ANC(4, 3, 4))
_check('determinant4433', lambda a: evaluable.determinant(a, (2, 3)), lambda a: numpy.linalg.det(a), ANY(4, 4, 3, 3))
_check('determinant4433-complex', lambda a: evaluable.determinant(a, (2, 3)), lambda a: numpy.linalg.det(a), ANC(4, 4, 3, 3))
_check('determinant200', lambda a: evaluable.determinant(a, (1, 2)), lambda a: numpy.linalg.det(a) if a.shape[-1] else numpy.ones(a.shape[:-2], float), numpy.empty((2, 0, 0)), zerograd=True)
_check('determinant200-complex', lambda a: evaluable.determinant(a, (1, 2)), lambda a: numpy.linalg.det(a) if a.shape[-1] else numpy.ones(a.shape[:-2], complex), numpy.empty((2, 0, 0), dtype=complex), zerograd=True)
_check('inverse141', lambda a: evaluable.inverse(a, (0, 2)), lambda a: numpy.linalg.inv(a.swapaxes(0, 1)).swapaxes(0, 1), NZ(1, 4, 1))
_check('inverse141-complex', lambda a: evaluable.inverse(a, (0, 2)), lambda a: numpy.linalg.inv(a.swapaxes(0, 1)).swapaxes(0, 1), NZC(1, 4, 1))
_check('inverse434', lambda a: evaluable.inverse(a, (0, 2)), lambda a: numpy.linalg.inv(a.swapaxes(0, 1)).swapaxes(0, 1), POS(4, 3, 4)+numpy.eye(4, 4)[:, numpy.newaxis, :])
_check('inverse434-complex', lambda a: evaluable.inverse(a, (0, 2)), lambda a: numpy.linalg.inv(a.swapaxes(0, 1)).swapaxes(0, 1), ANC(4, 3, 4)+numpy.eye(4, 4)[:, numpy.newaxis, :])
_check('inverse4422', lambda a: evaluable.inverse(a), lambda a: numpy.linalg.inv(a), POS(4, 4, 2, 2)+numpy.eye(2))
_check('inverse4422-complex', lambda a: evaluable.inverse(a), lambda a: numpy.linalg.inv(a), ANC(4, 4, 2, 2)+numpy.eye(2))
_check('repeat', lambda a: evaluable.repeat(a, evaluable.constant(3), 1), lambda a: numpy.repeat(a, 3, 1), ANY(4, 1, 4))
_check('diagonalize', lambda a: evaluable.diagonalize(a, 1, 3), lambda a: numeric.diagonalize(a, 1, 3), ANY(4, 4, 4, 4))
_check('multiply', evaluable.multiply, numpy.multiply, ANY(4, 4), ANY(4, 4))
_check('multiply-complex', evaluable.multiply, numpy.multiply, ANC(4, 4), 1-1j+ANC(4, 4))
_check('dot', lambda a, b: evaluable.dot(a, b, axes=1), lambda a, b: (a*b).sum(1), ANY(4, 2, 4), ANY(4, 2, 4))
_check('divide', evaluable.divide, lambda a, b: a * b**-1, ANY(4, 4), NZ(4, 4))
_check('divide2', lambda a: evaluable.asarray(a)/2., lambda a: a/2., ANY(4, 1))
_check('divide-complex', evaluable.divide, lambda a, b: a * b**-1, ANC(4, 4), NZC(4, 4).T)
_check('add', evaluable.add, numpy.add, ANY(4, 4), ANY(4, 4))
_check('add-complex', evaluable.add, numpy.add, ANC(4, 4), numpy.exp(.2j)*ANC(4, 4))
_check('subtract', evaluable.subtract, numpy.subtract, ANY(4, 4), ANY(4, 4))
_check('subtract-complex', evaluable.subtract, numpy.subtract, ANC(4, 4), numpy.exp(.2j)*ANC(4, 4))
_check('mulsum', lambda a, b: evaluable.multiply(a, b).sum(-2), lambda a, b: (a*b).sum(-2), ANY(4, 2, 4), ANY(4, 2, 4))
_check('min', lambda a, b: evaluable.Minimum(a, b), numpy.minimum, ANY(4, 4), ANY(4, 4))
_check('max', lambda a, b: evaluable.Maximum(a, b), numpy.maximum, ANY(4, 4), ANY(4, 4))
_check('equal', evaluable.Equal, numpy.equal, ANY(4, 4), ANY(4, 4), zerograd=True)
_check('greater', evaluable.Greater, numpy.greater, ANY(4, 4), ANY(4, 4), zerograd=True)
_check('less', evaluable.Less, numpy.less, ANY(4, 4), ANY(4, 4), zerograd=True)
_check('logical_and', evaluable.multiply, numpy.logical_and, numpy.array([[False, False], [True, True]], dtype=bool), numpy.array([[False, True], [False, True]], dtype=bool))
_check('logical_or', evaluable.add, numpy.logical_or, numpy.array([[False, False], [True, True], [False, True]], dtype=bool), numpy.array([[False, True], [False, True], [True, False]], dtype=bool))
_check('logical_not', evaluable.LogicalNot, numpy.logical_not, numpy.array([[False, False], [True, True], [False, True]], dtype=bool))
_check('logical_any', evaluable.Sum, lambda a: numpy.any(a, axis=-1), numpy.array([[False, False], [True, True], [False, True]], dtype=bool))
_check('logical_all', evaluable.Product, lambda a: numpy.all(a, axis=-1), numpy.array([[False, False], [True, True], [False, True]], dtype=bool))
_check('arctan2', evaluable.arctan2, numpy.arctan2, ANY(4, 4), ANY(4, 4))
_check('stack', lambda a, b: evaluable.stack([a, b], 0), lambda a, b: numpy.concatenate([a[numpy.newaxis, :], b[numpy.newaxis, :]], axis=0), ANY(4), ANY(4))
_check('eig', lambda a: evaluable.eig(a+a.swapaxes(0, 1), symmetric=True)[1], lambda a: numpy.linalg.eigh(a+a.swapaxes(0, 1))[1], ANY(4, 4), hasgrad=False)
_check('eig-complex', lambda a: evaluable.eig(a+a.swapaxes(0, 1))[1], lambda a: numpy.linalg.eig(a+a.swapaxes(0, 1))[1], ANC(4, 4), hasgrad=False)
_check('mod', lambda a, b: evaluable.mod(a, b), lambda a, b: numpy.mod(a, b), ANY(4), NZ(4), hasgrad=False)
_check('ravel', lambda f: evaluable.ravel(f, axis=1), lambda a: a.reshape(4, 4, 4, 4), ANY(4, 2, 2, 4, 4))
_check('unravel', lambda f: evaluable.unravel(f, axis=1, shape=[evaluable.constant(2), evaluable.constant(2)]), lambda a: a.reshape(4, 2, 2, 4, 4), ANY(4, 4, 4, 4))
_check('ravelindex', lambda a, b: evaluable.RavelIndex(a, b, evaluable.constant(12), evaluable.constant(20)), lambda a, b: a[..., numpy.newaxis, numpy.newaxis] * 20 + b, INT(3, 4), INT(4, 5))
_check('inflate', lambda f: evaluable._inflate(f, dofmap=evaluable.InRange(evaluable.Guard(evaluable.constant([0, 3])), evaluable.constant(4)), length=evaluable.constant(4), axis=1), lambda a: numpy.concatenate([a[:, :1], numpy.zeros_like(a), a[:, 1:]], axis=1), ANY(4, 2, 4))
_check('inflate-constant', lambda f: evaluable._inflate(f, dofmap=evaluable.constant([0, 3]), length=evaluable.constant(4), axis=1), lambda a: numpy.concatenate([a[:, :1], numpy.zeros_like(a), a[:, 1:]], axis=1), ANY(4, 2, 4))
_check('inflate-duplicate', lambda f: evaluable.Inflate(f, dofmap=evaluable.constant([0, 1, 0, 3]), length=evaluable.constant(4)), lambda a: numpy.stack([a[:, 0]+a[:, 2], a[:, 1], numpy.zeros_like(a[:, 0]), a[:, 3]], axis=1), ANY(2, 4))
_check('inflate-block', lambda f: evaluable.Inflate(f, dofmap=evaluable.constant([[5, 4, 3], [2, 1, 0]]), length=evaluable.constant(6)), lambda a: a.ravel()[::-1], ANY(2, 3))
_check('inflate-scalar', lambda f: evaluable.Inflate(f, dofmap=evaluable.constant(1), length=evaluable.constant(3)), lambda a: numpy.array([0, a, 0]), numpy.array(.5))
_check('inflate-diagonal', lambda f: evaluable.Inflate(evaluable.Inflate(f, evaluable.constant(1), evaluable.constant(3)), evaluable.constant(1), evaluable.constant(3)), lambda a: numpy.diag(numpy.array([0, a, 0])), numpy.array(.5))
_check('inflate-one', lambda f: evaluable.Inflate(f, evaluable.constant(0), evaluable.constant(1)), lambda a: numpy.array([a]), numpy.array(.5))
_check('inflate-range', lambda f: evaluable.Inflate(f, evaluable.Range(evaluable.constant(3)), evaluable.constant(3)), lambda a: a, ANY(3))
_check('inflate-twice', lambda f: evaluable._inflate(evaluable.Inflate(f, evaluable.constant([[0,1],[2,0]]), evaluable.constant(3)), evaluable.constant([0, 1, 0]), evaluable.constant(2), axis=0), lambda a: numpy.einsum('ij,jklm,lmn->ikn', [[1,0,1],[0,1,0]], a, [[[1,0,0],[0,1,0]],[[0,0,1],[1,0,0]]]), ANY(3,5,2,2))
_check('take', lambda f: evaluable.Take(f, evaluable.constant([0, 3, 2])), lambda a: a[:, [0, 3, 2]], ANY(2, 4))
_check('take-duplicate', lambda f: evaluable.Take(f, evaluable.constant([0, 3, 0])), lambda a: a[:, [0, 3, 0]], ANY(2, 4))
_check('choose', lambda a, b, c: evaluable.Choose(a % 2, evaluable.stack([b, c], -1)), lambda a, b, c: numpy.choose(a % 2, [b, c]), INT(3, 3), ANY(3, 3), ANY(3, 3))
_check('slice', lambda a: evaluable.asarray(a)[::2], lambda a: a[::2], ANY(5, 3))
_check('normal1d', evaluable.Orthonormal, lambda G, a: numpy.sign(a), numpy.zeros([3,1,0]), NZ(3, 1))
_check('normal2d', evaluable.Orthonormal, lambda G, a: numeric.normalize(a - numeric.normalize(G[...,0]) * ((a * numeric.normalize(G[...,0])).sum(-1))[...,numpy.newaxis]), POS(3, 2, 1), ANY(3, 2))
_check('normal3d', evaluable.Orthonormal, lambda G, a: numeric.normalize(a - numpy.einsum('pij,pj->pi', G, numpy.linalg.solve(numpy.einsum('pki,pkj->pij', G, G), numpy.einsum('pij,pi->pj', G, a)[...,numpy.newaxis])[...,0])), POS(2, 3, 2) + numpy.eye(3)[:,:2], ANY(2, 3))
_check('normalmanifold', evaluable.Orthonormal, lambda G, a: numeric.normalize(a - numpy.einsum('pij,pj->pi', G, numpy.linalg.solve(numpy.einsum('pki,pkj->pij', G, G), numpy.einsum('pij,pi->pj', G, a)[...,numpy.newaxis])[...,0])), POS(2, 3, 1), ANY(2, 3))
_check('loopsum1', lambda: evaluable.loop_sum(evaluable.loop_index('index', 3), evaluable.loop_index('index', 3)), lambda: numpy.array(3))
_check('loopsum2', lambda a: evaluable.loop_sum(a, evaluable.loop_index('index', 2)), lambda a: 2*a, ANY(3, 4, 2, 4))
_check('loopsum3', lambda a: evaluable.loop_sum(evaluable.get(a, 0, evaluable.loop_index('index', 3)), evaluable.loop_index('index', 3)), lambda a: numpy.sum(a, 0), ANY(3, 4, 2, 4) + 10 * numpy.eye(4)[:,numpy.newaxis])
_check('loopsum4', lambda: evaluable.loop_sum(evaluable.Inflate(evaluable.loop_index('index', 3), evaluable.constant(0), evaluable.constant(2)), evaluable.loop_index('index', 3)), lambda: numpy.array([3, 0]))
_check('loopsum5', lambda: evaluable.loop_sum(evaluable.loop_index('index', 1), evaluable.loop_index('index', 1)), lambda: numpy.array(0))
_check('loopsum6', lambda: evaluable.loop_sum(evaluable.Guard(evaluable.constant(1) + evaluable.loop_index('index', 4)), evaluable.loop_index('index', 4)) * evaluable.loop_sum(evaluable.loop_index('index', 4), evaluable.loop_index('index', 4)), lambda: numpy.array(60))
_check('loopconcatenate1', lambda a: evaluable.loop_concatenate(a+evaluable.prependaxes(evaluable.astype(evaluable.loop_index('index', 3), float), a.shape), evaluable.loop_index('index', 3)), lambda a: a+numpy.arange(3)[None], ANY(3, 1))
_check('loopconcatenate2', lambda: evaluable.loop_concatenate(evaluable.Elemwise(tuple(types.arraydata(numpy.arange(48).reshape(4, 4, 3)[:, :, a:b]) for a, b in util.pairwise([0, 2, 3])), evaluable.loop_index('index', 2), int), evaluable.loop_index('index', 2)), lambda: numpy.arange(48).reshape(4, 4, 3))
_check('legendre', lambda a: evaluable.Legendre(evaluable.asarray(a), 5), lambda a: numpy.moveaxis(numpy.polynomial.legendre.legval(a, numpy.eye(6)), 0, -1), ANY(3, 4, 3))

_check('polyval_1d_p0', lambda c, x: evaluable.Polyval(c, x), poly.eval_outer, POS(1), ANY(4, 1), ndim=1)
_check('polyval_1d_p1', lambda c, x: evaluable.Polyval(c, x), poly.eval_outer, NZ(2), ANY(4, 1), ndim=1)
_check('polyval_1d_p2', lambda c, x: evaluable.Polyval(c, x), poly.eval_outer, ANY(3), ANY(4, 1), ndim=1)
_check('polyval_2d_p0', lambda c, x: evaluable.Polyval(c, x), poly.eval_outer, POS(1), ANY(4, 2), ndim=2)
_check('polyval_2d_p1', lambda c, x: evaluable.Polyval(c, x), poly.eval_outer, ANY(3), ANY(4, 2), ndim=2)
_check('polyval_2d_p2', lambda c, x: evaluable.Polyval(c, x), poly.eval_outer, ANY(6), ANY(4, 2), ndim=2)
_check('polyval_2d_p1_23', lambda c, x: evaluable.Polyval(c, x), poly.eval_outer, ANY(2, 3, 3), ANY(4, 2), ndim=2)
_check('polymul_x3yz1', lambda l, r: evaluable.PolyMul(l, r, (poly.MulVar.Left, poly.MulVar.Right, poly.MulVar.Right)), lambda l, r: poly.mul(l, r, (poly.MulVar.Left, poly.MulVar.Right, poly.MulVar.Right)), ANY(4, 4, 4), ANY(4, 4, 3), hasgrad=False)
_check('polymul_x2y0', lambda l, r: evaluable.PolyMul(l, r, (poly.MulVar.Left, poly.MulVar.Right)), lambda l, r: poly.mul(l, r, (poly.MulVar.Left, poly.MulVar.Right)), ANY(4, 4, 3), ANY(4, 4, 1), hasgrad=False)
_check('polygrad_xy0', lambda c: evaluable.PolyGrad(c, 2), lambda c: poly.grad(c, 2), ANY(4, 1), hasgrad=False)
_check('polygrad_xy1', lambda c: evaluable.PolyGrad(c, 2), lambda c: poly.grad(c, 2), ANY(2, 2, 3), hasgrad=False)
_check('polygrad_xy2', lambda c: evaluable.PolyGrad(c, 2), lambda c: poly.grad(c, 2), ANY(4, 4, 6), hasgrad=False)

_check('searchsorted', lambda a: evaluable.SearchSorted(evaluable.asarray(a), array=evaluable.constant(numpy.linspace(0, 1, 9)), side='left', sorter=None), lambda a: numpy.searchsorted(numpy.linspace(0, 1, 9), a).astype(int), POS(4, 2))
_check('searchsorted_sorter', lambda a: evaluable.SearchSorted(evaluable.asarray(a), array=evaluable.constant([.2,.8,.4,0,.6,1]), side='left', sorter=evaluable.constant([3,0,2,4,1,5])), lambda a: numpy.searchsorted([.2,.8,.4,0,.6,1], a, sorter=[3,0,2,4,1,5]).astype(int), POS(4, 2))
_check('argsort', evaluable.ArgSort, lambda a: numpy.argsort(a, axis=-1, kind='stable').astype(int), ANY(3, 9))
_check('unique', evaluable.unique, numpy.unique, numpy.arange(10) % 3)
_check('unique-index', lambda a: evaluable.unique(a, return_index=True)[1], lambda a: numpy.unique(a, return_index=True)[1].astype(int), numpy.arange(10) % 3)
_check('unique-inverse', lambda a: evaluable.unique(a, return_inverse=True)[1], lambda a: numpy.unique(a, return_inverse=True)[1].astype(int), numpy.arange(10) % 3)


class compile(TestCase):

    def test_array_arg(self):
        a = evaluable.Argument('a', (), int)
        f = evaluable.compile(a)
        self.assertEqual(f(dict(a=1)), 1)

    def test_tuple_arg(self):
        a = evaluable.Argument('a', (), int)
        b = evaluable.Argument('b', (), int)
        f = evaluable.compile((a, b))
        self.assertEqual(f(dict(a=1, b=2)), (1, 2))

    def test_nested_arg(self):
        a = evaluable.Argument('a', (), int)
        b = evaluable.Argument('b', (), int)
        c = evaluable.Argument('c', (), int)
        f = evaluable.compile((a, (b, c)))
        self.assertEqual(f(dict(a=1, b=2, c=3)), (1, (2, 3)))

    def test_stats(self):
        a = evaluable.Argument('a', (), int)
        f = evaluable.compile(a, stats='log')
        with self.assertLogs('nutils', logging.INFO) as cm:
            f(dict(a=1))
            self.assertTrue(cm.output[0].startswith('INFO:nutils:total time:'))


class intbounds(TestCase):

    @staticmethod
    def R(start, shape):
        # A range of numbers starting at `start` with the given `shape`.
        if isinstance(shape, int):
            size = shape
            shape = shape,
        else:
            size = util.product(shape)
        return evaluable.constant(numpy.arange(start, start+size).reshape(*shape))

    class S(evaluable.Array):
        # An evaluable scalar argument with given bounds.
        argname: str
        lower: int
        upper: int

        dtype = int
        shape = ()

        @property
        def dependencies(self):
            return evaluable.Argument(self.argname, shape=(), dtype=int),

        def _compile(self, builder):
            value, = builder.compile(self.dependencies)
            block = builder.get_block_for_evaluable(self)
            block.assert_true(_pyast.BinOp(_pyast.LiteralInt(self.lower), '<=', value.get_item(_pyast.Tuple(()))))
            block.assert_true(_pyast.BinOp(value.get_item(_pyast.Tuple(())), '<=', _pyast.LiteralInt(self.upper)))
            return value

        @property
        def _intbounds(self):
            return self.lower, self.upper

    def assertBounds(self, func, *, tight_lower=True, tight_upper=True, **evalargs):
        lower, upper = func._intbounds
        value = evaluable.eval_once(func, arguments=evalargs)
        (self.assertEqual if tight_lower else self.assertLessEqual)(lower, value.min())
        (self.assertEqual if tight_upper else self.assertGreaterEqual)(upper, value.max())

    def test_default(self):
        class Test(evaluable.Array):
            dtype = int
            shape = ()

            @property
            def dependencies(self):
                return evaluable.Argument('dummy', (), int),

        self.assertEqual(Test()._intbounds, (float('-inf'), float('inf')))

    def test_constant(self):
        self.assertEqual(self.R(-4, [2, 3, 4])._intbounds, (-4, 19))

    def test_constant_empty(self):
        self.assertEqual(self.R(0, [0])._intbounds, (float('-inf'), float('inf')))

    def test_insertaxis(self):
        arg = self.R(-4, [2, 3, 4])
        self.assertEqual(evaluable.InsertAxis(arg, evaluable.constant(2))._intbounds, arg._intbounds)

    def test_transpose(self):
        arg = self.R(-4, [2, 3, 4])
        self.assertEqual(evaluable.Transpose(arg, (2, 0, 1))._intbounds, arg._intbounds)

    def test_multiply(self):
        args = tuple(self.R(low, [high+1-low]) for low, high in ((-13, -5), (-2, 7), (3, 11)))
        for arg1 in args:
            for arg2 in args:
                self.assertBounds(evaluable.multiply(evaluable.insertaxis(arg1, 1, arg2.shape[0]), evaluable.insertaxis(arg2, 0, arg1.shape[0])))

    def test_add(self):
        self.assertBounds(evaluable.add(evaluable.insertaxis(self.R(-5, [8]), 1, evaluable.constant(5)), evaluable.insertaxis(self.R(2, [5]), 0, evaluable.constant(8))))

    def test_sum_zero_axis(self):
        self.assertEqual(evaluable.Sum(self.R(0, [0]))._intbounds, (0, 0))

    def test_sum_variable_axis_including_zero(self):
        self.assertEqual(evaluable.Sum(evaluable.Argument('test', (self.S('n', 0, 4),), int))._intbounds, (float('-inf'), float('inf')))

    def test_sum_zero_size(self):
        self.assertEqual(evaluable.Sum(self.R(0, [2, 3, 0]))._intbounds, (0, 0))

    def test_sum_nonzero(self):
        self.assertBounds(evaluable.Sum(self.R(-3, [9, 1])))

    def test_sum_unknown(self):
        func = lambda l, h: evaluable.Sum(evaluable.InsertAxis(self.R(l, [h+1-l]), self.S('n', 2, 5)))
        self.assertBounds(func(-3, 5), n=5)
        self.assertBounds(func(-3, 5), n=5, tight_lower=False, tight_upper=False)
        self.assertBounds(func(3, 5), n=5, tight_lower=False)
        self.assertBounds(func(3, 5), n=2, tight_upper=False)
        self.assertBounds(func(-3, -2), n=5, tight_upper=False)
        self.assertBounds(func(-3, -2), n=2, tight_lower=False)

    def test_takediag(self):
        arg = self.R(-4, [2, 3, 3])
        self.assertEqual(evaluable.TakeDiag(arg)._intbounds, arg._intbounds)

    def test_take(self):
        arg = self.R(-4, [2, 3, 4])
        idx = self.R(0, [1])
        self.assertEqual(evaluable.Take(arg, idx)._intbounds, arg._intbounds)

    def test_negative(self):
        self.assertBounds(evaluable.Negative(self.R(-4, [2, 3, 4])))

    def test_absolute_negative(self):
        self.assertBounds(evaluable.Absolute(self.R(-4, [3])))

    def test_absolute_positive(self):
        self.assertBounds(evaluable.Absolute(self.R(1, [3])))

    def test_absolute_full(self):
        self.assertBounds(evaluable.Absolute(self.R(-3, [7])))

    def test_mod_nowrap(self):
        self.assertBounds(evaluable.Mod(evaluable.insertaxis(self.R(1, [4]), 1, evaluable.constant(3)), evaluable.insertaxis(self.R(5, [3]), 0, evaluable.constant(4))))

    def test_mod_wrap_negative(self):
        self.assertBounds(evaluable.Mod(evaluable.insertaxis(self.R(-3, [7]), 1, evaluable.constant(3)), evaluable.insertaxis(self.R(5, [3]), 0, evaluable.constant(7))))

    def test_mod_wrap_positive(self):
        self.assertBounds(evaluable.Mod(evaluable.insertaxis(self.R(3, [7]), 1, evaluable.constant(3)), evaluable.insertaxis(self.R(5, [3]), 0, evaluable.constant(7))))

    def test_mod_negative_divisor(self):
        self.assertEqual(evaluable.Mod(evaluable.Argument('d', (evaluable.constant(2),), int), self.R(-3, [2]))._intbounds, (float('-inf'), float('inf')))

    def test_floordiv(self):
        self.assertBounds(evaluable.FloorDivide(evaluable.insertaxis(self.R(2, [9]), 1, evaluable.constant(2)), evaluable.insertaxis(self.R(3, [2]), 0, evaluable.constant(9))))
        self.assertBounds(evaluable.FloorDivide(evaluable.insertaxis(self.R(-12, [9]), 1, evaluable.constant(2)), evaluable.insertaxis(self.R(3, [2]), 0, evaluable.constant(9))))
        self.assertBounds(evaluable.FloorDivide(evaluable.insertaxis(self.R(-15, [9]), 1, evaluable.constant(2)), evaluable.insertaxis(self.R(-3, [2]), 0, evaluable.constant(9))))
        self.assertBounds(evaluable.FloorDivide(evaluable.insertaxis(self.R(-4, [9]), 1, evaluable.constant(2)), evaluable.insertaxis(self.R(8, [2]), 0, evaluable.constant(9))))
        self.assertEqual(evaluable.FloorDivide(self.S('dividend', 2, 4), self.S('divisor', float('-inf'), float('inf')))._intbounds, (float('-inf'), float('inf')))

    def test_sign(self):
        for i in range(-2, 3):
            for j in range(i, 3):
                self.assertBounds(evaluable.Sign(self.R(i, [j-i+1])))

    def test_zeros(self):
        self.assertEqual(evaluable.Zeros((evaluable.constant(2), evaluable.constant(3)), int)._intbounds, (0, 0))

    def test_range(self):
        self.assertEqual(evaluable.Range(self.S('n', 0, 0))._intbounds, (0, 0))
        self.assertBounds(evaluable.Range(self.S('n', 1, 3)), n=3)

    def test_inrange_loose(self):
        self.assertEqual(evaluable.InRange(self.S('n', 3, 5), evaluable.constant(6))._intbounds, (3, 5))

    def test_inrange_strict(self):
        self.assertEqual(evaluable.InRange(self.S('n', float('-inf'), float('inf')), self.S('m', 2, 4))._intbounds, (0, 3))

    def test_inrange_empty(self):
        self.assertEqual(evaluable.InRange(self.S('n', float('-inf'), float('inf')), evaluable.constant(0))._intbounds, (0, 0))

    def test_bool_to_int(self):
        self.assertEqual(evaluable.BoolToInt(evaluable.constant(numpy.array([False, True], dtype=bool)))._intbounds, (0, 1))

    def test_array_from_tuple(self):
        A = self.S('A', 2, 4)
        B = self.S('B', -5, 3)
        tpl = evaluable.Tuple((A, B))
        self.assertEqual(evaluable.ArrayFromTuple(tpl, 0, A.shape, A.dtype)._intbounds, (2, 4))
        self.assertEqual(evaluable.ArrayFromTuple(tpl, 1, B.shape, B.dtype)._intbounds, (-5, 3))

    def test_inflate(self):
        self.assertEqual(evaluable.Inflate(self.R(4, (2, 3)), evaluable.constant(numpy.arange(6).reshape(2, 3)), evaluable.constant(7))._intbounds, (0, 9))

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


class commutativity(TestCase):

    def setUp(self):
        super().setUp()
        numpy.random.seed(0)
        self.A = evaluable.asarray(numpy.random.uniform(size=[2, 3]))
        self.B = evaluable.asarray(numpy.random.uniform(size=[2, 3]))

    def test_add(self):
        self.assertEqual(evaluable.add(self.A, self.B), evaluable.add(self.B, self.A))

    def test_multiply(self):
        self.assertEqual(evaluable.multiply(self.A, self.B), evaluable.multiply(self.B, self.A))

    def test_dot(self):
        self.assertEqual(evaluable.dot(self.A, self.B, axes=[0]), evaluable.dot(self.B, self.A, axes=[0]))

    def test_combined(self):
        self.assertEqual(evaluable.add(self.A, self.B) * evaluable.insertaxis(evaluable.dot(self.A, self.B, axes=[0]), 0, evaluable.constant(2)),
                         evaluable.insertaxis(evaluable.dot(self.B, self.A, axes=[0]), 0, evaluable.constant(2)) * evaluable.add(self.B, self.A))


class sampled(TestCase):

    def test_match(self):
        f = evaluable.Sampled(evaluable.constant([[1, 2], [3, 4]]), evaluable.constant([[1, 2], [3, 4]]), 'none')
        self.assertAllEqual(evaluable.eval_once(f), numpy.eye(2))

    def test_no_match(self):
        f = evaluable.Sampled(evaluable.constant([[1, 2], [3, 4]]), evaluable.constant([[3, 4], [1, 2]]), 'none')
        with self.assertRaises(Exception):
            evaluable.eval_once(f)


class elemwise(TestCase):

    def assertElemwise(self, items):
        items = tuple(map(types.arraydata, items))
        index = evaluable.InRange(evaluable.Argument('index', (), int), evaluable.constant(len(items)))
        elemwise = evaluable.compile(evaluable.Elemwise(items, index, int))
        for i, item in enumerate(items):
            self.assertEqual(elemwise(dict(index=i)).tolist(), numpy.asarray(item).tolist())

    def test_const_values(self):
        self.assertElemwise((numpy.arange(2*3*4).reshape(2, 3, 4),)*3)

    def test_const_shape(self):
        self.assertElemwise(numpy.arange(4*2*3*4).reshape(4, 2, 3, 4))

    def test_mixed_shape(self):
        self.assertElemwise(numpy.arange(4*i*j*3).reshape(4, i, j, 3) for i, j in ((1, 2), (2, 4)))

    def test_var_shape(self):
        self.assertElemwise(numpy.arange(i*j).reshape(i, j) for i, j in ((1, 2), (2, 4)))


class derivative(TestCase):

    def test_int(self):
        arg = evaluable.Argument('arg', (evaluable.constant(2),), int)
        self.assertEqual(evaluable.derivative(evaluable.insertaxis(arg, 0, evaluable.constant(1)), arg), evaluable.Zeros(tuple(map(evaluable.constant, (1, 2, 2))), int))

    def test_int_to_float(self):
        arg = evaluable.Argument('arg', (), float)
        func = evaluable.IntToFloat(evaluable.BoolToInt(evaluable.Greater(arg, evaluable.zeros(()))))
        self.assertTrue(evaluable.iszero(evaluable.derivative(func, arg)))

    def test_with_derivative(self):
        arg = evaluable.Argument('arg', (evaluable.constant(3),), float)
        deriv = numpy.arange(6, dtype=float).reshape(2, 3)
        func = evaluable.zeros((evaluable.constant(2),), float)
        func = evaluable.WithDerivative(func, arg, evaluable.asarray(deriv))
        self.assertAllAlmostEqual(evaluable.eval_once(evaluable.derivative(func, arg)), deriv)

    def test_default_derivative(self):
        # Tests whether `evaluable.Array._derivative` correctly raises an
        # exception when taking a derivative to one of the arguments present in
        # its `.arguments`.
        has_arg = evaluable.Argument('has_arg', (), float)
        has_not_arg = evaluable.Argument('has_not_arg', (), float)
        class DefaultDeriv(evaluable.Array):
            dtype = float
            shape = ()
            dependencies = evaluable.WithDerivative(evaluable.Zeros((), float), has_arg, evaluable.Zeros((), float)),
        func = DefaultDeriv()
        with self.assertRaises(NotImplementedError):
            evaluable.derivative(func, has_arg)
        self.assertTrue(evaluable.iszero(evaluable.derivative(func, has_not_arg)))


class asciitree(TestCase):

    @unittest.skipIf(sys.version_info < (3, 6), 'test requires dicts maintaining insertion order')
    def test_asciitree(self):
        n = evaluable.constant(2)
        f = evaluable.Sin(evaluable.InsertAxis(evaluable.Inflate(evaluable.constant(1.), evaluable.constant(1), n), n)**evaluable.Diagonalize(evaluable.Argument('arg', (n,))))
        self.assertEqual(f.asciitree(richoutput=True),
                         '%0 = Sin; f:2,2\n'
                         '└ %1 = Power; f:2,2\n'
                         '  ├ %2 = InsertAxis; f:~2,(2)\n'
                         '  │ ├ %3 = Inflate; f:~2\n'
                         '  │ │ ├ 1.0\n'
                         '  │ │ ├ 1\n'
                         '  │ │ └ 2\n'
                         '  │ └ 2\n'
                         '  └ %4 = Diagonalize; f:2/,2/\n'
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
                         '%B0 = LoopSum i\n'
                         '└ func = %B1 = LoopIndex i\n'
                         '  └ length = 2\n')

    @unittest.skipIf(sys.version_info < (3, 6), 'test requires dicts maintaining insertion order')
    def test_loop_concatenate(self):
        i = evaluable.loop_index('i', 2)
        f = evaluable.loop_concatenate(evaluable.InsertAxis(i, evaluable.constant(1)), i)
        self.assertEqual(f.asciitree(richoutput=True),
                         'SUBGRAPHS\n'
                         'A\n'
                         '└ B = Loop\n'
                         'NODES\n'
                         '%B0 = LoopConcatenate i\n'
                         '├ shape[0] = %A0 = Take; i:; [0,2]\n'
                         '│ ├ %A1 = _SizesToOffsets; i:3; [0,2]\n'
                         '│ │ └ %A2 = InsertAxis; i:(2); [1,1]\n'
                         '│ │   ├ 1\n'
                         '│ │   └ 2\n'
                         '│ └ 2\n'
                         '├ start = %B1 = Take; i:; [0,2]\n'
                         '│ ├ %A1\n'
                         '│ └ %B2 = LoopIndex i\n'
                         '│   └ length = 2\n'
                         '├ stop = %B3 = Take; i:; [0,2]\n'
                         '│ ├ %A1\n'
                         '│ └ %B4 = Add; i:; [1,2]\n'
                         '│   ├ %B2\n'
                         '│   └ 1\n'
                         '└ func = %B5 = InsertAxis; i:(1); [0,1]\n'
                         '  ├ %B2\n'
                         '  └ 1\n')


class simplify(TestCase):

    def test_minimum_maximum_bounds(self):

        class R(evaluable.Array):
            # An evaluable scalar argument with given bounds.
            lower: int
            upper: int

            dtype = int
            shape = ()

            @property
            def dependencies(self):
                return evaluable.Argument('R', shape=(), dtype=int),

            @property
            def _intbounds(self):
                return self.lower, self.upper

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

    def test_multiply_transpose(self):
        dummy = evaluable.Argument('dummy', shape=tuple(map(evaluable.constant, [2, 2, 2])), dtype=float)
        f = evaluable.multiply(dummy,
                               evaluable.Transpose(evaluable.multiply(dummy,
                                                                      evaluable.Transpose(dummy, (2, 0, 1))), (2, 0, 1)))
        # The test below is not only to verify that no simplifications are
        # performed, but also to make sure that simplified does not get stuck in a
        # circular dependence. This used to be the case prior to adding the
        # isinstance(other_trans, Transpose) restriction in Transpose._multiply.
        self.assertEqual(f.simplified, f)

    def test_add_sparse(self):
        a = evaluable.Inflate(
            func=evaluable.Argument('a', shape=tuple(map(evaluable.constant, [2, 3, 2])), dtype=float),
            dofmap=evaluable.InRange(evaluable.Argument('dofmap', shape=(evaluable.constant(2),), dtype=int), evaluable.constant(3)),
            length=evaluable.constant(3))
        b = evaluable.Diagonalize(
            func=evaluable.Argument('b', shape=tuple(map(evaluable.constant, [2, 3])), dtype=float))
        c = evaluable.Argument('c', shape=tuple(map(evaluable.constant, [2, 3, 3])), dtype=float)
        # Since a and b are both sparse, we expect (a+b)*c to be simplified to a*c+b*c.
        self.assertIsInstance(((a + b) * c).simplified, evaluable.Add)
        # If the sparsity of the terms is equal then sparsity propagates through the addition.
        self.assertIsInstance(((a + a) * c).simplified, evaluable.Inflate)
        self.assertIsInstance(((b + b) * c).simplified, evaluable.Diagonalize)
        # If either term in the addition is dense, the original structure remains.
        self.assertIsInstance(((a + c) * c).simplified, evaluable.Multiply)
        self.assertIsInstance(((c + b) * c).simplified, evaluable.Multiply)

    def test_insert_zero(self):
        a = evaluable.Argument('test', shape=(evaluable.constant(2,),))
        inserted = evaluable.InsertAxis(a, length=evaluable.constant(0))
        self.assertTrue(evaluable.iszero(inserted))

    def test_subtract_equals(self):
        a = evaluable.Argument('test', shape=(evaluable.constant(2,),))
        self.assertTrue(evaluable.iszero(a - a))

    def test_equal(self):
        r = evaluable.Range(evaluable.constant(3))
        self.assertEqual(
            evaluable.Equal(r, r).simplified,
            evaluable.ones(r.shape, bool))
        self.assertEqual(
            evaluable.Equal(evaluable.prependaxes(r, r.shape), evaluable.appendaxes(r, r.shape)).simplified,
            evaluable.Diagonalize(evaluable.ones(r.shape, bool)))

    def test_constant_range(self):
        self.assertEqual(
            evaluable.constant(numpy.arange(3)).simplified,
            evaluable.Range(evaluable.constant(3)))

    def test_swap_take_inflate(self):
        # test whether inflation to [0, 2] followed by take of [1] simplifies to zero
        a = evaluable.Argument('test', shape=(evaluable.constant(2),))
        inflated = evaluable.Inflate(a, dofmap=evaluable.constant([2,0]), length=evaluable.constant(3))
        taken = evaluable.Take(inflated, indices=evaluable.constant([1]))
        self.assertTrue(evaluable.iszero(taken))

    def test_double_logical_not(self):
        a = evaluable.Argument('test', shape=(), dtype=bool)
        self.assertEqual(evaluable.LogicalNot(evaluable.LogicalNot(a)).simplified, a)

    def test_logical_or_same_args(self):
        a = evaluable.Argument('test', shape=(), dtype=bool)
        self.assertEqual((a | a).simplified, a)

    def test_logical_and_same_args(self):
        a = evaluable.Argument('test', shape=(), dtype=bool)
        self.assertEqual((a & a).simplified, a)


class memory(TestCase):

    def assertCollected(self, ref):
        gc.collect()
        if ref() is not None:
            self.fail('object was not garbage collected')

    def test_general(self):
        # NOTE: The list of numbers must be unique in the entire test suite. If
        # not, a test leaking this specific array will cause this test to fail.
        A = evaluable.constant([1, 2, 3, 98, 513])
        A = weakref.ref(A)
        self.assertCollected(A)

    def test_simplified(self):
        # NOTE: The list of numbers must be unique in the entire test suite. If
        # not, a test leaking this specific array will cause this test to fail.
        A = evaluable.constant([1, 2, 3, 99, 514])
        A.simplified  # constant simplified to itself, which should be handled as a special case to avoid circular references
        A = weakref.ref(A)
        self.assertCollected(A)

    def test_replace(self):
        class MyException(Exception):
            pass

        class A(evaluable.Array):
            dtype = float
            shape = ()

            def _simplified(self):
                raise MyException
        t = evaluable.Tuple((A(),))
        with self.assertRaises(MyException):
            t.simplified
        with self.assertRaises(MyException):  # make sure no placeholders remain in the replacement cache
            t.simplified


class make_loop_ids_unique(TestCase):

    @staticmethod
    def loop_index(id):
        return evaluable._LoopIndex(evaluable._LoopId(id), evaluable.constant(3))

    @staticmethod
    def loop_sum(func, index):
        return evaluable.LoopSum(index.loop_id, index.length, func, ())

    def test_already_unique(self):
        i, j = map(self.loop_index, 'ij')
        A = self.loop_sum(self.loop_sum(i + j, j), i)
        self.assertEqual(evaluable._make_loop_ids_unique((A,)), (A,))

    def test_dependent(self):
        i0, i1, i2 = map(self.loop_index, range(3))
        A = self.loop_sum(self.loop_sum(i0, i0), i0)
        B = self.loop_sum(self.loop_sum(i2, i2), i1)
        self.assertEqual(evaluable._make_loop_ids_unique((A,)), (B,))

    def test_nested(self):
        i0, i1, i2, i3, i4 = map(self.loop_index, range(5))
        A = self.loop_sum(self.loop_sum(self.loop_sum(i0 + i2, i0) + i0, i2), i0)
        B = self.loop_sum(self.loop_sum(self.loop_sum(i4 + i3, i4) + i1, i3), i1)
        self.assertEqual(evaluable._make_loop_ids_unique((A,)), (B,))

    def test_cache_same_inner_loop(self):
        i0, i1, i2, i3, i4 = map(self.loop_index, range(5))
        A = self.loop_sum(self.loop_sum(i0, i0), i1) % self.loop_sum(self.loop_sum(i0, i0) + i1, i1)
        B = self.loop_sum(self.loop_sum(i3, i3), i4) % self.loop_sum(self.loop_sum(i3, i3) + i2, i2)
        self.assertEqual(evaluable._make_loop_ids_unique((A,)), (B,))

    def test_cache_different_inner_loop(self):
        i0, i1, i2, i3, i4, i5 = map(self.loop_index, range(6))
        # The inner loops look the same, but are different because index `i1` refers to different loops.
        A = self.loop_sum(self.loop_sum(i0 + i1, i0), i1) % self.loop_sum(self.loop_sum(i0 + i1, i0) + i1, i1)
        B = self.loop_sum(self.loop_sum(i5 + i4, i5), i4) % self.loop_sum(self.loop_sum(i3 + i2, i3) + i2, i2)
        self.assertEqual(evaluable._make_loop_ids_unique((A,)), (B,))


class Einsum(TestCase):

    def test_swapaxes(self):
        arg = numpy.arange(6).reshape(2, 3)
        ret = evaluable.einsum('ij->ji', evaluable.constant(arg))
        self.assertAllEqual(evaluable.eval_once(ret), arg.T)

    def test_rollaxes(self):
        arg = numpy.arange(6).reshape(1, 2, 3)
        ret = evaluable.einsum('Ai->iA', evaluable.constant(arg))
        self.assertAllEqual(evaluable.eval_once(ret), arg.transpose([2, 0, 1]))

    def test_swapgroups(self):
        arg = numpy.arange(24).reshape(1, 2, 3, 4)
        ret = evaluable.einsum('AB->BA', evaluable.constant(arg), B=2)
        self.assertAllEqual(evaluable.eval_once(ret), arg.transpose([2, 3, 0, 1]))

    def test_matvec(self):
        arg1 = numpy.arange(6).reshape(2, 3)
        arg2 = numpy.arange(6).reshape(3, 2)
        ret = evaluable.einsum('ij,jk->ik', evaluable.constant(arg1), evaluable.constant(arg2))
        self.assertAllEqual(evaluable.eval_once(ret), arg1 @ arg2)

    def test_multidot(self):
        arg1 = numpy.arange(6).reshape(2, 3)
        arg2 = numpy.arange(9).reshape(3, 3)
        arg3 = numpy.arange(6).reshape(3, 2)
        ret = evaluable.einsum('ij,jk,kl->il', evaluable.constant(arg1), evaluable.constant(arg2), evaluable.constant(arg3))
        self.assertAllEqual(evaluable.eval_once(ret), arg1 @ arg2 @ arg3)

    def test_wrong_args(self):
        arg = numpy.arange(6).reshape(2, 3)
        with self.assertRaisesRegex(ValueError, 'number of arguments does not match format string'):
            evaluable.einsum('ij,jk->ik', evaluable.constant(arg))

    def test_wrong_ellipse(self):
        arg = numpy.arange(6)
        with self.assertRaisesRegex(ValueError, 'argument dimensions are inconsistent with format string'):
            evaluable.einsum('iAj->jAi', evaluable.constant(arg))

    def test_wrong_dimension(self):
        arg = numpy.arange(9).reshape(3, 3)
        with self.assertRaisesRegex(ValueError, 'argument dimensions are inconsistent with format string'):
            evaluable.einsum('ijk->kji', evaluable.constant(arg))

    def test_wrong_multi_ellipse(self):
        arg = numpy.arange(6)
        with self.assertRaisesRegex(ValueError, 'cannot establish length of variable groups A, B'):
            evaluable.einsum('AB->BA', evaluable.constant(arg))

    def test_wrong_indices(self):
        arg = numpy.arange(9).reshape(3, 3)
        with self.assertRaisesRegex(ValueError, 'internal repetitions are not supported'):
            evaluable.einsum('kk->', evaluable.constant(arg))

    def test_wrong_shapes(self):
        arg1 = numpy.arange(6).reshape(2, 3)
        arg2 = numpy.arange(6).reshape(3, 2)
        with self.assertRaisesRegex(ValueError, 'shapes do not match for axis i0'):
            ret = evaluable.einsum('ij,ik->jk', evaluable.constant(arg1), evaluable.constant(arg2))

    def test_wrong_group_dimension(self):
        arg = numpy.arange(6)
        with self.assertRaisesRegex(ValueError, 'axis group dimensions cannot be negative'):
            evaluable.einsum('Aij->ijA', evaluable.constant(arg), A=-1)


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


class unalign(TestCase):

    def test_single_noop(self):
        ox = evaluable.asarray(numpy.arange(6).reshape(2,3))
        ux, where = evaluable.unalign(ox)
        self.assertEqual(where, (0, 1))
        self.assertEqual(evaluable.eval_once(evaluable.align(ux, where, ox.shape)).tolist(), evaluable.eval_once(ox).tolist())

    def test_single_trans(self):
        ox = evaluable.Transpose(evaluable.asarray(numpy.arange(6).reshape(2,3)), (1, 0))
        ux, where = evaluable.unalign(ox)
        self.assertEqual(where, (1, 0)) # transposed, because this is a single argument
        self.assertEqual(evaluable.eval_once(evaluable.align(ux, where, ox.shape)).tolist(), evaluable.eval_once(ox).tolist())

    def test_single_ins(self):
        ox = evaluable.InsertAxis(evaluable.asarray(numpy.arange(2)), evaluable.constant(3))
        ux, where = evaluable.unalign(ox)
        self.assertEqual(where, (0,))
        self.assertEqual(evaluable.eval_once(evaluable.align(ux, where, ox.shape)).tolist(), evaluable.eval_once(ox).tolist())

    def test_single_ins_trans(self):
        ox = evaluable.Transpose(evaluable.InsertAxis(evaluable.asarray(numpy.arange(3)), evaluable.constant(2)), (1, 0))
        ux, where = evaluable.unalign(ox)
        self.assertEqual(where, (1,))
        self.assertEqual(evaluable.eval_once(evaluable.align(ux, where, ox.shape)).tolist(), evaluable.eval_once(ox).tolist())

    def test_single_naxes_reins(self):
        # tests reinsertion of an uninserted axis >= naxes
        ox = evaluable.InsertAxis(evaluable.Transpose(evaluable.InsertAxis(evaluable.asarray(numpy.arange(3)), evaluable.constant(2)), (1, 0)), evaluable.constant(4))
        ux, where = evaluable.unalign(ox, naxes=2)
        self.assertEqual(where, (1,))
        self.assertEqual(evaluable.eval_once(evaluable.align(ux, where+(2,), ox.shape)).tolist(), evaluable.eval_once(ox).tolist())

    def test_single_naxes_trans(self):
        # tests the transpose of an axis >= naxes
        ox = evaluable.Transpose(evaluable.InsertAxis(evaluable.asarray(numpy.arange(12).reshape(4, 3)), evaluable.constant(2)), (2, 1, 0))
        ux, where = evaluable.unalign(ox, naxes=1)
        self.assertEqual(where, ())
        self.assertEqual(evaluable.eval_once(evaluable.align(ux, where+(1, 2), ox.shape)).tolist(), evaluable.eval_once(ox).tolist())

    def test_single_naxes_reins_trans(self):
        # tests the transpose of an uninserted axis >= naxes
        ox = evaluable.Transpose(evaluable.InsertAxis(evaluable.InsertAxis(evaluable.asarray(numpy.arange(4)), evaluable.constant(2)), evaluable.constant(3)), (1, 2, 0))
        ux, where = evaluable.unalign(ox, naxes=1)
        self.assertEqual(where, ())
        self.assertEqual(evaluable.eval_once(evaluable.align(ux, where+(1, 2), ox.shape)).tolist(), evaluable.eval_once(ox).tolist())

    def test_double_noins(self):
        ox = evaluable.asarray(numpy.arange(6).reshape(2,3))
        oy = evaluable.asarray(numpy.arange(6, 12).reshape(2,3))
        ux, uy, where = evaluable.unalign(ox, oy)
        self.assertEqual(where, (0, 1))
        self.assertEqual(evaluable.eval_once(evaluable.align(ux, where, ox.shape)).tolist(), evaluable.eval_once(ox).tolist())
        self.assertEqual(evaluable.eval_once(evaluable.align(uy, where, oy.shape)).tolist(), evaluable.eval_once(oy).tolist())

    def test_double_disjointins(self):
        ox = evaluable.Transpose(evaluable.InsertAxis(evaluable.asarray(numpy.arange(3)), evaluable.constant(2)), (1, 0))
        oy = evaluable.InsertAxis(evaluable.asarray(numpy.arange(2, 4)), evaluable.constant(3))
        ux, uy, where = evaluable.unalign(ox, oy)
        self.assertEqual(where, (0, 1)) # not transposed, despite the transpose of the first argument
        self.assertEqual(evaluable.eval_once(evaluable.align(ux, where, ox.shape)).tolist(), evaluable.eval_once(ox).tolist())
        self.assertEqual(evaluable.eval_once(evaluable.align(uy, where, oy.shape)).tolist(), evaluable.eval_once(oy).tolist())

    def test_double_commonins(self):
        ox = evaluable.Transpose(evaluable.InsertAxis(evaluable.asarray(numpy.arange(3)), evaluable.constant(2)), (1, 0))
        oy = evaluable.zeros((evaluable.constant(2), evaluable.constant(3)), dtype=int)
        ux, uy, where = evaluable.unalign(ox, oy)
        self.assertEqual(where, (1,))
        self.assertEqual(evaluable.eval_once(evaluable.align(ux, where, ox.shape)).tolist(), evaluable.eval_once(ox).tolist())
        self.assertEqual(evaluable.eval_once(evaluable.align(uy, where, oy.shape)).tolist(), evaluable.eval_once(oy).tolist())

    def test_too_few_axes(self):
        with self.assertRaises(ValueError):
            evaluable.unalign(evaluable.zeros((evaluable.constant(2), evaluable.constant(3))), naxes=3)

    def test_unequal_naxes(self):
        with self.assertRaises(ValueError):
            evaluable.unalign(evaluable.zeros(tuple(map(evaluable.constant, (2, 3)))), evaluable.zeros(tuple(map(evaluable.constant, (2, 3, 4)))))


class Poly(TestCase):

    def test_mul_variable_ncoeffs(self):
        vars = poly.MulVar.Left, poly.MulVar.Right
        const_coeffs_left = numpy.arange(6, dtype=float)
        const_coeffs_right = numpy.array([1, 2], dtype=float)
        eval_ncoeffs_left = evaluable.InRange(evaluable.Argument('ncoeffs_left', (), int), evaluable.constant(10))
        eval_coeffs_left = evaluable.IntToFloat(evaluable.Range(eval_ncoeffs_left))
        eval_coeffs_right = evaluable.asarray(const_coeffs_right)
        numpy.testing.assert_allclose(
            evaluable.eval_once(evaluable.PolyMul(eval_coeffs_left, eval_coeffs_right, vars), arguments=dict(ncoeffs_left=numpy.array(6))),
            poly.mul(const_coeffs_left, const_coeffs_right, vars),
        )

    def test_grad_variable_ncoeffs(self):
        const_coeffs = numpy.arange(6, dtype=float)
        eval_ncoeffs = evaluable.InRange(evaluable.Argument('ncoeffs', (), int), evaluable.constant(10))
        eval_coeffs = evaluable.IntToFloat(evaluable.Range(eval_ncoeffs))
        numpy.testing.assert_allclose(
            evaluable.eval_once(evaluable.PolyGrad(eval_coeffs, 2), arguments=dict(ncoeffs=numpy.array(6))),
            poly.grad(const_coeffs, 2),
        )


class factor(TestCase):

    def setUp(self):
        self.index = evaluable.loop_index('i', 4)
        values = evaluable.constant([1., 2., -3.])
        dofs = self.index + evaluable.Range(values.shape[0])
        length = evaluable.constant(6)
        basis = evaluable._inflate(values, dofs, length, axis=0)
        self.varg = evaluable.Argument('v', basis.shape, float)
        self.v = (basis * self.varg).sum(0)
        self.targ = evaluable.Argument('t', (), float)
        self.t = 10. * self.targ
        self.barg = evaluable.Argument('b', (*basis.shape, evaluable.constant(2)), float)
        self.b = (evaluable.InsertAxis(basis, self.barg.shape[1]) * self.barg).sum(0)

    def integral(self, f):
        return evaluable.loop_sum(f, self.index)

    def assertFactoredEqual(self, integrand, replacements=None, *, v=0, t=0, b=0):
        orig = self.integral(integrand)
        factored = evaluable.factor(orig)
        if replacements:
            orig = evaluable.replace_arguments(orig, replacements)
            factored = evaluable.replace_arguments(factored, replacements)

        for func in orig, factored:
            self.assertEqual(func.argument_degree(self.varg), v)
            self.assertEqual(func.argument_degree(self.targ), t)
            self.assertEqual(func.argument_degree(self.barg), b)

        testing_grid = [{}]
        if t: testing_grid = [dict(d, t=t) for d in testing_grid for t in [0., 1., -5.]]
        if v: testing_grid = [dict(d, v=v) for d in testing_grid for v in [numpy.zeros(6), numpy.ones(6), numpy.arange(6, dtype=float)]]
        if b: testing_grid = [dict(d, b=b) for d in testing_grid for b in [numpy.zeros((6,2)), numpy.ones((6,2)), numpy.arange(12, dtype=float).reshape(6,2)]]

        for deriv_args in [[self.targ] * i + [self.varg] * j + [self.barg] * k for i in range(t+1) for j in range(v+1) for k in range(b+1)]:
            with self.subTest('f/' + ''.join(arg.name for arg in deriv_args)):
                F = evaluable.compile(functools.reduce(evaluable.derivative, deriv_args, orig))
                G = evaluable.compile(functools.reduce(evaluable.derivative, deriv_args, factored))
                for eval_args in testing_grid:
                    self.assertAllAlmostEqual(F(eval_args), G(eval_args))

    def test_linear(self):
        self.assertFactoredEqual(1. + self.v, v=1)
        self.assertFactoredEqual(2. * self.v - 5. * self.t, v=1, t=1)
        self.assertFactoredEqual(2. * self.v - self.t, v=1, t=1)
        self.assertFactoredEqual(2. * self.v * self.t, v=1, t=1)
        self.assertFactoredEqual(3. * self.b * self.t, b=1, t=1)

    def test_quadratic(self):
        self.assertFactoredEqual(1. + self.v - self.v**2., v=2)
        self.assertFactoredEqual(3. * self.v**2., v=2)
        self.assertFactoredEqual(5. * self.t**2., t=2)
        self.assertFactoredEqual(self.v * evaluable.Sum(self.b * self.b) - 2. * self.v**2., v=2, b=2)

    def test_cubic(self):
        self.assertFactoredEqual(1. + self.v - self.v**2. + self.v**3., v=3)
        self.assertFactoredEqual(-self.v**3., v=3)
        self.assertFactoredEqual(3. * self.t**3., t=3)
        self.assertFactoredEqual(self.t**2. * (self.v**2. - 2. * self.t), t=3, v=2)
        self.assertFactoredEqual(self.t**2. * (self.b**2. - 2. * self.t * self.b), t=3, b=2)

    def test_not_polynomial(self):
        with self.assertRaisesRegex(evaluable.NotPolynomal, "nutils.evaluable.Sign<f:> is not polynomial in argument 'v'"):
            evaluable.factor(self.integral(evaluable.Sign(self.v)))

    def test_constant(self):
        self.assertFactoredEqual(evaluable.constant(1.))

    def test_replace(self):
        self.assertFactoredEqual(self.v, dict(v=evaluable.constant(numpy.arange(.5, 6)) * self.t), t=1)
        self.assertFactoredEqual(self.v * self.v, dict(v=evaluable.constant(numpy.arange(.5, 6)) * self.t), t=2)
        self.assertFactoredEqual(self.b, dict(b=evaluable.constant(numpy.arange(.5, 12).reshape(6,2)) * self.t), t=1)
        self.assertFactoredEqual(self.v * self.v * evaluable.Sum(self.b * self.b),
            dict(b=evaluable.constant(numpy.arange(.5, 12).reshape(6,2)) * self.t, v=evaluable.constant(numpy.arange(.5, 6)) * self.t), t=4)

    def test_derivative_sparsity(self):
        # check that the function tree produced by factor reveals its sparsity after derivative.

        v2 = evaluable.derivative(evaluable.derivative(evaluable.factor(self.integral(self.v * self.v)), self.varg), self.varg)
        values, indices, shape = evaluable.eval_once(v2.simplified.assparse)
        self.assertAllEqual(values,
            [2, 4, -6, 4, 10, -8, -6, -6, -8, 28, -8, -6, -6, -8, 28, -8, -6, -6, -8, 26, -12, -6, -12, 18])
        self.assertAllEqual(indices,
            [[0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5],
             [0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 4, 1, 2, 3, 4, 5, 2, 3, 4, 5, 3, 4, 5]])
        self.assertEqual(shape, (6, 6))

        b2 = evaluable.derivative(evaluable.derivative(evaluable.factor(self.integral(evaluable.Sum(self.b * self.b))), self.barg), self.barg)
        values, indices, shape = evaluable.eval_once(b2.simplified.assparse)
        self.assertAllEqual(values,
            [2, 4, -6, 2, 4, -6, 4, 10, -8, -6, 4, 10, -8, -6, -6, -8, 28, -8, -6, -6, -8, 28, -8, -6, -6, -8, 28, -8, -6, -6, -8, 28, -8, -6, -6, -8, 26, -12, -6, -8, 26, -12, -6, -12, 18, -6, -12, 18])
        self.assertAllEqual(indices,
           [[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5],
            [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1],
            [0, 1, 2, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 2, 3, 4, 5, 2, 3, 4, 5, 3, 4, 5, 3, 4, 5],
            [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1]])
        self.assertEqual(shape, (6, 2, 6, 2))
