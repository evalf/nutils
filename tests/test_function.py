from nutils import evaluable, function, mesh, numeric, types, points, transformseq, transform, element, warnings
from nutils.testing import TestCase, parametrize
import nutils_poly as poly
import numpy
import itertools
import warnings as _builtin_warnings
import functools
import fractions


class Array(TestCase):

    def test_cast_ndim_mismatch(self):
        with self.assertRaises(ValueError):
            function.Array.cast([1, 2], ndim=2)

    def test_cast_dtype_mismatch(self):
        with self.assertRaises(ValueError):
            function.Array.cast([1.2, 2.3], dtype=int)

    def test_cast_invalid_argument(self):
        with self.assertRaisesRegex(ValueError, "cannot convert '132' to Array: unsupported data type"):
            function.Array.cast('132')

    def test_cast_different_shapes(self):
        with self.assertRaisesRegex(ValueError, 'cannot convert \[\[1, 2, 3\], \[4, 5\]\] to Array: all input arrays must have the same shape'):
            function.Array.cast([[1,2,3],[4,5]])

    def test_ndim(self):
        self.assertEqual(function.Argument('a', (2, 3)).ndim, 2)

    def test_size_known(self):
        self.assertEqual(function.Argument('a', (2, 3)).size, 6)

    def test_size_0d(self):
        self.assertEqual(function.Argument('a', ()).size, 1)

    def test_len_0d(self):
        with self.assertRaisesRegex(Exception, '^len\\(\\) of unsized object$'):
            len(function.Array.cast(0))

    def test_len_known(self):
        self.assertEqual(len(function.Array.cast([1, 2])), 2)

    def test_iter_0d(self):
        with self.assertRaisesRegex(Exception, '^iteration over a 0-D array$'):
            iter(function.Array.cast(0))

    def test_iter_known(self):
        a, b = function.Array.cast([1, 2])
        self.assertEqual(a.as_evaluable_array.eval(), 1)
        self.assertEqual(b.as_evaluable_array.eval(), 2)

    def test_binop_notimplemented(self):
        with self.assertRaisesRegex(TypeError, '^operand type\(s\) all returned NotImplemented from __array_ufunc__'):
            function.Argument('a', ()) + '1'

    def test_rbinop_notimplemented(self):
        with self.assertRaisesRegex(TypeError, '^operand type\(s\) all returned NotImplemented from __array_ufunc__'):
            '1' + function.Argument('a', ())

    def test_deprecated_simplified(self):
        with self.assertWarns(warnings.NutilsDeprecationWarning):
            function.Array.cast([1, 2]).simplified

    def test_different_argument_shapes(self):
        with self.assertRaisesRegex(ValueError, "Argument 'a' has two different shapes"):
            function.Argument('a', (2,)).sum() + function.Argument('a', (3, 4)).sum(-1)

    def test_different_argument_dtypes(self):
        with self.assertRaisesRegex(ValueError, "Argument 'a' has two different dtypes"):
            function.Argument('a', (), dtype=float) + function.Argument('a', (), dtype=int)

    def test_index(self):
        self.assertEqual(function.Array.cast(2).__index__(), 2)
        with self.assertRaisesRegex(ValueError, "cannot convert non-constant array to index: arguments=foo"):
            function.Argument('foo', shape=(), dtype=int).__index__()
        with self.assertRaisesRegex(ValueError, "cannot convert non-scalar array to index: shape=\(2,\)"):
            function.Array.cast([2, 3]).__index__()
        with self.assertRaisesRegex(ValueError, "cannot convert non-integer array to index: dtype=float"):
            function.Array.cast(2.5).__index__()

    def test_truthiness(self):
        topo, geom = mesh.unitsquare(4, 'square')
        with self.assertRaisesRegex(ValueError, 'The truth value of a nutils Array is ambiguous'):
            min(geom, 1)

    def test_fraction(self):
        v = function.Array.cast(fractions.Fraction(1, 2))
        self.assertEqual(v.eval(), .5)


class integral_compatibility(TestCase):

    def test_eval(self):
        v = numpy.array([1, 2])
        a = function.Argument('a', (2,), dtype=float)
        self.assertAllAlmostEqual(a.eval(a=v), v)

    def test_derivative(self):
        v = numpy.array([1, 2])
        a = function.Argument('a', (2,), dtype=float)
        f = 2*a.sum()
        for name, obj in ('str', 'a'), ('argument', a):
            with self.subTest(name):
                self.assertAllAlmostEqual(f.derivative(obj).eval(a=v), numpy.array([2, 2]))

    def test_derivative_str_unknown_argument(self):
        f = function.zeros((2,), dtype=float)
        with self.assertRaises(ValueError):
            f.derivative('a')

    def test_derivative_invalid(self):
        f = function.zeros((2,), dtype=float)
        with self.assertRaises(ValueError):
            f.derivative(1.)

    def test_replace(self):
        v = numpy.array([1, 2])
        a = function.Argument('a', (2,), dtype=int)
        b = function.Argument('b', (2,), dtype=int)
        f = a.replace(dict(a=b))
        self.assertAllAlmostEqual(f.eval(b=v), v)

    def test_contains(self):
        f = 2*function.Argument('a', (2,), dtype=int)
        self.assertTrue(f.contains('a'))
        self.assertFalse(f.contains('b'))

    def test_argshapes(self):
        a = function.Argument('a', (2, 3), dtype=int)
        b = function.Argument('b', (3,), dtype=int)
        f = (a * b[None]).sum(-1)
        self.assertEqual(dict(f.argshapes), dict(a=(2, 3), b=(3,)))

    def test_argshapes_shape_mismatch(self):
        with self.assertRaises(Exception):
            f = function.Argument('a', (2,), dtype=int)[None] + function.Argument('a', (3,), dtype=int)[:, None]


@parametrize
class check(TestCase):

    def setUp(self):
        super().setUp()
        numpy.random.seed(0)

    def assertArrayAlmostEqual(self, actual, desired, decimal):
        if actual.shape != desired.shape:
            self.fail('shapes of actual {} and desired {} are incompatible.'.format(actual.shape, desired.shape))
        error = actual - desired if not actual.dtype.kind == desired.dtype.kind == 'b' else actual ^ desired
        approx = error.dtype.kind in 'fc'
        mask = numpy.greater_equal(abs(error), 1.5 * 10**-decimal) if approx else error
        indices = tuple(zip(*mask.nonzero())) if actual.ndim else ((),) if mask.any() else ()
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

    def test_lower_eval(self):
        actual = self.op(*self.args).as_evaluable_array.eval()
        desired = self.n_op(*self.args)
        self.assertArrayAlmostEqual(actual, desired, decimal=15)


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
        if real and imag:
            assert negative
            a = a * numpy.exp(1j * numpy.arange(size)**2)
        elif imag:
            a = a * 1j
    return a.reshape(shape)


INT = functools.partial(generate, real=False, imag=False, zero=True, negative=False)
NNI = functools.partial(generate, real=False, imag=False, zero=True, negative=False)
ANY = functools.partial(generate, real=True, imag=False, zero=True, negative=True)
NZ = functools.partial(generate, real=True, imag=False, zero=False, negative=True)
POS = functools.partial(generate, real=True, imag=False, zero=False, negative=False)
NN = functools.partial(generate, real=True, imag=False, zero=True, negative=False)
IM = functools.partial(generate, real=False, imag=True, zero=True, negative=True)
ANC = functools.partial(generate, real=True, imag=True, zero=True, negative=True)
NZC = functools.partial(generate, real=True, imag=True, zero=False, negative=True)


def _check(name, op, n_op, *args):
    check(name, op=op, n_op=n_op, args=args)


_check('asarray', function.asarray, lambda a: a, ANY(2, 4, 2))
_check('zeros', lambda: function.zeros([1, 4, 3, 4]), lambda: numpy.zeros([1, 4, 3, 4]))
_check('ones', lambda: function.ones([1, 4, 3, 4]), lambda: numpy.ones([1, 4, 3, 4]))
_check('eye', lambda: function.eye(3), lambda: numpy.eye(3))

_check('add', lambda a, b: numpy.add(a, function.Array.cast(b)), numpy.add, ANY(4), ANY(4))
_check('add-complex', lambda a, b: numpy.add(a, function.Array.cast(b)), numpy.add, ANY(4), ANC(4))
_check('Array_add', lambda a, b: function.Array.cast(a) + b, numpy.add, ANY(4, 4), ANY(4))
_check('Array_radd', lambda a, b: a + function.Array.cast(b), numpy.add, ANY(4, 4), ANY(4))
_check('subtract', lambda a, b: numpy.subtract(a, function.Array.cast(b)), numpy.subtract, ANY(4, 4), ANY(4))
_check('subtract-complex', lambda a, b: numpy.subtract(a, function.Array.cast(b)), numpy.subtract, ANY(4, 4), ANC(4))
_check('Array_sub', lambda a, b: function.Array.cast(a) - b, numpy.subtract, ANY(4, 4), ANY(4))
_check('Array_rsub', lambda a, b: a - function.Array.cast(b), numpy.subtract, ANY(4, 4), ANY(4))
_check('negative', lambda a: numpy.negative(function.Array.cast(a)), numpy.negative, ANY(4))
_check('negative-complex', lambda a: numpy.negative(function.Array.cast(a)), numpy.negative, ANC(4))
_check('positive', lambda a: numpy.positive(function.Array.cast(a)), numpy.positive, ANY(4))
_check('positive-complex', lambda a: numpy.positive(function.Array.cast(a)), numpy.positive, ANC(4))
_check('Array_neg', lambda a: -function.Array.cast(a), numpy.negative, ANY(4))
_check('Array_pos', lambda a: +function.Array.cast(a), lambda a: a, ANY(4))
_check('multiply', lambda a, b: numpy.multiply(a, function.Array.cast(b)), numpy.multiply, ANY(4, 4), ANY(4))
_check('multiply-complex', lambda a, b: numpy.multiply(a, function.Array.cast(b)), numpy.multiply, ANC(4, 4), 1-1j+ANC(4))
_check('Array_mul', lambda a, b: function.Array.cast(a) * b, numpy.multiply, ANY(4, 4), ANY(4))
_check('Array_rmul', lambda a, b: a * function.Array.cast(b), numpy.multiply, ANY(4, 4), ANY(4))
_check('divide', lambda a, b: numpy.divide(a, function.Array.cast(b)), numpy.divide, ANY(4), POS(4))
_check('divide-complex', lambda a, b: numpy.divide(a, function.Array.cast(b)), numpy.divide, ANC(4, 4), NZC(4, 4).T)
_check('Array_truediv', lambda a, b: function.Array.cast(a) / b, numpy.divide, ANY(4), POS(4))
_check('Array_rtruediv', lambda a, b: a / function.Array.cast(b), numpy.divide, ANY(4), POS(4))
_check('floor_divide', lambda a, b: numpy.floor_divide(a, function.Array.cast(b)), numpy.floor_divide, INT(4, 4), 1+NNI(4))
_check('Array_floordiv', lambda a, b: function.Array.cast(a) // b, numpy.floor_divide, INT(4, 4), 1+NNI(4))
_check('Array_rfloordiv', lambda a, b: a // function.Array.cast(b), numpy.floor_divide, INT(4, 4), 1+NNI(4))
_check('reciprocal', lambda a: numpy.reciprocal(function.Array.cast(a)), numpy.reciprocal, NZ(4))
_check('reciprocal-complex', lambda a: numpy.reciprocal(function.Array.cast(a)), numpy.reciprocal, NZC(4))
_check('power', lambda a, b: numpy.power(a, function.Array.cast(b)), numpy.power, POS(4, 4), ANY(4, 4))
_check('power-complex', lambda a, b: numpy.power(a, function.Array.cast(b)), numpy.power, NZC(4, 4), ANY(4, 4))
_check('Array_pow', lambda a, b: function.Array.cast(a) ** b, numpy.power, POS(4, 4), ANY(4, 4))
_check('Array_rpow', lambda a, b: a ** function.Array.cast(b), numpy.power, POS(4, 4), ANY(4, 4))
_check('sqrt', lambda a: numpy.sqrt(function.Array.cast(a)), numpy.sqrt, NN(4))
_check('sqrt-complex', lambda a: numpy.sqrt(function.Array.cast(a)), numpy.sqrt, ANC(4))
_check('abs', lambda a: numpy.abs(function.Array.cast(a)), numpy.abs, ANY(4))
_check('abs-complex', lambda a: numpy.abs(function.Array.cast(a)), numpy.abs, ANC(4))
_check('Array_abs', lambda a: abs(function.Array.cast(a)), numpy.abs, ANY(4))
_check('sign', lambda a: numpy.sign(function.Array.cast(a)), numpy.sign, ANY(4))
_check('mod', lambda a, b: numpy.mod(a, function.Array.cast(b)), numpy.mod, INT(4, 4), 1+NNI(4))
_check('Array_mod', lambda a, b: function.Array.cast(a) % b, numpy.mod, INT(4, 4), 1+NNI(4))
_check('Array_rmod', lambda a, b: a % function.Array.cast(b), numpy.mod, INT(4, 4), 1+NNI(4))
_check('divmod_div', lambda a, b: numpy.divmod(a, function.Array.cast(b))[0], lambda a, b: numpy.divmod(a, b)[0], INT(4, 4), 1+NNI(4))
_check('divmod_mod', lambda a, b: numpy.divmod(a, function.Array.cast(b))[1], lambda a, b: numpy.divmod(a, b)[1], INT(4, 4), 1+NNI(4))
_check('Array_divmod_div', lambda a, b: divmod(function.Array.cast(a), b)[0], lambda a, b: numpy.divmod(a, b)[0], INT(4, 4), 1+NNI(4))
_check('Array_divmod_mod', lambda a, b: divmod(function.Array.cast(a), b)[1], lambda a, b: numpy.divmod(a, b)[1], INT(4, 4), 1+NNI(4))
_check('Array_rdivmod_div', lambda a, b: divmod(a, function.Array.cast(b))[0], lambda a, b: numpy.divmod(a, b)[0], INT(4, 4), 1+NNI(4))
_check('Array_rdivmod_mod', lambda a, b: divmod(a, function.Array.cast(b))[1], lambda a, b: numpy.divmod(a, b)[1], INT(4, 4), 1+NNI(4))
_check('matmul', lambda a, b: numpy.matmul(a, function.Array.cast(b)), numpy.matmul, ANY(4), ANY(4))
_check('matmul-complex', lambda a, b: numpy.matmul(a, function.Array.cast(b)), numpy.matmul, ANY(4), ANC(4))
_check('Array_matmul_vecvec', lambda a, b: function.Array.cast(a) @ b, numpy.matmul, ANY(4), ANY(4))
_check('Array_matmul_vecmat', lambda a, b: function.Array.cast(a) @ b, numpy.matmul, ANY(4), ANY(4, 3))
_check('Array_matmul_matvec', lambda a, b: function.Array.cast(a) @ b, numpy.matmul, ANY(3, 4), ANY(4))
_check('Array_matmul_matmat', lambda a, b: function.Array.cast(a) @ b, numpy.matmul, ANY(3, 4), ANY(4, 5))
_check('real', lambda a: numpy.real(function.Array.cast(a)), numpy.real, ANY(4))
_check('real-complex', lambda a: numpy.real(function.Array.cast(a)), numpy.real, ANC(4))
_check('Aray_real-complex', lambda a: function.Array.cast(a).real, numpy.real, ANC(4))
_check('imag', lambda a: numpy.imag(function.Array.cast(a)), numpy.imag, ANY(4))
_check('imag-complex', lambda a: numpy.imag(function.Array.cast(a)), numpy.imag, ANC(4))
_check('Aray_imag-complex', lambda a: function.Array.cast(a).imag, numpy.imag, ANC(4))
_check('conjugate', lambda a: numpy.conjugate(function.Array.cast(a)), numpy.conjugate, ANY(4))
_check('conjugate-complex', lambda a: numpy.conjugate(function.Array.cast(a)), numpy.conjugate, ANC(4))
_check('Aray_conjugate-complex', lambda a: function.Array.cast(a).conjugate(), numpy.conjugate, ANC(4))

_check('cos', lambda a: numpy.cos(function.Array.cast(a)), numpy.cos, ANY(4))
_check('cos-complex', lambda a: numpy.cos(function.Array.cast(a)), numpy.cos, ANC(4))
_check('sin', lambda a: numpy.sin(function.Array.cast(a)), numpy.sin, ANY(4))
_check('sin-complex', lambda a: numpy.sin(function.Array.cast(a)), numpy.sin, ANC(4))
_check('tan', lambda a: numpy.tan(function.Array.cast(a)), numpy.tan, ANY(4))
_check('tan-complex', lambda a: numpy.tan(function.Array.cast(a)), numpy.tan, ANC(4))
_check('arccos', lambda a: numpy.arccos(function.Array.cast(a)), numpy.arccos, ANY(4))
_check('arccos-complex', lambda a: numpy.arccos(function.Array.cast(a)), numpy.arccos, ANC(4))
_check('arcsin', lambda a: numpy.arcsin(function.Array.cast(a)), numpy.arcsin, ANY(4))
_check('arcsin-complex', lambda a: numpy.arcsin(function.Array.cast(a)), numpy.arcsin, ANC(4))
_check('arctan', lambda a: numpy.arctan(function.Array.cast(a)), numpy.arctan, ANY(4))
_check('arctan-complex', lambda a: numpy.arctan(function.Array.cast(a)), numpy.arctan, ANC(4))
_check('arctan2', lambda a, b: numpy.arctan2(a, function.Array.cast(b)), numpy.arctan2, ANY(4, 1), ANY(1, 4))
_check('sinc', lambda a: numpy.sin(function.Array.cast(a)), numpy.sin, ANY(4))
_check('sinc-complex', lambda a: numpy.sin(function.Array.cast(a)), numpy.sin, ANC(4))
_check('cosh', lambda a: numpy.cosh(function.Array.cast(a)), numpy.cosh, ANY(4))
_check('cosh-complex', lambda a: numpy.cosh(function.Array.cast(a)), numpy.cosh, ANC(4))
_check('sinh', lambda a: numpy.sinh(function.Array.cast(a)), numpy.sinh, ANY(4))
_check('sinh-complex', lambda a: numpy.sinh(function.Array.cast(a)), numpy.sinh, ANC(4))
_check('tanh', lambda a: numpy.tanh(function.Array.cast(a)), numpy.tanh, ANY(4))
_check('tanh-complex', lambda a: numpy.tanh(function.Array.cast(a)), numpy.tanh, ANC(4))
_check('arctanh', lambda a: numpy.arctanh(function.Array.cast(a)), numpy.arctanh, ANY(4))
_check('arctanh-complex', lambda a: numpy.arctanh(function.Array.cast(a)), numpy.arctanh, ANC(4))
_check('exp', lambda a: numpy.exp(function.Array.cast(a)), numpy.exp, ANY(4))
_check('exp-complex', lambda a: numpy.exp(function.Array.cast(a)), numpy.exp, ANC(4))
_check('log', lambda a: numpy.log(function.Array.cast(a)), numpy.log, POS(4))
_check('log-complex', lambda a: numpy.log(function.Array.cast(a)), numpy.log, NZC(4))
_check('log2', lambda a: numpy.log2(function.Array.cast(a)), numpy.log2, POS(4))
_check('log2-complex', lambda a: numpy.log2(function.Array.cast(a)), numpy.log2, NZC(4))
_check('log10', lambda a: numpy.log10(function.Array.cast(a)), numpy.log10, POS(4))
_check('log10-complex', lambda a: numpy.log10(function.Array.cast(a)), numpy.log10, NZC(4))
_check('trignormal', function.trignormal, lambda x: numpy.stack([numpy.cos(x), numpy.sin(x)], axis=-1), ANY(4))
_check('trigtangent', function.trigtangent, lambda x: numpy.stack([-numpy.sin(x), numpy.cos(x)], axis=-1), ANY(4))

_check('greater', lambda a, b: numpy.greater(a, function.Array.cast(b)), numpy.greater, ANY(4, 1), ANY(1, 4))
_check('equal', lambda a, b: numpy.equal(a, function.Array.cast(b)), numpy.equal, ANY(4, 1), ANY(1, 4))
_check('equal-complex', lambda a, b: numpy.equal(a, function.Array.cast(b)), numpy.equal, ANC(4, 1), ANC(1, 4))
_check('less', lambda a, b: numpy.less(a, function.Array.cast(b)), numpy.less, ANY(4, 1), ANY(1, 4))
_check('min', lambda a, b: numpy.minimum(a, function.Array.cast(b)), numpy.minimum, ANY(4, 1), ANY(1, 4))
_check('max', lambda a, b: numpy.maximum(a, function.Array.cast(b)), numpy.maximum, ANY(4, 1), ANY(1, 4))
_check('heaviside', function.heaviside, lambda u: numpy.heaviside(u, .5), ANY(4, 4))

## TODO: opposite
## TODO: mean
## TODO: jump
#
_check('sum', lambda a: numpy.sum(function.Array.cast(a), 2), lambda a: a.sum(2), ANY(4, 3, 4))
_check('sum-bool', lambda a: numpy.sum(function.Array.cast(a > 0), 2), lambda a: (a > 0).sum(2), ANY(4, 3, 4))
_check('sum-complex', lambda a: numpy.sum(function.Array.cast(a), 2), lambda a: a.sum(2), ANC(4, 3, 4))
_check('Array_sum', lambda a: function.Array.cast(a).sum(2), lambda a: a.sum(2), ANY(4, 3, 4))
_check('product', lambda a: numpy.product(function.Array.cast(a), 2), lambda a: numpy.product(a, 2), ANY(4, 3, 4))
_check('product-bool', lambda a: numpy.product(function.Array.cast(a > 0), 2), lambda a: numpy.product((a > 0), 2), ANY(4, 3, 4))
_check('product-complex', lambda a: numpy.product(function.Array.cast(a), 2), lambda a: numpy.product(a, 2), ANC(4, 3, 4))
_check('Array_prod', lambda a: function.Array.cast(a).prod(2), lambda a: numpy.product(a, 2), ANY(4, 3, 4))

_check('dot', lambda a, b: numpy.dot(a, function.Array.cast(b)), numpy.dot, ANY(1, 2, 5), ANY(3, 5, 4))
_check('dot-complex', lambda a, b: numpy.dot(a, function.Array.cast(b)), numpy.dot, ANC(1, 2, 5), ANC(3, 5, 4))
_check('Array_dot', lambda a, b: function.Array.cast(a).dot(b), lambda a, b: a.dot(b), ANY(4), ANY(4))
_check('vdot', lambda a, b: numpy.vdot(a, function.Array.cast(b)), numpy.vdot, ANY(4, 2, 4), ANY(4, 2, 4).T)
_check('vdot-complex', lambda a, b: numpy.vdot(a, function.Array.cast(b)), numpy.vdot, ANC(4, 2, 4) / 10, ANC(4, 2, 4).T / 10)
_check('trace', lambda a: numpy.trace(function.Array.cast(a), 0, 0, 2), lambda a: numpy.trace(a, 0, 0, 2), ANY(3, 2, 3))
_check('norm', lambda a: numpy.linalg.norm(function.Array.cast(a), axis=1), lambda a: numpy.linalg.norm(a, axis=1), ANY(2, 3))
_check('norm-complex', lambda a: numpy.linalg.norm(function.Array.cast(a), axis=1), lambda a: numpy.linalg.norm(a, axis=1), ANC(2, 3))
_check('normalized', function.normalized, lambda a: a / numpy.linalg.norm(a, axis=1)[:, None], ANY(2, 3))
_check('normalized-complex', function.normalized, lambda a: a / numpy.linalg.norm(a, axis=1)[:, None], ANC(2, 3))
_check('Array_normalized', lambda a: function.Array.cast(a).normalized(), lambda a: a / numpy.linalg.norm(a, axis=1)[:, None], ANY(2, 3))
_check('inv', lambda a: numpy.linalg.inv(a+3*function.eye(3)), lambda a: numpy.linalg.inv(a+3*numpy.eye(3)), ANY(2, 3, 3))
_check('inv-complex', lambda a: numpy.linalg.inv(a+3*function.eye(3)), lambda a: numpy.linalg.inv(a+3*numpy.eye(3)), ANC(2, 3, 3))
_check('det', lambda a: numpy.linalg.det(a+3*function.eye(3)), lambda a: numpy.linalg.det(a+3*numpy.eye(3)), ANY(2, 3, 3))
_check('eigval', lambda a: numpy.linalg.eig(function.Array.cast(a))[0], lambda a: numpy.linalg.eig(a)[0], ANY(3, 3))
_check('eigvec', lambda a: numpy.linalg.eig(function.Array.cast(a))[1], lambda a: numpy.linalg.eig(a)[1], ANY(3, 3))
_check('eigval_symmetric', lambda a: numpy.linalg.eigh(function.Array.cast(a+a.T))[0], lambda a: numpy.linalg.eigh(a+a.T)[0], ANY(3, 3))
_check('eigvec_symmetric', lambda a: numpy.linalg.eigh(function.Array.cast(a+a.T))[1], lambda a: numpy.linalg.eigh(a+a.T)[1], ANY(3, 3))
_check('diagonal', lambda a: numpy.diagonal(function.Array.cast(a), axis1=0, axis2=2), lambda a: numpy.diagonal(a, axis1=0, axis2=2), ANY(3, 2, 3))
_check('diagonal-posoffset', lambda a: numpy.diagonal(function.Array.cast(a), +1, axis1=0, axis2=2), lambda a: numpy.diagonal(a, +1, axis1=0, axis2=2), ANY(3, 2, 3))
_check('diagonal-negoffset', lambda a: numpy.diagonal(function.Array.cast(a), -2, axis1=0, axis2=2), lambda a: numpy.diagonal(a, -2, axis1=0, axis2=2), ANY(3, 2, 3))
_check('diagonalize', function.diagonalize, numpy.diag, ANY(3))
_check('cross2', lambda a, b: numpy.cross(a, function.Array.cast(b)), numpy.cross, ANY(3,2), 1+ANY(3,2))
_check('cross3', lambda a, b: numpy.cross(a, function.Array.cast(b)), numpy.cross, ANY(2,3), 1+ANY(2,3))
_check('cross3-complex', lambda a, b: numpy.cross(function.Array.cast(a), b), numpy.cross, ANC(2,3), 1-1j+ANC(2,3))
_check('cross3-axes', lambda a, b: numpy.cross(a, function.Array.cast(b), axisa=2, axisb=0, axisc=1), lambda a, b: numpy.cross(a, b, axisa=2, axisb=0, axisc=1), ANY(2,1,3), ANY(3,1,4))
_check('outer', function.outer, lambda a, b: a[:, None]*b[None, :], ANY(2, 3), ANY(4, 3))
_check('outer_self', function.outer, lambda a: a[:, None]*a[None, :], ANY(2, 3))
_check('square', lambda a: numpy.square(function.Array.cast(a)), numpy.square, ANY(4))
_check('hypot', lambda a, b: numpy.hypot(a, function.Array.cast(b)), numpy.hypot, ANY(4, 4), ANY(4, 4))

_check('transpose', lambda a: numpy.transpose(function.Array.cast(a), [0, 1, 3, 2]), lambda a: a.transpose([0, 1, 3, 2]), INT(1, 2, 3, 4))
_check('Array_transpose', lambda a: function.Array.cast(a).transpose([0, 1, 3, 2]), lambda a: a.transpose([0, 1, 3, 2]), INT(1, 2, 3, 4))
_check('insertaxis', lambda a: function.insertaxis(a, 2, 3), lambda a: numpy.repeat(numpy.expand_dims(a, 2), 3, 2), INT(3, 2, 4))
_check('expand_dims', lambda a: function.expand_dims(a, 1), lambda a: numpy.expand_dims(a, 1), INT(2, 3))
_check('repeat', lambda a: numpy.repeat(function.Array.cast(a), 3, 1), lambda a: numpy.repeat(a, 3, 1), INT(2, 1, 4))
_check('swapaxes', lambda a: numpy.swapaxes(function.Array.cast(a), 1, 2), lambda a: numpy.transpose(a, (0, 2, 1)), INT(2, 3, 4))
_check('Array_swapaxes', lambda a: function.Array.cast(a).swapaxes(1, 2), lambda a: numpy.transpose(a, (0, 2, 1)), INT(2, 3, 4))
_check('reshape', lambda a: numpy.reshape(function.Array.cast(a), (2, -1, 5)), lambda a: numpy.reshape(a, (2, -1, 5)), INT(2, 3, 4, 5))
_check('ravel', lambda a: numpy.ravel(function.Array.cast(a)), lambda a: numpy.ravel(a), INT(2, 3, 4))
_check('unravel', lambda a: function.unravel(a, 1, (3, 4)), lambda a: numpy.reshape(a, (2, 3, 4, 5)), INT(2, 12, 5))
_check('take', lambda a: numpy.take(function.Array.cast(a), numpy.array([[0, 2], [1, 3]]), 1), lambda a: numpy.take(a, numpy.array([[0, 2], [1, 3]]), 1), INT(3, 4, 5))
_check('compress', lambda a: numpy.compress(numpy.array([False, True, False, True]), function.Array.cast(a), 1), lambda a: numpy.compress(numpy.array([False, True, False, True]), a, 1), INT(3, 4, 5))
_check('get', lambda a: function.get(a, 1, 1), lambda a: numpy.take(a, 1, 1), INT(3, 4, 5))
_check('scatter', lambda a: function.scatter(a, 3, [2, 0]), lambda a: numpy.stack([a[:, 1], numpy.zeros([4]), a[:, 0]], axis=1), INT(4, 2))
_check('kronecker', lambda a: function.kronecker(a, 1, 3, 1), lambda a: numpy.stack([numpy.zeros_like(a), a, numpy.zeros_like(a)], axis=1), INT(4, 4))
_check('concatenate', lambda a, b: numpy.concatenate([a, function.Array.cast(b)], axis=1), lambda a, b: numpy.concatenate([a, b], axis=1), INT(4, 2, 1), INT(4, 3, 1))
_check('stack', lambda a, b: numpy.stack([a, function.Array.cast(b)], 1), lambda a, b: numpy.stack([a, b], 1), INT(4, 2), INT(4, 2))
_check('choose', lambda a, b: numpy.choose([0,1], [a, function.Array.cast(b)]), lambda a, b: numpy.choose([0,1], [a, b]), INT(4, 1), INT(1, 2))
_check('einsum', lambda a, b: numpy.einsum('ik,jkl->ijl', a, function.Array.cast(b)), lambda a, b: numpy.einsum('ik,jkl->ijl', a, b), ANY(2, 4), ANY(3, 4, 2))
_check('einsum-diag', lambda a: numpy.einsum('ijii->ji', function.Array.cast(a)), lambda a: numpy.einsum('ijii->ji', a), ANY(3, 2, 3, 3))
_check('einsum-sum', lambda a: numpy.einsum('ijk->i', function.Array.cast(a)), lambda a: numpy.einsum('ijk->i', a), ANY(2, 3, 4))
_check('einsum-implicit', lambda a: numpy.einsum('i...i', function.Array.cast(a)), lambda a: numpy.einsum('i...i', a), ANY(3, 2, 3))

_check('Array_getitem_scalar', lambda a: function.Array.cast(a)[0], lambda a: a[0], INT(5, 3, 2))
_check('Array_getitem_scalar_scalar', lambda a: function.Array.cast(a)[0, 1], lambda a: a[0, 1], INT(5, 3, 2))
_check('Array_getitem_matrix', lambda a: function.Array.cast(a)[numpy.array([[4,3],[2,1]])], lambda a: a[numpy.array([[4,3],[2,1]])], INT(5, 3, 2))
_check('Array_getitem_slice_step', lambda a: function.Array.cast(a)[:, ::2], lambda a: a[:, ::2], INT(5, 3, 2))
_check('Array_getitem_ellipsis_scalar', lambda a: function.Array.cast(a)[..., 1], lambda a: a[..., 1], INT(5, 3, 2))
_check('Array_getitem_ellipsis_scalar_newaxis', lambda a: function.Array.cast(a)[..., 1, None], lambda a: a[..., 1, None], INT(5, 3, 2))

_check('add_T', lambda a: function.add_T(a, (1, 2)), lambda a: a + a.transpose((0, 2, 1)), INT(5, 2, 2))
_check('Array_add_T', lambda a: function.Array.cast(a).add_T((1, 2)), lambda a: a + a.transpose((0, 2, 1)), INT(5, 2, 2))

_check('searchsorted', lambda a: numpy.searchsorted(numpy.linspace(0, 1, 9), function.Array.cast(a)), lambda a: numpy.searchsorted(numpy.linspace(0, 1, 9), a), POS(4, 2))
_check('searchsorted_sorter', lambda a: numpy.searchsorted([.2,.8,.4,0,.6,1], function.Array.cast(a), sorter=[3,0,2,4,1,5]), lambda a: numpy.searchsorted([.2,.8,.4,0,.6,1], a, sorter=[3,0,2,4,1,5]), POS(4, 2))
_check('interp', lambda a: numpy.interp(function.Array.cast(a), [-.5,0,.5], [0,1,0]), lambda a: numpy.interp(a, [-.5,0,.5], [0,1,0]), ANY(4, 2))
_check('interp_lr', lambda a: numpy.interp(function.Array.cast(a), [-.5,0,.5], [0,1,0], left=-10, right=+10), lambda a: numpy.interp(a, [-.5,0,.5], [0,1,0], left=-10, right=+10), ANY(4, 2))


class Unlower(TestCase):

    def test(self):
        e = evaluable.Argument('arg', tuple(map(evaluable.constant, (2, 3, 4, 5))), int)
        arguments = {'arg': ((2, 3), int)}
        f = function._Unlower(e, frozenset(), arguments, function.LowerArgs((2, 3), {}, {}))
        self.assertEqual(f.shape, (4, 5))
        self.assertEqual(f.dtype, int)
        self.assertEqual(f.arguments, arguments)
        self.assertEqual(f.lower(function.LowerArgs((2, 3), {}, {})), e)
        with self.assertRaises(ValueError):
            f.lower(function.LowerArgs((3, 4), {}, {}))


class Custom(TestCase):

    def assertEvalAlmostEqual(self, factual, fdesired, **args):
        with self.subTest('0d-points'):
            self.assertAllAlmostEqual(factual.as_evaluable_array.eval(**args), fdesired.as_evaluable_array.eval(**args))
        with self.subTest('1d-points'):
            lower_args = function.LowerArgs((evaluable.asarray(5),), {}, {})
            self.assertAllAlmostEqual(factual.lower(lower_args).eval(**args), fdesired.lower(lower_args).eval(**args))
        with self.subTest('2d-points'):
            lower_args = function.LowerArgs((evaluable.asarray(5), evaluable.asarray(6)), {}, {})
            self.assertAllAlmostEqual(factual.lower(lower_args).eval(**args), fdesired.lower(lower_args).eval(**args))

    def assertMultipy(self, leftval, rightval):

        for npointwise in range(leftval. ndim):

            class Multiply(function.Custom):

                def __init__(self, left, right):
                    left = function.asarray(left)
                    right = function.asarray(right)
                    if left.shape != right.shape:
                        raise ValueError('left and right arguments not aligned')
                    super().__init__(args=(left, right), shape=left.shape[npointwise:], dtype=left.dtype, npointwise=npointwise)

                @staticmethod
                def evalf(left, right):
                    return left * right

                @staticmethod
                def partial_derivative(iarg, left, right):
                    if iarg == 0:
                        return functools.reduce(function.diagonalize, range(right.ndim), right)
                    elif iarg == 1:
                        return functools.reduce(function.diagonalize, range(left.ndim), left)
                    else:
                        raise NotImplementedError

            args = dict(left=leftval, right=rightval)
            left = function.Argument('left', args['left'].shape, dtype=float if leftval.dtype.kind == 'f' else int)
            right = function.Argument('right', args['right'].shape, dtype=float if rightval.dtype.kind == 'f' else int)
            actual = Multiply(left, right)
            desired = left * right
            self.assertEvalAlmostEqual(actual, desired, **args)
            self.assertEvalAlmostEqual(actual.derivative('left'), desired.derivative('left'), **args)
            self.assertEvalAlmostEqual(actual.derivative('right'), desired.derivative('right'), **args)

    def test_multiply_float_float(self):
        self.assertMultipy(numpy.array([[1, 2, 3], [4, 5, 6]], float), numpy.array([[2, 3, 4], [1, 2, 3]], float))

    def test_multiply_float_int(self):
        self.assertMultipy(numpy.array([[1, 2, 3], [4, 5, 6]], float), numpy.array([[2, 3, 4], [1, 2, 3]], int))

    def test_multiply_int_int(self):
        self.assertMultipy(numpy.array([[1, 2, 3], [4, 5, 6]], int), numpy.array([[2, 3, 4], [1, 2, 3]], int))

    def test_singleton_points(self):

        class Func(function.Custom):

            def __init__(self):
                super().__init__(args=(), shape=(3,), dtype=int)

            @staticmethod
            def evalf():
                return numpy.array([1, 2, 3])[None]

        self.assertEvalAlmostEqual(Func(), function.Array.cast([1, 2, 3]))

    def test_consts(self):

        class Func(function.Custom):

            def __init__(self, offset, base1, exp1, base2, exp2):
                base1 = function.asarray(base1)
                base2 = function.asarray(base2)
                assert base1.shape == base2.shape
                super().__init__(args=(offset, base1, exp1.__index__(), base2, exp2.__index__()), shape=base1.shape, dtype=float)

            @staticmethod
            def evalf(offset, base1, exp1, base2, exp2):
                return offset + base1**exp1 + base2**exp2

            @staticmethod
            def partial_derivative(iarg, offset, base1, exp1, base2, exp2):
                if iarg == 1:
                    if exp1 == 0:
                        return function.zeros(base1.shape + base1.shape)
                    else:
                        return functools.reduce(function.diagonalize, range(base1.ndim), exp1*base1**(exp1-1))
                    return exp1*base1**(exp1-1)
                elif iarg == 3:
                    if exp2 == 0:
                        return function.zeros(base2.shape + base2.shape)
                    else:
                        return functools.reduce(function.diagonalize, range(base2.ndim), exp2*base2**(exp2-1))
                else:
                    raise NotImplementedError

        b1 = function.Argument('b1', (3,))
        b2 = function.Argument('b2', (3,))
        actual = Func(4, b1, 2, b2, 3)
        desired = 4 + b1**2 + b2**3
        args = dict(b1=numpy.array([1, 2, 3]), b2=numpy.array([4, 5, 6]))
        self.assertEvalAlmostEqual(actual, desired, **args)
        self.assertEvalAlmostEqual(actual.derivative('b1'), desired.derivative('b1'), **args)
        self.assertEvalAlmostEqual(actual.derivative('b2'), desired.derivative('b2'), **args)

    def test_deduplication(self):

        class A(function.Custom):

            @staticmethod
            def evalf():
                pass

            @staticmethod
            def partial_derivative(iarg):
                pass

        class B(function.Custom):

            @staticmethod
            def evalf():
                pass

            @staticmethod
            def partial_derivative(iarg):
                pass

        a = A(args=(function.Argument('a', (2, 3)),), shape=(), dtype=float).as_evaluable_array
        b = A(args=(function.Argument('a', (2, 3)),), shape=(), dtype=float).as_evaluable_array
        c = B(args=(function.Argument('a', (2, 3)),), shape=(), dtype=float).as_evaluable_array
        d = A(args=(function.Argument('a', (2, 3)),), shape=(), dtype=int).as_evaluable_array
        e = A(args=(function.Argument('a', (2, 3)),), shape=(2, 3), dtype=float).as_evaluable_array
        f = A(args=(function.Argument('a', (2, 3)), 1), shape=(), dtype=float).as_evaluable_array
        g = A(args=(function.Argument('b', (2, 3)),), shape=(), dtype=float).as_evaluable_array
        h = A(args=(function.Argument('a', (2, 3)),), shape=(), dtype=float, npointwise=1).as_evaluable_array

        self.assertIs(a, b)
        self.assertEqual(len({b, c, d, e, f, g, h}), 7)

    def test_node_details(self):

        class A(function.Custom):
            pass

        self.assertEqual(A(args=(), shape=(), dtype=float).as_evaluable_array._node_details, 'A')

    def test_evaluable_argument(self):
        with self.assertRaisesRegex(ValueError, 'It is not allowed to call this function with a `nutils.evaluable.Evaluable` argument.'):
            function.Custom(args=(evaluable.Argument('a', ()),), shape=(), dtype=float)

    def test_pointwise_no_array_args(self):
        with self.assertRaisesRegex(ValueError, 'Pointwise axes can only be used in combination with at least one `function.Array` argument.'):
            function.Custom(args=(1, 2), shape=(), dtype=float, npointwise=3)

    def test_pointwise_missing_axes(self):
        with self.assertRaisesRegex(ValueError, 'All arrays must have at least 3 axes.'):
            function.Custom(args=(function.Argument('a', (2, 3)),), shape=(), dtype=float, npointwise=3)

    def test_no_array_args_invalid_shape(self):

        class Test(function.Custom):
            @staticmethod
            def evalf():
                return numpy.array([1, 2, 3])

        with self.assertRaises(ValueError):
            Test((), (), int).eval()

    def test_pointwise_singleton_expansion(self):

        class Test(function.Custom):
            def __init__(self, args, shapes, npointwise):
                super().__init__(args=(shapes, *args), shape=(), dtype=float, npointwise=npointwise)

            @staticmethod
            def evalf(shapes, *args):
                for shape, arg in zip(shapes, args):
                    self.assertEqual(tuple(map(int, arg.shape)), shape)
                return numpy.zeros(args[0].shape[:1], float)

        Z = lambda *s: function.zeros(s)
        Test((Z(1, 3, 2), Z(2, 1, 4)), ((6, 2), (6, 4)), 2).as_evaluable_array.eval()

    def test_partial_derivative_invalid_shape(self):

        class Test(function.Custom):
            @staticmethod
            def partial_derivative(iarg, arg):
                return function.zeros((2, 3, 4))

        arg = function.Argument('arg', (5,))
        with self.assertRaisesRegex(ValueError, '`partial_derivative` to argument 0 returned an array with shape'):
            Test((arg,), (5,), float).derivative(arg).as_evaluable_array

    def test_grad(self):

        class Test(function.Custom):
            def eval(arg):
                return arg
            @staticmethod
            def partial_derivative(iarg, arg):
                return function.ones(arg.shape)

        topo, geom = mesh.line(3)
        smpl = topo.sample('bezier', 2)
        test = Test((geom**2,), geom.shape, float)
        self.assertAllAlmostEqual(*smpl.eval([2 * geom, function.grad(test, geom)]))


class broadcasting(TestCase):

    def assertBroadcasts(self, desired, *from_shapes):
        with self.subTest('broadcast_arrays'):
            broadcasted = function.broadcast_arrays(*(function.Argument('arg{}'.format(i), s) for i, s in enumerate(from_shapes)))
            actual = tuple(array.shape for array in broadcasted)
            self.assertEqual(actual, (desired,)*len(from_shapes))
        with self.subTest('broadcast_shapes'):
            actual = function.broadcast_shapes(*from_shapes)
            self.assertEqual(actual, desired)

    def test_singleton_expansion(self):
        self.assertBroadcasts((3, 2, 3), (1, 2, 3), (3, 1, 3), (3, 1, 1))

    def test_prepend_axes(self):
        self.assertBroadcasts((3, 2, 3), (3, 2, 3), (3,), (2, 3))

    def test_both(self):
        self.assertBroadcasts((3, 2, 3), (3, 2, 3), (3,), (1, 1))

    def test_incompatible_shape(self):
        with self.assertRaisesRegex(ValueError, 'cannot broadcast'):
            function.broadcast_shapes((1, 2), (2, 3))

    def test_no_shapes(self):
        with self.assertRaisesRegex(ValueError, 'expected at least one shape but got none'):
            function.broadcast_shapes()

    def test_broadcast_to_decrease_dimension(self):
        with self.assertRaisesRegex(ValueError, 'cannot broadcast array .* because the dimension decreases'):
            numpy.broadcast_to(function.Argument('a', (2, 3, 4)), (3, 4))

    def test_broadcast_to_invalid_length(self):
        with self.assertRaisesRegex(ValueError, 'cannot broadcast array .* because input axis .* is neither singleton nor has the desired length'):
            numpy.broadcast_to(function.Argument('a', (2, 3, 4)), (2, 5, 4))


@parametrize
class sampled(TestCase):

    def setUp(self):
        super().setUp()
        self.domain, geom = mesh.unitsquare(4, self.etype)
        basis = self.domain.basis('std', degree=1)
        numpy.random.seed(0)
        self.f = basis.dot(numpy.random.uniform(size=len(basis)))
        sample = self.domain.sample('gauss', 2)
        print(sample.eval(self.f))
        self.f_sampled = sample.asfunction(sample.eval(self.f))

    def test_isarray(self):
        self.assertTrue(function.isarray(self.f_sampled))

    def test_values(self):
        diff = self.domain.integrate(self.f - self.f_sampled, ischeme='gauss2')
        self.assertAllAlmostEqual(diff, 0)

    def test_pointset(self):
        with self.assertRaises(ValueError):
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
            self.f = left + numpy.sin(x) * mid + x**2 * right
        else:
            self.f = function.piecewise(x, [.2, .8], 1, numpy.sin(x), x**2)

    def test_evalf(self):
        f_ = self.domain.sample('uniform', 4).eval(self.f)  # x=.125, .375, .625, .875
        assert numpy.equal(f_, [1, numpy.sin(.375), numpy.sin(.625), .875**2]).all()

    def test_deriv(self):
        g_ = self.domain.sample('uniform', 4).eval(function.grad(self.f, self.geom))  # x=.125, .375, .625, .875
        assert numpy.equal(g_, [[0], [numpy.cos(.375)], [numpy.cos(.625)], [2*.875]]).all()


piecewise(partition=False)
piecewise(partition=True)


class elemwise(TestCase):

    def setUp(self):
        super().setUp()
        self.index = function._Wrapper(lambda: evaluable.InRange(evaluable.Argument('index', (), int), evaluable.constant(5)), shape=(), dtype=int)
        self.data = tuple(map(types.frozenarray, (
            numpy.arange(1, 7, dtype=float).reshape(2, 3),
            numpy.arange(2, 8, dtype=float).reshape(2, 3),
            numpy.arange(3, 9, dtype=float).reshape(2, 3),
            numpy.arange(4, 10, dtype=float).reshape(2, 3),
            numpy.arange(6, 12, dtype=float).reshape(2, 3),
        )))
        with self.assertWarns(warnings.NutilsDeprecationWarning):
            self.func = function.Elemwise(self.data, self.index, float)

    def test_evalf(self):
        for i in range(5):
            with self.subTest(i=i):
                numpy.testing.assert_array_almost_equal(self.func.as_evaluable_array.eval(index=i), self.data[i])

    def test_shape(self):
        for i in range(5):
            with self.subTest(i=i):
                self.assertEqual(self.func.size, self.data[i].size)


class replace_arguments(TestCase):

    def test_array(self):
        a = function.Argument('a', (2,))
        b = function.Array.cast([1, 2.])
        self.assertEqual(function.replace_arguments(a, dict(a=b)).as_evaluable_array, b.as_evaluable_array)

    def test_argument(self):
        a = function.Argument('a', (2,))
        b = function.Argument('b', (2,))
        self.assertEqual(function.replace_arguments(a, dict(a='b')).as_evaluable_array, b.as_evaluable_array)

    def test_argument_array(self):
        a = function.Argument('a', (2,))
        b = function.Argument('b', (2,))
        c = function.Array.cast([1, 2.])
        self.assertEqual(function.replace_arguments(function.replace_arguments(a, dict(a=b)), dict(b=c)).as_evaluable_array, c.as_evaluable_array)

    def test_swap(self):
        a = function.Argument('a', (2,))
        b = function.Argument('b', (2,))
        self.assertEqual(function.replace_arguments(2*a+3*b, dict(a=b, b=a)).as_evaluable_array, (2*b+3*a).as_evaluable_array)

    def test_ignore_replaced(self):
        a = function.Argument('a', (2,))
        b = function.Array.cast([1, 2.])
        c = function.Array.cast([2, 3.])
        self.assertEqual(function.replace_arguments(function.replace_arguments(a, dict(a=b)), dict(a=c)).as_evaluable_array, b.as_evaluable_array)

    def test_ignore_recursion(self):
        a = function.Argument('a', (2,))
        self.assertEqual(function.replace_arguments(a, dict(a=2*a)).as_evaluable_array, (2*a).as_evaluable_array)

    def test_replace_derivative(self):
        a = function.Argument('a', ())
        b = function.Argument('b', ())
        actual = function.replace_arguments(function.derivative(a, a), dict(a=b))
        self.assertEqual(actual.as_evaluable_array.simplified, evaluable.ones(()).simplified)
        actual = function.replace_arguments(function.derivative(a, b), dict(a=b))
        self.assertEqual(actual.as_evaluable_array.simplified, evaluable.zeros(()).simplified)
        actual = function.derivative(function.replace_arguments(a, dict(a=b)), b)
        self.assertEqual(actual.as_evaluable_array.simplified, evaluable.ones(()).simplified)
        actual = function.derivative(function.replace_arguments(a, dict(a=b)), a)
        self.assertEqual(actual.as_evaluable_array.simplified, evaluable.zeros(()).simplified)

    def test_different_shape(self):
        with self.assertRaisesRegex(ValueError, "Argument 'foo' has shape \\(2,\\) but the replacement has shape \\(3, 4\\)."):
            function.replace_arguments(function.Argument('foo', (2,), dtype=float), dict(foo=function.zeros((3, 4), dtype=float)))

    def test_different_dtype(self):
        with self.assertRaisesRegex(ValueError, "Argument 'foo' has dtype int but the replacement has dtype float."):
            function.replace_arguments(function.Argument('foo', (), dtype=int), dict(foo=function.zeros((), dtype=float)))

    def test_nonempty_spaces(self):
        topo, geom = mesh.unitsquare(1, 'square')
        with self.assertRaisesRegex(ValueError, "replacement functions cannot contain spaces, but replacement for Argument 'foo' contains space X."):
            function.replace_arguments(function.Argument('foo', (2,), dtype=float), dict(foo=geom))


class dotarg(TestCase):

    def assertEvalAlmostEqual(self, f_actual, desired, **arguments):
        self.assertEqual(f_actual.shape, desired.shape)
        actual = f_actual.as_evaluable_array.eval(**arguments)
        self.assertEqual(actual.shape, desired.shape)
        self.assertAllAlmostEqual(actual, desired)

    def test(self):
        a = numpy.ones((), dtype=float)
        a2 = numpy.arange(2, dtype=float)
        a23 = numpy.arange(6, dtype=float).reshape(2, 3)
        a24 = numpy.arange(8, dtype=float).reshape(2, 4)
        a243 = numpy.arange(24, dtype=float).reshape(2, 4, 3)
        a4 = numpy.arange(4, dtype=float)
        a45 = numpy.arange(20, dtype=float).reshape(4, 5)
        self.assertEvalAlmostEqual(function.dotarg('arg'), a, arg=a)
        self.assertEvalAlmostEqual(function.dotarg('arg', shape=(2, 3)), a23, arg=a23)
        self.assertEvalAlmostEqual(function.dotarg('arg', a4), numpy.einsum('i,i->', a4, a4), arg=a4)
        self.assertEvalAlmostEqual(function.dotarg('arg', a45), numpy.einsum('i,ij->j', a4, a45), arg=a4)
        self.assertEvalAlmostEqual(function.dotarg('arg', a24, shape=(3,)), numpy.einsum('ij,ik->jk', a23, a24), arg=a23)
        self.assertEvalAlmostEqual(function.dotarg('arg', a2, a45), numpy.einsum('ij,i,jk->k', a24, a2, a45), arg=a24)
        self.assertEvalAlmostEqual(function.dotarg('arg', a2, a45, shape=(3,)), numpy.einsum('ijk,i,jl->kl', a243, a2, a45), arg=a243)


class jacobian(TestCase):

    def setUp(self):
        super().setUp()
        self.domain, self.geom = mesh.unitsquare(1, 'square')
        self.basis = self.domain.basis('std', degree=1)
        arg = function.Argument('dofs', [4])
        self.v = self.basis.dot(arg)
        self.X = (self.geom[numpy.newaxis, :] * [[0, 1], [-self.v, 0]]).sum(-1)  # X_i = <x_1, -2 x_0>_i
        self.J = function.J(self.X)
        self.dJ = function.derivative(self.J, arg)

    def test_shape(self):
        self.assertEqual(self.J.shape, ())
        self.assertEqual(self.dJ.shape, (4,))

    def test_value(self):
        values = self.domain.sample('uniform', 2).eval(self.J, dofs=[2]*4)
        numpy.testing.assert_almost_equal(values, [2]*4)
        values1, values2 = self.domain.sample('uniform', 2).eval([self.J,
                                                                  self.v + self.v.grad(self.geom)[0] * self.geom[0]], dofs=[1, 2, 3, 10])
        numpy.testing.assert_almost_equal(values1, values2)

    def test_derivative(self):
        values1, values2 = self.domain.sample('uniform', 2).eval([self.dJ,
                                                                  self.basis + self.basis.grad(self.geom)[:, 0] * self.geom[0]], dofs=[1, 2, 3, 10])
        numpy.testing.assert_almost_equal(values1, values2)

    def test_zeroderivative(self):
        otherarg = function.Argument('otherdofs', (10,))
        smpl = self.domain.sample('uniform', 2)
        values = smpl.eval(function.derivative(self.dJ, otherarg))
        self.assertEqual(values.shape[1:], self.dJ.shape + otherarg.shape)
        self.assertAllEqual(values, numpy.zeros((smpl.npoints, *self.dJ.shape, *otherarg.shape)))

    def test_invalid_dimension_spaceless_geometry(self):
        with self.assertRaisesRegex(ValueError, 'The jacobian of a constant \\(in space\\) geometry must have dimension zero.'):
            function.jacobian(function.ones((1,)))

    def test_tip_larger_than_geom(self):
        topo, geom = mesh.newrectilinear([2, 3])
        with self.assertRaisesRegex(ValueError, 'Expected a dimension of the tip coordinate system not greater than'):
            function.jacobian(geom, 3)

    def test_invalid_tip_dim(self):
        topo, geom = mesh.newrectilinear([2, 3])
        J = function.jacobian(geom, 2)
        with self.assertRaisesRegex(ValueError, 'Expected a tip dimension of 2 but got 1.'):
            topo.boundary.integral(J, degree=0).as_evaluable_array


@parametrize
class derivative(TestCase):

    def assertEvalAlmostEqual(self, topo, factual, fdesired):
        actual, desired = topo.sample('uniform', 2).eval([function.asarray(factual), function.asarray(fdesired)])
        self.assertAllAlmostEqual(actual, desired)

    def test_grad_0d(self):
        domain, (x,) = mesh.rectilinear([1])
        x = 2*x-0.5
        self.assertEvalAlmostEqual(domain, self.grad(x**2, x), 2*x)

    def test_grad_1d(self):
        domain, x = mesh.rectilinear([1]*2)
        x = 2*x-0.5
        self.assertEvalAlmostEqual(domain, self.grad([x[0]**2*x[1], x[1]**2], x), [[2*x[0]*x[1], x[0]**2], [0, 2*x[1]]])

    def test_grad_2d(self):
        domain, x = mesh.rectilinear([1]*4)
        x = 2*x-0.5
        x = function.unravel(x, 0, (2, 2))
        self.assertEvalAlmostEqual(domain, self.grad(x, x), numpy.eye(4, 4).reshape(2, 2, 2, 2))

    def test_grad_3d(self):
        domain, x = mesh.rectilinear([1]*4)
        x = 2*x-0.5
        x = function.unravel(function.unravel(x, 0, (2, 2)), 0, (2, 1))
        self.assertEvalAlmostEqual(domain, self.grad(x, x), numpy.eye(4, 4).reshape(2, 1, 2, 2, 1, 2))

    def test_curl(self):
        domain, geom = mesh.rectilinear([[-1, 1]]*3)
        x, y, z = geom
        self.assertEvalAlmostEqual(domain, self.curl([y, -x, z], geom), [0, 0, -2])
        self.assertEvalAlmostEqual(domain, self.curl([0, -x**2, 0], geom), [0, 0, -2*x])
        self.assertEvalAlmostEqual(domain, self.curl([[x, -z, y], [0, x*z, 0]], geom), [[2, 0, 0], [-x, 0, z]])
        self.assertEvalAlmostEqual(domain, self.curl(self.grad(x*y+z, geom), geom), [0, 0, 0])
        with self.assertRaisesRegex(ValueError, 'Expected a geometry with shape'):
            self.curl([x, y, z], geom[:2])
        with self.assertRaisesRegex(ValueError, 'Expected a function with at least 1 axis but got 0.'):
            self.curl(function.zeros(()), geom)
        with self.assertRaisesRegex(ValueError, 'Expected a function with a trailing axis of length 3'):
            self.curl(function.zeros((3, 2)), geom)

    def test_div(self):
        domain, x = mesh.rectilinear([1]*2)
        x = 2*x-0.5
        self.assertEvalAlmostEqual(domain, self.div([x[0]**2+x[1], x[1]**2-x[0]], x), 2*x[0]+2*x[1])

    def test_laplace(self):
        domain, x = mesh.rectilinear([1]*2)
        x = 2*x-0.5
        self.assertEvalAlmostEqual(domain, self.laplace(x[0]**2*x[1]-x[1]**2, x), 2*x[1]-2)

    def test_symgrad(self):
        domain, x = mesh.rectilinear([1]*2)
        x = 2*x-0.5
        self.assertEvalAlmostEqual(domain, self.symgrad([x[0]**2*x[1], x[1]**2], x), [[2*x[0]*x[1], 0.5*x[0]**2], [0.5*x[0]**2, 2*x[1]]])

    def test_normal_0d(self):
        domain, (x,) = mesh.rectilinear([1])
        x = 2*x-0.5
        self.assertEvalAlmostEqual(domain.boundary['right'], self.normal(x), 1)
        self.assertEvalAlmostEqual(domain.boundary['left'], self.normal(x), -1)

    def test_normal_1d(self):
        domain, x = mesh.rectilinear([1]*2)
        x = 2*x-0.5
        for bnd, n in ('right', [1, 0]), ('left', [-1, 0]), ('top', [0, 1]), ('bottom', [0, -1]):
            self.assertEvalAlmostEqual(domain.boundary[bnd], self.normal(x), n)

    def test_normal_2d(self):
        domain, x = mesh.rectilinear([1]*2)
        x = 2*x-0.5
        x = function.unravel(x, 0, [2, 1])
        for bnd, n in ('right', [1, 0]), ('left', [-1, 0]), ('top', [0, 1]), ('bottom', [0, -1]):
            self.assertEvalAlmostEqual(domain.boundary[bnd], self.normal(x), numpy.array(n)[:, numpy.newaxis])

    def test_normal_3d(self):
        domain, x = mesh.rectilinear([1]*2)
        x = 2*x-0.5
        x = function.unravel(function.unravel(x, 0, [2, 1]), 0, [1, 2])
        for bnd, n in ('right', [1, 0]), ('left', [-1, 0]), ('top', [0, 1]), ('bottom', [0, -1]):
            self.assertEvalAlmostEqual(domain.boundary[bnd], self.normal(x), numpy.array(n)[numpy.newaxis, :, numpy.newaxis])

    def test_normal_manifold(self):
        domain, geom = mesh.rectilinear([1]*2)
        x = numpy.stack([geom[0], geom[1], geom[0]**2 - geom[1]**2])
        n = self.normal(x) # boundary normal
        N = self.normal(x, geom) # exterior normal
        k = -.5 * self.div(N, x, -1) # mean curvature
        dA = function.jacobian(x, 2)
        dL = function.jacobian(x, 1)
        v1 = domain.integrate(2 * k * N * dA, degree=16)
        v2 = domain.boundary.integrate(n * dL, degree=16)
        self.assertAllAlmostEqual(v1, v2) # divergence theorem in curved space

    def test_dotnorm(self):
        domain, x = mesh.rectilinear([1]*2)
        x = 2*x-0.5
        for bnd, desired in ('right', 1), ('left', -1), ('top', 0), ('bottom', 0):
            self.assertEvalAlmostEqual(domain.boundary[bnd], self.dotnorm([1, 0], x), desired)

    def test_tangent(self):
        domain, x = mesh.rectilinear([1]*2)
        x = 2*x-0.5
        for bnd, desired in ('right', [0, 1]), ('left', [0, 1]), ('top', [-1, 0]), ('bottom', [-1, 0]):
            self.assertEvalAlmostEqual(domain.boundary[bnd], self.tangent(x, [-1, 1]), desired)

    def test_ngrad(self):
        domain, x = mesh.rectilinear([1]*2)
        x = 2*x-0.5
        for bnd, desired in ('right', [2*x[0]*x[1], 0]), ('left', [-2*x[0]*x[1], 0]), ('top', [x[0]**2, 2*x[1]]), ('bottom', [-x[0]**2, -2*x[1]]):
            self.assertEvalAlmostEqual(domain.boundary[bnd], self.ngrad([x[0]**2*x[1], x[1]**2], x), desired)

    def test_nsymgrad(self):
        domain, x = mesh.rectilinear([1]*2)
        x = 2*x-0.5
        for bnd, desired in ('right', [2*x[0]*x[1], 0.5*x[0]**2]), ('left', [-2*x[0]*x[1], -0.5*x[0]**2]), ('top', [0.5*x[0]**2, 2*x[1]]), ('bottom', [-0.5*x[0]**2, -2*x[1]]):
            self.assertEvalAlmostEqual(domain.boundary[bnd], self.nsymgrad([x[0]**2*x[1], x[1]**2], x), desired)

    def test_not_an_argument(self):
        with self.assertRaisesRegex(ValueError, 'Expected an instance of `Argument`'):
            function.derivative(function.ones(()), function.zeros(()))

    def test_different_argument_shape(self):
        with self.assertRaisesRegex(ValueError, "Argument 'foo' has shape \\(2,\\) in the function, but the derivative to 'foo' with shape \\(3, 4\\) was requested."):
            function.derivative(function.Argument('foo', (2,)), function.Argument('foo', (3, 4)))

    def test_different_argument_dtype(self):
        with self.assertRaisesRegex(ValueError, "Argument 'foo' has dtype int in the function, but the derivative to 'foo' with dtype float was requested."):
            function.derivative(function.Argument('foo', (), dtype=int), function.Argument('foo', (), dtype=float))


derivative('function',
           normal=function.normal,
           tangent=function.tangent,
           dotnorm=function.dotnorm,
           grad=function.grad,
           div=function.div,
           curl=function.curl,
           laplace=function.laplace,
           symgrad=function.symgrad,
           ngrad=function.ngrad,
           nsymgrad=function.nsymgrad)
derivative('method',
           normal=lambda geom, refgeom=None: function.Array.cast(geom).normal(refgeom),
           tangent=lambda geom, vec: function.Array.cast(geom).tangent(vec),
           dotnorm=lambda vec, geom: function.Array.cast(vec).dotnorm(geom),
           grad=lambda arg, geom: function.Array.cast(arg).grad(geom),
           div=lambda arg, geom, ndims=0: function.Array.cast(arg).div(geom, ndims),
           curl=lambda arg, geom: function.Array.cast(arg).curl(geom),
           laplace=lambda arg, geom: function.Array.cast(arg).laplace(geom),
           symgrad=lambda arg, geom: function.Array.cast(arg).symgrad(geom),
           ngrad=lambda arg, geom: function.Array.cast(arg).ngrad(geom),
           nsymgrad=lambda arg, geom: function.Array.cast(arg).nsymgrad(geom))


class deprecations(TestCase):

    def test_simplified(self):
        with self.assertWarns(warnings.NutilsDeprecationWarning):
            function.simplified(function.Argument('a', ()))

    def test_iszero(self):
        with self.assertWarns(warnings.NutilsDeprecationWarning):
            function.iszero(function.Argument('a', ()))


class CommonBasis:

    @staticmethod
    def mk_index_coords(coorddim, transforms):
        space = 'X'
        index = function.transforms_index(space, transforms)
        coords = function.transforms_coords(space, transforms)
        return index, coords

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

    def test_get_ndofs(self):
        for ielem in range(self.checknelems):
            self.assertEqual(self.basis.get_ndofs(ielem), len(self.checkdofs[ielem]))

    def test_dofs_array(self):
        for mask in itertools.product(*[[False, True]]*self.checknelems):
            mask = numpy.array(mask, dtype=bool)
            indices, = numpy.where(mask)
            for value in mask, indices:
                with self.subTest(tuple(value)):
                    self.assertEqual(sorted(self.basis.get_dofs(value)), sorted(set(itertools.chain.from_iterable(self.checkdofs[i] for i in indices))))

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
        checkmasks = getattr(self, 'checkmasks', itertools.product(*[[False, True]]*self.checkndofs))
        for mask in checkmasks:
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

    def test_getitem_slice(self):
        maskedbasis = self.basis[1:-1]
        indices = numpy.arange(self.checkndofs)[1:-1]
        for ielem in range(self.checknelems):
            m = numpy.asarray(numeric.sorted_contains(indices, self.checkdofs[ielem]))
            self.assertEqual(maskedbasis.get_dofs(ielem).tolist(), numeric.sorted_index(indices, numpy.compress(m, self.checkdofs[ielem], axis=0)).tolist())
            self.assertEqual(maskedbasis.get_coefficients(ielem).tolist(), numpy.compress(m, self.checkcoeffs[ielem], axis=0).tolist())

    def test_getitem_slice_all(self):
        maskedbasis = self.basis[:]
        for ielem in range(self.checknelems):
            self.assertEqual(maskedbasis.get_dofs(ielem).tolist(), self.checkdofs[ielem])
            self.assertEqual(maskedbasis.get_coefficients(ielem).tolist(), self.checkcoeffs[ielem])

    def checkeval(self, ielem, points):
        result = numpy.zeros((points.npoints, self.checkndofs,), dtype=float)
        if self.checkcoeffs[ielem]:
            numpy.add.at(result, (slice(None), numpy.array(self.checkdofs[ielem], dtype=int)), poly.eval_outer(numpy.array(self.checkcoeffs[ielem], dtype=float), points.coords))
        return result

    def test_lower(self):
        ref = element.PointReference() if self.basis.coords.shape[0] == 0 else element.LineReference()**self.basis.coords.shape[0]
        points = ref.getpoints('bezier', 4)
        coordinates = evaluable.constant(points.coords)
        lowerargs = function.LowerArgs.for_space('X', (self.checktransforms,), evaluable.Argument('ielem', (), int), coordinates)
        lowered = self.basis.lower(lowerargs)
        with _builtin_warnings.catch_warnings():
            _builtin_warnings.simplefilter('ignore', category=evaluable.ExpensiveEvaluationWarning)
            for ielem in range(self.checknelems):
                value = lowered.eval(ielem=ielem)
                if value.shape[0] == 1:
                    value = numpy.tile(value, (points.npoints, 1))
                self.assertAllAlmostEqual(value, self.checkeval(ielem, points))


class PlainBasis(CommonBasis, TestCase):

    def setUp(self):
        self.checktransforms = transformseq.IndexTransforms(0, 4)
        index, coords = self.mk_index_coords(0, self.checktransforms)
        self.checkcoeffs = [[[1.]], [[2.], [3.]], [[4.], [5.]], [[6.]]]
        self.checkdofs = [[0], [2, 3], [1, 3], [2]]
        self.basis = function.PlainBasis(self.checkcoeffs, self.checkdofs, 4, index, coords)
        self.checkndofs = 4
        super().setUp()


class DiscontBasis(CommonBasis, TestCase):

    def setUp(self):
        self.checktransforms = transformseq.IndexTransforms(0, 4)
        index, coords = self.mk_index_coords(0, self.checktransforms)
        self.checkcoeffs = [[[1.]], [[2.], [3.]], [[4.], [5.]], [[6.]]]
        self.basis = function.DiscontBasis(self.checkcoeffs, index, coords)
        self.checkdofs = [[0], [1, 2], [3, 4], [5]]
        self.checkndofs = 6
        super().setUp()


class LegendreBasis(CommonBasis, TestCase):

    def setUp(self):
        self.checktransforms = transformseq.IndexTransforms(1, 3)
        index, coords = self.mk_index_coords(0, self.checktransforms)
        self.checkcoeffs = [[[0, 0, 0, 1], [0, 0, 2, -1], [0, 6, -6, 1], [20, -30, 12, -1]]]*3
        self.basis = function.LegendreBasis(3, 3, index, coords)
        self.checkdofs = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]
        self.checkndofs = 12
        self.checkmasks = [[i in [0, 1, 4, 5, 7] for i in range(12)]]
        super().setUp()


class MaskedBasis(CommonBasis, TestCase):

    def setUp(self):
        self.checktransforms = transformseq.IndexTransforms(0, 4)
        index, coords = self.mk_index_coords(0, self.checktransforms)
        parent = function.PlainBasis([[[1.]], [[2.], [3.]], [[4.], [5.]], [[6.]]], [[0], [2, 3], [1, 3], [2]], 4, index, coords)
        self.basis = function.MaskedBasis(parent, [0, 2])
        self.checkcoeffs = [[[1.]], [[2.]], [], [[6.]]]
        self.checkdofs = [[0], [1], [], [1]]
        self.checkndofs = 2
        super().setUp()


class PrunedBasis(CommonBasis, TestCase):

    def setUp(self):
        parent_transforms = transformseq.IndexTransforms(0, 4)
        parent_index, parent_coords = self.mk_index_coords(0, parent_transforms)
        indices = types.frozenarray([0, 2])
        self.checktransforms = parent_transforms[indices]
        index, coords = self.mk_index_coords(0, self.checktransforms)
        parent = function.PlainBasis([[[1.]], [[2.], [3.]], [[4.], [5.]], [[6.]]], [[0], [2, 3], [1, 3], [2]], 4, parent_index, parent_coords)
        self.basis = function.PrunedBasis(parent, indices, index, coords)
        self.checkcoeffs = [[[1.]], [[4.], [5.]]]
        self.checkdofs = [[0], [1, 2]]
        self.checkndofs = 3
        super().setUp()


class StructuredBasis1D(CommonBasis, TestCase):

    def setUp(self):
        self.checktransforms = transformseq.IndexTransforms(1, 4)
        index, coords = self.mk_index_coords(1, self.checktransforms)
        self.basis = function.StructuredBasis([[[[1.], [2.]], [[3.], [4.]], [[5.], [6.]], [[7.], [8.]]]], [[0, 1, 2, 3]], [[2, 3, 4, 5]], [5], [4], index, coords)
        self.checkcoeffs = [[[1.], [2.]], [[3.], [4.]], [[5.], [6.]], [[7.], [8.]]]
        self.checkdofs = [[0, 1], [1, 2], [2, 3], [3, 4]]
        self.checkndofs = 5
        super().setUp()


class StructuredBasis1DPeriodic(CommonBasis, TestCase):

    def setUp(self):
        self.checktransforms = transformseq.IndexTransforms(1, 4)
        index, coords = self.mk_index_coords(1, self.checktransforms)
        self.basis = function.StructuredBasis([[[[1.], [2.]], [[3.], [4.]], [[5.], [6.]], [[7.], [8.]]]], [[0, 1, 2, 3]], [[2, 3, 4, 5]], [4], [4], index, coords)
        self.checkcoeffs = [[[1.], [2.]], [[3.], [4.]], [[5.], [6.]], [[7.], [8.]]]
        self.checkdofs = [[0, 1], [1, 2], [2, 3], [3, 0]]
        self.checkndofs = 4
        super().setUp()


class StructuredBasis2D(CommonBasis, TestCase):

    def setUp(self):
        self.checktransforms = transformseq.IndexTransforms(2, 4)
        index, coords = self.mk_index_coords(2, self.checktransforms)
        self.basis = function.StructuredBasis([[[[1.], [2.]], [[3.], [4.]]], [[[5.], [6.]], [[7.], [8.]]]], [[0, 1], [0, 1]], [[2, 3], [2, 3]], [3, 3], [2, 2], index, coords)
        self.checkcoeffs = [[[5.], [6.], [10.], [12.]], [[7.], [8.], [14.], [16.]], [[15.], [18.], [20.], [24.]], [[21.], [24.], [28.], [32.]]]
        self.checkdofs = [[0, 1, 3, 4], [1, 2, 4, 5], [3, 4, 6, 7], [4, 5, 7, 8]]
        self.checkndofs = 9
        super().setUp()


@parametrize
class SurfaceGradient(TestCase):

    K = 3  # mean curvature

    def setUp(self):
        super().setUp()
        if self.boundary:
            if self.etype == 'cube':
                topo, (x, y, z) = mesh.rectilinear([1, 2, 3])
                self.u = x * y * (2-y) * z * (3-z)
            else:
                topo, (x, y) = mesh.unitsquare(nelems=2, etype=self.etype)
                self.u = x * y * (1-y)
            self.manifold = topo.boundary['right']
            refgeom = None
        else:
            if self.etype == 'line':
                self.manifold, y = mesh.line(2)
                self.u = y * (2-y)
                refgeom = numpy.stack([y])
            else:
                self.manifold, (y, z) = mesh.unitsquare(nelems=2, etype=self.etype)
                self.u = y * (1-y) * z * (1-z)
                refgeom = numpy.stack([y, z])
            x = 1
        # geometry describes a circle/sphere with curvature K
        self.geom = (x/self.K) * numpy.stack(
            (numpy.cos(y), numpy.sin(y)) if self.manifold.ndims == 1
            else (numpy.cos(y), numpy.sin(y) * numpy.cos(z), numpy.sin(y) * numpy.sin(z)))
        self.normal = function.normal(self.geom, refgeom=refgeom)

    @property
    def P(self):
        n = len(self.normal)
        return function.eye(n) - self.normal[:, numpy.newaxis] * self.normal

    def test_grad_u(self):
        grad = function.surfgrad(self.u, self.geom)
        if self.boundary:  # test the entire vector
            expect = (self.P * function.grad(self.u, self.geom)).sum(-1)
            self.assertAllAlmostEqual(*self.manifold.sample('uniform', 2).eval([grad, expect]))
        else:  # test that vector is tangent to the manifold
            ngrad = (grad * self.normal).sum(-1)
            self.assertAllAlmostEqual(*self.manifold.sample('uniform', 2).eval([ngrad, 0]))

    def test_grad_x(self):
        P = function.surfgrad(self.geom, self.geom)
        self.assertAllAlmostEqual(*self.manifold.sample('uniform', 2).eval([P, self.P]))

    def test_div_n(self):
        # https://en.wikipedia.org/wiki/Mean_curvature#Surfaces_in_3D_space
        K = function.div(self.normal, self.geom, -1) / self.manifold.ndims
        self.assertAllAlmostEqual(*self.manifold.sample('uniform', 2).eval([K, self.K]))

    def test_stokes(self):
        # https://en.wikipedia.org/wiki/Laplace%E2%80%93Beltrami_operator#Formal_self-adjointness
        grad = function.surfgrad(self.u, self.geom)
        lapl = function.laplace(self.u, self.geom, -1)
        J = function.J(self.geom)
        self.assertAlmostEqual(*self.manifold.integrate([(grad @ grad) * J, -self.u * lapl * J], degree=9))


SurfaceGradient(boundary=False, etype='line')
SurfaceGradient(boundary=False, etype='square')
SurfaceGradient(boundary=False, etype='triangle')
SurfaceGradient(boundary=True, etype='square')
SurfaceGradient(boundary=True, etype='triangle')
SurfaceGradient(boundary=True, etype='cube')


class Eval(TestCase):

    def test_single(self):
        f = function.dotarg('v', numpy.array([1, 2, 3]))
        retval = function.eval(f, v=numpy.array([4, 5, 6]))
        self.assertEqual(retval, 4+10+18)

    def test_multiple(self):
        f = function.dotarg('v', numpy.array([1, 2, 3]))
        g = function.dotarg('v', numpy.array([3, 2, 1]))
        retvals = function.eval([f, g], v=numpy.array([4, 5, 6]))
        self.assertEqual(retvals, (4+10+18, 12+10+6))


class linearize(TestCase):

    def test(self):
        f = function.linearize(function.Argument('u', shape=(3, 4), dtype=float)**3
                             + function.Argument('p', shape=(), dtype=float), 'u:v,p:q')
        # test linearization of u**3 + p -> 3 u**2 v + q through evaluation
        _u = numpy.arange(3, dtype=float)[:,numpy.newaxis].repeat(4, 1)
        _v = numpy.arange(4, dtype=float)[numpy.newaxis,:].repeat(3, 0)
        _q = 5.
        self.assertAllEqual(f.eval(u=_u, v=_v, q=_q).export('dense'), 3 * _u**2 * _v + _q)


class attributes(TestCase):

    def test(self):
        A = function.Argument('test', (2,3))
        self.assertEqual(numpy.shape(A), (2,3))
        self.assertEqual(numpy.size(A), 6)
        self.assertEqual(numpy.ndim(A), 2)
