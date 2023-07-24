from nutils import numeric, _util as util, types
from nutils.testing import TestCase, parametrize
import numpy
import itertools


@parametrize
class pack(TestCase):

    def setUp(self):
        super().setUp()
        assert self.nbits in (8, 16, 32)
        self.dtype = numpy.dtype('int{}'.format(self.nbits))
        self.nnan, self.nnil, self.nmin, self.nmax, self.ninf = n = numpy.array(
            [-128, 0, 1, 126, 127] if self.nbits == 8 else
            [-32768, 0, 1, 32766, 32767] if self.nbits == 16 else
            [-2147483648, 0, 1, 2147483646, 2147483647], dtype=self.dtype)
        self.amin, self.amax, self.aclip = numpy.sinh(n[2:]*self.rtol)*(self.atol/self.rtol)

    def pack(self, a):
        return numeric.pack(a, atol=self.atol, rtol=self.rtol, dtype=self.dtype)

    def unpack(self, n):
        return numeric.unpack(n, atol=self.atol, rtol=self.rtol)

    def test_decode(self):
        self.assertTrue(numpy.isnan(self.unpack(self.nnan)))
        self.assertEqual(self.unpack(-self.ninf), -numpy.inf)
        self.assertEqual(self.unpack(-self.nmax), -self.amax)
        self.assertEqual(self.unpack(-self.nmin), -self.amin)
        self.assertEqual(self.unpack(self.nnil), 0)
        self.assertEqual(self.unpack(self.nmin), self.amin)
        self.assertEqual(self.unpack(self.nmax), self.amax)
        self.assertEqual(self.unpack(self.ninf), numpy.inf)

    def test_encode(self):
        self.assertEqual(self.pack(numpy.nan), self.nnan)
        self.assertEqual(self.pack(-numpy.inf), -self.ninf)
        self.assertEqual(self.pack(-self.amax), -self.nmax)
        self.assertEqual(self.pack(-self.amin), -self.nmin)
        self.assertEqual(self.pack(0), self.nnil)
        self.assertEqual(self.pack(self.amin), self.nmin)
        self.assertEqual(self.pack(self.amax), self.nmax)
        self.assertEqual(self.pack(numpy.inf), self.ninf)

    def test_clip(self):
        with self.assertWarns(RuntimeWarning):
            self.assertEqual(self.pack(-self.aclip), -self.ninf)
        with self.assertWarns(RuntimeWarning):
            self.assertEqual(self.pack(self.aclip), self.ninf)

    def test_round(self):
        b01 = numpy.sinh(0.5*self.rtol)*(self.atol/self.rtol)
        b12 = numpy.sinh(1.5*self.rtol)*(self.atol/self.rtol)
        a = -b12*1.001, -b12*.999, -b01*1.001, -b01*.999, b01*.999, b01*1.001, b12*.999, b12*1.001
        n = self.pack(a)
        self.assertEqual(tuple(n), (-2, -1, -1, 0, 0, 1, 1, 2))

    def test_spacing(self):
        for a in -1., 0., 1.:
            n = self.pack(a)
            da = numpy.sqrt(self.atol**2 + (a*self.rtol)**2)
            self.assertLess(self.unpack(n-1), a - da/2)
            self.assertGreater(self.unpack(n), a - da/2)
            self.assertLess(self.unpack(n), a + da/2)
            self.assertGreater(self.unpack(n+1), a + da/2)


pack('int8', atol=2e-6, rtol=2e-1, nbits=8)
pack('int16', atol=2e-15, rtol=2e-3, nbits=16)
pack('int32', atol=2e-96, rtol=2e-7, nbits=32)


class sorted_index(TestCase):

    def test_None(self):
        for a, b in ([], []), ([1], []), ([1, 2], [2, 1]):
            self.assertEqual(numeric.sorted_index(numpy.array(a, int), b).tolist(), [a.index(v) for v in b if v in a])

    def test_None_exception(self):
        for a, b in ([], [1]), ([1], [2]), ([1, 2], [3, 1]):
            with self.assertRaises(ValueError):
                numeric.sorted_index(numpy.array(a, int), b)

    def test_int(self):
        for a, b in ([], []), ([1], []), ([1, 2], [2, 1]), ([], [1]), ([1], [2]), ([1, 2], [3, 1]):
            self.assertEqual(numeric.sorted_index(numpy.array(a, int), b, missing=-1).tolist(), [a.index(v) if v in a else -1 for v in b])

    def test_mask(self):
        for a, b in ([], []), ([1], []), ([1, 2], [2, 1]), ([], [1]), ([1], [2]), ([1, 2], [3, 1]):
            self.assertEqual(numeric.sorted_index(numpy.array(a, int), b, missing='mask').tolist(), [a.index(v) for v in b if v in a])

    def test_invalid(self):
        with self.assertRaises(ValueError):
            numeric.sorted_index(numpy.array([1], int), [1], missing='foo')


class sorted_contains(TestCase):

    def test(self):
        for a, b in ([], []), ([1], []), ([1, 2], [2, 1]), ([], [1]), ([1], [2]), ([1, 2], [3, 1]):
            self.assertEqual(numeric.sorted_contains(numpy.array(a, int), b).tolist(), [v in a for v in b])


class asboolean(TestCase):

    def test_bool(self):
        self.assertAllEqual(numeric.asboolean([True, False], 2), [True, False])

    def test_int(self):
        self.assertAllEqual(numeric.asboolean([1], 2), [False, True])
        self.assertAllEqual(numeric.asboolean([0], 2), [True, False])

    def test_none(self):
        self.assertAllEqual(numeric.asboolean(None, 2), [False, False])
        self.assertAllEqual(numeric.asboolean([], 2), [False, False])
        self.assertAllEqual(numeric.asboolean((), 2), [False, False])

    def test_float(self):
        with self.assertRaises(Exception):
            numeric.asboolean([1.5, 2.5], 2)

    def test_wrongsize(self):
        with self.assertRaises(Exception):
            numeric.asboolean([True, False, True], 2)

    def test_wrongdimension(self):
        with self.assertRaises(Exception):
            numeric.asboolean([[True, False, True], [False, True, True]], 2)

    def test_outofbounds(self):
        with self.assertRaises(Exception):
            numeric.asboolean([-1], 2)
        with self.assertRaises(Exception):
            numeric.asboolean([2], 2)

    def test_unordered(self):
        self.assertAllEqual(numeric.asboolean([2, 1], 3, ordered=False), [False, True, True])
        with self.assertRaises(Exception):
            numeric.asboolean([2, 1], 3)


class invmap(TestCase):

    def test(self):
        indices = numpy.array([3, 1])
        invmap = numeric.invmap(indices, length=5, missing=9)
        self.assertAllEqual(invmap, [9, 1, 9, 0, 9])


class istype(TestCase):

    def test_isint(self):
        self.assertTrue(numeric.isint(1))
        self.assertFalse(numeric.isint(numpy.array(1)))
        self.assertTrue(numeric.isint(numpy.int32(1)))
        self.assertTrue(numeric.isint(numpy.uint32(1)))
        self.assertFalse(numeric.isint(1.5))
        self.assertFalse(numeric.isint(numpy.array([1])))

    def test_isbool(self):
        self.assertTrue(numeric.isbool(True))
        self.assertTrue(numeric.isbool(numpy.bool_(True)))
        self.assertTrue(numeric.isbool(numpy.array(True)))
        self.assertFalse(numeric.isbool(numpy.array([True])))

    def test_isnumber(self):
        self.assertTrue(numeric.isnumber(1))
        self.assertFalse(numeric.isnumber(numpy.array(1)))
        self.assertTrue(numeric.isnumber(numpy.int32(1)))
        self.assertTrue(numeric.isnumber(numpy.uint32(1)))
        self.assertTrue(numeric.isnumber(1.5))
        self.assertTrue(numeric.isnumber(numpy.float64(1.5)))
        self.assertFalse(numeric.isnumber(numpy.array(1.5)))
        self.assertFalse(numeric.isnumber(numpy.array([1])))

    def test_isarray(self):
        self.assertTrue(numeric.isarray(numpy.array([1, 2, 3])))
        self.assertTrue(numeric.isarray(types.frozenarray([1, 2, 3])))
        self.assertTrue(numeric.isarray(numpy.array(1)))

    def test_isboolarray(self):
        self.assertTrue(numeric.isboolarray(numpy.array(True)))
        self.assertTrue(numeric.isboolarray(numpy.array([True])))
        self.assertTrue(numeric.isboolarray(types.frozenarray([True])))
        self.assertFalse(numeric.isboolarray(numpy.array([1])))
        self.assertFalse(numeric.isboolarray(True))

    def test_isboolarray(self):
        self.assertTrue(numeric.isintarray(numpy.array(1)))
        self.assertTrue(numeric.isintarray(numpy.array([1])))
        self.assertTrue(numeric.isintarray(types.frozenarray([1])))
        self.assertFalse(numeric.isintarray(numpy.array([1.5])))
        self.assertFalse(numeric.isintarray(1.5))


class levicivita(TestCase):

    def test_1d(self):
        with self.assertRaisesRegex(ValueError, '^The Levi-Civita symbol is undefined for dimensions lower than 2.'):
            numeric.levicivita(1)

    def test_2d(self):
        self.assertAllEqual(numeric.levicivita(2, int), numpy.array([[0, 1], [-1, 0]]))

    def test_nd(self):
        sign = lambda v: -1 if v < 0 else 1 if v > 0 else 0
        for n in range(2, 6):
            with self.subTest(n=n):
                desired = numpy.empty((n,)*n, int)
                for I in itertools.product(*[range(n)]*n):
                    desired[I] = util.product(sign(b-a) for a, b in itertools.combinations(I, 2))
                self.assertAllEqual(numeric.levicivita(n, int), desired)


class meshgrid(TestCase):

    def test_unary(self):
        m = numeric.meshgrid([1, 2, 3])
        self.assertEqual(m.dtype, int)
        self.assertAllEqual(m, [[1, 2, 3]])

    def test_binary(self):
        m = numeric.meshgrid([1, 2, 3], [.4, .5])
        self.assertEqual(m.dtype, float)
        self.assertAllEqual(m, [[[1, 1], [2, 2], [3, 3]], [[.4, .5], [.4, .5], [.4, .5]]])

    def test_ternary(self):
        m = numeric.meshgrid([1, 2, 3], 1j, [.4, .5])
        self.assertEqual(m.dtype, complex)
        self.assertAllEqual(m, [[[1, 1], [2, 2], [3, 3]], [[1j, 1j], [1j, 1j], [1j, 1j]], [[.4, .5], [.4, .5], [.4, .5]]])

    def test_dtype(self):
        m = numeric.meshgrid(1, dtype=float)
        self.assertEqual(m.dtype, float)
        self.assertAllEqual(m, numpy.ones((1,)))


class simplex_grid(TestCase):

    def simplex_grid(self, shape, spacing):
        coords = numeric.simplex_grid(shape, spacing)
        self.assertEqual(coords.ndim, 2)
        self.assertEqual(coords.shape[1], len(shape))
        self.assertTrue((coords > 0).all())
        self.assertTrue((coords < shape).all())
        mindist = min(numpy.linalg.norm(c1 - c2) for i, c1 in enumerate(coords) for c2 in coords[:i])
        self.assertAlmostEqual(mindist, spacing)
        return coords

    def test_1d(self):
        coords = self.simplex_grid([2], .8)
        self.assertEqual(len(coords), 3)
        self.assertAllAlmostEqual(coords[:, 0], [.2, 1, 1.8])

    def test_2d(self):
        coords = self.simplex_grid([2, 3], .8)
        self.assertEqual(len(coords), 13)

    def test_3d(self):
        coords = self.simplex_grid([2, 3, 4], .8)
        self.assertEqual(len(coords), 82)


class overlapping(TestCase):

    def test_pairwise(self):
        a = numpy.arange(4)
        b = numeric.overlapping(a)
        self.assertIs(b.base, a)
        self.assertEqual(b.shape, (3, 2))
        self.assertEqual(b.tolist(), [[0, 1], [1, 2], [2, 3]])

    def test_multidim(self):
        a = numpy.arange(.5, 16).reshape(2, 4, 2)
        b = numeric.overlapping(a, axis=1, n=3)
        self.assertIs(b.base, a.base)
        self.assertEqual(b.shape, (2, 2, 3, 2))
        for i in range(2):
            for j in range(2):
                self.assertAllEqual(b[i, ..., j], a[i, 0, j]+numpy.array([[0, 2, 4], [2, 4, 6]]))


class full(TestCase):

    def test(self):
        f = numeric.full((2, 3), fill_value=1, dtype=int)
        self.assertIsNone(f.base)
        self.assertFalse(f.flags.writeable)
        self.assertEqual(f.dtype, int)
        self.assertEqual(f.strides, (0, 0))
        self.assertAllEqual(f, numpy.ones((2, 3)))


class sinc(TestCase):

    def check(self, n, psin, pcos, f0):
        x = numpy.arange(-4, 5) / 4
        nz = x != 0
        (z,), = (~nz).nonzero()
        f = numpy.polyval(psin, x) * numpy.sin(x) + numpy.polyval(pcos, x) * numpy.cos(x)
        f[nz] /= x[nz]**(n+1)
        f[z] = f0
        self.assertAllAlmostEqual(numeric.sinc(x, n), f)

    def test_f0(self):
        self.check(0, [1], [], 1)

    def test_f1(self):
        self.check(1, [-1], [1,0], 0)

    def test_f2(self):
        self.check(2, [-1,0,2], [-2,0], -1/3)

    def test_f3(self):
        self.check(3, [3,0,-6], [-1,0,6,0], 0)

    def test_f4(self):
        self.check(4, [1,0,-12,0,24], [4,0,-24,0], 1/5)

    def test_scalar(self):
        self.assertEqual(numeric.sinc(0), 1)
        self.assertEqual(numeric.sinc(numpy.pi/2), 2/numpy.pi)
        self.assertEqual(numeric.sinc(0, 1), 0)
        self.assertAlmostEqual(numeric.sinc(numpy.pi/2, 1), -4/numpy.pi**2, delta=1e-16)

    def test_errors(self):
        for n in range(5):
            f = numeric.sinc(numpy.linspace(-1e-12, 1e-12, 10), n)
            self.assertTrue(all(abs(numpy.diff(f)) < 1e-13))


class sanitize_einsum_subscripts(TestCase):

    def sanitize_einsum_subscripts(self, subscripts, *shapes):
        args = [numpy.arange(numpy.prod(shape, dtype=int)).reshape(shape) for shape in shapes]
        try:
            ret_orig = numpy.einsum(subscripts, *args)
        except:
            ret_orig = None
        try:
            sanitized = numeric.sanitize_einsum_subscripts(subscripts, *shapes)
        except Exception as e:
            if ret_orig is not None:
                self.fail(f'subscript is valid, but sanitize_einsum_subscripts failed with error "{e}"')
            raise
        else:
            if ret_orig is None:
                self.fail(f'subscript is invalid, but sanitize_einsum_subscripts found no problem')
            ret_parsed = numpy.einsum(','.join(sanitized[:-1]) + '->' + sanitized[-1], *args)
            self.assertAllEqual(ret_orig, ret_parsed)
            return sanitized

    def test_valid(self):
        with self.subTest('matmat-explicit'):
            self.assertEqual(self.sanitize_einsum_subscripts('ij,jk->ik', (2,3), (3,4)), ('ij', 'jk', 'ik'))
        with self.subTest('matmat-implicit'):
            self.assertEqual(self.sanitize_einsum_subscripts('ij,jk', (2,3), (3,4)), ('ij', 'jk', 'ik'))
        with self.subTest('matmat-ellipses'):
            self.assertEqual(self.sanitize_einsum_subscripts('...ij,...jk->...ik', (5,2,3), (5,3,4)), ('Aij', 'Ajk', 'Aik'))
        with self.subTest('transpose'):
            self.assertEqual(self.sanitize_einsum_subscripts('...ji', (5,2,3)), ('Aji', 'Aij'))
        with self.subTest('trace'):
            self.assertEqual(self.sanitize_einsum_subscripts('i...i', (3,2,4,3)), ('iBAi', 'BA'))
        with self.subTest('diag'):
            self.assertEqual(self.sanitize_einsum_subscripts('i...i->...i', (3,2,4,3)), ('iBAi', 'BAi'))
        with self.subTest('diag3'):
            self.assertEqual(self.sanitize_einsum_subscripts('i...ii->i...', (3,2,3,3)), ('iAii', 'iA'))
        with self.subTest('sum'):
            self.assertEqual(self.sanitize_einsum_subscripts('ij->...', (1,2)), ('ij', ''))
        with self.subTest('combined'):
            self.assertEqual(self.sanitize_einsum_subscripts('...ij,j,...i->...i', (5,2,3), (3,), (5,2)), ('Aij', 'j', 'Ai', 'Ai'))
        with self.subTest('broadcast'):
            self.assertEqual(self.sanitize_einsum_subscripts('...ij,k...->...', (2,2,3), (4,5,1)), ('Aij', 'kBA', 'BA'))

    def test_invalid(self):
        with self.assertRaisesRegex(ValueError, 'first einsum argument must be a string of subscript labels'):
            self.sanitize_einsum_subscripts(123)
        with self.assertRaisesRegex(ValueError, 'non-empty ellipses in input require ellipsis in output'):
            self.sanitize_einsum_subscripts('...ij->ij', (1,2,3))
        with self.assertRaisesRegex(ValueError, 'number of arguments does not match subscript labels'):
            self.sanitize_einsum_subscripts('ij, jk', (1,2))
        with self.assertRaisesRegex(ValueError, 'argument shapes are inconsistent with subscript labels'):
            self.sanitize_einsum_subscripts('...i,...i->...', (2,3), (4,5,3))
        with self.assertRaisesRegex(ValueError, 'invalid subscripts argument'):
            self.sanitize_einsum_subscripts('i$->i', (2,3))
        with self.assertRaisesRegex(ValueError, 'argument dimensions are inconsistent with subscript labels'):
            self.sanitize_einsum_subscripts('ji->ij', (2,3,4))
        with self.assertRaisesRegex(ValueError, 'argument shapes are inconsistent with subscript labels'):
            self.sanitize_einsum_subscripts('...i,...i->...', (2,3), (4,5))
        with self.assertRaisesRegex(ValueError, 'argument dimensions are inconsistent with subscript labels'):
            self.sanitize_einsum_subscripts('i...j->...', (2,))
        with self.assertRaisesRegex(ValueError, "einstein sum subscripts string included output subscript 'j' which never appeared in an input"):
            self.sanitize_einsum_subscripts('i->j', (2,))
