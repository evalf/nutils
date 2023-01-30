from nutils import *
from nutils.testing import *
import tempfile
import pathlib
import os
import io
import contextlib
import inspect


@parametrize
class tri(TestCase):

    # Triangles and node numbering:
    #
    #   2/4-(5)
    #    | \ |
    #   (0)-1/3

    def setUp(self):
        super().setUp()
        self.x = numpy.array([[0, 0], [1, 0], [0, 1], [1, 0], [0, 1], [1, 1]], dtype=float)
        self.tri = numpy.array([[0, 1, 2], [3, 4, 5]])

    @requires('scipy')
    def test_merge(self):
        tri_merged = util.tri_merge(self.tri, self.x, mergetol=self.mergetol).tolist()
        tri_expected = self.tri.tolist() if self.mergetol < 0 else [[0, 1, 2], [1, 2, 5]] if self.mergetol < 1 else [[0, 0, 0], [0, 0, 0]]
        self.assertEqual(tri_merged, tri_expected)

    @requires('matplotlib', 'scipy')
    def test_interpolate(self):
        interpolate = util.tri_interpolator(self.tri, self.x, mergetol=self.mergetol)
        x = [.1, .9],
        if self.mergetol < 0:
            with self.assertRaises(RuntimeError):
                interpolate[x]
        else:
            f = interpolate[x]
            vtri = [0, 0], [1, 0], [0, 1], [10, 10], [10, 10], [1, 1]
            vx = f(vtri)  # note: v[3] and v[4] should be ignored, leaving a linear ramp
            self.assertEqual(vx.shape, (1, 2))
            if self.mergetol < 1:
                self.assertEqual(vx.tolist(), list(x))
            else:
                self.assertTrue(numpy.isnan(vx).all())

    @parametrize.enable_if(lambda mergetol: 0 <= mergetol < 1)
    @requires('matplotlib', 'scipy')
    def test_outofbounds(self):
        interpolate = util.tri_interpolator(self.tri, self.x, mergetol=self.mergetol)
        x = [.5, .5], [1.5, .5]
        vtri = 0, 1, 0, 10, 10, 1
        vx = interpolate[x](vtri)
        self.assertEqual(vx.shape, (2,))
        self.assertEqual(vx[0], .5)
        self.assertTrue(numpy.isnan(vx[1]))


tri(mergetol=-1)
tri(mergetol=0)
tri(mergetol=.1)
tri(mergetol=2)


class linreg(TestCase):

    def test_linear(self):
        a = numpy.array([[0, 1], [-1, 0]])
        b = numpy.array([[0, 1], [0, 1]])
        linreg = util.linear_regressor()
        ab0, ab1, ab2 = [linreg.add(x, a * x + b) for x in range(3)]
        self.assertTrue(numpy.isnan(ab0).all())
        self.assertEqual([a.tolist(), b.tolist()], ab1.tolist())
        self.assertEqual([a.tolist(), b.tolist()], ab2.tolist())


class pairwise(TestCase):

    def test_normal(self):
        for n in range(5):
            with self.subTest(length=n):
                self.assertEqual(list(util.pairwise(range(n))), list(zip(range(n-1), range(1, n))))

    def test_periodic(self):
        self.assertEqual(list(util.pairwise((), periodic=True)), [])
        for n in range(1, 5):
            with self.subTest(length=n):
                self.assertEqual(list(util.pairwise(range(n), periodic=True)), [*zip(range(n-1), range(1, n)), (n-1, 0)])


class readtext(TestCase):

    def _test(self, method):
        try:
            with tempfile.NamedTemporaryFile('w', delete=False) as f:
                f.write('foobar')
            self.assertEqual(util.readtext(method(f.name)), 'foobar')
        finally:  # this instead of simply setting delete=True is required for windows
            os.remove(str(f.name))

    def test_str(self):
        self._test(str)

    def test_path(self):
        self._test(pathlib.Path)

    def test_file(self):
        self.assertEqual(util.readtext(io.StringIO('foobar')), 'foobar')

    def test_typeerror(self):
        with self.assertRaises(TypeError):
            util.readtext(None)


class binaryfile(TestCase):

    def setUp(self):
        super().setUp()
        fid, self.path = tempfile.mkstemp()
        self.addCleanup(os.unlink, self.path)
        os.write(fid, b'foobar')
        os.close(fid)

    def test_str(self):
        with util.binaryfile(self.path) as f:
            self.assertEqual(f.read(), b'foobar')

    def test_path(self):
        with util.binaryfile(pathlib.Path(self.path)) as f:
            self.assertEqual(f.read(), b'foobar')

    def test_file(self):
        with open(self.path, 'rb') as F, util.binaryfile(F) as f:
            self.assertEqual(f.read(), b'foobar')

    def test_typeerror(self):
        with self.assertRaises(TypeError):
            util.binaryfile(None)


class single_or_multiple(TestCase):

    def test_function(self):
        @util.single_or_multiple
        def square(values):
            self.assertIsInstance(values, tuple)
            return [value**2 for value in values]
        self.assertEqual(square(2), 4)
        self.assertEqual(square([2, 3]), (4, 9))

    def test_method(self):
        class T:
            @util.single_or_multiple
            def square(self_, values):
                self.assertIsInstance(self_, T)
                self.assertIsInstance(values, tuple)
                return [value**2 for value in values]
        t = T()
        self.assertEqual(t.square(2), 4)
        self.assertEqual(t.square([2, 3]), (4, 9))


class positional_only(TestCase):

    def test_simple(self):
        @util.positional_only
        def f(x):
            return x
        self.assertEqual(f(1), 1)
        self.assertEqual(str(inspect.signature(f)), '(x, /)')

    def test_mixed(self):
        @util.positional_only
        def f(x, *, y):
            return x, y
        self.assertEqual(f(1, y=2), (1, 2))
        self.assertEqual(str(inspect.signature(f)), '(x, /, *, y)')

    def test_varkw(self):
        @util.positional_only
        def f(x, y=...):
            return x, y
        self.assertEqual(f(1, x=2, y=3), (1, {'x': 2, 'y': 3}))
        self.assertEqual(str(inspect.signature(f)), '(x, /, **y)')

    def test_simple_method(self):
        class T:
            @util.positional_only
            def f(self_, x):
                self.assertIsInstance(self_, T)
                return x
        t = T()
        self.assertEqual(t.f(1), 1)
        self.assertEqual(str(inspect.signature(T.f)), '(self_, x, /)')
        self.assertEqual(str(inspect.signature(t.f)), '(x, /)')


class index(TestCase):

    def _check(self, items):
        for t in list, tuple, iter:
            for i in range(2):
                with self.subTest('{}:{}'.format(t.__name__, i)):
                    self.assertEqual(util.index(t(items), items[i]), i)

    def test_int(self):
        self._check([1, 2, 3, 2, 1])

    def test_set(self):
        self._check([{1, 2}, {2, 3}, {3, 4}, {2, 3}, {1, 2}])


class unique(TestCase):

    def test_nokey(self):
        unique, indices = util.unique([1, 2, 3, 2])
        self.assertEqual(unique, [1, 2, 3])
        self.assertEqual(indices, [0, 1, 2, 1])

    def test_key(self):
        unique, indices = util.unique([[1, 2], [2, 3], [2, 1]], key=frozenset)
        self.assertEqual(unique, [[1, 2], [2, 3]])
        self.assertEqual(indices, [0, 1, 0])


class cached_property(TestCase):

    def test(self):
        class A:
            def __init__(self):
                self.counter = 0

            @util.cached_property
            def x(self):
                self.counter += 1
                return 'x'
        a = A()
        self.assertEqual(a.x, 'x')
        self.assertEqual(a.x, 'x')
        self.assertEqual(a.counter, 1)


class merge_index_map(TestCase):

    def test_empty_merge_sets(self):
        index_map, count = util.merge_index_map(4, [])
        self.assertEqual(index_map.tolist(), [0, 1, 2, 3])
        self.assertEqual(count, 4)

    def test_merge_set_one(self):
        index_map, count = util.merge_index_map(4, [[1]])
        self.assertEqual(index_map.tolist(), [0, 1, 2, 3])
        self.assertEqual(count, 4)

    def test_multihop1(self):
        index_map, count = util.merge_index_map(4, [[0, 2], [1, 3], [0, 3]])
        self.assertEqual(index_map.tolist(), [0, 0, 0, 0])
        self.assertEqual(count, 1)

    def test_multihop2(self):
        index_map, count = util.merge_index_map(4, [[2, 3], [1, 2], [0, 3]])
        self.assertEqual(index_map.tolist(), [0, 0, 0, 0])
        self.assertEqual(count, 1)
