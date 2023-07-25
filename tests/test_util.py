from nutils import _util as util, warnings
from nutils.testing import TestCase, parametrize, requires
import tempfile
import pathlib
import os
import io
import contextlib
import inspect
import treelog
import datetime
import numpy
import sys
import contextlib


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


class gather(TestCase):

    def test(self):
        items = ('z',1), ('a', 2), ('a', 3), ('z', 4), ('b', 5)
        self.assertEqual(list(util.gather(items)), [('z', [1,4]), ('a', [2,3]), ('b', [5])])


class set_current(TestCase):

    def test(self):

        @util.set_current
        def f(x=1):
            return x

        self.assertEqual(f.current, 1)
        with f(2):
            self.assertEqual(f.current, 2)
        self.assertEqual(f.current, 1)


class defaults_from_env(TestCase):

    def setUp(self):
        self.old = os.environ.pop('NUTILS_TEST_ARG', None)

    def tearDown(self):
        if self.old:
            os.environ['NUTILS_TEST_ARG'] = self.old
        else:
            os.environ.pop('NUTILS_TEST_ARG', None)

    def check_retvals(self, expect):
        @util.defaults_from_env
        def f(test_arg: int = 1):
            return test_arg
        self.assertEqual(f(-1), -1)
        self.assertEqual(f(), expect)

    def test_no_env(self):
        self.check_retvals(1)

    def test_valid_env(self):
        os.environ['NUTILS_TEST_ARG'] = '2'
        self.check_retvals(2)

    def test_invalid_env(self):
        os.environ['NUTILS_TEST_ARG'] = 'x'
        with self.assertWarns(warnings.NutilsWarning):
            self.check_retvals(1)


class time(TestCase):

    def assertFormatEqual(self, seconds, formatted):
        self.assertEqual(util.format_timedelta(datetime.timedelta(seconds=seconds)), formatted)

    def test_timedelta(self):
        self.assertFormatEqual(0, '0:00')
        self.assertFormatEqual(1, '0:01')
        self.assertFormatEqual(59, '0:59')
        self.assertFormatEqual(60, '1:00')
        self.assertFormatEqual(3599, '59:59')
        self.assertFormatEqual(3600, '1:00:00')

    def test_timeit(self):
        with self.assertLogs('nutils') as cm, util.timeit():
            treelog.error('test')
        self.assertEqual(len(cm.output), 3)
        self.assertEqual(cm.output[0][:17], 'INFO:nutils:start')
        self.assertEqual(cm.output[1], 'ERROR:nutils:test')
        self.assertEqual(cm.output[2][:18], 'INFO:nutils:finish')

    def test_timer(self):
        self.assertEqual(str(util.timer()), '0:00')


class in_context(TestCase):

    def test(self):

        x_value = None

        @contextlib.contextmanager
        def c(x: int):
            nonlocal x_value
            x_value = x
            yield

        @util.in_context(c)
        def f(s: str):
            return s

        retval = f('test', x=10)

        self.assertEqual(retval, 'test')
        self.assertEqual(x_value, 10)


class log_arguments(TestCase):

    def test(self):

        @util.log_arguments
        def f(foo, bar):
            pass

        with self.assertLogs('nutils') as cm:
            f('x', 10)

        self.assertEqual(cm.output, ['INFO:nutils:arguments > foo=x', 'INFO:nutils:arguments > bar=10'])


class log_traceback(TestCase):

    def test(self):
        with self.assertRaises(SystemExit), self.assertLogs('nutils') as cm, util.log_traceback(gracefulexit=True):
            1/0
        self.assertEqual(cm.output, ['ERROR:nutils:ZeroDivisionError: division by zero'])

    def test_cause(self):
        with self.assertRaises(SystemExit), self.assertLogs('nutils') as cm, util.log_traceback(gracefulexit=True):
            try:
                1/0
            except Exception as e:
                raise RuntimeError('something went wrong') from e
        self.assertEqual(cm.output, ['ERROR:nutils:RuntimeError: something went wrong',
            'ERROR:nutils:.. caused by ZeroDivisionError: division by zero'])

    def test_context(self):
        with self.assertRaises(SystemExit), self.assertLogs('nutils') as cm, util.log_traceback(gracefulexit=True):
            try:
                1/0
            except Exception:
                raise RuntimeError('something went wrong')
        self.assertEqual(cm.output, ['ERROR:nutils:RuntimeError: something went wrong',
            'ERROR:nutils:.. while handling ZeroDivisionError: division by zero'])

    def test_nograce(self):
        with self.assertRaises(ZeroDivisionError), util.log_traceback(gracefulexit=False):
            1/0


class signal_handler(TestCase):

    def test(self):

        try:
            from signal import SIGABRT, raise_signal
        except ImportError:
            raise self.skipTest('test is not possible on this platform')

        caught = False
        def f(sig, frame):
            nonlocal caught
            caught = True

        with util.signal_handler('SIGABRT', f):
            raise_signal(SIGABRT)

        self.assertTrue(caught)


class name_of_main(TestCase):

    def setUp(self):
        self.__main__ = sys.modules['__main__']

    def tearDown(self):
        sys.modules['__main__'] = self.__main__

    def test_package(self):
        class test_main:
            __package__ = 'foo.bar'
            __file__ = '/path/to/foo/bar.py'
        sys.modules['__main__'] = test_main
        self.assertEqual(util.name_of_main(), 'foo.bar')

    def test_file(self):
        class test_main:
            __file__ = '/path/to/foo/bar.py'
        sys.modules['__main__'] = test_main
        self.assertEqual(util.name_of_main(), 'bar')

    def test_interactive(self):
        class test_main:
            pass
        sys.modules['__main__'] = test_main
        self.assertEqual(util.name_of_main(), 'interactive')


class add_htmllog(TestCase):

    def test(self):

        with tempfile.TemporaryDirectory() as outdir:
            with util.add_htmllog(outdir=outdir):
                treelog.info('hi there')
            self.assertTrue(os.path.isfile(os.path.join(outdir, 'log.html')))


class deep_reduce(TestCase):

    def test(self):
        L = 1, 2, [3, (4, 5), 6], 7
        self.assertEqual(util.deep_reduce(min, L), 1)
        self.assertEqual(util.deep_reduce(max, L), 7)
        self.assertEqual(util.deep_reduce(sum, L), 28)


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


class nutils_dispatch(TestCase):

    class Ten:
        @classmethod
        def __nutils_dispatch__(cls, func, args, kwargs):
            return func(*[10 if isinstance(v, cls) else v for v in args], **kwargs)

    class Twenty:
        @classmethod
        def __nutils_dispatch__(cls, func, args, kwargs):
            return func(*[20 if isinstance(v, cls) else v for v in args], **kwargs)

    class NotImp:
        @classmethod
        def __nutils_dispatch__(cls, func, args, kwargs):
            return NotImplemented

    @staticmethod
    @util.nutils_dispatch
    def f(a, b=1, *, c=2):
        return a, b, c

    def test_single(self):
        ten = self.Ten()
        self.assertEqual(self.f(ten), (10, 1, 2))
        self.assertEqual(self.f(1, ten), (1, 10, 2))
        self.assertEqual(self.f(2, c=ten), (2, 1, ten))

    def test_double(self):
        self.assertEqual(self.f(self.Ten(), self.Twenty()), (10, 20, 2))

    def test_notimp(self):
        notimp = self.NotImp()
        self.assertEqual(self.f(notimp, self.Ten()), (notimp, 10, 2))
