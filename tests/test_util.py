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

    def check(self, square):
        self.assertEqual(square(2), 4)
        self.assertEqual(square([2, 3]), (4, 9))
        self.assertEqual(square([2, [(3, 4), [[[[[[5, 6]]]]]]], 7]), (4, ((9, 16), ((((((25, 36),),),),),)), 49))

    def test_function(self):
        @util.single_or_multiple
        def square(values):
            self.assertIsInstance(values, tuple)
            return [value**2 for value in values]
        self.check(square)

    def test_method(self):
        class T:
            @util.single_or_multiple
            def square(self_, values):
                self.assertIsInstance(self_, T)
                self.assertIsInstance(values, tuple)
                return [value**2 for value in values]
        self.check(T().square)


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

    def test_elapsed(self):
        self.assertEqual(str(util.elapsed()), '0:00')


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

    def test_nocondense(self):
        # combination of multihop1, multihop2, and implicit singletons
        index_map, count = util.merge_index_map(11, [[1, 3], [2, 4], [1, 4], [8, 9], [7, 8], [6, 9]], condense=False)
        self.assertEqual(count, 5)
        self.assertEqual(index_map.tolist(), [0, 1, 1, 1, 1, 5, 6, 6, 6, 6, 10])


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


class IDDict(TestCase):

    def setUp(self):
        self.d = util.IDDict()
        self.a, self.b = 'ab'
        self.d[self.a] = 1
        self.d[self.b] = 2

    def test_getitem(self):
        self.assertEqual(self.d[self.a], 1)
        self.assertEqual(self.d[self.b], 2)

    def test_setitem(self):
        c = 'c'
        self.d[c] = 3
        self.assertEqual(self.d[c], 3)

    def test_delitem(self):
        del self.d[self.a]
        self.assertNotIn(self.a, self.d)
        self.assertIn(self.b, self.d)

    def test_contains(self):
        self.assertIn(self.a, self.d)
        self.assertIn(self.b, self.d)
        c = 'c'
        self.assertNotIn('c', self.d)

    def test_len(self):
        self.assertEqual(len(self.d), 2)

    def test_get(self):
        self.assertEqual(self.d.get(self.a, 10), 1)
        self.assertEqual(self.d.get(self.b, 10), 2)
        self.assertEqual(self.d.get('c', 10), 10)

    def test_keys(self):
        self.assertEqual(list(self.d.keys()), ['a', 'b'])

    def test_iter(self):
        self.assertEqual(list(self.d), ['a', 'b'])

    def test_values(self):
        self.assertEqual(list(self.d.values()), [1, 2])

    def test_items(self):
        self.assertEqual(list(self.d.items()), [('a', 1), ('b', 2)])

    def test_str(self):
        self.assertEqual(str(self.d), "{'a': 1, 'b': 2}")

    def test_repr(self):
        self.assertEqual(repr(self.d), "{'a': 1, 'b': 2}")


class IDSet(TestCase):

    def setUp(self):
        self.a, self.b, self.c = 'abc'
        self.ab = util.IDSet([self.a, self.b])
        self.ac = util.IDSet([self.a, self.c])

    def test_union(self):
        union = self.ab | self.ac
        self.assertEqual(list(union), ['a', 'b', 'c'])
        union = self.ac.union([self.a, self.b])
        self.assertEqual(list(union), ['a', 'c', 'b'])

    def test_union_update(self):
        self.ab |= self.ac
        self.assertEqual(list(self.ab), ['a', 'b', 'c'])
        self.ac.update([self.a, self.b])
        self.assertEqual(list(self.ac), ['a', 'c', 'b'])

    def test_intersection(self):
        intersection = self.ab & self.ac
        self.assertEqual(list(intersection), ['a'])
        intersection = self.ab.intersection([self.a, self.c])
        self.assertEqual(list(intersection), ['a'])

    def test_intersection_update(self):
        self.ab &= self.ac
        self.assertEqual(list(self.ab), ['a'])
        self.ac.intersection_update([self.a, self.b])
        self.assertEqual(list(self.ac), ['a'])

    def test_difference(self):
        difference = self.ab - self.ac
        self.assertEqual(list(difference), ['b'])
        difference = self.ac - self.ab
        self.assertEqual(list(difference), ['c'])

    def test_difference_update(self):
        self.ab -= self.ac
        self.assertEqual(list(self.ab), ['b'])
        self.ac.difference_update([self.a, self.b])
        self.assertEqual(list(self.ac), ['c'])

    def test_add(self):
        self.ab.add(self.a)
        self.assertEqual(list(self.ab), ['a', 'b'])
        self.ab.add(self.c)
        self.assertEqual(list(self.ab), ['a', 'b', 'c'])
        self.ac.add(self.b)
        self.assertEqual(list(self.ac), ['a', 'c', 'b'])

    def test_isdisjoint(self):
        self.assertTrue(self.ab.isdisjoint(self.c))
        self.assertTrue(self.ac.isdisjoint(self.b))
        self.assertFalse(self.ac.isdisjoint(self.a))

    def test_pop(self):
        self.assertEqual(self.ab.pop(), 'b')
        self.assertEqual(list(self.ab), ['a'])

    def test_copy(self):
        copy = self.ab.copy()
        self.ab.pop()
        self.assertEqual(list(self.ab), ['a'])
        self.assertEqual(list(copy), ['a', 'b'])

    def test_view(self):
        view = self.ab.view()
        self.ab.pop()
        self.assertEqual(list(view), ['a'])
        with self.assertRaises(AttributeError):
            view.pop()

    def test_str(self):
        self.assertEqual(str(self.ab), "{'a', 'b'}")

    def test_repr(self):
        self.assertEqual(repr(self.ab), "{'a', 'b'}")


class replace(TestCase):

    class Base:
        def __init__(self, *args):
            self.args = args
            self.called = False
        def __reduce__(self):
            return type(self), self.args
        @util.deep_replace_property
        def simple(self):
            assert not self.called, 'caching failure: simple called twice on the same object'
            self.called = True
            if isinstance(self, replace.Ten):
                return replace.Intermediate() # to test recursion
            elif isinstance(self, replace.Intermediate):
                return 10
            else:
                return self

    class Ten(Base): pass
    class Intermediate(Base): pass

    @staticmethod
    @util.shallow_replace
    def subs10(obj, value):
        if isinstance(obj, replace.Ten):
            return value

    def test_deep_simple(self):
        ten = self.Ten()
        self.assertEqual(ten.simple, 10)
        self.assertIn('simple', ten.__dict__)
        self.assertEqual(ten.simple, 10)

    def test_deep_nested(self):
        ten = self.Ten()
        obj = self.Base(5, {7, ten})
        self.assertEqual(type(obj.simple), type(obj))
        self.assertEqual(obj.simple.args, (5, {7, 10}))
        self.assertIn('simple', obj.__dict__)
        self.assertIn('simple', ten.__dict__)

    def test_shallow_simple(self):
        ten = self.Ten()
        self.assertEqual(self.subs10(ten, 20), 20)

    def test_shallow_nested(self):
        ten = self.Ten()
        obj = self.Base(5, {7, ten})
        newobj = self.subs10(obj, 20)
        self.assertEqual(type(newobj), type(obj))
        self.assertEqual(newobj.args, (5, {7, 20}))

    def test_shallow_direct(self):
        ten = self.Ten()
        obj = self.Base(5, {7, ten})
        def subs(arg):
            if isinstance(arg, self.Ten):
                return 20
        newobj = util.shallow_replace(subs, obj)
        self.assertEqual(type(newobj), type(obj))
        self.assertEqual(newobj.args, (5, {7, 20}))


class untake(TestCase):

    def test_default(self):
        self.assertEqual(util.untake([1,2,0]), (2,0,1))

    def test_target(self):
        self.assertEqual(util.untake([1,2,0], 'abc'), ('c','a','b'))


class abstract_property(TestCase):

    def test(self):
        class A:
            x = util.abstract_property()
        a = A()
        with self.assertRaisesRegex(NotImplementedError, 'class A fails to implement x'):
            a.x
