from nutils import *
from nutils.testing import *
import sys, contextlib, tempfile, pathlib, threading

@contextlib.contextmanager
def tmpcache():
  with tempfile.TemporaryDirectory() as tmpdir:
    with cache.enable(tmpdir):
      yield pathlib.Path(tmpdir)

class TestException(Exception): pass

class function(TestCase):

  def test_nocache(self):

    @cache.function
    def func():
      return 'spam'

    self.assertEqual(func(), 'spam')

  def test_cache(self):

    @cache.function
    def func():
      nonlocal ncalls
      ncalls += 1
      return 'spam'

    with tmpcache():
      ncalls = 0
      self.assertEqual(func(), 'spam')
      self.assertEqual(ncalls, 1)
      self.assertEqual(func(), 'spam')
      self.assertEqual(ncalls, 1)

  def test_cache_with_args(self):

    @cache.function
    def func(a, b):
      nonlocal ncalls
      ncalls += 1
      return a + b

    with tmpcache():
      ncalls = 0
      self.assertEqual(func(1, 2), 3)
      self.assertEqual(ncalls, 1)
      self.assertEqual(func(2, 1), 3)
      self.assertEqual(ncalls, 2)
      self.assertEqual(func(1, 2), 3)
      self.assertEqual(ncalls, 2)
      self.assertEqual(func(a=2, b=1), 3)
      self.assertEqual(ncalls, 2)

  def test_cache_with_kwonly_args(self):

    @cache.function
    def func(a, *, b):
      nonlocal ncalls
      ncalls += 1
      return a + b

    with tmpcache():
      ncalls = 0
      self.assertEqual(func(1, b=2), 3)
      self.assertEqual(ncalls, 1)
      self.assertEqual(func(2, b=1), 3)
      self.assertEqual(ncalls, 2)
      self.assertEqual(func(1, b=2), 3)
      self.assertEqual(ncalls, 2)
      self.assertEqual(func(a=2, b=1), 3)
      self.assertEqual(ncalls, 2)

  def test_cache_with_kwargs(self):

    @cache.function
    def func(a, *args, **kwargs):
      nonlocal ncalls
      ncalls += 1
      return a + sum(args) + sum(kwargs.values())

    with tmpcache():
      ncalls = 0
      self.assertEqual(func(1, 2, b=3), 6)
      self.assertEqual(ncalls, 1)
      self.assertEqual(func(1, b=2, c=3), 6)
      self.assertEqual(ncalls, 2)
      self.assertEqual(func(2, 1, b=3), 6)
      self.assertEqual(ncalls, 3)
      self.assertEqual(func(a=1, c=3, b=2), 6)
      self.assertEqual(ncalls, 3)

  def test_cache_with_version(self):

    @cache.function
    def func():
      nonlocal ncalls
      ncalls += 1
      return 'spam'

    func0 = func

    @cache.function(version=0)
    def func():
      nonlocal ncalls
      ncalls += 1
      return 'spam'

    func1 = func

    @cache.function(version=1)
    def func():
      nonlocal ncalls
      ncalls += 1
      return 'eggs'

    func2 = func

    # The first two functions should have the same hash, the last should be
    # different from the first two because of the version number.

    with tmpcache():
      ncalls = 0
      self.assertEqual(func0(), 'spam')
      self.assertEqual(ncalls, 1)
      self.assertEqual(func1(), 'spam')
      self.assertEqual(ncalls, 1)
      self.assertEqual(func2(), 'eggs')
      self.assertEqual(ncalls, 2)

  def test_cache_invalid_version(self):

    with self.assertRaisesRegex(ValueError, "^'version' should be of type 'int'"):
      @cache.function(version='1')
      def func():
        pass

  def test_cache_exception(self):

    @cache.function
    def func():
      nonlocal ncalls
      ncalls += 1
      raise TestException('spam')

    with tmpcache():
      ncalls = 0
      with self.assertRaises(TestException) as cm:
        func()
      self.assertEqual(cm.exception.args[0], 'spam')
      self.assertEqual(ncalls, 1)
      with self.assertRaises(TestException) as cm:
        func()
      self.assertEqual(cm.exception.args[0], 'spam')
      self.assertEqual(ncalls, 1)

  def test_corruption(self):

    @cache.function
    def func():
      nonlocal ncalls
      ncalls += 1
      return 'spam'

    for corruption in '', 'bogus':
      with self.subTest(corruption=corruption), tmpcache() as cachedir:

        ncalls = 0

        self.assertEqual(func(), 'spam')
        self.assertEqual(ncalls, 1)

        cache_files = tuple(cachedir.iterdir())
        self.assertEqual(len(cache_files), 1)
        cache_file, = cache_files
        with cache_file.open('wb') as f:
          f.write(corruption.encode())

        self.assertEqual(func(), 'spam')
        self.assertEqual(ncalls, 2)

        self.assertEqual(func(), 'spam')
        self.assertEqual(ncalls, 2)

  @unittest.skipIf(cache._lock_file is cache._lock_file_fallback, 'platform does not support file locks')
  def test_concurrent_access(self):

    @cache.function
    def func():
      nonlocal ncalls
      ncalls += 1
      return 'spam'

    def wrapper():
      nonlocal nsuccess
      assert func() == 'spam'
      nsuccess += 1

    with tmpcache() as cachedir:

      ncalls = 0
      nsuccess = 0

      # Call `wrapper`.  Since the cache is clean `func` should be called.
      wrapper()
      self.assertEqual(ncalls, 1)
      self.assertEqual(nsuccess, 1)

      # Find the cache file, obtain a lock and call `wrapper` in a thread.
      # `wrapper` should block on acquiring the file lock in `function.cache`.
      cache_files = tuple(cachedir.iterdir())
      self.assertEqual(len(cache_files), 1)
      cache_file, = cache_files
      with cache_file.open('r+b') as f:
        cache._lock_file(f)

        # We use `daemon=True` to make sure this thread won't keep the
        # interpreter alive when something goes wrong with the thread.
        t = threading.Thread(target=wrapper, daemon=True)
        t.start()
        # Give the thread some time to start.
        t.join(timeout=1)
        # Assert the thread is still running, but `func` is not called.
        self.assertTrue(t.is_alive())
        self.assertEqual(ncalls, 1)
        self.assertEqual(nsuccess, 1)

      # The lock has been released by closing the file.  The thread should
      # continue with loading the cache.
      t.join(timeout=5)
      self.assertFalse(t.is_alive())
      self.assertEqual(ncalls, 1)
      self.assertEqual(nsuccess, 2)


class Recursion(TestCase):

  def test_nocache(self):

    untouched = object()

    class R(cache.Recursion, length=1):
      def resume(R_self, history):
        nonlocal received_history
        received_history =tuple(history)
        yield from range(0 if not history else history[-1]+1, 10)

    received_history = untouched
    self.assertEqual(tuple(R()), tuple(range(10)))
    self.assertEqual(received_history, ())

  def test_cache(self):

    read = lambda iterable, n: tuple(item for i, item in zip(range(n), iterable))
    untouched = object()

    for length in 1, 2, 3:
      with self.subTest(length=length), tmpcache():

        class R(cache.Recursion, length=length):
          def resume(R_self, history):
            nonlocal received_history
            received_history = tuple(history)
            yield from range(0 if not history else history[-1]+1, 10)

        received_history = untouched
        self.assertEqual(read(R(), 4), tuple(range(4)))
        self.assertEqual(received_history, ())

        received_history = untouched
        self.assertEqual(read(R(), 3), tuple(range(3)))
        self.assertEqual(received_history, untouched)

        received_history = untouched
        self.assertEqual(read(R(), 6), tuple(range(6)))
        self.assertEqual(received_history, tuple(range(4-length,4)))

        received_history = untouched
        self.assertEqual(read(R(), 12), tuple(range(10)))
        self.assertEqual(received_history, tuple(range(6-length,6)))

        received_history = untouched
        self.assertEqual(read(R(), 12), tuple(range(10)))
        self.assertEqual(received_history, untouched)

  def test_cache_exception(self):

    read = lambda iterable, n: tuple(item for i, item in zip(range(n), iterable))
    untouched = object()

    class R(cache.Recursion, length=1):
      def resume(R_self, history):
        nonlocal received_history
        received_history = tuple(history)
        yield from range(0 if not history else history[-1]+1, 2)
        raise TestException('spam')

    with tmpcache():

      received_history = untouched
      with self.assertRaises(TestException) as cm:
        read(R(), 3)
      self.assertEqual(cm.exception.args[0], 'spam')
      self.assertEqual(received_history, ())

      received_history = untouched
      self.assertEqual(read(R(), 2), tuple(range(2)))
      self.assertEqual(received_history, untouched)

      received_history = untouched
      with self.assertRaises(TestException) as cm:
        read(R(), 4)
      self.assertEqual(cm.exception.args[0], 'spam')
      self.assertEqual(received_history, untouched)

  def test_corruption(self):

    read = lambda iterable, n: tuple(item for i, item in zip(range(n), iterable))
    untouched = object()

    class R(cache.Recursion, length=1):
      def resume(R_self, history):
        nonlocal received_history
        received_history = tuple(history)
        yield from range(0 if not history else history[-1]+1, 10)

    for icorrupted in range(3):
      for corruption in '', 'bogus':
        with self.subTest(corruption=corruption, icorrupted=icorrupted), tmpcache() as cachedir:

          received_history = untouched
          self.assertEqual(read(R(), 4), tuple(range(4)))
          self.assertEqual(received_history, ())

          cache_files = tuple(cachedir.iterdir())
          self.assertEqual(len(cache_files), 1)
          cache_file, = cache_files
          self.assertTrue((cache_file/'{:04d}'.format(icorrupted)).exists())
          with (cache_file/'{:04d}'.format(icorrupted)).open('wb') as f:
            f.write(corruption.encode())

          received_history = untouched
          self.assertEqual(read(R(), 6), tuple(range(6)))
          self.assertEqual(received_history, (icorrupted-1,) if icorrupted else ())

  @unittest.skipIf(cache._lock_file is cache._lock_file_fallback, 'platform does not support file locks')
  def test_concurrent_access(self):

    read = lambda iterable, n: tuple(item for i, item in zip(range(n), iterable))
    untouched = object()

    class R(cache.Recursion, length=1):
      def resume(R_self, history):
        nonlocal received_history
        received_history = tuple(history)
        yield from range(0 if not history else history[-1]+1, 10)

    def wrapper(n):
      nonlocal nsuccess
      assert read(R(), n) == tuple(range(n))
      nsuccess += 1

    for ilock in range(3):
      with self.subTest(ilock=ilock), tmpcache() as cachedir:

        nsuccess = 0

        # Call `wrapper`.  Since the cache is clean `R.resume` should be called with empty history.
        received_history = untouched
        wrapper(4)
        self.assertEqual(received_history, ())
        self.assertEqual(nsuccess, 1)

        # Find the cache file of iteration `ilock`, obtain a lock and call
        # `wrapper` in a thread.  `wrapper` should block on acquiring the file
        # lock in `function.Recursion`.
        cache_files = tuple(cachedir.iterdir())
        self.assertEqual(len(cache_files), 1)
        cache_file = cache_files[0]/'{:04d}'.format(ilock)
        assert cache_file.exists()
        with cache_file.open('r+b') as f:
          cache._lock_file(f)

          # We use `daemon=True` to make sure this thread won't keep the
          # interpreter alive when something goes wrong with the thread.
          received_history = untouched
          t = threading.Thread(target=lambda: wrapper(5), daemon=True)
          t.start()
          # Give the thread some time to start.
          t.join(timeout=1)
          # Assert the thread is still running, but `R.resume` is not called.
          self.assertEqual(received_history, untouched)
          self.assertEqual(nsuccess, 1)

        # The lock has been released by closing the file.  The thread should
        # continue with loading the cache and ultimately calling `R.resume
        t.join(timeout=5)
        self.assertFalse(t.is_alive())
        self.assertEqual(received_history, (3,))
        self.assertEqual(nsuccess, 2)
