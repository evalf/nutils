from nutils import *
from . import *
import sys, contextlib, tempfile, pathlib, threading

@contextlib.contextmanager
def tmpcache():
  with tempfile.TemporaryDirectory() as tmpdir:
    with config(cache=True, cachedir=str(tmpdir)):
      yield pathlib.Path(tmpdir)

class refcount(TestCase):

  def setUp(self):
    self.x = object()
    self.d = {'referenced': self.x, 'dangling': object()}

  def test_noremove(self):
    keep = set(k for k, v in self.d.items() if sys.getrefcount(v) > 3)
    assert keep == {'referenced', 'dangling'}

  def test_remove(self):
    keep = set(k for k, v in self.d.items() if sys.getrefcount(v) > 4)
    assert keep == {'referenced'}

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
