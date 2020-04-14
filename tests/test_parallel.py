import unittest, os, multiprocessing, time, sys
from nutils import parallel

canfork = hasattr(os, 'fork')

@unittest.skipIf(sys.platform == 'darwin', 'fork is unreliable (in combination with matplotlib)')
class Test(unittest.TestCase):

  def setUp(self):
    parallel._maxprocs = 3

  def tearDown(self):
    parallel._maxprocs = 1

  def test_maxprocs(self):
    with parallel.maxprocs(4):
      self.assertEqual(parallel._maxprocs, 4)
    self.assertEqual(parallel._maxprocs, 3)

  def test_fork(self):
    mask = multiprocessing.RawValue('i', 0)
    lock = multiprocessing.Lock()
    with parallel.fork() as procid, lock:
      mask.value |= 1 << procid
    self.assertEqual(mask.value, 0b111 if canfork else 1)

  def test_shzeros(self):
    a = parallel.shzeros([3], dtype=int)
    with parallel.fork() as procid:
      a[procid] = 1
    self.assertEqual(a.tolist(), [1,1,1] if canfork else [1,0,0])

  def test_failinmain(self):
    with self.assertRaises(ZeroDivisionError), parallel.fork() as procid:
      if procid == 0:
        1/0

  @unittest.skipIf(not canfork, 'fork is not available on this system')
  def test_failinchild(self):
    with self.assertRaisesRegex(Exception, 'fork failed in 2 out of 3 processes'), parallel.fork() as procid:
      if procid != 0:
        1/0

  def test_range(self):
    a = parallel.shempty([32], dtype=int)
    a[:] = -1
    r = parallel.range(len(a))
    with parallel.fork() as procid:
      for i in r:
        a[i] = procid
        time.sleep(.01)
    self.assertEqual(min(a), 0)
    self.assertEqual(max(a), 2 if canfork else 0)

  def test_ctxrange(self):
    a = parallel.shzeros([32], dtype=int)
    with parallel.ctxrange('test', len(a)) as r:
      for i in r:
        a[i] = 1
        time.sleep(.01)
    self.assertEqual(a.tolist(), [1]*len(a))
