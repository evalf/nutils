import __init__ as numeric
import numpy

class unittest:
  TESTS = []
  WIDTH = 60
  def __init__( self, func ):
    self.func = func
    self.TESTS.append( self )
  def __call__( self ):
    s = '%s ... ' % self.func.func_name
    try:
      self.func()
    except Exception, e:
      s += repr( e )
      print s + 'ERROR'.rjust(self.WIDTH-len(s))
      return 1
    else:
      print s + 'OK'.rjust(self.WIDTH-len(s))
      return 0
  @classmethod
  def runall( cls ):
    print '-' * cls.WIDTH
    errcount = sum( test() for test in cls.TESTS )
    print '-' * cls.WIDTH
    print '%d tests successful, %d tests failed' % (len(cls.TESTS)-errcount,errcount)
    return errcount

@unittest
def equal_self():
  A = numeric.asarray([1,2])
  assert A == A

@unittest
def equal_identical_other():
  A = numeric.asarray([1,2])
  B = numeric.asarray([1,2])
  assert A == B

@unittest
def notequal_other_shape():
  A = numeric.asarray([1,2])
  B = numeric.asarray([[1,2]])
  assert A != B

@unittest
def notequal_other_dtype():
  A = numeric.asarray([1,2])
  B = numeric.asarray([1.,2.])
  assert A != B

@unittest
def notequal_other_data():
  A = numeric.asarray([1,2])
  B = numeric.asarray([1,3])
  assert A != B

@unittest
def scalar_equal_int_int():
  A = numeric.asarray(1)
  B = 1
  assert A == B

@unittest
def scalar_equal_int_float():
  A = numeric.asarray(1)
  B = 1.
  assert A == B

@unittest
def scalar_notequal_int_other_float():
  A = numeric.asarray(1)
  B = 2.
  assert A != B

@unittest
def zerosize_equal_sameshape():
  A = numeric.asarray( numpy.ones([2,0]) )
  B = numeric.asarray( numpy.zeros([2,0]) )
  assert A == B

@unittest
def zerosize_notequal_othershape():
  A = numeric.asarray( numpy.ones([2,0]) )
  B = numeric.asarray( numpy.zeros([0,2]) )
  assert A != B

@unittest
def retrieve_from_dict():
  A = numeric.asarray([1,2])
  d = { A: 1 }
  assert d[ numeric.asarray([1,2]) ] == 1

@unittest
def bisection():
  target1 = numeric.asarray([1,2])
  target2 = numeric.asarray([1,2,3])
  arrays = [ target1, 2, False, numeric.asarray([1,3]), None, (), numeric.asarray([5]), target2 ]
  arrays.sort()
  i1 = i2 = 0
  for n in range(2,-1,-1): # list is length 8 = 2**3
    if target1 >= arrays[i1|(1<<n)]: i1 |= 1<<n
    if target2 >= arrays[i2|(1<<n)]: i2 |= 1<<n
  assert arrays[i1] == target1
  assert arrays[i2] == target2

if __name__ == '__main__':
  status = unittest.runall()
  raise SystemExit( status )
