class UnitTest( object ):
  ok = error = 0
  def __init__( self, width=60 ):
    self.width = width
    print '-' * self.width
  def __call__( self, func ):
    s = '%s ... ' % func.func_name
    try:
      func()
    except Exception, e:
      s += repr(e)
      print s + 'ERROR'.rjust(self.width-len(s))
      self.error += 1
    else:
      print s + 'OK'.rjust(self.width-len(s))
      self.ok += 1
    return func
  def exit( self ):
    print '-' * self.width
    print '%d tests successful, %d tests failed' % ( self.ok, self.error )
    raise SystemExit( self.error )

unittest = UnitTest()

## BEGIN UNIT TESTS ##

import __init__ as numeric
import numpy

@unittest
def equal_self():
  A = numeric.arange(1,3)
  assert A == A, 'A==A'

@unittest
def nonzero():
  A = numeric.arange(1,3)
  assert A, 'A'
  B = numeric.empty((0,3))
  assert not B, 'not B'

@unittest
def notequal_object():
  A = numeric.arange(1,3)
  assert A != (1,2), 'A!=tuple'

@unittest
def equal_identical_other():
  A = numeric.arange(1,3)
  B = numeric.arange(1,3)
  assert A == B, 'A==B'
  assert B == A, 'B==A'

@unittest
def notequal_other_shape():
  A = numeric.arange(1,3)
  B = numeric.arange(1,3).reshape(1,2)
  assert A != B, 'A!=B'
  assert B != A, 'B!=A'

@unittest
def notequal_other_dtype():
  A = numeric.arange(1,3)
  B = numeric.arange(1,3,dtype=float)
  assert A != B, 'A!=B'
  assert B != A, 'B!=A'

@unittest
def notequal_other_data():
  A = numeric.arange(1,3)
  B = numeric.arange(1,4)
  assert A != B, 'A!=B'
  assert B != A, 'B!=A'

@unittest
def scalar_equal_int_int():
  A = numeric.asarray(1)
  B = 1
  assert A == B, 'A==B'
  assert B == A, 'B==A'

@unittest
def scalar_equal_int_float():
  A = numeric.asarray(1)
  B = 1.
  assert A == B, 'A==B'
  assert B == A, 'B==A'

@unittest
def scalar_less():
  A = numeric.asarray(1)
  B = numeric.asarray(2.)
  assert A <  B, 'A<B'
  assert A <= B, 'A<=B'
  assert not A >= B, '!A>=B'
  assert not A >  B, '!A>B'

@unittest
def scalar_notequal_int_other_float():
  A = numeric.asarray(1)
  B = 2.
  assert A != B, 'A!=B'
  assert B != A, 'B!=A'

@unittest
def zerosize_equal_sameshape():
  A = numeric.ones([2,0])
  B = numeric.zeros([2,0])
  assert A == B, 'A==B'
  assert B == A, 'B==A'

@unittest
def zerosize_notequal_othershape():
  A = numeric.ones([2,0])
  B = numeric.zeros([0,2])
  assert A != B, 'A!=B'
  assert B != A, 'B!=A'

@unittest
def retrieve_from_dict():
  A = numeric.arange(1,3)
  B = numeric.arange(3,5)
  d = { A: 1, B: 2 }
  assert d[ numeric.asarray([1,2]) ] == 1, 'A'
  assert d[ numeric.asarray([3,4]) ] == 2, 'B'

@unittest
def bisection():
  target1 = numeric.arange(1,3)
  target2 = numeric.arange(1,4)
  arrays = [ target1, 2, False, numeric.asarray([1,3]), None, (), numeric.asarray([5]), target2 ]
  arrays.sort()
  i1 = i2 = 0
  for n in range(2,-1,-1): # list is length 8 = 2**3
    if target1 >= arrays[i1|(1<<n)]: i1 |= 1<<n
    if target2 >= arrays[i2|(1<<n)]: i2 |= 1<<n
  assert arrays[i1] == target1, 'target1'
  assert arrays[i2] == target2, 'target2'

@unittest
def dot_matrix_vector():
  A = numeric.arange(1,5).reshape(2,2)
  B = numeric.arange(5,7)
  AB = numeric.dot(A,B)
  assert AB.ndim == 1, 'A,B ndim'
  assert AB.dtype == float, 'A,B dtype'
  assert numpy.equal( AB, [17,39] ).all(), 'A,B data'
  BA = numeric.dot(B,A)
  assert BA.ndim == 1, 'B,A ndim'
  assert BA.dtype == float, 'B,A dtype'
  assert numpy.equal( BA, [23,34] ).all(), 'B,A data'

@unittest
def dot_matrix_matrix():
  A = numeric.arange(1,5).reshape(2,2)
  B = numeric.arange(5,9).reshape(2,2)
  AB = numeric.dot(A,B)
  assert AB.ndim == 2, 'A,B ndim'
  assert AB.dtype == float, 'A,B ndim'
  assert numpy.equal( AB, [[19,22],[43,50]] ).all(), 'A,B data'
  ATB = numeric.dot(A.T,B)
  assert ATB.ndim == 2, 'A.T,B ndim'
  assert ATB.dtype == float, 'A.T,B dtype'
  assert numpy.equal( ATB, [[26,30],[38,44]] ).all(), 'A.T,B data'
  ABT = numeric.dot(A,B.T)
  assert ABT.ndim == 2, 'A,B.T ndim'
  assert ABT.dtype == float, 'A,B.T dtype'
  assert numpy.equal( ABT, [[17,23],[39,53]] ).all(), 'A,B.T data'

@unittest
def dot_tensor_tensor():
  A = numeric.arange(1,7).reshape(2,1,3)
  B = numeric.arange(7,13).reshape(3,1,2)
  AB = numeric.dot(A,B)
  assert AB.ndim == 4, 'A,B ndim'
  assert AB.dtype == float, 'A,B dtype'
  assert numpy.equal( AB, [[[[58,64]]],[[[139,154]]]] ).all(), 'A,B data'
  BA = numeric.dot(B,A)
  assert BA.ndim == 4, 'B,A ndim'
  assert BA.dtype == float, 'B,A dtype'
  assert numpy.equal( BA, [[[[39,54,69]]],[[[49,68,87]]],[[[59,82,105]]]] ).all(), 'B,A data'

@unittest
def contract_matrix_tensor():
  A = numeric.arange(9,13).reshape(2,2)
  B = numeric.arange(1,9).reshape(2,2,2)
  AB0 = numeric.contract(A,B,axis=0)
  assert AB0.ndim == 2, 'axis==0 ndim'
  assert AB0.dtype == float, 'axis==0 dtype'
  assert numpy.equal( AB0, [[54,80],[110,144]] ).all(), 'axis=0 data'
  AB1 = numeric.contract(A,B,axis=1)
  assert AB1.ndim == 2, 'axis==1 ndim'
  assert AB1.dtype == float, 'axis==1 ndim'
  assert numpy.equal( AB1, [[42,68],[122,156]] ).all(), 'axis=1 data'
  AB2 = numeric.contract(A,B)
  assert AB2.ndim == 2, 'axis==2 ndim'
  assert AB2.dtype == float, 'axis==2 dtype'
  assert numpy.equal( AB2, [[29,81],[105,173]] ).all(), 'axis=2 data'

## END UNIT TESTS ##

unittest.exit()
