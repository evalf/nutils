import numpy

print '''\
WARNING: You are using the _numeric.py fallback module. This
may severely impact performance! Please compile the extension
module by running 'make' in the nutils/numeric directory.
'''

def _contract( A, B, axes ):
  assert A.shape == B.shape and axes > 0
  return ((A*B).reshape(A.shape[:-axes]+(-1,))).sum(-1)

class NumericArray( numpy.ndarray ):
  __slots__ = ()
  def __new__( cls, arr ):
    return numpy.asarray( arr ).view( cls )
  # rich comparisons
  def __eq__( self, other ): return self.__cmp__( other ) == 0
  def __gt__( self, other ): return self.__cmp__( other ) >  0
  def __ge__( self, other ): return self.__cmp__( other ) >= 0
  def __lt__( self, other ): return self.__cmp__( other ) <  0
  def __le__( self, other ): return self.__cmp__( other ) <= 0
  def __ne__( self, other ): return self.__cmp__( other ) != 0
  # classical comparisons
  def __cmp__( self, other ):
    if self is other:
      return 0
    elif self.ndim == 0 and ( not isinstance( other, numpy.ndarray ) or other.ndim == 0 ):
      return cmp( self[()], other )
    elif not isinstance( other, numpy.ndarray ):
      return -1
    elif self.dtype != other.dtype:
      return cmp( self.dtype, other.dtype )
    elif self.shape != other.shape:
      return cmp( self.shape, other.shape )
    elif self.__array_interface__['data'] == other.__array_interface__['data'] and self.strides == other.strides:
      return 0
    else:
      ne, = numpy.not_equal( self.flat, other.flat ).nonzero()
      return ne.size and cmp( self.flat[ne[0]], other.flat[ne[0]] )
  def __hash__( self ):
    return hash( self[()] if self.ndim == 0
      else (self.dtype,) + self.shape + tuple( self.flat[::self.size//4+1] ) ) # incompatible with c hash!
