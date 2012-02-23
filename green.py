from . import util, element, numpy, function, _

class AddPart( function.ArrayFunc ):
  'add partial'

  needxi = True

  def __init__( self, func1, func2 ):
    'constructor'

    self.args = func1, func2, func2.shape[0]
    self.shape = func1.shape

  @staticmethod
  def eval( xi, func1, func2, idxmap ):
    'evaluate'

    retval = func1.copy()
    index = idxmap.get( xi.elem )
    while index is None:
      xi = xi.next
      index = idxmap.get( xi.elem )
    retval[ index ] += func2
    return retval

class BEMFunc( function.ArrayFunc ):
  'BEM convoluted function'

  needxi = True

  def __init__( self, topo, coords, funcsp ):
    'constructor'

    self.topo = topo
    self.coords = coords
    self.funcsp = funcsp
    self.shape = tuple( int(n) for n in funcsp.shape ) + self.Kernel.Shape
    for i in range( len(self.Kernel.Shape) ):
      funcsp = funcsp[...,_] # TODO: combine getitems
    self.args = coords, (self.Kernel,funcsp,topo,coords)

  def dot( self, weights ):
    'dot'

    return self.__class__( self.topo, self.coords, self.funcsp.dot(weights) )

  @staticmethod
  def eval( xi, x, (kernel,func,topo,coords) ):
    'convolute shapes'

    detJ = function.Jacobian( coords )[:,0].norm2( 0 )
    integrand = func[...,_] * kernel( coords, x )
    integrate = function.Tuple(( function.ArrayIndex(integrand), function.Integrate( integrand, detJ ) ))
    u = numpy.zeros( integrand.shape )
    for elem in topo:
      xi_ = elem( 'uniform1000' if elem is xi.elem else 'gauss10' )
      I, data = integrate( xi_ )
      u[ I ] += data
    return u

class Laplacelet( BEMFunc ):
  'Laplacelet'

  class Kernel( function.ArrayFunc ):
    'laplacelet'

    Shape = ()
    
    def __init__( self, coords, x ):
      'constructor'
  
      self.args = coords, x
      self.shape = x.shape[1:]
  
    @staticmethod
    def eval( x1, x2 ):
      'evaluate'
  
      D = x2[:,:,_] - x1[:,_,:]
      R2 = ( D * D ).sum( 0 )
      logR = .5 * numpy.log( R2 )
      return logR / (-2*numpy.pi)

  def grad( self, coords ):
    'gradient'

    assert coords == self.coords
    return LaplaceletGrad( self.topo, self.coords, self.funcsp )

  def ngrad( self, coords ):
    'flux'

    if self.funcsp.shape: # HACK!!
      return AddPart( ( self.grad( coords ) * coords.normal() ).sum(), .5 * self.funcsp )
    else:
      return ( self.grad( coords ) * coords.normal() ).sum() + .5 * self.funcsp

class LaplaceletGrad( BEMFunc ):
  'Laplacelet flux'

  class Kernel( function.ArrayFunc ):
    'laplacelet'

    Shape = 2,
    
    def __init__( self, coords, x ):
      'constructor'
  
      self.args = coords, x
      self.shape = x.shape
  
    @staticmethod
    def eval( x1, x2 ):
      'evaluate'
  
      D = x2[:,:,_] - x1[:,_,:]
      R2 = ( D * D ).sum( 0 )
      return D[:,:,:] / R2[_,:,:] / (-2*numpy.pi)

class Stokeslet( BEMFunc ):
  'Laplacelet'

  class Kernel( function.ArrayFunc ):
    'laplacelet'

    Shape = ()
    
    def __init__( self, coords, x ):
      'constructor'
  
      self.args = coords, x
      self.shape = x.shape[1:]
  
    @staticmethod
    def eval( x1, x2, MU ):
      'evaluate'
  
      D = x2[:,:,_] - x1[:,_,:]
      R2 = ( D * D ).sum( 0 )
      logR = .5 * numpy.log( R2 )
      v = ( D[:,_,:] * D[_,:,:] ) / R2[_,_,:]
      v[0,0] -= logR
      v[1,1] -= logR
      v /= 4 * numpy.pi * MU
      return v

  def grad( self, coords ):
    'gradient'

    assert coords == self.coords
    return LaplaceletGrad( self.topo, self.coords, self.funcsp )

  def ngrad( self, coords ):
    'flux'

    if self.funcsp.shape: # HACK!!
      return AddPart( ( self.grad( coords ) * coords.normal() ).sum(), .5 * self.funcsp )
    else:
      return ( self.grad( coords ) * coords.normal() ).sum() + .5 * self.funcsp

# vim:shiftwidth=2:foldmethod=indent:foldnestmax=2
