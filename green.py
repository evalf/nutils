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

class IterData( function.Evaluable ):
  'integrate iterator'

  needxi = True

  def __init__( self, topo, coords, funcsp ):
    'constructor'

    self.args = coords, (topo,coords,funcsp)

  @staticmethod
  def eval( xi, x, (topo,coords,funcsp) ):
    'convolute shapes'

    D = function.StaticArray( x ) - coords[:,_]
    R2 = ( D * D ).sum( 0 )
    logR = .5 * function.Log( R2 )
    detJ = function.Jacobian( coords )[:,0].norm2( 0 )
    T = function.Tuple(( D, R2, logR, detJ * funcsp ))

    iterdata = []
    I = slice(None)
    for elem in topo:
      xi_ = elem( 'uniform1000' if elem is xi.elem else 'gauss10' )
      d, r2, logr, f_detj = T( xi_ )
      if funcsp.shape:
        I = funcsp.shape[0].get( xi_.elem )
        assert I is not None
      iterdata.append(( d, r2, logr, xi_.weights * f_detj, I ))
    return tuple( int(n) for n in funcsp.shape ) + xi.points.coords.shape[1:], iterdata

class Laplacelet( function.ArrayFunc ):
  'Laplacelet'

  def __init__( self, topo, coords, funcsp ):
    'constructor'

    self.topo = topo
    self.coords = coords
    self.funcsp = funcsp
    self.shape = tuple( int(n) for n in funcsp.shape )
    self.args = IterData( topo, coords, funcsp ),

  def dot( self, weights ):
    'dot'

    return self.__class__( self.topo, self.coords, self.funcsp.dot(weights) )

  @staticmethod
  def eval( (shape,iterdata) ):
    'evaluate'

    retval = numpy.zeros( shape )
    for D, R2, logR, wf, I in iterdata:
      kernel = logR / (-2*numpy.pi)
      retval[ I ] += numpy.tensordot( wf, kernel, (-1,-1) )
    return retval

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

class LaplaceletGrad( function.ArrayFunc ):
  'laplacelet gradient'

  def __init__( self, topo, coords, funcsp ):
    'constructor'

    self.topo = topo
    self.coords = coords
    self.funcsp = funcsp
    self.shape = tuple( int(n) for n in funcsp.shape ) + (2,)
    self.args = IterData( topo, coords, funcsp ),

  def dot( self, weights ):
    'dot'

    return self.__class__( self.topo, self.coords, self.funcsp.dot(weights) )

  @staticmethod
  def eval( (shape,iterdata) ):
    'evaluate'

    retval = numpy.zeros( shape[:-1] + (2,) + shape[-1:] )
    for D, R2, logR, wf, I in iterdata:
      kernel = D / ( R2 * (-2*numpy.pi) )
      retval[ I ] += numpy.tensordot( wf, kernel, (-1,-1) )
    return retval

class Stokeslet( function.ArrayFunc ):
  'stokeslet'

  def __init__( self, topo, coords, funcsp, mu ):
    'constructor'

    self.topo = topo
    self.coords = coords
    self.funcsp = funcsp
    self.shape = int( funcsp.shape[0] ) * 2, 2
    self.args = IterData( topo, coords, funcsp ), mu

  @staticmethod
  def eval( (shape,iterdata), mu ):
    'evaluate'

    retval = numpy.zeros( (shape[0]*2,2) + shape[1:] )
    retval_swap = retval.reshape( 2, -1, *retval.shape[1:] ).swapaxes(0,1) # follows ordering Vectorize
    for D, R2, logR, wf, I in iterdata:
      kernel = D[:,_] * D[_,:]
      kernel /= R2
      kernel[0,0] -= logR
      kernel[1,1] -= logR
      kernel /= 4 * numpy.pi * mu
      retval_swap[I] += numpy.tensordot( wf, kernel, (-1,-1) )
    return retval

  def stress( self, coords ):
    'gradient'

    assert coords == self.coords
    return StokesletStress( self.topo, self.coords, self.funcsp )

  def traction( self, coords ):
    'flux'

    if self.funcsp.shape: # HACK!!
      return AddPart( ( self.stress( coords ) * coords.normal() ).sum(), .5 * self.funcsp.vector(2) )
    else:
      return ( self.stress( coords ) * coords.normal() ).sum() + .5 * self.funcsp.vector(2)

class StokesletStress( function.ArrayFunc ):
  'stokeslet stress'

  def __init__( self, topo, coords, funcsp ):
    'constructor'

    self.topo = topo
    self.coords = coords
    self.funcsp = funcsp
    self.shape = int( funcsp.shape[0] ) * 2, 2, 2
    self.args = IterData( topo, coords, funcsp ),

  @staticmethod
  def eval( (shape,iterdata) ):
    'evaluate'

    retval = numpy.zeros( (shape[0]*2,2,2) + shape[1:] )
    retval_swap = retval.reshape( 2, -1, *retval.shape[1:] ).swapaxes(0,1) # follows ordering Vectorize
    for D, R2, logR, wf, I in iterdata:
      kernel = D[:,_,_] * D[_,:,_] * D[_,_,:]
      kernel /= (R2**2) * -numpy.pi
      retval_swap[I] += numpy.tensordot( wf, kernel, (-1,-1) )
    return retval

# vim:shiftwidth=2:foldmethod=indent:foldnestmax=2
