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

class CacheFunc( object ):
  'cached function evaluation'

  def __init__( self, topo, func ):
    'compare'

    self.topo = topo
    self.func = func
    self.data = [ (elem,) + func(elem('gauss10')) for elem in topo ]

  def __eq__( self, other ):
    'compare'

    return isinstance( other, CacheFunc ) and self.topo == other.topo and self.func == other.func

class IterData( function.Evaluable ):
  'integrate iterator'

  needxi = True

  def __init__( self, topo, coords, *funcs ):
    'constructor'

    Y_Funcs = function.Tuple(( coords, function.Tuple(funcs) ))
    self.args = coords, CacheFunc( topo, Y_Funcs )

  @staticmethod
  def eval( xi, x, cachefunc ):
    'convolute shapes'

    iterdata = []
    for elem, y, funcs in cachefunc.data:
      if elem is xi.elem:
        y, funcs = cachefunc.func(elem('uniform1000'))
      d = x[:,:,_] - y[:,_,:] # FIX count number of axes in x
      r2 = ( d * d ).sum( 0 )
      logr = .5 * numpy.log( r2 )
      iterdata.append( (d,r2,logr) + funcs )
    return xi.points.coords.shape[1:], iterdata

class Laplacelet( function.ArrayFunc ):
  'Laplacelet'

  def __init__( self, topo, coords, funcsp ):
    'constructor'

    self.topo = topo
    self.coords = coords
    self.funcsp = funcsp
    self.shape = int(funcsp.shape[0]),
    self.args = int(funcsp.shape[0]), IterData( topo, coords, funcsp * function.IntegrationWeights( coords, ndims=1 ), function.ArrayIndex(funcsp) )

  @staticmethod
  def eval( ndofs, (shape,iterdata) ):
    'evaluate'

    retval = numpy.zeros( (ndofs,)+shape )
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

    return AddPart( ( self.grad( coords ) * coords.normal() ).sum(), .5 * self.funcsp )

  def reconstruct( self, bval, flux ):
    'reconstruct'

    return LaplaceletReconstruct( self.topo, self.coords, bval, flux )

class LaplaceletGrad( function.ArrayFunc ):
  'laplacelet gradient'

  def __init__( self, topo, coords, funcsp ):
    'constructor'

    self.shape = int(funcsp.shape[0]), 2
    self.args = int(funcsp.shape[0]), IterData( topo, coords, funcsp * function.IntegrationWeights( coords, ndims=1 ), function.ArrayIndex(funcsp) )

  @staticmethod
  def eval( ndofs, (shape,iterdata) ):
    'evaluate'

    retval = numpy.zeros( (ndofs,2) + shape )
    for D, R2, logR, wf, I in iterdata:
      kernel = D / ( R2 * (-2*numpy.pi) )
      retval[ I ] += numpy.tensordot( wf, kernel, (-1,-1) )
    return retval

class LaplaceletReconstruct( function.ArrayFunc ):
  'laplacelet reconstruction'

  def __init__( self, topo, coords, bval, flux ):
    'constructor'

    self.shape = ()
    self.args = IterData( topo, coords, function.IntegrationWeights( coords, ndims=1 ), bval, flux, coords.normal() ),

  @staticmethod
  def eval( (shape,iterdata) ):
    'evaluate'

    retval = 0
    for D, R2, logR, w, bval, flux, normal in iterdata:
      retval += numpy.dot( logR * flux + ( D * normal[:,_,:] ).sum(0) * bval / R2, w ) # TODO fix for arbitrary axes
    return retval / (-2*numpy.pi)

class Stokeslet( function.ArrayFunc ):
  'stokeslet'

  def __init__( self, topo, coords, funcsp, mu ):
    'constructor'

    self.topo = topo
    self.coords = coords
    self.funcsp = funcsp
    self.mu = mu
    self.shape = int( funcsp.shape[0] ) * 2, 2
    self.args = int(funcsp.shape[0]), IterData( topo, coords, funcsp * function.IntegrationWeights( coords, ndims=1 ), function.ArrayIndex(funcsp) ), mu

  @staticmethod
  def eval( ndofs, (shape,iterdata), mu ):
    'evaluate'

    retval = numpy.zeros( (ndofs*2,2) + shape )
    retval_swap = retval.reshape( 2, ndofs, 2, *shape ).swapaxes(0,1) # follows ordering Vectorize
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

    return AddPart( ( self.stress( coords ) * coords.normal() ).sum(), .5 * self.funcsp.vector(2) )

  def reconstruct( self, bval, flux ):
    'reconstruct'

    return StokesletReconstruct( self.topo, self.coords, bval, flux, self.mu )

class StokesletStress( function.ArrayFunc ):
  'stokeslet stress'

  def __init__( self, topo, coords, funcsp ):
    'constructor'

    self.shape = int( funcsp.shape[0] ) * 2, 2, 2
    self.args = int(funcsp.shape[0]), IterData( topo, coords, funcsp * function.IntegrationWeights( coords, ndims=1 ), function.ArrayIndex(funcsp) )

  @staticmethod
  def eval( ndofs, (shape,iterdata) ):
    'evaluate'

    retval = numpy.zeros( (ndofs*2,2,2) + shape )
    retval_swap = retval.reshape( 2, ndofs, 2, 2, *shape ).swapaxes(0,1) # follows ordering Vectorize
    for D, R2, logR, wf, I in iterdata:
      kernel = D[:,_,_] * D[_,:,_] * D[_,_,:]
      kernel /= (R2**2) * -numpy.pi
      retval_swap[I] += numpy.tensordot( wf, kernel, (-1,-1) )
    return retval

class StokesletReconstruct( function.ArrayFunc ):
  'stokeslet reconstruction'

  def __init__( self, topo, coords, velo, trac, mu ):
    'constructor'

    self.shape = 2,
    self.args = IterData( topo, coords, function.IntegrationWeights( coords, ndims=1 ), velo, trac, coords.normal() ), mu

  @staticmethod
  def eval( (shape,iterdata), mu ):
    'evaluate'

    retval = 0
    for D, R2, logR, w, velo, trac, norm in iterdata:
      D_R2 = D / R2
      Dtrac = ( D_R2 * trac[:,_,:] ).sum(0)
      Dvelo = ( D_R2 * velo[:,_,:] ).sum(0)
      Dnorm = ( D_R2 * norm[:,_,:] ).sum(0)
      retval += numpy.dot( ( D * Dtrac - trac[:,_,:] * logR ) / (4*mu) - D * Dnorm * Dvelo, w )
    return retval / numpy.pi

# vim:shiftwidth=2:foldmethod=indent:foldnestmax=2
