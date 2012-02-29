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

  def __init__( self, coords, topo ):
    'constructor'

    self.args = coords, (coords,topo)

  @staticmethod
  def eval( xi, x1, (coords,topo) ):
    'convolute shapes'

    D = function.StaticArray( x1 ) - coords[:,_]
    R2 = ( D * D ).sum( 0 )
    logR = .5 * function.Log( R2 )
    detJ = function.Jacobian( coords )[:,0].norm2( 0 )
    T = function.Tuple(( D, R2, logR, detJ ))

    iterdata = []
    for elem in topo:
      xi_ = elem( 'uniform1000' if elem is xi.elem else 'gauss10' )
      d, r2, logr, detj = T( xi_ )
      iterdata.append(( xi_, d, r2, logr, xi_.weights * detj ))
    return iterdata

class Laplacelet( function.ArrayFunc ):
  'Laplacelet'

  needxi = True

  def __init__( self, topo, coords, funcsp, igrad=0 ):
    'constructor'

    self.topo = topo
    self.coords = coords
    self.funcsp = funcsp
    self.igrad = igrad
    kernel = getattr( self, 'kernel%d' % igrad )
    self.shape = tuple( int(n) for n in funcsp.shape ) + (2,)*igrad
    self.args = IterData( coords, topo ), (kernel,funcsp,igrad)

  def dot( self, weights ):
    'dot'

    return self.__class__( self.topo, self.coords, self.funcsp.dot(weights), self.igrad )

  @staticmethod
  def eval( xi, iterdata, (kernel,func,igrad) ):
    'convolute shapes'

    shape = func.shape + (2,)*igrad + xi.points.coords.shape[1:]
    retval = numpy.zeros( shape )
    isfuncspace = func.shape and isinstance( func.shape[0], function.DofAxis )
    if not isfuncspace:
      I = slice(None)
    for xi, D, R2, logR, w in iterdata:
      if isfuncspace:
        I = func.shape[0].get( xi.elem )
        while I is None:
          xi = xi.next
          I = func.shape[0].get( xi.elem )
      retval[ I ] += numpy.tensordot( func( xi ) * w, kernel( D, R2, logR ), (-1,-1) )
    return retval

  @staticmethod
  def kernel0( D, R2, logR ):
    return logR / (-2*numpy.pi)

  @staticmethod
  def kernel1( D, R2, logR ):
    return D / ( R2 * (-2*numpy.pi) )

  def grad( self, coords ):
    'gradient'

    assert coords == self.coords
    return Laplacelet( self.topo, self.coords, self.funcsp, self.igrad+1 )

  def ngrad( self, coords ):
    'flux'

    if self.funcsp.shape: # HACK!!
      return AddPart( ( self.grad( coords ) * coords.normal() ).sum(), .5 * self.funcsp )
    else:
      return ( self.grad( coords ) * coords.normal() ).sum() + .5 * self.funcsp

class Stokeslet( function.ArrayFunc ):
  'Laplacelet'

  needxi = True

  def __init__( self, topo, coords, funcsp, mu, igrad=0 ):
    'constructor'

    self.topo = topo
    self.coords = coords
    self.funcsp = funcsp
    self.igrad = igrad
    self.mu = mu
    kernel = getattr( self, 'kernel%d' % igrad )
    self.shape = (int( funcsp.shape[0] ) * 2,) + (2,) * (igrad+1)
    self.args = IterData( coords, topo ), (kernel,funcsp,igrad,mu)

  @staticmethod
  def eval( xi, iterdata, (kernel,func,igrad,mu) ):
    'convolute shapes'

    shape = (int(func.shape[0])*2,) + (2,)*(igrad+1) + xi.points.coords.shape[1:]
    retval = numpy.zeros( shape )
    retval_swap = retval.reshape( 2, func.shape[0], *shape[1:] ).swapaxes(0,1) # follows ordering Vectorize
    for xi, D, R2, logR, w in iterdata:
      I = func.shape[0].get( xi.elem )
      while I is None:
        xi = xi.next
        I = func.shape[0].get( xi.elem )
      retval_swap[I] += numpy.tensordot( func( xi ) * w, kernel( D, R2, logR, mu ), (-1,-1) )
    return retval

  @staticmethod
  def kernel0( D, R2, logR, mu ):
    v = D[:,_] * D[_,:]
    v /= R2
    v[0,0] -= logR
    v[1,1] -= logR
    v /= 4 * numpy.pi * mu
    return v

  @staticmethod
  def kernel1( D, R2, logR, mu ):
    v = D[:,_,_] * D[_,:,_] * D[_,_,:]
    v /= (R2**2) * numpy.pi
    return v

  def stress( self, coords ):
    'gradient'

    assert coords == self.coords
    return Stokeslet( self.topo, self.coords, self.funcsp, self.mu, self.igrad+1 )

  def traction( self, coords ):
    'flux'

    if self.funcsp.shape: # HACK!!
      return AddPart( ( self.stress( coords ) * coords.normal() ).sum(), .5 * self.funcsp.vector(2) )
    else:
      return ( self.stress( coords ) * coords.normal() ).sum() + .5 * self.funcsp.vector(2)

# vim:shiftwidth=2:foldmethod=indent:foldnestmax=2
