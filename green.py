from . import util, element, numpy, function, _

class AddPart( function.ArrayFunc ):
  'add partial'

  needxi = True

  def __init__( self, func1, func2 ):
    'constructor'

    self.args = func1, func2, func2.shape[0]
    self.shape = func1.shape

  @staticmethod
  def eval( xi, func1, func2, I ):
    'evaluate'

    retval = func1.copy()
    retval[ I ] += func2
    return retval

class CacheFunc( object ):
  'cached function evaluation'

  def __init__( self, topo, coords, func, other ):
    'compare'

    J = coords.localgradient( topo )
    detJ = J[:,0].norm2( 0 )
    self.F = function.Tuple([ coords, func * detJ, function.Tuple(other) ])
    self.data = [ (elem,) + self.get(elem.eval('gauss10')) for elem in topo ]

  def get( self, xi ):
    'generate data'

    y, func, other = self.F( xi )
    func *= xi.weights
    return y, (func,) + other

  def __eq__( self, other ):
    'compare'

    return isinstance( other, CacheFunc ) and self.F == other.F

class IterData( function.Evaluable ):
  'integrate iterator'

  needxi = True

  def __init__( self, mycoords, topo, coords, func, *other ):
    'constructor'

    self.args = mycoords, CacheFunc( topo, coords, func, other )

  @staticmethod
  def eval( xi, x, cachefunc ):
    'convolute shapes'

    iterdata = []
    for elem, y, funcs in cachefunc.data:
      if elem is xi.elem:
        y, funcs = cachefunc.get( elem.eval('uniform1000') )
      d = x[:,:,_] - y[:,_,:] # FIX count number of axes in x
      r2 = util.contract( d, d, 0 )
      logr = .5 * numpy.log( r2 )
      iterdata.append( (d,r2,logr) + funcs )
    return xi.points.coords.shape[1:], iterdata

class Laplacelet( function.ArrayFunc ):
  'Laplacelet'

  def __init__( self, mycoords, topo, coords, funcsp ):
    'constructor'

    self.topo = topo
    self.mycoords = mycoords
    self.coords = coords
    self.funcsp = funcsp
    self.shape = int(funcsp.shape[0]),
    self.args = int(funcsp.shape[0]), IterData( mycoords, topo, coords, funcsp, funcsp.shape[0] )

  @staticmethod
  def eval( ndofs, (shape,iterdata) ):
    'evaluate'

    retval = numpy.zeros( (ndofs,)+shape )
    for D, R2, logR, wf, I in iterdata:
      kernel = logR / (-2*numpy.pi)
      retval[ I ] += numpy.tensordot( wf, kernel, (-1,-1) )
    return retval

  def grad( self ):
    'gradient'

    return LaplaceletGrad( self.mycoords, self.topo, self.coords, self.funcsp )

  def flux( self ):
    'flux'

    return AddPart( ( self.grad() * self.coords.normal(self.topo) ).sum(), .5 * self.funcsp )

  def reconstruct( self, bval, flux ):
    'reconstruct'

    return LaplaceletReconstruct( self.mycoords, self.topo, self.coords, bval, flux )

class LaplaceletGrad( function.ArrayFunc ):
  'laplacelet gradient'

  def __init__( self, mycoords, topo, coords, funcsp ):
    'constructor'

    self.shape = int(funcsp.shape[0]), 2
    self.args = int(funcsp.shape[0]), IterData( mycoords, topo, coords, funcsp, funcsp.shape[0] )

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

  def __init__( self, mycoords, topo, coords, bval, flux ):
    'constructor'

    self.shape = ()
    self.args = IterData( mycoords, topo, coords, 1, bval, flux, coords.normal(topo) ),

  @staticmethod
  def eval( (shape,iterdata) ):
    'evaluate'

    retval = 0
    for D, R2, logR, w, bval, flux, normal in iterdata:
      retval += numpy.dot( logR * flux + util.contract( D, normal[:,_,:], 0 ) * bval / R2, w ) # TODO fix for arbitrary axes
    return retval / (-2*numpy.pi)

class Stokeslet( function.ArrayFunc ):
  'stokeslet'

  def __init__( self, mycoords, topo, coords, funcsp, mu ):
    'constructor'

    self.topo = topo
    self.coords = coords
    self.mycoords = mycoords
    self.funcsp = funcsp
    self.mu = mu
    self.shape = int( funcsp.shape[0] ) * 2, 2
    self.args = int(funcsp.shape[0]), IterData( mycoords, topo, coords, funcsp, funcsp.shape[0] ), mu

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

  def stress( self ):
    'gradient'

    return StokesletStress( self.mycoords, self.topo, self.coords, self.funcsp )

  def traction( self ):
    'flux'

    return AddPart( ( self.stress() * self.coords.normal( self.topo ) ).sum(), .5 * self.funcsp.vector(2) )

  def reconstruct( self, bval, flux ):
    'reconstruct'

    return StokesletReconstruct( self.mycoords, self.topo, self.coords, bval, flux, self.mu )

class StokesletGrad( function.ArrayFunc ):
  'stokeslet gradient'

  def __init__( self, mycoords, topo, coords, funcsp, mu ):
    'constructor'

    self.shape = int( funcsp.shape[0] ) * 2, 2, 2
    self.args = int(funcsp.shape[0]), IterData( mycoords, topo, coords, funcsp, funcsp.shape[0] ), mu

  @staticmethod
  def eval( ndofs, (shape,iterdata), mu ):
    'evaluate'

    retval = numpy.zeros( (ndofs*2,2,2) + shape )
    retval_swap = retval.reshape( 2, ndofs, 2, 2, *shape ).swapaxes(0,1) # follows ordering Vectorize
    for D, R2, logR, wf, I in iterdata:
      kernel = D[:,_,_] * D[_,:,_] * D[_,_,:]
      kernel /= -.5 * R2
      kernel[:,0,0] += D
      kernel[:,1,1] += D
      kernel[1,0,1] += D[0]
      kernel[1,1,0] -= D[0]
      kernel[0,0,1] -= D[1]
      kernel[0,1,0] += D[1]
      kernel /= (4 * numpy.pi * mu) * R2
      retval_swap[I] += numpy.tensordot( wf, kernel, (-1,-1) )
    return retval

class StokesletStress( function.ArrayFunc ):
  'stokeslet stress'

  def __init__( self, mycoords, topo, coords, funcsp ):
    'constructor'

    self.shape = int( funcsp.shape[0] ) * 2, 2, 2
    self.args = int(funcsp.shape[0]), IterData( mycoords, topo, coords, funcsp, funcsp.shape[0] )

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

  def __init__( self, mycoords, topo, coords, velo, trac, mu ):
    'constructor'

    self.shape = 2,
    assert velo.shape == (2,)
    assert trac.shape == (2,)
    self.args = IterData( mycoords, topo, coords, 1, velo, trac, coords.normal(topo) ), mu

  @staticmethod
  def eval( (shape,iterdata), mu ):
    'evaluate'

    retval = 0
    for D, R2, logR, w, velo, trac, norm in iterdata:
      D_R2 = D / R2
      Dtrac = util.contract( D_R2, trac[:,_,:], 0 )
      Dvelo = util.contract( D_R2, velo[:,_,:], 0 )
      Dnorm = util.contract( D_R2, norm[:,_,:], 0 )
      retval += numpy.dot( ( D * Dtrac - trac[:,_,:] * logR ) / (4*mu) - D * Dnorm * Dvelo, w )
    return retval / numpy.pi

def stokeslet_multidom( domains, coordss, funcsps, mu ):
  'create stokeslet on multiple domains'

  assert len(domains) == len(coordss) == len(funcsps)
  for mycoords, mydomain in zip( coordss, domains ):
    stokeslets = []
    tracs = []
    for domain, coords, funcsp in zip( domains, coordss, funcsps ):
      stokeslet = Stokeslet( mycoords, domain, coords, funcsp, mu )
      stress = StokesletStress( mycoords, domain, coords, funcsp )
      trac = ( stress * mycoords.normal( mydomain ) ).sum()
      stokeslets.append( stokeslet )
      tracs.append( trac if coords is not mycoords else AddPart( trac, .5 * funcsp.vector(2) ) )
    yield function.Stack( stokeslets ), function.Stack( tracs )

def stokeslet_reconstruct_multidom( mycoords, domains, coordss, velos, tracs, mu ):
  'create stokeslet on multiple domains'

  assert len(domains) == len(coordss) == len(velos) == len(tracs)
  return sum( StokesletReconstruct( mycoords, domain, coords, velo, trac, mu )
                for domain, coords, velo, trac in zip( domains, coordss, velos, tracs ) )

def testgrad( domain, coords, funcsp ):
  'numeric differentiation test'

  import mesh
  testdomain, testcoords = mesh.rectilinear( [0,1], [0,1] )
  elem, = testdomain
  eps = 1e-10
  p = elem.eval( numpy.array([[.5],[.5]]) )
  px = elem.eval( numpy.array([[.5-.5*eps,.5+.5*eps],[.5,.5]]) )
  py = elem.eval( numpy.array([[.5,.5],[.5-.5*eps,.5+.5*eps]]) )

  fval = Stokeslet( testcoords, domain, coords, funcsp, 1. )
  grad = StokesletGrad( testcoords, domain, coords, funcsp, 1. )

  dx = fval(px)
  print numpy.hstack([ ( dx[...,1] - dx[...,0] ) / eps, grad( p )[...,0,0] ])
  print ( dx[...,1] - dx[...,0] ) / eps - grad( p )[...,0,0]
  print

  dy = fval(py)
  print numpy.hstack([ ( dy[...,1] - dy[...,0] ) / eps, grad( p )[...,1,0] ])
  print ( dy[...,1] - dy[...,0] ) / eps - grad( p )[...,1,0]

  raise SystemExit

# vim:shiftwidth=2:foldmethod=indent:foldnestmax=2
