#!/usr/bin/env python

from nutils import *
import numpy, warnings
almostEquals = lambda val, places=7: numpy.abs( val ) < 10.**(-places)
infnorm = lambda f: numpy.linalg.norm( f, numpy.inf )
grid = lambda n: numpy.linspace( -n/2., n/2., n+1 )
pi = numpy.pi
def V( x, y ):
  rInv = function.norm2( x-y )**-1.
  return 0.125*pi**-1. * (function.eye(3)*rInv + (x-y)[:,_]*(x-y)[_,:]*rInv**3)
def K( x, y ):
  rInv = function.norm2( x-y )**-1.
  return 0.75*pi**-1. * (x-y)[:,_]*(x-y)[_,:] * ((x-y)*y.normal()).sum() * rInv**5

class TestGaussDoubleInt( object ):
  'Gauss quadrature on product domain.'
  def distance( self, val ):
    assert almostEquals( val - 0. )

  def distancesquared( self, val ):
    assert almostEquals( val - 1./6 )

  def test_polynomials( self ):
    domain, coords = mesh.rectilinear( [[0,.5,1]] )
    ddomain = domain * domain

    iw = function.iwscale( coords, domain.ndims )
    iweights = iw * function.opposite( iw ) * function.IWeights()

    x = coords[0]
    y = function.opposite( coords[0] )

    self.distance( ddomain.integrate( x-y, iweights=iweights, ischeme='gauss2' ) )
    self.distancesquared( ddomain.integrate( (x-y)**2, iweights=iweights, ischeme='gauss2' ) )

class TestSingularDoubleInt( object ):
  'Regularized quadrature on product domain.'
  def patch( self, val ):
    assert almostEquals( val - 1. )

  def distance( self, val ):
    assert almostEquals( val - 1./3 )

  def test_Integration( self ):
    grid = numpy.linspace( 0., 1., 4 )
    domain, coords = mesh.rectilinear( 2*(grid,) )
    ddomain = domain * domain

    iw = function.iwscale( coords, domain.ndims )
    iweights = iw * function.opposite( iw ) * function.IWeights()

    x = coords
    y = function.opposite( coords[0] )
    r = function.norm2( x-y )
    
    self.patch( ddomain.integrate( 1, iweights=iweights, ischeme='singular2' ) )
    self.distance( ddomain.integrate( r**2, iweights=iweights, ischeme='singular3' ) )

class TestNormalInKernelOfV( object ):
  'Convolute normal with single-layer to verify it is in the kernel, note that this works with all gauss schemes!'
  def __init__( self ):
    'Geometry definitions.'
    domain, coords = mesh.rectilinear( (grid(4),grid(2)), periodic=(0,) )
    self.coords, self.domain, self.ddomain = coords, domain, domain * domain

  def template( self, degree, geometry, dump=False ):
    'Template for Vn = 0 tests on different geometries.'
    trac = self.domain.splinefunc( degree=2*(3,) ).vector(3)
    if dump:
      geo = domain.projection( geometry, onto=trac, coords=self.coords )
      refine = 3 if geometry is sphere else 0
      plot.writevtu( 'geometry.vtu', domain.refine(3), geometry )
      if refine: warnings.warn( 'The plotted geometry is a projection, and a bad approximation.' )

    iw = function.iwscale( geometry, self.domain.ndims )
    iweights = iw * function.opposite( iw ) * function.IWeights()

    x = geometry
    y = function.opposite( geometry )

    return self.ddomain.integrate( (V(x,y)*x.normal()).sum(), iweights=iweights, ischeme='singular{0}'.format(degree) )

  def test_SphericalGeometry( self ):
    'n in ker(V), case: sphere.'
    cos, sin, pi = function.cos, function.sin, numpy.pi
    phi, theta = .5*pi*self.coords # |phi| < pi, |theta| < pi/2
    self.sphere = function.stack( [cos(phi)*cos(theta), sin(phi)*cos(theta), sin(theta)] )

    err = self.template( 4, self.sphere )
    assert almostEquals( infnorm(err) - 0., places=3 )

  def test_PolyhedronGeometry( self ):
    'n in ker(V), case: polyhedron'
    # raise NotImplemented( 'Choose has no .opposite yet' )

    abs = function.abs
    xi, eta = self.coords
    self.octahedron = function.Concatenate( [[(1.-abs( eta ))*function.piecewise( xi, (-1., 0., 1.), 1, -2*xi-1, -1, 2*xi-3 )], [(1.-abs( eta ))*function.piecewise( xi, (-1., 0., 1.), 2*xi+3, 1, 1-2*xi, -1 )], [numpy.sqrt(2)*eta]] )

    err = self.template( 4, self.octahedron )
    assert almostEquals( infnorm( err ) - 0., places=3 )

class TestKroneckerKernelGivesSurface( object ):
  'Convolute a uniform velocity field with the identity to verify it gives the surface.'

  def test_SphericalGeometry( self ):
    'Integrating Id gives surf, case: sphere.'
    domain, coords = mesh.rectilinear( (grid(4),grid(2)), periodic=(0,) )
    cos, sin, pi = function.cos, function.sin, numpy.pi
    phi, theta = .5*pi*coords # |phi| < pi, |theta| < pi/2
    sphere = function.stack( [cos(phi)*cos(theta), sin(phi)*cos(theta), sin(theta)] )
    velo = domain.splinefunc( degree=2*(3,) ).vector(3)
    vinf = function.stack( (0.,0.,1.) )
    val = domain.integrate( (velo*vinf).sum(-1), coords=sphere, ischeme='gauss5' ).sum()

    surf = 4.*pi
    assert almostEquals( val-surf )

class TestOneInKernelOfK( object ):
  'Convolute a uniform velocity field with the dual-layer to verify it is in the kernel.'
  def __init__( self ):
    'Geometry definitions.'
    domain, coords = mesh.rectilinear( (grid(4),grid(2)), periodic=(0,) )
    self.coords, self.domain, self.ddomain = coords, domain, domain * domain

  def template( self, geometry, degree ):
    trac = self.domain.splinefunc( degree=2*(3,) ).vector(3)
    vinf = function.stack( (0.,0.,1.) )

    iw = function.iwscale( geometry, self.domain.ndims )
    iweights = iw * function.opposite( iw ) * function.IWeights()

    x = geometry
    y = function.opposite( geometry )

    Kvinf = (K(x, y)[:,:]*vinf[_,:]).sum(-1)
    doublelayer = self.ddomain.integrate( (trac[:,:]*Kvinf[_,:]).sum(), iweights=iweights, ischeme='singular{0}'.format(degree) )
    identity = self.domain.integrate( (trac*vinf).sum(), coords=geometry, ischeme='gauss3' )

    return .5*identity + doublelayer

  def test_SphericalGeometry( self ):
    '1 in ker(K), case: sphere.'
    cos, sin, pi = function.cos, function.sin, numpy.pi
    phi, theta = .5*pi*self.coords # |phi| < pi, |theta| < pi/2
    self.sphere = function.stack( [cos(phi)*cos(theta), sin(phi)*cos(theta), sin(theta)] )

    err = infnorm( self.template( self.sphere, 4 ) )
    assert almostEquals( err - 0., places=2 )

class TestShearFlow( object ):
  'Torus cut of shear flow.'
  def test_InteriorProblem( self, N=4 ):
    'Interior Dirichlet on torus, shear flow.'
    # domain, space
    cos, sin = function.cos, function.sin
    grid = lambda n: numpy.linspace( -2., 2., n+1 )
    # domain, coords = mesh.rectilinear( 2*(grid(4),), periodic=(0,1) )
    domain, coords = mesh.rectilinear( 2*(grid(N),), periodic=(0,1) )
    ddomain = domain*domain
    funcsp = domain.splinefunc( degree=2*(5,) ).vector(3)
    
    # geometry
    R, r = 3, 1
    phi, theta = .5*pi*coords
    torus = function.stack( [
        (r*cos(theta) + R)*cos(phi),
        (r*cos(theta) + R)*sin(phi),
         r*sin(theta)] )
  
    # boundary data
    velo_shear = function.stack( [torus[2], 0., 0.] )
    trac_shear = function.stack( [torus.normal()[2], 0., torus.normal()[0]] )
    assert numpy.abs( domain.integrate( (velo_shear*torus.normal()).sum(-1), coords=torus, ischeme='gauss2' ) ) < 1.e-12, 'int v.n = 0 condition violated.'
  
    l2norm = lambda self, func: numpy.sqrt( self.domain.integrate( func**2, coords=self.torus, ischeme='gauss4' ).sum() )
  
    iw = function.iwscale( torus, domain.ndims )
    iweights = iw * function.opposite( iw ) * function.IWeights()
    x = torus
    y = function.opposite( x )
    
    rhs = 0.5*domain.integrate( (funcsp*velo_shear).sum(-1), coords=x, ischeme='gauss3' ) \
        + ddomain.integrate( (funcsp*(K(x,y)*function.opposite(velo_shear)).sum()).sum(),
          iweights=iweights, ischeme='singular5', title='bem[K]' )
    mat = ddomain.integrate_symm( (funcsp*(V(x,y)*function.opposite(funcsp)[:,_,_,:]).sum()).sum(),
          iweights=iweights, ischeme='singular3', force_dense=True, title='bem[V]' )
    lhs = mat.solve( rhs, tol=1.e-8 )
    trac = funcsp.dot(lhs)
    trac_err, surf = domain.integrate( ((trac-trac_shear)**2, 1), coords=x, ischeme='gauss4' )
    err = numpy.sqrt( trac_err.sum() )/surf
    assert almostEquals( err, places=2 ), 'err = %.3e'%err

def main( N=8 ):
  a = TestShearFlow()
  a.test_InteriorProblem( N=N )
  raw_input( 'hit any key to exit' )

util.run( main )
# vim:shiftwidth=2:foldmethod=indent:foldnestmax=2
