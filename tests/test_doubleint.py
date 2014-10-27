#!/usr/bin/env python

from nutils import *
import numpy, warnings
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
  # Gauss quadrature on product domain.

  def test_polynomials( self ):
    domain, geom = mesh.rectilinear( [[0,.5,1]] )
    ddomain = domain * domain
    iwscale = function.iwdscale(geom,domain.ndims)

    x = geom[0]
    y = function.opposite( geom[0] )

    numpy.testing.assert_almost_equal(
      ddomain.integrate( x-y, iwscale=iwscale, ischeme='gauss2' ), 0 )

    numpy.testing.assert_almost_equal(
      ddomain.integrate( (x-y)**2, iwscale=iwscale, ischeme='gauss2' ), 1./6 )

class TestSingularDoubleInt( object ):
  # Regularized quadrature on product domain.

  def patch( self, val ):
    numpy.testing.assert_almost_equal( val, 1 )

  def distance( self, val ):
    numpy.testing.assert_almost_equal( val, 1./3 )

  def test_Integration( self ):
    grid = numpy.linspace( 0., 1., 4 )
    domain, geom = mesh.rectilinear( 2*(grid,) )
    ddomain = domain * domain
    iwscale = function.iwdscale(geom,domain.ndims)

    x = geom
    y = function.opposite( geom[0] )
    r = function.norm2( x-y )
    
    self.patch( ddomain.integrate( 1, iwscale=iwscale, ischeme='singular2' ) )
    self.distance( ddomain.integrate( r**2, iwscale=iwscale, ischeme='singular3' ) )

class TestNormalInKernelOfV( object ):
  # Convolute normal with single-layer to verify it is in the kernel, note that
  # this works with all gauss schemes!'

  def __init__( self ):
    'Geometry definitions.'
    domain, geom = mesh.rectilinear( (grid(4),grid(2)), periodic=(0,) )
    self.geom, self.domain, self.ddomain = geom, domain, domain * domain

  def template( self, degree, geometry, dump=False ):
    'Template for Vn = 0 tests on different geometries.'
    trac = self.domain.splinefunc( degree=2*(2,) ).vector(3)
    if dump:
      geo = domain.projection( geometry, onto=trac, geometry=self.geom )
      refine = 3 if geometry is sphere else 0
      plot.writevtu( 'geometry.vtu', domain.refine(3), geometry )
      if refine: warnings.warn( 'The plotted geometry is a projection, and a bad approximation.' )

    iwscale = function.iwdscale(geometry,self.domain.ndims)

    x = geometry
    y = function.opposite( geometry )

    tmp = self.ddomain.integrate( (V(x,y)).sum(), iwscale=iwscale, ischeme='singular{0}'.format(degree) )
    print( tmp )

    return self.ddomain.integrate( (V(x,y)*x.normal()).sum(), iwscale=iwscale, ischeme='singular{0}'.format(degree) )

  def test_SphericalGeometry( self ):
    # n in ker(V), case: sphere.
    cos, sin, pi = function.cos, function.sin, numpy.pi
    phi, theta = .5*pi*self.geom # |phi| < pi, |theta| < pi/2
    self.sphere = function.stack( [cos(phi)*cos(theta), sin(phi)*cos(theta), sin(theta)] )

    err = self.template( 4, self.sphere )
    numpy.testing.assert_almost_equal( infnorm(err), 0., decimal=3 )

  def test_PolyhedronGeometry( self ):
    # n in ker(V), case: polyhedron
    # raise NotImplemented( 'Choose has no .opposite yet' )

    abs = function.abs
    xi, eta = self.geom
    self.octahedron = function.Concatenate( [[(1.-abs( eta ))*function.piecewise( xi, (-1., 0., 1.), 1, -2*xi-1, -1, 2*xi-3 )], [(1.-abs( eta ))*function.piecewise( xi, (-1., 0., 1.), 2*xi+3, 1, 1-2*xi, -1 )], [numpy.sqrt(2)*eta]] )

    err = self.template( 4, self.octahedron )
    numpy.testing.assert_almost_equal( infnorm( err ), 0, decimal=3 )

class TestKroneckerKernelGivesSurface( object ):
  # Convolute a uniform velocity field with the identity to verify it gives the
  # surface.

  def test_SphericalGeometry( self ):
    #Integrating Id gives surf, case: sphere.

    domain, geom = mesh.rectilinear( (grid(4),grid(2)), periodic=(0,) )
    cos, sin, pi = function.cos, function.sin, numpy.pi
    phi, theta = .5*pi*geom # |phi| < pi, |theta| < pi/2
    sphere = function.stack( [cos(phi)*cos(theta), sin(phi)*cos(theta), sin(theta)] )
    velo = domain.splinefunc( degree=2*(2,) ).vector(3)
    vinf = function.stack( (0.,0.,1.) )
    val = domain.integrate( (velo*vinf).sum(-1), geometry=sphere, ischeme='gauss8' ).sum()

    surf = 4.*pi
    numpy.testing.assert_almost_equal( val, surf )

class TestOneInKernelOfK( object ):
  # Convolute a uniform velocity field with the dual-layer to verify it is in
  # the kernel.

  def __init__( self ):
    'Geometry definitions.'
    domain, geom = mesh.rectilinear( (grid(4),grid(2)), periodic=(0,) )
    self.geom, self.domain, self.ddomain = geom, domain, domain * domain

  def template( self, geometry, degree ):
    trac = self.domain.splinefunc( degree=2*(2,) ).vector(3)
    vinf = function.stack( (0.,0.,1.) )
    iwscale = function.iwdscale( geometry, self.domain.ndims )

    x = geometry
    y = function.opposite( geometry )

    Kvinf = (K(x, y)[:,:]*vinf[_,:]).sum(-1)
    doublelayer = self.ddomain.integrate( (trac[:,:]*Kvinf[_,:]).sum(), iwscale=iwscale, ischeme='singular{0}'.format(degree) )
    identity = self.domain.integrate( (trac*vinf).sum(), geometry=geometry, ischeme='gauss4' )

    return .5*identity + doublelayer

  def test_SphericalGeometry( self ):
    # 1 in ker(K), case: sphere.

    cos, sin, pi = function.cos, function.sin, numpy.pi
    phi, theta = .5*pi*self.geom # |phi| < pi, |theta| < pi/2
    self.sphere = function.stack( [cos(phi)*cos(theta), sin(phi)*cos(theta), sin(theta)] )

    err = infnorm( self.template( self.sphere, 4 ) )
    numpy.testing.assert_almost_equal( err, 0, decimal=2 )

class TestShearFlow( object ):
  # Torus cut of shear flow.

  def test_InteriorProblem( self, N=4 ):
    # Interior Dirichlet on torus, shear flow.
    # domain, space
    grid = numpy.linspace( -2., 2., N+1 )
    domain, geom = mesh.rectilinear( [grid,grid], periodic=(0,1) )
    ddomain = domain * domain
    funcsp = domain.splinefunc( degree=4 ).vector(3)
    
    # geometry
    R, r = 3, 1
    phi, theta = .5*pi*geom
    torus = function.stack( [
        (r*function.cos(theta) + R)*function.cos(phi),
        (r*function.cos(theta) + R)*function.sin(phi),
         r*function.sin(theta)] )
  
    dtorus = function.concatenate([ torus, function.opposite(torus) ])

    # boundary data
    velo_shear = function.stack( [torus[2], 0., 0.] )
    trac_shear = function.stack( [torus.normal()[2], 0., torus.normal()[0]] )
    assert numpy.abs( domain.integrate( (velo_shear*torus.normal()).sum(-1), geometry=torus, ischeme='gauss2' ) ) < 1.e-12, 'int v.n = 0 condition violated.'
  
    x = torus
    y = function.opposite( x )
    
    rhs = 0.5*domain.integrate( (funcsp*velo_shear).sum(-1), geometry=x, ischeme='gauss4' ) \
        + ddomain.integrate( (funcsp*(K(x,y)*function.opposite(velo_shear)).sum()).sum(),
          geometry=dtorus, ischeme='singular5', title='bem[K]' )
    mat = ddomain.integrate_symm( (funcsp*(V(x,y)*function.opposite(funcsp)[:,_,_,:]).sum()).sum(),
          geometry=dtorus, ischeme='singular3', force_dense=True, title='bem[V]' )
    lhs = mat.solve( rhs, tol=1.e-8 )
    trac = funcsp.dot(lhs)
    trac_err, surf = domain.integrate( ((trac-trac_shear)**2, 1), geometry=x, ischeme='gauss6' )
    err = numpy.sqrt( trac_err.sum() )/surf
    numpy.testing.assert_almost_equal( err, 0, decimal=2 ), 'err = %.3e'%err

def main( N=8 ):
  a = TestShearFlow()
  a.test_InteriorProblem( N=N )
  raw_input( 'hit any key to exit' )

if __name__ == '__main__':
  util.run( main )

# vim:shiftwidth=2:foldmethod=indent:foldnestmax=2
