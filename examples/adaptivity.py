#! /usr/bin/env python3

from nutils import *
import fractions, unittest


class MakePlots( object ):

  def __init__( self, geom, exact, optimalrate ):
    self.geom = geom
    self.exact = exact
    self.index = 0
    self.ndofs = []
    self.error_exact = []
    self.error_estimate = []
    self.optimalrate = optimalrate

  def __call__( self, domain, sol, ndofs, error_estimate ):
    self.index += 1

    error_exact = domain['aoi'].integrate( self.exact - sol, geometry=self.geom, ischeme='gauss9' )
    log.user( 'error estimate: %.2e (%.1f%% accurate)' % ( error_estimate, 100.*error_estimate/error_exact ) )

    points, colors = domain.elem_eval( [ self.geom, sol ], ischeme='bezier9', separate=True )
    aoi = domain['aoi'].boundary.elem_eval( self.geom, ischeme='bezier2', separate=True )
    with plot.PyPlot( 'sol', index=self.index ) as plt:
      plt.mesh( points, colors )
      plt.colorbar()
      plt.segments( aoi )

    self.ndofs.append( ndofs )
    self.error_exact.append( error_exact )
    self.error_estimate.append( error_estimate )
    with plot.PyPlot( 'conv', index=self.index ) as plt:
      plt.loglog( self.ndofs, self.error_exact, 'k-^', label='exact' )
      plt.loglog( self.ndofs, self.error_estimate, 'k--', label='estimate' )
      plt.slope_marker( ndofs, min( error_exact, error_estimate ), slope=-self.optimalrate )
      plt.legend( loc=3, frameon=False )
      plt.grid()
      plt.xlabel( 'degrees of freedom' )
      plt.ylabel( 'error' )


def main(
    degree: 'number of elements' = 1,
    circle: 'use circular area of interest (default square)' = False,
    uniform: 'use uniform refinement (default adaptive)' = False,
    basistype: 'basis function' = 'std',
    nrefine: 'maximum allowed number of refinements' = 7,
    figures: 'create figures' = True,
  ):

  # construct domain
  verts = numpy.linspace( -1, 1, 7 )
  basetopo, geom = mesh.rectilinear( [verts,verts] )
  aoi = basetopo.trim( .04 - ((geom+.5)**2).sum(-1), maxrefine=5 ) if circle else basetopo[1:2,1:2]
  domain = ( basetopo.withboundary( outside=... )
           - basetopo[3:,:3].withboundary( inside=... ) ).withsubdomain( aoi=aoi )

  # construct exact sulution (used for boundary conditions and error evaluation)
  exact = ( geom**2 ).sum(-1)**(1./3) * function.sin( (2./3) * function.arctan2(-geom[1],-geom[0]) )
  flux = exact.ngrad( geom )

  # sanity check
  harmonicity = numpy.sqrt( domain.integrate( exact.laplace(geom)**2, geometry=geom, ischeme='gauss9' ) )
  log.info( 'exact solution lsqr harmonicity:', harmonicity )

  # prepare plotting
  makeplots = MakePlots( geom, exact, fractions.Fraction(2 if uniform else degree*3,3) ) if figures else lambda *args, **kwargs: None

  # start adaptive refinement
  for irefine in log.count( 'level', start=1 ):

    # construct, solve course domain primal/dual problem
    basis = domain.basis( basistype, degree=degree )
    laplace = function.outer( basis.grad(geom) ).sum(-1)
    matrix = domain.integrate( laplace, geometry=geom, ischeme='gauss5' )
    rhsprimal = domain.boundary['inside'].integrate( basis * flux, geometry=geom, ischeme='gauss99' )
    rhsdual = domain['aoi'].integrate( basis, geometry=geom, ischeme='gauss5' )
    cons = domain.boundary['outside'].project( exact, ischeme='gauss9', geometry=geom, onto=basis )
    lhsprimal = matrix.solve( rhsprimal, constrain=cons )
    lhsdual = matrix.solve( rhsdual, constrain=cons&0 )
    primal = basis.dot( lhsprimal )
    dual = basis.dot( lhsdual )

    # construct, solve refined domain primal/dual problem
    finedomain = domain.refined
    finebasis = finedomain.basis( basistype, degree=degree )
    finelaplace = function.outer( finebasis.grad(geom) ).sum(-1)
    finematrix = finedomain.integrate( finelaplace, geometry=geom, ischeme='gauss5' )
    finerhsdual = finedomain['aoi'].integrate( finebasis, geometry=geom, ischeme='gauss5' )
    finecons = finedomain.boundary['outside'].project( 0, ischeme='gauss5', geometry=geom, onto=finebasis )
    finelhsdual = finematrix.solve( finerhsdual, constrain=finecons )

    # evaluate error estimate
    dlhsdual = finelhsdual - finedomain.project( dual, onto=finebasis, geometry=geom, ischeme='gauss5' )
    ddualw = finebasis * dlhsdual
    error_est_w = finedomain.boundary['inside'].integrate( ddualw * flux, geometry=geom, ischeme='gauss99' )
    error_est_w -= finedomain.integrate( ( ddualw.grad(geom) * primal.grad(geom) ).sum(-1), geometry=geom, ischeme='gauss5' )

    # plot solution and error convergence
    makeplots( domain, primal, len(lhsprimal), error_estimate=abs(error_est_w).sum() )

    if irefine >= nrefine:
      break

    # refine mesh
    if uniform:
      domain = domain.refined
    else:
      mask = error_est_w**2 > numpy.mean(error_est_w**2)
      domain = domain.refined_by( elem.transform[:-1] for elem in domain.refined.supp( finebasis, mask ) )

  return lhsprimal, error_est_w


class test(unittest.TestCase):

  def test_p1_std(self):
    lhsprimal, error_est_w = main(degree=1, circle=False, uniform=False, basistype='std', nrefine=2, figures=False)
    numeric.assert_allclose64(lhsprimal,
      'eNpVkckNQzEIRBuyJfallijH9N9CbOYrUk5glgeMbbGvV0jXZ9syVb5WhGneQsfW2iegN7BPhY0TSlMS'
      'lTKl2TmtloParBafHafX2tGb3eitnN7tGTI1ShyIuICfYXCMBcAITNwV9izDDLJ3gVxEc8DucgDDWwBs'
      '3Ljb69mQ5RTrAO2/65diCbqOpZxMLmula9WS7+ZaUEoFUEmeccIyO7IBneanzlcGJkTI9Lk/SitBvs7J'
      's+tcVCwTT8Zh3tDwaDlcNZ08F0EKCaORtDvyOhU+md9kr7Fugo9jfK14gcB1kX5UrVQ4ehHH4dAbeX8B'
      'W2l7pA==')
    numeric.assert_allclose64(error_est_w,
      'eNp1VMmRGzEMTGhYhfuIxeWn80/BJADO7GdLD0EQzu4G5UF9/sDz+8cJ4d/Kh9nofMMTkfFvIW4ThX2c'
      'KNvCm7UcoMKXEEtFE2JUxGLPylpqkf0fR0djmFYQEnQMG1rHUN4OnAHd1lKkLfLUtuA0Ra7YTMopjFke'
      'Ya5WIqfBrqvJt26E8lRT4FsNGStD08u33LkBcHKb1EVgk7HISF4s0N2nMWIZIdDDhVnPZOJaBTkOyL0j'
      'kHU9FJJbWZTfyivIew60aGglsZda4YplfbX9gFa0sdNtI0I5psZE/i4GY7/Us0gzpVTr7AYMFw3cyPTI'
      'glbDJ9jPMYYnDY0LoNipne3OiVkyOjqSUrJvecK4cmoW9tKk3qKxmG6OWCiugJHh1nO0+Dxk0lwriHoB'
      'PKKD/FohWLzj+GU5QPTOW0Q2Q6ivcBYl6XCfByPatYhqtXDQ8phSldkCbcIiBftygnAGBK4YRp2RLfkT'
      'Atn7o6VXlyE4GEq6XwhZ9aMIL/EpP1S1ceVBKrFGRICeaHt6aJELtI9mzJrWbY9ERV16DxyJLdxqq0vy'
      '6Bh2KWSVqR3oKRe+4FfsW0QD8H5N7lrIo4x9l2PtQ8BZn0W/G4S50yVSquHHSqGnY56CZ8uQeYNY+ogo'
      'uabj0mA9QV7Nd+f6g/Q8KrEJIi2CtpLagdasbuY6o0iNg+9+SC8dCnPuS8mbDRrpW1DLGzHvctBd4e9/'
      'Be02XA==')

  def test_p2_spline(self):
    lhsprimal, error_est_w = main(degree=2, circle=False, uniform=False, basistype='spline', nrefine=1, figures=False)
    numeric.assert_allclose64(lhsprimal,
      'eNpVkFEOg0EEhC/0b4LFcpamj73/FWqNNOkTiZkPow/b83LJ/Cx9bOu+dUvKrcxst65qvJuatGKVVLvx'
      'TXQbN+uJWUQzlMEQzIsRw1APMOwcMMwadkS4WQzFTj+4I7GMVYeRDqqLgFHe3hdMXQ/Dq+Z9F2dAyDEO'
      'DWVYKYAvb2tTuPefnN+CW8BuNOtpvqcJJjbWr/Imvku1aUBdJxHvygoct5CVaxdHjw1nMkkfTmUfrakU'
      'qJs6k2eX0D/np2G5Ae33F29saGg=')
    numeric.assert_allclose64(error_est_w,
      'eNp9U0lu7TAMu5ADaB7OUnTZ+1/hy5Lyir8psgghDxRJWQ7q+YLz14cS/PMgHGORAnQy3LrCqd4AAbSX'
      'kLIqWf/AXnkoIgrheTTZe9PDnIPQzaaEJlO6lARic4Y5FiHktIEuuldjSCNiG/qHiGH2Sy4vM8H2ptM/'
      'J9wKHyb7JUXwvK0/VkeHAAWiS4K4JXblWzLBWSJEmCVnwGYkjW1CSQeZ03aIJe7DWe5AjipxWYaxGY4Q'
      'rXPxuqmIvna4++43zq2lcyOX5sCDiTy0Wkdb86V1ZH7JeGkJFFsQAvsAtOKPGpErJw6FvL50FnlUaKlZ'
      'bFDZ8QaGPqiFBm6/hPTeksBrYbbzpin37wo0HZA0yICdM0r7HJ7JESZqoBj8G6ZQxyMk6y/eVqM6jR1n'
      'ZK0bvML1O21FC9T9lLreqk5jJwKONayoXclAXm0zuF4uOTUBkbQK8rUL2WSdvbquSazv8aeO5UqKeTti'
      'vm9HmPGdfty4y4vRG0K2el291Vas7RfFjDIGbBij8VILjIQabXglxEZCvt6UiKBNVvcONd1mBPb5PqT9'
      'VO5UVdranJmwj1JpLjM3nIjcdy6hDagX6Gmf0P77vv8BTa37JQ==')

  def test_p1_std_circle(self):
    lhsprimal, error_est_w = main(degree=1, circle=True, uniform=False, basistype='std', nrefine=1, figures=False)
    numeric.assert_allclose64(lhsprimal,
      'eNpVj9sNQzEIQxdKpPAyMEvVz+6/QrkQVeoXCHNs0EW2XuDMz9alIvJUZuKnbiPoZxOtXaOWdu1oN5Bz'
      'ngaBgd0HVqNeYGgNUI3mIOrpw4Y3672iyzSt2YhJwXEaNnMmZjxsMe0W0Mk3oPNJ/OZ6XW8VR3HGza5J'
      '3LyS5vhteTC2mfNzhuFSsCtdnzSfR4i4zpM21H/qJxGjKHl/Aa51TUM=')
    numeric.assert_allclose64(error_est_w,
      'eNp9ksltRTEIRRt6SMxDLVGWv/8WYsD5STYRGwQXfADrQ/Z84POfkRq9II+XIvGC6pipv4Am7W4bLCa5'
      'wTag4OxCoMSpcyLewDj1QGRWO4x1pETPkdJkWKimGaFNf3ChGA2r/HknNXIBwbUJ2/VyvqLguoTCkust'
      'oKG0WpCxwxRyHo+TwKgBZfYdQYNtcUg3pYkbMVPt6lKr4fPw33wqpstEFN79zyaDLhKJyWaZyX/gMnet'
      'rLbNhHRGg9CcFRGHLK3bQBL6BlRZWmFV0xFM03e9tJmzHw+8S0vFCwOFXG/0czfPPYbmKDxrysV9r0TH'
      'G71zQ02hyCbP5FR4Gzv2jd6NVXd3EIFbBxZ9pcMIgr5aRvP7B0q/j4lMb7dH2E/If77EL/v8AhMYnBA=')


if __name__ == '__main__':
  cli.run(main)
