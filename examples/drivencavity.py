#! /usr/bin/env python3

from nutils import mesh, plot, cli, log, function, debug, _
import numpy


class MakePlots( object ):

  def __init__( self, domain, geom, dpres=.01 ):
    self.domain = domain
    self.geom = geom
    self.dpres = dpres
    self.index = 0

  def __call__( self, velo, pres ):
    self.index += 1
    points, velo, pres = self.domain.elem_eval( [ self.geom, velo, pres ], ischeme='bezier9', separate=True )
    with plot.PyPlot( 'flow', index=self.index, ndigits=4 ) as plt:
      tri = plt.mesh( points, mergetol=1e-5 )
      plt.tricontour( tri, pres, every=self.dpres, linestyles='solid', alpha=.333 )
      plt.colorbar()
      plt.streamplot( tri, velo, spacing=.01, linewidth=-10, color='k', zorder=9 )


def main( nelems=12, viscosity=1e-3, density=1, degree=2, warp=False, withplots=True, tol=1e-5, maxiter=numpy.inf ):

  Re = density / viscosity # based on unit length and velocity
  log.user( 'reynolds number: {:.1f}'.format(Re) )

  # construct mesh
  verts = numpy.linspace( 0, 1, nelems+1 )
  domain, geom = mesh.rectilinear( [verts,verts] )

  # construct bases
  vxbasis, vybasis, pbasis, lbasis = function.chain([
    domain.basis( 'spline', degree=(degree+1,degree), removedofs=((0,-1),None) ),
    domain.basis( 'spline', degree=(degree,degree+1), removedofs=(None,(0,-1)) ),
    domain.basis( 'spline', degree=degree ),
    [1], # lagrange multiplier
  ])
  if not warp:
    vbasis = function.stack( [ vxbasis, vybasis ], axis=1 )
  else:
    gridgeom = geom
    xi, eta = gridgeom
    geom = (eta+2) * function.rotmat(xi*.4)[:,1] - (0,2) # slight downward bend
    J = geom.grad( gridgeom )
    detJ = function.determinant( J )
    vbasis = ( vxbasis[:,_] * J[:,0] + vybasis[:,_] * J[:,1] ) / detJ # piola transform
    pbasis /= detJ
  stressbasis = (2*viscosity) * vbasis.symgrad(geom) - (pbasis)[:,_,_] * function.eye( domain.ndims )

  # construct matrices
  A = function.outer( vbasis.grad(geom), stressbasis ).sum([2,3]) \
    + function.outer( pbasis, vbasis.div(geom)+lbasis ) \
    + function.outer( lbasis, pbasis )
  Ad = function.outer( vbasis.div(geom) )
  stokesmat, divmat = domain.integrate( [ A, Ad ], geometry=geom, ischeme='gauss9' )

  # define boundary conditions
  normal = geom.normal()
  utop = function.asarray([ normal[1], -normal[0] ])
  h = domain.boundary.elem_eval( 1, geometry=geom, ischeme='gauss9', asfunction=True )
  nietzsche = (2*viscosity) * ( ((degree+1)*2.5/h) * vbasis - vbasis.symgrad(geom).dotnorm(geom) )
  stokesmat += domain.boundary.integrate( function.outer( nietzsche, vbasis ).sum(-1), geometry=geom, ischeme='gauss9' )
  rhs = domain.boundary['top'].integrate( ( nietzsche * utop ).sum(-1), geometry=geom, ischeme='gauss9' )

  # prepare plotting
  makeplots = MakePlots( domain, geom ) if withplots else lambda *args: None

  # start picard iterations
  lhs = stokesmat.solve( rhs, tol=tol, solver='cg', precon='spilu' )
  for iiter in log.count( 'picard' ):
    log.info( 'velocity divergence:', divmat.matvec(lhs).dot(lhs) )
    makeplots( vbasis.dot(lhs), pbasis.dot(lhs) )
    ugradu = ( vbasis.grad(geom) * vbasis.dot(lhs) ).sum(-1)
    convection = density * function.outer( vbasis, ugradu ).sum(-1)
    matrix = stokesmat + domain.integrate( convection, ischeme='gauss9', geometry=geom )
    lhs, info = matrix.solve( rhs, lhs0=lhs, tol=tol, info=True, precon='spilu', restart=999 )
    if iiter == maxiter-1 or info.niter == 0:
      break

  return rhs, lhs


def unittest():

  retvals = main( nelems=4, viscosity=1e-3, degree=1, warp=False, tol=0, maxiter=1, withplots=False )
  assert debug.checkdata( retvals, '''
    eNrFkl1uxCAMhK+zK0Hl8S8+UO5/hYKdbJ8q9a0SEpE9DJ+HYLx0wN7j9aKv0cvd45oxnIiu6ePTwF3I
    Pzd+tfr3tSeexsuuaUOD5OwTknxNHa4SpwA6Ah2T8/BvBUO0K2vR+eAgnF0TqxrKmi3NbCk8S4q0ktiK
    cnWgTW0BfWS5n46YF5BZlPkDOhVcSpYo0A8gcMh3Q7sgGdp3ohw9FPVC7Ded94gaKxoC8XAL2jPsHsBQ
    rR9gQwfmvFrykMI7qL2XAqFaXORVd0rvvkXXubhnhPABnDu+ukxcVjtA+0Q77QyiiVOWNwWoUtnD5i2V
    O0/SMpU7npmJ+hH3m7QrtHmygxUW6oCtAq4/5f0NMWaoJA==''' )

  retvals = main( nelems=4, viscosity=1e-3, degree=2, warp=False, tol=0, maxiter=1, withplots=False )
  assert debug.checkdata( retvals, '''
    eNrVU1mK3UAMvM4bsIP25UBz/yvEUvUbmAmE/AYMaloqqark5utlF/vH9XrRr+vrYyL6vOMKR/yWO3ed
    /kdODo51ov8z7m/z/uvvMfa2SBkzspIn3sxen/fju6ROvDk75yBsMvFBnEolQYUjc5szzaHMfKIaORJm
    b0zFHoo4JhVysNzAdOvOjYwat29t7YXoMwfd1JZic220ct5E0ZTGpct9IFHAsmdjjNm2L3Y9auToJsjU
    yFOJirfuL6s4RJDo7clUD0KvUM7JK+nG21JWdRCBnyWICmccbwl2mI9T9RwqtrmnFEqiV3Ua2foRdvy3
    ErhbDXEcP3zwNPjgKucggbZv/SKyDnHLipOGiHFsL6JlX4tRE9QxAIa8xuxt6HDpDM0SxTPDf8P57lwM
    +xSLTTukgmaPw87rkMH+pA4kCxD8WE9lkmFt0aAhrnFobS8V3kpmBhuH03UevzAHcLSjHgvXMtmd1dWk
    i9es9Wmfy8dvwCb0Ug==''' )

  retvals = main( nelems=4, viscosity=1e-3, degree=1, warp=True, tol=0, maxiter=1, withplots=False )
  assert debug.checkdata( retvals, '''
    eNrFUltuxDAIvM6uZCremAPl/leogU2kflT9rGSJhBlgGJvWSxfZe71e+LXmbES8IJalyAW+HoC4AF87
    J/4N/Nrq38/ZGFx1X2DLmKwikEZcoCslvRIevgdw6QSwuhUD2GLXBwdLRd3YEVSYmiqozQCy4KbSzorG
    2VT3MuUw8x5zmvE0G0EWU3kLBcndzdnMG3gqFVsghzYgW3sTRu9ZJMqFB3lOwbmrLlDrCNa7F+KUo1t9
    VpRsNY9e4qo9DA8ZN26hJNwySKLHUMTIsuz/EMEPzoOP0UCk05txMsfG/MFU1Sbo41rG7G796g6lTemS
    2UR33ZrXssYzBvFDGYNTylg/Nm2dqdJ5dtfapx/K+xso/qhh''' )


if __name__ == '__main__':
  cli.run( main, unittest )
