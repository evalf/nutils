#! /usr/bin/env python3

from nutils import mesh, util, cli, log, function, plot, debug, _
import numpy


class MakePlots( object ):

  def __init__( self, domain, geom, timestep, bbox=((-2,6),(-3,3)), video=False ):
    self.bbox = numpy.asarray(bbox)
    self.plotdomain = domain.select( function.piecewise( geom[0], self.bbox[0], 0, 1, 0 ) * function.piecewise( geom[1], self.bbox[1], 0, 1, 0 ), 'bezier3' )
    self.geom = geom
    self.plt = video and plot.PyPlotVideo( 'flow' )
    self.every = .01
    self.index = 0
    self.timestep = timestep
    self.spacing = .075
    self.xy = util.regularize( self.bbox, self.spacing )

  def __call__( self, velo, pres, angle ):
    self.index += 1
    points, velo, flow, pres = self.plotdomain.elem_eval( [ self.geom, velo, function.norm2(velo), pres ], ischeme='bezier9', separate=True )
    with self.plt if self.plt else plot.PyPlot( 'flow', index=self.index ) as plt:
      plt.axes( [0,0,1,1], yticks=[], xticks=[], frame_on=False )
      tri = plt.mesh( points, flow, mergetol=1e-5 )
      plt.clim( 0, 1.5 )
      plt.tricontour( tri, pres, every=self.every, cmap='gray', linestyles='solid', alpha=.8 )
      uv = plot.interpolate( tri, self.xy, velo )
      plt.vectors( self.xy, uv, zorder=9, pivot='mid', stems=False )
      plt.plot( 0, 0, 'k', marker=(3,2,angle*180/numpy.pi-90), markersize=20 )
      plt.xlim( self.bbox[0] )
      plt.ylim( self.bbox[1] )
    self.xy = util.regularize( self.bbox, self.spacing, self.xy + uv * self.timestep )


def main(
    nelems: 'number of elements' = 12,
    viscosity: 'fluid viscosity' = 1e-2,
    density: 'fluid density' = 1,
    tol: 'solver tolerance' = 1e-12,
    rotation: 'cylinder rotation speed' = 0,
    timestep: 'time step' = 1/24,
    maxradius: 'approximate domain size' = 25,
    tmax: 'end time' = numpy.inf,
    withplots: 'create plots' = True,
  ):

  uinf = numpy.array([ 1, 0 ])
  log.user( 'reynolds number:', density/viscosity )

  # construct mesh
  rscale = numpy.pi / nelems
  melems = numpy.ceil( numpy.log(2*maxradius) / rscale ).astype( int )
  log.info( 'creating {}x{} mesh, outer radius {:.2f}'.format( melems, 2*nelems, .5*numpy.exp(rscale*melems) ) )
  domain, gridgeom = mesh.rectilinear( [range(melems+1),numpy.linspace(0,2*numpy.pi,2*nelems+1)], periodic=(1,) )
  rho, phi = gridgeom
  phi += 1e-3 # tiny nudge (0.057 deg) to break element symmetry
  cylvelo = -.5 * rotation * function.trignormal(phi)
  radius = .5 * function.exp( rscale * rho )
  geom = radius * function.trigtangent(phi)

  # prepare bases
  J = geom.grad( gridgeom )
  detJ = function.determinant( J )
  vnbasis, vtbasis, pbasis = function.chain([ # compatible spaces using piola transformation
    domain.basis( 'spline', degree=(3,2), removedofs=((0,),None) )[:,_] * J[:,0] / detJ,
    domain.basis( 'spline', degree=(2,3) )[:,_] * J[:,1] / detJ,
    domain.basis( 'spline', degree=2 ) / detJ,
  ])
  vbasis = vnbasis + vtbasis
  stressbasis = (2*viscosity) * vbasis.symgrad(geom) - pbasis[:,_,_] * function.eye( domain.ndims )

  # prepare matrices
  A = function.outer( vbasis.grad(geom), stressbasis ).sum([2,3]) + function.outer( pbasis, vbasis.div(geom) )
  Ao = function.outer( vbasis, density * ( vbasis.grad(geom) * uinf ).sum(-1) ).sum(-1)
  At = (density/timestep) * function.outer( vbasis ).sum(-1)
  Ad = function.outer( vbasis.div(geom) )
  stokesmat, uniconvec, inertmat, divmat = domain.integrate( [ A, Ao, At, Ad ], geometry=geom, ischeme='gauss9' )

  # interior boundary condition (weak imposition of shear verlocity component)
  inner = domain.boundary['left']
  h = inner.elem_eval( 1, geometry=geom, ischeme='gauss9', asfunction=True )
  nietzsche = (2*viscosity) * ( (7.5/h) * vbasis - vbasis.symgrad(geom).dotnorm(geom) )
  B = function.outer( nietzsche, vbasis ).sum(-1)
  b = ( nietzsche * cylvelo ).sum(-1)
  bcondmat, rhs = inner.integrate( [ B, b ], geometry=geom, ischeme='gauss9' )
  stokesmat += bcondmat

  # exterior boundary condition (full velocity vector imposed at inflow)
  inflow = domain.boundary['right'].select( -( uinf * geom.normal() ).sum(-1), ischeme='gauss1' )
  cons = inflow.project( uinf, onto=vbasis, geometry=geom, ischeme='gauss9', tol=1e-12 )

  # initial condition (stationary oseen flow)
  lhs = (stokesmat+uniconvec).solve( rhs, constrain=cons, tol=0 )

  # prepare plotting
  if not withplots:
    makeplots = lambda *args: None
  else:
    makeplots = MakePlots( domain, geom, video=withplots=='video', timestep=timestep )
    mesh_ = domain.elem_eval( geom, ischeme='bezier5', separate=True )
    inflow_ = inflow.elem_eval( geom, ischeme='bezier5', separate=True )
    with plot.PyPlot( 'mesh', ndigits=0 ) as plt:
      plt.mesh( mesh_ )
      plt.rectangle( makeplots.bbox[:,0], *( makeplots.bbox[:,1] - makeplots.bbox[:,0] ), ec='green' )
      plt.segments( inflow_, colors='red' )

  # start time stepping
  stokesmat += inertmat
  precon = (stokesmat+uniconvec).getprecon( 'splu', constrain=cons )
  for istep in log.count( 'timestep', start=1 ):
    log.info( 'velocity divergence:', divmat.matvec(lhs).dot(lhs) )
    convection = density * ( vbasis.grad(geom) * vbasis.dot(lhs) ).sum(-1)
    matrix = stokesmat + domain.integrate( function.outer( vbasis, convection ).sum(-1), ischeme='gauss9', geometry=geom )
    lhs = matrix.solve( rhs + inertmat.matvec(lhs), lhs0=lhs, constrain=cons, tol=tol, restart=9999, precon=precon )
    makeplots( vbasis.dot(lhs), pbasis.dot(lhs), istep*timestep*rotation )
    if istep*timestep > tmax:
      break

  return lhs
  

def unittest():

  retvals = main( nelems=3, viscosity=1e-2, timestep=.1, tmax=.05, rotation=0, withplots=False )
  assert debug.checkdata( retvals, '''
    eNo1kteNxDAMRNvZBWyAORR0/bdwTOsfjskhJT0Jn488qN/nww769+rzkqKMQA5cEUotGCE2AWYtXLOi
    PK+jRQtW34QpYgvLmPiKuI5DKP9eft4SMQLZfCxkNhmykI5IAF1A4jJQOYl0hHhnuhdwSghMHWsV3FaX
    a5EVJBq/DEwLeXYkzvknQFhDEo9gt7wM2lkmssashu6LLMN4weSCqR0PscAe3gX1cbAiLLGQEQyke/rY
    iiMtQi7vVpJXMMQiI8lpKlI0kQO6wIDXIogLqPa459flTrQzityAIRRbp8ad3/iObTDIUC+GTgHFaQ1x
    w6XUrgJ6pGwY/goVdjgF5AIx1RM9zapUW9oHmGIrZvMlLHjNEnJPkjl/5s2I8gmMZU7SHhnzMWZbUZZ7
    c25H+2em+hYyxpFhOJgKeaJvs4X4gK/r7GssEWojLK9irtttGDM4QMbBDLxsltWrGcM3Hef9E9NeBLmO
    4/sPCbTLPw==''' )

  retvals = main( nelems=3, viscosity=1e-2, timestep=.1, tmax=.05, rotation=1, withplots=False )
  assert debug.checkdata( retvals, '''
    eNo1ktltADEIRNtJJFuC4TAuKP23EHPsl2e5Fh7w+tHF9rt++JD8bVubr1gL+CkB83LpZS9DmD7hy3Ff
    hK5tN24KcbIynKNIEbhtUHWviFfkb8vaT1gJVirPhmcHsuDwfBlcNRh4KciGmEro0dO5JGVhyoawVIM6
    9eikaAuoxWfpFJybL+TWN2iK84WUkON3LOwTUq9YoErcLP4aVKrpn79wQLmGFoDK76e+H7WBwVYJOgX2
    5VOomUIHSpfaOMbFXAhF5xIs2ZvI/Exv2s0Pdwb8dg3uSfcbEQVWpd5Hq0fHtwLTgeMyqC0beiOalIfD
    +zVCB0QHvFmiBciGkhe/z/Geb0fIScA5dB0aRQvibp1pAIknIFvumeNJmoaCf8d50DkP4twg1XUGS8Vy
    BA2Qoaxm0qEqfXKCvIqsYuEfihGiDQeUbZbQmOXcjnHnjjmQ2ptHL+nd/+0lHPJKOtfqZEVsbgd9mE5h
    jcf6DtV6BQgUz99/q5rJZA==''' )


if __name__ == '__main__':
  cli.run( main, unittest )
