#! /usr/bin/env python3

from nutils import *


def makeplots( name, domain, geom, stress ):
  sigma_dev = stress - (function.trace(stress)/domain.ndims) * function.eye(domain.ndims)
  vonmises = function.sqrt( ( sigma_dev**2 ).sum([0,1]) * 1.5 ) # TODO check fix for 2D
  points, colors = domain.simplex.elem_eval( [ geom, vonmises ], ischeme='bezier3', separate=True )
  with plot.PyPlot( name ) as plt:
    plt.mesh( points, colors, triangulate='bezier' )
    plt.colorbar()
    plt.axis( 'equal' )
    plt.clim( 0, 1 )


def main(
    nelems: 'number of elements' = 12,
    lmbda: 'first lamé constant' = 1.,
    mu: 'second lamé constant' = 1.,
    angle: 'bend angle (degrees)' = 20,
    restol: 'residual tolerance' = 1e-10,
    trim: 'create circular-shaped hole' = False,
    plots: 'create plots' = True,
  ):

  verts = numpy.linspace( 0, 1, nelems+1 )
  domain, geom0 = mesh.rectilinear( [verts,verts] )
  if trim:
    levelset = function.norm2( geom0 - (.5,.5) ) - .2
    domain = domain.trim( levelset, maxrefine=2 )

  basis = domain.basis( 'spline', degree=2 ).vector( domain.ndims )
  phi = angle * numpy.pi / 180
  a = numpy.cos(phi) - 1, -numpy.sin(phi)
  b = numpy.sin(2*phi), numpy.cos(2*phi) - 1
  x, y = geom0
  cons = domain.boundary['left,right'].project( x*(a+y*b), onto=basis, ischeme='gauss6', geometry=geom0 )

  dofs = function.DerivativeTarget( [len(basis)] )
  disp = basis.dot( dofs )
  geom = geom0 + disp
  eye = function.eye(len(geom))

  strain = disp.symgrad( geom0 )
  stress = lmbda * function.trace(strain) * eye + (2*mu) * strain
  residual = model.Integral( basis['ni,j'] * stress['ij'], domain=domain, geometry=geom0, degree=7 )

  lhs0 = model.solve_linear( dofs, residual=residual, constrain=cons )
  if plots:
    makeplots( 'linear', domain, function.replace(dofs,lhs0,geom), function.replace(dofs,lhs0,stress) )

  strain = .5 * eye - .5 * ( geom0.grad(geom)[:,:,_] * geom0.grad(geom)[:,_,:] ).sum(0)
  stress = lmbda * function.trace(strain) * eye + (2*mu) * strain
  residual = model.Integral( basis['ni,j'] * stress['ij'], domain=domain, geometry=geom, degree=7 )

  lhs1 = model.newton( dofs, residual=residual, lhs0=lhs0, freezedofs=cons.where ).solve( tol=restol )
  if plots:
    makeplots( 'linear', domain, function.replace(dofs,lhs1,geom), function.replace(dofs,lhs1,stress) )

  return lhs0, lhs1


def unittest():

  retvals = main( nelems=4, angle=10, plots=False )
  assert debug.checkdata( retvals, '''
    eNqtU0luBDEI/M5EsiP25UHz/y/EBvqS5JZII1FjFxiKalwvWagf6/WCz/XttzGD39sWukdFZHhvXYQp
    NzKQ36h4eXoSUPUCI/WbwGFFcBC6ERHwvc+LjHnjRseoC9YimF/C+U/cxAC5kZCsE4zraUyrxHTqBGcv
    ot6CJyZQJ+h96vTsat3CVM6oihRS/5msEn/KwK41vSJVZO+4USAbEFf3mwm0haiqF1j2CZFYc9hKxK3q
    DcxHPvPsOhbUwFlKwB0eUxAu+Q5GPeltAxtwwJzYXBlKg0BtkDrA2Z6rUSqtK9PRqAFZpxNLVT4++cUm
    gnF08KUQtfWzzOqVqHvm81bZBGYG0+sjX2haSxTHEsmtCYhUz2EVkNLsVIiz3SwRPXVsQ9UpetRUBNyz
    mGM5l6GlRbBsP/VopNLbd8V/sgkaQE2vUCpshvbNzpD2CUHieMCzXfGYgZTGHiJ9JeGPK6BPTmsyJ5N+
    bhqEUI49ng+FaD4x5nEF5ayelZ+hxycOw0lqzjGy/cUnH19iCe5m''' )


if __name__ == '__main__':
  cli.run( main, unittest )
