from nutils import model, mesh, function
from . import register, unittest
import numpy


@register
def laplace():
  domain, geom = mesh.rectilinear( [8,8] )
  basis = domain.basis( 'std', degree=1 )
  cons = domain.boundary['left'].project( 0, onto=basis, geometry=geom, ischeme='gauss2' )
  dofs = function.DerivativeTarget( [len(basis)] )
  u = basis.dot( dofs )
  residual = model.Integral( ( basis.grad(geom) * u.grad(geom) ).sum(-1), domain=domain, geometry=geom, degree=2 ) \
           + model.Integral( basis, domain=domain.boundary['top'], geometry=geom, degree=2 )

  @unittest
  def res():
    lhs = model.solve_linear( dofs, residual=residual, constrain=cons )
    res = residual.replace( dofs, lhs ).eval()
    resnorm = numpy.linalg.norm( res[~cons.where] )
    assert resnorm < 1e-13


@register
def navierstokes():
  domain, geom = mesh.rectilinear( [numpy.linspace(0,1,9)] * 2 )
  ubasis, pbasis = function.chain([
    domain.basis( 'std', degree=2 ).vector(2),
    domain.basis( 'std', degree=1 ),
  ])
  dofs = function.DerivativeTarget( [len(ubasis)] )
  u = ubasis.dot( dofs )
  p = pbasis.dot( dofs )
  viscosity = 1
  inertia = model.Integral( (ubasis * u).sum(-1), domain=domain, geometry=geom, degree=5 )
  stokesres = model.Integral( viscosity * ubasis['ni,j'] * (u['i,j']+u['j,i']) - ubasis['nk,k'] * p + pbasis['n'] * u['k,k'], domain=domain, geometry=geom, degree=5 )
  residual = stokesres + model.Integral( ubasis['ni'] * u['i,j'] * u['j'], domain=domain, geometry=geom, degree=5 )
  cons = domain.boundary['top,bottom'].project( [0,0], onto=ubasis, geometry=geom, ischeme='gauss2' ) \
       | domain.boundary['left'].project( [geom[1]*(1-geom[1]),0], onto=ubasis, geometry=geom, ischeme='gauss2' )
  lhs0 = model.solve_linear( dofs, residual=stokesres, constrain=cons )

  for solver in model.newton( dofs, residual=residual, lhs0=lhs0, freezedofs=cons.where ), \
                model.pseudotime( dofs, residual=residual, lhs0=lhs0, freezedofs=cons.where, inertia=inertia, timestep=1 ):
    @unittest( name=solver.__name__ )
    def res():
      tol = 1e-10
      lhs = model.solve( solver, tol=tol, maxiter=2 if solver.__name__=='newton' else 3 )
      res = residual.replace( dofs, lhs ).eval()
      resnorm = numpy.linalg.norm( res[~cons.where] )
      assert resnorm < tol
