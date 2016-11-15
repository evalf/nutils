from nutils import model, mesh
from . import register, unittest
import numpy

class Laplace( model.Model ):
  def bases( self, domain ):
    yield 'u', domain.basis( 'std', degree=1 )
  def evalres( self, domain, geom, ns ):
    ubasis, = self.chained( domain )
    integral = model.Integral( ( ubasis.grad(geom) * ns.u.grad(geom) ).sum(-1), domain=domain, geometry=geom, degree=2 )
    integral += model.Integral( ubasis, domain=domain.boundary['top'], geometry=geom, degree=2 )
    cons = domain.boundary['left'].project( 0, onto=ubasis, geometry=geom, ischeme='gauss2' )
    return integral, cons

class NavierStokes( model.Model ):
  def bases( self, domain ):
    yield 'u', domain.basis( 'std', degree=2 ).vector(2)
    yield 'p', domain.basis( 'std', degree=1 )
  def evalres( self, domain, geom, ns ):
    ubasis, pbasis = self.chained( domain )
    viscosity = 1.
    integral = model.Integral( ubasis['ni'] * ns.u['i,j'] * ns.u['j'] + viscosity * ubasis['ni,j'] * (ns.u['i,j']+ns.u['j,i']) - ubasis['nk,k'] * ns.p + pbasis['n'] * ns.u['k,k'], domain=domain, geometry=geom, degree=5 )
    cons = domain.boundary['top,bottom'].project( [0,0], onto=ubasis, geometry=geom, ischeme='gauss2' ) \
         | domain.boundary['left'].project( [geom[1]*(1-geom[1]),0], onto=ubasis, geometry=geom, ischeme='gauss2' )
    return integral, cons

@register
def laplace():
  domain, geom = mesh.rectilinear( [8,8] )
  model = Laplace()

  @unittest
  def res():
    ns = model.solve_namespace( domain, geom )
    integral, cons = model.evalres( domain, geom, ns )
    res = numpy.linalg.norm( (cons&0) | integral.eval() )
    assert res < 1e-13

@register
def navierstokes():
  vecs = []
  domain, geom = mesh.rectilinear( [numpy.linspace(0,1,9)] * 2 )
  model = NavierStokes()
  tol = 1e-10

  @unittest
  def res():
    ns = model.solve_namespace( domain, geom, tol=tol, callback=vecs.append )
    integral, cons = model.evalres( domain, geom, ns )
    res = numpy.linalg.norm( (cons&0) | integral.eval() )
    assert res < tol

  @unittest
  def callback():
    assert len(vecs) == 2, 'expected 2 iterations, found {}'.format( len(vecs) )
    assert all( isinstance(vec,numpy.ndarray) and vec.ndim == 1 for vec in vecs )
