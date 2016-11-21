from nutils import model, mesh
from . import register, unittest
import numpy

class Laplace( model.Model ):
  def bases( self, domain ):
    yield 'u', domain.basis( 'std', degree=1 )
  def constraints( self, domain, geom ):
    ubasis, = self.chained( domain )
    return domain.boundary['left'].project( 0, onto=ubasis, geometry=geom, ischeme='gauss2' )
  def evalres( self, domain, geom, ns ):
    ubasis, = self.chained( domain )
    return model.Integral( ( ubasis.grad(geom) * ns.u.grad(geom) ).sum(-1), domain=domain, geometry=geom, degree=2 ) \
         + model.Integral( ubasis, domain=domain.boundary['top'], geometry=geom, degree=2 )

class NavierStokes( model.Model ):
  def __init__( self ):
    self.viscosity = 1
  def bases( self, domain ):
    yield 'u', domain.basis( 'std', degree=2 ).vector(2)
    yield 'p', domain.basis( 'std', degree=1 )
  def constraints( self, domain, geom ):
    ubasis, pbasis = self.chained( domain )
    return domain.boundary['top,bottom'].project( [0,0], onto=ubasis, geometry=geom, ischeme='gauss2' ) \
         | domain.boundary['left'].project( [geom[1]*(1-geom[1]),0], onto=ubasis, geometry=geom, ischeme='gauss2' )
  def evalres0( self, domain, geom, ns ):
    ubasis, pbasis = self.chained( domain )
    return model.Integral( self.viscosity * ubasis['ni,j'] * (ns.u['i,j']+ns.u['j,i']) - ubasis['nk,k'] * ns.p + pbasis['n'] * ns.u['k,k'], domain=domain, geometry=geom, degree=5 )
  def evalres( self, domain, geom, ns ):
    ubasis, pbasis = self.chained( domain )
    return self.evalres0( domain, geom, ns ) + model.Integral( ubasis['ni'] * ns.u['i,j'] * ns.u['j'], domain=domain, geometry=geom, degree=5 )

@register
def laplace():
  domain, geom = mesh.rectilinear( [8,8] )
  model = Laplace()

  @unittest
  def res():
    ns = model.solve_namespace( domain, geom )
    cons = model.constraints( domain, geom )
    res = model.evalres( domain, geom, ns )
    resnorm = numpy.linalg.norm( (cons&0) | res.eval() )
    assert resnorm < 1e-13

@register
def navierstokes():
  vecs = []
  domain, geom = mesh.rectilinear( [numpy.linspace(0,1,9)] * 2 )
  model = NavierStokes()
  tol = 1e-10

  @unittest
  def res():
    ns = model.solve_namespace( domain, geom, tol=tol, callback=vecs.append )
    cons = model.constraints( domain, geom )
    res = model.evalres( domain, geom, ns )
    resnorm = numpy.linalg.norm( (cons&0) | res.eval() )
    assert resnorm < tol

  @unittest
  def callback():
    assert len(vecs) == 2, 'expected 2 iterations, found {}'.format( len(vecs) )
    assert all( isinstance(vec,numpy.ndarray) and vec.ndim == 1 for vec in vecs )
