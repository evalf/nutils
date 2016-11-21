from nutils import model, mesh
from . import register, unittest
import numpy

class Laplace( model.Model ):
  def __init__( self, domain ):
    self.domain = domain
  def bases( self ):
    yield 'u', self.domain.basis( 'std', degree=1 )
  def constraints( self, geom ):
    ubasis, = self.chained()
    return self.domain.boundary['left'].project( 0, onto=ubasis, geometry=geom, ischeme='gauss2' )
  def evalres( self, geom, ns ):
    ubasis, = self.chained()
    return model.Integral( ( ubasis.grad(geom) * ns.u.grad(geom) ).sum(-1), domain=self.domain, geometry=geom, degree=2 ) \
         + model.Integral( ubasis, domain=self.domain.boundary['top'], geometry=geom, degree=2 )

class NavierStokes( model.Model ):
  def __init__( self, domain ):
    self.domain = domain
    self.viscosity = 1
  def bases( self ):
    yield 'u', self.domain.basis( 'std', degree=2 ).vector(2)
    yield 'p', self.domain.basis( 'std', degree=1 )
  def constraints( self, geom ):
    ubasis, pbasis = self.chained()
    return self.domain.boundary['top,bottom'].project( [0,0], onto=ubasis, geometry=geom, ischeme='gauss2' ) \
         | self.domain.boundary['left'].project( [geom[1]*(1-geom[1]),0], onto=ubasis, geometry=geom, ischeme='gauss2' )
  def evalres0( self, geom, ns ):
    ubasis, pbasis = self.chained()
    return model.Integral( self.viscosity * ubasis['ni,j'] * (ns.u['i,j']+ns.u['j,i']) - ubasis['nk,k'] * ns.p + pbasis['n'] * ns.u['k,k'], domain=self.domain, geometry=geom, degree=5 )
  def evalres( self, geom, ns ):
    ubasis, pbasis = self.chained()
    return self.evalres0( geom, ns ) + model.Integral( ubasis['ni'] * ns.u['i,j'] * ns.u['j'], domain=self.domain, geometry=geom, degree=5 )

@register
def laplace():
  domain, geom = mesh.rectilinear( [8,8] )
  model = Laplace( domain )

  @unittest
  def res():
    ns = model.solve_namespace( geom )
    cons = model.constraints( geom )
    res = model.evalres( geom, ns )
    resnorm = numpy.linalg.norm( (cons&0) | res.eval() )
    assert resnorm < 1e-13

@register
def navierstokes():
  vecs = []
  domain, geom = mesh.rectilinear( [numpy.linspace(0,1,9)] * 2 )
  model = NavierStokes( domain )
  tol = 1e-10

  @unittest
  def res():
    ns = model.solve_namespace( geom, tol=tol, callback=vecs.append )
    cons = model.constraints( geom )
    res = model.evalres( geom, ns )
    resnorm = numpy.linalg.norm( (cons&0) | res.eval() )
    assert resnorm < tol

  @unittest
  def callback():
    assert len(vecs) == 2, 'expected 2 iterations, found {}'.format( len(vecs) )
    assert all( isinstance(vec,numpy.ndarray) and vec.ndim == 1 for vec in vecs )
