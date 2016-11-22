from nutils import model, mesh, function
from . import register, unittest
import numpy

class Laplace( model.Model ):
  def __init__( self, domain ):
    self.domain = domain
    self.basis = self.domain.basis( 'std', degree=1 )
    model.Model.__init__( self, ndofs=len(self.basis) )
  def namespace( self, coeffs ):
    return { 'u': self.basis.dot(coeffs) }
  def constraints( self, geom ):
    return self.domain.boundary['left'].project( 0, onto=self.basis, geometry=geom, ischeme='gauss2' )
  def evalres( self, geom, ns ):
    return model.Integral( ( self.basis.grad(geom) * ns['u'].grad(geom) ).sum(-1), domain=self.domain, geometry=geom, degree=2 ) \
         + model.Integral( self.basis, domain=self.domain.boundary['top'], geometry=geom, degree=2 )

class ConvectionDiffusion( Laplace ):
  def evalres( self, geom, ns ):
    return super().evalres(geom,ns) + model.Integral( self.basis['n,k'] * ns.u['k'], domain=self.domain, geometry=geom, degree=2 )

class NavierStokes( model.Model ):
  def __init__( self, domain ):
    self.domain = domain
    self.ubasis, self.pbasis = function.chain([
      self.domain.basis( 'std', degree=2 ).vector(2),
      self.domain.basis( 'std', degree=1 ),
    ])
    self.viscosity = 1
    model.Model.__init__( self, ndofs=len(self.ubasis) )
  def namespace( self, coeffs ):
    return model.AttrDict( u=self.ubasis.dot(coeffs), p=self.pbasis.dot(coeffs) )
  def constraints( self, geom ):
    return self.domain.boundary['top,bottom'].project( [0,0], onto=self.ubasis, geometry=geom, ischeme='gauss2' ) \
         | self.domain.boundary['left'].project( [geom[1]*(1-geom[1]),0], onto=self.ubasis, geometry=geom, ischeme='gauss2' )
  def evalres0( self, geom, ns ):
    return model.Integral( self.viscosity * self.ubasis['ni,j'] * (ns.u['i,j']+ns.u['j,i']) - self.ubasis['nk,k'] * ns.p + self.pbasis['n'] * ns.u['k,k'], domain=self.domain, geometry=geom, degree=5 )
  def evalres( self, geom, ns ):
    return self.evalres0( geom, ns ) + model.Integral( self.ubasis['ni'] * ns.u['i,j'] * ns.u['j'], domain=self.domain, geometry=geom, degree=5 )

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

@register
def coupled():
  vecs = []
  domain, geom = mesh.rectilinear( [numpy.linspace(0,1,9)] * 2 )
  model = NavierStokes( domain ) | ConvectionDiffusion( domain )
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
