from nutils import model, mesh, function
from . import register, unittest
import numpy

class Laplace( model.Model ):
  def __init__( self, domain, geom ):
    self.domain = domain
    self.geom = geom
    self.basis = self.domain.basis( 'std', degree=1 )
    cons = domain.boundary['left'].project( 0, onto=self.basis, geometry=geom, ischeme='gauss2' )
    super().__init__( cons )
  def namespace( self, coeffs ):
    return { 'u': self.basis.dot(coeffs) }
  def residual( self, ns ):
    return model.Integral( ( self.basis.grad(self.geom) * ns['u'].grad(self.geom) ).sum(-1), domain=self.domain, geometry=self.geom, degree=2 ) \
         + model.Integral( self.basis, domain=self.domain.boundary['top'], geometry=self.geom, degree=2 )

class ConvectionDiffusion( Laplace ):
  def residual( self, ns ):
    return super().residual(ns) + model.Integral( self.basis['n,k'] * ns.u['k'], domain=self.domain, geometry=self.geom, degree=2 )

class NavierStokes( model.Model ):
  def __init__( self, domain, geom ):
    self.domain = domain
    self.geom = geom
    self.ubasis, self.pbasis = function.chain([
      self.domain.basis( 'std', degree=2 ).vector(2),
      self.domain.basis( 'std', degree=1 ),
    ])
    self.viscosity = 1
    cons = domain.boundary['top,bottom'].project( [0,0], onto=self.ubasis, geometry=geom, ischeme='gauss2' ) \
         | domain.boundary['left'].project( [geom[1]*(1-geom[1]),0], onto=self.ubasis, geometry=geom, ischeme='gauss2' )
    super().__init__( cons )
  def namespace( self, coeffs ):
    return model.AttrDict( u=self.ubasis.dot(coeffs), p=self.pbasis.dot(coeffs) )
  def inertia( self, ns ):
    return model.Integral( (self.ubasis * ns.u).sum(-1), domain=self.domain, geometry=self.geom, degree=5 )
  def initial( self, ns ):
    return model.Integral( self.viscosity * self.ubasis['ni,j'] * (ns.u['i,j']+ns.u['j,i']) - self.ubasis['nk,k'] * ns.p + self.pbasis['n'] * ns.u['k,k'], domain=self.domain, geometry=self.geom, degree=5 )
  def residual( self, ns ):
    return self.initial( ns ) + model.Integral( self.ubasis['ni'] * ns.u['i,j'] * ns.u['j'], domain=self.domain, geometry=self.geom, degree=5 )

@register
def laplace():
  domain, geom = mesh.rectilinear( [8,8] )
  model = Laplace( domain, geom )

  @unittest
  def res():
    ns = model.solve_namespace()
    res = model.residual( ns )
    resnorm = numpy.linalg.norm( (model.constraints&0) | res.eval() )
    assert resnorm < 1e-13

@register
def navierstokes():
  vecs = []
  domain, geom = mesh.rectilinear( [numpy.linspace(0,1,9)] * 2 )
  model = NavierStokes( domain, geom )
  tol = 1e-10

  @unittest
  def res_newton():
    ns = model.solve_namespace( tol=tol, callback=vecs.append )
    res = model.residual( ns )
    resnorm = numpy.linalg.norm( (model.constraints&0) | res.eval() )
    assert resnorm < tol

  @unittest
  def callback():
    assert len(vecs) == 2, 'expected 2 iterations, found {}'.format( len(vecs) )
    assert all( isinstance(vec,numpy.ndarray) and vec.ndim == 1 for vec in vecs )

  @unittest
  def res_pseudo():
    for ns in model.pseudo_timestep_namespace( tol=tol, timestep=1 ):
      pass
    res = model.residual( ns )
    resnorm = numpy.linalg.norm( (model.constraints&0) | res.eval() )
    assert resnorm < tol

@register
def coupled():
  vecs = []
  domain, geom = mesh.rectilinear( [numpy.linspace(0,1,9)] * 2 )
  model = NavierStokes( domain, geom ) | ConvectionDiffusion( domain, geom )
  tol = 1e-10

  @unittest
  def res():
    ns = model.solve_namespace( tol=tol, callback=vecs.append )
    res = model.residual( ns )
    resnorm = numpy.linalg.norm( (model.constraints&0) | res.eval() )
    assert resnorm < tol

  @unittest
  def callback():
    assert len(vecs) == 2, 'expected 2 iterations, found {}'.format( len(vecs) )
    assert all( isinstance(vec,numpy.ndarray) and vec.ndim == 1 for vec in vecs )
