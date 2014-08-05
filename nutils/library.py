# -*- coding: utf8 -*-
#
# Module LIBRARY
#
# Part of Nutils: open source numerical utilities for Python. Jointly developed
# by HvZ Computational Engineering, TU/e Multiscale Engineering Fluid Dynamics,
# and others. More info at http://nutils.org <info@nutils.org>. (c) 2014

"""
The library module provides a collection of application specific functions, that
nevertheless have a wide enough range of applicability to be useful as generic
building blocks.
"""

from . import function, _, numeric
import numpy

def _tryall( obj, prefix, kwargs ):
  for name in dir( obj ):
    if name.startswith( prefix ):
      try:
        return getattr( obj, name )( **kwargs )
      except TypeError:
        pass
  raise Exception, 'not supported: ' + ', '.join( kwargs.keys() )

class Hooke:

  def __init__( self, **kwargs ):
    
    verify = kwargs.pop( 'verify', True )

    if len(kwargs)!=2:
      raise ValueError( 'exactly two arguments expected, found %d' % len(kwargs) )

    _tryall( self, '_set_from_', kwargs )

    if verify:
      for key, value in kwargs.items():
        numpy.testing.assert_almost_equal( value, getattr(self,key) )

  def _set_from_lame( self, lmbda, mu ):
    self.lmbda = float(lmbda)
    self.mu = float(mu)

  def _set_from_poisson_young( self, nu, E ):
    self.lmbda = (nu*E)/((1.+nu)*(1.-2.*nu))
    self.mu = E/(2.*(1.+nu))

  def __call__ ( self, epsilon ):
    ndims = epsilon.shape[-2]
    assert epsilon.shape[-1] == ndims
    return self.lmbda * function.trace( epsilon )[...,_,_] * function.eye(ndims) + 2 * self.mu * epsilon
    
  def __str__( self ):
    return 'Hooke(mu=%s,lmbda=%s)' % ( self.mu, self.lmbda )

  @property
  def E( self ):
    return self.mu * (3.*self.lmbda+2.*self.mu) / (self.lmbda+self.mu)

  @property
  def nu( self ):
    return self.lmbda / (2.*(self.lmbda+self.mu))

class Orthotropic(object):
  
  def __init__( self, E, G, nu ):

    vbar = function.product( nu )
    vv = nu[:,_]*nu[_,:]
    NE = -function.sqrt( E[:,_]*E[_,:] ) * ( vbar/vv + vv )  + function.diagonalize(E*(2.*nu**2-1.+vbar/nu**2))
    NE /= (nu**2).sum() + 2*vbar - 1
    
    self.NE     = NE
    self.Gbar   = function.product( G )
    self.Ginv   = 1./G

  def __call__( self, epsilson ):
    assert epsilon.shape[-1] == epsilon.shape[-2]
    epsdiag = function.takediag( epsilon )
    GG = self.Ginv[:,_] * self.Ginv[_,:] * self.Gbar
    return function.diagonalize( (self.NE * epsdiag[...,_,:]).sum() - 2.*self.Gbar*self.Ginv**2*epsdiag ) + 2 * GG * epsilon
      
