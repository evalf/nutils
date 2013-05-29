from . import function, _
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
    
    if len(kwargs)!=2:
      raise ValueError( 'exactly two arguments expected, found %d' % len(kwargs) )

    _tryall( self, '_set_from_', kwargs )
    for key, value in kwargs.items():
      numpy.testing.assert_almost_equal( value, getattr(self,key) )

  def _set_from_lame( self, lmbda, mu ):
    self.lmbda = lmbda
    self.mu = mu

  def _set_from_poisson_young( self, nu, E ):
    self.lmbda = (nu*E)/((1.+nu)*(1.-2.*nu))
    self.mu = E/(2.*(1.+nu))

  def __call__ ( self, disp, coords ): 
    return self.lmbda * disp.div(coords)[...,_,_] * function.eye( coords.shape[0] ) \
      + (2*self.mu) * disp.symgrad(coords)
    
  def __str__( self ):
    return 'Hooke(%s,%s)' % ( self.mu, self.lmbda )

  @property
  def E( self ):
    return self.mu * (3.*self.lmbda+2.*self.mu) / (self.lmbda+self.mu)

  @property
  def nu( self ):
    return self.lmbda / (2.*(self.lmbda+self.mu))
