from finity import util, function, _

@util.withrepr
def Hooke( **kwargs ):

  if len(kwargs)!=2:
    raise ValueError( 'Exactly two arguments should be provided' )

  if 'lmbda' in kwargs and 'mu' in kwargs:
    lmbda = kwargs['lmbda']
    mu    = kwargs['mu']
  elif 'E' in kwargs and 'nu' in kwargs:
    lmbda = kwargs['E']*kwargs['nu']/((1.+kwargs['nu'])*(1.-2.*kwargs['nu']))
    mu    = kwargs['E']/(2.*(1.+kwargs['nu']))
  else:
    raise ValueError('Illegal argument combination. Valid combinations are: (lmbda,mu), (E,nu)')

  return lambda disp, coords: \
    lmbda * disp.div(coords)[...,_,_] * function.eye( coords.shape[0] ) \
      + (2*mu) * disp.symgrad(coords)
