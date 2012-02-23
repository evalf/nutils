from . import util, function, _

def laplace( func, coords ):
  'laplace matrix'

  g = func.grad( coords )
  return ( g[:,_,:] * g[_,:,:] ).sum(2)

def elasticity( disp, coords, LAMBDA=1., MU=1. ):
  'linear elasticity matrix'

  symgrad = disp.symgrad( coords )
  div = disp.div( coords )
  return LAMBDA * div[:,_] * div[_,:] + (2*MU) * ( symgrad[:,_,:,:] * symgrad[_,:,:,:] ).sum(2,3)

def stokes( velo, pres, coords, REYNOLDS=1. ):
  'stokes matrix'

  symgrad = velo.symgrad( coords )
  div = velo.div( coords )
  Auu = (2./REYNOLDS) * ( symgrad[:,_,:,:] * symgrad[_,:,:,:] ).sum(2,3)
  Aup = -div[:,_] * pres[_,:]
  return function.Stack( [[Auu,Aup],[Aup.T,0]] )

# vim:shiftwidth=2:foldmethod=indent:foldnestmax=1
