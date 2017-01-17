#! /usr/bin/env python3

from nutils import *


class MakePlots( object ):

  def __init__( self, geom, exact, optimalrate ):
    self.geom = geom
    self.exact = exact
    self.index = 0
    self.ndofs = []
    self.error_exact = []
    self.error_estimate = []
    self.optimalrate = optimalrate

  def __call__( self, domain, sol, ndofs, error_estimate ):
    self.index += 1

    error_exact = domain['aoi'].integrate( self.exact - sol, geometry=self.geom, ischeme='gauss9' )
    log.user( 'error estimate: %.2e (%.1f%% accurate)' % ( error_estimate, 100.*error_estimate/error_exact ) )

    points, colors = domain.elem_eval( [ self.geom, sol ], ischeme='bezier9', separate=True )
    aoi = domain['aoi'].boundary.elem_eval( self.geom, ischeme='bezier2', separate=True )
    with plot.PyPlot( 'sol', index=self.index ) as plt:
      plt.mesh( points, colors )
      plt.colorbar()
      plt.segments( aoi )

    self.ndofs.append( ndofs )
    self.error_exact.append( error_exact )
    self.error_estimate.append( error_estimate )
    with plot.PyPlot( 'conv', index=self.index ) as plt:
      plt.loglog( self.ndofs, self.error_exact, 'k-^', label='exact' )
      plt.loglog( self.ndofs, self.error_estimate, 'k--', label='estimate' )
      plt.slope_marker( ndofs, min( error_exact, error_estimate ), slope=-self.optimalrate )
      plt.legend( loc=3, frameon=False )
      plt.grid()
      plt.xlabel( 'degrees of freedom' )
      plt.ylabel( 'error' )


def main(
    degree: 'number of elements' = 1,
    solvetol: 'solver tolerance' = 1e-10,
    circle: 'use circular area of interest (default square)' = False,
    uniform: 'use uniform refinement (default adaptive)' = False,
    basistype: 'basis function' = 'std',
    nrefine: 'maximum allowed number of refinements' = 7,
    withplots: 'create plots' = True,
  ):

  # construct domain
  verts = numpy.linspace( -1, 1, 7 )
  basetopo, geom = mesh.rectilinear( [verts,verts] )
  aoi = basetopo.trim( .04 - ((geom+.5)**2).sum(-1), maxrefine=5 ) if circle else basetopo[1:2,1:2]
  domain = ( basetopo.withboundary( outside=... )
           - basetopo[3:,:3].withboundary( inside=... ) ).withsubdomain( aoi=aoi )

  # construct exact sulution (used for boundary conditions and error evaluation)
  exact = ( geom**2 ).sum(-1)**(1./3) * function.sin( (2./3) * function.arctan2(-geom[1],-geom[0]) )
  flux = exact.ngrad( geom )

  # sanity check
  harmonicity = numpy.sqrt( domain.integrate( exact.laplace(geom)**2, geometry=geom, ischeme='gauss9' ) )
  log.info( 'exact solution lsqr harmonicity:', harmonicity )

  # prepare plotting
  makeplots = MakePlots( geom, exact, rational.frac(2 if uniform else degree*3,3) ) if withplots else lambda *args, **kwargs: None

  # start adaptive refinement
  for irefine in log.count( 'level', start=1 ):

    # construct, solve course domain primal/dual problem
    basis = domain.basis( basistype, degree=degree )
    laplace = function.outer( basis.grad(geom) ).sum(-1)
    matrix = domain.integrate( laplace, geometry=geom, ischeme='gauss5' )
    rhsprimal = domain.boundary['inside'].integrate( basis * flux, geometry=geom, ischeme='gauss99' )
    rhsdual = domain['aoi'].integrate( basis, geometry=geom, ischeme='gauss5' )
    cons = domain.boundary['outside'].project( exact, ischeme='gauss9', geometry=geom, onto=basis )
    lhsprimal = matrix.solve( rhsprimal, constrain=cons, tol=solvetol, symmetric=True )
    lhsdual = matrix.solve( rhsdual, constrain=cons&0, tol=solvetol, symmetric=True )
    primal = basis.dot( lhsprimal )
    dual = basis.dot( lhsdual )

    # construct, solve refined domain primal/dual problem
    finedomain = domain.refined
    finebasis = finedomain.basis( basistype, degree=degree )
    finelaplace = function.outer( finebasis.grad(geom) ).sum(-1)
    finematrix = finedomain.integrate( finelaplace, geometry=geom, ischeme='gauss5' )
    finerhsdual = finedomain['aoi'].integrate( finebasis, geometry=geom, ischeme='gauss5' )
    finecons = finedomain.boundary['outside'].project( 0, ischeme='gauss5', geometry=geom, onto=finebasis )
    finelhsdual = finematrix.solve( finerhsdual, constrain=finecons, tol=solvetol, symmetric=True )

    # evaluate error estimate
    dlhsdual = finelhsdual - finedomain.project( dual, onto=finebasis, geometry=geom, ischeme='gauss5' )
    ddualw = finebasis * dlhsdual
    error_est_w = finedomain.boundary['inside'].integrate( ddualw * flux, geometry=geom, ischeme='gauss99' )
    error_est_w -= finedomain.integrate( ( ddualw.grad(geom) * primal.grad(geom) ).sum(-1), geometry=geom, ischeme='gauss5' )

    # plot solution and error convergence
    makeplots( domain, primal, len(lhsprimal), error_estimate=abs(error_est_w).sum() )

    if irefine >= nrefine:
      break

    # refine mesh
    if uniform:
      domain = domain.refined
    else:
      mask = error_est_w**2 > numpy.mean(error_est_w**2)
      domain = domain.refined_by( elem.transform[:-1] for elem in domain.refined.supp( finebasis, mask ) )

  return lhsprimal, error_est_w


def unittest():

  retvals = main( degree=1, solvetol=0, circle=False, uniform=False, basistype='std', nrefine=2, withplots=False )
  assert debug.checkdata( retvals, '''
    eNqNVVmuJDkIvM57UuaIfTlQ3/8KjcF21fy1VJIpzBJAmMTnRx7U3+fnxyjzzyuPMNM6iRD6P0Gd8byl
    6Iu3LLiF42LhrRBPb9cS2gBZ9M9r5Ssp4+vj8lrYCOqGbcOAOhqlMXaTSSRIMV4GDeoNkw0GcYw1Y2wC
    tk2GTkDTHONKPjap2xiRypg7IP/f614hWQviVOX5I8mwTi7FQs6BuDyYoksixw5V/eoTRRqii5adPm5q
    3TajBqq6O80w7UvvE1VaH7t6R+gybBdYvey4LNz3GDDIyZZjtTTTbAmxM97MGp1BZRCeJpLGrj3YFtRX
    e7RLYJUW0Lg1xRn47/nHnzeZ8mE2XmepItYsEJeMXcGoUQoawsf3dVh1Z82aWNqFEKONXva0vlOLCUcc
    naJqMGmjmqO3hg1tbCjjJOEM2Lkt16BapB13Q5DmfjlkUu7oG8K8l3xEVpYKrsk3eITeiAr8iYiM7aXp
    g9adx71IdhvwEtx+1VTXsG5n0P0A2NhCYAoOW2+rBBPXDsphdAsGOqWhkNzwot/h3yCfhuJ0thIljuYN
    V2jpk8ADzjTZ6SYTWW9l5OKSHZt/+RmvNbJjytokFUcJhgPMYCcyOh3+CJouMcHsIMOvSRbD73ReMT9+
    dZGHB69sxjX7lOy7L4RxuEe7+JfUh2E2nKspIo4mYHO2nsAwRz1ku7k2VppScDEU4jsZgsUHlF9QAaIX
    eE97DxGVvoqlJN0cWasCaS0z6jLD17IojSl1qCI0NYhI2c8tCDdQ4LZh1F2xJdMXUDKW87d4ekhlgqf3
    kp9We++SOzq8BOn391V9rKfcvVsLvMAiAM5zoV2QyGm9+7TVbAZesg5GUdepqDfpml3t5Hl/Mwwomkh3
    UZn21vDUTyvjG7LpYUotJLjE49P6etNHRLvsM94U3u8XfHj1ijSj+DHf8DAFp96QvcxYZj1QcuNkAN67
    zBtCpe8L0sWz/l5vdrmOAm0mXdMcjx50dKtrMfslFOxl8Wp9ylqi/TgsaDYLYp7FVF+lw7/f5/cvNmnD
    XQ==''' )

  retvals = main( degree=2, solvetol=0, circle=False, uniform=False, basistype='spline', nrefine=1, withplots=False )
  assert debug.checkdata( retvals, '''
    eNqNVFmuHDEIvM4bqR2ZzcCBcv8rhM2tiZSPSCOZwVBQBW54fvgB+Tw/P4f2/r34EWLOk9ApTwA4ea7X
    uDfrhq6be0TKIWJeGAxYGNj3gWGDwccaQ7Q9N1cRK+dAR5D3CeBdDJgHw0+jHsT23FyDXR3qYLCc5uLW
    VcEmg42hU7c1/M11hOKgviviGEhhHIEpv4f97ptl1OXXzV3iLGPsvlLniTnYMbATkAKH9eKoTszghPZa
    MaHCLiPa3H/XenHeGMBTRox3/3r+/wdsIQHs51C2D/i4ZUfhIZc2onxfAXpo6nEa1M1Cs4yBZE2nghaR
    a4fraWNB8C2ri+LOsWYWkY0F27sR0LpMcJjekM5gI9LueK75hkWEMN3lVoVBNUag2OvzXRa27Ww/1Pdm
    Fgu2i9HiUj1dpELpOgx9hdFaXykVMDwoBkMb2cs6ijJkcfNX1dBo8oGVp0a9prAYsVksu5oKwLhAdZSg
    Qz4+VyxLuapALDpQF5ZILd5dWIHeckQjaS2Op9ykbUBqZPFlSHHsQeNh2xPxRxineL/k5F/fh+oHcpHD
    GrIG0zMCXpx6DyVkr88RlzxVNnUPyNWVW04jlUI/b3LLyoRNU3or3qEyFijj9A1ARWSRzfoA5SJrDLlW
    Kwpv5Nrjelj2iOJouaFbJYGm5VbqFr/eYg2plKoCIhcO6q1Nh0feuykkb/6KOja0zKsTPjXO8DDdzY4m
    hnzo0attjOdyVlGv5muoUdzmv+2ZSdPM4jwurs0fFjZNoNrdCrzrTLKbtxzBQdt+H5/k/Hq/YsxFADw/
    QPVGZdCO5qcoJ6V6Za27eJDq3w/yn7/P8/kDV7pq1w==''' )

  retvals = main( degree=1, solvetol=0, circle=True, uniform=False, basistype='std', nrefine=1, withplots=False )
  assert debug.checkdata( retvals, '''
    eNqNU1mOXDEIvM609ByxLwea+18hBuxOevKTliXTQFEF5uHzJQ/q6/n6Msr8XvIIM9dNhFT3UjT5XojP
    ertuzrogCxuwuzRYFDuBTHambUNyIOJpgz0YN2mMygQoQiYBHAebB6tKg/UuK0+YNI+pefMjx+EtrG46
    9JxqmiN+Y04XMOKXJpyymSM7Q/WgTE8oYEI6Le6J0PZwF+RP1DuEZG3sAcOv5/8Pim6NUWZwtZTtVdk6
    Edq2EtbuJJTrnrPQyQu/MMArzRB5HPtXjuURWQZBRj/vTqWOEGN0PQRtjmWM1jkk/JMqxONIXXsQfmxL
    45vplFcrE+cxr1SFVsZALQyd9+07AFN4r52NIU7Djig5noDxqNYS5ZOi2UrN//DPEa5uWhyie3HUeB2v
    NmSVk7A33T9kRpj1tESnLKOMKJcY3VR7UbpNewgINg4R4srQzC66VGLGufdEDo2Zwx1jCFxNK4Hirz72
    g9oUIYnGWmTfbDYPi2yzJUalbKDME605YF6zPrHP6iLa/5c7DHip5/n+GYp7+wjUzobkex+8H/Da/raR
    fu7Mv+f1vH4Dyjj09Q==''' )


if __name__ == '__main__':
  cli.run( main, unittest )
