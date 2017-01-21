#!/usr/bin/env python3

"""
This example demonstrates how to perform NURBS-based isogeometric analysis
for the elastic analysis of an infinite plate with a circular hole under
tension. A detailed description of the testcase can be found in:
Hughes et al. 'Isogeometric analysis: CAD, finite elements, NURBS, exact 
geometry and mesh refinement', Computer Methods in Applied Mechanics and 
Engineering, Elsevier, 2005, 194, 4135-4195.
"""

from nutils import util, mesh, function, plot, library, log, debug, _
import numpy


def main ( L  = 4. , # Domain size
           R  = 1. , # Hole radius
           E  = 1e5, # Young's modulus
           nu = 0.3, # Poisson's ratio
           T  = 10 , # Far field traction
           nr = 2  , # Number of h-refinements
           withplots = True ):

  #Create the coarsest level parameter domain
  domain, geometry = mesh.rectilinear( [1,2] )

  #Define the control points and control point weights
  controlpoints = numpy.array([[0,R],[R*(1-2**.5),R],[-R,R*(2**.5-1)],[-R,0],[0,.5*(R+L)],[-.15*(R+L),.5*(R+L)],[-.5*(R+L),.15*(R*L)],[-.5*(R+L),0],[0,L],[-L,L],[-L,L],[-L,0]])
  weights       = numpy.array([1,.5+.5/2**.5,.5+.5/2**.5,1,1,1,1,1,1,1,1,1])

  #Create the second-order B-spline basis over the coarsest domain
  bsplinebasis = domain.basis( 'spline', degree=2 )

  #Create the NURBS basis
  weightfunc = bsplinebasis.dot(weights)
  nurbsbasis = (bsplinebasis*weights)/weightfunc

  #Create the isogeometric map
  geometry = (nurbsbasis[:,_]*controlpoints).sum(0)

  #Create the computational domain by h-refinement
  domain = domain.refine( nr )

  #Create the B-spline basis
  bsplinebasis = domain.basis( 'spline', degree=2 )

  #Create the NURBS basis
  weights    = domain.project( weightfunc, onto=bsplinebasis, geometry=geometry, ischeme='gauss9' )
  nurbsbasis = (bsplinebasis*weights)/weightfunc

  #Create the displacement field basis
  ubasis = nurbsbasis.vector(2)

  #Define the plane stress function
  stress = library.Hooke( lmbda=E*nu/(1-nu**2), mu=.5*E/(1+nu) )

  #Get the exact solution
  uexact     = exact_solution( geometry, T, R, E, nu )
  sigmaexact = stress( uexact.symgrad(geometry) )

  #Define the linear and bilinear forms
  mat_func = function.outer( ubasis.symgrad(geometry), stress(ubasis.symgrad(geometry)) ).sum([2,3])
  rhs_func = ( ubasis*sigmaexact.dotnorm(geometry) ).sum(-1)

  #Compute the matrix and rhs
  mat = domain.integrate( mat_func, geometry=geometry, ischeme='gauss9' )
  rhs = domain.boundary['right'].integrate( rhs_func, geometry=geometry, ischeme='gauss9' )

  #Compute the constraints vector for the symmetry conditions
  cons = domain.boundary['top,bottom'].project( 0, onto=ubasis.dotnorm(geometry), geometry=geometry, ischeme='gauss9' )

  #Solve the system of equations
  sol = mat.solve( rhs=rhs, constrain=cons )

  #Compute the approximate displacement and stress functions
  u     = ubasis.dot( sol )
  sigma = stress( u.symgrad(geometry) )

  #Post-processing
  if withplots:
    points, colors = domain.simplex.elem_eval( [ geometry, sigma[0,0] ], ischeme='bezier8', separate=True )
    with plot.PyPlot( 'solution', index=nr ) as plt:
      plt.mesh( points, colors )
      plt.colorbar()

  #Compute the L2-norm of the error in the stress
  err  = numpy.sqrt( domain.integrate( ((sigma-sigmaexact)*(sigma-sigmaexact)).sum([0,1]), geometry=geometry, ischeme='gauss9' ) )

  #Compute the mesh parameter (maximum physical distance between diagonally opposite knot locations)
  hmax = numpy.max([max( numpy.linalg.norm(verts[0]-verts[3]), numpy.linalg.norm(verts[1]-verts[2]) ) for verts in domain.elem_eval( geometry, ischeme='bezier2', separate=True )])

  return err, hmax


def convergence( nrefine=5 ):

  err = []
  h   = []

  for irefine in log.range( 'refine', nrefine ):
    serr, hmax = main( nr=irefine )
    err.append( serr )
    h.append( hmax )

  with plot.PyPlot( 'convergence' ) as plt:
    plt.loglog( h, err, 'k*--' )
    plt.slope_triangle( h, err )
    plt.ylabel( 'L2 error of stress' )
    plt.grid( True )


def unittest():

  retvals = main( nr=0, withplots=False )
  assert debug.checkdata( retvals, '''
    eNoz1NEw0TE01dTRMLY0tEjVNdYxNTAwANGaAEntBW4=''' )

  retvals = main( nr=2, withplots=False )
  assert debug.checkdata( retvals, '''
    eNoz1NEw0TE01dTRMDQxN0vVNdYxMjCyBNGaAEniBXM=''' )

  retvals = main( L=3, R=1.5, E=1e6, nu=0.4, T=15, nr=3, withplots=False )
  assert debug.checkdata( retvals, '''
    eNoz1NEw0TE01dTRMDI1MUrVNdExN7MwA9GaAEohBX4=''' )


def exact_solution( geometry, T, R, E, nu ):

  mu = .5*E/(1+nu)
  k  = (3-nu)/(1+nu) #Plane stress parameter

  x, y = geometry/R #Dimensionless coordinates
  r2 = x**2 + y**2

  return T/(4*mu)*geometry*( [(k+1)/2,(k-3)/2] + [1+k,1-k]/r2 + (1-1/r2)*[x**2-3*y**2,3*x**2-y**2]/r2**2 )


if __name__ == '__main__':
  util.run( main, convergence, unittest )
