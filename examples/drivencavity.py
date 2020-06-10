#! /usr/bin/env python3
#
# In this script we solve the lid driven cavity problem for stationary Stokes
# and Navier-Stokes flow. That is, a unit square domain, with no-slip left,
# bottom and right boundaries and a top boundary that is moving at unit
# velocity in positive x-direction.

from nutils import mesh, function, solver, export, cli, testing
import numpy, treelog

def main(nelems:int, etype:str, degree:int, reynolds:float):
  '''
  Driven cavity benchmark problem.

  .. arguments::

     nelems [12]
       Number of elements along edge.
     etype [square]
       Element type (square/triangle/mixed).
     degree [2]
       Polynomial degree for velocity; the pressure space is one degree less.
     reynolds [1000]
       Reynolds number, taking the domain size as characteristic length.
  '''

  domain, geom = mesh.unitsquare(nelems, etype)

  ns = function.Namespace()
  ns.Re = reynolds
  ns.x = geom
  ns.ubasis = domain.basis('std', degree=degree).vector(domain.ndims)
  ns.pbasis = domain.basis('std', degree=degree-1)
  ns.u_i = 'ubasis_ni ?u_n'
  ns.p = 'pbasis_n ?p_n'
  ns.stress_ij = '(d(u_i, x_j) + d(u_j, x_i)) / Re - p δ_ij'

  usqr = domain.boundary.integral('u_k u_k d:x' @ ns, degree=degree*2)
  wallcons = solver.optimize('u', usqr, droptol=1e-15)

  usqr = domain.boundary['top'].integral('(u_0 - 1)^2 d:x' @ ns, degree=degree*2)
  lidcons = solver.optimize('u', usqr, droptol=1e-15)

  ucons = numpy.choose(numpy.isnan(lidcons), [lidcons, wallcons])
  pcons = numpy.zeros(len(ns.pbasis), dtype=bool)
  pcons[-1] = True # constrain pressure to zero in a point
  cons = dict(u=ucons, p=pcons)

  ures = domain.integral('d(ubasis_ni, x_j) stress_ij d:x' @ ns, degree=degree*2)
  pres = domain.integral('pbasis_n d(u_k, x_k) d:x' @ ns, degree=degree*2)
  with treelog.context('stokes'):
    state0 = solver.solve_linear(('u', 'p'), (ures, pres), constrain=cons)
    postprocess(domain, ns, **state0)

  ures += domain.integral('.5 (ubasis_ni d(u_i, x_j) - d(ubasis_ni, x_j) u_i) u_j d:x' @ ns, degree=degree*3)
  with treelog.context('navierstokes'):
    state1 = solver.newton(('u', 'p'), (ures, pres), arguments=state0, constrain=cons).solve(tol=1e-10)
    postprocess(domain, ns, **state1)

  return state0, state1

# Postprocessing in this script is separated so that it can be reused for the
# results of Stokes and Navier-Stokes, and because of the extra steps required
# for establishing streamlines.

def postprocess(domain, ns, every=.05, spacing=.01, **arguments):

  ns = ns.copy_() # copy namespace so that we don't modify the calling argument
  ns.streambasis = domain.basis('std', degree=2)[1:] # remove first dof to obtain non-singular system
  ns.stream = 'streambasis_n ?streamdofs_n' # stream function
  ns.ε = function.levicivita(2)
  sqr = domain.integral('(u_i - ε_ij d(stream, x_j)) (u_i - ε_ij d(stream, x_j)) d:x' @ ns, degree=4)
  arguments['streamdofs'] = solver.optimize('streamdofs', sqr, arguments=arguments) # compute streamlines

  bezier = domain.sample('bezier', 9)
  x, u, p, stream = bezier.eval(['x_i', 'sqrt(u_k u_k)', 'p', 'stream'] @ ns, **arguments)
  with export.mplfigure('flow.png') as fig: # plot velocity as field, pressure as contours, streamlines as dashed
    ax = fig.add_axes([.1,.1,.8,.8], yticks=[], aspect='equal')
    import matplotlib.collections
    ax.add_collection(matplotlib.collections.LineCollection(x[bezier.hull], colors='w', linewidths=.5, alpha=.2))
    ax.tricontour(x[:,0], x[:,1], bezier.tri, stream, 16, colors='k', linestyles='dotted', linewidths=.5, zorder=9)
    caxu = fig.add_axes([.1,.1,.03,.8], title='velocity')
    imu = ax.tripcolor(x[:,0], x[:,1], bezier.tri, u, shading='gouraud', cmap='jet')
    fig.colorbar(imu, cax=caxu)
    caxu.yaxis.set_ticks_position('left')
    caxp = fig.add_axes([.87,.1,.03,.8], title='pressure')
    imp = ax.tricontour(x[:,0], x[:,1], bezier.tri, p, 16, cmap='gray', linestyles='solid')
    fig.colorbar(imp, cax=caxp)

# If the script is executed (as opposed to imported), :func:`nutils.cli.run`
# calls the main function with arguments provided from the command line. To
# keep with the default arguments simply run :sh:`python3 drivencavity.py`.

if __name__ == '__main__':
  cli.run(main)

# Once a simulation is developed and tested, it is good practice to save a few
# strategic return values for regression testing. The :mod:`nutils.testing`
# module, which builds on the standard :mod:`unittest` framework, facilitates
# this by providing :func:`nutils.testing.TestCase.assertAlmostEqual64` for the
# embedding of desired results as compressed base64 data.

class test(testing.TestCase):

  @testing.requires('matplotlib')
  def test_square(self):
    state0, state1 = main(nelems=3, etype='square', reynolds=100, degree=3)
    with self.subTest('stokes-velocity'): self.assertAlmostEqual64(state0['u'], '''
      eNpjYCAMgswhdJ1O+uWtl7/rp13hMyq4rmx8/cI/44qLW0wMLliZMhrxGMHUPT//WmfruW8GWudSjJ6d
      FTeWP/vf+NH5TuNVhurGPqZvL8LUbTi3X+P1WV2D2rPthjZnw4yenflpdODSFaO9RpuNZple14Wp+362
      VzP87Ce9b2f0DHaduW+w7EyEIYPeH4MW4z6Dh6apSOqKr4Wf5bv47cyl87vOKJ5fdmbFOQY9lvMtxkXn
      H5rOvoSw1/H667OXz9eerTxnc3bV2Wdn2M8euKRzdq+R79lZppqXEP4Qvbz1HNd5rXNzzj47+/KM/FnG
      M4/Ol59ZZXjzjI+psB4iXGbqbL3MeSHtyqezBdfvnrl+gelMxUWf0wYXjp1iNPpyFqaOmHAGAFFhkvE=''')
    with self.subTest('stokes-pressure'): self.assertAlmostEqual64(state0['p'], '''
      eNp7dOb9mRdnHp2pPbPw9MzTN848OXPpzJ4z1Wcizqw/c//Ma6DM9TMHzsw/s+PMFxTIdfbvGfazwmf1
      zkaftTgrdJblrORZ47NTz94463qW/ezPM4xAcsLZjZcZGAAX7kL6''')
    with self.subTest('navier-stokes-velocity'): self.assertAlmostEqual64(state1['u'], '''
      eNpjYCAMgswh9ErtA5c9ruTp/7xiZbhX28jo8MVbRn1XLIxVLhcbBxuIGsDUHbhgoxNx3tnA9vwcQ4fz
      PkbC59mNP2rONOIxnGiUaOx9GaYu5PxOjfJzB/Xdz5kbWp9jNYo9t81ont4so1OGXUbNJtX6MHVd515p
      XT4rqT//7EWDprPzDR+eTTY6pBdltNhoq9ELE+3zMHWe58rVl53lv2R1Vvbq+zOTdCLPmhpONkgwlDMO
      NawyvWAIU/f7nMRN03PyF3rOSp6XOqt3bv+ZA+cirnSclzf2vxhvGgo3T/pC5xXW80Ln751ddzb1rOpZ
      wzOXTp89l3iu1vic0WrTImOYugCd8uvrriy/aHf1w9nkizPPvDl/7rTe+cRTBmf3nVQx0T4DU0dMOAMA
      p1CDeg==''')
    with self.subTest('navier-stokes-pressure'): self.assertAlmostEqual64(state1['p'], '''
      eNoNiT0OQEAYBdd5xD1cSOIMiIJGVJuIiliS9VNYKolkNG6j9BUvmZlnmJnoSAnx6RmFNSUVjgErRVOQ
      iFtWKTUZMQ2n/BuGnJaFi52bl48D73Fis+wjCpT6AWgsRHE=''')

  @testing.requires('matplotlib')
  def test_mixed(self):
    state0, state1 = main(nelems=3, etype='mixed', reynolds=100, degree=2)
    with self.subTest('stokes-velocity'): self.assertAlmostEqual64(state0['u'], '''
      eNpjYEAFEy++uHzs/EYjEFv73Hm9T2eVDGFyMWdfGJ/TVTGxMvQyqjxbAlYTZM7AsNWIxwhEG5hs0wTR
      1Wd5DDTOHrx05OzmczC9Cud8riSccbrqYJR2dfMZ07M+hkznLpuongepB+Eyk/TzIHVbL4QbrLqw9Qyy
      m+aee31awWALXEzP+NIZkB6Y/QAD2Dbr''')
    with self.subTest('stokes-pressure'): self.assertAlmostEqual64(state0['p'], '''
      eNoBIADf/1zNa81jzWHNs8w3zV3MUs2HzZrNzc26zanNd82qzgAAR/MTmQ==''')
    with self.subTest('navier-stokes-velocity'): self.assertAlmostEqual64(state1['u'], '''
      eNpjYEAFvRcfXLa8eNsQxP5xLkBv17ktBjC5iHOfjbr1XxotNbAy0j4nbQQSCzJnYLA0lNIH0XLGi3VB
      dNK5HoNjZ2frXj+77BxM775zfbq6Z+cYxBpd1vc543jursGSC0omz86D1IPwBJOzYDszL5oaLbgQcQbZ
      TTZnec9svix4FsZXNbl9GqQHZj8AAcY1/g==''')
    with self.subTest('navier-stokes-pressure'): self.assertAlmostEqual64(state1['p'], '''
      eNoBIADf/wXMDswMzAfMzMvry73LAcwIzCDM/ssJzDTM9MvRzAAAHqQRsw==''')
