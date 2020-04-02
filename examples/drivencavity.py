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
  ns.ubasis, ns.pbasis = function.chain([
    domain.basis('std', degree=degree).vector(2),
    domain.basis('std', degree=degree-1),
  ])
  ns.u_i = 'ubasis_ni ?lhs_n'
  ns.p = 'pbasis_n ?lhs_n'
  ns.stress_ij = '(u_i,j + u_j,i) / Re - p δ_ij'

  sqr = domain.boundary.integral('u_k u_k d:x' @ ns, degree=degree*2)
  wallcons = solver.optimize('lhs', sqr, droptol=1e-15)

  sqr = domain.boundary['top'].integral('(u_0 - 1)^2 d:x' @ ns, degree=degree*2)
  lidcons = solver.optimize('lhs', sqr, droptol=1e-15)

  cons = numpy.choose(numpy.isnan(lidcons), [lidcons, wallcons])
  cons[-1] = 0 # pressure point constraint

  res = domain.integral('(ubasis_ni,j stress_ij + pbasis_n u_k,k) d:x' @ ns, degree=degree*2)
  with treelog.context('stokes'):
    lhs0 = solver.solve_linear('lhs', res, constrain=cons)
    postprocess(domain, ns, lhs=lhs0)

  res += domain.integral('.5 (ubasis_ni u_i,j - ubasis_ni,j u_i) u_j d:x' @ ns, degree=degree*3)
  with treelog.context('navierstokes'):
    lhs1 = solver.newton('lhs', res, lhs0=lhs0, constrain=cons).solve(tol=1e-10)
    postprocess(domain, ns, lhs=lhs1)

  return lhs0, lhs1

# Postprocessing in this script is separated so that it can be reused for the
# results of Stokes and Navier-Stokes, and because of the extra steps required
# for establishing streamlines.

def postprocess(domain, ns, every=.05, spacing=.01, **arguments):

  ns = ns.copy_() # copy namespace so that we don't modify the calling argument
  ns.streambasis = domain.basis('std', degree=2)[1:] # remove first dof to obtain non-singular system
  ns.stream = 'streambasis_n ?streamdofs_n' # stream function
  sqr = domain.integral('((u_0 - stream_,1)^2 + (u_1 + stream_,0)^2) d:x' @ ns, degree=4)
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
    lhs0, lhs1 = main(nelems=3, etype='square', reynolds=100, degree=3)
    with self.subTest('stokes'): self.assertAlmostEqual64(lhs0, '''
      eNqNkE8og3EYx99JLZKDJTktOfjT9nr3+3kjuTi4zcUB5cpcxcGkOa3kQC4OGyUXaRfESiI7OdjzfX/v
      3nd5OUiWhpbUWlsr/1Za48Tzrec5PJ/69v1K0t8z3PN9F11TZtQsdPmS9WzaauWW/sH9iaNuRe9TbayO
      lblHkXFFtbzSqU2wNJq4E588JZZ5xNPGvepLoszta+ftGbiVAJY8/RhhaSqymJFkZ+yQhVXLXeYKWOkY
      RVbOk6yc0J2yQ2MeSX5TgnxVuVcnf3CzV6OoT+TJECfUInZoV5PkahHkM+Je3TAqvgNWBqYIYF7rRwRp
      siNmuHDGhhBWO4xKjkYzqtWKTm0TaTyTEzZKiTmKeG7IqzrkSi8hV9Ss0X3JLKatW7L0KvInvHFFv7i0
      sRzK3H96TtErPVGKArQdD8Wv6YEMOqUFGqM9uqNM6WNRjLbomHK/VIv3UgoHZIyjFw2oRjM41nGNQdhR
      JFtpr+HAlKQvrHfV6g==''')
    with self.subTest('navier-stokes'): self.assertAlmostEqual64(lhs1, '''
      eNpjYCAMgswh9ErtA5c9ruTp/7xiZbhX28jo8MVbRn1XLIxVLhcbBxuIGsDUHbhgoxNx3tnA9vwcQ4fz
      PkbC59mNP2rONOIxnGiUaOx9GaYu5PxOjfJzB/Xdz5kbWp9jNYo9t81ont4so1OGXUbNJtX6MHVd515p
      XT4rqT//7EWDprPzDR+eTTY6pBdltNhoq9ELE+3zMHWe58rVl53lv2R1Vvbq+zOTdCLPmhpONkgwlDMO
      NawyvWAIU/f7nMRN03PyF3rOSp6XOqt3bv+ZA+cirnSclzf2vxhvGgo3T/pC5xXW80Ln751ddzb1rOpZ
      wzOXTp89l3iu1vic0WrTImOYugCd8uvrriy/aHf1w9nkizPPvDl/7rTe+cRTBmf3nVQx0T4DU0dMOK8/
      vfX0xtOrT3ef9jitfXrN6Q1A9oLTk0/POL339LrTW4AiC05POt0F5G85vR0oMut0z+mK04tP7wfK7zi9
      /nTf6aWnt50+enrP6ROnL57+cXrfacYze4G8rUD843SpLgMDAGbLx+o=''')

  @testing.requires('matplotlib')
  def test_mixed(self):
    lhs0, lhs1 = main(nelems=3, etype='mixed', reynolds=100, degree=2)
    with self.subTest('stokes'): self.assertAlmostEqual64(lhs0, '''
      eNpjYEAFEy++uHzs/EYjEFv73Hm9T2eVDGFyMWdfGJ/TVTGxMvQyqjxbAlYTZM7AsNWIxwhEG5hs0wTR
      1Wd5DDTOHrx05OzmczC9Cud8riSccbrqYJR2dfMZ07M+hkznLpuongepB+Eyk/TzIHVbL4QbrLqw9Qyy
      m+aee31awWALXEzP+NIZkB6Y/TFns88mn008u/mM+dnYM0Fn28/OOnv27K6zK8+Wn10FdAEAKXRKgw==''')
    with self.subTest('navier-stokes'): self.assertAlmostEqual64(lhs1, '''
      eNpjYEAFvRcfXLa8eNsQxP5xLkBv17ktBjC5iHOfjbr1XxotNbAy0j4nbQQSCzJnYLA0lNIH0XLGi3VB
      dNK5HoNjZ2frXj+77BxM775zfbq6Z+cYxBpd1vc543jursGSC0omz86D1IPwBJOzYDszL5oaLbgQcQbZ
      TTZnec9svix4FsZXNbl9GqQHZj/rGb4zPGfYz5w5/fr03tOMZzjOKJz5d5rzjMmZL6cvAk0CAOBkR7A=''')
