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
      eNp1zj1IQlEUB/BrCJKEQxLRFNFQxvN1vTcpoqWhzZaGElr7WKOGirApiIaipcEKoiXCpaKEiCKnhjzn
      XX1PejaEJGGFRCCiCH153YrXOXCG3+Fw/oT8rZFeQpaVqDGVmjHNxEKSJmxM2rOIal1aDlsxKyK+gF/a
      sZbHEA5gDmL6FduuWRnHsAQXcABEXeGP/5rVrdUPqyxWma1q2ih3u1g7/+JnPf3+BiYtr5ToBGvm33yN
      d/C3pLTrTi9d9Y2yCkuxU2Z6pa17CqpKMzTo+6AbdLJmc3eupC7axKFmF7NiR5c2aBpiUYugAxUcRk/N
      mgyn2MVXsME83INblRZW6hMFfIA6CMRvbotonTgL7/ACWQjBfjwcT8MT6HAJSxCEI8hAvroxIQZ7cA7F
      X+3ET3CgG1Ucxz5sRDu2IMctTONQNVkFbNW5iScGIT8HbdXq''')
    with self.subTest('navier-stokes'): self.assertAlmostEqual64(lhs1, '''
      eNptzkFoE0EUBuD1ELwIBUUwLdpLiq2bJjuzhkhbReihlQqKGkgLpQ3Bq4cWaVG86EXTUrC5mCCmEFRK
      JLHQRAumiYqg+XdmdrMbpB4EbyIF21uKl+7sUfLgwXvfezM8Rfk/bkQVZV0ddw6cqvrZWnEC9k1NWt2M
      i4visjgh9geOkYQu7ZZY4GN8mE/zF6EGeXReWorbLMcesl/sUyhPf3t2hb9iQ+yvMcnS2hn9XkTaPx7h
      y6yb1Yy406vPeNZj+sRPdpsRg/EHesGz68Gic6mVtHZFSGgs4P3X6eZOUbfvhIcIpT/oBX1eP6lJGwmO
      as/JVXpUz9CndMKWttX/MRwlPlqhWZqi98PS/pzzhy0tR5J0ipapKqQtnO1qnm6tBiNklsSISaSd2uk1
      /SLE6/yxuGbFvL0nznFRZH2siQTndE733n5/be2xjMGRaGx/U43OF5dQxgYKWMI4VLzBW7deQxrPUEUR
      m66sYRUpt9/EO1eyWMYi8qi58/coYQUvUcEXfMBXWGhjG0eMqtuV3Wzj7qCiHALXRMfq''')

  @testing.requires('matplotlib')
  def test_mixed(self):
    lhs0, lhs1 = main(nelems=3, etype='mixed', reynolds=100, degree=2)
    with self.subTest('stokes'): self.assertAlmostEqual64(lhs0, '''
      eNpjYICAiRePnWdg0D736SyIF3P2nK6VYSWQHWS+1SjI3MAkyLz6rMbZI2BZhXMJZxyMNp/xMbwMFA8y
      LzNhYNh6YdUFiElzzykYgGg94yBzkH6oBQwvLm80YmA4r6dkCOYZq5h4GZUYgdg8QHKbJpA2OHhp8zmQ
      iM8Vp6tpV03PMp1TPQ/ipwPJcIOtZyAmvT69Bcy6BOXHnM0+m3w28ezmM+ZnY88EnW0/O+vs2bO7zq48
      W352FdA8ABC3SoM=''')
    with self.subTest('navier-stokes'): self.assertAlmostEqual64(lhs1, '''
      eNpjYICA3ouWFxkYfpzbdQ7EizjXrb/UQBvIDjK3NAwylzMOMk86d+zs9bMg2X3ndM/GGvmcuWugZBJk
      HmQ+wYSBIfPiggsQk2zObr4MolWBciD9UAsYHly+bcjAEKC3xQDE+2z00sjKSNoIxJbSZ2BYrMvA0GMw
      W3cZ2P4+3TkGl/Udzy258Ow8iH8WqNPUKOIMxCTeM4Jgd9w+DeGznuE7w3OG/cyZ069P7z3NeIbjjMKZ
      f6c5z5ic+XL6IlAPAPejR7A=''')
