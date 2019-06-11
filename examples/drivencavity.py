#! /usr/bin/env python3
#
# In this script we solve the lid driven cavity problem for stationary Stokes
# and Navier-Stokes flow. That is, a unit square domain, with no-slip left,
# bottom and right boundaries and a top boundary that is moving at unit
# velocity in positive x-direction.

import nutils, numpy

# The main function defines the parameter space for the script. Configurable
# parameters are the mesh density (in number of elements along an edge),
# element type (square, triangle, or mixed), polynomial degree, and Reynolds
# number.

def main(nelems: 'number of elements' = 12,
         etype: 'type of elements (square/triangle/mixed)' = 'square',
         degree: 'polynomial degree for velocity' = 3,
         reynolds: 'reynolds number' = 1000.):

  domain, geom = nutils.mesh.unitsquare(nelems, etype)

  ns = nutils.function.Namespace()
  ns.Re = reynolds
  ns.x = geom
  ns.ubasis, ns.pbasis = nutils.function.chain([
    domain.basis('std', degree=degree).vector(2),
    domain.basis('std', degree=degree-1),
  ])
  ns.u_i = 'ubasis_ni ?lhs_n'
  ns.p = 'pbasis_n ?lhs_n'
  ns.stress_ij = '(u_i,j + u_j,i) / Re - p δ_ij'

  sqr = domain.boundary.integral('u_k u_k d:x' @ ns, degree=degree*2)
  wallcons = nutils.solver.optimize('lhs', sqr, droptol=1e-15)

  sqr = domain.boundary['top'].integral('(u_0 - 1)^2 d:x' @ ns, degree=degree*2)
  lidcons = nutils.solver.optimize('lhs', sqr, droptol=1e-15)

  cons = numpy.choose(numpy.isnan(lidcons), [lidcons, wallcons])
  cons[-1] = 0 # pressure point constraint

  res = domain.integral('(ubasis_ni,j stress_ij + pbasis_n u_k,k) d:x' @ ns, degree=degree*2)
  with nutils.log.context('stokes'):
    lhs0 = nutils.solver.solve_linear('lhs', res, constrain=cons)
    postprocess(domain, ns, lhs=lhs0)

  res += domain.integral('ubasis_ni u_i,j u_j d:x' @ ns, degree=degree*3)
  with nutils.log.context('navierstokes'):
    lhs1 = nutils.solver.newton('lhs', res, lhs0=lhs0, constrain=cons).solve(tol=1e-10)
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
  arguments['streamdofs'] = nutils.solver.optimize('streamdofs', sqr, arguments=arguments) # compute streamlines

  bezier = domain.sample('bezier', 9)
  x, u, p, stream = bezier.eval(['x_i', 'sqrt(u_k u_k)', 'p', 'stream'] @ ns, **arguments)
  with nutils.export.mplfigure('flow.png') as fig: # plot velocity as field, pressure as contours, streamlines as dashed
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
  nutils.cli.run(main)

# Once a simulation is developed and tested, it is good practice to save a few
# strategicly chosen return values for routine regression testing. Here we use
# the standard :mod:`unittest` framework, with
# :func:`nutils.numeric.assert_allclose64` facilitating the embedding of
# desired results as compressed base64 data.

class test(nutils.testing.TestCase):

  @nutils.testing.requires('matplotlib')
  def test_square(self):
    lhs0, lhs1 = main(nelems=3, etype='square', reynolds=100, degree=3)
    nutils.numeric.assert_allclose64(lhs0, 'eNp1zj1IQlEUB/BrCJKEQxLRFNFQxvN1vTcpo'
      'qWhzZaGElr7WKOGirApiIaipcEKoiXCpaKEiCKnhjznXX1PejaEJGGFRCCiCH153YrXOXCG3+F'
      'w/oT8rZFeQpaVqDGVmjHNxEKSJmxM2rOIal1aDlsxKyK+gF/asZbHEA5gDmL6FduuWRnHsAQXc'
      'ABEXeGP/5rVrdUPqyxWma1q2ih3u1g7/+JnPf3+BiYtr5ToBGvm33yNd/C3pLTrTi9d9Y2yCku'
      'xU2Z6pa17CqpKMzTo+6AbdLJmc3eupC7axKFmF7NiR5c2aBpiUYugAxUcRk/Nmgyn2MVXsME83'
      'INblRZW6hMFfIA6CMRvbotonTgL7/ACWQjBfjwcT8MT6HAJSxCEI8hAvroxIQZ7cA7FX+3ET3C'
      'gG1Ucxz5sRDu2IMctTONQNVkFbNW5iScGIT8HbdXq')
    nutils.numeric.assert_allclose64(lhs1, 'eNptzktoU0EUBuC7KeLGguKioS4MBdPekNyZS'
      'WIwEihowVVBxJW0pYuiFgpiXSh0F0ltELvoC2zAVuorRuiTJlRLC6Hof2cml0wwCxVqCl1XFOq'
      'i4p27LPlXP985HI5hHM/1i4aRMzvVL7VqOs4j5VMhS9un8k2ZkEnZLL+271v3mLYb8oG4KuKiR'
      '0yGtkk6om1MODzLH/Ma/xZK0b+eXROveJzX7Vs8ZcXYUFTbkYiJp7yFb9i3VTO765m/fFL+5IM'
      '8ZBfFHJvybCD4WvVWi86BZPIsj3j3Gv3cKKXKUDhJovQ7TbBhdsrSdjl4xcqSbtrEZukM7VDa3'
      'ge2wnHSRAt0lmboSFjbCfNMuGItkH7aSxdpi9Q2c+Gf80JFgpdIHxkgdaJtt3aufFq2iRXxUPq'
      'chLfnV63yLT/Pd2CKLXqfadsL9DmGmLeruPPl42diN/44jyV8wBuMogvteIe827MYxwTWkMOiK'
      '1k8QxrTbl9xZQpPMIzn2EDR3cgjg5dYxzYKKIHjDzbx252sY9mdHuKHaRj/AYh1yFc=')

  @nutils.testing.requires('matplotlib')
  def test_mixed(self):
    lhs0, lhs1 = main(nelems=3, etype='mixed', reynolds=100, degree=2)
    nutils.numeric.assert_allclose64(lhs0, 'eNpjYICAiRePnWdg0D736SyIF3P2nK6VYSWQH'
      'WS+1SjI3MAkyLz6rMbZI2BZhXMJZxyMNp/xMbwMFA8yLzNhYNh6YdUFiElzzykYgGg94yBzkH6'
      'oBQwvLm80YmA4r6dkCOYZq5h4GZUYgdg8QHKbJpA2OHhp8zmQiM8Vp6tpV03PMp1TPQ/ipwPJc'
      'IOtZyAmvT69Bcy6BOXHnM0+m3w28ezmM+ZnY88EnW0/O+vs2bO7zq48W352FdA8ABC3SoM=')
    nutils.numeric.assert_allclose64(lhs1, 'eNpjYICA1RezLjIwPD639hyIl31umX6vgQGQH'
      'WTuaRhkLmYcZB54bvvZq2dBsofPqZ4tMoo4o22oaxJkHmReasLAsOrihAsQkxzOJl0B0TJAOZB'
      '+qAUMtZefGzIwxOjtNgDxfho9MbI1UjcCsV/pMTA802VgqDNYqrsEbL+I7nGD0/o655ouMIFN3'
      'QLUqWSUcQZiEvMZbrA7npyG8IXPyJ2RPiN65ubpn6dPn+Y9I3XG4AwfUMzlDPuZ60A9AH73RT0'
      '=')
