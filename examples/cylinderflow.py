#! /usr/bin/env python3

import nutils, numpy
import matplotlib.collections

def main(nelems: 'number of elements' = 12,
         reynolds: 'reynolds number' = 100.,
         rotation: 'cylinder rotation speed' = 0.,
         timestep: 'time step' = 1/24,
         maxradius: 'approximate domain size' = 50.,
         endtime: 'end time' = numpy.inf,
         degree: 'polynomial degree' = 2):

  # construct mesh
  rscale = numpy.pi / nelems
  melems = numpy.ceil(numpy.log(2*maxradius) / rscale).astype(int)
  nutils.log.info('creating {}x{} mesh, outer radius {:.2f}'.format(melems, 2*nelems, .5*numpy.exp(rscale*melems)))
  domain, geom = nutils.mesh.rectilinear([range(melems+1),numpy.linspace(0,2*numpy.pi,2*nelems+1)], periodic=(1,))
  domain = domain.withboundary(inner='left', outer='right')

  ns = nutils.function.Namespace()
  ns.uinf = 1, 0
  ns.r = .5 * nutils.function.exp(rscale * geom[0])

  s = .01
  ns.Re = reynolds# * (1 + s - nutils.function.power(s, 1-geom[0]/melems))

  ns.phi = geom[1] + 2/3 * numpy.pi / nelems # nudge to break element symmetry
  ns.x_i = 'r <-cos(phi), -sin(phi)>_i'
  ns.J = ns.x.grad(geom)
  ns.unbasis, ns.utbasis, ns.pbasis = nutils.function.chain([ # compatible spaces using piola transformation
    domain.basis('spline', degree=(degree,degree-1), removedofs=((0,),None)),
    domain.basis('spline', degree=(degree-1,degree)),
    domain.basis('spline', degree=degree-1),
  ]) / nutils.function.determinant(ns.J)
  ns.ubasis_ni = 'unbasis_n J_i0 + utbasis_n J_i1'
  ns.u_i = 'ubasis_ni ?lhs_n'
  ns.p = 'pbasis_n ?lhs_n'
  ns.sigma_ij = '(u_i,j + u_j,i) / Re - p δ_ij'
  ns.N = 5 * degree * nelems / numpy.pi
  ns.rotation = rotation
  ns.uwall_i = '0.5 rotation <sin(phi), -cos(phi)>_i'

  # create residual vector components
  res = domain.integral('ubasis_ni,j sigma_ij + pbasis_n u_k,k' @ ns, geometry=ns.x, degree=9)
  res += domain.boundary['inner'].integral('(N ubasis_ni - (ubasis_ni,j + ubasis_nj,i) n_j) (u_i - uwall_i) / Re' @ ns, geometry=ns.x, degree=9)
  oseen = domain.integral('ubasis_ni u_i,j uinf_j' @ ns, geometry=ns.x, degree=9)
  convec = domain.integral('ubasis_ni u_i,j u_j' @ ns, geometry=ns.x, degree=9)
  inertia = domain.integral('ubasis_ni u_i' @ ns, geometry=ns.x, degree=9)

  # constrain full velocity vector at inflow
  #sqr = domain.boundary['outer'].integral('(u_i - uinf_i) (u_i - uinf_i)' @ ns, degree=9)
  sqr = domain.boundary['outer'].integral('(u_i - uinf_i) (u_i - uinf_i)' @ ns * nutils.function.min(0, 'uinf_k n_k' @ ns), degree=9)
  cons = nutils.solver.optimize('lhs', sqr, droptol=1e-15)
  #cons[-1] = 0

  lhs0 = nutils.solver.solve_linear('lhs', res, constrain=cons)

  bbox = numpy.array([[-2,6],[-3,3]])
  bezier0 = domain.sample('bezier', 5)
  bezier = bezier0.subset((bezier0.eval((ns.x-bbox[:,0]) * (bbox[:,1]-ns.x)) > 0).all(axis=1))
  interpolate = nutils.util.tri_interpolator(bezier.tri, bezier.eval(ns.x), mergetol=1e-5)

  spacing = .075
  xgrd = nutils.util.regularize(bbox, spacing)

  for istep, lhs in nutils.log.enumerate('timestep', nutils.solver.impliciteuler('lhs', residual=res+convec, inertia=inertia, lhs0=lhs0, timestep=timestep, constrain=cons, newtontol=1e-10)):

    x, u, div, re, detJ = bezier0.eval([ns.x, ns.u, 'u_k,k'@ns, ns.Re, nutils.function.determinant(ns.J)], arguments=dict(lhs=lhs))
    nutils.export.triplot('divergence.jpg', x, div, tri=bezier0.tri, hull=bezier0.hull)
    nutils.export.triplot('u0.jpg', x, u[:,0], tri=bezier0.tri, hull=bezier0.hull)
    nutils.export.triplot('u1.jpg', x, u[:,1], tri=bezier0.tri, hull=bezier0.hull)
    nutils.export.triplot('re.jpg', x, re, tri=bezier0.tri, hull=bezier0.hull)
    nutils.export.triplot('detJ.jpg', x, detJ, tri=bezier0.tri, hull=bezier0.hull)

    t = istep * timestep
    x, u, normu, p, Re = bezier.eval([ns.x, ns.u, nutils.function.norm2(ns.u), ns.p, ns.Re], arguments=dict(lhs=lhs))
    ugrd = interpolate[xgrd](u)

    with nutils.export.mplfigure('flow.jpg') as fig:
      ax = fig.add_axes([0,0,1,1], yticks=[], xticks=[], frame_on=False, xlim=bbox[0], ylim=bbox[1])
      im = ax.tripcolor(x[:,0], x[:,1], bezier.tri, p, shading='gouraud', cmap='jet')
      ax.add_collection(matplotlib.collections.LineCollection(x[bezier.hull], colors='k', linewidths=.1, alpha=.5))
      ax.quiver(xgrd[:,0], xgrd[:,1], ugrd[:,0], ugrd[:,1], angles='xy', width=1e-3, headwidth=3e3, headlength=5e3, headaxislength=2e3, zorder=9)
      ax.plot(0, 0, 'k', marker=(3,2,t*rotation*180/numpy.pi-90), markersize=20)
      cax = fig.add_axes([0.8, 0.1, 0.02, 0.8])
      fig.colorbar(im, cax=cax)

    if t >= endtime:
      break

    xgrd = nutils.util.regularize(bbox, spacing, xgrd + ugrd * timestep)

  return lhs0, lhs

if __name__ == '__main__':
  nutils.cli.run(main)

import unittest

class test(unittest.TestCase):

  def test_rot0(self):
    lhs0, lhs = main(nelems=3, reynolds=1e-2, timestep=.1, endtime=.05, rotation=0)
    nutils.numeric.assert_allclose64(lhs0, 'eNoB2AAn/0nRvS+1MHguQNBiz7bLOTUiNvIzw'
      '8oByg3IwzjkOTY3NcdgxmDE6zvmPCk6AsTKw57BYzrDQJ/BFsQ5wSXA3TxTyHdBt8LFvzsxHzI'
      '9MMTO980C0Cc1HTY0NNXKAsoqzJw4kTnfN1LHqcbKyBc7mjzZOpDE38PkxkY9/D40P8vAnsONw'
      'xc9ckDoP75CiL/7v7U74MPmwi7FpDxwPe0+u8FzwEDErz9cQAhCC8LXv/w9kkIFQ25FVESaPEt'
      'F0UVaRoNJ+kalSfhGJkqbSVtNcE0IsoFPoE0LTkLyaJA=')
    nutils.numeric.assert_allclose64(lhs, 'eNoB2AAn/0rRvC+0MH0uQtBhz7jLODUgNvgzxc'
      'r/yQ7IwzjiOTw3NsdexmLE6zvjPDg6AsTJw57BYzrDQJ/BFsQ5wSXA3TxTyHdBt8LFvzsxHzI+'
      'MMfO9s3/zyg1HDY0NNjKAcomzJw4kTnfN1THqcbHyBc7mTzZOpfE38PhxkQ9+z4zP8nAm8OLwx'
      'c9ckDoP75CiL/7v5871MPUwgnFljxoPcM+g8FewI/DkT9JQMJBwsAhv4fGXkLeQhBFcUNkv9xE'
      'eUUeRixJDEb2SCFC4klJSQhNIk0IskRPQk24TYBVaVU=')

  def test_rot1(self):
    lhs0, lhs = main(nelems=3, reynolds=1e-2, timestep=.1, endtime=.05, rotation=1)
    nutils.numeric.assert_allclose64(lhs0, 'eNoB2AAn/0nRvS+1MHguQNBiz7bLOTUiNvIzw'
      '8oByg3IwzjkOTY3NcdgxmDE6zvmPCk6AsTKw57BYzrDQJ/BFsQ5wSXA3TxTyHdBt8LFvyQ0YTT'
      '1M/kygzJRM742OjdgNnAyU82VNP04zjlmOMXH+cbqyTQ7pzz5OqnE8cM9x089/z43P87ArMOaw'
      'xc9ckDoP75CiL/7v7U74MPmwi7FpDxwPe0+u8FzwEDErz9cQAhCC8LXv/w9kkIFQ25FVESaPEt'
      'F0UVaRoNJ+kalSfhGJkqbSVtNcE0IsoFPoE0LTlkWaMs=')
    nutils.numeric.assert_allclose64(lhs, 'eNoB2AAn/03Rvi+0MHouQNBhz7rLOTUgNvYzxM'
      'r/yQ/IwzjiOTs3NsdexmLE6zvjPDg6AsTJw57BYzrDQJ/BFsQ5wSXA3TxTyHdBt8LFvyU0YTT1'
      'M/kygzJRM782OjdgNncyUs2TNP04zjlmOMfH+MblyTQ7pjz5OrHE8MM5x0w9/j42P8zAqcOYwx'
      'c9ckDoP75CiL/7v5g7z8PUwgzFlzxnPb4+fcFdwJjDkj9IQL5BtsAgvyvHX0LdQg5FbENfv91E'
      'eUUdRixJCEb1SB1C4klJSQdNIU0IskRPQk24TftIaAU=')
