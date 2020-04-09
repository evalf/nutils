# Copyright (c) 2014 Evalf
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""
The mesh module provides mesh generators: methods that return a topology and an
accompanying geometry function. Meshes can either be generated on the fly, e.g.
:func:`rectilinear`, or read from external an externally prepared file,
:func:`gmsh`, and converted to nutils format. Note that no mesh writers are
provided at this point.
"""

from . import topology, function, util, element, elementseq, numpy, numeric, transform, transformseq, warnings, types, cache, _
import os, itertools, re, math, treelog as log, io, contextlib

# MESH GENERATORS

@log.withcontext
def rectilinear(richshape, periodic=(), name='rect'):
  'rectilinear mesh'

  ndims = len(richshape)
  shape = []
  offset = []
  scale = []
  uniform = True
  for v in richshape:
    if numeric.isint(v):
      assert v > 0
      shape.append(v)
      scale.append(1)
      offset.append(0)
    elif numpy.equal(v, numpy.linspace(v[0],v[-1],len(v))).all():
      shape.append(len(v)-1)
      scale.append((v[-1]-v[0]) / float(len(v)-1))
      offset.append(v[0])
    else:
      shape.append(len(v)-1)
      uniform = False

  root = transform.Identifier(ndims, name)
  axes = [transformseq.DimAxis(0,n,idim in periodic) for idim, n in enumerate(shape)]
  topo = topology.StructuredTopology(root, axes)

  if uniform:
    if all(o == offset[0] for o in offset[1:]):
      offset = offset[0]
    if all(s == scale[0] for s in scale[1:]):
      scale = scale[0]
    geom = function.rootcoords(ndims) * scale + offset
  else:
    funcsp = topo.basis('spline', degree=1, periodic=())
    coords = numeric.meshgrid(*richshape).reshape(ndims, -1)
    geom = (funcsp * coords).sum(-1)

  return topo, geom

def line(nodes, periodic=False, bnames=None):
  if isinstance(nodes, int):
    uniform = True
    assert nodes > 0
    nelems = nodes
    scale = 1
    offset = 0
  else:
    nelems = len(nodes)-1
    scale = (nodes[-1]-nodes[0]) / nelems
    offset = nodes[0]
    uniform = numpy.equal(nodes, offset + numpy.arange(nelems+1) * scale).all()
  root = transform.Identifier(1, 'line')
  domain = topology.StructuredLine(root, 0, nelems, periodic=periodic, bnames=bnames)
  geom = function.rootcoords(1) * scale + offset if uniform else domain.basis('std', degree=1, periodic=False).dot(nodes)
  return domain, geom

def newrectilinear(nodes, periodic=None, bnames=[['left','right'],['bottom','top'],['front','back']]):
  if periodic is None:
    periodic = numpy.zeros(len(nodes), dtype=bool)
  else:
    periodic = numpy.asarray(periodic)
    assert len(periodic) == len(nodes) and periodic.ndim == 1 and periodic.dtype == bool
  dims = [line(nodesi, periodici, bnamesi) for nodesi, periodici, bnamesi in zip(nodes, periodic, tuple(bnames)+(None,)*len(nodes))]
  domain, geom = dims.pop(0)
  for domaini, geomi in dims:
    domain = domain * domaini
    geom = function.concatenate(function.bifurcate(geom,geomi))
  return domain, geom

@log.withcontext
def multipatch(patches, nelems, patchverts=None, name='multipatch'):
  '''multipatch rectilinear mesh generator

  Generator for a :class:`~nutils.topology.MultipatchTopology` and geometry.
  The :class:`~nutils.topology.MultipatchTopology` consists of a set patches,
  where each patch is a :class:`~nutils.topology.StructuredTopology` and all
  patches have the same number of dimensions.

  The ``patches`` argument, a :class:`numpy.ndarray`-like with shape
  ``(npatches, 2*ndims)`` or ``(npatches,)+(2,)*ndims``, defines the
  connectivity by labelling the patch vertices.  For example, three
  one-dimensional patches can be connected at one edge by::

      # connectivity:     3
      #                   │
      #                1──0──2

      patches=[[0,1], [0,2], [0,3]]

  Or two two-dimensional patches along an edge by::

      # connectivity:  3──4──5
      #                │  │  │
      #                0──1──2

      patches=[[[0,3],[1,4]], [[1,4],[2,5]]]

  The geometry is specified by the ``patchverts`` argument: a
  :class:`numpy.ndarray`-like with shape ``(nverts,ngeomdims)`` specifying for
  each vertex a coordinate.  Note that the dimension of the geometry may be
  higher than the dimension of the patches.  The created geometry is a
  patch-wise linear interpolation of the vertex coordinates.  If the
  ``patchverts`` argument is omitted the geometry describes a unit hypercube
  per patch.

  The ``nelems`` argument is either an :class:`int` defining the number of
  elements per patch per dimension, or a :class:`dict` with edges (a pair of
  vertex numbers) as keys and the number of elements (:class:`int`) as values,
  with key ``None`` specifying the default number of elements.  Example::

      # connectivity:  3─────4─────5
      #                │ 4x3 │ 8x3 │
      #                0─────1─────2

      patches=[[[0,3],[1,4]], [[1,4],[2,5]]]
      nelems={None: 4, (1,2): 8, (4,5): 8, (0,3): 3, (1,4): 3, (2,5): 3}

  Since the patches are structured topologies, the number of elements per
  patch per dimension should be unambiguous.  In above example specifying
  ``nelems={None: 4, (1,2): 8}`` will raise an exception because the patch on
  the right has 8 elements along edge ``(1,2)`` and 4 along ``(4,5)``.

  Example
  -------

  An L-shaped domain can be generated by::

  >>> # connectivity:  2──5
  >>> #                │  |
  >>> #                1──4─────7     y
  >>> #                │  │     │     │
  >>> #                0──3─────6     └──x

  >>> domain, geom = multipatch(
  ...   patches=[[0,1,3,4], [1,2,4,5], [3,4,6,7]],
  ...   patchverts=[[0,0], [0,1], [0,2], [1,0], [1,1], [1,2], [3,0], [3,1]],
  ...   nelems={None: 4, (3,6): 8, (4,7): 8})

  The number of elements is chosen such that all elements in the domain have
  the same size.

  A topology and geometry describing the surface of a sphere can be generated
  by creating a multipatch cube surface and inflating the cube to a sphere:

  >>> # connectivity:    3────7
  >>> #                 ╱│   ╱│
  >>> #                2────6 │     y
  >>> #                │ │  │ │     │
  >>> #                │ 1──│─5     │ z
  >>> #                │╱   │╱      │╱
  >>> #                0────4       *────x

  >>> import itertools
  >>> from nutils import function
  >>> topo, cube = multipatch(
  ...   patches=[
  ...     [0,1,2,3], # left,   normal: x
  ...     [4,5,6,7], # right,  normal: x
  ...     [0,1,4,5], # bottom, normal: -y
  ...     [2,3,6,7], # top,    normal: -y
  ...     [0,2,4,6], # front,  normal: z
  ...     [1,3,5,7], # back,   normal: z
  ...   ],
  ...   patchverts=tuple(itertools.product(*([[-1,1]]*3))),
  ...   nelems=1)
  >>> sphere = function.normalized(cube)

  The normals of the patches are determined by the order of the vertex numbers.
  An outward normal for the cube is obtained by flipping the left, top and
  front faces:

  >>> cubenormal = cube.normal(exterior=True) * topo.basis('patch').dot([-1,1,1,-1,-1,1])

  At the centroids of the faces the outward normal should equal the cube geometry:

  >>> numpy.testing.assert_allclose(*topo.sample('gauss', 1).eval([cubenormal, cube]))

  Similarly, the outward normal of the sphere is obtained by:

  >>> spherenormal = sphere.normal(exterior=True) * topo.basis('patch').dot([-1,1,1,-1,-1,1])
  >>> numpy.testing.assert_allclose(*topo.sample('gauss', 1).eval([spherenormal, cube]))

  Args
  ----
  patches:
      A :class:`numpy.ndarray` with shape sequence of patches with each patch being a list of vertex indices.
  patchverts:
      A sequence of coordinates of the vertices.
  nelems:
      Either an :class:`int` specifying the number of elements per patch per
      dimension, or a :class:`dict` with edges (a pair of vertex numbers) as
      keys and the number of elements (:class:`int`) as values, with key
      ``None`` specifying the default number of elements.

  Returns
  -------
  :class:`nutils.topology.MultipatchTopology`:
      The multipatch topology.
  :class:`nutils.function.Array`:
      The geometry defined by the ``patchverts`` or a unit hypercube per patch
      if ``patchverts`` is not specified.
  '''

  patches = numpy.array(patches)
  if patches.dtype != int:
    raise ValueError('`patches` should be an array of ints.')
  if patches.ndim < 2 or patches.ndim == 2 and patches.shape[-1] % 2 != 0:
    raise ValueError('`patches` should be an array with shape (npatches,2,...,2) or (npatches,2*ndims).')
  elif patches.ndim > 2 and patches.shape[1:] != (2,) * (patches.ndim - 1):
    raise ValueError('`patches` should be an array with shape (npatches,2,...,2) or (npatches,2*ndims).')
  patches = patches.reshape(patches.shape[0], -1)

  # determine topological dimension of patches

  ndims = 0
  while 2**ndims < patches.shape[1]:
    ndims += 1
  if 2**ndims > patches.shape[1]:
    raise ValueError('Only hyperrectangular patches are supported: ' \
      'number of patch vertices should be a power of two.')
  patches = patches.reshape([patches.shape[0]] + [2]*ndims)

  # group all common patch edges (and/or boundaries?)

  if isinstance(nelems, int):
    nelems = {None: nelems}
  elif isinstance(nelems, dict):
    nelems = {(k and frozenset(k)): v for k, v in nelems.items()}
  else:
    raise ValueError('`nelems` should be an `int` or `dict`')

  # create patch topologies, geometries

  if patchverts is not None:
    patchverts = numpy.array(patchverts)
    indices = set(patches.flat)
    if tuple(sorted(indices)) != tuple(range(len(indices))):
      raise ValueError('Patch vertices in `patches` should be numbered consecutively, starting at 0.')
    if len(patchverts) != len(indices):
      raise ValueError('Number of `patchverts` does not equal number of vertices specified in `patches`.')
    if len(patchverts.shape) != 2:
      raise ValueError('Every patch vertex should be an array of dimension 1.')

  topos = []
  coords = []
  for i, patch in enumerate(patches):
    # find shape of patch and local patch coordinates
    shape = []
    for dim in range(ndims):
      nelems_sides = []
      sides = [(0,1)]*ndims
      sides[dim] = slice(None),
      for side in itertools.product(*sides):
        sideverts = frozenset(patch[side])
        if sideverts in nelems:
          nelems_sides.append(nelems[sideverts])
        else:
          nelems_sides.append(nelems[None])
      if len(set(nelems_sides)) != 1:
        raise ValueError('duplicate number of elements specified for patch {} in dimension {}'.format(i, dim))
      shape.append(nelems_sides[0])
    # create patch topology
    topos.append(rectilinear(shape, name='{}{}'.format(name, i))[0])
    # compute patch geometry
    patchcoords = [numpy.linspace(0, 1, n+1) for n in shape]
    patchcoords = numeric.meshgrid(*patchcoords).reshape(ndims, -1)
    if patchverts is not None:
      patchcoords = numpy.array([
        sum(
          patchverts[j]*util.product(c if s else 1-c for c, s in zip(coord, side))
          for j, side in zip(patch.flat, itertools.product(*[[0,1]]*ndims))
       )
        for coord in patchcoords.T
      ]).T
    coords.append(patchcoords)

  # build patch boundary data

  boundarydata = topology.MultipatchTopology.build_boundarydata(patches)

  # join patch topologies, geometries

  topo = topology.MultipatchTopology(tuple(map(topology.Patch, topos, patches, boundarydata)))
  funcsp = topo.basis('spline', degree=1, patchcontinuous=False)
  geom = (funcsp * numpy.concatenate(coords, axis=1)).sum(-1)

  return topo, geom

@cache.function
def parsegmsh(mshdata):
  """Gmsh parser

  Parser for Gmsh data in ``msh2`` or ``msh4`` format. See the `Gmsh manual
  <http://geuz.org/gmsh/doc/texinfo/gmsh.html>`_ for details.

  Parameters
  ----------
  mshdata : :class:`io.BufferedIOBase`
      Msh file contents.

  Returns
  -------
  :class:`dict`:
      Keyword arguments for :func:`simplex`
  """

  try:
    from meshio import gmsh
  except ImportError as e:
    raise Exception('parsegmsh requires the meshio module to be installed') from e

  msh = gmsh.main.read_buffer(mshdata)

  # Coords is a 2d float-array such that coords[inode,idim] == coordinate.
  coords = msh.points

  # Nodes is a dictionary that maps a topological dimension to a 2d int-array
  # dictionary such that nodes[nd][ielem,ilocal] == inode, where ilocal < nd+1
  # for linear geometries or larger for higher order geometries. Since meshio
  # stores nodes by simplex type and cell, simplex types are mapped to
  # dimensions and gathered, after which cells are concatenated under the
  # assumption that there is only one simplex type per dimension.
  nodes = {('ver','lin','tri','tet').index(typename[:3]): numpy.concatenate(datas, axis=0)
    for typename, datas in util.gather((cells.type, cells.data) for cells in msh.cells)}

  # Identities is a 2d [master, slave] int-aray that pairs matching nodes on
  # periodic walls. For the topological connectivity, all slaves in the nodes
  # arrays will be replaced by their master counterpart.
  identities = numpy.zeros((0, 2), dtype=int) if not msh.gmsh_periodic \
    else numpy.concatenate([d for a, b, c, d in msh.gmsh_periodic], axis=0)

  # Tags is a list of (nd, name, ndelems) tuples that define topological groups
  # per dimension. Since meshio associates group names with cells, which are
  # concatenated in nodes, element ids are offset and concatenated to match.
  tags = [(nd, name, numpy.concatenate([(data == itag).nonzero()[0]
    + sum(len(cells.data) for cells in msh.cells[:icell] if cells.type == msh.cells[icell].type) # offset into nodes
                                         for icell, data in enumerate(msh.cell_data['gmsh:physical'])]))
           for name, (itag, nd) in msh.field_data.items()]

  # determine the dimension of the topology
  ndims = max(nodes)

  # determine the dimension of the geometry
  assert not numpy.isnan(coords).any()
  while coords.shape[1] > ndims and not coords[:,-1].any():
    coords = coords[:,:-1]

  # separate geometric, topological nodes
  cnodes = nodes[ndims]
  if cnodes.shape[1] > ndims+1: # higher order geometry
    nodes = {nd: n[:,:nd+1] for nd, n in nodes.items()} # remove high order info

  if len(identities):
    slaves, masters = identities.T
    keep = numpy.ones(len(coords), dtype=bool)
    keep[slaves] = False
    assert keep[masters].all()
    renumber = keep.cumsum()-1
    renumber[slaves] = renumber[masters]
    nodes = {nd: renumber[n] for nd, n in nodes.items()}

  vnodes = nodes[ndims]
  bnodes = nodes.get(ndims-1)
  pnodes = nodes.get(0)

  if cnodes is vnodes: # geometry is linear and non-periodic, dofs follow in-place sorting of nodes
    degree = 1
  elif cnodes.shape[1] == ndims+1: # linear elements: match sorting of nodes
    degree = 1
    shuffle = vnodes.argsort(axis=1)
    cnodes = cnodes[numpy.arange(len(cnodes))[:,_], shuffle] # gmsh conveniently places the primary ndim+1 vertices first
  else: # higher order elements: match sorting of nodes and renumber higher order coefficients
    degree, nodeorder = { # for meshio's node ordering conventions see http://www.vtk.org/VTK/img/file-formats.pdf
      (2, 6): (2, (0,3,1,5,4,2)),
      (2,10): (3, (0,3,4,1,8,9,5,7,6,2)),
      (2,15): (4, (0,3,4,5,1,11,12,13,6,10,14,7,9,8,2)),
      (3,10): (2, (0,4,1,6,5,2,7,8,9,3))}[ndims, cnodes.shape[1]]
    enum = numpy.empty([degree+1]*(ndims+1), dtype=int)
    bari = tuple(numpy.array([index[::-1] for index in numpy.ndindex(*enum.shape) if sum(index) == degree]).T)
    enum[bari] = numpy.arange(cnodes.shape[1]) # maps baricentric index to corresponding enumerated index
    shuffle = vnodes.argsort(axis=1)
    cnodes = cnodes[:,nodeorder] # convert from gmsh to nutils order
    for i in range(ndims): # strategy: apply shuffle to cnodes by sequentially swapping vertices...
      for j in range(i+1, ndims+1): # ...considering all j > i pairs...
        m = shuffle[:,i] == j # ...and swap vertices if vertex j is shuffled into i...
        r = enum.swapaxes(i,j)[bari] # ...using the enum table to generate the appropriate renumbering
        cnodes[m,:] = cnodes[numpy.ix_(m,r)]
        m = shuffle[:,j] == i
        shuffle[m,j] = shuffle[m,i] # update shuffle to track changed vertex positions

  vnodes.sort(axis=1)
  nnodes = vnodes[:,-1].max()+1

  vtags, btags, ptags = {}, {}, {}
  edge_vertices = numpy.arange(ndims+1).repeat(ndims).reshape(ndims, ndims+1)[:,::-1].T # nedges x ndims
  for nd, name, ielems in tags:
    if nd == ndims:
      vtags[name] = numpy.array(ielems)
    elif nd == ndims-1:
      edgenodes = bnodes[ielems]
      nodemask = numeric.asboolean(edgenodes.ravel(), size=nnodes, ordered=False)
      ielems, = (nodemask[vnodes].sum(axis=1) >= ndims).nonzero() # all elements sharing at least ndims edgenodes
      edgemap = {tuple(b): (ielem, iedge) for ielem, a in zip(ielems, vnodes[ielems[:,_,_], edge_vertices[_,:,:]]) for iedge, b in enumerate(a)}
      btags[name] = numpy.array([edgemap[tuple(sorted(n))] for n in edgenodes])
    elif nd == 0:
      ptags[name] = pnodes[ielems][...,0]

  log.info('\n- '.join(['loaded {}d gmsh topology consisting of #{} elements'.format(ndims, len(cnodes))]
    + [name + ' groups: ' + ', '.join('{} #{}'.format(n, len(e)) for n, e in tags.items())
      for name, tags in (('volume', vtags), ('boundary', btags), ('point', ptags)) if tags]))

  return dict(nodes=vnodes, cnodes=cnodes, coords=coords, tags=vtags, btags=btags, ptags=ptags)

@log.withcontext
@types.apply_annotations
def gmsh(fname:util.binaryfile, name='gmsh'):
  """Gmsh parser

  Parser for Gmsh files in `.msh` format. Only files with physical groups are
  supported. See the `Gmsh manual
  <http://geuz.org/gmsh/doc/texinfo/gmsh.html>`_ for details.

  Parameters
  ----------
  fname : :class:`str` or :class:`io.BufferedIOBase`
      Path to mesh file or mesh file object.
  name : :class:`str` or :any:`None`
      Name of parsed topology, defaults to 'gmsh'.

  Returns
  -------
  topo : :class:`nutils.topology.SimplexTopology`
      Topology of parsed Gmsh file.
  geom : :class:`nutils.function.Array`
      Isoparametric map.
  """

  with fname as f:
    return simplex(name=name, **parsegmsh(f))

def simplex(nodes, cnodes, coords, tags, btags, ptags, name='simplex'):
  '''Simplex topology.

  Parameters
  ----------
  nodes : :class:`numpy.ndarray`
      Vertex indices as (nelems x ndims+1) integer array, sorted along the
      second dimension. This table fully determines the connectivity of the
      simplices.
  cnodes : :class:`numpy.ndarray`
      Coordinate indices as (nelems x ncnodes) integer array following Nutils'
      conventions for Bernstein polynomials. The polynomial degree is inferred
      from the array shape.
  coords : :class:`numpy.ndarray`
      Coordinates as (nverts x ndims) float array to be indexed by ``cnodes``.
  tags : :class:`dict`
      Dictionary of name->element numbers. Element order is preserved in the
      resulting volumetric groups.
  btags : :class:`dict`
      Dictionary of name->edges, where edges is a (nedges x 2) integer array
      containing pairs of element number and edge number. The segments are
      assigned to boundary or interfaces groups automatically while otherwise
      preserving order.
  ptags : :class:`dict`
      Dictionary of name->node numbers referencing the ``nodes`` table.
  name : :class:`str`
      Name of simplex topology.

  Returns
  -------
  topo : :class:`nutils.topology.SimplexTopology`
      Topology with volumetric, boundary and interface groups.
  geom : :class:`nutils.function.Array`
      Geometry function.
  '''

  nverts = len(coords)
  nelems, ncnodes = cnodes.shape
  ndims = nodes.shape[1] - 1
  assert len(nodes) == nelems
  assert numpy.greater(nodes[:,1:], nodes[:,:-1]).all(), 'nodes must be sorted'

  if ncnodes == ndims+1:
    degree = 1
    vnodes = cnodes
  else:
    degree = int((ncnodes * math.factorial(ndims))**(1/ndims))-1  # degree**ndims/ndims! < ncnodes < (degree+1)**ndims/ndims!
    dims = numpy.arange(ndims)
    strides = (dims+1+degree).cumprod() // (dims+1).cumprod() # (i+1+degree)!/(i+1)!
    assert strides[-1] == ncnodes
    vnodes = cnodes[:,(0,*strides-1)]

  assert vnodes.shape == nodes.shape
  transforms = transformseq.IdentifierTransforms(ndims=ndims, name=name, length=nelems)
  topo = topology.SimplexTopology(nodes, transforms, transforms)
  coeffs = element.getsimplex(ndims).get_poly_coeffs('lagrange', degree=degree)
  basis = function.PlainBasis([coeffs] * nelems, cnodes, nverts, topo.transforms)
  geom = (basis[:,_] * coords).sum(0)

  connectivity = topo.connectivity

  bgroups = {}
  igroups = {}
  for name, elems_edges in btags.items():
    bitems = [], [], None
    iitems = [], [], []
    for ielem, iedge in elems_edges:
      ioppelem = connectivity[ielem, iedge]
      simplices, transforms, opposites = bitems if ioppelem == -1 else iitems
      simplices.append(tuple(nodes[ielem][:iedge])+tuple(nodes[ielem][iedge+1:]))
      transforms.append(topo.transforms[ielem] + (transform.SimplexEdge(ndims, iedge),))
      if opposites is not None:
        opposites.append(topo.transforms[ioppelem] + (transform.SimplexEdge(ndims, tuple(connectivity[ioppelem]).index(ielem)),))
    for groups, (simplices, transforms, opposites) in (bgroups, bitems), (igroups, iitems):
      if simplices:
        transforms = transformseq.PlainTransforms(transforms, ndims-1)
        opposites = transforms if opposites is None else transformseq.PlainTransforms(opposites, ndims-1)
        groups[name] = topology.SimplexTopology(simplices, transforms, opposites)

  pgroups = {}
  if ptags:
    ptrans = [transform.Matrix(linear=numpy.zeros(shape=(ndims,0)), offset=offset) for offset in numpy.eye(ndims+1)[:,1:]]
    pmap = {inode: numpy.array(numpy.equal(nodes, inode).nonzero()).T for inode in set.union(*map(set, ptags.values()))}
    for pname, inodes in ptags.items():
      ptransforms = transformseq.PlainTransforms([topo.transforms[ielem] + (ptrans[ivertex],) for inode in inodes for ielem, ivertex in pmap[inode]], 0)
      preferences = elementseq.asreferences([element.getsimplex(0)], 0)*len(ptransforms)
      pgroups[pname] = topology.Topology(preferences, ptransforms, ptransforms)

  vgroups = {}
  for name, ielems in tags.items():
    if len(ielems) == nelems and numpy.equal(ielems, numpy.arange(nelems)).all():
      vgroups[name] = topo.withgroups(bgroups=bgroups, igroups=igroups, pgroups=pgroups)
      continue
    transforms = topo.transforms[ielems]
    vtopo = topology.SimplexTopology(nodes[ielems], transforms, transforms)
    keep = numpy.zeros(nelems, dtype=bool)
    keep[ielems] = True
    vbgroups = {}
    vigroups = {}
    for bname, elems_edges in btags.items():
      bitems = [], [], []
      iitems = [], [], []
      for ielem, iedge in elems_edges:
        ioppelem = connectivity[ielem, iedge]
        if ioppelem == -1:
          keepopp = False
        else:
          keepopp = keep[ioppelem]
          ioppedge = tuple(connectivity[ioppelem]).index(ielem)
        if keepopp and keep[ielem]:
          simplices, transforms, opposites = iitems
        elif keepopp or keep[ielem]:
          simplices, transforms, opposites = bitems
          if keepopp:
            ielem, iedge, ioppelem, ioppedge = ioppelem, ioppedge, ielem, iedge
        else:
          continue
        simplices.append(tuple(nodes[ielem][:iedge])+tuple(nodes[ielem][iedge+1:]))
        transforms.append(topo.transforms[ielem] + (transform.SimplexEdge(ndims, iedge),))
        if ioppelem != -1:
          opposites.append(topo.transforms[ioppelem] + (transform.SimplexEdge(ndims, ioppedge),))
      for groups, (simplices, transforms, opposites) in (vbgroups, bitems), (vigroups, iitems):
        if simplices:
          transforms = transformseq.PlainTransforms(transforms, ndims-1)
          opposites = transformseq.PlainTransforms(opposites, ndims-1) if len(opposites) == len(transforms) else transforms
          groups[bname] = topology.SimplexTopology(simplices, transforms, opposites)
    vpgroups = {}
    for pname, inodes in ptags.items():
      ptransforms = transformseq.PlainTransforms([topo.transforms[ielem] + (ptrans[ivertex],) for inode in inodes for ielem, ivertex in pmap[inode] if keep[ielem]], 0)
      preferences = elementseq.asreferences([element.getsimplex(0)], 0)*len(ptransforms)
      vpgroups[pname] = topology.Topology(preferences, ptransforms, ptransforms)
    vgroups[name] = vtopo.withgroups(bgroups=vbgroups, igroups=vigroups, pgroups=vpgroups)

  return topo.withgroups(vgroups=vgroups, bgroups=bgroups, igroups=igroups, pgroups=pgroups), geom

def fromfunc(func, nelems, ndims, degree=1):
  'piecewise'

  if isinstance(nelems, int):
    nelems = [nelems]
  assert len(nelems) == func.__code__.co_argcount
  topo, ref = rectilinear([numpy.linspace(0,1,n+1) for n in nelems])
  funcsp = topo.basis('spline', degree=degree).vector(ndims)
  coords = topo.projection(func, onto=funcsp, coords=ref, exact_boundaries=True)
  return topo, coords

def unitsquare(nelems, etype):
  '''Unit square mesh.

  Args
  ----
  nelems : :class:`int`
      Number of elements along boundary
  etype : :class:`str`
      Type of element used for meshing. Supported are:

      * ``"square"``: structured mesh of squares.

      * ``"triangle"``: unstructured mesh of triangles.

      * ``"mixed"``: unstructured mesh of triangles and squares.

  Returns
  -------
  :class:`nutils.topology.Topology`:
      The structured/unstructured topology.
  :class:`nutils.function.Array`:
      The geometry function.
  '''

  root = transform.Identifier(2, 'unitsquare')

  if etype == 'square':
    topo = topology.StructuredTopology(root, [transformseq.DimAxis(0, nelems, False)] * 2)

  elif etype in ('triangle', 'mixed'):
    simplices = numpy.concatenate([
      numpy.take([i*(nelems+1)+j, i*(nelems+1)+j+1, (i+1)*(nelems+1)+j, (i+1)*(nelems+1)+j+1], [[0,1,2],[1,2,3]] if i%2==j%2 else [[0,1,3],[0,2,3]], axis=0)
        for i in range(nelems) for j in range(nelems)])

    v = numpy.arange(nelems+1, dtype=float)
    coords = numeric.meshgrid(v, v).reshape(2,-1).T
    transforms = transformseq.PlainTransforms([(root, transform.Square((c[1:]-c[0]).T, c[0])) for c in coords[simplices]], 2)
    topo = topology.SimplexTopology(simplices, transforms, transforms)

    if etype == 'mixed':
      references = list(topo.references)
      transforms = list(topo.transforms)
      square = element.getsimplex(1)**2
      connectivity = list(topo.connectivity)
      isquares = [i * nelems + j for i in range(nelems) for j in range(nelems) if i%2==j%3]
      for n in sorted(isquares, reverse=True):
        i, j = divmod(n, nelems)
        references[n*2:(n+1)*2] = square,
        transforms[n*2:(n+1)*2] = (root, transform.Shift([float(i),float(j)])),
        connectivity[n*2:(n+1)*2] = numpy.concatenate(connectivity[n*2:(n+1)*2])[[3,2,4,1] if i%2==j%2 else [3,2,0,5]],
        connectivity = [c-numpy.greater(c,n*2) for c in connectivity]
      topo = topology.ConnectedTopology(elementseq.asreferences(references, 2), transformseq.PlainTransforms(transforms, 2),transformseq.PlainTransforms(transforms, 2), tuple(types.frozenarray(c, copy=False) for c in connectivity))

    x, y = topo.boundary.elem_mean(function.rootcoords(2), degree=1).T
    bgroups = dict(left=x==0, right=x==nelems, bottom=y==0, top=y==nelems)
    topo = topo.withboundary(**{name: topo.boundary[numpy.where(mask)[0]] for name, mask in bgroups.items()})

  else:
    raise Exception('invalid element type {!r}'.format(etype))

  return topo, function.rootcoords(2) / nelems

# vim:sw=2:sts=2:et
