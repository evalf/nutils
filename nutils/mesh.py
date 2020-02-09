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
import os, itertools, re, math, treelog as log

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

      # connectivity:  2──5
      #                │  |
      #                1──4─────7     y
      #                │  │     │     │
      #                0──3─────6     └──x

      domain, geom = mesh.multipatch(
        patches=[[0,1,3,4], [1,2,4,5], [3,4,6,7]],
        patchverts=[[0,0], [0,1], [0,2], [1,0], [1,1], [1,2], [3,0], [3,1]],
        nelems={None: 4, (3,6): 8, (4,7): 8})

  The number of elements is chosen such that all elements in the domain have
  the same size.

  A topology and geometry describing the surface of a sphere can be generated
  by creating a multipatch cube surface and inflating the cube to a sphere::

      # connectivity:    3────7
      #                 ╱│   ╱│
      #                2────6 │     y
      #                │ │  │ │     │
      #                │ 1──│─5     │ z
      #                │╱   │╱      │╱
      #                0────4       *────x

      topo, cube = multipatch(
        patches=[
          # The order of the vertices is chosen such that normals point outward.
          [2,3,0,1],
          [4,5,6,7],
          [4,6,0,2],
          [1,3,5,7],
          [1,5,0,4],
          [2,6,3,7],
        ],
        patchverts=tuple(itertools.product(*([[-1,1]]*3))),
        nelems=10,
     )
      sphere = cube / function.sqrt((cube**2).sum(0))

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

def _parsegmsh2(sections):
  '''helper function to parse msh2 format'''

  # parse section PhysicalNames
  if 'PhysicalNames' not in sections:
    tagmapbydim = None
  else:
    N, *PhysicalNames = sections.pop('PhysicalNames').splitlines()
    assert int(N) == len(PhysicalNames)
    tagmapbydim = {}, {}, {}, {} # tagid->tagname dictionary
    for line in PhysicalNames:
      nd, tagid, tagname = line.split(' ', 2)
      nd = int(nd)
      tagmapbydim[nd][tagid] = tagname.strip('"')

  # parse section Nodes
  N, *Nodes = sections.pop('Nodes').splitlines()
  nnodes = len(Nodes)
  assert int(N) == nnodes
  nodes = numpy.empty((nnodes, 3))
  nodemap = {}
  for i, line in enumerate(Nodes):
    n, *c = line.split()
    nodemap[n] = i
    nodes[i] = c
  assert not numpy.isnan(nodes).any()

  # parse section Elements
  N, *Elements = sections.pop('Elements').splitlines()
  assert int(N) == len(Elements)
  inodesbydim = [], [], [], [] # nelems-list of 4-tuples of node numbers
  tagnamesbydim = {}, {}, {}, {} # tag->ielems dictionary
  etype2nd = {'15': 0, '1': 1, '2': 2, '4': 3, '8': 1, '9': 2, '11': 3, '26': 1, '21': 2, '23': 2, '27': 1}
  for line in Elements:
    n, e, t, m, *w = line.split()
    nd = etype2nd[e]
    ntags = int(t) - 1
    assert ntags >= 0
    inodes = tuple(nodemap[nodeid] for nodeid in w[ntags:])
    if not inodesbydim[nd] or inodesbydim[nd][-1] != inodes: # multiple tags are repeated in consecutive lines
      inodesbydim[nd].append(inodes)
    if tagmapbydim:
      tagname = tagmapbydim[nd][m]
      tagnamesbydim[nd].setdefault(tagname, []).append(len(inodesbydim[nd])-1)
  inodesbydim = [numpy.array(e) if e else numpy.empty((0,nd+1), dtype=int) for nd, e in enumerate(inodesbydim)]

  # parse section Periodic
  N, *Periodic = sections.pop('Periodic', '0').splitlines()
  nperiodic = int(N)
  vertex_identities = [] # slave, master
  n = 0
  for line in Periodic:
    words = line.split()
    if words[0] == 'Affine':
      pass
    elif len(words) == 1:
      n = int(words[0]) # initialize for counting backwards
    elif len(words) == 2:
      vertex_identities.append([nodemap[w] for w in words])
      n -= 1
    else:
      assert len(words) == 3 # discard content
      assert n == 0 # check if batch of slave/master nodes matches announcement
      nperiodic -= 1
  assert nperiodic == 0 # check if number of periodic blocks matches announcement
  assert n == 0 # check if last batch of slave/master nodes matches announcement

  return inodesbydim, vertex_identities, nodes, tagnamesbydim

def _parsegmsh4(sections):
  '''helper function to parse msh4 format'''

  # parse section PhysicalNames
  PhysicalNames = sections.pop('PhysicalNames').splitlines()
  nnames = int(PhysicalNames[0])
  lines = PhysicalNames[1:]
  assert len(lines) == nnames
  names = {} # nd, nametag -> name
  for line in lines:
    nd, nametag, tagname = line.split(' ', 2)
    names[int(nd), nametag] = tagname.strip('"')

  # parse section Entities
  Entities = sections.pop('Entities').splitlines()
  header = Entities[0]
  lines = Entities[1:]
  entities = {} # nd, entitytag -> nametags
  for nd, nentities in enumerate(map(int, header.split())):
    for line in lines[:nentities]:
      parts = line.split()
      ntags, *tags = parts[7 if nd else 4:]
      entities[nd, parts[0]] = tags[:int(ntags)]
    lines = lines[nentities:]
  assert not lines

  # parse section Nodes
  Nodes = sections.pop('Nodes').splitlines()
  lines = iter(Nodes)
  countdown = int(next(lines).split()[0])
  coords = numpy.empty([0, 3], dtype=float) # inode x 3
  nodemap = {} # nodeid -> inode
  for header in lines:
    ndims, tag, parametric, nnodes = header.split()
    a = len(coords)
    b = a + int(nnodes)
    for i in range(a, b):
      n = next(lines)
      assert n not in nodemap
      nodemap[n] = i
    coords.resize((b, 3), refcheck=False)
    for i in range(a, b):
      coords[i] = [float(s) for s in next(lines).split()]
    countdown -= 1
  assert countdown == 0

  # parse section Elements
  Elements = sections.pop('Elements').splitlines()
  lines = iter(Elements)
  countdown = int(next(lines).split()[0])
  elements = {} # nd -> nodeids = ielem x nd+1
  elemtags = [] # nd, entitytag, elemslice
  for header in lines:
    ndims, tag, elemtype, nelems = header.split()
    inodes = elements.setdefault(int(ndims), [])
    a = len(inodes)
    inodes.extend(next(lines).split()[1:] for i in range(int(nelems)))
    b = len(inodes)
    elemtags.append((int(ndims), tag, range(a, b)))
    countdown -= 1
  assert countdown == 0

  # parse section Periodic
  Periodic = sections.pop('Periodic', '0').splitlines()
  lines = iter(Periodic)
  countdown = int(next(lines))
  vertex_identities = [] # slave, master
  for header in lines:
    ndims, tag, mastertag = header.split()
    affine = next(lines)
    npairs = int(next(lines))
    vertex_identities.extend(next(lines).split() for ipair in range(npairs))
    countdown -= 1
  assert countdown == 0

  # process data
  nodemap = numpy.vectorize(nodemap.__getitem__, otypes=[int])
  inodesbydim = [nodemap(elements[nd]) if nd in elements else numpy.empty([0,nd+1], dtype=int) for nd in range(4)]
  tagnamesbydim = [{} for nd in range(4)]
  for nd, tag, ielems in elemtags:
    for n in entities[nd, tag]:
      tagnamesbydim[nd].setdefault(names[nd, n], []).extend(ielems)
  vertex_identities = nodemap(vertex_identities).tolist()

  return inodesbydim, vertex_identities, coords, tagnamesbydim

@types.apply_annotations
@cache.function
def parsegmsh(fname:util.readtext, name='gmsh'):
  """Gmsh parser

  Parser for Gmsh files in `.msh` format. Only files with physical groups are
  supported. See the `Gmsh manual
  <http://geuz.org/gmsh/doc/texinfo/gmsh.html>`_ for details.

  Parameters
  ----------
  fname : :class:`str`
      Path to mesh file.

  Returns
  -------
  Keyword arguments for :func:`simplex`
  """

  # split sections
  sections = dict(re.findall(r'^\$(\w+)\n(.*)\n\$End\1$', fname, re.MULTILINE|re.DOTALL))

  # parse section MeshFormat
  MeshFormat = sections.pop('MeshFormat', None)
  if not MeshFormat:
    raise ValueError('invalid or incomplete gmsh data')
  version, filetype, datasize = MeshFormat.split()
  if filetype != '0':
    raise ValueError('binary gmsh data is not supported')

  # parse remaining sections
  if version.startswith('2.'):
    _parse = _parsegmsh2
  elif version.startswith('4.'):
    _parse = _parsegmsh4
  else:
    raise ValueError('gmsh version {} is not supported; please use -format msh4'.format(version))

  inodesbydim, vertex_identities, nodes, tagnamesbydim = _parse(sections)

  # determine the dimension of the topology
  while not inodesbydim[-1].size:
    inodesbydim.pop()
  ndims = len(inodesbydim)-1 # topological dimension

  # determine the dimension of the geometry
  assert not numpy.isnan(nodes).any()
  while nodes.shape[1] > ndims and not nodes[:,-1].any():
    nodes = nodes[:,:-1]

  # warn about unused sections
  for section in sections:
    warnings.warn('section {!r} defined but not used'.format(section))

  # separate geometric dofs and sort vertices
  geomdofs = inodesbydim[ndims]
  if geomdofs.shape[1] > ndims+1: # higher order geometry
    inodesbydim = [n[:,:i+1] for i, n in enumerate(inodesbydim)] # remove high order info

  if vertex_identities:
    slaves, masters = numpy.array(vertex_identities).T
    keep = numpy.ones(len(nodes), dtype=bool)
    keep[slaves] = False
    assert keep[masters].all()
    renumber = keep.cumsum()-1
    renumber[slaves] = renumber[masters]
    inodesbydim = [renumber[n] for n in inodesbydim]

  if geomdofs is inodesbydim[ndims]: # geometry is linear and non-periodic, dofs follow in-place sorting of inodesbydim
    degree = 1
  elif geomdofs.shape[1] == ndims+1: # linear elements: match sorting of inodesbydim
    degree = 1
    shuffle = inodesbydim[ndims].argsort(axis=1)
    geomdofs = geomdofs[numpy.arange(len(geomdofs))[:,_], shuffle] # gmsh conveniently places the primary ndim+1 vertices first
  else: # higher order elements: match sorting of inodesbydim and renumber higher order coefficients
    degree, nodeorder = { # for gmsh node ordering conventions see http://gmsh.info/doc/texinfo/gmsh.html#Node-ordering
      (2, 6): (2, (0,3,1,5,4,2)),
      (2,10): (3, (0,3,4,1,8,9,5,7,6,2)),
      (2,15): (4, (0,3,4,5,1,11,12,13,6,10,14,7,9,8,2)),
      (3,10): (2, (0,4,1,6,5,2,7,9,8,3))}[ndims, geomdofs.shape[1]]
    enum = numpy.empty([degree+1]*(ndims+1), dtype=int)
    bari = tuple(numpy.array([index[::-1] for index in numpy.ndindex(*enum.shape) if sum(index) == degree]).T)
    enum[bari] = numpy.arange(geomdofs.shape[1]) # maps baricentric index to corresponding enumerated index
    shuffle = inodesbydim[ndims].argsort(axis=1)
    geomdofs = geomdofs[:,nodeorder] # convert from gmsh to nutils order
    for i in range(ndims): # strategy: apply shuffle to geomdofs by sequentially swapping vertices...
      for j in range(i+1, ndims+1): # ...considering all j > i pairs...
        m = shuffle[:,i] == j # ...and swap vertices if vertex j is shuffled into i...
        r = enum.swapaxes(i,j)[bari] # ...using the enum table to generate the appropriate renumbering
        geomdofs[m,:] = geomdofs[numpy.ix_(m,r)]
        m = shuffle[:,j] == i
        shuffle[m,j] = shuffle[m,i] # update shuffle to track changed vertex positions

  inodesbydim[ndims].sort(axis=1)
  if tagnamesbydim[ndims-1]:
    inodesbydim[ndims-1].sort(axis=1)
    edges = {tuple(inodes[:iedge])+tuple(inodes[iedge+1:]): (ielem, iedge) for ielem, inodes in enumerate(inodesbydim[ndims]) for iedge in range(ndims+1)}

  vtags = {name: numpy.array(inodes) for name, inodes in tagnamesbydim[ndims].items()}
  btags = {name: numpy.array([edges[tuple(inodesbydim[ndims-1][ibelem])] for ibelem in ibelems]) for name, ibelems in tagnamesbydim[ndims-1].items()}
  ptags = {name: inodesbydim[0][ipelems][...,0] for name, ipelems in tagnamesbydim[0].items()}

  log.info('\n- '.join(['loaded {}d gmsh topology consisting of #{} elements'.format(ndims, len(geomdofs))]
    + [name + ' groups: ' + ', '.join('{} #{}'.format(n, len(e)) for n, e in tags.items())
      for name, tags in (('volume', vtags), ('boundary', btags), ('point', ptags)) if tags]))

  return dict(nodes=inodesbydim[ndims], cnodes=geomdofs, coords=nodes, tags=vtags, btags=btags, ptags=ptags)

@log.withcontext
def gmsh(fname, name='gmsh'):
  """Gmsh parser

  Parser for Gmsh files in `.msh` format. Only files with physical groups are
  supported. See the `Gmsh manual
  <http://geuz.org/gmsh/doc/texinfo/gmsh.html>`_ for details.

  Parameters
  ----------
  fname : :class:`str`
      Path to mesh file.
  name : :class:`str` or :any:`None`
      Name of parsed topology, defaults to 'gmsh'.

  Returns
  -------
  topo : :class:`nutils.topology.SimplexTopology`
      Topology of parsed Gmsh file.
  geom : :class:`nutils.function.Array`
      Isoparametric map.
  """

  return simplex(name=name, **parsegmsh(fname))

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
    transforms = transformseq.PlainTransforms([(root, transform.Simplex(coords[s])) for s in simplices], 2)
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
