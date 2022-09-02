"""
The mesh module provides mesh generators: methods that return a topology and an
accompanying geometry function. Meshes can either be generated on the fly, e.g.
:func:`rectilinear`, or read from external an externally prepared file,
:func:`gmsh`, and converted to nutils format. Note that no mesh writers are
provided at this point.
"""

from . import topology, function, util, element, numeric, transform, transformseq, warnings, types, cache
from .elementseq import References
from .transform import TransformItem
from .topology import Topology
from typing import Optional, Sequence, Tuple, Union
import numpy
import os
import itertools
import re
import math
import treelog as log
import io
import contextlib
_ = numpy.newaxis

# MESH GENERATORS


@log.withcontext
def rectilinear(richshape: Sequence[Union[int, Sequence[float]]], periodic: Sequence[int] = (), name: Optional[str] = None, space: str = 'X', root: Optional[TransformItem] = None) -> Tuple[Topology, function.Array]:
    'rectilinear mesh'

    verts = [numpy.arange(v + 1) if numeric.isint(v) else v for v in richshape]
    shape = [len(v) - 1 for v in verts]
    ndims = len(shape)

    if name is not None:
        warnings.deprecation('Argument `name` is deprecated; use `root` with a `TransformItem` instead.')
        if root is not None:
            raise ValueError('Arguments `name` and `root` cannot be used simultaneously.')
        root = transform.Index(hash(name))
    elif root is None:
        root = transform.Index(ndims, 0)

    axes = [transformseq.DimAxis(i=0, j=n, mod=n if idim in periodic else 0, isperiodic=idim in periodic) for idim, n in enumerate(shape)]
    topo = topology.StructuredTopology(space, root, axes)

    funcsp = topo.basis('spline', degree=1, periodic=())
    coords = numeric.meshgrid(*verts).reshape(ndims, -1)
    geom = (funcsp * coords).sum(-1)

    return topo, geom


_oldrectilinear = rectilinear  # reference for internal unittests


def line(nodes: Union[int, Sequence[float]], periodic: bool = False, bnames: Optional[Sequence[Tuple[str, str]]] = None, *, name: Optional[str] = None, space: str = 'X', root: Optional[TransformItem] = None) -> Tuple[Topology, function.Array]:
    if name is not None:
        warnings.deprecation('Argument `name` is deprecated; use `root` with a `transform.transformitem` instead.')
        if root is not None:
            raise ValueError('Arguments `name` and `root` cannot be used simultaneously.')
        root = transform.Index(hash(name))
    elif root is None:
        root = transform.Index(1, 0)
    if isinstance(nodes, int):
        nodes = numpy.arange(nodes + 1)
    domain = topology.StructuredLine(space, root, 0, len(nodes) - 1, periodic=periodic, bnames=bnames)
    geom = domain.basis('std', degree=1, periodic=[]).dot(nodes)
    return domain, geom


def newrectilinear(nodes: Sequence[Union[int, Sequence[float]]], periodic: Optional[Sequence[int]] = None, name: Optional[str] = None, bnames=[['left', 'right'], ['bottom', 'top'], ['front', 'back']], spaces: Optional[Sequence[str]] = None, root: Optional[TransformItem] = None) -> Tuple[Topology, function.Array]:
    if periodic is None:
        periodic = []
    if not spaces:
        spaces = 'XYZ' if len(nodes) <= 3 else map('R{}'.format, range(len(nodes)))
    else:
        assert len(spaces) == len(nodes)
    domains, geoms = zip(*(line(nodesi, i in periodic, bnamesi, name=name, space=spacei, root=root) for i, (nodesi, bnamesi, spacei) in enumerate(zip(nodes, tuple(bnames)+(None,)*len(nodes), spaces))))
    return util.product(domains), function.stack(geoms)


if os.environ.get('NUTILS_TENSORIAL'):
    def rectilinear(richshape: Sequence[Union[int, Sequence[float]]], periodic: Sequence[int] = (), name: Optional[str] = None, space: str = 'X', root: Optional[TransformItem] = None) -> Tuple[Topology, function.Array]:
        spaces = tuple(space+str(i) for i in range(len(richshape)))
        return newrectilinear(richshape, periodic, name=name, spaces=spaces, root=root)


@log.withcontext
def multipatch(patches, nelems, patchverts=None, name: Optional[str] = None):
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
    ...     [0,1,2,3], # x=-1
    ...     [4,5,6,7], # x= 1
    ...     [0,1,4,5], # y=-1
    ...     [2,3,6,7], # y= 1
    ...     [0,2,4,6], # z=-1
    ...     [1,3,5,7], # z= 1
    ...   ],
    ...   patchverts=tuple(itertools.product(*([[-1,1]]*3))),
    ...   nelems=1)
    >>> sphere = function.normalized(cube)

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

    if name is not None:
        warnings.deprecation('Argument `name` is deprecated and can safely be omitted.')

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
        raise ValueError('Only hyperrectangular patches are supported: '
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
            sides = [(0, 1)]*ndims
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
        topos.append(rectilinear(shape, root=transform.Index(ndims, i))[0])
        # compute patch geometry
        patchcoords = [numpy.linspace(0, 1, n+1) for n in shape]
        patchcoords = numeric.meshgrid(*patchcoords).reshape(ndims, -1)
        if patchverts is not None:
            patchcoords = numpy.array([
                sum(
                    patchverts[j]*util.product(c if s else 1-c for c, s in zip(coord, side))
                    for j, side in zip(patch.flat, itertools.product(*[[0, 1]]*ndims))
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

    if not msh.cell_sets:
        # Old versions of the gmsh file format repeat elements that have multiple
        # tags. To support this we edit the meshio data to bring it in the same
        # form as the new files by deduplicating cells and creating cell_sets.
        renums = []
        for icell, cells in enumerate(msh.cells):
            keep = (cells.data[1:] != cells.data[:-1]).any(axis=1)
            if keep.all():
                renum = numpy.arange(len(cells.data))
            else:
                msh.cells[icell] = type(cells)(cells.type, cells.data[numpy.hstack([True, keep])])
                renum = numpy.hstack([0, keep.cumsum()])
            renums.append(renum)
        for name, (itag, nd) in msh.field_data.items():
            msh.cell_sets[name] = [renum[data == itag] for data, renum in zip(msh.cell_data['gmsh:physical'], renums)]

    # Coords is a 2d float-array such that coords[inode,idim] == coordinate.
    coords = msh.points

    # Nodes is a dictionary that maps a topological dimension to a 2d int-array
    # dictionary such that nodes[nd][ielem,ilocal] == inode, where ilocal < nd+1
    # for linear geometries or larger for higher order geometries. Since meshio
    # stores nodes by simplex type and cell, simplex types are mapped to
    # dimensions and gathered, after which cells are concatenated under the
    # assumption that there is only one simplex type per dimension.
    nodes = {('ver', 'lin', 'tri', 'tet').index(typename[:3]): numpy.concatenate(datas, axis=0)
             for typename, datas in util.gather((cells.type, cells.data) for cells in msh.cells)}

    # Identities is a 2d [master, slave] int-aray that pairs matching nodes on
    # periodic walls. For the topological connectivity, all slaves in the nodes
    # arrays will be replaced by their master counterpart.
    identities = numpy.zeros((0, 2), dtype=int) if not msh.gmsh_periodic \
        else numpy.concatenate([d for a, b, c, d in msh.gmsh_periodic], axis=0)

    # It may happen that meshio provides periodicity relations for nodes that
    # have no associated coordinate, typically because they are not part of any
    # physical group. We need to filter these out to avoid errors further down.
    mask = identities < len(coords)
    keep = mask.any(axis=1)
    assert mask[keep].all()
    identities = identities[keep]

    # Tags is a list of (nd, name, ndelems) tuples that define topological groups
    # per dimension. Since meshio associates group names with cells, which are
    # concatenated in nodes, element ids are offset and concatenated to match.
    tags = [(nd, name, numpy.concatenate([selection
                                          + sum(len(cells.data) for cells in msh.cells[:icell] if cells.type == msh.cells[icell].type)  # offset into nodes
                                          for icell, selection in enumerate(msh.cell_sets[name]) if len(selection)]))
            for name, (itag, nd) in msh.field_data.items()]

    # determine the dimension of the topology
    ndims = max(nodes)

    # determine the dimension of the geometry
    assert not numpy.isnan(coords).any()
    while coords.shape[1] > ndims and not coords[:, -1].any():
        coords = coords[:, :-1]

    # separate geometric, topological nodes
    cnodes = nodes[ndims]
    if cnodes.shape[1] > ndims+1:  # higher order geometry
        nodes = {nd: n[:, :nd+1] for nd, n in nodes.items()}  # remove high order info

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

    if cnodes is vnodes:  # geometry is linear and non-periodic, dofs follow in-place sorting of nodes
        degree = 1
    elif cnodes.shape[1] == ndims+1:  # linear elements: match sorting of nodes
        degree = 1
        shuffle = vnodes.argsort(axis=1)
        cnodes = cnodes[numpy.arange(len(cnodes))[:, _], shuffle]  # gmsh conveniently places the primary ndim+1 vertices first
    else:  # higher order elements: match sorting of nodes and renumber higher order coefficients
        degree, nodeorder = {  # for meshio's node ordering conventions see http://www.vtk.org/VTK/img/file-formats.pdf
            (2, 6): (2, (0, 3, 1, 5, 4, 2)),
            (2, 10): (3, (0, 3, 4, 1, 8, 9, 5, 7, 6, 2)),
            (2, 15): (4, (0, 3, 4, 5, 1, 11, 12, 13, 6, 10, 14, 7, 9, 8, 2)),
            (3, 10): (2, (0, 4, 1, 6, 5, 2, 7, 8, 9, 3))}[ndims, cnodes.shape[1]]
        enum = numpy.empty([degree+1]*(ndims+1), dtype=int)
        bari = tuple(numpy.array([index[::-1] for index in numpy.ndindex(*enum.shape) if sum(index) == degree]).T)
        enum[bari] = numpy.arange(cnodes.shape[1])  # maps baricentric index to corresponding enumerated index
        shuffle = vnodes.argsort(axis=1)
        cnodes = cnodes[:, nodeorder]  # convert from gmsh to nutils order
        for i in range(ndims):  # strategy: apply shuffle to cnodes by sequentially swapping vertices...
            for j in range(i+1, ndims+1):  # ...considering all j > i pairs...
                m = shuffle[:, i] == j  # ...and swap vertices if vertex j is shuffled into i...
                r = enum.swapaxes(i, j)[bari]  # ...using the enum table to generate the appropriate renumbering
                cnodes[m, :] = cnodes[numpy.ix_(m, r)]
                m = shuffle[:, j] == i
                shuffle[m, j] = shuffle[m, i]  # update shuffle to track changed vertex positions

    vnodes.sort(axis=1)
    nnodes = vnodes[:, -1].max()+1

    vtags, btags, ptags = {}, {}, {}
    edge_vertices = numpy.arange(ndims+1).repeat(ndims).reshape(ndims, ndims+1)[:, ::-1].T  # nedges x ndims
    for nd, name, ielems in tags:
        if nd == ndims:
            vtags[name] = numpy.array(ielems)
        elif nd == ndims-1:
            edgenodes = bnodes[ielems]  # all edge elements in msh file
            nodemask = numeric.asboolean(edgenodes.ravel(), size=nnodes, ordered=False)  # all elements sharing at least 1 edge node
            ielems, = (nodemask[vnodes].sum(axis=1) >= ndims).nonzero()  # all elements sharing at least ndims edge nodes
            edgemap = {tuple(b): (ielem, iedge) for ielem, a in zip(ielems, vnodes[ielems[:, _, _], edge_vertices[_, :, :]]) for iedge, b in enumerate(a)}
            belems = (edgemap.get(tuple(sorted(n))) for n in edgenodes)  # map every edge element to its corresponding (ielem, iedge) combination
            belems = filter(None, belems)  # remove spurious edge elements that have no adjacent volume element
            btags[name] = numpy.array(list(belems))
        elif nd == 0:
            ptags[name] = pnodes[ielems][..., 0]

    log.info('\n- '.join(['loaded {}d gmsh topology consisting of #{} elements'.format(ndims, len(cnodes))]
                         + [name + ' groups: ' + ', '.join('{} #{}'.format(n, len(e)) for n, e in tags.items())
                            for name, tags in (('volume', vtags), ('boundary', btags), ('point', ptags)) if tags]))

    return dict(nodes=vnodes, cnodes=cnodes, coords=coords, tags=vtags, btags=btags, ptags=ptags)


@log.withcontext
@types.apply_annotations
def gmsh(fname: util.binaryfile, name='gmsh', *, space='X'):
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
        return simplex(name=name, **parsegmsh(f), space=space)


def simplex(nodes, cnodes, coords, tags, btags, ptags, name='simplex', *, space='X'):
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
    degree = 1 if ncnodes == ndims+1 else int((ncnodes * math.factorial(ndims))**(1/ndims))-1

    assert len(nodes) == nelems, 'number of simplex vertices and coordinates do not match'
    assert numpy.greater(nodes[:, 1:], nodes[:, :-1]).all(), 'nodes must be sorted'
    assert ncnodes == _comb(ndims + degree, degree), 'number of coordinate nodes does not correspond to uniformly refined simplex'

    transforms = transformseq.IndexTransforms(ndims=ndims, length=nelems)
    topo = topology.SimplexTopology(space, nodes, transforms, transforms)
    coeffs = element.getsimplex(ndims).get_poly_coeffs('lagrange', degree=degree)
    basis = function.PlainBasis([coeffs] * nelems, cnodes, nverts, topo.f_index, topo.f_coords)
    geom = (basis[:, _] * coords).sum(0)

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
                transforms = transformseq.PlainTransforms(transforms, ndims, ndims-1)
                opposites = transforms if opposites is None else transformseq.PlainTransforms(opposites, ndims, ndims-1)
                groups[name] = topology.SimplexTopology(space, simplices, transforms, opposites)

    pgroups = {}
    if ptags:
        ptrans = [transform.Point(offset) for offset in numpy.eye(ndims+1)[:, 1:]]
        pmap = {inode: numpy.array(numpy.equal(nodes, inode).nonzero()).T for inode in set.union(*map(set, ptags.values()))}
        for pname, inodes in ptags.items():
            ptransforms = transformseq.PlainTransforms([topo.transforms[ielem] + (ptrans[ivertex],) for inode in inodes for ielem, ivertex in pmap[inode]], ndims, 0)
            preferences = References.uniform(element.getsimplex(0), len(ptransforms))
            pgroups[pname] = topology.TransformChainsTopology(space, preferences, ptransforms, ptransforms)

    vgroups = {}
    for name, ielems in tags.items():
        if len(ielems) == nelems and numpy.equal(ielems, numpy.arange(nelems)).all():
            vgroups[name] = topo.withgroups(bgroups=bgroups, igroups=igroups, pgroups=pgroups)
            continue
        transforms = topo.transforms[ielems]
        vtopo = topology.SimplexTopology(space, nodes[ielems], transforms, transforms)
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
                    transforms = transformseq.PlainTransforms(transforms, ndims, ndims-1)
                    opposites = transformseq.PlainTransforms(opposites, ndims, ndims-1) if len(opposites) == len(transforms) else transforms
                    groups[bname] = topology.SimplexTopology(space, simplices, transforms, opposites)
        vpgroups = {}
        for pname, inodes in ptags.items():
            ptransforms = transformseq.PlainTransforms([topo.transforms[ielem] + (ptrans[ivertex],) for inode in inodes for ielem, ivertex in pmap[inode] if keep[ielem]], ndims, 0)
            preferences = References.uniform(element.getsimplex(0), len(ptransforms))
            vpgroups[pname] = topology.TransformChainsTopology(space, preferences, ptransforms, ptransforms)
        vgroups[name] = vtopo.withgroups(bgroups=vbgroups, igroups=vigroups, pgroups=vpgroups)

    return topo.withgroups(vgroups=vgroups, bgroups=bgroups, igroups=igroups, pgroups=pgroups), geom


def fromfunc(func, nelems, ndims, degree=1):
    'piecewise'

    if isinstance(nelems, int):
        nelems = [nelems]
    assert len(nelems) == func.__code__.co_argcount
    topo, ref = rectilinear([numpy.linspace(0, 1, n+1) for n in nelems])
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
    :class:`nutils.topology.TransformChainsTopology`:
        The structured/unstructured topology.
    :class:`nutils.function.Array`:
        The geometry function.
    '''

    space = 'X'

    if etype == 'square':
        topo, geom = rectilinear([nelems, nelems], space=space)
        return topo, geom / nelems

    elif etype in ('triangle', 'mixed'):
        simplices = numpy.concatenate([
            numpy.take([i*(nelems+1)+j, i*(nelems+1)+j+1, (i+1)*(nelems+1)+j, (i+1)*(nelems+1)+j+1], [[0, 1, 2], [1, 2, 3]] if i % 2 == j % 2 else [[0, 1, 3], [0, 2, 3]], axis=0)
            for i in range(nelems) for j in range(nelems)])

        v = numpy.arange(nelems+1, dtype=float)
        coords = numeric.meshgrid(v, v).reshape(2, -1).T
        transforms = transformseq.IndexTransforms(2, len(simplices))
        topo = topology.SimplexTopology(space, simplices, transforms, transforms)

        if etype == 'mixed':
            references = list(topo.references)
            square = element.getsimplex(1)**2
            connectivity = list(topo.connectivity)
            isquares = [i * nelems + j for i in range(nelems) for j in range(nelems) if i % 2 == j % 3]
            dofs = list(simplices)
            for n in sorted(isquares, reverse=True):
                i, j = divmod(n, nelems)
                references[n*2:(n+1)*2] = square,
                connectivity[n*2:(n+1)*2] = numpy.concatenate(connectivity[n*2:(n+1)*2])[[3, 2, 4, 1] if i % 2 == j % 2 else [3, 2, 0, 5]],
                connectivity = [c-numpy.greater(c, n*2) for c in connectivity]
                dofs[n*2:(n+1)*2] = numpy.unique([*dofs[n*2], *dofs[n*2+1]]),
            coords = coords[numpy.argsort(numpy.unique(numpy.concatenate(dofs), return_index=True)[1])]
            transforms = transformseq.IndexTransforms(2, len(connectivity))
            topo = topology.ConnectedTopology(space, References.from_iter(references, 2), transforms, transforms, connectivity)

        geom = (topo.basis('std', degree=1) * coords.T).sum(-1)
        x, y = topo.boundary.sample('_centroid', None).eval(geom).T
        bgroups = dict(left=x < .1, right=x > nelems-.1, bottom=y < .1, top=y > nelems-.1)
        topo = topo.withboundary(**{name: topo.boundary[numpy.where(mask)[0]] for name, mask in bgroups.items()})
        return topo, geom / nelems

    else:
        raise Exception('invalid element type {!r}'.format(etype))


try:
    from math import comb as _comb  # new in Python 3.8
except ImportError:
    _comb = lambda n, k: numpy.arange(1+max(k, n-k), 1+n).prod() // math.factorial(min(k, n-k))

# vim:sw=2:sts=2:et
