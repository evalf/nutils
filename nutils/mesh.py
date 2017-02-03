# -*- coding: utf8 -*-
#
# Module MESH
#
# Part of Nutils: open source numerical utilities for Python. Jointly developed
# by HvZ Computational Engineering, TU/e Multiscale Engineering Fluid Dynamics,
# and others. More info at http://nutils.org <info@nutils.org>. (c) 2014

"""
The mesh module provides mesh generators: methods that return a topology and an
accompanying geometry function. Meshes can either be generated on the fly, e.g.
:func:`rectilinear`, or read from external an externally prepared file,
:func:`gmsh`, :func:`igatool`, and converted to nutils format. Note that no
mesh writers are provided at this point; output is handled by the
:mod:`nutils.plot` module.
"""

from . import topology, function, util, element, numpy, numeric, transform, log, _
import os, warnings

# MESH GENERATORS

@log.title
def rectilinear( richshape, periodic=(), name='rect' ):
  'rectilinear mesh'

  ndims = len(richshape)
  shape = []
  offset = []
  scale = []
  uniform = True
  for v in richshape:
    if numeric.isint( v ):
      assert v > 0
      shape.append( v )
      scale.append( 1 )
      offset.append( 0 )
    elif numpy.equal( v, numpy.linspace(v[0],v[-1],len(v)) ).all():
      shape.append( len(v)-1 )
      scale.append( (v[-1]-v[0]) / float(len(v)-1) )
      offset.append( v[0] )
    else:
      shape.append( len(v)-1 )
      uniform = False

  if isinstance( name, str ):
    wrap = tuple( sh if i in periodic else 0 for i, sh in enumerate(shape) )
    root = transform.roottrans( name, wrap )
  else:
    assert all( ( name.take(0,i) == name.take(2,i) ).all() for i in periodic )
    root = transform.roottransedges( name, shape )

  axes = [ topology.DimAxis(0,n,idim in periodic) for idim, n in enumerate(shape) ]
  topo = topology.StructuredTopology( root, axes )

  if uniform:
    if all( o == offset[0] for o in offset[1:] ):
      offset = offset[0]
    if all( s == scale[0] for s in scale[1:] ):
      scale = scale[0]
    geom = function.rootcoords(ndims) * scale + offset
  else:
    funcsp = topo.splinefunc( degree=1, periodic=() )
    coords = numeric.meshgrid( *richshape ).reshape( ndims, -1 )
    geom = ( funcsp * coords ).sum( -1 )

  return topo, geom

def line( nodes, periodic=False, bnames=None ):
  if isinstance( nodes, int ):
    uniform = True
    assert nodes > 0
    nelems = nodes
    scale = 1
    offset = 0
  else:
    nelems = len(nodes)-1
    scale = (nodes[-1]-nodes[0]) / nelems
    offset = nodes[0]
    uniform = numpy.equal( nodes, offset + numpy.arange(nelems+1) * scale ).all()
  root = transform.roottrans( 'rect', shape=[ nelems if periodic else 0 ] )
  domain = topology.StructuredLine( root, 0, nelems, periodic=periodic, bnames=bnames )
  geom = function.rootcoords(1) * scale + offset if uniform else domain.basis( 'std', degree=1, periodic=False ).dot( nodes )
  return domain, geom

def newrectilinear( nodes, periodic=None, bnames=[['left','right'],['bottom','top'],['front','back']] ):
  if periodic is None:
    periodic = numpy.zeros( len(nodes), dtype=bool )
  else:
    periodic = numpy.asarray( periodic )
    assert len(periodic) == len(nodes) and periodic.ndim == 1 and periodic.dtype == bool
  dims = [ line( nodesi, periodici, bnamesi ) for nodesi, periodici, bnamesi in zip( nodes, periodic, tuple(bnames)+(None,)*len(nodes) ) ]
  domain, geom = dims.pop(0)
  for domaini, geomi in dims:
    domain = domain * domaini
    geom = function.concatenate( function.bifurcate(geom,geomi) )
  return domain, geom

@log.title
def gmsh( fname, name=None ):
  """Gmsh parser

  Parser for Gmsh files in `.msh` format. Only files with physical groups are
  supported. See the `Gmsh manual
  <http://geuz.org/gmsh/doc/texinfo/gmsh.html>`_ for details.

  Args:
      fname (str): Path to mesh file
      name (str, optional): Name of parsed topology, defaults to None

  Returns:
      topo (:class:`nutils.topology.Topology`): Topology of parsed Gmsh file
      geom (:class:`nutils.function.Array`): Isoparametric map

  """

  ndims = 2

  # split sections
  sections = {}
  lines = iter( open(fname,'r') if isinstance(fname,str) else fname )
  for line in lines:
    line = line.strip()
    assert line[0]=='$'
    sname = line[1:]
    slines = []
    for sline in lines:
      sline = sline.strip()
      if sline=='$End'+sname:
        break
      slines.append( sline ) 
    sections[sname] = slines

  # discard section MeshFormat
  sections.pop( 'MeshFormat', None )

  # parse section PhysicalNames
  PhysicalNames = sections.pop( 'PhysicalNames', [0] )
  assert int(PhysicalNames[0]) == len(PhysicalNames)-1
  tagmapbydim = {}, {}, {}
  for line in PhysicalNames[1:]:
    nd, tagid, tagname = line.split( ' ', 2 )
    nd = int(nd)
    tagmapbydim[nd][int(tagid)] = tagname.strip( '"' )

  # parse section Nodes
  Nodes = sections.pop( 'Nodes' )
  assert int(Nodes[0]) == len(Nodes)-1
  nodes = numpy.empty((len(Nodes)-1,3))
  nodemap = {}
  for i, line in enumerate( Nodes[1:] ):
    words = line.split()
    nodemap[int(words[0])] = i
    nodes[i] = [ float(n) for n in words[1:] ]
  assert not numpy.isnan(nodes).any()
  assert numpy.all( nodes[:,2] ) == 0, 'ndims=3 case not yet implemented.'
  nodes = nodes[:,:2]

  # parse section Elements
  Elements = sections.pop( 'Elements' )
  assert int(Elements[0]) == len(Elements)-1
  inodesbydim = [], [], []
  tagnamesbydim = [], [], []
  etype2nd = { 15:0, 1:1, 2:2 }
  for line in Elements[1:]:
    words = line.split()
    nd = etype2nd[int(words[1])]
    ntags = int(words[2])
    assert ntags >= 1
    tagname = tagmapbydim[nd][int(words[3])]
    inodes = tuple( nodemap[int(nodeid)] for nodeid in words[3+ntags:] )
    if inodesbydim[nd] and inodesbydim[nd][-1] == inodes:
      tagnamesbydim[nd][-1].append( tagname )
    else:
      inodesbydim[nd].append( inodes )
      tagnamesbydim[nd].append( [tagname] )
  assert len(inodesbydim) == len(tagnamesbydim)
  inodesbydim = [ numpy.array(e) if e else numpy.empty( (0,nd), dtype=int ) for nd, e in enumerate(inodesbydim) ]

  # check orientation
  vinodes = inodesbydim[2] # save for geom
  elemnodes = nodes[vinodes] # nelems x 3 x 2
  elemareas = numpy.linalg.det( elemnodes[:,:2] - elemnodes[:,2:] )
  assert numpy.all( elemareas > 0 )

  # parse section Periodic
  Periodic = sections.pop( 'Periodic', [0] )
  nperiodic = int(Periodic[0])
  renumber = numpy.arange( len(nodes) )
  master = numpy.ones( len(nodes), dtype=bool )
  n = 0
  for line in Periodic[1:]:
    words = line.split()
    if len(words) == 1:
      n = int(words[0]) # initialize for counting backwards
    elif len(words) == 2:
      islave = nodemap[int(words[0])]
      imaster = nodemap[int(words[1])]
      renumber[islave] = renumber[imaster]
      master[islave] = False
      n -= 1
    else:
      assert len(words) == 3 # discard content
      assert n == 0 # check if batch of slave/master nodes matches announcement
      nperiodic -= 1
  assert nperiodic == 0 # check if number of periodic blocks matches announcement
  assert n == 0 # check if last batch of slave/master nodes matches announcement
  renumber = master.cumsum()[renumber]-1
  inodesbydim = [ renumber[e] for e in inodesbydim ]

  # warn about unused sections
  for section in sections:
    warnings.warn('section {!r} defined but not used'.format(section) )

  # create volume, boundary and interface elements
  triref = element.getsimplex(2)
  velems = [] # same order as inodesbydim[2]
  belems = {}
  ielems = {}
  flip = transform.affine( -1, [1] )
  for inodes in inodesbydim[2]:
    trans = transform.maptrans( triref.vertices, inodes if not name else [name+str(inode) for inode in inodes] )
    elem = element.Element( triref, trans )
    velems.append( elem )
    for iedge, binodes in enumerate([ inodes[1:], inodes[::-2], inodes[:2] ]):
      try:
        belem = belems.pop( tuple(binodes[::-1]) )
      except KeyError:
        belems[tuple(binodes)] = elem.edge(iedge)
      else:
        oppbelem = elem.edge(iedge)
        assert belem.reference == oppbelem.reference
        opptrans = oppbelem.transform
        if belem.transform.isflipped == opptrans.isflipped:
          opptrans <<= flip
        ielems[tuple(binodes)] = element.Element( belem.reference, belem.transform, opptrans )

  # separate volume elements by tag
  tagsvelems = {}
  for elem, tagnames in zip( velems, tagnamesbydim[2] ):
    for tagname in tagnames:
      tagsvelems.setdefault( tagname, [] ).append( elem )

  # separate boundary and interface elements by tag
  tagsbelems = {}
  tagsielems = {}
  istagged = set()
  for inodes, tagnames in zip( inodesbydim[1], tagnamesbydim[1] ):
    inodes = tuple(inodes)
    try:
      elem = belems.get(inodes)
      if elem is None:
        elem = belems[inodes[::-1]]
        tagnames = [ '~'+tagname for tagname in tagnames ]
      tagselems = tagsbelems
    except KeyError:
      elem = ielems.get(inodes)
      if elem is None:
        elem = ielems.pop(inodes[::-1]).flipped
        assert inodes[::-1] not in istagged, 'opposing interface elements found in group {!r}'.format(','.join(tagnames))
        ielems[inodes] = elem # flip element to match interface group
      istagged.add( inodes ) # check for back/forth modification
      tagselems = tagsielems
    for tagname in tagnames:
      tagselems.setdefault( tagname, [] ).append( elem )

  # create and separate point elements by tag
  pelems = []
  tagspelems = {}
  pref = element.getsimplex(0)
  for (pinode,), tagnames in zip( inodesbydim[0], tagnamesbydim[0] ):
    pelem = []
    for inodes, elem in zip( inodesbydim[2], velems ):
      if pinode in inodes:
        ivertex = tuple(inodes).index( pinode )
        offset = elem.reference.vertices[ivertex]
        trans = elem.transform << transform.affine( linear=numpy.zeros(shape=(ndims,0),dtype=int), offset=offset, isflipped=False )
        pelem.append( element.Element( pref, trans ) )
    pelems.extend( pelem )
    for tagname in tagnames:
      tagspelems.setdefault( tagname, [] ).extend( pelem )

  # log statistics
  log.info( 'topology (#{}) with groups: {}'.format( len(velems), ', '.join('{} (#{})'.format(n,len(e)) for n, e in tagsvelems.items()) ) )
  log.info( 'boundary (#{}) with groups: {}'.format( len(belems), ', '.join('{} (#{})'.format(n,len(e)) for n, e in tagsbelems.items() ) ) )
  log.info( 'interfaces (#{}) with groups: {}'.format( len(ielems), ', '.join('{} (#{})'.format(n,len(e)) for n, e in tagsielems.items() ) ) )
  log.info( 'points (#{}) with groups: {}'.format( len(pelems), ', '.join('{} (#{})'.format(n,len(e)) for n, e in tagspelems.items() ) ) )

  # create base topology
  basetopo = topology.UnstructuredTopology( ndims, velems )
  basetopo.boundary = topology.UnstructuredTopology( ndims-1, belems.values() )
  basetopo.interfaces = topology.UnstructuredTopology( ndims-1, ielems.values() )
  basetopo.points = topology.UnstructuredTopology( 0, pelems )

  # create volume, boundary, interface, point subtopologies
  vgroups = { tagname: topology.UnstructuredTopology( ndims, tagvelems ) for tagname, tagvelems in tagsvelems.items() }
  bgroups = { tagname: topology.UnstructuredTopology( ndims-1, tagbelems ) for tagname, tagbelems in tagsbelems.items() }
  igroups = { tagname: topology.UnstructuredTopology( ndims-1, tagielems ) for tagname, tagielems in tagsielems.items() }
  pgroups = { tagname: topology.UnstructuredTopology( 0, tagpelems ) for tagname, tagpelems in tagspelems.items() }
  topo = basetopo.withgroups( vgroups=vgroups, bgroups=bgroups, igroups=igroups, pgroups=pgroups )

  # create geometry
  nmap = { elem.transform: inodes for inodes, elem in zip( vinodes, velems ) }
  fmap = dict.fromkeys( nmap, triref.stdfunc(1) )
  basis = function.function( fmap=fmap, nmap=nmap, ndofs=len(nodes) )
  geom = ( basis[:,_] * nodes ).sum(0)

  return topo, geom

def gmesh( fname, tags={}, name=None, use_elementary=False ):
  warnings.warn( 'mesh.gmesh has been renamed to mesh.gmsh; please update your code', DeprecationWarning )
  assert not use_elementary, 'support of non-physical gmsh files has been deprecated'
  assert not tags, 'support of external group names has been deprecated; please provide physical names via gmsh'
  return gmsh( fname, name )

def triangulation( vertices, nvertices ):
  'triangulation'

  raise NotImplementedError

  bedges = {}
  nmap = {}
  I = numpy.array( [[2,0],[0,1],[1,2]] )
  for n123 in vertices:
    elem = element.getsimplex(2)
    nmap[ elem ] = n123
    for iedge, (n1,n2) in enumerate( n123[I] ):
      try:
        del bedges[ (n2,n1) ]
      except KeyError:
        bedges[ (n1,n2) ] = elem, iedge

  dofaxis = function.DofAxis( nvertices, nmap )
  stdelem = element.PolyTriangle( 1 )
  linearfunc = function.Function( dofaxis=dofaxis, stdmap=dict.fromkeys(nmap,stdelem) )

  connectivity = dict( bedges.iterkeys() )
  N = list( connectivity.popitem() )
  while connectivity:
    N.append( connectivity.pop( N[-1] ) )
  assert N[0] == N[-1]

  structure = []
  for n12 in zip( N[:-1], N[1:] ):
    elem, iedge = bedges[ n12 ]
    structure.append( elem.edge( iedge ) )
    
  topo = topology.UnstructuredTopology( ndims, nmap )
  topo.boundary = topology.StructuredTopology( structure, periodic=(1,) )
  return topo

def igatool( path, name=None ):
  'igatool mesh'

  if name is None:
    name = os.path.basename(path)

  import vtk

  reader = vtk.vtkXMLUnstructuredGridReader()
  reader.SetFileName( path )
  reader.Update()

  mesh = reader.GetOutput()

  FieldData = mesh.GetFieldData()
  CellData = mesh.GetCellData()

  NumberOfPoints = int( mesh.GetNumberOfPoints() )
  NumberOfElements = mesh.GetNumberOfCells()
  NumberOfArrays = FieldData.GetNumberOfArrays()

  points = util.arraymap( mesh.GetPoint, float, range(NumberOfPoints) )
  Cij = FieldData.GetArray( 'Cij' )
  Cv = FieldData.GetArray( 'Cv' )
  Cindi = CellData.GetArray( 'Elem_extr_indi')

  elements = []
  degree = 3
  ndims = 2
  nmap = {}
  fmap = {}

  poly = element.PolyLine( element.PolyLine.bernstein_poly( degree ) )**ndims

  for ielem in range(NumberOfElements):

    cellpoints = vtk.vtkIdList()
    mesh.GetCellPoints( ielem, cellpoints )
    nids = util.arraymap( cellpoints.GetId, int, range(cellpoints.GetNumberOfIds()) )

    assert mesh.GetCellType(ielem) == vtk.VTK_HIGHER_ORDER_QUAD
    nb = (degree+1)**2
    assert len(nids) == nb

    n = range( *util.arraymap( Cindi.GetComponent, int, ielem, [0,1] ) )
    I = util.arraymap( Cij.GetComponent, int, n, 0 )
    J = util.arraymap( Cij.GetComponent, int, n, 1 )
    Ce = numpy.zeros(( nb, nb ))
    Ce[I,J] = util.arraymap( Cv.GetComponent, float, n, 0 )

    vertices = [ element.PrimaryVertex( '%s(%d:%d)' % (name,ielem,ivertex) ) for ivertex in range(2**ndims) ]
    elem = element.QuadElement( vertices=vertices, ndims=ndims )
    elements.append( elem )

    fmap[ elem ] = element.ExtractionWrapper( poly, Ce.T )
    nmap[ elem ] = nids

  splinefunc = function.function( fmap, nmap, NumberOfPoints )

  boundaries = {}
  elemgroups = {}
  vertexgroups = {}
  renumber   = (0,3,1,2)
  for iarray in range( NumberOfArrays ):
    name = FieldData.GetArrayName( iarray )
    index = name.find( '_group_' )
    if index == -1:
      continue
    grouptype = name[:index]
    groupname = name[index+7:]
    A = FieldData.GetArray( iarray )
    I = util.arraymap( A.GetComponent, int, range(A.GetSize()), 0 )
    if grouptype == 'edge':
      belements = [ elements[i//4].edge( renumber[i%4] ) for i in I ]
      boundaries[ groupname ] = topology.UnstructuredTopology( ndims-1, belements )
    elif grouptype == 'vertex':
      vertexgroups[ groupname ] = I
    elif grouptype == 'element':
      elemgroups[ groupname ] = topology.UnstructuredTopology( ndims, [ elements[i] for i in I ] )
    else:
      raise Exception( 'unknown group type: %r' % grouptype )

  topo = topology.UnstructuredTopology( ndims, elements )
  for groupname, grouptopo in elemgroups.items():
    topo[groupname] = grouptopo

  if boundaries:
    topo.boundary = topology.UnstructuredTopology( ndims-1, [ elem for topo in boundaries.values() for elem in topo ] )
    for groupname, grouptopo in boundaries.items():
      topo.boundary[groupname] = grouptopo

  for group in elemgroups.values():
    myboundaries = {}
    for name, boundary in boundaries.items():
      belems = [ belem for belem in boundary.elements if belem.parent[0] in group ]
      if belems:
        myboundaries[ name ] = topology.UnstructuredTopology( ndims-1, belems )
    if myboundaries:
      group.boundary = topology.UnstructuredTopology( ndims-1, [ elem for topo in myboundaries.values() for elem in topo ] )
      for groupname, grouptopo in myboundaries.items():
        group.boundary[groupname] = grouptopo

  funcsp = topo.splinefunc( degree=degree )
  coords = ( funcsp[:,_] * points ).sum( 0 )
  return topo, coords #, vertexgroups

def fromfunc( func, nelems, ndims, degree=1 ):
  'piecewise'

  if isinstance( nelems, int ):
    nelems = [ nelems ]
  assert len( nelems ) == func.__code__.co_argcount
  topo, ref = rectilinear( [ numpy.linspace(0,1,n+1) for n in nelems ] )
  funcsp = topo.splinefunc( degree=degree ).vector( ndims )
  coords = topo.projection( func, onto=funcsp, coords=ref, exact_boundaries=True )
  return topo, coords

def demo( xmin=0, xmax=1, ymin=0, ymax=1 ):
  'demo triangulation of a rectangle'

  phi = numpy.arange( 1.5, 13 ) * (2*numpy.pi) / 12
  P = numpy.array([ numpy.cos(phi), numpy.sin(phi) ])
  P /= abs(P).max(axis=0)
  phi = numpy.arange( 1, 9 ) * (2*numpy.pi) / 8
  Q = numpy.array([ numpy.cos(phi), numpy.sin(phi) ])
  Q /= 2 * numpy.sqrt( abs(Q).max(axis=0) / numpy.sqrt(2) )
  R = numpy.zeros([2,1])

  coords = numpy.round( numpy.hstack( [P,Q,R] ).T * 2**5 ) / 2**5

  vertices = numpy.array(
    [ [ 12+(i-i//3)%8, i, (i+1)%12 ] for i in range(12) ]
  + [ [ i+1+(i//2), 12+(i+1)%8, 12+i ] for i in range(8) ]
  + [ [ 20, 12+i, 12+(i+1)%8 ] for i in range(8) ] )
  
  root = transform.roottrans( 'demo', shape=(0,0) )
  reference = element.getsimplex(2)
  elems = [ element.Element( reference, root << transform.simplex(coords[iverts]) ) for iverts in vertices ]
  topo = topology.UnstructuredTopology( 2, elems )

  belems = [ elem.edge(0) for elem in elems[:12] ]
  btopos = [ topology.UnstructuredTopology( 1, subbelems ) for subbelems in (belems[0:3], belems[3:6], belems[6:9], belems[9:12]) ]
  topo.boundary = topology.UnionTopology( btopos, ['top','left','bottom','right'] )

  geom = [.5*(xmin+xmax),.5*(ymin+ymax)] \
       + [.5*(xmax-xmin),.5*(ymax-ymin)] * function.rootcoords(2)

  return topo, geom

# vim:shiftwidth=2:softtabstop=2:expandtab:foldmethod=indent:foldnestmax=1
