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
:func:`gmsh`, and converted to nutils format. Note that no mesh writers are
provided at this point; output is handled by the :mod:`nutils.plot` module.
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
  tagmapbydim = {}, {}, {} # tagid->tagname dictionary
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
  inodesbydim = [], [], [] # nelems-list of 3-tuples of node numbers
  tagnamesbydim = {}, {}, {} # tag->ielems dictionary
  etype2nd = { 15:0, 1:1, 2:2 }
  for line in Elements[1:]:
    words = line.split()
    nd = etype2nd[int(words[1])]
    ntags = int(words[2])
    assert ntags >= 1
    tagname = tagmapbydim[nd][int(words[3])]
    inodes = tuple( nodemap[int(nodeid)] for nodeid in words[3+ntags:] )
    if not inodesbydim[nd] or inodesbydim[nd][-1] != inodes: # multiple tags are repeated in consecutive lines
      inodesbydim[nd].append( inodes )
    tagnamesbydim[nd].setdefault( tagname, [] ).append( len(inodesbydim[nd])-1 )
  inodesbydim = [ numpy.array(e) if e else numpy.empty( (0,nd), dtype=int ) for nd, e in enumerate(inodesbydim) ]
  if tagnamesbydim[2]:
    log.info( 'topology groups:', ', '.join('{} (#{})'.format(n,len(e)) for n, e in tagnamesbydim[2].items()) )

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

  # create base topology
  triref = element.getsimplex(2)
  elements = [ element.Element( triref, transform.maptrans( linear=[[-1,-1],[1,0],[0,1]], offset=[1,0,0], vertices=inodes if not name else [name+str(inode) for inode in inodes] ) )
    for ielem, inodes in log.enumerate( 'elem', inodesbydim[2] ) ]
  basetopo = topology.UnstructuredTopology( ndims, elements )
  log.info( 'created topology consisting of {} elements'.format(len(elements)) )

  # create connectivity matrix
  connectivity = -numpy.ones( (len(inodesbydim[2]),3), dtype=int )
  edges = {} # binodes->(ielem,iedge) dictionary
  for ielem, inodes in log.enumerate( 'elem', inodesbydim[2] ):
    for iedge, binodes in enumerate([ inodes[1:], inodes[::-2], inodes[:2] ]):
      key = tuple(sorted(binodes))
      try:
        jelem, jedge = edges[key]
      except KeyError:
        edges[key] = ielem, iedge
      else:
        connectivity[ielem][iedge] = jelem
        connectivity[jelem][jedge] = ielem

  # insert connectivity in place of cached property
  basetopo.connectivity = connectivity

  # separate boundary and interface elements by tag
  tagsbelems = {}
  tagsielems = {}
  for name, ibelems in tagnamesbydim[1].items():
    for ibelem in ibelems:
      binodes = inodesbydim[1][ibelem]
      ielem, iedge = edges[ min(binodes), max(binodes) ]
      elem = elements[ielem].edge(iedge)
      ioppelem = connectivity[ielem][iedge]
      if ioppelem == -1:
        tagsbelems.setdefault( name, [] ).append( elem )
      else:
        ioppedge = tuple(connectivity[ioppelem]).index(ielem)
        tagsielems.setdefault( name, [] ).append( elem.withopposite( elements[ioppelem].edge(ioppedge) ) )
  if tagsbelems:
    log.info( 'boundary groups:', ', '.join('{} (#{})'.format(n,len(e)) for n, e in tagsbelems.items() ) )
  if tagsielems:
    log.info( 'interface groups:', ', '.join('{} (#{})'.format(n,len(e)) for n, e in tagsielems.items() ) )

  # create points topology and separate point elements by tag
  tagspelems = {}
  if tagnamesbydim[0]: # point gorups defined
    pelems = { inodes[0]: [] for inodes in inodesbydim[0] }
    pref = element.getsimplex(0)
    for inodes, elem in zip( inodesbydim[2], elements ):
      for ivertex, inode in enumerate(inodes):
        if inode in pelems:
          offset = elem.reference.vertices[ivertex]
          trans = elem.transform << transform.affine( linear=numpy.zeros(shape=(ndims,0),dtype=int), offset=offset, isflipped=False )
          pelems[inode].append( element.Element( pref, trans ) )
    for name, ipelems in tagnamesbydim[0].items():
      tagspelems[name] = [ pelem for ipelem in ipelems for inode in inodesbydim[0][ipelem] for pelem in pelems[inode] ]
    basetopo.points = topology.UnstructuredTopology( 0, sum( pelems.values(), [] ) )
    log.info( 'points groups:', ', '.join('{} (#{})'.format(n,len(e)) for n, e in tagspelems.items() ) )

  # add volume, boundary, interface, point subtopologies
  topo = basetopo.withgroups(
    bgroups={ tagname: topology.UnstructuredTopology( ndims-1, tagbelems ) for tagname, tagbelems in tagsbelems.items() },
    igroups={ tagname: topology.UnstructuredTopology( ndims-1, tagielems ) for tagname, tagielems in tagsielems.items() },
    pgroups={ tagname: topology.UnstructuredTopology( 0, tagpelems ) for tagname, tagpelems in tagspelems.items() } )

  # create vgroups
  vgroups = {}
  for name, ielems in tagnamesbydim[2].items():
    if len(ielems) == len(elements):
      vgroups[name] = ...
    elif ielems:
      refs = numpy.array( [None] * len(elements), dtype=object )
      refs[ielems] = triref
      vgroups[name] = topology.SubsetTopology( topo, refs )

  # create geometry
  nmap = { elem.transform: inodes for inodes, elem in zip( vinodes, elements ) }
  fmap = dict.fromkeys( nmap, triref.stdfunc(1) )
  basis = function.function( fmap=fmap, nmap=nmap, ndofs=len(nodes) )
  geom = ( basis[:,_] * nodes ).sum(0)

  return topo.withgroups( vgroups=vgroups ), geom

def gmesh( fname, tags={}, name=None, use_elementary=False ):
  warnings.warn( 'mesh.gmesh has been renamed to mesh.gmsh; please update your code', DeprecationWarning )
  assert not use_elementary, 'support of non-physical gmsh files has been deprecated'
  assert not tags, 'support of external group names has been deprecated; please provide physical names via gmsh'
  return gmsh( fname, name )

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
