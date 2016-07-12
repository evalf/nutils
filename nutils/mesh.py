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
:func:`gmesh`, :func:`igatool`, and converted to nutils format. Note that no
mesh writers are provided at this point; output is handled by the
:mod:`nutils.plot` module.
"""

from . import topology, function, util, element, numpy, numeric, transform, log, _
import os, warnings

# MESH GENERATORS

def rectilinear( richshape, periodic=(), name='rect', revolved=False ):
  'rectilinear mesh'

  ndims = len(richshape)
  shape = []
  offset = []
  scale = []
  uniform = True
  for v in richshape:
    if isinstance( v, int ):
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
  topo = topology.StructuredTopology( root, axes ).withsubs()

  if uniform:
    if all( o == offset[0] for o in offset[1:] ):
      offset = offset[0]
    if all( s == scale[0] for s in scale[1:] ):
      scale = scale[0]
    geom = function.LocalCoords( ndims ) * scale + offset
  else:
    funcsp = topo.splinefunc( degree=1, periodic=() )
    coords = numeric.meshgrid( *richshape ).reshape( ndims, -1 )
    geom = ( funcsp * coords ).sum( -1 )

  if revolved:
    topo = topology.RevolvedTopology(topo)
    theta = function.RevolutionAngle()
    if topo.ndims == 1:
      r, = function.revolved(geom)
      geom = r * function.stack([ function.cos(theta), function.sin(theta) ])
    elif topo.ndims == 2:
      r, y = function.revolved(geom)
      geom = function.stack([ r * function.cos(theta), y, r * function.sin(theta) ])
    else:
      raise NotImplementedError( 'ndims={}'.format( topo.ndims ) )

  return topo, geom

def gmesh( fname, tags={}, name=None, use_elementary=False ):
  """Gmesh parser

  Parser for Gmesh files in `.msh` format. See the `Gmesh manual <http://geuz.org/gmsh/doc/texinfo/gmsh.html>`_ for details.

  Args:
      fname (str): Path to mesh file
      tags (dict, optional): Dictionary mapping gmesh group IDs to names
      name (str, optional): Name of parsed topology, defaults to None
      use_elementary (bool, optional): Option to indicate whether Gmsh is used with elementary groups only (i.e. no physical groups are defined), defaults to False

  Returns:
      topo (:class:`nutils.topology.Topology`): Topology of parsed Gmesh file
      geom (:class:`nutils.function.Array`): Isoparametric map

  """

  ndims = 2

  if isinstance( tags, str ):
    warnings.warn('String format for groups is depricated, please use dictionary format instead with (key,value)=(physical ID,group name)',DeprecationWarning)
    tags = { i+1: tag for i, tag in enumerate( tags.split(',') ) }

  #Parse the file
  sections = {}
  lines = iter(open(fname,'r') if isinstance(fname,str) else fname)
  for line in lines:
    line = line.strip()
    assert line[0]=='$'
    sname = line[1:]
    slines = []
    for sline in lines:
      sline = sline.strip()
      if sline=='$End%s'%sname:
        break
      slines.append( sline ) 
    sections[sname] = slines  

  #PhysicalNames
  names = sections.pop( 'PhysicalNames', [0] )
  assert int(names.pop(0)) == len(names)
  for line in names:
    words = line.split()
    nid = int(words[1])
    name = words[2].strip( '"' )
    if nid not in tags:
      tags[nid] = name
    else:
      warnings.warn( 'renamed physical group name {!r} to {!r}'.format( name, tags[nid] ) )
        
  #Nodes
  nodedata = sections.pop('Nodes')
  nnodes = int(nodedata.pop(0))
  assert len(nodedata)==nnodes
        
  coords = numpy.empty((nnodes,3))
  coords[:] = numpy.nan
  nidmap = {}
  for line in nodedata:
    words = line.split()
    nid = len(nidmap)
    nidmap[int(words[0])] = nid
    coords[nid] = [ float(n) for n in words[1:] ]
  assert not numpy.isnan(coords).any()
  assert numpy.all( coords[:,2] ) == 0, 'ndims=3 case not yet implemented.'
  coords = coords[:,:2]

  #Elements
  elemdata = sections.pop('Elements')
  nelems = int(elemdata.pop(0))
  assert len(elemdata)==nelems
  flip = transform.affine( -1, [1] )

  elems = {}
  elemgroups = {}
  edges = {}
  ifaces = {}
  vertexgroups = {}
  vertices = {}
  edgegroups = {}
  fmap = {}
  nmap = {}
  for line in elemdata:
    words = line.split()
    etype = int(words[1])
    ntags = int(words[2])
    if use_elementary:
      assert words[3] == '0', 'option use_elementary=True conflicts with non-zero physical tag'
      tag = tags.get( int( words[4] ), 'elementary' + words[4] )
    else:
      tag = tags.get( int( words[3] ), 'physical' + words[3] )

    nids = numpy.array([ nidmap[int(gmshid)] for gmshid in words[3+ntags:] ])
    elemkey = tuple(sorted(nids))
    if etype == 1: # Linear line
      edgegroups.setdefault(tag,[]).append(elemkey)
    elif etype == 2: # linear triangle
      assert len(nids)==3
      try:
        elem = elems[elemkey]
      except KeyError:
        elemcoords = coords[nids]
        if numpy.linalg.det( elemcoords[:2] - elemcoords[2] ) < 0:
          nids[:2] = nids[1], nids[0]
        ref = element.getsimplex(2)
        maptrans = transform.maptrans(ref.vertices,nids if not name else [name+str(nid) for nid in nids])
        elem = element.Element(ref,maptrans)
        elems[elemkey] = elem
        fmap[maptrans] = (ref.stdfunc(1),None),
        nmap[maptrans] = nids

        #Extract the edges
        for iedge, iverts in enumerate([[1,2],[0,2],[0,1]]):
          edge = elem.edge(iedge)
          key = tuple(sorted(nids[iverts]))
          try:
            opposite = edges.pop( key )
          except KeyError:
            edges[key] = edge
          else:
            assert edge.reference == opposite.reference
            iface = element.Element( edge.reference, edge.transform, opposite.transform << flip )
            #assert iface.transform.apply( iface.reference.vertices ) == iface.opposite.apply( iface.reference.vertices )
            ifaces[key] = iface

        #Extract the vertices
        vref = element.getsimplex(0)
        for ivertex in range(3): #GMSH and Nutils node ordering coincide
          zeroD_to_twoD = transform.affine( linear=numpy.zeros(shape=(2,0),dtype=int), offset=ref.vertices[ivertex], isflipped=False )
          vmaptrans = maptrans << zeroD_to_twoD
          vertexkey = (nids[ivertex],)
          velem = element.Element(vref,vmaptrans)
          vertices.setdefault(vertexkey,[]).append(velem)

      elemgroups.setdefault(tag,[]).append(elem)
    elif etype == 15:
      if not use_elementary:
        vertexgroups.setdefault(tag,[]).append(elemkey)
    else:
      raise NotImplementedError('Unknown GMSH element type %i' % etype)

  subtopos = { name: topology.UnstructuredTopology( ndims, subtopo ).withsubs() for name, subtopo in elemgroups.items() }
  subbtopos = {}
  subitopos = {}
  for name, keys in edgegroups.items():
    bgrouptopo = []
    igrouptopo = []
    for key in keys:
      try:
        bgrouptopo.append( edges[key] )
      except KeyError:
        igrouptopo.append( ifaces[key] )
    if bgrouptopo:
      subbtopos[name] = topology.UnstructuredTopology( ndims-1, bgrouptopo ).withsubs()
    if igrouptopo:
      subitopos[name] = topology.UnstructuredTopology( ndims-1, igrouptopo ).withsubs()
  subptopos = { name: topology.UnstructuredTopology( 0, [vertex for vertexkey in vertexkeys for vertex in vertices[vertexkey]] ).withsubs() for name, vertexkeys in vertexgroups.items() }

  topo = topology.UnstructuredTopology( ndims, elems.values() ).withsubs( subtopos )
  topo.boundary = topology.UnstructuredTopology( ndims-1, edges.values() ).withsubs( subbtopos )
  #topo.interfaces = topology.UnstructuredTopology( ndims-1, ifaces.values() ).withsubs( subitopos )
  topo.points = topology.UnstructuredTopology( 0, [vertex for vertexkeys in vertexgroups.values() for vertexkey in vertexkeys for vertex in vertices[vertexkey]] ).withsubs( subptopos )

  alltags = set(subtopos) | set(subbtopos) | set(subitopos) | set(subptopos)
  for tag in set(tags.values()) - alltags:
    warnings.warn('tag %r defined but not used' % tag )

  log.info('parsed GMSH file:')
  log.info('* nodes (#%d)' % nnodes)
  log.info('* topology (#%d) with groups: %s' % (len(topo), ', '.join('%s (#%d)' % (name,len(topo[name])) for name in subtopos)))
  log.info('* boundary (#%d) with groups: %s' % (len(topo.boundary), ', '.join('%s (#%d)' % (name,len(topo.boundary[name])) for name in subbtopos)))
  log.info('* interfaces (#%d) with groups: %s' % (len(topo.interfaces), ', '.join('%s (#%d)' % (name,len(topo.interfaces[name])) for name in subitopos)))
  log.info('* points (#%d) with groups: %s' % (len(topo.points), ', '.join('%s (#%d)' % (name,len(topo.points[name])) for name in subitopos)))

  linearfunc = function.function( fmap=fmap, nmap=nmap, ndofs=nnodes, ndims=topo.ndims )
  geom = ( linearfunc[:,_] * coords ).sum(0)
  return topo, geom

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
    
  topo = topology.UnstructuredTopology( ndims, nmap ).withsubs()
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

  splinefunc = function.function( fmap, nmap, NumberOfPoints, ndims )

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
  elements = [ element.Element( reference, root << transform.simplex(coords[iverts]) ) for iverts in vertices ]
  belems = [ elem.edge(0) for elem in elements[:12] ]
  bgroups = { 'top': belems[0:3], 'left': belems[3:6], 'bottom': belems[6:9], 'right': belems[9:12] }

  topo = topology.UnstructuredTopology( 2, elements ).withsubs()
  subbtopos = { name: topology.UnstructuredTopology( 1, subtopo ) for name, subtopo in bgroups.items() }
  topo.boundary = topology.UnstructuredTopology( 1, belems ).withsubs( subbtopos )

  geom = [.5*(xmin+xmax),.5*(ymin+ymax)] \
       + [.5*(xmax-xmin),.5*(ymax-ymin)] * function.LocalCoords( 2 )

  return topo, geom

# vim:shiftwidth=2:softtabstop=2:expandtab:foldmethod=indent:foldnestmax=1
