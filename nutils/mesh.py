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

from . import topology, function, util, element, numpy, numeric, transform, rational, _
import os

# MESH GENERATORS

def rectilinear( richshape, periodic=(), name='rect' ):
  'rectilinear mesh'

  ndims = len(richshape)
  shape = []
  offset = []
  scale = []
  for v in richshape:
    if isinstance( v, int ):
      assert v > 0
      shape.append( v )
      scale.append( 1 )
      offset.append( 0 )
    else:
      assert numpy.equal( v, numpy.linspace(v[0],v[-1],len(v)) ).all()
      shape.append( len(v)-1 )
      scale.append( (v[-1]-v[0]) / float(len(v)-1) )
      offset.append( v[0] )
  if all( o == 0 for o in offset[1:] ):
    offset = 0
  if all( s == scale[0] for s in scale[1:] ):
    scale = scale[0]
  indices = numeric.grid( shape )
  structure = numpy.empty( indices.shape[1:], dtype=object )

  if isinstance( name, str ):
    wrap = tuple( sh if i in periodic else 0 for i, sh in enumerate(shape) )
    root = element.RootTrans( name, wrap )
  else:
    assert all( ( name.take(0,i) == name.take(2,i) ).all() for i in periodic )
    root = element.RootTransEdges( name, shape )

  reference = element.SimplexReference(1)**ndims
  for index in indices.reshape( ndims, -1 ).T:
    structure[tuple(index)] = element.Element( reference, transform.shift(index) >> root )
  topo = topology.StructuredTopology( structure, periodic=periodic )
  coords = function.ElemFunc( ndims ) * scale + offset
  return topo, coords

def revolve( topo, coords, nelems, degree=3, axis=0 ):
  'revolve coordinates'

  # This is a hack. We need to be able to properly multiply topologies.
  DEGREE = (2,) # Degree of coords element

  structure = numpy.array([ [ element.QuadElement( ndims=topo.ndims+1 ) for elem in topo ] for ielem in range(nelems) ])
  revolved_topo = topology.StructuredTopology( structure.reshape( nelems, *topo.structure.shape ), periodic=0 )
  if nelems % 2 == 0:
    revolved_topo.groups[ 'top' ] = revolved_topo[:nelems//2]
    revolved_topo.groups[ 'bottom' ] = revolved_topo[nelems//2:]

  print 'topo:', revolved_topo.structure.shape
  revolved_func = revolved_topo.splinefunc( degree=(degree,)+DEGREE )

  assert isinstance( coords, function.StaticDot )
  assert coords.array.ndim == 2
  nvertices, ndims = coords.array.shape

  phi = ( .5 + numpy.arange(nelems) - .5*degree ) * ( 2 * numpy.pi / nelems )
  weights = numpy.empty(( nelems, nvertices, ndims+1 ))
  weights[...,:axis] = coords.array[:,:axis]
  weights[...,axis] = numpy.cos(phi)[:,_] * coords.array[:,axis]
  weights[...,axis+1] = numpy.sin(phi)[:,_] * coords.array[:,axis]
  weights[...,axis+2:] = coords.array[:,axis+1:]
  weights = numeric.reshape( weights, 2, 1 )

  return revolved_topo, revolved_func.dot( weights )

def gmesh( path, btags={}, name=None ):
  'gmesh'

  if name is None:
    name = os.path.basename(path)

  if isinstance( btags, str ):
    btags = { i+1: btag for i, btag in enumerate( btags.split(',') ) }

  lines = iter( open( path, 'r' ) )

  assert lines.next() == '$MeshFormat\n'
  version, filetype, datasize = lines.next().split()
  assert lines.next() == '$EndMeshFormat\n'

  assert lines.next() == '$Nodes\n'
  nvertices = int( lines.next() )
  coords = numpy.empty(( nvertices, 3 ))
  for iNode in range( nvertices ):
    items = lines.next().split()
    assert int( items[0] ) == iNode + 1
    coords[ iNode ] = map( float, items[1:] )
  assert lines.next() == '$EndNodes\n'

  assert numpy.all( abs( coords[:,2] ) < 1e-5 ), 'ndims=3 case not yet implemented.'
  coords = coords[:,:2]

  boundary = []
  elements = []
  connected = [ set() for i in range( nvertices ) ]
  nmap = {}
  fmap = {}

  assert lines.next() == '$Elements\n'
  domainelem = element.Element( ndims=2, vertices=[] )
  vertexobjs = numpy.array( [ element.PrimaryVertex( '%s(%d)' % (name,ivertex) ) for ivertex in range(nvertices) ], dtype=object )
  for ielem in range( int( lines.next() ) ):
    items = lines.next().split()
    assert int( items[0] ) == ielem + 1
    elemtype = int( items[1] )
    ntags = int( items[2] )
    tags = [ int(tag) for tag in set( items[3:3+ntags] ) ]
    elemvertices = numpy.asarray( items[3+ntags:], dtype=int ) - 1
    elemcoords = coords[ elemvertices ]
    if elemtype == 1: # boundary edge
      boundary.append(( elemvertices, tags ))
    elif elemtype in (2,4):
      if elemtype == 2: # interior element, triangle
        if numpy.linalg.det( elemcoords[:2] - elemcoords[2] ) < 0:
          elemvertices[:2] = elemvertices[1], elemvertices[0]
        elem = element.TriangularElement( vertices=vertexobjs[ elemvertices ] )
        stdelem = element.PolyTriangle( 1 )
      else: # interior element, quadrilateral
        raise NotImplementedError
        elem = element.QuadElement( ndims=2 )
        stdelem = element.PolyQuad( (2,2) )
      elements.append( elem )
      fmap[ elem ] = stdelem
      nmap[ elem ] = elemvertices
      for n in elemvertices:
        connected[ n ].add( elem )
    elif elemtype == 15: # boundary vertex
      pass
    else:
      raise Exception, 'element type #%d not supported' % elemtype
  assert lines.next() == '$EndElements\n'

  belements = []
  bgroups = {}
  for vertices, tags in boundary:
    n1, n2 = vertices
    elem, = connected[n1] & connected[n2]
    loc_vert_indices = [elem.vertices.index(vertexobjs[v]) for v in vertices] # in [0,1,2]
    match = numpy.array( loc_vert_indices ).sum()-1
    iedge = [1, 0, 2][match]
    belem = elem.edge( iedge )
    belements.append( belem )
    for tag in tags:
      bgroups.setdefault( tag, [] ).append( belem )

  structured = True
  for i, el in enumerate( belements ):
    if not set(belements[i-1].vertices) & set(el.vertices):
      structured = False
      numpy.warnings.warn( 'Boundary elements are not sorted: boundary group will be an unstructured Topology.' )
      break

  linearfunc = function.function( fmap, nmap, nvertices, 2 )
  # Extend linearfunc by bubble functions for the P^1+bubble basis
  fmap_b, nmap_b = {}, {}
  for i, (key,val) in enumerate( nmap.iteritems() ): # enumerate bubble functions
    fmap_b[key] = element.BubbleTriangle( 1 )
    nmap_b[key] = numpy.concatenate( [val, [nvertices+i]] )
  bubblefunc = function.function( fmap_b, nmap_b, nvertices+len(nmap), 2 )

  topo = topology.Topology( elements )
  topo.boundary = topology.Topology( belements )
  topo.boundary.groups = {}
  for tag, group in bgroups.items():
    try:
      tag = btags[tag]
    except:
      pass
    topo.boundary.groups[tag] = topology.Topology( group )

  geom = linearfunc.dot( coords )
  return topo, geom

def triangulation( vertices, nvertices ):
  'triangulation'

  bedges = {}
  nmap = {}
  I = numpy.array( [[2,0],[0,1],[1,2]] )
  for n123 in vertices:
    elem = element.TriangularElement()
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
    
  topo = topology.Topology( nmap )
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
      boundaries[ groupname ] = topology.Topology( belements )
    elif grouptype == 'vertex':
      vertexgroups[ groupname ] = I
    elif grouptype == 'element':
      elemgroups[ groupname ] = topology.Topology( elements[i] for i in I )
    else:
      raise Exception, 'unknown group type: %r' % grouptype

  topo = topology.Topology( elements )
  topo.groups = elemgroups
  if boundaries:
    topo.boundary = topology.Topology( elem for topo in boundaries.values() for elem in topo )
    topo.boundary.groups = boundaries

  for group in elemgroups.values():
    myboundaries = {}
    for name, boundary in boundaries.iteritems():
      belems = [ belem for belem in boundary.elements if belem.parent[0] in group ]
      if belems:
        myboundaries[ name ] = topology.Topology( belems )
    if myboundaries:
      group.boundary = topology.Topology( elem for topo in myboundaries.values() for elem in topo )
      group.boundary.groups = myboundaries

  funcsp = topo.splinefunc( degree=degree )
  coords = ( funcsp[:,_] * points ).sum( 0 )
  return topo, coords #, vertexgroups

def fromfunc( func, nelems, ndims, degree=1 ):
  'piecewise'

  if isinstance( nelems, int ):
    nelems = [ nelems ]
  assert len( nelems ) == func.func_code.co_argcount
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

  scale = rational.Scalar([1,1,1])
  coords = numeric.round( numpy.hstack( [P,Q,R] ).T * float(scale) )

  vertices = numpy.array(
    [ ( i, (i+1)%12, 12+(i-i//3)%8 )   for i in range(12) ]
  + [ ( 12+(i+1)%8, 12+i, i+1+(i//2) ) for i in range( 8) ]
  + [ ( 12+i, 12+(i+1)%8, 20 )         for i in range( 8) ] )
  
  elements = []
  root = element.RootTrans( 'demo', shape=(0,0) )
  reference = element.SimplexReference(2)
  for ielem, elemvertices in enumerate( vertices ):
    elemcoords = coords[ numpy.array(elemvertices) ]
    trans = transform.linear((elemcoords[:2]-elemcoords[2]).T,scale) >> transform.shift(elemcoords[2],scale)
    elem = element.Element( reference, trans >> root )
    elements.append( elem )

  belems = [ elem.edge(2) for elem in elements[:12] ]
  bgroups = { 'top': belems[0:3], 'left': belems[3:6], 'bottom': belems[6:9], 'right': belems[9:12] }

  topo = topology.Topology( elements )
  topo.boundary = topology.Topology( belems )
  topo.boundary.groups = dict( ( tag, topology.Topology( group ) ) for tag, group in bgroups.items() )

  geom = [.5*(xmin+xmax),.5*(ymin+ymax)] \
       + [.5*(xmax-xmin),.5*(ymax-ymin)] * function.ElemFunc( 2 )

  return topo, geom

# vim:shiftwidth=2:foldmethod=indent:foldnestmax=1
