from . import topology, function, util, element, numpy, numeric, _
import os

# MESH GENERATORS

def rectilinear( nodes, periodic=(), name='rect' ):
  'rectilinear mesh'

  nodes = [ numpy.linspace(*n) if len(n) == 3 and isinstance(n,tuple) else numpy.asarray(n) for n  in nodes ]
  ndims = len(nodes)
  indices = numpy.ogrid[ tuple( slice(len(n)-1) for n in nodes ) ]
  domainelem = element.Element( ndims=ndims, id=name )
  structure = util.objmap( lambda *index: element.QuadElement(
    ndims=ndims,
    parent=( domainelem, element.AffineTransformation(
      offset=[ n[i] for n,i in zip(nodes,index) ],
      transform=numpy.diag([ n[i+1]-n[i] for n,i in zip(nodes,index) ]) ) ),
    id='{}.quad({})'.format(name,','.join(str(i) for i in index)) ), *indices )
  topo = topology.StructuredTopology( structure )
  coords = function.ElemFunc( domainelem )
  if periodic:
    topo = topo.make_periodic( periodic )
  return topo, coords

def revolve( topo, coords, nelems, degree=4, axis=0 ):
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
  nnodes, ndims = coords.array.shape

  phi = ( 1 + numpy.arange(nelems) - .5*degree ) * ( 2 * numpy.pi / nelems )
  weights = numpy.empty(( nelems, nnodes, ndims+1 ))
  weights[...,:axis] = coords.array[:,:axis]
  weights[...,axis] = numpy.cos(phi)[:,_] * coords.array[:,axis]
  weights[...,axis+1] = numpy.sin(phi)[:,_] * coords.array[:,axis]
  weights[...,axis+2:] = coords.array[:,axis+1:]
  weights = numeric.reshape( weights, 2, 1 )

  return revolved_topo, revolved_func.dot( weights )

def gmesh( path, btags=[], name=None ):
  'gmesh'

  if name is None:
    name = os.path.basename(path)

  if isinstance( btags, str ):
    btags = ( 'all,' + btags ).split( ',' )

  lines = iter( open( path, 'r' ) )

  assert lines.next() == '$MeshFormat\n'
  version, filetype, datasize = lines.next().split()
  assert lines.next() == '$EndMeshFormat\n'

  assert lines.next() == '$Nodes\n'
  nnodes = int( lines.next() )
  coords = numpy.empty(( nnodes, 3 ))
  for iNode in range( nnodes ):
    items = lines.next().split()
    assert int( items[0] ) == iNode + 1
    coords[ iNode ] = map( float, items[1:] )
  assert lines.next() == '$EndNodes\n'

  if numpy.all( abs( coords[:,2] ) < 1e-5 ):
    ndims = 2
    coords = coords[:,:2]
  else:
    ndims = 3

  boundary = []
  elements = []
  connected = [ set() for i in range( nnodes ) ]

  nmap = {}
  fmap = {}

  assert lines.next() == '$Elements\n'
  domainelem = element.Element( ndims=2, id=name )
  for ielem in range( int( lines.next() ) ):
    items = lines.next().split()
    assert int( items[0] ) == ielem + 1
    elemtype = int( items[1] )
    ntags = int( items[2] )
    tags = [ int(tag) for tag in set( items[3:3+ntags] ) ]
    elemnodes = numpy.asarray( items[3+ntags:], dtype=int ) - 1
    elemcoords = coords[ elemnodes ]
    if elemtype == 1:
      boundary.append(( elemnodes, tags ))
    elif elemtype in (2,4):
      if elemtype == 2:
        parent = domainelem, element.AffineTransformation( offset=elemcoords[2], transform=(elemcoords[:2]-elemcoords[2]).T )
        elem = element.TriangularElement( id='{}.tri({})'.format(domainelem.id,ielem), parent=parent )
        stdelem = element.PolyTriangle( 1 )
      else:
        raise NotImplementedError
        elem = element.QuadElement( ndims=2 )
        stdelem = element.PolyQuad( (2,2) )
      elements.append( elem )
      fmap[ elem ] = stdelem
      nmap[ elem ] = elemnodes
      for n in elemnodes:
        connected[ n ].add( elem )
    elif elemtype == 15: # vertex?
      pass
    else:
      raise Exception, 'element type #%d not supported' % elemtype
  assert lines.next() == '$EndElements\n'

  belements = []
  bgroups = {}
  for nodes, tags in boundary:
    n1, n2 = nodes
    elem, = connected[n1] & connected[n2]
    e1, e2, e3 = nmap[ elem ]
    if e1==n1 and e2==n2:
      iedge = 1
    elif e2==n1 and e3==n2:
      iedge = 2
    elif e3==n1 and e1==n2:
      iedge = 0
    else:
      raise Exception, 'cannot match edge, perhaps order is reversed in gmesh'
    belem = elem.edge( iedge )
    belements.append( belem )
    for tag in tags:
      bgroups.setdefault( tag, [] ).append( belem )

  linearfunc = function.function( fmap, nmap, nnodes, ndims )
  namedfuncs = { 'spline2': linearfunc }
  topo = topology.UnstructuredTopology( elements, ndims=2, namedfuncs=namedfuncs )
  topo.boundary = topology.UnstructuredTopology( belements, ndims=1 )
  topo.boundary.groups = dict( ( btags[tag], topology.UnstructuredTopology( group, ndims=1 ) ) for tag, group in bgroups.items() )

  return topo, function.ElemFunc( domainelem )

def triangulation( nodes, nnodes ):
  'triangulation'

  bedges = {}
  nmap = {}
  I = numpy.array( [[2,0],[0,1],[1,2]] )
  for n123 in nodes:
    elem = element.TriangularElement()
    nmap[ elem ] = n123
    for iedge, (n1,n2) in enumerate( n123[I] ):
      try:
        del bedges[ (n2,n1) ]
      except KeyError:
        bedges[ (n1,n2) ] = elem, iedge

  dofaxis = function.DofAxis( nnodes, nmap )
  stdelem = element.PolyTriangle( 1 )
  linearfunc = function.Function( dofaxis=dofaxis, stdmap=dict.fromkeys(nmap,stdelem) )
  namedfuncs = { 'spline2': linearfunc }

  connectivity = dict( bedges.iterkeys() )
  N = list( connectivity.popitem() )
  while connectivity:
    N.append( connectivity.pop( N[-1] ) )
  assert N[0] == N[-1]

  structure = []
  for n12 in zip( N[:-1], N[1:] ):
    elem, iedge = bedges[ n12 ]
    structure.append( elem.edge( iedge ) )
    
  topo = topology.UnstructuredTopology( list(nmap), ndims=2, namedfuncs=namedfuncs )
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
  degree = 4
  ndims = 2
  nmap = {}
  fmap = {}

  poly = element.PolyLine( element.PolyLine.bernstein_poly( degree ) )**ndims

  for ielem in range(NumberOfElements):

    cellpoints = vtk.vtkIdList()
    mesh.GetCellPoints( ielem, cellpoints )
    nids = util.arraymap( cellpoints.GetId, int, range(cellpoints.GetNumberOfIds()) )

    assert mesh.GetCellType(ielem) == vtk.VTK_HIGHER_ORDER_QUAD
    nb = degree**2
    assert len(nids) == nb

    n = range( *util.arraymap( Cindi.GetComponent, int, ielem, [0,1] ) )
    I = util.arraymap( Cij.GetComponent, int, n, 0 )
    J = util.arraymap( Cij.GetComponent, int, n, 1 )
    Ce = numpy.zeros(( nb, nb ))
    Ce[I,J] = util.arraymap( Cv.GetComponent, float, n, 0 )

    elem = element.QuadElement( id='%s.quad(%d)' % ( name, ielem ), ndims=ndims )
    elements.append( elem )

    fmap[ elem ] = element.ExtractionWrapper( poly, Ce.T )
    nmap[ elem ] = nids

  linearfunc = function.function( fmap, nmap, NumberOfPoints, ndims )
  namedfuncs = { 'spline%d' % degree: linearfunc }

  boundaries = {}
  elemgroups = {}
  nodegroups = {}
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
      boundaries[ groupname ] = topology.UnstructuredTopology( belements, ndims=ndims-1 )
    elif grouptype == 'node':
      nodegroups[ groupname ] = I
    elif grouptype == 'element':
      elemgroups[ groupname ] = topology.UnstructuredTopology( [ elements[i] for i in I ], namedfuncs=namedfuncs, ndims=2 )
    else:
      raise Exception, 'unknown group type: %r' % grouptype

  topo = topology.UnstructuredTopology( elements, namedfuncs=namedfuncs, ndims=ndims )
  topo.groups = elemgroups
  topo.boundary = topology.UnstructuredTopology( elements=[], ndims=ndims-1 )
  topo.boundary.groups = boundaries

  for group in elemgroups.values():
    myboundaries = {}
    for name, boundary in boundaries.iteritems():
      belems = [ belem for belem in boundary.elements if belem.parent[0] in group ]
      if belems:
        myboundaries[ name ] = topology.UnstructuredTopology( belems, ndims=ndims-1 )
    group.boundary = topology.UnstructuredTopology( elements=[], ndims=ndims-1 )
    group.boundary.groups = myboundaries

  funcsp = topo.splinefunc( degree=degree )
  coords = ( funcsp[:,_] * points ).sum( 0 )
  return topo, coords #, nodegroups

def fromfunc( func, nelems, ndims, degree=2 ):
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

  coords = [.5*(xmin+xmax),.5*(ymin+ymax)] \
         + [.5*(xmax-xmin),.5*(ymax-ymin)] * numpy.hstack( [P,Q,R] ).T
  vertices = numpy.array(
    [ ( i, (i+1)%12, 12+(i-i//3)%8 )   for i in range(12) ]
  + [ ( 12+(i+1)%8, 12+i, i+1+(i//2) ) for i in range( 8) ]
  + [ ( 12+i, 12+(i+1)%8, 20 )         for i in range( 8) ] )
  
  domainelem = element.Element( ndims=2, id='demo' )
  elements = []
  for ielem, elemnodes in enumerate( vertices ):
    elemcoords = coords[ numpy.array(elemnodes) ]
    parent = domainelem, element.AffineTransformation( offset=elemcoords[2], transform=(elemcoords[:2]-elemcoords[2]).T )
    elem = element.TriangularElement( id='{}.tri({})'.format(domainelem.id,ielem), parent=parent )
    elements.append( elem )

  fmap = dict.fromkeys( elements, element.PolyTriangle(1) )
  nmap = dict( zip( elements, vertices ) )
  belems = [ elem.edge(1) for elem in elements[:12] ]
  bgroups = { 'top': belems[0:3], 'left': belems[3:6], 'bottom': belems[6:9], 'right': belems[9:12] }

  linearfunc = function.function( fmap, nmap, ndofs=21, ndims=2 )
  namedfuncs = { 'spline2': linearfunc }
  topo = topology.UnstructuredTopology( elements, ndims=2, namedfuncs=namedfuncs )
  topo.boundary = topology.UnstructuredTopology( belems, ndims=1 )
  topo.boundary.groups = dict( ( tag, topology.UnstructuredTopology( group, ndims=1 ) ) for tag, group in bgroups.items() )

  return topo, function.ElemFunc( domainelem )

# vim:shiftwidth=2:foldmethod=indent:foldnestmax=1
