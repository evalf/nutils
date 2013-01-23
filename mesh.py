from . import topology, function, util, element, numpy, numeric, _
import os

# MESH GENERATORS

class ElemScale( function.ArrayFunc ):
  'trivial scale'

  def __init__( self, rectelem ):
    'constructor'

    self.rectelem = rectelem
    function.ArrayFunc.__init__( self, args=[function.ELEM,rectelem], evalf=self.elemscale, shape=[rectelem.ndims] )

  @staticmethod
  def elemscale( elem, rectelem ):
    'evaluate'

    assert elem.ndims <= rectelem.ndims
    while elem.ndims < rectelem.ndims:
      elem, transform = elem.context or elem.parent
    elem, transform = elem.parent
    scale = transform.transform
    while elem is not rectelem:
      elem, transform = elem.parent
      scale = scale * transform.transform
    return scale

class ElemFunc( function.ArrayFunc ):
  'trivial func'

  def __init__( self, rectelem ):
    'constructor'

    self.rectelem = rectelem
    function.ArrayFunc.__init__( self, args=[function.ELEM,function.POINTS,rectelem], evalf=self.elemfunc, shape=[rectelem.ndims] )

  @staticmethod
  def elemfunc( elem, points, rectelem ):
    'evaluate'

    assert elem.ndims <= rectelem.ndims
    while elem.ndims < rectelem.ndims:
      elem, transform = elem.context or elem.parent
      points = transform.eval( points )
    while elem is not rectelem:
      elem, transform = elem.parent
      points = transform.eval( points )
    return points.coords

  def localgradient( self, ndims ):
    'local gradient'

    scale = ElemScale( self.rectelem )
    if ndims == self.rectelem.ndims:
      if ndims == 1:
        return scale[_]
      return function.Diagonalize( scale, [0,1] )
    assert ndims < self.rectelem.ndims
    return function.Transform( self.rectelem.ndims, ndims ) * scale[:,_]

  def find( self, elem, C ):
    'find coordinates'

    assert C.ndim == 2 and C.shape[1] == self.rectelem.ndims
    assert elem.ndims == self.rectelem.ndims # for now
    elem, transform = elem.parent
    offset = transform.offset
    scale = transform.transform
    while elem is not self.rectelem:
      elem, transform = elem.parent
      offset = transform.offset + offset * transform.transform
      scale = scale * transform.transform
    selection = numpy.ones( C.shape[0], dtype=bool )
    for idim in range( self.rectelem.ndims ):
      for side in range(2):
        newsel = C[:,idim] >= offset[idim] if side == 0 else C[:,idim] <= offset[idim] + scale[idim]
        selection[selection] &= newsel
        C = C[newsel]
        if not C.size:
          return None, None
    C -= offset
    C /= scale
    return element.LocalPoints( C ), selection

def rectilinear( nodes, periodic=(), name='rect' ):
  'rectilinear mesh'

  nodes = [ numpy.linspace(*n) if len(n) == 3 and isinstance(n,tuple) else numpy.asarray(n) for n  in nodes ]
  ndims = len(nodes)
  indices = numpy.ogrid[ tuple( slice(len(n)-1) for n in nodes ) ]
  rectelem = element.Element( ndims=ndims, id=name )
  structure = util.objmap( lambda *index: element.QuadElement(
    ndims=ndims,
    parent=( rectelem, element.AffineTransformation(
      offset=[ n[i] for n,i in zip(nodes,index) ],
      transform=[ n[i+1]-n[i] for n,i in zip(nodes,index) ] ) ),
    id='{}.quad({})'.format(name,','.join(str(i) for i in index)) ), *indices )
  topo = topology.StructuredTopology( structure )
  coords = ElemFunc( rectelem )
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
  nNodes = int( lines.next() )
  coords = numpy.empty(( nNodes, 3 ))
  for iNode in range( nNodes ):
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
  connected = [ set() for i in range( nNodes ) ]

  nmap = {}
  fmap = {}

  assert lines.next() == '$Elements\n'
  for iElem in range( int( lines.next() ) ):
    items = lines.next().split()
    assert int( items[0] ) == iElem + 1
    elemType = int( items[1] )
    nTags = int( items[2] )
    tags = [ int(tag) for tag in set( items[3:3+nTags] ) ]
    elemnodes = numpy.asarray( items[3+nTags:], dtype=int ) - 1
    if elemType == 1:
      boundary.append(( elemnodes, tags ))
    elif elemType in (2,4):
      if elemType == 2:
        elem = element.TriangularElement( id='{}.tri({})'.format(name,iElem), index=iElem )
        stdelem = element.PolyTriangle( 1 )
      else:
        elem = element.QuadElement( ndims=2 )
        stdelem = element.PolyQuad( (2,2) )
      elements.append( elem )
      fmap[ elem ] = stdelem
      nmap[ elem ] = elemnodes
      for n in elemnodes:
        connected[ n ].add( elem )
    elif elemType == 15: # vertex?
      pass
    else:
      raise Exception, 'element type #%d not supported' % elemType
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

  dofaxis = function.DofMap( nNodes, topology.ElemMap(nmap,ndims,overlap=False) )
  linearfunc = function.Function( dofaxis=dofaxis, stdmap=topology.ElemMap(fmap,ndims,overlap=False), igrad=0 )
  namedfuncs = { 'spline2': linearfunc }
  topo = topology.UnstructuredTopology( elements, ndims=2, namedfuncs=namedfuncs )
  topo.boundary = topology.UnstructuredTopology( belements, ndims=1 )
  topo.boundary.groups = dict( ( btags[tag], topology.UnstructuredTopology( group, ndims=1 ) ) for tag, group in bgroups.items() )

  return topo, linearfunc.dot( coords )

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

def igatool( path ):
  'igatool mesh'

  import vtk

  reader = vtk.vtkXMLUnstructuredGridReader()
  reader.SetFileName( path )
  reader.Update()

  mesh = reader.GetOutput()

  FieldData = mesh.GetFieldData()
  CellData = mesh.GetCellData()

  NumberOfPoints = mesh.GetNumberOfPoints()
  NumberOfElements = mesh.GetNumberOfCells()
  NumberOfArrays = FieldData.GetNumberOfArrays()

  points = util.arraymap( mesh.GetPoint, float, range(NumberOfPoints) )
  Cij = FieldData.GetArray( 'Cij' )
  Cv = FieldData.GetArray( 'Cv' )
  Cindi = CellData.GetArray( 'Elem_extr_indi')

  elements = []
  splineinfo = {}
  degree = 4
  poly = element.PolyQuad( (degree,degree) )

  for ie in range(NumberOfElements):

    cellpoints = vtk.vtkIdList()
    mesh.GetCellPoints( ie, cellpoints )
    nids = util.arraymap( cellpoints.GetId, int, range(cellpoints.GetNumberOfIds()) )

    assert mesh.GetCellType(ie) == vtk.VTK_HIGHER_ORDER_QUAD
    nb = degree**2
    assert len(nids) == nb

    n = range( *util.arraymap( Cindi.GetComponent, int, ie, [0,1] ) )
    I = util.arraymap( Cij.GetComponent, int, n, 0 )
    J = util.arraymap( Cij.GetComponent, int, n, 1 )
    Ce = numpy.zeros(( nb, nb ))
    Ce[I,J] = util.arraymap( Cv.GetComponent, float, n, 0 )

    elem = element.QuadElement( ndims=2 )
    elements.append( elem )
    splineinfo[ elem ] = function.ExtractionWrapper( poly, Ce ), nids

  funinfo = { 'spline%d' % degree: splineinfo }

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
      boundaries[ groupname ] = topology.UnstructuredTopology( belements, ndims=1 )
    elif grouptype == 'node':
      nodegroups[ groupname ] = I
    elif grouptype == 'element':
      elemgroups[ groupname ] = topology.UnstructuredTopology( [ elements[i] for i in I ], funinfo=funinfo, ndims=2 )
    else:
      raise Exception, 'unknown group type: %r' % grouptype

  topo = topology.UnstructuredTopology( elements, funinfo=funinfo, ndims=2 )
  topo.groups = elemgroups
  topo.boundary = topology.UnstructuredTopology( elements=[], ndims=1 )
  topo.boundary.groups = boundaries

  for group in elemgroups.values():
    myboundaries = {}
    for name, boundary in boundaries.iteritems():
      belems = [ belem for belem in boundary.elements if belem.parent[0] in group ]
      if belems:
        myboundaries[ name ] = topology.UnstructuredTopology( belems, ndims=1 )
    group.boundary = topology.UnstructuredTopology( elements=[], ndims=1 )
    group.boundary.groups = myboundaries

  coords = topo.splinefunc( degree=degree, weights=points )
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

# vim:shiftwidth=2:foldmethod=indent:foldnestmax=1
