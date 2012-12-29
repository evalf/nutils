from . import topology, function, util, element, numpy, numeric, _

class ElemScale( function.ArrayFunc ):
  'element-constant array'

  def __init__( self, scalemap ):
    'constructor'

    self.scalemap = scalemap
    self.__class__.__base__.__init__( self, args=[function.ELEM,scalemap], evalf=self.evalscale, shape=[scalemap.ndims] )

  @staticmethod
  def evalscale( elem, scalemap ):
    'evaluate'

    assert elem.ndims <= scalemap.ndims
    while elem.ndims < scalemap.ndims:
      elem, transform = elem.context or elem.parent

    scale = 1
    elemscale = scalemap.get( elem )
    while elemscale is None:
      elem, transform = elem.parent
      scale *= transform.transform
      elemscale = scalemap.get( elem )
    scale *= elemscale

    return scale

  def localgradient( self, ndims ):
    'local gradient'

    return function.ZERO( self.shape + (ndims,) )

class ElemCoords( function.ArrayFunc ):
  'define function by affine transform of elem-local coords'

  def __init__( self, offsetmap ):
    'constructor'

    self.offsetmap = offsetmap
    function.ArrayFunc.__init__( self, args=[function.ELEM,function.POINTS,offsetmap], evalf=self.evaloffset, shape=[offsetmap.ndims] )

  @staticmethod
  def evaloffset( elem, points, offsetmap ):
    'evaluate'

    assert elem.ndims <= offsetmap.ndims
    while elem.ndims < offsetmap.ndims:
      elem, transform = elem.context or elem.parent
      points = transform.eval( points )

    offset = 0
    scale = 1
    elemoffset = offsetmap.get( elem )
    while elemoffset is None:
      elem, transform = elem.parent
      scale /= transform.transform
      offset += transform.offset * scale
      elemoffset = offsetmap.get( elem )
    offset += elemoffset * scale

    return points.coords + offset

  def localgradient( self, ndims ):
    'local gradient'

    if ndims == self.offsetmap.ndims:
      return function.Diagonalize( function.Expand( function.StaticArray([1]), [ndims] ), [0,1] )

    assert ndims < self.offsetmap.ndims
    return function.Transform( self.offsetmap.ndims, ndims )

# MESH GENERATORS

def rectilinear( nodes, periodic=() ):
  'rectilinear mesh'

  uniform = all( len(n) == 3 and isinstance(n,tuple) for n in nodes )
  ndims = len(nodes)
  indices = numpy.ogrid[ tuple( slice( n[2] if uniform else len(n)-1 ) for n in nodes ) ]
  if uniform:
    indices = numpy.ogrid[ tuple( slice(n-1) for (a,b,n) in nodes ) ]
    scale = numpy.array( [ (b-a)/float(n-1) for (a,b,n) in nodes ], dtype=float )
  else:
    indices = numpy.ogrid[ tuple( slice(len(n)-1) for n in nodes ) ]
  scalemap = {}
  structure = numpy.frompyfunc( lambda *s: element.QuadElement( ndims ), ndims, 1 )( *indices )
  topo = topology.StructuredTopology( structure )
  offsetmap = {}
  for elem_index in numpy.broadcast( structure, *indices ):
    elem = elem_index[0]
    index = elem_index[1:]
    if uniform:
      scalemap[elem] = scale
      offsetmap[elem] = index + numpy.array([ a for (a,b,n) in nodes ]) / scale
    else:
      offset0 = numpy.array([ n[i  ] for n, i in zip( nodes, index ) ])
      offset1 = numpy.array([ n[i+1] for n, i in zip( nodes, index ) ])
      scalemap[elem] = offset1 - offset0
      offsetmap[elem] = offset0 / ( offset1 - offset0 )
  coords = ElemCoords( topology.ElemMap(offsetmap,ndims,overlap=False) ) \
         * ElemScale( topology.ElemMap(scalemap,ndims,overlap=False) )

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

def gmesh( path, btags=[] ):
  'gmesh'

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
        elem = element.TriangularElement()
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
