from . import element, function, util, numpy, _

class Topology( object ):
  'topology base class'

  def __add__( self, other ):
    'add topologies'

    if self is other:
      return self
    assert self.ndims == other.ndims
    return UnstructuredTopology( set(self) | set(other), ndims=self.ndims )

  def integrate( self, func, coords, ischeme='gauss2' ):
    'integrate'

    if coords:
      J = function.Jacobian( coords )
      ndims, = coords.shape
      if ndims == self.ndims:
        detJ = J.det( 0, 1 )
      elif self.ndims == 1:
        detJ = J[:,0].norm2( 0 )
      elif ndims == 3 and self.ndims == 2:
        detJ = function.Cross( J[:,0], J[:,1], axis=1 ).norm2( 0 )
      elif self.ndims == 0:
        detJ = 1.
      else:
        raise NotImplementedError, 'cannot compute determinant for %dx%d jacobian' % J.shape[:2]
    else:
      detJ = 1.

    topo = util.progressbar( self, title='integrating' )
    if isinstance( func, (list,tuple) ):
      A = [ numpy.zeros( f.shape ) for f in func ]
      func = function.Tuple( function.Tuple(( function.ArrayIndex(f), function.Integrate(f,detJ) )) for f in func )
      for elem in topo:
        xi = elem(ischeme)
        for Ai, (index,data) in zip( A, func(xi) ):
          Ai[ index ] += data
    else:
      A = numpy.zeros( func.shape )
      func = function.Tuple(( function.ArrayIndex(func), function.Integrate(func,detJ) ))
      for elem in topo:
        xi = elem(ischeme)
        index, data = func(xi)
        A[ index ] += data
    return A

class StructuredTopology( Topology ):
  'structured topology'

  def __init__( self, structure, periodic=None ):
    'constructor'

    assert isinstance( structure, numpy.ndarray )
    self.ndims = structure.ndim
    self.structure = structure
    self.periodic = periodic
    self.groups = {}

  def __len__( self ):
    'number of elements'

    return self.structure.size

  def __iter__( self ):
    'iterate'

    return self.structure.flat

  def __getitem__( self, item ):
    'subtopology'

    return self.groups[ item ] if isinstance( item, str ) \
      else StructuredTopology( self.structure[item] )

  @util.cacheprop
  def boundary( self ):
    'boundary'

    shape = numpy.asarray( self.structure.shape ) + 1
    nodes = numpy.arange( numpy.product(shape) ).reshape( shape )
    stdelem = element.PolyQuad( (2,)*(self.ndims-1) )

    boundaries = []
    for iedge in range( 2 * self.ndims ):
      idim = iedge // 2
      iside = iedge % 2
      s = [ slice(None,None,1-2*iside) ] * idim \
        + [ -iside ] \
        + [ slice(None,None,2*iside-1) ] * (self.ndims-idim-1)
      # TODO: check that this is correct for all dimensions; should match conventions in elem.edge
      belems = numpy.frompyfunc( lambda elem: elem.edge( iedge ), 1, 1 )( self.structure[s] )
      boundaries.append( StructuredTopology( belems ) )

    if self.ndims == 2:
      structure = numpy.concatenate([ boundaries[i].structure for i in [0,2,1,3] ])
      topo = StructuredTopology( structure, periodic=0 )
    else:
      allbelems = [ belem for belem in boundary.structure.flat for boundary in boundaries ]
      topo = UnstructuredTopology( allbelems, ndims=self.ndims-1 )

    topo.groups = dict( zip( ( 'left', 'right', 'bottom', 'top', 'front', 'back' ), boundaries ) )
    return topo

  def splinefunc( self, degree ):
    'spline from nodes'

    if isinstance( degree, int ):
      degree = ( degree, ) * self.ndims

    extractions = numpy.ones( (1,1,1), dtype=float )
    indices = numpy.array( 0 )
    slices = []
    for p, nelems in zip( degree, self.structure.shape ):
      n = min( nelems, 2*(p-1)-1 )
      ex = numpy.empty(( n, p, p ))
      ex[0] = numpy.eye( p )
      for i in range( 1, n ):
        ex[i] = numpy.eye( p )
        for j in range( 2, p ):
          for k in reversed( range( j, p ) ):
            alpha = 1. / min( 2+k-j, n-i+1 )
            ex[i-1,:,k] = alpha * ex[i-1,:,k] + (1-alpha) * ex[i-1,:,k-1]
          ex[i,-j-1:-1,-j-1] = ex[i-1,-j:,-1]
      extractions = util.reshape( extractions[:,_,:,_,:,_]
                                         * ex[_,:,_,:,_,:], 2, 2, 2 )
      if self.periodic == len( slices ):
        I = [p-2] * nelems
      else:
        I = range( n )
        if n < nelems:
          I[p-2:p-1] *= nelems - n + 1
      indices = indices[...,_] * n + I
      slices.append( [ slice(j,j+p) for j in range(nelems) ] )

    poly = element.PolyQuad( degree )
    stdelems = numpy.array( [ poly ] if all( p==2 for p in degree )
                       else [ element.ExtractionWrapper( poly, ex ) for ex in extractions ] )

    shape = [ n + p - 1 for n, p in zip( self.structure.shape, degree ) ]
    nodes_structure = numpy.arange( numpy.product(shape) ).reshape( shape )
    if self.periodic is None:
      dofcount = nodes_structure.size
    else:
      tmp = nodes_structure.swapaxes( 0, self.periodic )
      overlap = degree[self.periodic] - 1
      tmp[ -overlap: ] = tmp[ :overlap ]
      dofcount = tmp[ :-overlap ].size
    mapping = {}
    for item in numpy.broadcast( self.structure, *numpy.ix_(*slices) ):
      elem = item[0]
      S = item[1:]
      mapping[ elem ] = nodes_structure[S].ravel()
    shape = function.DofAxis( dofcount, mapping ),
    mapping = dict( ( elem, wrapper ) for elem, wrapper in numpy.broadcast( self.structure, stdelems[indices] ) )

    return function.Function( topodims=self.ndims, shape=shape, mapping=mapping )

  def linearfunc( self ):
    'linears'

    return self.splinefunc( degree=2 )

  def rectilinearfunc( self, gridnodes ):
    'rectilinear func'

    return function.RectilinearFunc( self, gridnodes )

  def refine( self, n ):
    'refine entire topology'

    if n == 1:
      return self

    N = n**self.ndims
    structure = numpy.array( [ elem.refined(n) for elem in self.structure.flat ] )
    structure = structure.reshape( self.structure.shape + (n,)*self.ndims )
    structure = structure.transpose( sum( [ ( i, self.ndims+i ) for i in range(self.ndims) ], () ) )
    structure = structure.reshape( [ self.structure.shape[i] * n for i in range(self.ndims) ] )

    return StructuredTopology( structure)

  def __str__( self ):
    'string representation'

    return '%s(%s)' % ( self.__class__.__name__, 'x'.join(map(str,self.structure.shape)) )

class UnstructuredTopology( Topology ):
  'externally defined topology'

  groups = ()

  def __init__( self, elements, ndims, namedfuncs={} ):
    'constructor'

    self.elements = elements
    self.namedfuncs = namedfuncs
    self.ndims = ndims

  def __len__( self ):
    'number of elements'

    return len( self.elements )

  def __getitem__( self, item ):
    'subtopology'

    return self.groups[item]

  def __iter__( self ):
    'iterate'

    return iter( self.elements )

  def splinefunc( self, degree ):
    'spline func'

    return self.namedfuncs[ 'spline%d' % degree ]

  def linearfunc( self ):
    'linear func'

    return self.splinefunc( degree=2 )

  def refine( self, n ):
    'refine entire topology'

    if n == 1:
      return self

    elements = []
    for elem in self:
      elements.extend( elem.refined(n) )
    return UnstructuredTopology( elements, ndims=self.ndims )

# vim:shiftwidth=2:foldmethod=indent:foldnestmax=2
