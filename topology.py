from . import element, function, util, numpy, parallel, matrix, _

class Topology( set ):
  'topology base class'

  def __add__( self, other ):
    'add topologies'

    if self is other:
      return self
    assert self.ndims == other.ndims
    return UnstructuredTopology( set(self) | set(other), ndims=self.ndims )

  def __getitem__( self, item ):
    'subtopology'

    items = ( self.groups[it] for it in item.split( ',' ) )
    return sum( items, items.next() )

  def integrate( self, funcs, ischeme=None, coords=None, title=True ):
    'integrate'

    if title is True:
      title = 'integrating %d elements sparse' % len(self)

    topo = self if not title else util.progressbar( self, title=title )

    single_arg = not isinstance(funcs,list)
    if single_arg:
      funcs = funcs,

    iweights = coords.iweights( self.ndims )
    integrands = []
    retvals = []
    for ifunc, func in enumerate( funcs ):
      if not isinstance(func,tuple):
        func = func,
      ndim = func[0].ndim
      assert all( f.ndim == ndim for f in func[1:] )
      func = filter( None, func ) # skip over zero integrands
      if ndim == 0:
        for f in func:
          integrands.append( function.Tuple([ ifunc, (), f, iweights ]) )
        A = numpy.array( 0, dtype=float )
      elif ndim == 1:
        length = 0
        for f in func:
          sh0, = f.shape
          length = max( length, int(sh0) )
          shape = slice(None) if isinstance(sh0,int) else sh0
          integrands.append( function.Tuple([ ifunc, shape, f, iweights ]) )
        A = numpy.zeros( length, dtype=float )
      elif ndim == 2:
        graph = []
        ncols = 0
        for f in func:
          sh0, sh1 = f.shape
          graph += [[]] * ( int(sh0) - len(graph) )
          ncols = max( ncols, int(sh1) )
          IJ = function.Tuple([ sh0, sh1 ])
          for elem in self:
            I, J = IJ( elem, None )
            for i in I:
              graph[i] = util.addsorted( graph[i], J, inplace=True )
          integrands.append( function.Tuple([ ifunc, IJ, f, iweights ]) )
        A = matrix.SparseMatrix( graph, ncols )
      else:
        raise NotImplementedError, 'ndim=%d' % func.ndim
      retvals.append( A )

    idata = function.Tuple( integrands )
    for elem in topo:
      for ifunc, index, data, w in idata( elem, ischeme ):
        retvals[ifunc][index] += util.contract( data.T, w ).T

    if single_arg:
      retvals, = retvals
    return retvals

  def projection( self, fun, onto, coords, **kwargs ):
    'project and return as function'

    weights = self.project( fun, onto, coords, **kwargs )
    return onto.dot( weights )

  def project( self, fun, onto, coords, tol=0, ischeme='gauss8', title=True, droptol=1e-8, exact_boundaries=False, constrain=None, verify=None ):
    'L2 projection of function onto function space'

    if exact_boundaries:
      assert constrain is None
      constrain = self.boundary.project( fun, onto, coords, ischeme=ischeme, title=None, tol=tol, droptol=droptol )
    elif constrain is None:
      constrain = util.NanVec( onto.shape[0] )
    else:
      assert isinstance( constrain, util.NanVec )
      assert constrain.shape == onto.shape[:1]

    if len( onto.shape ) == 1:
      Afun = onto[:,_] * onto[_,:]
      bfun = onto * fun
    elif len( onto.shape ) == 2:
      Afun = ( onto[:,_,:] * onto[_,:,:] ).sum( 2 )
      bfun = onto * fun
      if bfun:
        bfun = bfun.sum( 1 )
    else:
      raise Exception
    A, b = self.integrate( [Afun,bfun], coords=coords, ischeme=ischeme, title=title )
    supp = A.rowsupp( droptol )
    if verify is not None:
      assert (~supp).sum() == verify, 'number of constraints does not meet expectation'
    constrain[supp] = 0
    if numpy.all( b == 0 ):
      u = constrain | 0
    else:
      u = A.solve( b, constrain, tol=tol, title='projecting', symmetric=True )
    u[supp] = numpy.nan
    return u.view( util.NanVec )

  def trim( self, levelset, maxrefine, lscheme='bezier3' ):
    'create new domain based on levelset'

    newelems = []
    for elem in util.progressbar( self, title='selecting/refining elements' ):
      elempool = [ elem ]
      for level in range( maxrefine ):
        nextelempool = []
        for elem in elempool:
          inside = levelset( elem, lscheme ) > 0
          if inside.all():
            newelems.append( elem )
          elif inside.any():
            nextelempool.extend( elem.refined(2) )
        elempool = nextelempool
      # TODO select >50% overlapping elements from elempool
      newelems.extend( elempool )
    return UnstructuredTopology( newelems, ndims=self.ndims )

class StructuredTopology( Topology ):
  'structured topology'

  def __init__( self, structure, periodic=() ):
    'constructor'

    structure = numpy.asarray(structure)
    self.ndims = structure.ndim
    self.structure = structure
    self.periodic = tuple(periodic)
    self.groups = {}

    Topology.__init__( self, structure.flat )

  def make_periodic( self, periodic ):
    'add periodicity'

    return StructuredTopology( self.structure, periodic=periodic )

  def __len__( self ):
    'number of elements'

    return self.structure.size

  def __iter__( self ):
    'iterate'

    return self.structure.flat

  def __getitem__( self, item ):
    'subtopology'

    if isinstance( item, str ):
      return Topology.__getitem__( self, item )
    return StructuredTopology( self.structure[item] )

  @util.cacheprop
  def boundary( self ):
    'boundary'

    shape = numpy.asarray( self.structure.shape ) + 1
    nodes = numpy.arange( numpy.product(shape) ).reshape( shape )

    boundaries = []
    for iedge in range( 2 * self.ndims ):
      idim = iedge // 2
      iside = iedge % 2
      if self.ndims > 1:
        s = ( slice(None,None,1-2*iside), ) * idim \
          + ( -iside, ) \
          + ( slice(None,None,2*iside-1), ) * (self.ndims-idim-1)
        # TODO: check that this is correct for all dimensions; should match conventions in elem.edge
        belems = numpy.frompyfunc( lambda elem: elem.edge( iedge ), 1, 1 )( self.structure[s] )
      else:
        belems = numpy.array( self.structure[-iside].edge( iedge ) )
      boundaries.append( StructuredTopology( belems ) )

    if self.ndims == 2:
      structure = numpy.concatenate([ boundaries[i].structure for i in [0,2,1,3] ])
      topo = StructuredTopology( structure, periodic=[0] )
    else:
      allbelems = [ belem for boundary in boundaries for belem in boundary.structure.flat ]
      topo = UnstructuredTopology( allbelems, ndims=self.ndims-1 )

    topo.groups = dict( zip( ( 'left', 'right', 'bottom', 'top', 'front', 'back' ), boundaries ) )
    return topo

  @util.cachefunc
  def splinefunc( self, degree, neumann=() ):
    'spline from nodes'

    if isinstance( degree, int ):
      degree = ( degree, ) * self.ndims

    nodes_structure = numpy.array( 0 )
    dofcount = 1
    slices = []

    for idim in range( self.ndims ):
      periodic = idim in self.periodic
      n = self.structure.shape[idim]
      p = degree[idim]

      neumann_i = (idim*2 in neumann and 1) | (idim*2+1 in neumann and 2)
      stdelems_i = element.PolyLine.spline( degree=p, nelems=n, periodic=periodic, neumann=neumann_i )
      stdelems = stdelems[...,_] * stdelems_i if idim else stdelems_i

      nd = n + p - 1
      numbers = numpy.arange( nd )
      if periodic:
        overlap = p - 1
        numbers[ -overlap: ] = numbers[ :overlap ]
        nd -= overlap
      nodes_structure = nodes_structure[...,_] * nd + numbers
      dofcount *= nd
      slices.append( [ slice(i,i+p) for i in range(n) ] )

    indexmap = {}
    for item in numpy.broadcast( self.structure, *numpy.ix_(*slices) ):
      elem = item[0]
      S = item[1:]
      indexmap[ elem ] = nodes_structure[S].ravel()

    shape = function.DofAxis( dofcount, indexmap ),
    funcmap = dict( numpy.broadcast( self.structure, stdelems ) )

    return function.Function( shape=shape, mapping=funcmap )

  def linearfunc( self ):
    'linears'

    return self.splinefunc( degree=2 )

  def rectilinearfunc( self, gridnodes ):
    'rectilinear func'

    assert len( gridnodes ) == self.ndims
    nodes_structure = numpy.empty( map( len, gridnodes ) + [self.ndims] )
    for idim, inodes in enumerate( gridnodes ):
      shape = [1,] * self.ndims
      shape[idim] = -1
      nodes_structure[...,idim] = numpy.asarray( inodes ).reshape( shape )
    return self.linearfunc().dot( nodes_structure.reshape( -1, self.ndims ) )

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

  def manifold( self, xi0, xi1 ):
    'create lower dimensional manifold in parent'

    assert self.ndims == 2
    scale = numpy.array( self.structure.shape )
    i0, j0 = numpy.asarray(xi0) * scale
    i1, j1 = numpy.asarray(xi1) * scale
    # j = A + B * i
    # j0 = A + B * i0
    # j1 = A + B * i1
    # B = (j1-j0) / float(i1-i0)
    # A = (j0*i1-j1*i0) / float(i1-i0)
    Ia = numpy.arange( int(i0), int(i1) ) + 1
    Ja = ( j0*i1 - j1*i0 + (j1-j0) * Ia ) / float(i1-i0)
    # i = C + D * j
    # i0 = C + D * j0
    # i1 = C + D * j1
    # D = (i1-i0) / float(j1-j0)
    # C = (i0*j1-i1*j0) / float(j1-j0)
    Jb = numpy.arange( int(j0), int(j1) ) + 1
    Ib = ( i0*j1 - i1*j0 + (i1-i0) * Jb ) / float(j1-j0)

    points = numpy.array( sorted( [(i0,j0),(i1,j1)] + zip(Ia,Ja) + zip(Ib,Jb) ) )
    keep = numpy.hstack( [ ( numpy.diff( points, axis=0 )**2 ).sum( axis=1 ) > 1e-9, [True] ] )
    points = points[keep]

    offsets = points - points.astype(int)
    transforms = numpy.diff( points, axis=0 )
    n, m = ( points[:-1] + .5 * transforms ).astype( int ).T
    pelems = self.structure[ n, m ]

    structure = []
    for pelem, offset, transform in zip( pelems, offsets, transforms ):
      trans = element.AffineTransformation( offset=offset, transform=transform[:,_] )
      elem = element.QuadElement( ndims=1, parent=(pelem,trans) )
      structure.append( elem )

    topo = StructuredTopology( numpy.asarray(structure) )

    weights = numpy.sqrt( ( ( points - points[0] )**2 ).sum( axis=1 ) / ( (i1-i0)**2 + (j1-j0)**2 ) )
    coords = topo.splinefunc( degree=2 ).dot( weights )

    return topo, coords

  def manifold2d( self, C0, C1, C2 ):
    'manifold 2d'

    np = 100
    n = numpy.arange( .5, np ) / np
    i = n[_,:,_]
    j = n[_,_,:]

    xyz = C0[:,_,_] + i * C1 + j * C2
    nxyz = int( xyz )
    fxyz = xyz - nxyz

    while len(n):
      ielem = nxyz[:,0]
      select = ( nxyz == ielem ).all( axis=0 )
      pelem = self.structure[ ielem ]

      trans = element.AffineTransformation( offset=offset, transform=transform[:,_] )
      elem = element.QuadElement( ndims=1, parent=(pelem,trans) )
      structure.append( elem )
    
class UnstructuredTopology( Topology ):
  'externally defined topology'

  groups = ()

  def __init__( self, elements, ndims, namedfuncs={} ):
    'constructor'

    self.namedfuncs = namedfuncs
    self.ndims = ndims

    Topology.__init__( self, elements )

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
