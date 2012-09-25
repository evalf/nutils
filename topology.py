from . import element, function, util, numpy, parallel, _

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

  def integrate( self, func, ischeme=None, coords=None, title=True, dense=None, shape=None ):
    'integrate'

    funcs = func if isinstance( func, (list,tuple) ) else [func]
    shapes = [ map(int,f.shape) for f in funcs ]

    if self.ndims == 0:
      # simple point evaluation
      iweights = 1
      ischeme = 'none'
    else:
      assert coords, 'must specify coords'
      assert ischeme, 'must specify integration scheme'
      iweights = coords.iweights( self.ndims )
    idata = function.Tuple([ function.Tuple([ f, f.indices(), iweights ]) for f in funcs ])

    if not dense and ( isinstance( func, tuple ) or isinstance( func, function.ArrayFunc ) and len( func.shape ) == 2 ):
      # quickly implemented single array for now, needs to be extended for
      # multiple inputs. requires thinking of separating types for separate
      # arguments.

      topo = self if not title \
        else util.progressbar( self, title='integrating %d elements sparse' % len(self) if title is True else title )

      import scipy.sparse

      #assert all( sh == shapes[0] for sh in shapes[1:] )
      if shape:
        assert len(shape) == 2
        assert all( n1 <= shape[0] and n2 <= shape[1] for (n1,n2) in shapes )
      else:
        shape = max( sh[0] for sh in shapes ), max( sh[1] for sh in shapes )

      indices = []
      values = []
      length = 0
      for elem in topo:
        for data, (I,J), w in idata( elem, ischeme ):
          evalues = util.contract( data.T, w ).T # TEMP
          eindices = numpy.empty( (2,) + evalues.shape )
          eindices[0] = I[:,_]
          eindices[1] = J[_,:]
          values.append( evalues.ravel() )
          indices.append( eindices.reshape(2,-1) )
          length += evalues.size

      v = numpy.empty( length, dtype=float )
      ij = numpy.empty( [2,length], dtype=float )
      n0 = 0
      for val, ind in zip( values, indices ):
        n1 = n0 + val.size
        v[n0:n1] = val
        ij[:,n0:n1] = ind
        n0 = n1
      assert n0 == length
      A = scipy.sparse.csr_matrix( (v,ij), shape=shape )

    else:

      topo = self if not title \
        else util.progressbar( self, title='integrating %d elems dense nproc=%d' % ( len(self), parallel.nprocs ) if title is True else title )

      A = map( parallel.shzeros, shapes )

      lock = parallel.Lock()
      for elem in parallel.pariter( topo ):
        for i, (data,index,w) in enumerate( idata( elem, ischeme ) ):
        #for i, (data,index,w) in enumerate( idata.eval( elem, ischeme ) ):

          # BEGIN
          # This is of course a big waste but the whole numpy indexing way is
          # ridiculous anyway. Will go away soon.
          where, = numpy.where( [ isinstance(ni,numpy.ndarray) for ni in index ] )
          if where.size > 1:
            index = list(index)
            for ni in where:
              I = [_] * where.size
              I[ni] = slice(None)
              index[ni] = index[ni][ tuple(I) ]
            index = tuple(index)
          # END

          emat = util.contract( data.T, w ).T # TEMP
          with lock:
            A[i][index] += emat

      if funcs is not func: # unpack single function
        A, = A

    return A

  def projection( self, fun, onto, coords, **kwargs ):
    'project and return as function'

    weights = self.project( fun, onto, coords, **kwargs )
    return onto.dot( weights )

  def project( self, fun, onto, coords, ischeme='gauss8', title=True, tol=1e-8, exact_boundaries=False, constrain=None ):
    'L2 projection of function onto function space'

    if exact_boundaries:
      assert constrain is None
      constrain = self.boundary.project( fun, onto, coords, ischeme=ischeme, title=None, tol=tol )
    elif constrain is None:
      constrain = util.NanVec( onto.shape[0] )
    else:
      assert isinstance( constrain, util.NanVec )
      assert constrain.shape == onto.shape[:1]

    if not isinstance( fun, function.Evaluable ):
      if callable( fun ):
        fun = function.UFunc( coords, fun )

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

    zero = ( numpy.abs( A ) < tol ).all( axis=0 )
    constrain[zero] = 0
    if bfun == 0:
      u = constrain | 0
    else:
      u = util.solve( A, b, constrain )
    u[zero] = numpy.nan
    return u.view( util.NanVec )

  def trim( self, levelset, maxrefine ):
    'create new domain based on levelset'

    newelems = []
    for elem in util.progressbar( self, title='selecting/refining elements' ):
      elempool = [ elem ]
      for level in range( maxrefine ):
        nextelempool = []
        for elem in elempool:
          inside = levelset( elem, 'bezier3' ) > 0
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

    extractions = numpy.ones( (1,1,1), dtype=float )
    indices = numpy.array( 0 )
    slices = []
    for idim in range( self.ndims ):
      p = degree[idim]
      nelems = self.structure.shape[idim]
      n = (2*(p-1)-1) if idim in self.periodic else min( nelems, 2*(p-1)-1 )
      ex = numpy.empty(( n, p, p ))
      ex[0] = numpy.eye( p )
      for i in range( 1, n ):
        ex[i] = numpy.eye( p )
        for j in range( 2, p ):
          for k in reversed( range( j, p ) ):
            alpha = 1. / min( 2+k-j, n-i+1 )
            ex[i-1,:,k] = alpha * ex[i-1,:,k] + (1-alpha) * ex[i-1,:,k-1]
          ex[i,-j-1:-1,-j-1] = ex[i-1,-j:,-1]

      if idim * 2 in neumann:
        ex[0,1,:] += ex[0,0,:]
      if idim * 2 + 1 in neumann:
        ex[-1,-2,:] += ex[-1,-1,:]

      extractions = util.reshape( extractions[:,_,:,_,:,_]
                                         * ex[_,:,_,:,_,:], 2, 2, 2 )
      if idim in self.periodic:
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

#   shape = [ n + p - 1 for n, p in zip( self.structure.shape, degree ) ]
#   nodes_structure = numpy.arange( numpy.product(shape) ).reshape( shape )

#   if not self.periodic:
#     dofcount = nodes_structure.size
#   else:
#     tmp = nodes_structure.swapaxes( 0, self.periodic )
#     overlap = degree[self.periodic] - 1
#     tmp[ -overlap: ] = tmp[ :overlap ]
#     dofcount = tmp[ :-overlap ].size

#   shape = numpy.asarray( nodes_structure.shape )
#   for idim in self.periodic:
#     tmp = nodes_structure.swapaxes( 0, idim )
#     overlap = degree[idim] - 1
#     tmp[ -overlap: ] = tmp[ :overlap ]
#     shape[idim] -= overlap
#   dofcount = int( numpy.product( shape ) )
#   print dofcount

    nodes_structure = numpy.array( 0 )
    dofcount = 1
    for idim in range( self.ndims ):
      n = self.structure.shape[idim]
      p = degree[idim]
      nd = n + p - 1
      numbers = numpy.arange( nd )
      if idim in self.periodic:
        overlap = p - 1
        numbers[ -overlap: ] = numbers[ :overlap ]
      else:
        overlap = 0
      nodes_structure = nodes_structure[...,_] * nd + numbers
      dofcount *= nd - overlap

    mapping = {}
    for item in numpy.broadcast( self.structure, *numpy.ix_(*slices) ):
      elem = item[0]
      S = item[1:]
      mapping[ elem ] = nodes_structure[S].ravel()
    shape = function.DofAxis( dofcount, mapping ),
    mapping = dict( ( elem, wrapper ) for elem, wrapper in numpy.broadcast( self.structure, stdelems[indices] ) )

    return function.Function( shape=shape, mapping=mapping )

  def linearfunc( self, periodic=None ):
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
