from . import element, function, util, cache, parallel, matrix, log, numeric, prop, transform, pointset, _
import warnings, itertools

class ElemMap( dict ):
  'dictionary-like element mapping'

  def __init__( self, mapping, ndims ):
    'constructor'

    self.ndims = ndims
    dict.__init__( self, mapping )

  def __eq__( self, other ):
    'test equal'

    return self is other

  def __str__( self ):
    'string representation'

    return 'ElemMap(#%d,%dD)' % ( len(self), self.ndims )


class Topology( object ):

  def __init__( self, ndims, elements, groups=None ):
    self.ndims = ndims
    self.elements = numeric.empty( len(elements), dtype=object )
    self.elements[:] = elements
    self.elements.sort()
    self.groups = groups or {}

  def __getitem__( self, item ):
    if isinstance( item, str ):
      return eval( item.replace(',','|'), self.groups )
    return self.elements[ item ]

  def _set( self, other, op ):
    assert self.ndims == other.ndims
    return Topology( self.ndims, op( self.elements, other.elements ) )
    
  def __and__( self, other ): return self._set( other, numeric.intersect1d )
  def __or__ ( self, other ): return self._set( other, numeric.union1d     )
  def __xor__( self, other ): return self._set( other, numeric.setxor1d    )
  def __sub__( self, other ): return self._set( other, numeric.setdiff1d   )

  def __iter__( self ):
    return iter( self.elements )

  def __len__( self ):
    return len( self.elements )

  def build_graph( self, func ):
    'get matrix sparsity'

    log.context( 'graph' )

    nrows, ncols = func.shape
    graph = [ [] for irow in range(nrows) ]
    IJ = function.Tuple([ function.Tuple(ind) for f, ind in function.blocks( func ) ]).compiled()

    for elem in log.iterate('elem',self):
      for I, J in IJ.eval( elem, None ):
        for i in I:
          graph[i].append(J)

    for irow, g in log.iterate( 'dof', enumerate(graph), nrows ):
      # release memory as we go
      if g: graph[irow] = numeric.unique( numeric.concatenate( g ) )

    return graph

  def integrate( self, funcs, ischeme, geometry=None, iweights=None, force_dense=False, title='integrating' ):
    'integrate'

    log.context( title )

    single_arg = not isinstance(funcs,(list,tuple))
    if single_arg:
      funcs = funcs,

    if iweights is None:
      assert geometry is not None, 'conflicting arguments geometry and iweights'
      iweights = function.iwscale( geometry, self.ndims ) * function.IWeights( self.ndims )
    else:
      assert geometry is None, 'conflicting arguments geometry and iweights'
    assert iweights.ndim == 0

    integrands = []
    retvals = []
    for ifunc, func in enumerate( funcs ):
      func = function.asarray( func )
      lock = parallel.Lock()
      if function._isfunc( func ):
        array = parallel.shzeros( func.shape, dtype=float ) if func.ndim != 2 \
           else matrix.DenseMatrix( func.shape ) if force_dense \
           else matrix.SparseMatrix( self.build_graph(func), func.shape[1] )
        for f, ind in function.blocks( func ):
          integrands.append( function.Tuple([ ifunc, lock, function.Tuple(ind), function.elemint( f, iweights ) ]) )
      else:
        array = parallel.shzeros( func.shape, dtype=float )
        if not function._iszero( func ):
          integrands.append( function.Tuple([ ifunc, lock, (), function.elemint( func, iweights ) ]) )
      retvals.append( array )
    idata = function.Tuple( integrands ).compiled()

    points = pointset.aspointset( ischeme )
    for elem in parallel.pariter( log.iterate('elem',self) ):
      for ifunc, lock, index, data in idata.eval( elem, points ):
        retval = retvals[ifunc]
        with lock:
          retval[index] += data

    log.info( 'cache', idata.cache.summary() )
    log.info( 'created', ', '.join( '%s(%s)' % ( retval.__class__.__name__, ','.join(map(str,retval.shape)) ) for retval in retvals ) )
    if single_arg:
      retvals, = retvals

    return retvals

  def projection( self, fun, onto, geometry, **kwargs ):
    'project and return as function'

    weights = self.project( fun, onto, geometry, **kwargs )
    return onto.dot( weights )

  def project( self, fun, onto, geometry, tol=0, ischeme=None, title='projecting', droptol=1e-8, exact_boundaries=False, constrain=None, verify=None, maxiter=0, ptype='lsqr' ):
    'L2 projection of function onto function space'

    log.context( title + ' [%s]' % ptype )

    points = pointset.aspointset( ischeme )
    if exact_boundaries:
      assert constrain is None
      constrain = self.boundary.project( fun, onto, geometry, title='boundaries', ischeme=points, tol=tol, droptol=droptol, ptype=ptype )
    elif constrain is None:
      constrain = util.NanVec( onto.shape[0] )
    else:
      assert isinstance( constrain, util.NanVec )
      assert constrain.shape == onto.shape[:1]

    if ptype == 'lsqr':
      assert points is not None, 'please specify an integration scheme for lsqr-projection'
      if len( onto.shape ) == 1:
        Afun = function.outer( onto )
        bfun = onto * fun
      elif len( onto.shape ) == 2:
        Afun = function.outer( onto ).sum( 2 )
        bfun = function.sum( onto * fun )
      else:
        raise Exception
      A, b = self.integrate( [Afun,bfun], geometry=geometry, ischeme=points, title='building system' )
      N = A.rowsupp(droptol)
      if numeric.equal( b, 0 ).all():
        constrain[~constrain.where&N] = 0
      else:
        solvecons = constrain.copy()
        solvecons[~(constrain.where|N)] = 0
        u = A.solve( b, solvecons, tol=tol, symmetric=True, maxiter=maxiter )
        constrain[N] = u[N]

    elif ptype == 'convolute':
      assert points is not None, 'please specify an integration scheme for convolute-projection'
      if len( onto.shape ) == 1:
        ufun = onto * fun
        afun = onto
      elif len( onto.shape ) == 2:
        ufun = function.sum( onto * fun )
        afun = function.norm2( onto )
      else:
        raise Exception
      u, scale = self.integrate( [ ufun, afun ], geometry=geometry, ischeme=points )
      N = ~constrain.where & ( scale > droptol )
      constrain[N] = u[N] / scale[N]

    elif ptype == 'nodal':

      ## data = function.Tuple([ fun, onto ])
      ## F = W = 0
      ## for elem in self:
      ##   f, w = data( elem, 'bezier2' )
      ##   W += w.sum( axis=-1 ).sum( axis=0 )
      ##   F += numeric.contract( f[:,_,:], w, axis=[0,2] )
      ## I = (W!=0)

      F = numeric.zeros( onto.shape[0] )
      W = numeric.zeros( onto.shape[0] )
      I = numeric.zeros( onto.shape[0], dtype=bool )
      fun = function.asarray( fun )
      data = function.Tuple( function.Tuple([ fun, f, function.Tuple(ind) ]) for f, ind in function.blocks( onto ) )
      for elem in self:
        for f, w, ind in data( elem, 'bezier2' ):
          w = w.swapaxes(0,1) # -> dof axis, point axis, ...
          wf = w * f[ (slice(None),)+numeric.ix_(*ind[1:]) ]
          W[ind[0]] += w.reshape(w.shape[0],-1).sum(1)
          F[ind[0]] += wf.reshape(w.shape[0],-1).sum(1)
          I[ind[0]] = True

      I[constrain.where] = False
      constrain[I] = F[I] / W[I]

    else:
      raise Exception, 'invalid projection %r' % ptype

    errfun2 = ( onto.dot( constrain | 0 ) - fun )**2
    if errfun2.ndim == 1:
      errfun2 = errfun2.sum()
    error2, area = self.integrate( [ errfun2, 1 ], geometry=geometry, ischeme=points or 'gauss2' )
    avg_error = numeric.sqrt(error2) / area

    numcons = constrain.where.sum()
    if verify is not None:
      assert numcons == verify, 'number of constraints does not meet expectation: %d != %d' % ( numcons, verify )

    log.info( 'constrained %d/%d dofs, error %.2e/area' % ( numcons, constrain.size, avg_error ) )

    return constrain

  def elem_eval( self, funcs, ischeme, separate=False, title='evaluating' ):
    'element-wise evaluation'

    log.context( title )

    single_arg = not isinstance(funcs,(tuple,list))
    if single_arg:
      funcs = funcs,

    slices = []
    pointshape = function.PointShape().compiled()
    npoints = 0
    separators = []
    points = pointset.aspointset( ischeme )
    for elem in log.iterate('elem',self):
      np, = pointshape.eval( elem, points )
      slices.append( slice(npoints,npoints+np) )
      npoints += np
      if separate:
        separators.append( npoints )
        npoints += 1
    if separate:
      separators = numeric.array( separators[:-1], dtype=int )
      npoints -= 1

    retvals = []
    idata = []
    for ifunc, func in enumerate( funcs ):
      func = function.asarray( func )
      retval = parallel.shzeros( (npoints,)+func.shape, dtype=func.dtype )
      if separate:
        retval[separators] = numeric.nan
      if function._isfunc( func ):
        for f, ind in function.blocks( func ):
          idata.append( function.Tuple( [ ifunc, function.Tuple(ind), f ] ) )
      else:
        idata.append( function.Tuple( [ ifunc, (), func ] ) )
      retvals.append( retval )
    idata = function.Tuple( idata ).compiled()

    for ielem, elem in parallel.pariter( enumerate( log.iterate('elem',self) ) ):
      s = slices[ielem],
      for ifunc, index, data in idata.eval( elem, points ):
        retvals[ifunc][s+index] = data

    log.info( 'cache', idata.cache.summary() )
    log.info( 'created', ', '.join( '%s(%s)' % ( retval.__class__.__name__, ','.join(map(str,retval.shape)) ) for retval in retvals ) )
    if single_arg:
      retvals, = retvals

    return retvals

  def trim( self, levelset, maxrefine=0, minrefine=0, title='trimming' ):
    'trim element along levelset'

    levelset = function.ascompiled( levelset )
    pos, neg = [], []
    for elem in log.iterate( title, self ):
      p, n = elem[-1].trim( levelset=(elem[:-1]+(levelset,)), maxrefine=maxrefine, minrefine=minrefine )
      if p: pos.append( elem[:-1]+(p,) )
      if n: neg.append( elem[:-1]+(n,) )
    return Topology( ndims=self.ndims, elements=pos ), Topology( ndims=self.ndims, elements=neg )

  @cache.property
  def simplex( self ):
    simplices = [ elem[:-1] + simplex for elem in self for simplex in elem[-1].simplices ]
    return Topology( ndims=self.ndims, elements=simplices )

  def get_simplices( self ):
    warnings.warn( '.getsimplices() is replaced by .simplex', DeprecationWarning )
    return self.simplex

  @cache.property
  def refined( self ):
    children = [ elem[:-1] + child for elem in self for child in elem[-1].children ]
    groups = { key: topo.refined for key, topo in self.groups.items() }
    return Topology( ndims=self.ndims, elements=children, groups=groups )

class StructuredTopology( Topology ):

  def __init__( self, structure, periodic=(), groups=None ):
    self.structure = numeric.asarray(structure)
    self.periodic = tuple(periodic)
    Topology.__init__( self, ndims=self.structure.ndim, elements=self.structure.ravel(), groups=groups )

  @cache.property
  def boundary( self ):
    'boundary'

    shape = numeric.asarray( self.structure.shape ) + 1
    vertices = numeric.arange( numeric.product(shape) ).reshape( shape )

    boundaries = []
    for iedge in range( 2 * self.ndims ):
      idim = iedge // 2
      iside = iedge % 2
      if self.ndims > 1:
        s = [ slice(None,None,-1) ] * idim \
          + [ -iside, ] \
          + [ slice(None,None,1) ] * (self.ndims-idim-1)
        if not iside: # TODO: check that this is correct for all dimensions; should match conventions in elem.edge
          s[idim-1] = slice(None,None,1 if idim else -1)
        elems = self.structure[tuple(s)]
        belems = numeric.empty( elems.shape, dtype=object )
        for index, elem in numeric.enumerate_nd( elems ):
          belems[ index ] = elem[:-1] + elem[-1].edges[iedge]
      else:
        belems = numeric.array( self.structure[-iside].edge( 1-iedge ) )
      periodic = [ d - (d>idim) for d in self.periodic if d != idim ] # TODO check that dimensions are correct for ndim > 2
      boundaries.append( StructuredTopology( belems, periodic=periodic ) )
    groups = dict( zip( ( 'left', 'right', 'bottom', 'top', 'front', 'back' ), boundaries ) )

    if self.ndims == 2:
      structure = numeric.concatenate([ boundaries[i].structure for i in [0,2,1,3] ])
      topo = StructuredTopology( structure, periodic=[0], groups=groups )
    else:
      allbelems = [ belem for boundary in boundaries for belem in boundary.structure.flat if belem is not None ]
      topo = Topology( elements=allbelems, ndims=self.ndims-1, groups=groups )

    return topo

  def splinefunc( self, degree, neumann=(), periodic=None, closed=False, removedofs=None ):
    'spline from vertices'

    if periodic is None:
      periodic = self.periodic

    if isinstance( degree, int ):
      degree = ( degree, ) * self.ndims

    if removedofs == None:
      removedofs = [None] * self.ndims
    else:
      assert len(removedofs) == self.ndims

    vertex_structure = numeric.array( 0 )
    dofcount = 1
    slices = []

    for idim in range( self.ndims ):
      periodic_i = idim in periodic
      n = self.structure.shape[idim]
      p = degree[idim]

      if closed == False:
        neumann_i = (idim*2 in neumann and 1) | (idim*2+1 in neumann and 2)
        stdelems_i = element.PolyLine.spline( degree=p, nelems=n, periodic=periodic_i, neumann=neumann_i )
      elif closed == True:
        assert periodic==(), 'Periodic option not allowed for closed spline'
        assert neumann ==(), 'Neumann option not allowed for closed spline'
        stdelems_i = element.PolyLine.spline( degree=p, nelems=n, periodic=True )

      stdelems = stdelems[...,_] * stdelems_i if idim else stdelems_i

      nd = n + p
      numbers = numeric.arange( nd )
      if periodic_i and p > 0:
        overlap = p
        numbers[ -overlap: ] = numbers[ :overlap ]
        nd -= overlap
      remove = removedofs[idim]
      if remove is None:
        vertex_structure = vertex_structure[...,_] * nd + numbers
      else:
        mask = numeric.zeros( nd, dtype=bool )
        mask[numeric.array(remove)] = True
        nd -= mask.sum()
        numbers -= mask.cumsum()
        vertex_structure = vertex_structure[...,_] * nd + numbers
        vertex_structure[...,mask] = -1
      dofcount *= nd
      slices.append( [ slice(i,i+p+1) for i in range(n) ] )

    dofmap = {}
    funcmap = {}
    hasnone = False
    for item in numeric.broadcast( self.structure, stdelems, *numeric.ix_(*slices) ):
      elem = item[0][:-1]
      std = item[1]
      if elem is None:
        hasnone = True
      else:
        S = item[2:]
        dofs = vertex_structure[S].ravel()
        mask = numeric.greater_equal( dofs, 0 )
        if mask.all():
          dofmap[ elem ] = dofs
          funcmap[elem] = std
        elif mask.any():
          dofmap[ elem ] = dofs[mask]
          funcmap[elem] = std, mask

    if hasnone:
      touched = numeric.zeros( dofcount, dtype=bool )
      for dofs in dofmap.itervalues():
        touched[ dofs ] = True
      renumber = touched.cumsum()
      dofcount = int(renumber[-1])
      dofmap = dict( ( elem, renumber[dofs]-1 ) for elem, dofs in dofmap.iteritems() )

    return function.function( funcmap, dofmap, dofcount, self.ndims )



## OLD

class _Topology( object ):
  'topology base class'

  def __init__( self, ndims ):
    'constructor'

    self.ndims = ndims

  def refined_by( self, refine ):
    'create refined space by refining dofs in existing one'

    refine = list( refine )
    refined = [] # all elements of refined topology, max 1 level finer than current
    for elem in self:
      if elem in refine:
        refine.remove( elem )
        refined.extend( elem.children )
      else:
        refined.append( elem )
        # only for argument checking: remove parents from refine
        pelem = elem
        while pelem.parent:
          pelem, trans = pelem.parent
          if pelem in refine:
            refine.remove( pelem )

    assert not refine, 'not all refinement elements were found: %s' % '\n '.join( str(e) for e in refine )
    return HierarchicalTopology( self, refined )

  def stdfunc( self, degree ):
    'spline from vertices'

    assert degree == 1 # for now!
    dofmap = { n: i for i, n in enumerate( sorted( set( n for elem in self for n in elem.vertices ) ) ) }
    fmap = { elem: elem.simplex.stdfunc(degree) for elem in self }
    nmap = { elem: numeric.array([ dofmap[n] for n in elem.vertices ]) for elem in self }
    return function.function( fmap=fmap, nmap=nmap, ndofs=len(dofmap), ndims=2 )

  def __add__( self, other ):
    'add topologies'

    if self is other:
      return self
    assert self.ndims == other.ndims
    return UnstructuredTopology( set(self) | set(other), ndims=self.ndims )

  def __sub__( self, other ):
    'add topologies'

    if self is other:
      return self
    assert self.ndims == other.ndims
    return UnstructuredTopology( set(self) - set(other), ndims=self.ndims )

  def __mul__( self, other ):
    'element products'

    elems = util.Product( self, other )
    return UnstructuredTopology( elems, ndims=self.ndims+other.ndims )

  def __getitem__( self, item ):
    'subtopology'

    items = ( self.groups[it] for it in item.split( ',' ) )
    return sum( items, items.next() )

  def elem_eval( self, funcs, ischeme, separate=False, title='evaluating' ):
    'element-wise evaluation'

    log.context( title )

    single_arg = not isinstance(funcs,(tuple,list))
    if single_arg:
      funcs = funcs,

    slices = []
    pointshape = function.PointShape().compiled()
    npoints = 0
    separators = []
    for elem in log.iterate('elem',self):
      np, = pointshape.eval( elem, ischeme )
      slices.append( slice(npoints,npoints+np) )
      npoints += np
      if separate:
        separators.append( npoints )
        npoints += 1
    if separate:
      separators = numeric.array( separators[:-1], dtype=int )
      npoints -= 1

    retvals = []
    idata = []
    for ifunc, func in enumerate( funcs ):
      func = function.asarray( func )
      retval = parallel.shzeros( (npoints,)+func.shape, dtype=func.dtype )
      if separate:
        retval[separators] = numeric.nan
      if function._isfunc( func ):
        for f, ind in function.blocks( func ):
          idata.append( function.Tuple( [ ifunc, function.Tuple(ind), f ] ) )
      else:
        idata.append( function.Tuple( [ ifunc, (), func ] ) )
      retvals.append( retval )
    idata = function.Tuple( idata ).compiled()

    for ielem, elem in parallel.pariter( enumerate( self ) ):
      s = slices[ielem],
      for ifunc, index, data in idata.eval( elem, ischeme ):
        retvals[ifunc][s+index] = data

    log.info( 'cache', idata.cache.summary() )
    log.info( 'created', ', '.join( '%s(%s)' % ( retval.__class__.__name__, ','.join(map(str,retval.shape)) ) for retval in retvals ) )
    if single_arg:
      retvals, = retvals

    return retvals

  def elem_mean( self, funcs, geometry, ischeme, title='computing mean values' ):
    'element-wise integration'

    log.context( title )

    single_arg = not isinstance(funcs,(tuple,list))
    if single_arg:
      funcs = funcs,

    retvals = []
    #iweights = geometry.iweights( self.ndims )
    iweights = function.iwscale( geometry, self.ndims ) * function.IWeights( self.ndims )
    idata = [ iweights ]
    for func in funcs:
      func = function.asarray( func )
      if not function._isfunc( func ):
        func = function.Const( func )
      assert all( isinstance(sh,int) for sh in func.shape )
      idata.append( function.elemint( func, iweights ) )
      retvals.append( numeric.empty( (len(self),)+func.shape ) )
    idata = function.Tuple( idata )

    for ielem, elem in enumerate( self ):
      area_data = idata( elem, ischeme )
      area = area_data[0].sum()
      for retval, data in zip( retvals, area_data[1:] ):
        retval[ielem] = data / area

    log.info( 'created', ', '.join( '%s(%s)' % ( retval.__class__.__name__, ','.join(map(str,retval.shape)) ) for retval in retvals ) )
    if single_arg:
      retvals, = retvals

    return retvals

  def grid_eval( self, funcs, geometry, C, title='grid-evaluating' ):
    'evaluate grid points'

    log.context( title )

    single_arg = not isinstance(funcs,(tuple,list))
    if single_arg:
      funcs = funcs,

    C = numeric.asarray( C )
    assert C.shape[0] == self.ndims
    shape = C.shape
    C = C.reshape( self.ndims, -1 )

    funcs = [ function.asarray(func) for func in funcs ]
    retvals = [ numeric.empty( C.shape[1:] + func.shape ) for func in funcs ]
    for retval in retvals:
      retval[:] = numeric.nan

    data = function.Tuple([ function.Tuple([ func, retval ]) for func, retval in zip( funcs, retvals ) ])

    for elem in log.iterate('elem',self):
      points, selection = geometry.find( elem, C.T )
      if selection is not None:
        for func, retval in data( elem, points ):
          retval[selection] = func

    retvals = [ retval.reshape( shape[1:] + func.shape ) for func, retval in zip( funcs, retvals ) ]
    log.info( 'created', ', '.join( '%s(%s)' % ( retval.__class__.__name__, ','.join(map(str,retval.shape)) ) for retval in retvals ) )
    if single_arg:
      retvals, = retvals

    return retvals

  def integrate_symm( self, funcs, ischeme, geometry=None, iweights=None, force_dense=False, title='integrating' ):
    'integrate a symmetric integrand on a product domain' # TODO: find a proper home for this

    log.context( title )

    single_arg = not isinstance(funcs,list)
    if single_arg:
      funcs = funcs,

    if iweights is None:
      assert geometry is not None, 'conflicting arguments geometry and iweights'
      iweights = function.iwscale( geometry, self.ndims ) * function.IWeights( self.ndims )
    else:
      assert geometry is None, 'conflicting arguments geometry and iweights'
    assert iweights.ndim == 0

    integrands = []
    retvals = []
    for ifunc, func in enumerate( funcs ):
      func = function.asarray( func )
      lock = parallel.Lock()
      if function._isfunc( func ):
        array = parallel.shzeros( func.shape, dtype=float ) if func.ndim != 2 \
           else matrix.DenseMatrix( func.shape ) if force_dense \
           else matrix.SparseMatrix( self.build_graph(func), func.shape[1] )
        for f, ind in function.blocks( func ):
          integrands.append( function.Tuple([ ifunc, lock, function.Tuple(ind), function.elemint( f, iweights ) ]) )
      else:
        array = parallel.shzeros( func.shape, dtype=float )
        if not function._iszero( func ):
          integrands.append( function.Tuple([ ifunc, lock, (), function.elemint( func, iweights ) ]) )
      retvals.append( array )
    idata = function.Tuple( integrands ).compiled()

    for elem in parallel.pariter( log.iterate('elem',self) ):
      assert isinstance( elem, element.ProductElement )
      compare_elem = cmp( elem.elem1, elem.elem2 )
      if compare_elem < 0:
        continue
      for ifunc, lock, index, data in idata.eval( elem, ischeme ):
        with lock:
          retvals[ifunc][index] += data
          if compare_elem > 0:
            retvals[ifunc][index[::-1]] += data.T

    log.info( 'cache', idata.cache.summary() )
    log.info( 'created', ', '.join( '%s(%s)' % ( retval.__class__.__name__, ','.join(map(str,retval.shape)) ) for retval in retvals ) )
    if single_arg:
      retvals, = retvals

    return retvals

  def refinedfunc( self, dofaxis, refine, degree, title='refining' ):
    'create refined space by refining dofs in existing one'

    warnings.warn( 'refinedfunc is replaced by refined_by + splinefunc; this function will be removed in future' % ischeme, DeprecationWarning )
    log.context( title )

    refine = set(refine) # make unique and equip with set operations
  
    # initialize
    topoelems = [] # non-overlapping 'top-level' elements, will make up the new domain
    parentelems = [] # all parents, grandparents etc of topoelems
    nrefine = 0 # number of nested topologies after refinement

    dofmap = dofaxis.dofmap
    topo = self
    while topo: # elements to examine in next level refinement
      nexttopo = []
      refined = set() # refined dofs in current refinement level
      for elem in log.iterate('elem',topo): # loop over remaining elements in refinement level 'nrefine'
        dofs = dofmap.get( elem ) # dof numbers for current funcsp object
        if dofs is not None: # elem is a top-level element
          supp = refine.intersection(dofs) # supported dofs that are tagged for refinement
          if supp: # elem supports dofs for refinement
            parentelems.append( elem ) # elem will become a parent
            topoelems.extend( filter(None,elem.children) ) # children will become top-level elements
            refined.update( supp ) # dofs will not be considered in following refinement levels
          else: # elem does not support dofs for refinement
            topoelems.append( elem ) # elem remains a top-level elemnt
        else: # elem is not a top-level element
          parentelems.append( elem ) # elem is a parent
          nexttopo.extend( filter(None,elem.children) ) # examine children in next iteration
      refine -= refined # discard dofs to prevent further consideration
      topo = nexttopo # prepare for next iteration
      nrefine += 1 # update refinement level
    assert not refine, 'unrefined leftover: %s' % refine
    if refined: # last considered level contained refinements
      nrefine += 1 # this raises the total level to nrefine + 1

    # initialize
    dofmap = {} # IEN mapping of new function object
    stdmap = {} # shape function mapping of new function object, plus boolean vector indicating which shapes to retain
    ndofs = 0 # total number of dofs of new function object
  
    topo = self # topology to examine in next level refinement
    for irefine in log.iterate( 'level', range(nrefine), showpct=False ):
  
      funcsp = topo.splinefunc( degree ) # shape functions for level irefine
      (func,(dofaxis,)), = function.blocks( funcsp ) # separate elem-local funcs and global placement index
  
      supported = numeric.ones( funcsp.shape[0], dtype=bool ) # True if dof is contained in topoelems or parentelems
      touchtopo = numeric.zeros( funcsp.shape[0], dtype=bool ) # True if dof touches at least one topoelem
      myelems = [] # all top-level or parent elements in level irefine
      for elem, idofs in log.iterate( 'element', dofaxis.dofmap.items() ):
        if elem in topoelems:
          touchtopo[idofs] = True
          myelems.append( elem )
        elif elem in parentelems:
          myelems.append( elem )
        else:
          supported[idofs] = False
  
      keep = numeric.logical_and( supported, touchtopo ) # THE refinement law
      if keep.all() and irefine == nrefine - 1:
        return topo, funcsp
  
      for elem in myelems: # loop over all top-level or parent elements in level irefine
        idofs = dofaxis.dofmap[elem] # local dof numbers
        mykeep = keep[idofs]
        std = func.stdmap[elem]
        assert isinstance(std,element.StdElem)
        if mykeep.all():
          stdmap[elem] = std # use all shapes from this level
        elif mykeep.any():
          stdmap[elem] = std, mykeep # use some shapes from this level
        newdofs = [ ndofs + keep[:idof].sum() for idof in idofs if keep[idof] ] # new dof numbers
        if elem not in self: # at lowest level
          pelem, transform = elem.parent
          newdofs.extend( dofmap[pelem] ) # add dofs of all underlying 'broader' shapes
        dofmap[elem] = numeric.array(newdofs) # add result to IEN mapping of new function object
  
      ndofs += keep.sum() # update total number of dofs
      topo = topo.refined # proceed to next level
  
    for elem in parentelems:
      del dofmap[elem] # remove auxiliary elements

    funcsp = function.function( stdmap, dofmap, ndofs, self.ndims )
    domain = UnstructuredTopology( topoelems, ndims=self.ndims )

    if hasattr( topo, 'boundary' ):
      allbelems = []
      bgroups = {}
      topo = self # topology to examine in next level refinement
      for irefine in range( nrefine ):
        belemset = set()
        for belem in topo.boundary:
          celem, transform = belem.context
          if celem in topoelems:
            belemset.add( belem )
        allbelems.extend( belemset )
        for btag, belems in topo.boundary.groups.iteritems():
          bgroups.setdefault( btag, [] ).extend( belemset.intersection(belems) )
        topo = topo.refined # proceed to next level
      domain.boundary = UnstructuredTopology( allbelems, ndims=self.ndims-1 )
      domain.boundary.groups = dict( ( tag, UnstructuredTopology( group, ndims=self.ndims-1 ) ) for tag, group in bgroups.items() )

    if hasattr( topo, 'interfaces' ):
      allinterfaces = []
      topo = self # topology to examine in next level refinement
      for irefine in range( nrefine ):
        for ielem in topo.interfaces:
          (celem1,transform1), (celem2,transform2) = ielem.interface
          if celem1 in topoelems:
            while True:
              if celem2 in topoelems:
                allinterfaces.append( ielem )
                break
              if not celem2.parent:
                break
              celem2, transform2 = celem2.parent
          elif celem2 in topoelems:
            while True:
              if celem1 in topoelems:
                allinterfaces.append( ielem )
                break
              if not celem1.parent:
                break
              celem1, transform1 = celem1.parent
        topo = topo.refined # proceed to next level
      domain.interfaces = UnstructuredTopology( allinterfaces, ndims=self.ndims-1 )
  
    return domain, funcsp

  def refine( self, n ):
    'refine entire topology n times'

    return self if n <= 0 else self.refined.refine( n-1 )

  def get_simplices( self, maxrefine, title='getting simplices' ):
    'Getting simplices'

    log.context( title )

    return [ simplex for elem in self for simplex in elem.get_simplices( maxrefine ) ]

  def get_trimmededges( self, maxrefine, title='getting trimmededges' ):
    'Getting trimmed edges'

    log.context( title )

    return [ trimmededge for elem in self for trimmededge in elem.get_trimmededges( maxrefine ) ]

class _StructuredTopology( Topology ):
  'structured topology'

  def __init__( self, structure, periodic=() ):
    'constructor'

    structure = numeric.asarray(structure)
    self.structure = structure
    self.periodic = tuple(periodic)
    self.groups = {}
    Topology.__init__( self, structure.ndim )

  def make_periodic( self, periodic ):
    'add periodicity'

    return StructuredTopology( self.structure, periodic=periodic )

  def __len__( self ):
    'number of elements'

    return sum( elem is not None for elem in self.structure.flat )

  def __iter__( self ):
    'iterate over elements'

    return itertools.ifilter( None, self.structure.flat )

  def __getitem__( self, item ):
    'subtopology'

    if isinstance( item, str ):
      return Topology.__getitem__( self, item )
    if not isinstance( item, tuple ):
      item = item,
    periodic = [ idim for idim in self.periodic if idim < len(item) and item[idim] == slice(None) ]
    return StructuredTopology( self.structure[item], periodic=periodic )

  @cache.property
  def interfaces( self ):
    'interfaces'

    interfaces = []
    eye = numeric.eye(self.ndims-1)
    for idim in range(self.ndims):
      s1 = (slice(None),)*idim + (slice(-1),)
      s2 = (slice(None),)*idim + (slice(1,None),)
      for elem1, elem2 in numeric.broadcast( self.structure[s1], self.structure[s2] ):
        A = numeric.zeros((self.ndims,self.ndims-1))
        A[:idim] = eye[:idim]
        A[idim+1:] = -eye[idim:]
        b = numeric.hstack( [ numeric.zeros(idim+1), numeric.ones(self.ndims-idim) ] )
        context1 = elem1, element.AffineTransformation( b[1:], A )
        context2 = elem2, element.AffineTransformation( b[:-1], A )
        vertices = numeric.asarray( elem1.vertices ).reshape( [2]*elem1.ndims )[s2].ravel()
        assert numeric.equal( vertices == numeric.asarray( elem2.vertices ).reshape( [2]*elem1.ndims )[s1].ravel() ).all()
        ielem = element.QuadElement( ndims=self.ndims-1, vertices=vertices, interface=(context1,context2) )
        interfaces.append( ielem )
    return UnstructuredTopology( interfaces, ndims=self.ndims-1 )

  def discontfunc( self, degree ):
    'discontinuous shape functions'

    if isinstance( degree, int ):
      degree = ( degree, ) * self.ndims

    dofs = numeric.arange( numeric.product(numeric.array(degree)+1) * len(self) ).reshape( len(self), -1 )
    dofmap = dict( zip( self, dofs ) )

    stdelem = util.product( element.PolyLine( element.PolyLine.bernstein_poly( d ) ) for d in degree )
    funcmap = dict( numeric.broadcast( self.structure, stdelem ) )

    return function.function( funcmap, dofmap, dofs.size, self.ndims )

  def curvefreesplinefunc( self ):
    'spline from vertices'

    p = 2
    periodic = self.periodic

    vertex_structure = numeric.array( 0 )
    dofcount = 1
    slices = []

    for idim in range( self.ndims ):
      periodic_i = idim in periodic
      n = self.structure.shape[idim]

      stdelems_i = element.PolyLine.spline( degree=p, nelems=n, curvature=True )

      stdelems = stdelems[...,_] * stdelems_i if idim else stdelems_i

      nd = n + p - 2
      numbers = numeric.arange( nd )

      vertex_structure = vertex_structure[...,_] * nd + numbers

      dofcount *= nd

      myslice = [ slice(0,2) ]
      for i in range(n-2):
        myslice.append( slice(i,i+p+1) )
      myslice.append( slice(n-2,n) )

      slices.append( myslice )

    dofmap = {}
    for item in numeric.broadcast( self.structure, *numeric.ix_(*slices) ):
      elem = item[0]
      S = item[1:]
      dofmap[ elem ] = vertex_structure[S].ravel()

    dofaxis = function.DofMap( ElemMap(dofmap,self.ndims) )
    funcmap = dict( numeric.broadcast( self.structure, stdelems ) )

    return function.Function( dofaxis=dofaxis, stdmap=ElemMap(funcmap,self.ndims), igrad=0 )

  def stdfunc( self, degree ):
    'spline from vertices'

    if isinstance( degree, int ):
      degree = ( degree, ) * self.ndims

    vertex_structure = numeric.array( 0 )
    dofcount = 1
    slices = []

    stdelem = util.product( element.PolyLine( element.PolyLine.bernstein_poly( d ) ) for d in degree )

    for idim in range( self.ndims ):
      n = self.structure.shape[idim]
      p = degree[idim]

      nd = n * p + 1
      numbers = numeric.arange( nd )
      if idim in self.periodic:
        numbers[-1] = numbers[0]
        nd -= 1
      vertex_structure = vertex_structure[...,_] * nd + numbers
      dofcount *= nd
      slices.append( [ slice(p*i,p*i+p+1) for i in range(n) ] )

    dofmap = {}
    hasnone = False
    for item in numeric.broadcast( self.structure, *numeric.ix_(*slices) ):
      elem = item[0]
      if elem is None:
        hasnone = True
      else:
        S = item[1:]
        dofmap[ elem ] = vertex_structure[S].ravel()

    if hasnone:
      touched = numeric.zeros( dofcount, dtype=bool )
      for dofs in dofmap.itervalues():
        touched[ dofs ] = True
      renumber = touched.cumsum()
      dofcount = int(renumber[-1])
      dofmap = dict( ( elem, renumber[dofs]-1 ) for elem, dofs in dofmap.iteritems() )

    funcmap = dict( numeric.broadcast( self.structure, stdelem ) )
    return function.function( funcmap, dofmap, dofcount, self.ndims )

  @cache.property
  def refined( self ):
    'refine entire topology'

    structure = numeric.array( [ elem.children if elem is not None else [None]*(2**self.ndims) for elem in self.structure.flat ] )
    structure = structure.reshape( self.structure.shape + (2,)*self.ndims )
    structure = structure.transpose( sum( [ ( i, self.ndims+i ) for i in range(self.ndims) ], () ) )
    structure = structure.reshape([ sh * 2 for sh in self.structure.shape ])
    refined = StructuredTopology( structure )
    refined.groups = { key: group.refined for key, group in self.groups.items() }
    return refined

  def __str__( self ):
    'string representation'

    return '%s(%s)' % ( self.__class__.__name__, 'x'.join(map(str,self.structure.shape)) )

  @cache.property
  def multiindex( self ):
    'Inverse map of self.structure: given an element find its location in the structure.'
    return dict( (self.structure[alpha], alpha) for alpha in numeric.ndindex( self.structure.shape ) )

  def neighbor( self, elem0, elem1 ):
    'Neighbor detection, returns codimension of interface, -1 for non-neighboring elements.'

    return elem0.neighbor( elem1 )

    # REPLACES:
    alpha0 = self.multiindex[elem0]
    alpha1 = self.multiindex[elem1]
    diff = numeric.array(alpha0) - numeric.array(alpha1)
    for i, shi in enumerate( self.structure.shape ):
      if diff[i] in (shi-1, 1-shi) and i in self.periodic:
        diff[i] = -numeric.sign( shi )
    if set(diff).issubset( (-1,0,1) ):
      return numeric.sum(numeric.abs(diff))
    return -1
    
class _UnstructuredTopology( Topology ):
  'externally defined topology'

  def __init__( self, elements, ndims, namedfuncs={} ):
    'constructor'

    self.namedfuncs = namedfuncs
    self.elements = elements
    self.groups = {}
    Topology.__init__( self, ndims )

  def __iter__( self ):
    'iterate over elements'

    return iter( self.elements )

  def __len__( self ):
    'number of elements'

    return len(self.elements)

  def splinefunc( self, degree ):
    'spline func'

    return self.namedfuncs[ 'spline%d' % degree ]

  def bubblefunc( self ):
    'linear func + bubble'

    return self.namedfuncs[ 'bubble1' ]

class _HierarchicalTopology( Topology ):
  'collection of nested topology elments'

  def __init__( self, basetopo, elements ):
    'constructor'

    if isinstance( basetopo, HierarchicalTopology ):
      basetopo = basetopo.basetopo
    self.basetopo = basetopo
    self.elements = tuple(elements)
    Topology.__init__( self, basetopo.ndims )

  def __iter__( self ):
    'iterate over elements'

    return iter(self.elements)

  def __len__( self ):
    'number of elements'

    return len(self.elements)

  @cache.property
  def boundary( self ):
    'boundary elements & groups'

    assert hasattr( self.basetopo, 'boundary' )
    allbelems = []
    bgroups = {}
    topo = self.basetopo # topology to examine in next level refinement
    for irefine in range( nrefine ):
      belemset = set()
      for belem in topo.boundary:
        celem, transform = belem.context
        if celem in self.elems:
          belemset.add( belem )
      allbelems.extend( belemset )
      for btag, belems in topo.boundary.groups.iteritems():
        bgroups.setdefault( btag, [] ).extend( belemset.intersection(belems) )
      topo = topo.refined # proceed to next level
    boundary = UnstructuredTopology( allbelems, ndims=self.ndims-1 )
    boundary.groups = dict( ( tag, UnstructuredTopology( group, ndims=self.ndims-1 ) ) for tag, group in bgroups.items() )
    return boundary

  @cache.property
  def interfaces( self ):
    'interface elements & groups'

    assert hasattr( self.basetopo, 'interfaces' )
    allinterfaces = []
    topo = self.basetopo # topology to examine in next level refinement
    for irefine in range( nrefine ):
      for ielem in topo.interfaces:
        (celem1,transform1), (celem2,transform2) = ielem.interface
        if celem1 in self.elems:
          while True:
            if celem2 in self.elems:
              allinterfaces.append( ielem )
              break
            if not celem2.parent:
              break
            celem2, transform2 = celem2.parent
        elif celem2 in self.elems:
          while True:
            if celem1 in self.elems:
              allinterfaces.append( ielem )
              break
            if not celem1.parent:
              break
            celem1, transform1 = celem1.parent
      topo = topo.refined # proceed to next level
    return UnstructuredTopology( allinterfaces, ndims=self.ndims-1 )

  def _funcspace( self, mkspace ):

    log.context( 'generating refined space' )

    dofmap = {} # IEN mapping of new function object
    stdmap = {} # shape function mapping of new function object, plus boolean vector indicating which shapes to retain
    ndofs = 0 # total number of dofs of new function object
    remaining = len(self) # element count down (know when to stop)
  
    topo = self.basetopo # topology to examine in next level refinement
    newdiscard = []
    parentelems = []
    maxrefine = 9
    for irefine in range( maxrefine ):

      funcsp = mkspace( topo ) # shape functions for level irefine
      (func,(dofaxis,)), = function.blocks( funcsp ) # separate elem-local funcs and global placement index
  
      discard = set(newdiscard)
      newdiscard = []
      supported = numeric.ones( funcsp.shape[0], dtype=bool ) # True if dof is contained in topoelems or parentelems
      touchtopo = numeric.zeros( funcsp.shape[0], dtype=bool ) # True if dof touches at least one topoelem
      myelems = [] # all top-level or parent elements in level irefine
      for elem, idofs in dofaxis.dofmap.items():
        if elem in self.elements:
          remaining -= 1
          touchtopo[idofs] = True
          myelems.append( elem )
          newdiscard.append( elem )
        else:
          pelem, trans = elem.parent
          if pelem in discard:
            newdiscard.append( elem )
            supported[idofs] = False
          else:
            parentelems.append( elem )
            myelems.append( elem )
  
      keep = numeric.logical_and( supported, touchtopo ) # THE refinement law

      for elem in myelems: # loop over all top-level or parent elements in level irefine
        idofs = dofaxis.dofmap[elem] # local dof numbers
        mykeep = keep[idofs]
        std = func.stdmap[elem]
        assert isinstance(std,element.StdElem)
        if mykeep.all():
          stdmap[elem] = std # use all shapes from this level
        elif mykeep.any():
          stdmap[elem] = std, mykeep # use some shapes from this level
        newdofs = [ ndofs + keep[:idof].sum() for idof in idofs if keep[idof] ] # new dof numbers
        if irefine: # not at lowest level
          pelem, transform = elem.parent
          newdofs.extend( dofmap[pelem] ) # add dofs of all underlying 'broader' shapes
        dofmap[elem] = numeric.array(newdofs) # add result to IEN mapping of new function object
  
      ndofs += int( keep.sum() ) # update total number of dofs
      if not remaining:
        break
      topo = topo.refined # proceed to next level
  
    else:

      raise Exception, 'elements remaining after %d iterations' % maxrefine

    for elem in parentelems:
      del dofmap[elem] # remove auxiliary elements

    return function.function( stdmap, dofmap, ndofs, self.ndims )

  def stdfunc( self, *args, **kwargs ):
    return self._funcspace( lambda topo: topo.stdfunc( *args, **kwargs ) )

  def splinefunc( self, *args, **kwargs ):
    return self._funcspace( lambda topo: topo.splinefunc( *args, **kwargs ) )

def glue( master, slave, geometry, tol=1.e-10, verbose=False ):
  'Glue topologies along boundary group __glue__.'
  log.context('glue')

  gluekey = '__glue__'

  # Checks on input
  assert gluekey in master.boundary.groups and \
         gluekey in slave.boundary.groups, 'Must identify glue boundary first.'
  assert len(master.boundary[gluekey]) == \
          len(slave.boundary[gluekey]), 'Minimum requirement is that cardinality is equal.'
  assert master.ndims == 2 and slave.ndims == 2, '1D boundaries for now.' # see dists computation and update_vertices

  _mg, _sg = geometry if isinstance( geometry, tuple ) else (geometry,) * 2
  master_geom = _mg.compiled()
  slave_geom = master_geom if _mg == _sg else _sg.compiled()

  nglue = len(master.boundary[gluekey])
  assert len(slave.boundary[gluekey]) == nglue

  log.info( 'pairing elements [%i]' % nglue )
  masterelems = [ ( masterelem, master_geom.eval( masterelem, 'gauss1' )[0] ) for masterelem in master.boundary[gluekey] ]
  elempairs = []
  maxdist = 0
  for slaveelem in slave.boundary[gluekey]:
    slavepoint = slave_geom.eval( slaveelem, 'gauss1' )[0]
    distances = [ numeric.norm2( masterpoint - slavepoint ) for masterelem, masterpoint in masterelems ]
    i = numeric.numpy.argmin( distances )
    maxdist = max( maxdist, distances[i] )
    elempairs.append(( masterelems.pop(i)[0], slaveelem ))
  assert not masterelems
  assert maxdist < tol, 'maxdist exceeds tolerance: %.2e >= %.2e' % ( maxdist, tol )
  log.info( 'all elements matched within %.2e radius' % maxdist )

  # convert element pairs to vertex map
  vtxmap = {}
  for masterelem, slave_elem in elempairs:
    for oldvtx, newvtx in zip( slave_elem.vertices, reversed(masterelem.vertices) ):
      assert vtxmap.setdefault( oldvtx, newvtx ) == newvtx, 'conflicting vertex info'

  emap = {} # elem->newelem map
  for belem in slave.boundary:
    if not any( vtx in vtxmap for vtx in belem.vertices ):
      continue
    emap[belem] = element.QuadElement( belem.ndims,
      vertices=[ vtxmap.get(vtx,vtx) for vtx in belem.vertices ],
      parent=(belem,transform.Identity(belem.ndims)) )
    elem, trans = belem.context
    emap[elem] = element.QuadElement( elem.ndims,
      vertices=[ vtxmap.get(vtx,vtx) for vtx in elem.vertices ],
      parent=(elem,transform.Identity(elem.ndims)) )

  _wrapelem = lambda elem: emap.get(elem,elem)
  def _wraptopo( topo ):
    elems = map( _wrapelem, topo )
    return UnstructuredTopology( elems, ndims=topo.ndims ) if not isinstance( topo, UnstructuredTopology ) \
      else StructuredTopology( numeric.asarray(elems).reshape(slave.structure.shape) )

  # generate glued topology
  elems = list( master ) + map( _wrapelem, slave )
  union = UnstructuredTopology( elems, master.ndims )
  union.groups['master'] = master
  union.groups['slave'] = _wraptopo(slave)
  union.groups.update({ 'master_'+key: topo for key, topo in master.groups.iteritems() })
  union.groups.update({ 'slave_' +key: _wraptopo(topo) for key, topo in slave.groups.iteritems() })

  # generate topology boundary
  belems = [ belem for belem in master.boundary if belem not in master.boundary[gluekey] ] \
    + [ _wrapelem(belem) for belem in slave.boundary if belem not in slave.boundary[gluekey] ]
  union.boundary = UnstructuredTopology( belems, master.ndims-1 )
  union.boundary.groups['master'] = master.boundary
  union.boundary.groups['slave'] = _wraptopo(slave.boundary)
  union.boundary.groups.update({ 'master_'+key: topo for key, topo in master.boundary.groups.iteritems() if key != gluekey })
  union.boundary.groups.update({ 'slave_' +key: _wraptopo(topo) for key, topo in slave.boundary.groups.iteritems() if key != gluekey })

  log.info( 'created glued topology [%i]' % len(union) )
  return union

# vim:shiftwidth=2:foldmethod=indent:foldnestmax=2
