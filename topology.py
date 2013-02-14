from . import element, function, util, numpy, parallel, matrix, log, core, numeric, _

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

  def elem_eval( self, funcs, ischeme, stack=False, title='evaluating' ):
    'element-wise evaluation'

    pbar = log.ProgressBar( self, title=title )
    pbar.add( '[#%d]' % len(self) )

    single_arg = not isinstance(funcs,list)
    if single_arg:
      funcs = funcs,

    retvals = []
    idata = []
    for func in funcs:
      assert isinstance( func, function.ArrayFunc )
      assert all( isinstance(sh,int) for sh in func.shape )
      idata.append( func )
      retvals.append( numpy.empty( len(self), dtype=object ) )
    idata = function.Tuple( idata )

    if pbar.out:
      name = idata.graphviz()
      if name:
        pbar.add( name )

    for ielem, elem in enumerate( pbar ):
      for retval, data in zip( retvals, idata( elem, ischeme ) ):
        retval[ielem] = data

    if stack:
      stacked = []
      nansep = ( stack == 'nan' )
      for retval in retvals:
        npoints = sum( val.shape[0] for val in retval )
        nelems = len( retval )
        newretval = numpy.empty( (npoints+nansep*(nelems-1),)+retval[0].shape[1:] )
        ptr = 0
        for val in retval:
          if ptr and nansep:
            newretval[ptr:ptr+1] = numpy.nan
            ptr += 1
          newretval[ptr:ptr+val.shape[0]] = val
          ptr += val.shape[0]
        assert ptr == newretval.shape[0]
        stacked.append( newretval )
      retvals = stacked
    if single_arg:
      retvals, = retvals
    return retvals

  def elem_mean( self, funcs, coords, ischeme, title='computing mean values' ):
    'element-wise integration'

    pbar = log.ProgressBar( self, title=title )
    pbar.add( '[#%d]' % len(self) )

    single_arg = not isinstance(funcs,list)
    if single_arg:
      funcs = funcs,

    retvals = []
    iweights = coords.iweights( self.ndims )
    idata = [ iweights ]
    for func in funcs:
      assert isinstance( func, function.ArrayFunc )
      assert all( isinstance(sh,int) for sh in func.shape )
      idata.append( function.elemint( func, iweights ) )
      retvals.append( numpy.empty( (len(self),)+func.shape ) )
    idata = function.Tuple( idata )

    if pbar.out:
      name = idata.graphviz()
      if name:
        pbar.add( name )

    for ielem, elem in enumerate( pbar ):
      area_data = idata( elem, ischeme )
      area = area_data[0].sum()
      for retval, data in zip( retvals, area_data[1:] ):
        retval[ielem] = data / area

    if single_arg:
      retvals, = retvals
    return retvals

  def grid_eval( self, funcs, coords, C, title='grid-evaluating' ):
    'evaluate grid points'

    pbar = log.ProgressBar( self, title=title )
    pbar.add( '[#%d]' % len(self) )

    single_arg = not isinstance(funcs,list)
    if single_arg:
      funcs = funcs,

    C = numpy.asarray( C )
    assert C.shape[0] == self.ndims
    shape = C.shape
    C = C.reshape( self.ndims, -1 )

    funcs = [ function._asarray(func) for func in funcs ]
    retvals = [ numpy.empty( C.shape[1:] + func.shape ) for func in funcs ]
    for retval in retvals:
      retval[:] = numpy.nan

    data = function.Tuple([ function.Tuple([ func, retval ]) for func, retval in zip( funcs, retvals ) ])

    for elem in pbar:
      points, selection = coords.find( elem, C.T )
      if selection is not None:
        for func, retval in data( elem, points ):
          retval[selection] = func

    retvals = [ retval.reshape( shape[1:] + func.shape ) for func, retval in zip( funcs, retvals ) ]
    if single_arg:
      retvals, = retvals
    return retvals

  def integrate( self, funcs, ischeme, coords=None, iweights=None, title='integrating' ):
    'integrate'

    pbar = log.ProgressBar( self, title=title )
    pbar.add( '[#%d]' % len(self) )

    single_arg = not isinstance(funcs,list)
    if single_arg:
      funcs = funcs,

    if iweights is None:
      assert coords is not None, 'conflicting arguments coords and iweights'
      iweights = coords.iweights( self.ndims )
    else:
      assert coords is None, 'conflicting arguments coords and iweights'
    assert iweights.ndim == 0

    integrands = []
    retvals = []
    for ifunc, func in enumerate( funcs ):
      func = function._asarray( func )
      if isinstance( func, function.Inflate ):
        if func.ndim == 2:
          nrows, ncols = func.shape
          graph = [[]] * nrows
          IJ = function.Tuple([ function.Tuple(ind) for f, ind in func.blocks ])
          for elem in self:
            for I, J in IJ( elem, None ):
              for i in I:
                graph[i] = numeric.addsorted( graph[i], J, inplace=True )
          A = matrix.SparseMatrix( graph, ncols )
        else:
          A = numpy.zeros( func.shape, dtype=float )
        for f, ind in func.blocks:
          integrands.append( function.Tuple([ ifunc, function.Tuple(ind), function.elemint( f, iweights ) ]) )
      else:
        A = numpy.zeros( func.shape, dtype=float )
        if not function._iszero( func ):
          integrands.append( function.Tuple([ ifunc, (), function.elemint( func, iweights ) ]) )
      retvals.append( A )
    idata = function.Tuple( integrands )

    if pbar.out:
      name = idata.graphviz()
      if name:
        pbar.add( name )

    for elem in pbar:
      for ifunc, index, data in idata( elem, ischeme ):
        retvals[ifunc][index] += data

    if single_arg:
      retvals, = retvals
    return retvals

  def projection( self, fun, onto, coords, **kwargs ):
    'project and return as function'

    weights = self.project( fun, onto, coords, **kwargs )
    return onto.dot( weights )

  def project( self, fun, onto, coords, tol=0, ischeme='gauss8', title='projecting', droptol=1e-8, exact_boundaries=False, constrain=None, verify=None, maxiter=9999999999 ):
    'L2 projection of function onto function space'

    if exact_boundaries:
      assert constrain is None
      constrain = self.boundary.project( fun, onto, coords, title=title+' boundaries', ischeme=ischeme, tol=tol, droptol=droptol )
    elif constrain is None:
      constrain = util.NanVec( onto.shape[0] if isinstance(onto.shape[0],int) else onto.shape[0].stop )
    else:
      assert isinstance( constrain, util.NanVec )
      assert constrain.shape == onto.shape[:1]

    if len( onto.shape ) == 1:
      Afun = onto.outer()
      bfun = onto * fun
    elif len( onto.shape ) == 2:
      Afun = onto.outer().sum( 2 )
      bfun = function.sum( onto * fun )
    else:
      raise Exception
    A, b = self.integrate( [Afun,bfun], coords=coords, ischeme=ischeme, title=title )
    supp = A.rowsupp( droptol ) & numpy.isnan( constrain )
    if verify is not None:
      numcons = (~supp).sum()
      assert numcons == verify, 'number of constraints does not meet expectation: %d != %d' % ( numcons, verify )
    constrain[supp] = 0
    if numpy.all( b == 0 ):
      u = constrain | 0
    else:
      u = A.solve( b, constrain, tol=tol, title=title, symmetric=True, maxiter=maxiter )
    u[supp] = numpy.nan
    return u.view( util.NanVec )

# def trim( self, levelset, maxrefine, lscheme='bezier3' ):
#   'create new domain based on levelset'

#   newelems = []
#   for elem in log.ProgressBar( self, title='selecting/refining elements' ):
#     elempool = [ elem ]
#     for level in range( maxrefine ):
#       nextelempool = []
#       for elem in elempool:
#         inside = levelset( elem, lscheme ) > 0
#         if inside.all():
#           newelems.append( elem )
#         elif inside.any():
#           nextelempool.extend( elem.children )
#       elempool = nextelempool
#     # TODO select >50% overlapping elements from elempool
#     newelems.extend( elempool )
#   return UnstructuredTopology( newelems, ndims=self.ndims )

#   def select( elem, level=0 ):
#     try:
#       inside = levelset( elem, lscheme ) > 0
#     except function.EvaluationError:
#       pass
#     else:
#       return inside.any()
#     assert level < maxrefine, 'failed to evaluate levelset within maxrefine={}'.format( maxrefine )
#     return any( select(child,level+1) for child in elem.children )

#   return IndexedTopology( self, select=select )

  def refinedfunc( self, dofaxis, refine, degree, title='refining' ):
    'create refined space by refining dofs in existing one'

    pbar = log.ProgressBar( None, title )
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
      for elem in topo: # loop over remaining elements in refinement level 'nrefine'
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

    pbar.setmax( len(topoelems) + len(parentelems) )
  
    # initialize
    dofmap = {} # IEN mapping of new function object
    stdmap = {} # shape function mapping of new function object, plus boolean vector indicating which shapes to retain
    ndofs = 0 # total number of dofs of new function object
  
    topo = self # topology to examine in next level refinement
    for irefine in range( nrefine ):
      pbar.write( irefine )
  
      funcsp = topo.splinefunc( degree ) # shape functions for level irefine
      func, (dofaxis,) = funcsp.get_func_ind() # separate elem-local funcs and global placement index
  
      supported = numpy.ones( funcsp.shape[0], dtype=bool ) # True if dof is contained in topoelems or parentelems
      touchtopo = numpy.zeros( funcsp.shape[0], dtype=bool ) # True if dof touches at least one topoelem
      myelems = [] # all top-level or parent elements in level irefine
      for elem, idofs in dofaxis.dofmap.iteritems():
        if elem in topoelems:
          pbar.update()
          touchtopo[idofs] = True
          myelems.append( elem )
        elif elem in parentelems:
          pbar.update()
          myelems.append( elem )
        else:
          supported[idofs] = False
  
      keep = numpy.logical_and( supported, touchtopo ) # THE refinement law
      if keep.all() and irefine == nrefine - 1:
        pbar.close()
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
        dofmap[elem] = numpy.array(newdofs) # add result to IEN mapping of new function object
  
      ndofs += keep.sum() # update total number of dofs
      topo = topo.refined # proceed to next level
  
    for elem in parentelems:
      del dofmap[elem] # remove auxiliary elements

    ndofs = int(ndofs) # make sure we have a python int
    ind = function.DofMap( ElemMap(dofmap,self.ndims) ),
    func = function.Function( stdmap=ElemMap(stdmap,self.ndims), igrad=0 )
    funcsp = function.Inflate( (ndofs,), [(func,ind)] )
    domain = UnstructuredTopology( topoelems, ndims=self.ndims )
  
    pbar.close()
    return domain, funcsp

  def refine( self, n ):
    'refine entire topology n times'

    return self if n <= 0 else self.refined.refine( n-1 )

  def get_simplices( self, title='Getting simplices', **kwargs ):
    'Getting simplices'

    return [simplex for elem in log.ProgressBar( self, title=title ) for simplex in elem.get_simplices( **kwargs )]

class StructuredTopology( Topology ):
  'structured topology'

  def __init__( self, structure, periodic=() ):
    'constructor'

    structure = numpy.asarray(structure)
    self.ndims = structure.ndim
    self.structure = structure
    self.periodic = tuple(periodic)
    self.groups = {}

  def make_periodic( self, periodic ):
    'add periodicity'

    return StructuredTopology( self.structure, periodic=periodic )

  def __len__( self ):
    'number of elements'

    return sum( elem is not None for elem in self.structure.flat )

  def __iter__( self ):
    'iterate'

    return ( elem for elem in self.structure.flat if elem is not None )

  def __getitem__( self, item ):
    'subtopology'

    if isinstance( item, str ):
      return Topology.__getitem__( self, item )
    return StructuredTopology( self.structure[item] )

  @core.cacheprop
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
        belems = numpy.frompyfunc( lambda elem: elem.edge( iedge ) if elem is not None else None, 1, 1 )( self.structure[s] )
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

  @core.cachefunc
  def splinefunc( self, degree, neumann=(), periodic=None, closed=False ):
    'spline from nodes'

    if periodic is None:
      periodic = self.periodic

    if isinstance( degree, int ):
      degree = ( degree, ) * self.ndims

    nodes_structure = numpy.array( 0 )
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

      nd = n + p - 1
      numbers = numpy.arange( nd )
      if periodic_i:
        overlap = p - 1
        numbers[ -overlap: ] = numbers[ :overlap ]
        nd -= overlap
      nodes_structure = nodes_structure[...,_] * nd + numbers
      dofcount *= nd
      slices.append( [ slice(i,i+p) for i in range(n) ] )

    dofmap = {}
    hasnone = False
    for item in numpy.broadcast( self.structure, *numpy.ix_(*slices) ):
      elem = item[0]
      if elem is None:
        hasnone = True
      else:
        S = item[1:]
        dofmap[ elem ] = nodes_structure[S].ravel()

    if hasnone:
      touched = numpy.zeros( dofcount, dtype=bool )
      for dofs in dofmap.itervalues():
        touched[ dofs ] = True
      renumber = touched.cumsum()
      dofcount = int(renumber[-1])
      dofmap = dict( ( elem, renumber[dofs]-1 ) for elem, dofs in dofmap.iteritems() )

    ind = function.DofMap( ElemMap(dofmap,self.ndims) ),
    funcmap = dict( numpy.broadcast( self.structure, stdelems ) )
    funcsp = function.Function( stdmap=ElemMap(funcmap,self.ndims), igrad=0 )
    return function.Inflate( [dofcount], [(funcsp,ind)] )

  @core.cachefunc
  def curvefreesplinefunc( self ):
    'spline from nodes'

    p = 3
    periodic = self.periodic

    nodes_structure = numpy.array( 0 )
    dofcount = 1
    slices = []

    for idim in range( self.ndims ):
      periodic_i = idim in periodic
      n = self.structure.shape[idim]

      stdelems_i = element.PolyLine.spline( degree=p, nelems=n, curvature=True )

      stdelems = stdelems[...,_] * stdelems_i if idim else stdelems_i

      nd = n + p - 3
      numbers = numpy.arange( nd )

      nodes_structure = nodes_structure[...,_] * nd + numbers

      dofcount *= nd

      myslice = [ slice(0,2) ]
      for i in range(n-2):
        myslice.append( slice(i,i+p) )
      myslice.append( slice(n-2,n) )

      slices.append( myslice )

    dofmap = {}
    for item in numpy.broadcast( self.structure, *numpy.ix_(*slices) ):
      elem = item[0]
      S = item[1:]
      dofmap[ elem ] = nodes_structure[S].ravel()

    dofaxis = function.DofMap( ElemMap(dofmap,self.ndims) )
    funcmap = dict( numpy.broadcast( self.structure, stdelems ) )

    return function.Function( dofaxis=dofaxis, stdmap=ElemMap(funcmap,self.ndims), igrad=0 )

  def linearfunc( self ):
    'linears'

    return self.splinefunc( degree=2 )

  @core.cachefunc
  def stdfunc( self, degree ):
    'spline from nodes'

    if isinstance( degree, int ):
      degree = ( degree, ) * self.ndims

    nodes_structure = numpy.array( 0 )
    dofcount = 1
    slices = []

    stdelem = util.product( element.PolyLine( element.PolyLine.bernstein_poly( d ) ) for d in degree )

    for idim in range( self.ndims ):
      n = self.structure.shape[idim]
      p = degree[idim]

      nd = n * (p-1) + 1
      numbers = numpy.arange( nd )
      nodes_structure = nodes_structure[...,_] * nd + numbers
      dofcount *= nd
      slices.append( [ slice((p-1)*i,(p-1)*i+p) for i in range(n) ] )

    dofmap = {}
    hasnone = False
    for item in numpy.broadcast( self.structure, *numpy.ix_(*slices) ):
      elem = item[0]
      if elem is None:
        hasnone = True
      else:
        S = item[1:]
        dofmap[ elem ] = nodes_structure[S].ravel()

    if hasnone:
      touched = numpy.zeros( dofcount, dtype=bool )
      for dofs in dofmap.itervalues():
        touched[ dofs ] = True
      renumber = touched.cumsum()
      dofcount = int(renumber[-1])
      dofmap = dict( ( elem, renumber[dofs]-1 ) for elem, dofs in dofmap.iteritems() )

    ind = function.DofMap( ElemMap(dofmap,self.ndims) ),
    funcmap = dict( numpy.broadcast( self.structure, stdelem ) )
    funcsp = function.Function( stdmap=ElemMap(funcmap,self.ndims), igrad=0 )
    return function.Inflate( [dofcount], [(funcsp,ind)] )

  def rectilinearfunc( self, gridnodes ):
    'rectilinear func'

    assert len( gridnodes ) == self.ndims
    nodes_structure = numpy.empty( map( len, gridnodes ) + [self.ndims] )
    for idim, inodes in enumerate( gridnodes ):
      shape = [1,] * self.ndims
      shape[idim] = -1
      nodes_structure[...,idim] = numpy.asarray( inodes ).reshape( shape )
    return self.linearfunc().dot( nodes_structure.reshape( -1, self.ndims ) )

  @core.weakcacheprop
  def refined( self ):
    'refine entire topology'

    structure = numpy.array( [ list(elem.children) if elem is not None else [None]*(2**self.ndims) for elem in self.structure.flat ] )
    structure = structure.reshape( self.structure.shape + (2,)*self.ndims )
    structure = structure.transpose( sum( [ ( i, self.ndims+i ) for i in range(self.ndims) ], () ) )
    structure = structure.reshape( [ self.structure.shape[i] * 2 for i in range(self.ndims) ] )

    return StructuredTopology( structure )

  def trim( self, levelset, maxrefine, lscheme='bezier3', finestscheme='uniform2', evalrefine=0, title='trimming' ):
    'trim element along levelset'

    pbar = log.ProgressBar( self.structure.size, title )
    def trimelem( elem ):
      pbar.update()
      return elem.trim( levelset=levelset, maxrefine=maxrefine, lscheme=lscheme, finestscheme=finestscheme, evalrefine=evalrefine )
    trimmedstructure = util.objmap( trimelem, self.structure )
    pbar.close()
    return StructuredTopology( trimmedstructure, periodic=self.periodic )

#   elems = filter( None, [ elem.trim( levelset=levelset, maxrefine=maxrefine, lscheme=lscheme, finestscheme=finestscheme, evalrefine=evalrefine ) for elem in log.ProgressBar( self, title ) ] )
#   return IndexedTopology( self, elems )

  def __str__( self ):
    'string representation'

    return '%s(%s)' % ( self.__class__.__name__, 'x'.join(map(str,self.structure.shape)) )
    
class IndexedTopology( Topology ):
  'trimmed topology'
  
  def __init__( self, topo, elements ):
    'constructor'

    self.topo = topo
    self.ndims = topo.ndims
    self.elements = elements

  def __iter__( self ):
    'number of elements'

    return iter(self.elements)

  def __len__( self ):
    'number of elements'

    return len(self.elements)

  def splinefunc( self, degree ):
    'create spline function space'

    funcsp = self.topo.splinefunc( degree )
    func, (dofaxis,) = funcsp.get_func_ind()
    touched = numpy.zeros( funcsp.shape[0], dtype=bool )
    for elem in self:
      dofs = dofaxis(elem,None)
      touched[ dofs ] = True
    renumber = touched.cumsum()
    ndofs = int(renumber[-1])
    dofmap = dict( ( elem, renumber[ dofaxis(elem,None) ]-1 ) for elem in self )
    ind = function.DofMap( ElemMap(dofmap,self.ndims) ),
    return function.Inflate( (ndofs,), [(func,ind)] )

  @core.weakcacheprop
  def refined( self ):
    'refine all elements 2x'

    elements = [ child for elem in self.elements for child in elem.children if child is not None ]
    return IndexedTopology( self.topo.refined, elements )

  @core.cacheprop
  def boundary( self ):
    'boundary'

    return self.topo.boundary

class UnstructuredTopology( Topology ):
  'externally defined topology'

  groups = ()

  def __init__( self, elements, ndims, namedfuncs={} ):
    'constructor'

    self.namedfuncs = namedfuncs
    self.ndims = ndims
    self.elements = elements

  def __iter__( self ):
    'number of elements'

    return iter(self.elements)

  def __len__( self ):
    'number of elements'

    return len(self.elements)

  def splinefunc( self, degree ):
    'spline func'

    return self.namedfuncs[ 'spline%d' % degree ]

  def linearfunc( self ):
    'linear func'

    return self.splinefunc( degree=2 )

  @core.weakcacheprop
  def refined( self ):
    'refined (=refine(2))'

    try:
      linearfunc = self.linearfunc()
      func, (dofaxis,) = linearfunc.get_func_ind()
      ndofs = linearfunc.shape[0]
      edges = {}
      nmap = {}
    except:
      dofaxis = None

    elements = []
    for elem in self:
      children = list( elem.children )
      elements.extend( children )
      if not dofaxis:
        continue

      nodedofs = dofaxis(elem,None)
      edgedofs = []
      if isinstance( elem, element.TriangularElement ):
        for i in range(3):
          j = (i+1)%3
          try:
            edgedof = edges.pop(( nodedofs[i], nodedofs[j] ))
          except KeyError:
            edgedof = edges[( nodedofs[j], nodedofs[i] )] = ndofs
            ndofs += 1
          edgedofs.append( edgedof )
        nmap[ children[0] ] = numpy.array([ edgedofs[2], edgedofs[1], nodedofs[2] ])
        nmap[ children[1] ] = numpy.array([ edgedofs[0], nodedofs[1], edgedofs[1] ])
        nmap[ children[2] ] = numpy.array([ nodedofs[0], edgedofs[0], edgedofs[2] ])
        nmap[ children[3] ] = numpy.array([ edgedofs[1], edgedofs[2], edgedofs[0] ])
      else:
        raise NotImplementedError

    #print 'boundary:', edges

    if dofaxis:
      ind = function.DofMap( ElemMap(nmap,self.ndims) ),
      fmap = dict.fromkeys( elements, element.PolyTriangle(1) )
      func = function.Function( stdmap=ElemMap(fmap,self.ndims), igrad=0 )
      linearfunc = function.Inflate( (ndofs,), [(func,ind)] )
      namedfuncs = { 'spline2': linearfunc }
    else:
      namedfuncs = {}

    return UnstructuredTopology( elements, ndims=2, namedfuncs=namedfuncs )

# vim:shiftwidth=2:foldmethod=indent:foldnestmax=2
