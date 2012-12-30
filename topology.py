from . import element, function, util, numpy, parallel, matrix, log, core, numeric, _

class ElemMap( dict ):
  'dictionary-like element mapping'

  def __init__( self, mapping, ndims, overlap ):
    'constructor'

    self.ndims = ndims
    self.overlap = overlap
    dict.__init__( self, mapping )

  def __str__( self ):
    'string representation'

    return 'ElemMap(#%d,%dD,%s)' % ( len(self), self.ndims, 'ML' if self.overlap else 'SL' )

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

  def elemintegrate( self, funcs, ischeme, coords, title='integrating element-wise' ):
    'element-wise integration'

    pbar = log.ProgressBar( self, title=title )
    pbar.add( '[#%d]' % len(self) )

    single_arg = not isinstance(funcs,list)
    if single_arg:
      funcs = funcs,

    retvals = []
    iweights = coords.iweights( self.ndims )
    idata = []
    for ifunc, func in enumerate( funcs ):
      if not isinstance( func, function.ArrayFunc ):
        func = function.StaticArray( func )
      arr = numpy.empty( (len(self),)+func.shape )
      idata.append( function.Tuple([ function.StaticArray(arr), func, iweights ]) )
      retvals.append( arr )
    idata = function.Tuple( idata )

    if pbar.out:
      name = idata.graphviz()
      if name:
        pbar.add( name )

    for i, elem in enumerate( pbar ):
      if isinstance(ischeme,str):
        points = elem.eval(ischeme)
      else:
        points = ischeme[elem]
        if points is None:
          for arr in retvals:
            arr[i] = numpy.nan
          continue
      for arr, data, w in idata( elem, points ):
        arr[i] = numeric.contract( data.T, w ).T

    if single_arg:
      retvals, = retvals
    return retvals

  def integrate( self, funcs, ischeme=None, coords=None, title='integrating' ):
    'integrate'

    pbar = log.ProgressBar( self, title=title )
    pbar.add( '[#%d]' % len(self) )

    single_arg = not isinstance(funcs,list)
    if single_arg:
      funcs = funcs,

    iweights = coords.iweights( self.ndims )
    integrands = []
    retvals = []
    for ifunc, func in enumerate( funcs ):
      if not isinstance(func,tuple):
        func = func,
      func = filter( None, func ) # skip over zero integrands
      ndim = func[0].ndim if func else 0
      assert all( f.ndim == ndim for f in func[1:] )
      if ndim == 0:
        for f in func:
          integrands.append( function.Tuple([ ifunc, (), f, iweights ]) )
        A = numpy.array( 0, dtype=float )
      elif ndim == 1:
        if len( func ) == 1 and isinstance( func[0].shape[0], int ): # special case dense vector
          length = func[0].shape[0]
          integrands.append( function.Tuple([ ifunc, slice(None), func[0], iweights ]) )
        else:
          length = max( f.shape[0].stop for f in func )
          for f in func:
            sh, = f.shape
            integrands.append( function.Tuple([ ifunc, sh, f, iweights ]) )
        A = numpy.zeros( length, dtype=float )
      elif ndim == 2:
        nrows = max( f.shape[0].stop for f in func )
        ncols = max( f.shape[1].stop for f in func )
        if any( isinstance(sh,slice) for f in func for sh in f.shape ):
          for f in func:
            IJ = function.Tuple( f.shape )
            integrands.append( function.Tuple([ ifunc, IJ, f, iweights ]) )
          A = matrix.DenseMatrix( (nrows,ncols) )
        else:
          graph = [[]] * nrows
          for f in func:
            IJ = function.Tuple( f.shape )
            for elem in self:
              I, J = IJ( elem, None )
              for i in I:
                graph[i] = numeric.addsorted( graph[i], J, inplace=True )
            integrands.append( function.Tuple([ ifunc, IJ, f, iweights ]) )
          A = matrix.SparseMatrix( graph, ncols )
      else:
        raise NotImplementedError, 'ndim=%d' % func.ndim
      retvals.append( A )
    idata = function.Tuple( integrands )

    if pbar.out:
      name = idata.graphviz()
      if name:
        pbar.add( name )

    for elem in pbar:
      if isinstance(ischeme,str):
        points = elem.eval(ischeme)
      else:
        points = ischeme[elem]
        if points is None:
          continue
      for ifunc, index, data, w in idata( elem, points ):
        retvals[ifunc][index] += numeric.contract( data.T, w ).T

    if single_arg:
      retvals, = retvals
    return retvals

  def projection( self, fun, onto, coords, **kwargs ):
    'project and return as function'

    weights = self.project( fun, onto, coords, **kwargs )
    return onto.dot( weights )

  def project( self, fun, onto, coords, tol=0, ischeme='gauss8', title='projecting', droptol=1e-8, exact_boundaries=False, constrain=None, verify=None ):
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
    for elem in log.ProgressBar( self, title='selecting/refining elements' ):
      elempool = [ elem ]
      for level in range( maxrefine ):
        nextelempool = []
        for elem in elempool:
          inside = levelset( elem, lscheme ) > 0
          if inside.all():
            newelems.append( elem )
          elif inside.any():
            nextelempool.extend( elem.children )
        elempool = nextelempool
      # TODO select >50% overlapping elements from elempool
      newelems.extend( elempool )
    return UnstructuredTopology( newelems, ndims=self.ndims )

  def refinedfunc( self, dofaxis, degree, refine ):
    'create refined space by refining dofs in existing one'
  
    refine = set(refine) # make unique and equip with set operations
  
    # initialize
    topoelems = [] # non-overlapping 'top-level' elements, will make up the new domain
    parentelems = [] # all parents, grandparents etc of topoelems
    nrefine = 0 # number of nested topologies after refinement
  
    topo = self # elements to examine in next level refinement
    while topo:
      nexttopo = []
      refined = set() # refined dofs in current refinement level
      for elem in topo: # loop over remaining elements in refinement level 'nrefine'
        dofs = dofaxis.dofmap.get(elem) # dof numbers for current funcsp object
        if dofs is None: # elem is not a top-level element
          parentelems.append( elem ) # elem is a parent
          nexttopo.extend( elem.children ) # examine children in next iteration
        else: # elem is a top-level element
          supp = refine.intersection(dofs) # supported dofs that are tagged for refinement
          if supp: # elem supports dofs for refinement
            parentelems.append( elem ) # elem will become a parent
            topoelems.extend( elem.children ) # children will become top-level elements
            refined.update( supp ) # dofs will not be considered in following refinement levels
          else: # elem does not support dofs for refinement
            topoelems.append( elem ) # elem remains a top-level elemnt
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
    for irefine in range( nrefine ):
  
      funcsp = topo.splinefunc( degree ) # shape functions for level irefine
      dofaxis = funcsp.shape[0] # IEN mapping local to level irefine
  
      supported = numpy.ones( int(dofaxis), dtype=bool ) # True if dof is contained in topoelems or parentelems
      touchtopo = numpy.zeros( int(dofaxis), dtype=bool ) # True if dof touches at least one topoelem
      myelems = [] # all top-level or parent elements in level irefine
      for elem, idofs in dofaxis.dofmap.iteritems():
        if elem in topoelems:
          touchtopo[idofs] = True
          myelems.append( elem )
        elif elem in parentelems:
          myelems.append( elem )
        else:
          supported[idofs] = False
  
      keep = numpy.logical_and( supported, touchtopo ) # THE refinement law
      if keep.all() and irefine == nrefine - 1:
        return topo, funcsp
  
      for elem in myelems: # loop over all top-level or parent elements in level irefine
        idofs = dofaxis.dofmap[elem] # local dof numbers
        mykeep = keep[idofs]
        std = funcsp.stdmap[elem]
        assert isinstance(std,element.StdElem)
        if mykeep.all():
          stdmap[elem] = std # use all shapes from this level
        elif mykeep.any():
          stdmap[elem] = std, mykeep # use some shapes from this level
        newdofs = [ ndofs + keep[:idof].sum() for idof in idofs if keep[idof] ] # new dof numbers
        if elem.parent:
          pelem, transform = elem.parent
          newdofs.extend( dofmap[pelem] ) # add dofs of all underlying 'broader' shapes
        dofmap[elem] = numpy.array(newdofs) # add result to IEN mapping of new function object
  
      ndofs += keep.sum() # update total number of dofs
      topo = topo.refined # proceed to next level
  
    for elem in parentelems:
      del dofmap[elem] # remove auxiliary elements
  
    dofaxis = function.DofMap( ndofs, ElemMap(dofmap,self.ndims,overlap=False) )
    funcsp = function.Function( dofaxis=dofaxis, stdmap=ElemMap(stdmap,self.ndims,overlap=True), igrad=0 )
    domain = UnstructuredTopology( topoelems, ndims=self.ndims )
  
    return domain, funcsp

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
    for item in numpy.broadcast( self.structure, *numpy.ix_(*slices) ):
      elem = item[0]
      S = item[1:]
      dofmap[ elem ] = nodes_structure[S].ravel()

    dofaxis = function.DofMap( dofcount, ElemMap(dofmap,self.ndims,overlap=False) )
    funcmap = dict( numpy.broadcast( self.structure, stdelems ) )

    return function.Function( dofaxis=dofaxis, stdmap=ElemMap(funcmap,self.ndims,overlap=False), igrad=0 )

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

    print slices

    dofmap = {}
    for item in numpy.broadcast( self.structure, *numpy.ix_(*slices) ):
      elem = item[0]
      S = item[1:]
      dofmap[ elem ] = nodes_structure[S].ravel()

    dofaxis = function.DofMap( dofcount, ElemMap(dofmap,self.ndims,overlap=False) )
    funcmap = dict( numpy.broadcast( self.structure, stdelems ) )

    return function.Function( dofaxis=dofaxis, stdmap=ElemMap(funcmap,self.ndims,overlap=False), igrad=0 )

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

    structure = numpy.array( [ elem.refined(n) for elem in self.structure.flat ] )
    structure = structure.reshape( self.structure.shape + (n,)*self.ndims )
    structure = structure.transpose( sum( [ ( i, self.ndims+i ) for i in range(self.ndims) ], () ) )
    structure = structure.reshape( [ self.structure.shape[i] * n for i in range(self.ndims) ] )

    return StructuredTopology( structure )

  @core.weakcacheprop
  def refined( self ):
    'refined (=refine(2))'

    structure = numpy.array( [ list(elem.children) for elem in self.structure.flat ] )
    structure = structure.reshape( self.structure.shape + (2,)*self.ndims )
    structure = structure.transpose( sum( [ ( i, self.ndims+i ) for i in range(self.ndims) ], () ) )
    structure = structure.reshape( [ self.structure.shape[i] * 2 for i in range(self.ndims) ] )

    return StructuredTopology( structure )

  def __str__( self ):
    'string representation'

    return '%s(%s)' % ( self.__class__.__name__, 'x'.join(map(str,self.structure.shape)) )
    
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

  @core.weakcacheprop
  def refined( self ):
    'refined (=refine(2))'

    try:
      linearfunc = self.linearfunc()
      dofaxis, = linearfunc.shape
      ndofs = int(dofaxis)
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
      dofaxis = function.DofMap( ndofs, ElemMap(nmap,self.ndims,overlap=False) )
      fmap = dict.fromkeys( elements, element.PolyTriangle(1) )
      linearfunc = function.Function( dofaxis=dofaxis, stdmap=ElemMap(fmap,self.ndims,overlap=False), igrad=0 )
      namedfuncs = { 'spline2': linearfunc }
    else:
      namedfuncs = {}

    return UnstructuredTopology( elements, ndims=2, namedfuncs=namedfuncs )

# vim:shiftwidth=2:foldmethod=indent:foldnestmax=2
