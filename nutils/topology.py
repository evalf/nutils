from . import element, function, util, numpy, parallel, matrix, log, core, numeric, prop, _
import warnings, itertools, libmatrix


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

  def __init__( self, ndims ):
    'constructor'

    self.ndims = ndims

  def refined_by( self, refine ):
    'create refined space by refining dofs in existing one'

    refine = list( refine )
    refined = []
    for elem in self:
      if elem in refine:
        refine.remove( elem )
        refined.extend( elem.children )
      else:
        refined.append( elem )
        pelem = elem # only for argument checking:
        while pelem.parent:
          pelem, trans = pelem.parent
          if pelem in refine:
            refine.remove( pelem )

    assert not refine, 'not all refinement elements were found: %s' % '\n '.join( str(e) for e in refine )
    return HierarchicalTopology( self, refined )

  @core.cache
  def stdfunc( self, degree ):
    'spline from vertices'

    if isinstance( degree, int ):
      degree = ( degree, ) * self.ndims

    assert all( n == 1 for n in degree ) # for now!

    dofmap = { n: i for i, n in enumerate( sorted( set( n for elem in self for n in elem.vertices ) ) ) }
    fmap = dict.fromkeys( self, element.PolyTriangle(1) )
    nmap = { elem: numpy.array([ dofmap[n] for n in elem.vertices ]) for elem in self }
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

  def elem_eval( self, funcs, ischeme, stack=False, separate=None, title='evaluating' ):
    'element-wise evaluation'

    log.context( title )

    if separate is None:
      warnings.warn( '''in elem_eval: stack is deprecated and will be removed.
  List-of-arrays (was stack=False) is no longer supported, replaced by nan-separation.
  In plot.PyPlot use separate=True for mesh, separate=False for e.g. quiver.''' )
      separate = stack is not True

    single_arg = not isinstance(funcs,(tuple,list))
    if single_arg:
      funcs = funcs,

    slices = []
    pointshape = function.PointShape()
    npoints = 0
    separators = []
    for elem in log.iterate('elem',self):
      np, = pointshape( elem, ischeme )
      slices.append( slice(npoints,npoints+np) )
      npoints += np
      if separate:
        separators.append( npoints )
        npoints += 1
    if separate:
      separators = numpy.array( separators[:-1], dtype=int )
      npoints -= 1

    retvals = []
    idata = []
    for ifunc, func in enumerate( funcs ):
      func = function._asarray( func )
      retval = parallel.shzeros( (npoints,)+func.shape, dtype=func.dtype )
      if separate:
        retval[separators] = numpy.nan
      if function._isfunc( func ):
        for f, ind in function.blocks( func ):
          idata.append( function.Tuple( [ ifunc, function.Tuple(ind), f ] ) )
      else:
        idata.append( function.Tuple( [ ifunc, (), func ] ) )
      retvals.append( retval )
    idata = function.Tuple( idata )

    for ielem, elem in parallel.pariter( enumerate( self ) ):
      s = slices[ielem],
      for ifunc, index, data in idata( elem, ischeme ):
        retvals[ifunc][s+index] = data

    log.info( 'created', ', '.join( '%s(%s)' % ( retval.__class__.__name__, ','.join(map(str,retval.shape)) ) for retval in retvals ) )
    if single_arg:
      retvals, = retvals

    return retvals

  def elem_mean( self, funcs, coords, ischeme, title='computing mean values' ):
    'element-wise integration'

    log.context( title )

    single_arg = not isinstance(funcs,(tuple,list))
    if single_arg:
      funcs = funcs,

    retvals = []
    #iweights = coords.iweights( self.ndims )
    iweights = function.iwscale( coords, self.ndims ) * function.IWeights()
    idata = [ iweights ]
    for func in funcs:
      func = function._asarray( func )
      if not function._isfunc( func ):
        func = function.Const( func )
      assert all( isinstance(sh,int) for sh in func.shape )
      idata.append( function.elemint( func, iweights ) )
      retvals.append( numpy.empty( (len(self),)+func.shape ) )
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

  def grid_eval( self, funcs, coords, C, title='grid-evaluating' ):
    'evaluate grid points'

    log.context( title )

    single_arg = not isinstance(funcs,(tuple,list))
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

    for elem in log.iterate('elem',self):
      points, selection = coords.find( elem, C.T )
      if selection is not None:
        for func, retval in data( elem, points ):
          retval[selection] = func

    retvals = [ retval.reshape( shape[1:] + func.shape ) for func, retval in zip( funcs, retvals ) ]
    log.info( 'created', ', '.join( '%s(%s)' % ( retval.__class__.__name__, ','.join(map(str,retval.shape)) ) for retval in retvals ) )
    if single_arg:
      retvals, = retvals

    return retvals

  def integrate( self, funcs, ischeme, coords=None, iweights=None, force_dense=False, title='integrating' ):
    'integrate'

    if force_dense: raise NotImplementedError

    log.context( title )

    single_arg = not isinstance(funcs,(list,tuple))
    if single_arg:
      funcs = funcs,

    if iweights is None:
      assert coords is not None, 'conflicting arguments coords and iweights'
      iweights = function.iwscale( coords, self.ndims ) * function.IWeights()
    else:
      assert coords is None, 'conflicting arguments coords and iweights'
    assert iweights.ndim == 0

    integrands = []
    retvals = []
    for ifunc, func in enumerate( funcs ):
      func = function._asarray( func )
      if function._isfunc( func ):
        maps, = [ f.shape for f, ind in function.blocks( func ) ]
        array = libmatrix.Array( maps )
        for f, ind in function.blocks( func ):
          integrands.append( function.Tuple([ ifunc, function.Tuple(ind), function.elemint( f, iweights ) ]) )
      else:
        raise NotImplementedError
        array = parallel.shzeros( func.shape, dtype=float )
        if not function._iszero( func ):
          integrands.append( function.Tuple([ ifunc, (), function.elemint( func, iweights ) ]) )
      retvals.append( array )
    idata = function.Tuple( integrands )

    idata.compile()
    for elem in parallel.pariter( log.iterate('elem',self) ):
      for ifunc, index, data in idata( elem, ischeme ):
        retvals[ifunc].add_global( index, data )

    for retval in retvals:
      retval.complete()

    log.info( 'created', ', '.join( '%s(%s)' % ( retval.__class__.__name__, ','.join(map(str,retval.shape)) ) for retval in retvals ) )
    if single_arg:
      retvals, = retvals

    return retvals

  def integrate_symm( self, funcs, ischeme, coords=None, iweights=None, force_dense=False, title='integrating' ):
    'integrate a symmetric integrand on a product domain' # TODO: find a proper home for this

    log.context( title )

    single_arg = not isinstance(funcs,list)
    if single_arg:
      funcs = funcs,

    if iweights is None:
      assert coords is not None, 'conflicting arguments coords and iweights'
      iweights = function.iwscale( coords, self.ndims ) * function.IWeights()
    else:
      assert coords is None, 'conflicting arguments coords and iweights'
    assert iweights.ndim == 0

    integrands = []
    retvals = []
    for ifunc, func in enumerate( funcs ):
      func = function._asarray( func )
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
    idata = function.Tuple( integrands )

    for elem in parallel.pariter( log.iterate('elem',self) ):
      assert isinstance( elem, element.ProductElement )
      compare_elem = cmp( elem.elem1, elem.elem2 )
      if compare_elem < 0:
        continue
      for ifunc, lock, index, data in idata( elem, ischeme ):
        with lock:
          retvals[ifunc][index] += data
          if compare_elem > 0:
            retvals[ifunc][index[::-1]] += data.T

    log.info( 'created', ', '.join( '%s(%s)' % ( retval.__class__.__name__, ','.join(map(str,retval.shape)) ) for retval in retvals ) )
    if single_arg:
      retvals, = retvals

    return retvals

  def projection( self, fun, onto, coords, **kwargs ):
    'project and return as function'

    weights = self.project( fun, onto, coords, **kwargs )
    return onto.dot( weights )

  def project( self, fun, onto, coords, tol=0, ischeme=None, title='projecting', droptol=1e-8, exact_boundaries=False, constrain=None, verify=None, maxiter=0, ptype='lsqr' ):
    'L2 projection of function onto function space'

    log.context( title + ' [%s]' % ptype )

    if exact_boundaries:
      assert constrain is None
      constrain = self.boundary.project( fun, onto, coords, title='boundaries', ischeme=ischeme, tol=tol, droptol=droptol, ptype=ptype )
    elif constrain is None:
      constrain = util.NanVec( onto.shape[0] )
    else:
      assert isinstance( constrain, util.NanVec )
      assert constrain.shape == onto.shape[:1]

    if ptype == 'lsqr':
      if ischeme is None:
        ischeme = 'gauss8'
        warnings.warn( 'please specify an integration scheme for project; the current default ischeme=%r will be removed in future' % ischeme, DeprecationWarning )
      #assert ischeme is not None, 'ptype %r requires an ischeme' % ptype
      if len( onto.shape ) == 1:
        Afun = function.outer( onto )
        bfun = onto * fun
      elif len( onto.shape ) == 2:
        Afun = function.outer( onto ).sum( 2 )
        bfun = function.sum( onto * fun )
      else:
        raise Exception
      A, b = self.integrate( [Afun,bfun], coords=coords, ischeme=ischeme, title='building system' )
      constrain = A.solve( b )
      constrain.nan_from_supp( A )
      #N = A.rowsupp(droptol)
      #if numpy.all( b == 0 ):
      #  constrain[~constrain.where&N] = 0
      #else:
      #  solvecons = constrain.copy()
      #  solvecons[~(constrain.where|N)] = 0
      #  u = A.solve( b, solvecons, tol=tol, symmetric=True, maxiter=maxiter )
      #  constrain[N] = u[N]

    elif ptype == 'convolute':
      if ischeme is None:
        ischeme = 'gauss8'
        warnings.warn( 'please specify an integration scheme for project; the current default ischeme=%r will be removed in future' % ischeme, DeprecationWarning )
      #assert ischeme is not None, 'ptype %r requires an ischeme' % ptype
      if len( onto.shape ) == 1:
        ufun = onto * fun
        afun = onto
      elif len( onto.shape ) == 2:
        ufun = function.sum( onto * fun )
        afun = function.norm2( onto )
      else:
        raise Exception
      u, scale = self.integrate( [ ufun, afun ], coords=coords, ischeme=ischeme )
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

      F = numpy.zeros( onto.shape[0] )
      W = numpy.zeros( onto.shape[0] )
      I = numpy.zeros( onto.shape[0], dtype=bool )
      fun = function._asarray( fun )
      data = function.Tuple( function.Tuple([ fun, f, function.Tuple(ind) ]) for f, ind in function.blocks( onto ) )
      for elem in self:
        for f, w, ind in data( elem, 'bezier2' ):
          w = w.swapaxes(0,1) # -> dof axis, point axis, ...
          wf = w * f[ (slice(None),)+numpy.ix_(*ind[1:]) ]
          W[ind[0]] += w.reshape(w.shape[0],-1).sum(1)
          F[ind[0]] += wf.reshape(w.shape[0],-1).sum(1)
          I[ind[0]] = True

      I[constrain.where] = False
      constrain[I] = F[I] / W[I]

    else:
      raise Exception, 'invalid projection %r' % ptype

#   errfun2 = ( onto.dot( constrain | 0 ) - fun )**2
#   if errfun2.ndim == 1:
#     errfun2 = errfun2.sum()
#   error2, area = self.integrate( [ errfun2, 1 ], coords=coords, ischeme=ischeme or 'gauss2' )
#   avg_error = numpy.sqrt(error2) / area

#   numcons = constrain.where.sum()
#   if verify is not None:
#     assert numcons == verify, 'number of constraints does not meet expectation: %d != %d' % ( numcons, verify )

#   log.info( 'constrained %d/%d dofs, error %.2e/area' % ( numcons, constrain.size, avg_error ) )

    return constrain

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
  
      supported = numpy.ones( funcsp.shape[0], dtype=bool ) # True if dof is contained in topoelems or parentelems
      touchtopo = numpy.zeros( funcsp.shape[0], dtype=bool ) # True if dof touches at least one topoelem
      myelems = [] # all top-level or parent elements in level irefine
      for elem, idofs in log.iterate( 'element', dofaxis.dofmap.items() ):
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

class StructuredTopology( Topology ):
  'structured topology'

  def __init__( self, structure, decompose, periodic=() ):
    'constructor'

    structure = numpy.asarray(structure)
    self.structure = structure
    self.periodic = tuple(periodic)
    self.groups = {}
    Topology.__init__( self, structure.ndim )

    # domain decomposition
    if isinstance( decompose, libmatrix.LibMatrix ):
      self.comm = decompose
    else:
      log.info( 'starting libmatrix' )
      self.comm = libmatrix.LibMatrix( nprocs=decompose )
      log.info( 'libmatrix running' )
      iax = numpy.argmax( self.structure.shape )
      bounds = ( numpy.arange( decompose+1 ) * self.structure.shape[iax] ) / decompose
      for ipart in range(decompose):
        elemrange = slice( *bounds[ipart:ipart+2] )
        for elem in self.structure[(slice(None),)*iax+(elemrange,)].flat:
          elem.subdom = ipart

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
    return StructuredTopology( self.structure[item] )

  @property
  @core.cache
  def boundary( self ):
    'boundary'

    shape = numpy.asarray( self.structure.shape ) + 1
    vertices = numpy.arange( numpy.product(shape) ).reshape( shape )

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
        s = tuple(s)
        belems = numpy.frompyfunc( lambda elem: elem.edge( iedge ) if elem is not None else None, 1, 1 )( self.structure[s] )
      else:
        belems = numpy.array( self.structure[-iside].edge( 1-iedge ) )
      boundaries.append( StructuredTopology( belems, decompose=self.comm ) )

    if self.ndims == 2:
      structure = numpy.concatenate([ boundaries[i].structure for i in [0,2,1,3] ])
      topo = StructuredTopology( structure, periodic=[0], decompose=self.comm )
    else:
      allbelems = [ belem for boundary in boundaries for belem in boundary.structure.flat if belem is not None ]
      topo = UnstructuredTopology( allbelems, ndims=self.ndims-1 )

    topo.groups = dict( zip( ( 'left', 'right', 'bottom', 'top', 'front', 'back' ), boundaries ) )
    return topo

  @property
  @core.cache
  def interfaces( self ):
    'interfaces'

    interfaces = []
    eye = numpy.eye(self.ndims-1)
    for idim in range(self.ndims):
      s1 = (slice(None),)*idim + (slice(-1),)
      s2 = (slice(None),)*idim + (slice(1,None),)
      for elem1, elem2 in numpy.broadcast( self.structure[s1], self.structure[s2] ):
        A = numpy.zeros((self.ndims,self.ndims-1))
        A[:idim] = eye[:idim]
        A[idim+1:] = -eye[idim:]
        b = numpy.hstack( [ numpy.zeros(idim+1), numpy.ones(self.ndims-idim) ] )
        context1 = elem1, element.AffineTransformation( b[1:], A )
        context2 = elem2, element.AffineTransformation( b[:-1], A )
        vertices = numpy.reshape( elem1.vertices, [2]*elem1.ndims )[s2].ravel()
        assert numpy.all( vertices == numpy.reshape( elem2.vertices, [2]*elem1.ndims )[s1].ravel() )
        ielem = element.QuadElement( ndims=self.ndims-1, vertices=vertices, interface=(context1,context2) )
        interfaces.append( ielem )
    return UnstructuredTopology( interfaces, ndims=self.ndims-1 )

  @core.cache
  def splinefunc( self, degree, neumann=(), knots=None, periodic=None, closed=False, removedofs=None ):
    'spline from vertices'

    if periodic is None:
      periodic = self.periodic

    if isinstance( degree, int ):
      degree = ( degree, ) * self.ndims

    if removedofs == None:
      removedofs = [None] * self.ndims
    else:
      assert len(removedofs) == self.ndims

    vertex_structure = numpy.array( 0 )
    dofcount = 1
    slices = []

    for idim in range( self.ndims ):
      periodic_i = idim in periodic
      n = self.structure.shape[idim]
      p = degree[idim]
      #k = knots[idim]

      if closed == False:
        neumann_i = (idim*2 in neumann and 1) | (idim*2+1 in neumann and 2)
        stdelems_i = element.PolyLine.spline( degree=p, nelems=n, periodic=periodic_i, neumann=neumann_i )
      elif closed == True:
        assert periodic==(), 'Periodic option not allowed for closed spline'
        assert neumann ==(), 'Neumann option not allowed for closed spline'
        stdelems_i = element.PolyLine.spline( degree=p, nelems=n, periodic=True )

      stdelems = stdelems[...,_] * stdelems_i if idim else stdelems_i

      nd = n + p
      numbers = numpy.arange( nd )
      if periodic_i and p > 0:
        overlap = p
        numbers[ -overlap: ] = numbers[ :overlap ]
        nd -= overlap
      remove = removedofs[idim]
      if remove is None:
        vertex_structure = vertex_structure[...,_] * nd + numbers
      else:
        mask = numpy.zeros( nd, dtype=bool )
        mask[numpy.array(remove)] = True
        nd -= mask.sum()
        numbers -= mask.cumsum()
        vertex_structure = vertex_structure[...,_] * nd + numbers
        vertex_structure[...,mask] = -1
      dofcount *= nd
      slices.append( [ slice(i,i+p+1) for i in range(n) ] )

    dofmap = {}
    funcmap = {}
    hasnone = False
    masks = numpy.zeros( [ self.comm.nprocs, dofcount ], dtype=bool )
    for item in numpy.broadcast( self.structure, stdelems, *numpy.ix_(*slices) ):
      elem = item[0]
      std = item[1]
      if elem is None:
        hasnone = True
      else:
        S = item[2:]
        dofs = vertex_structure[S].ravel()
        mask = dofs >= 0
        if mask.all():
          dofmap[ elem ] = mydofs = dofs
          funcmap[elem] = std
        elif mask.any():
          dofmap[ elem ] = mydofs = dofs[mask]
          funcmap[elem] = std, mask
        masks[ elem.subdom, mydofs ] = True

    domainmap = libmatrix.Map( self.comm, masks )

    if hasnone:
      raise NotImplementedError
      touched = numpy.zeros( dofcount, dtype=bool )
      for dofs in dofmap.itervalues():
        touched[ dofs ] = True
      renumber = touched.cumsum()
      dofcount = int(renumber[-1])
      dofmap = dict( ( elem, renumber[dofs]-1 ) for elem, dofs in dofmap.iteritems() )

    return function.function( funcmap, dofmap, dofcount, self.ndims, domainmap )

  @core.cache
  def discontfunc( self, degree ):
    'discontinuous shape functions'

    if isinstance( degree, int ):
      degree = ( degree, ) * self.ndims

    dofs = numpy.arange( numpy.product(degree+1) * len(self) ).reshape( len(self), -1 )
    dofmap = dict( zip( self, dofs ) )

    stdelem = util.product( element.PolyLine( element.PolyLine.bernstein_poly( d ) ) for d in degree )
    funcmap = dict( numpy.broadcast( self.structure, stdelem ) )

    return function.function( funcmap, dofmap, dofs.size, self.ndims )

  @core.cache
  def curvefreesplinefunc( self ):
    'spline from vertices'

    p = 2
    periodic = self.periodic

    vertex_structure = numpy.array( 0 )
    dofcount = 1
    slices = []

    for idim in range( self.ndims ):
      periodic_i = idim in periodic
      n = self.structure.shape[idim]

      stdelems_i = element.PolyLine.spline( degree=p, nelems=n, curvature=True )

      stdelems = stdelems[...,_] * stdelems_i if idim else stdelems_i

      nd = n + p - 2
      numbers = numpy.arange( nd )

      vertex_structure = vertex_structure[...,_] * nd + numbers

      dofcount *= nd

      myslice = [ slice(0,2) ]
      for i in range(n-2):
        myslice.append( slice(i,i+p+1) )
      myslice.append( slice(n-2,n) )

      slices.append( myslice )

    dofmap = {}
    for item in numpy.broadcast( self.structure, *numpy.ix_(*slices) ):
      elem = item[0]
      S = item[1:]
      dofmap[ elem ] = vertex_structure[S].ravel()

    dofaxis = function.DofMap( ElemMap(dofmap,self.ndims) )
    funcmap = dict( numpy.broadcast( self.structure, stdelems ) )

    return function.Function( dofaxis=dofaxis, stdmap=ElemMap(funcmap,self.ndims), igrad=0 )

  def linearfunc( self ):
    'linears'

    return self.splinefunc( degree=1 )

  @core.cache
  def stdfunc( self, degree ):
    'spline from vertices'

    if isinstance( degree, int ):
      degree = ( degree, ) * self.ndims

    vertex_structure = numpy.array( 0 )
    dofcount = 1
    slices = []

    stdelem = util.product( element.PolyLine( element.PolyLine.bernstein_poly( d ) ) for d in degree )

    for idim in range( self.ndims ):
      n = self.structure.shape[idim]
      p = degree[idim]

      nd = n * p + 1
      numbers = numpy.arange( nd )
      if idim in self.periodic:
        numbers[-1] = numbers[0]
        nd -= 1
      vertex_structure = vertex_structure[...,_] * nd + numbers
      dofcount *= nd
      slices.append( [ slice(p*i,p*i+p+1) for i in range(n) ] )

    dofmap = {}
    hasnone = False
    for item in numpy.broadcast( self.structure, *numpy.ix_(*slices) ):
      elem = item[0]
      if elem is None:
        hasnone = True
      else:
        S = item[1:]
        dofmap[ elem ] = vertex_structure[S].ravel()

    if hasnone:
      touched = numpy.zeros( dofcount, dtype=bool )
      for dofs in dofmap.itervalues():
        touched[ dofs ] = True
      renumber = touched.cumsum()
      dofcount = int(renumber[-1])
      dofmap = dict( ( elem, renumber[dofs]-1 ) for elem, dofs in dofmap.iteritems() )

    funcmap = dict( numpy.broadcast( self.structure, stdelem ) )
    return function.function( funcmap, dofmap, dofcount, self.ndims )

  def rectilinearfunc( self, gridvertices ):
    'rectilinear func'

    assert len( gridvertices ) == self.ndims
    vertex_structure = numpy.empty( map( len, gridvertices ) + [self.ndims] )
    for idim, ivertices in enumerate( gridvertices ):
      shape = [1,] * self.ndims
      shape[idim] = -1
      vertex_structure[...,idim] = numpy.asarray( ivertices ).reshape( shape )
    return self.linearfunc().dot( vertex_structure.reshape( -1, self.ndims ) )

  @core.weakcache
  def refine_nu( self, N ):
    'refine non-uniformly'

    N = tuple(N)
    assert len(N) == self.ndims
    structure = numpy.array( [ elem.children_by(N) if elem is not None else [None]*numpy.product(N) for elem in self.structure.flat ] )
    structure = structure.reshape( self.structure.shape + tuple(N) )
    structure = structure.transpose( sum( [ ( i, self.ndims+i ) for i in range(self.ndims) ], () ) )
    structure = structure.reshape( self.structure.shape * numpy.asarray(N) )
    refined = StructuredTopology( structure )
    refined.groups = { key: group.refine_nu( N ) for key, group in self.groups.items() }
    return refined

  @property
  def refined( self ):
    'refine entire topology'

    return self.refine_nu( [2]*self.ndims )

  def trim( self, levelset, maxrefine, lscheme='bezier3', finestscheme='uniform2', evalrefine=0, title='trimming', log=log ):
    'trim element along levelset'

    trimmedelems = [ elem.trim( levelset=levelset, maxrefine=maxrefine, lscheme=lscheme, finestscheme=finestscheme, evalrefine=evalrefine ) for elem in log.iterate( title, self.structure.ravel() ) ]
    trimmedstructure = numpy.array( trimmedelems ).reshape( self.structure.shape )
    return StructuredTopology( trimmedstructure, periodic=self.periodic )

  def __str__( self ):
    'string representation'

    return '%s(%s)' % ( self.__class__.__name__, 'x'.join(map(str,self.structure.shape)) )

  @property
  @core.cache
  def multiindex( self ):
    'Inverse map of self.structure: given an element find its location in the structure.'
    return dict( (self.structure[alpha], alpha) for alpha in numpy.ndindex( self.structure.shape ) )

  def neighbor( self, elem0, elem1 ):
    'Neighbor detection, returns codimension of interface, -1 for non-neighboring elements.'

    return elem0.neighbor( elem1 )

    # REPLACES:
    alpha0 = self.multiindex[elem0]
    alpha1 = self.multiindex[elem1]
    diff = numpy.array(alpha0) - numpy.array(alpha1)
    for i, shi in enumerate( self.structure.shape ):
      if diff[i] in (shi-1, 1-shi) and i in self.periodic:
        diff[i] = -numpy.sign( shi )
    if set(diff).issubset( (-1,0,1) ):
      return numpy.sum(numpy.abs(diff))
    return -1
    
class IndexedTopology( Topology ):
  'trimmed topology'
  
  def __init__( self, topo, elements ):
    'constructor'

    self.topo = topo
    self.elements = elements
    Topology.__init__( self, topo.ndims )

  def __iter__( self ):
    'iterate over elements'

    return iter( self.elements )

  def __len__( self ):
    'number of elements'

    return len(self.elements)

  def splinefunc( self, degree ):
    'create spline function space'

    raise NotImplementedError
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

  @property
  @core.weakcache
  def refined( self ):
    'refine all elements 2x'

    elements = [ child for elem in self.elements for child in elem.children if child is not None ]
    return IndexedTopology( self.topo.refined, elements )

  @property
  @core.cache
  def boundary( self ):
    'boundary'

    return self.topo.boundary

class UnstructuredTopology( Topology ):
  'externally defined topology'

  groups = {}

  def __init__( self, elements, ndims, namedfuncs={} ):
    'constructor'

    self.namedfuncs = namedfuncs
    self.elements = elements
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

  def linearfunc( self ):
    'linear func'

    return self.splinefunc( degree=1 )

  @property
  @core.weakcache
  def refined( self ):
    'refined (=refine(2))'

    try:
      linearfunc = self.linearfunc()
      (func,(dofaxis,)), = function.blocks( linearfunc )
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

      vertexdofs = dofaxis(elem,None)
      edgedofs = []
      if isinstance( elem, element.TriangularElement ):
        for i in range(3):
          j = (i+1)%3
          try:
            edgedof = edges.pop(( vertexdofs[i], vertexdofs[j] ))
          except KeyError:
            edgedof = edges[( vertexdofs[j], vertexdofs[i] )] = ndofs
            ndofs += 1
          edgedofs.append( edgedof )
        nmap[ children[0] ] = numpy.array([ edgedofs[2], edgedofs[1], vertexdofs[2] ])
        nmap[ children[1] ] = numpy.array([ edgedofs[0], vertexdofs[1], edgedofs[1] ])
        nmap[ children[2] ] = numpy.array([ vertexdofs[0], edgedofs[0], edgedofs[2] ])
        nmap[ children[3] ] = numpy.array([ edgedofs[1], edgedofs[2], edgedofs[0] ])
      else:
        dofaxis = None

    #print 'boundary:', edges

    if dofaxis:
      fmap = dict.fromkeys( elements, element.PolyTriangle(1) )
      linearfunc = function.function( fmap, nmap, ndofs, self.ndims )
      namedfuncs = { 'spline2': linearfunc }
    else:
      namedfuncs = {}

    return UnstructuredTopology( elements, ndims=2, namedfuncs=namedfuncs )

class HierarchicalTopology( Topology ):
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

  @property
  @core.cache
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
        if celem in topoelems:
          belemset.add( belem )
      allbelems.extend( belemset )
      for btag, belems in topo.boundary.groups.iteritems():
        bgroups.setdefault( btag, [] ).extend( belemset.intersection(belems) )
      topo = topo.refined # proceed to next level
    boundary = UnstructuredTopology( allbelems, ndims=self.ndims-1 )
    boundary.groups = dict( ( tag, UnstructuredTopology( group, ndims=self.ndims-1 ) ) for tag, group in bgroups.items() )
    return boundary

  @property
  @core.cache
  def interfaces( self ):
    'interface elements & groups'

    assert hasattr( self.basetopo, 'interfaces' )
    allinterfaces = []
    topo = self.basetopo # topology to examine in next level refinement
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
      supported = numpy.ones( funcsp.shape[0], dtype=bool ) # True if dof is contained in topoelems or parentelems
      touchtopo = numpy.zeros( funcsp.shape[0], dtype=bool ) # True if dof touches at least one topoelem
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
  
      keep = numpy.logical_and( supported, touchtopo ) # THE refinement law

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
        dofmap[elem] = numpy.array(newdofs) # add result to IEN mapping of new function object
  
      ndofs += keep.sum() # update total number of dofs
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

  def linearfunc( self, *args, **kwargs ):
    return self._funcspace( lambda topo: topo.linearfunc( *args, **kwargs ) )

  def splinefunc( self, *args, **kwargs ):
    return self._funcspace( lambda topo: topo.splinefunc( *args, **kwargs ) )

def glue( master, slave, coords, tol=1.e-10, verbose=False ):
  'Glue topologies along boundary group __glue__.'
  log.context('glue')

  # Checks on input
  assert master.boundary.groups.has_key( '__glue__' ) and \
          slave.boundary.groups.has_key( '__glue__' ), 'Must identify glue boundary first.'
  assert len(master.boundary.groups['__glue__']) == \
          len(slave.boundary.groups['__glue__']), 'Minimum requirement is that cardinality is equal.'
  assert master.ndims == 2 and slave.ndims == 2, '1D boundaries for now.' # see dists computation and update_vertices

  # Handy local function definitions
  def replace_elem( index, elem ):
    'Put elem in new at index with slave as parent.'
    assert isinstance( elem, element.QuadElement ), 'glue() is very restrictive: only QuadElement meshes for now.'
    new_elem = lambda parent: element.QuadElement( elem.ndims, elem.vertices, parent=parent ) 
    if isinstance( slave, StructuredTopology ): # slave is of the same type as new
      ndindex = numpy.unravel_index( index, slave.structure.shape )
      new.elements[index] = new_elem( (slave.structure[ndindex], element.IdentityTransformation(elem.ndims)) )
    else:
      new.elements[index] = new_elem( (slave.elements[index], element.IdentityTransformation(elem.ndims)) )

  def update_vertices( edge, vertices ):
    'Update the new element. Tedious construction to avoid updating only a copy.'
    # Find copy of parent
    for index, elem in enumerate( new ):
      n0, n1, o0, o1 = vertices+edge.vertices # new and old vertex labels
      included = lambda vertex: set( elem.vertices ).issuperset( (vertex,) )
      if (included(n0) or included(o0)) and \
         (included(n1) or included(o1)): break # elem contains edge
    else:
      raise ValueError( 'edge not in elem' )

    # Create copy of vertices
    elem.vertices = list(elem.vertices) # elem.vertices = list(elem.vertices)
    for new_vertex, old_vertex in zip( vertices, edge.vertices ):
      try:
        j = elem.vertices.index( old_vertex )
        elem.vertices[j] = new_vertex
      except:
        pass # vertex has been updated via a previous edge
    elem.vertices = tuple(elem.vertices)

    # Place new element in copy
    replace_elem( index, elem )

  def elem_list( topo ):
    if isinstance( topo, StructuredTopology ): return list( topo.structure.flat )
    return topo.elements

  # 0. Create copy of slave, so that slave is not altered
  new = UnstructuredTopology( elem_list( slave ), slave.ndims )
  for index, elem in enumerate( slave ):
    replace_elem( index, elem )

  # 1. Determine vertex locations
  master_vertex_locations = {}
  master_coords, slave_coords = coords if isinstance( coords, tuple ) else 2*(coords,)
  for elem in master.boundary.groups['__glue__']:
    master_vertex_locations[elem] = master_coords( elem, 'bezier2' )
  slave_vertex_locations = {}
  for elem in slave.boundary.groups['__glue__']:
    slave_vertex_locations[elem] = slave_coords( elem, 'bezier2' )

  # 2. Update vertices of elements in new topology
  log.info( 'pairing elements [%i]' % len(master.boundary.groups['__glue__']) )
  for master_elem, master_locs in master_vertex_locations.iteritems():
    meshwidth = numpy.linalg.norm( numpy.diff( master_locs, axis=0 ) )
    assert meshwidth > tol, 'tol. (%.2e) > element size (%.2e)' % (tol, meshwidth)
    if verbose: pos = {}
    for slave_elem, slave_locs in slave_vertex_locations.iteritems():
      dists = (numpy.linalg.norm( master_locs-slave_locs ),
               numpy.linalg.norm( master_locs-slave_locs[::-1] ))
      if verbose:
        key = tuple(slave_locs[:,:2].T.flatten())
        pos[key] = min(*dists)
      if min(*dists) < tol:
        slave_vertex_locations.pop( slave_elem )
        new_vertices = master_elem.vertices if dists[0] < tol else master_elem.vertices[::-1]
        update_vertices( slave_elem, new_vertices )
        break # don't check if a second element can be paired.
    else:
      if verbose:
        from matplotlib import pyplot
        pyplot.clf()
        pyplot.plot( master_locs[:,0], master_locs[:,1], '.-', label='master' )
        for verts, dist in pos.iteritems():
          pyplot.plot( verts[:2], verts[2:], label='%.3f'%dist )
          title = min( locals().get('title',dist), dist )
        pyplot.legend()
        pyplot.axis('equal')
        pyplot.title('min dist: %.3e'%title)
        it = locals().get('it',-1) + 1
        name = 'glue%i.jpg'%it
        pyplot.savefig(prop.dumpdir+name)
        log.path(name)
      raise AssertionError( 'Could not pair master element: %s (maybe tol is set too low?)' % master_elem )
  assert not len( slave_vertex_locations ), 'Could not pair slave elements: ' + ', '.join(
    [elem.__repr__() for elem in slave_vertex_locations.iterkeys()] )

  # 3. Generate union topo
  union = UnstructuredTopology( elem_list(master) + elem_list(new), master.ndims )
  log.info( 'created union topo [%i]' % len(union) )
  # Update interior groups
  union.groups.update( dict(
      [(key+'.m',val) for key, val in master.groups.iteritems()] +
      [(key+'.s',val) for key, val in slave.groups.iteritems()] ) )
  # Update boundary groups
  # 1st level boundaries, not e.g. boundary['top'].boundary, groups structure needs reorganization anyways
  union.groups.update( dict( [(key+'.mb',val) for key, val in master.boundary.groups.iteritems()] ) )
  union.groups.update( dict( [(key+'.sb',val) for key, val in slave.boundary.groups.iteritems()] ) )
  # Glue-related groups added last (overwrites slave groups in case warning is given above)
  union.groups.update( {'__master__':elem_list(master),
                         '__slave__':elem_list(new),
                    '__master_bnd__':master.boundary,
                     '__slave_bnd__':slave.boundary} )
  union.groups['__glued__'] = union.groups.pop('__glue__.mb')
  union.groups.pop('__glue__.sb')
  return union

# vim:shiftwidth=2:foldmethod=indent:foldnestmax=2
