# -*- coding: utf8 -*-
#
# Module TOPOLOGY
#
# Part of Nutils: open source numerical utilities for Python. Jointly developed
# by HvZ Computational Engineering, TU/e Multiscale Engineering Fluid Dynamics,
# and others. More info at http://nutils.org <info@nutils.org>. (c) 2014

"""
The topology module defines the topology objects, notably the
:class:`StructuredTopology` and :class:`UnstructuredTopology`. Maintaining
strict separation of topological and geometrical information, the topology
represents a set of elements and their interconnectivity, boundaries,
refinements, subtopologies etc, but not their positioning in physical space. The
dimension of the topology represents the dimension of its elements, not that of
the the space they are embedded in.

The primary role of topologies is to form a domain for :mod:`nutils.function`
objects, like the geometry function and function bases for analysis, as well as
provide tools for their construction. It also offers methods for integration and
sampling, thus providing a high level interface to operations otherwise written
out in element loops. For lower level operations topologies can be used as
:mod:`nutils.element` iterators.
"""

from . import element, function, util, numpy, parallel, matrix, log, core, numeric, cache, _
import warnings, itertools

class Topology( object ):
  'topology base class'

  def __init__( self, ndims ):
    'constructor'

    self.ndims = ndims

  def refined_by( self, refine ):
    'create refined space by refining dofs in existing one'

    refine = set( refine )
    refined = []
    for elem in self:
      if elem in refine:
        refine.remove( elem )
        refined.extend( elem.children )
      else:
        refined.append( elem )
      while elem.parent: # only for argument checking:
        elem, trans = elem.parent
        refine.discard( elem )

    assert not refine, 'not all refinement elements were found: %s' % '\n '.join( str(e) for e in refine )
    return HierarchicalTopology( self, refined )

  def stdfunc( self, degree ):
    'spline from vertices'

    if isinstance( degree, int ):
      degree = ( degree, ) * self.ndims

    assert all( n == 1 for n in degree ) # for now!

    dofmap = { n: i for i, n in enumerate( sorted( set( n for elem in self for n in elem.vertices ) ) ) }
    fmap = dict.fromkeys( self, element.PolyTriangle(1) )
    nmap = { elem: numpy.array([ dofmap[n] for n in elem.vertices ]) for elem in self }
    return function.function( fmap=fmap, nmap=nmap, ndofs=len(dofmap), ndims=2 )

  @cache.property
  def simplex( self ):
    simplices = [ simplex for elem in self for simplex in elem.simplices ]
    return UnstructuredTopology( ndims=self.ndims, elements=simplices )

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

  @log.title
  def elem_eval( self, funcs, ischeme, separate=False ):
    'element-wise evaluation'

    single_arg = not isinstance(funcs,(tuple,list))
    if single_arg:
      funcs = funcs,

    slices = []
    pointshape = function.PointShape().compiled()
    npoints = 0
    separators = []
    __log__ = log.iter( 'elem', self )
    for elem in __log__:
      np, = pointshape.eval( elem, ischeme )
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
      func = function.asarray( func )
      retval = parallel.shzeros( (npoints,)+func.shape, dtype=func.dtype )
      if separate:
        retval[separators] = numpy.nan
      if function._isfunc( func ):
        for f, ind in function.blocks( func ):
          idata.append( function.Tuple( [ ifunc, function.Tuple(ind), f ] ) )
      else:
        idata.append( function.Tuple( [ ifunc, (), func ] ) )
      retvals.append( retval )
    idata = function.Tuple( idata ).compiled()

    __log__ = log.enumerate( 'elem', self )
    for ielem, elem in parallel.pariter( __log__ ):
      s = slices[ielem],
      for ifunc, index, data in idata.eval( elem, ischeme ):
        retvals[ifunc][s+index] = data

    log.debug( 'cache', idata.cache.summary() )
    log.info( 'created', ', '.join( '%s(%s)' % ( retval.__class__.__name__, ','.join(map(str,retval.shape)) ) for retval in retvals ) )
    if single_arg:
      retvals, = retvals

    return retvals

  @log.title
  def elem_mean( self, funcs, geometry, ischeme ):
    'element-wise integration'

    single_arg = not isinstance(funcs,(tuple,list))
    if single_arg:
      funcs = funcs,

    retvals = []
    #iweights = geometry.iweights( self.ndims )
    iweights = function.iwscale( geometry, self.ndims ) * function.IWeights()
    idata = [ iweights ]
    for func in funcs:
      func = function.asarray( func )
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

    log.debug( 'cache', idata.cache.summary() )
    log.info( 'created', ', '.join( '%s(%s)' % ( retval.__class__.__name__, ','.join(map(str,retval.shape)) ) for retval in retvals ) )
    if single_arg:
      retvals, = retvals

    return retvals

  @log.title
  def grid_eval( self, funcs, geometry, C ):
    'evaluate grid points'

    single_arg = not isinstance(funcs,(tuple,list))
    if single_arg:
      funcs = funcs,

    C = numpy.asarray( C )
    assert C.shape[0] == self.ndims
    shape = C.shape
    C = C.reshape( self.ndims, -1 )

    funcs = [ function.asarray(func) for func in funcs ]
    retvals = [ numpy.empty( C.shape[1:] + func.shape ) for func in funcs ]
    for retval in retvals:
      retval[:] = numpy.nan

    data = function.Tuple([ function.Tuple([ func, retval ]) for func, retval in zip( funcs, retvals ) ]).compiled()

    __log__ = log.iter( 'elem', self )
    for elem in __log__:
      points, selection = geometry.find( elem, C.T )
      if selection is not None:
        for func, retval in data( elem, points ):
          retval[selection] = func

    retvals = [ retval.reshape( shape[1:] + func.shape ) for func, retval in zip( funcs, retvals ) ]

    log.debug( 'cache', data.cache.summary() )
    log.info( 'created', ', '.join( '%s(%s)' % ( retval.__class__.__name__, ','.join(map(str,retval.shape)) ) for retval in retvals ) )
    if single_arg:
      retvals, = retvals

    return retvals

  @log.title
  def build_graph( self, func ):
    'get matrix sparsity'

    nrows, ncols = func.shape
    graph = [ [] for irow in range(nrows) ]
    IJ = function.Tuple([ function.Tuple(ind) for f, ind in function.blocks( func ) ]).compiled()

    __log__ = log.iter( 'elem', self )
    for elem in __log__:
      for I, J in IJ.eval( elem, None ):
        for i in I:
          graph[i].append(J)

    __log__ = log.enumerate( 'dof', graph )
    for irow, g in __log__:
      # release memory as we go
      if g: graph[irow] = numpy.unique( numpy.concatenate( g ) )

    return graph

  @log.title
  def integrate( self, funcs, ischeme, geometry=None, iweights=None, force_dense=False ):
    'integrate'

    single_arg = not isinstance(funcs,(list,tuple))
    if single_arg:
      funcs = funcs,

    if iweights is None:
      assert geometry is not None, 'conflicting arguments geometry and iweights'
      iweights = function.iwscale( geometry, self.ndims ) * function.IWeights()
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

    __log__ = log.iter( 'elem', self )
    for elem in parallel.pariter( __log__ ):
      for ifunc, lock, index, data in idata.eval( elem, ischeme ):
        with lock:
          retvals[ifunc][index] += data

    log.debug( 'cache', idata.cache.summary() )
    log.info( 'created', ', '.join( '%s(%s)' % ( retval.__class__.__name__, ','.join(map(str,retval.shape)) ) for retval in retvals ) )
    if single_arg:
      retvals, = retvals

    return retvals

  @log.title
  def integrate_symm( self, funcs, ischeme, geometry=None, iweights=None, force_dense=False ):
    'integrate a symmetric integrand on a product domain' # TODO: find a proper home for this

    single_arg = not isinstance(funcs,list)
    if single_arg:
      funcs = funcs,

    if iweights is None:
      assert geometry is not None, 'conflicting arguments geometry and iweights'
      iweights = function.iwscale( geometry, self.ndims ) * function.IWeights()
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

    __log__ = log.iter( 'elem', self )
    for elem in parallel.pariter( __log__ ):
      assert isinstance( elem.reference, element.ProductReference )
      (elem1,trans1), (elem2,trans2) = elem.interface
      compare_elem = cmp( elem1, elem2 )
      if compare_elem < 0:
        continue
      for ifunc, lock, index, data in idata.eval( elem, ischeme ):
        with lock:
          retvals[ifunc][index] += data
          if compare_elem > 0:
            retvals[ifunc][index[::-1]] += data.T

    log.debug( 'cache', idata.cache.summary() )
    log.info( 'created', ', '.join( '%s(%s)' % ( retval.__class__.__name__, ','.join(map(str,retval.shape)) ) for retval in retvals ) )
    if single_arg:
      retvals, = retvals

    return retvals

  def projection( self, fun, onto, geometry, **kwargs ):
    'project and return as function'

    weights = self.project( fun, onto, geometry, **kwargs )
    return onto.dot( weights )

  @log.title
  def project( self, fun, onto, geometry, tol=0, ischeme=None, droptol=1e-8, exact_boundaries=False, constrain=None, verify=None, maxiter=0, ptype='lsqr' ):
    'L2 projection of function onto function space'

    log.debug( 'projection type:', ptype )

    if exact_boundaries:
      assert constrain is None
      constrain = self.boundary.project( fun, onto, geometry, title='boundaries', ischeme=ischeme, tol=tol, droptol=droptol, ptype=ptype )
    elif constrain is None:
      constrain = util.NanVec( onto.shape[0] )
    else:
      assert isinstance( constrain, util.NanVec )
      assert constrain.shape == onto.shape[:1]

    if ptype == 'lsqr':
      assert ischeme is not None, 'please specify an integration scheme for lsqr-projection'
      if len( onto.shape ) == 1:
        Afun = function.outer( onto )
        bfun = onto * fun
      elif len( onto.shape ) == 2:
        Afun = function.outer( onto ).sum( 2 )
        bfun = function.sum( onto * fun )
      else:
        raise Exception
      A, b = self.integrate( [Afun,bfun], geometry=geometry, ischeme=ischeme, title='building system' )
      N = A.rowsupp(droptol)
      if numpy.all( b == 0 ):
        constrain[~constrain.where&N] = 0
      else:
        solvecons = constrain.copy()
        solvecons[~(constrain.where|N)] = 0
        u = A.solve( b, solvecons, tol=tol, symmetric=True, maxiter=maxiter )
        constrain[N] = u[N]

    elif ptype == 'convolute':
      assert ischeme is not None, 'please specify an integration scheme for convolute-projection'
      if len( onto.shape ) == 1:
        ufun = onto * fun
        afun = onto
      elif len( onto.shape ) == 2:
        ufun = function.sum( onto * fun )
        afun = function.norm2( onto )
      else:
        raise Exception
      u, scale = self.integrate( [ ufun, afun ], geometry=geometry, ischeme=ischeme )
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
      fun = function.asarray( fun )
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

    errfun2 = ( onto.dot( constrain | 0 ) - fun )**2
    if errfun2.ndim == 1:
      errfun2 = errfun2.sum()
    error2, area = self.integrate( [ errfun2, 1 ], geometry=geometry, ischeme=ischeme or 'gauss2' )
    avg_error = numpy.sqrt(error2) / area

    numcons = constrain.where.sum()
    if verify is not None:
      assert numcons == verify, 'number of constraints does not meet expectation: %d != %d' % ( numcons, verify )

    log.info( 'constrained %d/%d dofs, error %.2e/area' % ( numcons, constrain.size, avg_error ) )

    return constrain

  @log.title
  def refinedfunc( self, dofaxis, refine, degree ):
    'create refined space by refining dofs in existing one'

    warnings.warn( 'refinedfunc is replaced by refined_by + splinefunc; this function will be removed in future' % ischeme, DeprecationWarning )

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
      __log__ = log.iter( 'elem', topo )
      for elem in __log__: # loop over remaining elements in refinement level 'nrefine'
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
    __log__ = log.range( 'level', nrefine )
    for irefine in __log__:
  
      funcsp = topo.splinefunc( degree ) # shape functions for level irefine
      (func,(dofaxis,)), = function.blocks( funcsp ) # separate elem-local funcs and global placement index
  
      supported = numpy.ones( funcsp.shape[0], dtype=bool ) # True if dof is contained in topoelems or parentelems
      touchtopo = numpy.zeros( funcsp.shape[0], dtype=bool ) # True if dof touches at least one topoelem
      myelems = [] # all top-level or parent elements in level irefine

      __log__ = log.iter( 'element', dofaxis.dofmap.items() )
      for elem, idofs in __log__:
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

  @log.title
  def get_simplices( self, maxrefine ):
    'Getting simplices'

    return [ simplex for elem in self for simplex in elem.get_simplices( maxrefine ) ]

  @log.title
  def get_trimmededges( self, maxrefine ):
    'Getting trimmed edges'

    return [ trimmededge for elem in self for trimmededge in elem.get_trimmededges( maxrefine ) ]

class StructuredTopology( Topology ):
  'structured topology'

  def __init__( self, structure, periodic=() ):
    'constructor'

    structure = numpy.asarray(structure)
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
      periodic = [ d - (d>idim) for d in self.periodic if d != idim ] # TODO check that dimensions are correct for ndim > 2
      boundaries.append( StructuredTopology( belems, periodic=periodic ) )

    if self.ndims == 2:
      structure = numpy.concatenate([ boundaries[i].structure for i in [0,2,1,3] ])
      topo = StructuredTopology( structure, periodic=[0] )
    else:
      allbelems = [ belem for boundary in boundaries for belem in boundary.structure.flat if belem is not None ]
      topo = UnstructuredTopology( allbelems, ndims=self.ndims-1 )

    topo.groups = dict( zip( ( 'left', 'right', 'bottom', 'top', 'front', 'back' ), boundaries ) )
    return topo

  @cache.property
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
          dofmap[ elem ] = dofs
          funcmap[elem] = std
        elif mask.any():
          dofmap[ elem ] = dofs[mask]
          funcmap[elem] = std, mask

    if hasnone:
      touched = numpy.zeros( dofcount, dtype=bool )
      for dofs in dofmap.itervalues():
        touched[ dofs ] = True
      renumber = touched.cumsum()
      dofcount = int(renumber[-1])
      dofmap = dict( ( elem, renumber[dofs]-1 ) for elem, dofs in dofmap.iteritems() )

    return function.function( funcmap, dofmap, dofcount, self.ndims )

  def discontfunc( self, degree ):
    'discontinuous shape functions'

    if isinstance( degree, int ):
      degree = ( degree, ) * self.ndims

    dofs = numpy.arange( numpy.product(numpy.array(degree)+1) * len(self) ).reshape( len(self), -1 )
    dofmap = dict( zip( self, dofs ) )

    stdelem = util.product( element.PolyLine( element.PolyLine.bernstein_poly( d ) ) for d in degree )
    funcmap = dict( numpy.broadcast( self.structure, stdelem ) )

    return function.function( funcmap, dofmap, dofs.size, self.ndims )

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

  def stdfunc( self, degree, removedofs=None ):
    'spline from vertices'

    if isinstance( degree, int ):
      degree = ( degree, ) * self.ndims

    if removedofs == None:
      removedofs = [None] * self.ndims
    else:
      assert len(removedofs) == self.ndims

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
      slices.append( [ slice(p*i,p*i+p+1) for i in range(n) ] )

    dofmap = {}
    funcmap = {}
    hasnone = False
    for item in numpy.broadcast( self.structure, *numpy.ix_(*slices) ):
      elem = item[0]
      if elem is None:
        hasnone = True
      else:
        S = item[1:]
        dofs = vertex_structure[S].ravel()
        mask = dofs >= 0
        if mask.all():
          dofmap[ elem ] = dofs
          funcmap[elem] = stdelem
        elif mask.any():
          dofmap[ elem ] = dofs[mask]
          funcmap[elem] = stdelem, mask

    if hasnone:
      touched = numpy.zeros( dofcount, dtype=bool )
      for dofs in dofmap.itervalues():
        touched[ dofs ] = True
      renumber = touched.cumsum()
      dofcount = int(renumber[-1])
      dofmap = dict( ( elem, renumber[dofs]-1 ) for elem, dofs in dofmap.iteritems() )

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

  @property
  def refined( self ):
    'refine non-uniformly'

    structure = numpy.array( [ elem.children if elem is not None else [None]*(2**self.ndims) for elem in self.structure.flat ] )
    structure = structure.reshape( self.structure.shape + (2,)*self.ndims )
    structure = structure.transpose( sum( [ ( i, self.ndims+i ) for i in range(self.ndims) ], () ) )
    structure = structure.reshape( numpy.array(self.structure.shape) * 2 )
    refined = StructuredTopology( structure )
    refined.groups = { key: group.refined for key, group in self.groups.items() }
    return refined

  def trim( self, levelset, maxrefine, lscheme='bezier3', finestscheme='uniform2', evalrefine=0, title='trimming', log=log ):
    'trim element along levelset'

    __log__ = log.iter( title, self.structure.ravel() )
    levelset = function.ascompiled( levelset )
    trimmedelems = [ elem.trim( levelset=levelset, maxrefine=maxrefine ) for elem in __log__ ]
    trimmedstructure = numpy.array( trimmedelems ).reshape( self.structure.shape )
    return StructuredTopology( trimmedstructure, periodic=self.periodic )

  def __str__( self ):
    'string representation'

    return '%s(%s)' % ( self.__class__.__name__, 'x'.join(map(str,self.structure.shape)) )

  @cache.property
  def multiindex( self ):
    'Inverse map of self.structure: given an element find its location in the structure.'
    return dict( (self.structure[alpha], alpha) for alpha in numpy.ndindex( self.structure.shape ) )

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

  @cache.property
  def refined( self ):
    'refine all elements 2x'

    elements = [ child for elem in self.elements for child in elem.children if child is not None ]
    return IndexedTopology( self.topo.refined, elements )

  @cache.property
  def boundary( self ):
    'boundary'

    return self.topo.boundary

class UnstructuredTopology( Topology ):
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

  def linearfunc( self ):
    'linear func'

    return self.splinefunc( degree=1 )

  def bubblefunc( self ):
    'linear func + bubble'

    return self.namedfuncs[ 'bubble1' ]

  @cache.property
  def refined( self ):
    'refined (=refine(2))'

    try:
      linearfunc = self.linearfunc()
      (func,(dofaxis,)), = function.blocks( linearfunc )
      dofaxis = dofaxis.compiled()
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

      vertexdofs = dofaxis.eval(elem,None)
      edgedofs = []
      if isinstance( elem.reference, element.TriangularReference ):
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

    if dofaxis:
      fmap = dict.fromkeys( elements, element.PolyTriangle(1) )
      linearfunc = function.function( fmap, nmap, ndofs, self.ndims )
      namedfuncs = { 'spline1': linearfunc }
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

  @cache.property
  def boundary( self ):
    'boundary elements & groups'

    assert hasattr( self.basetopo, 'boundary' )
    allbelems = []
    bgroups = {}
    topo = self.basetopo # topology to examine in next level refinement
    elems = set( self )
    while elems:
      belemset = set()
      myelems = elems.intersection( topo )
      for belem in topo.boundary:
        celem, transform = belem.context
        if celem in myelems:
          belemset.add( belem )
      allbelems.extend( belemset )
      for btag, belems in topo.boundary.groups.iteritems():
        bgroups.setdefault( btag, [] ).extend( belemset.intersection(belems) )
      topo = topo.refined # proceed to next level
      elems -= myelems
    boundary = UnstructuredTopology( allbelems, ndims=self.ndims-1 )
    boundary.groups = dict( ( tag, UnstructuredTopology( group, ndims=self.ndims-1 ) ) for tag, group in bgroups.items() )
    return boundary

  @cache.property
  def interfaces( self ):
    'interface elements & groups'

    assert hasattr( self.basetopo, 'interfaces' )
    allinterfaces = []
    topo = self.basetopo # topology to examine in next level refinement
    elems = set( self )
    while elems:
      myelems = elems.intersection( topo )
      for ielem in topo.interfaces:
        (celem1,transform1), (celem2,transform2) = ielem.interface
        if celem1 in myelems:
          while True:
            if celem2 in self.elements:
              allinterfaces.append( ielem )
              break
            if not celem2.parent:
              break
            celem2, transform2 = celem2.parent
        elif celem2 in myelems:
          while True:
            if celem1 in self.elements:
              allinterfaces.append( ielem )
              break
            if not celem1.parent:
              break
            celem1, transform1 = celem1.parent
      topo = topo.refined # proceed to next level
      elems -= myelems
    return UnstructuredTopology( allinterfaces, ndims=self.ndims-1 )

  def _funcspace( self, mkspace ):

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

  @log.title
  def stdfunc( self, *args, **kwargs ):
    return self._funcspace( lambda topo: topo.stdfunc( *args, **kwargs ) )

  @log.title
  def linearfunc( self, *args, **kwargs ):
    return self._funcspace( lambda topo: topo.linearfunc( *args, **kwargs ) )

  @log.title
  def splinefunc( self, *args, **kwargs ):
    return self._funcspace( lambda topo: topo.splinefunc( *args, **kwargs ) )

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

@log.title
def glue( master, slave, geometry, tol=1.e-10, verbose=False ):
  'Glue topologies along boundary group __glue__.'

  gluekey = '__glue__'

  # Checks on input
  assert gluekey in master.boundary.groups and \
         gluekey in slave.boundary.groups, 'Must identify glue boundary first.'
  assert len(master.boundary[gluekey]) == \
          len(slave.boundary[gluekey]), 'Minimum requirement is that cardinality is equal.'
  assert master.ndims == 2 and slave.ndims == 2, '1D boundaries for now.' # see dists computation and update_vertices

  if isinstance( geometry, tuple ):
    master_geom, slave_geom = map( function.ascompiled, geometry )
  else:
    master_geom = slave_geom = function.ascompiled( geometry )

  vtxmap = {} # THE old vertex -> nex vertex mapping

  log.info( 'pairing elements [%i]' % len(master.boundary[gluekey]) )
  slave_vertex_locations = { slave_elem:
    slave_geom.eval( slave_elem, 'bezier2' ) for slave_elem in slave.boundary[gluekey] }
  for master_elem in master.boundary[gluekey]:
    master_locs = master_geom.eval( master_elem, 'bezier2' )
    meshwidth = numpy.linalg.norm( numpy.diff( master_locs, axis=0 ) )
    assert meshwidth > tol, 'tol. (%.2e) > element size (%.2e)' % (tol, meshwidth)
    for slave_elem, slave_locs in slave_vertex_locations.iteritems():
      dists = (numpy.linalg.norm( master_locs-slave_locs ),
               numpy.linalg.norm( master_locs-slave_locs[::-1] ))
      if min(*dists) < tol:
        break # don't check if a second element can be paired.
    else:
      if verbose:
        from matplotlib import pyplot
        pyplot.clf()
        pyplot.plot( master_locs[:,0], master_locs[:,1], '.-', label='master' )
        mindist = numpy.inf
        for slave_elem, slave_locs in slave_vertex_locations.iteritems():
          verts = slave_locs[:,:2].T.flatten()
          pyplot.plot( verts[:2], verts[2:], label='%.3f'%dist )
          mindist = min( mindist,
            numpy.linalg.norm( master_locs-slave_locs ),
            numpy.linalg.norm( master_locs-slave_locs[::-1] ) )
        pyplot.legend()
        pyplot.axis('equal')
        pyplot.title('min dist: %.3e'%mindist)
        it = locals().get('it',-1) + 1
        name = 'glue%i.jpg'%it
        pyplot.savefig( core.getprop( 'dumpdir' )+name )
        log.path(name)
      raise AssertionError( 'Could not pair master element: %s (maybe tol is set too low?)' % master_elem )
    slave_vertex_locations.pop( slave_elem )
    new_vertices = master_elem.vertices if dists[0] < tol \
              else reversed( master_elem.vertices )

    for oldvtx, newvtx in zip( slave_elem.vertices, new_vertices ):
      assert vtxmap.setdefault( oldvtx, newvtx ) == newvtx, 'conflicting vertex info'

  assert not slave_vertex_locations, 'Could not pair slave elements: %s' % slave_vertex_locations.keys()

  # we can forget everything now and continue with the vtxmap

  emap = {} # elem->newelem map
  for belem in slave.boundary:
    if not any( vtx in vtxmap for vtx in belem.vertices ):
      continue
    emap[belem] = element.QuadElement( belem.ndims,
      vertices=[ vtxmap.get(vtx,vtx) for vtx in belem.vertices ],
      parent=(belem,element.IdentityTransformation(belem.ndims)) )
    elem, trans = belem.context
    emap[elem] = element.QuadElement( elem.ndims,
      vertices=[ vtxmap.get(vtx,vtx) for vtx in elem.vertices ],
      parent=(elem,element.IdentityTransformation(elem.ndims)) )

  _wrapelem = lambda elem: emap.get(elem,elem)
  def _wraptopo( topo ):
    elems = map( _wrapelem, topo )
    return UnstructuredTopology( elems, ndims=topo.ndims ) if not isinstance( topo, UnstructuredTopology ) \
      else StructuredTopology( numpy.asarray(elems).reshape(slave.structure.shape) )

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
