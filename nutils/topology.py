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

from . import element, function, util, numpy, parallel, matrix, log, core, numeric, cache, rational, transform, _
import warnings, itertools

class Topology( object ):
  'topology base class'

  def __init__( self, elements ):
    'constructor'

    self.elements = tuple(elements)
    self.ndims = self.elements[0].ndims # assume all equal

  def __len__( self ):
    return len( self.elements )

  def __iter__( self ):
    return iter( self.elements )

  def stdfunc( self, degree ):
    'spline from vertices'

    assert degree == 1 # for now!
    dofmap = {}
    fmap = {}
    nmap = {}
    for elem in self:
      dofs = numpy.empty( elem.nverts, dtype=int )
      for i, v in enumerate( elem.vertices ):
        dof = dofmap.get(v)
        if dof is None:
          dof = len(dofmap)
          dofmap[v] = dof
        dofs[i] = dof
      stdfunc = elem.reference.stdfunc(1)
      assert stdfunc.nshapes == elem.nverts
      fmap[elem.transform] = stdfunc
      nmap[elem.transform] = dofs
    return function.function( fmap=fmap, nmap=nmap, ndofs=len(dofmap), ndims=self.ndims )

  def splinefunc( self, degree ):

    assert degree == 1
    return self.stdfunc( degree )

  def discontfunc( self, degree ):
    'discontinuous shape functions'

    assert isinstance( degree, int ) and degree >= 0
    fmap = {}
    nmap = {}
    ndofs = 0
    for elem in self:
      stdfunc = elem.reference.stdfunc(degree)
      fmap[elem.transform] = stdfunc
      nmap[elem.transform] = ndofs + numpy.arange(stdfunc.nshapes)
      ndofs += stdfunc.nshapes
    return function.function( fmap=fmap, nmap=nmap, ndofs=ndofs, ndims=self.ndims )

  @cache.property
  def simplex( self ):
    simplices = [ simplex for elem in self for simplex in elem.simplices ]
    return Topology( simplices )

  def refined_by( self, refine ):
    'create refined space by refining dofs in existing one'

    refine = set( refine )
    refined = []
    for elem in self:
      if elem.transform in refine:
        refine.remove( elem.transform )
        refined.extend( elem.children )
      else:
        refined.append( elem )
      for trans1, trans2 in elem.transform:
        refine.discard( trans2 )
    assert not refine, 'not all refinement elements were found: %s' % '\n '.join( str(e) for e in refine )
    return HierarchicalTopology( self, refined )

  def __add__( self, other ):
    'add topologies'

    if self is other:
      return self
    assert self.ndims == other.ndims
    return Topology( set(self) | set(other) )

  def __sub__( self, other ):
    'add topologies'

    if self is other:
      return self
    assert self.ndims == other.ndims
    return Topology( set(self) - set(other) )

  def __mul__( self, other ):
    'element products'

    quad = element.SimplexReference(1)**2
    ndims = self.ndims + other.ndims
    eye = numpy.eye( ndims, dtype=int )
    self_trans = transform.updim(eye[:self.ndims],1)
    other_trans = transform.updim(eye[self.ndims:],1)

    if any( elem.reference != quad for elem in self ) or any( elem.reference != quad for elem in other ):
      return Topology( element.Element( elem1.reference * elem2.reference, self_trans >> elem1.transform, other_trans >> elem2.transform )
        for elem1 in self for elem2 in other )

    elements = []
    self_vertices = [ elem.vertices for elem in self ]
    if self == other:
      other_vertices = self_vertices
      issym = False#True
    else:
      other_vertices = [ elem.vertices for elem in other ]
      issym = False
    for i, elemi in enumerate(self):
      lookup = { v: n for n, v in enumerate(self_vertices[i]) }
      for j, elemj in enumerate(other):
        if issym and i == j:
          reference = element.NeighborhoodTensorReference( elemi.reference, elemj.reference, 0, (0,0) )
          elements.append( element.Element( reference, self_trans >> elemi.transform, other_trans >> elemj.transform ) )
          break
        common = [ (lookup[v],n) for n, v in enumerate(other_vertices[j]) if v in lookup ]
        if not common:
          neighborhood = -1
          transf = 0, 0
        elif len(common) == 4:
          neighborhood = 0
          assert elemi == elemj
          transf = 0, 0
        elif len(common) == 2:
          neighborhood = 1
          vertex = (0,2), (2,3), (3,1), (1,0), (2,0), (3,2), (1,3), (0,1)
          transf = tuple( vertex.index(v) for v in zip(*common) )
        elif len(common) == 1:
          neighborhood = 2
          transf = tuple( (0,3,1,2)[v] for v in common[0] )
        else:
          raise ValueError( 'Unknown neighbor type %i' % neighborhood )
        reference = element.NeighborhoodTensorReference( elemi.reference, elemj.reference, neighborhood, transf )
        elements.append( element.Element( reference, self_trans >> elemi.transform, other_trans >> elemj.transform ) )
        if issym:
          reference = element.NeighborhoodTensorReference( elemj.reference, elemi.reference, neighborhood, transf[::-1] )
          elements.append( element.Element( reference, self_trans >> elemj.transform, other_trans >> elemi.transform ) )
    return Topology( elements )

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
    iweights = function.iweights( geometry, self.ndims )
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
  def integrate( self, funcs, ischeme, geometry, force_dense=False ):
    'integrate'

    single_arg = not isinstance(funcs,(list,tuple))
    if single_arg:
      funcs = funcs,

    iweights = function.iweights( geometry, self.ndims )
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
  def integrate_symm( self, funcs, ischeme, geometry, force_dense=False ):
    'integrate a symmetric integrand on a product domain' # TODO: find a proper home for this

    single_arg = not isinstance(funcs,list)
    if single_arg:
      funcs = funcs,

    iweights = function.iweights( geometry, self.ndims )
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
      assert isinstance( elem.reference, element.NeighborhoodTensorReference )
      elemcmp = cmp( elem.transform.trans2, elem.opposite.trans2 )
      if elemcmp < 0:
        continue
      for ifunc, lock, index, data in idata.eval( elem, ischeme ):
        with lock:
          retvals[ifunc][index] += data
          if elemcmp:
            retvals[ifunc][index[::-1]] += data.T
          else:
            numpy.testing.assert_almost_equal( data, data.T, err_msg='symmetry check failed' )

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

  @property
  def refined( self ):
    return RefinedTopology( self )

  def refine( self, n ):
    'refine entire topology n times'

    return self if n <= 0 else self.refined.refine( n-1 )

class StructuredTopology( Topology ):
  'structured topology'

  def __init__( self, structure, periodic=() ):
    'constructor'

    structure = numpy.asarray(structure)
    self.structure = structure
    self.periodic = tuple(periodic)
    self.groups = {}
    Topology.__init__( self, filter(None,structure.flat) )

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
      topo = Topology( allbelems )

    topo.groups = dict( zip( ( 'left', 'right', 'bottom', 'top', 'front', 'back' ), boundaries ) )
    return topo

  @cache.property
  def interfaces( self ):
    'interfaces'

    interfaces = []
    eye = numpy.eye( self.ndims-1, dtype=int )
    for idim in range(self.ndims):
      if idim in self.periodic:
        t1 = (slice(None),)*idim + (slice(None),)
        t2 = (slice(None),)*idim + (numpy.array( range(1,self.structure.shape[idim]) + [0] ),)
      else:
        t1 = (slice(None),)*idim + (slice(-1),)
        t2 = (slice(None),)*idim + (slice(1,None),)
      A = numpy.zeros( (self.ndims,self.ndims-1), dtype=int )
      A[:idim] = -eye[:idim]
      A[idim+1:] = eye[idim:]
      b = numpy.hstack( [ numpy.ones( idim+1, dtype=int ), numpy.zeros( self.ndims-idim, dtype=int ) ] )
      trans1 = transform.updim(A,sign=1) >> transform.shift(b[:-1])
      trans2 = transform.updim(A,sign=-1) >> transform.shift(b[1:])
      edge = element.SimplexReference(1)**(self.ndims-1)
      for elem1, elem2 in numpy.broadcast( self.structure[t1], self.structure[t2] ):
        assert elem1.transform == elem1.opposite
        assert elem2.transform == elem2.opposite
        ielem = element.Element( edge, trans1 >> elem1.transform, trans2 >> elem2.transform )
        interfaces.append( ielem )
    return Topology( interfaces )

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
        assert len(numbers) >= 2 * overlap
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
          dofmap[elem.transform] = dofs
          funcmap[elem.transform] = std
        elif mask.any():
          dofmap[elem.transform] = dofs[mask]
          funcmap[elem.transform] = std, mask

    if hasnone:
      touched = numpy.zeros( dofcount, dtype=bool )
      for dofs in dofmap.itervalues():
        touched[ dofs ] = True
      renumber = touched.cumsum()
      dofcount = int(renumber[-1])
      dofmap = dict( ( trans, renumber[dofs]-1 ) for trans, dofs in dofmap.iteritems() )

    return function.function( funcmap, dofmap, dofcount, self.ndims )

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

  def trim( self, levelset, maxrefine, evalrefine=0, eps=.01 ):
    'trim element along levelset'

    numer = rational.asrational( numeric.round(1./eps) )
    levelset = function.ascompiled( levelset )
    __log__ = log.iter( 'elem', self.structure.ravel() )
    trimmedelems = [ elem.trim( levelset=levelset, maxrefine=maxrefine, numer=numer ) for elem in __log__ ]
    trimmedstructure = numpy.array( trimmedelems ).reshape( self.structure.shape )
    return StructuredTopology( trimmedstructure, periodic=self.periodic )

  def __str__( self ):
    'string representation'

    return '%s(%s)' % ( self.__class__.__name__, 'x'.join(map(str,self.structure.shape)) )

  @cache.property
  def multiindex( self ):
    'Inverse map of self.structure: given an element find its location in the structure.'
    return dict( (self.structure[alpha], alpha) for alpha in numpy.ndindex( self.structure.shape ) )

class HierarchicalTopology( Topology ):
  'collection of nested topology elments'

  def __init__( self, basetopo, elements ):
    'constructor'

    if isinstance( basetopo, HierarchicalTopology ):
      basetopo = basetopo.basetopo
    self.basetopo = basetopo
    self.edict = { elem.transform: elem.reference for elem in elements }
    Topology.__init__( self, elements )

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
        celem, transform = belem.parent
        if celem in myelems:
          belemset.add( belem )
      allbelems.extend( belemset )
      for btag, belems in topo.boundary.groups.iteritems():
        bgroups.setdefault( btag, [] ).extend( belemset.intersection(belems) )
      topo = topo.refined # proceed to next level
      elems -= myelems
    boundary = Topology( allbelems )
    boundary.groups = dict( ( tag, Topology( group ) ) for tag, group in bgroups.items() )
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
        (celem1,transform1), (celem2,transform2) = ielem.parents
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
    return Topology( allinterfaces )

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
      for trans, idofs in dofaxis.dofmap.items():
        ref = self.edict.get( trans )
        if ref:
          remaining -= 1
          touchtopo[idofs] = True
          myelems.append( trans )
          newdiscard.append( trans )
        else:
          trans1, trans2 = trans[1]
          if trans2 in discard:
            newdiscard.append( trans )
            supported[idofs] = False
          else:
            parentelems.append( trans )
            myelems.append( trans )
  
      keep = numpy.logical_and( supported, touchtopo ) # THE refinement law

      for trans in myelems: # loop over all top-level or parent elements in level irefine
        idofs = dofaxis.dofmap[trans] # local dof numbers
        mykeep = keep[idofs]
        std = func.stdmap[trans]
        assert isinstance(std,element.StdElem)
        if mykeep.all():
          stdmap[trans] = std # use all shapes from this level
        elif mykeep.any():
          stdmap[trans] = std, mykeep # use some shapes from this level
        newdofs = [ ndofs + keep[:idof].sum() for idof in idofs if keep[idof] ] # new dof numbers
        if irefine: # not at lowest level
          trans1, trans2 = trans[1]
          newdofs.extend( dofmap[trans2] ) # add dofs of all underlying 'broader' shapes
        dofmap[trans] = numpy.array(newdofs) # add result to IEN mapping of new function object
  
      ndofs += keep.sum() # update total number of dofs
      if not remaining:
        break
      topo = topo.refined # proceed to next level
  
    else:

      raise Exception, 'elements remaining after %d iterations' % maxrefine

    for trans in parentelems:
      del dofmap[trans] # remove auxiliary elements

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

class RefinedTopology( Topology ):
  'refinement'

  def __init__( self, basetopo ):
    self.basetopo = basetopo
    elements = [ child for elem in basetopo for child in elem.children ]
    Topology.__init__( self, elements )

  def __getitem__( self, key ):
    return self.basetopo[key].refined

  @cache.property
  def boundary( self ):
    return self.basetopo.boundary.refined

# vim:shiftwidth=2:foldmethod=indent:foldnestmax=2
