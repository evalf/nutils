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

from __future__ import print_function, division
from . import element, function, util, numpy, parallel, matrix, log, core, numeric, cache, rational, transform, _
import warnings

class Topology( object ):
  'topology base class'

  def __init__( self, elements, ndims=None ):
    'constructor'

    self.elements = tuple(elements)
    self.ndims = self.elements[0].ndims if ndims is None else ndims # assume all equal
    self.__groups = {}
    self.__boundary = None

  def set_boundary( self, boundary ):
    assert self.__class__.boundary == Topology.boundary, 'cannot set boundary of %s' % self.__class__.__name__
    assert isinstance( boundary, Topology )
    assert boundary.ndims == self.ndims - 1
    self.__boundary = boundary

  @property
  def boundary( self ):
    if not self.__boundary:
      edges = {}
      __log__ = log.iter( 'elem', self )
      for elem in __log__:
        elemcoords = elem.vertices
        for iedge, iverts in enumerate( elem.reference.edge2vertices ):
          edgekey = tuple( sorted( c for c, n in zip( elemcoords, iverts ) if n ) )
          try:
            edges.pop( edgekey )
          except KeyError:
            edges[edgekey] = elem.edge(iedge)
      self.__boundary = Topology( edges.values() )
    return self.__boundary

  @property
  def groupnames( self ):
    return self.__groups.keys()

  def __contains__( self, element ):
    return self.edict.get( element.transform ) == element

  def __len__( self ):
    return len( self.elements )

  def __iter__( self ):
    return iter( self.elements )

  def __add__( self, other ):
    'add topologies'

    assert self.ndims == other.ndims
    return Topology( set(self) | set(other) )

  def __sub__( self, other ):
    'subtract topologies'

    assert self.ndims == other.ndims
    return Topology( set(self) - set(other), self.ndims )

  def __mul__( self, other ):
    'element products'

    quad = element.SimplexReference(1)**2
    ndims = self.ndims + other.ndims
    eye = numpy.eye( ndims, dtype=int )
    self_trans = transform.affine(eye[:self.ndims])
    other_trans = transform.affine(eye[self.ndims:])

    if any( elem.reference != quad for elem in self ) or any( elem.reference != quad for elem in other ):
      return Topology( element.Element( elem1.reference * elem2.reference, elem1.transform << self_trans, elem2.transform << other_trans )
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
          elements.append( element.Element( reference, elemi.transform << self_trans, elemj.transform << other_trans ) )
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
        elements.append( element.Element( reference, elemi.transform << self_trans, elemj.transform << other_trans ) )
        if issym:
          reference = element.NeighborhoodTensorReference( elemj.reference, elemi.reference, neighborhood, transf[::-1] )
          elements.append( element.Element( reference, elemj.transform << self_trans, elemi.transform << other_trans ) )
    return Topology( elements )

  def __getitem__( self, item ):
    'subtopology'

    if not isinstance( item, str ):
      raise KeyError( str(item) )
    items = ( self.__groups[it] for it in item.split( ',' ) )
    return sum( items, next(items) )

  def __setitem__( self, item, topo ):
    assert isinstance( topo, Topology ), 'wrong type: got %s, expected Topology' % type(topo)
    assert topo.ndims == self.ndims, 'wrong dimension: got %d, expected %d' % ( topo.ndims, self.ndims )
    for elem in topo:
      assert self.edict[elem.transform] == elem, 'group %r is not a subtopology' % item
    self.__groups[item] = topo

  @cache.property
  def edict( self ):
    '''transform -> element mapping'''
    return { elem.transform: elem for elem in self }

  @cache.property
  def transrange( self ):
    nmin = nmax = len(self.elements[0].transform)
    for elem in self.elements[1:]:
      n = len(elem.transform)
      nmin = min( n, nmin )
      nmax = max( n, nmax )
    return nmin, nmax

  @property
  def refine_iter( self ):
    topo = self
    __log__ = log.count( 'refinement level' )
    for irefine in __log__:
      yield topo
      topo = topo.refined

  stdfunc     = lambda self, *args, **kwargs: self.basis( 'std', *args, **kwargs )
  linearfunc  = lambda self, *args, **kwargs: self.basis( 'std', degree=1, *args, **kwargs )
  splinefunc  = lambda self, *args, **kwargs: self.basis( 'spline', *args, **kwargs )
  bubblefunc  = lambda self, *args, **kwargs: self.basis( 'bubble', *args, **kwargs )
  discontfunc = lambda self, *args, **kwargs: self.basis( 'discont', *args, **kwargs )

  def basis( self, name, *args, **kwargs ):
    f = getattr( self, 'basis_' + name )
    return f( *args, **kwargs )

  def basis_std( self, degree=1 ):
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
      fmap[elem.transform] = (stdfunc,None),
      nmap[elem.transform] = dofs
    return function.function( fmap=fmap, nmap=nmap, ndofs=len(dofmap), ndims=self.ndims )

  def basis_bubble( self ):
    'spline from vertices'

    assert self.ndims == 2
    dofmap = {}
    fmap = {}
    nmap = {}
    stdfunc = element.BubbleTriangle()
    for ielem, elem in enumerate(self):
      assert elem.reference == element.SimplexReference(2)
      dofs = numpy.empty( elem.nverts+1, dtype=int )
      for i, v in enumerate( elem.vertices ):
        dof = dofmap.get(v)
        if dof is None:
          dof = len(self) + len(dofmap)
          dofmap[v] = dof
        dofs[i] = dof
      dofs[ elem.nverts ] = ielem
      fmap[elem.transform] = (stdfunc,None),
      nmap[elem.transform] = dofs
    return function.function( fmap=fmap, nmap=nmap, ndofs=len(self)+len(dofmap), ndims=self.ndims )

  def basis_spline( self, degree ):

    assert degree == 1
    return self.stdfunc( degree )

  def basis_discont( self, degree ):
    'discontinuous shape functions'

    assert isinstance( degree, int ) and degree >= 0
    fmap = {}
    nmap = {}
    ndofs = 0
    for elem in self:
      stdfunc = elem.reference.stdfunc(degree)
      fmap[elem.transform] = (stdfunc,None),
      nmap[elem.transform] = ndofs + numpy.arange(stdfunc.nshapes)
      ndofs += stdfunc.nshapes
    return function.function( fmap=fmap, nmap=nmap, ndofs=ndofs, ndims=self.ndims )

  @cache.property
  def simplex( self ):
    simplices = [ simplex for elem in self for simplex in elem.simplices ]
    return Topology( simplices )

  def refined_by( self, refine ):
    'create refined space by refining dofs in existing one'

    refine = set( item.transform if isinstance(item,element.Element) else item for item in refine )
    refined = []
    for elem in self:
      if elem.transform in refine:
        refined.extend( elem.children )
      else:
        refined.append( elem )
    return HierarchicalTopology( self, refined )

  @log.title
  def elem_eval( self, funcs, ischeme, separate=False ):
    'element-wise evaluation'

    single_arg = not isinstance(funcs,(tuple,list))
    if single_arg:
      funcs = funcs,

    slices = []
    pointshape = function.PointShape()
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
        for ind, f in function.blocks( func ):
          idata.append( function.Tuple([ ifunc, ind, f ]) )
      else:
        idata.append( function.Tuple([ ifunc, (), func ]) )
      retvals.append( retval )
    idata = function.Tuple( idata )
    fcache = cache.CallDict()

    __log__ = log.enumerate( 'elem', self )
    for ielem, elem in parallel.pariter( __log__ ):
      s = slices[ielem],
      for ifunc, index, data in idata.eval( elem, ischeme, fcache ):
        retvals[ifunc][s+index] += data

    log.debug( 'cache', fcache.summary() )
    log.info( 'created', ', '.join( '%s(%s)' % ( retval.__class__.__name__, ','.join( str(n) for n in retval.shape ) ) for retval in retvals ) )
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
    iwscale = function.iwscale( geometry, self.ndims )
    idata = [ iwscale ]
    for func in funcs:
      func = function.asarray( func )
      assert all( numeric.isint(sh) for sh in func.shape )
      idata.append( func * iwscale )
      retvals.append( numpy.empty( (len(self),)+func.shape ) )
    idata = function.Tuple( idata )

    fcache = cache.CallDict()
    for ielem, elem in enumerate( self ):
      ipoints, iweights = fcache( elem.reference.getischeme, ischeme[elem] if isinstance(ischeme,dict) else ischeme )
      area_data = idata.eval( elem, ischeme, fcache )
      area = numeric.dot( iweights, area_data[0] )
      for retval, data in zip( retvals, area_data[1:] ):
        retval[ielem] = numeric.dot( iweights, data ) / area

    log.debug( 'cache', fcache.summary() )
    log.info( 'created', ', '.join( '%s(%s)' % ( retval.__class__.__name__, ','.join( str(n) for n in retval.shape ) ) for retval in retvals ) )
    if single_arg:
      retvals, = retvals

    return retvals

  def _integrate( self, funcs, ischeme ):

    # Functions may consist of several blocks, such as originating from
    # chaining. Here we make a list of all blocks consisting of triplets of
    # argument id, evaluable index, and evaluable values.

    blocks = [ ( ifunc, ind, f )
      for ifunc, func in enumerate( funcs )
        for ind, f in function.blocks( func ) ]

    block2func, indices, values = zip( *blocks ) if blocks else ([],[],[])
    indexfunc = function.Tuple( indices )
    valuefunc = function.Tuple( values )

    log.debug( 'integrating %s distinct blocks' % '+'.join(
      str(block2func.count(ifunc)) for ifunc in range(len(funcs)) ) )

    if core.getprop( 'dot', False ):
      valuefunc.graphviz()

    fcache = cache.CallDict()

    # To allocate (shared) memory for all block data we evaluate indexfunc to
    # build an nblocks x nelems+1 offset array, and nblocks index lists of
    # length nelems.

    offsets = numpy.zeros( ( len(blocks), len(self)+1 ), dtype=int )
    indices = [ [] for i in range( len(blocks) ) ]

    for ielem, elem in enumerate( self ):
      for iblock, index in enumerate( indexfunc.eval( elem, None, fcache ) ):
        n = util.product( len(ind) for ind in index ) if index else 1
        offsets[iblock,ielem+1] = offsets[iblock,ielem] + n
        indices[iblock].append( index )

    # Since several blocks may belong to the same function, we post process the
    # offsets to form consecutive intervals in longer arrays. The length of
    # these arrays is captured in the nfuncs-array nvals.

    nvals = numpy.zeros( len(funcs), dtype=int )
    for iblock, ifunc in enumerate( block2func ):
      offsets[iblock] += nvals[ifunc]
      nvals[ifunc] = offsets[iblock,-1]

    # The data_index list contains shared memory index and value arrays for
    # each function argument.

    data_index = [
      ( parallel.shzeros( n, dtype=float ),
        parallel.shzeros( (funcs[ifunc].ndim,n), dtype=int ) )
            for ifunc, n in enumerate(nvals) ]

    # In a second, parallel element loop, valuefunc is evaluated to fill the
    # data part of data_index using the offsets array for location. Each
    # element has its own location so no locks are required. The index part of
    # data_index is filled in the same loop. It does not use valuefunc data but
    # benefits from parallel speedup.

    __log__ = log.enumerate( 'elem', self )
    for ielem, elem in parallel.pariter( __log__ ):
      ipoints, iweights = fcache( elem.reference.getischeme, ischeme[elem] if isinstance(ischeme,dict) else ischeme )
      for iblock, intdata in enumerate( valuefunc.eval( elem, ipoints, fcache ) ):
        s = slice(*offsets[iblock,ielem:ielem+2])
        data, index = data_index[ block2func[iblock] ]
        w_intdata = numeric.dot( iweights, intdata )
        data[s] = w_intdata.ravel()
        si = (slice(None),) + (_,) * (w_intdata.ndim-1)
        for idim, ii in enumerate( indices[iblock][ielem] ):
          index[idim,s].reshape(w_intdata.shape)[...] = ii[si]
          si = si[:-1]

    log.debug( 'cache', fcache.summary() )

    return data_index

  @log.title
  def integrate( self, funcs, ischeme, geometry, force_dense=False ):
    'integrate'

    iwscale = function.iwscale( geometry, self.ndims )
    single_arg = not isinstance( funcs, (list,tuple) )
    integrands = [ funcs * iwscale ] if single_arg else [ func * iwscale for func in funcs ]
    data_index = self._integrate( integrands, ischeme )
    retvals = [ matrix.assemble( data, index, integrand.shape, force_dense ) for integrand, (data,index) in zip( integrands, data_index ) ]
    return retvals[0] if single_arg else retvals

  @log.title
  def integrate_symm( self, funcs, ischeme, geometry, force_dense=False ):
    'integrate a symmetric integrand on a product domain' # TODO: find a proper home for this

    iwscale = function.iwscale( geometry, self.ndims )
    single_arg = not isinstance( funcs, (list,tuple) )
    integrands = [ funcs * iwscale ] if single_arg else [ func * iwscale for func in funcs ]
    assert all( integrand.ndim == 2 for integrand in integrands )
    diagelems = []
    trielems = []
    for elem in self:
      assert isinstance( elem.reference, element.NeighborhoodTensorReference )
      head1 = elem.transform[:-1]
      head2 = elem.opposite[:-1]
      if head1 == head2:
        diagelems.append( elem )
      elif head1 < head2:
        trielems.append( elem )
    diag_data_index = Topology( diagelems, self.ndims )._integrate( integrands, ischeme )
    tri_data_index = Topology( trielems, self.ndims )._integrate( integrands, ischeme )
    retvals = []
    for integrand, (diagdata,diagindex), (tridata,triindex) in zip( integrands, diag_data_index, tri_data_index ):
      data = numpy.concatenate( [ diagdata, tridata, tridata ], axis=0 )
      index = numpy.concatenate( [ diagindex, triindex, triindex[::-1] ], axis=1 )
      retvals.append( matrix.assemble( data, index, integrand.shape, force_dense ) )
    return retvals[0] if single_arg else retvals

  def projection( self, fun, onto, geometry, **kwargs ):
    'project and return as function'

    weights = self.project( fun, onto, geometry, **kwargs )
    return onto.dot( weights )

  @log.title
  def project( self, fun, onto, geometry, tol=0, ischeme=None, droptol=1e-8, exact_boundaries=False, constrain=None, verify=None, maxiter=0, ptype='lsqr' ):
    'L2 projection of function onto function space'

    log.debug( 'projection type:', ptype )

    if exact_boundaries:
      constrain |= self.boundary.project( fun, onto, geometry, constrain=constrain, title='boundaries', ischeme=ischeme, tol=tol, droptol=droptol, ptype=ptype )
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
      data = function.Tuple( function.Tuple([ fun, f, ind ]) for ind, f in function.blocks( onto ) )
      for elem in self:
        for f, w, ind in data.eval( elem, 'bezier2' ):
          w = w.swapaxes(0,1) # -> dof axis, point axis, ...
          wf = w * f[ (slice(None),)+numpy.ix_(*ind[1:]) ]
          W[ind[0]] += w.reshape(w.shape[0],-1).sum(1)
          F[ind[0]] += wf.reshape(w.shape[0],-1).sum(1)
          I[ind[0]] = True

      I[constrain.where] = False
      constrain[I] = F[I] / W[I]

    else:
      raise Exception( 'invalid projection %r' % ptype )

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

  def trim( self, levelset, maxrefine, eps=.01 ):
    'trim element along levelset'

    numer = rational.round(1./eps)
    poselems = []
    negelems = []
    __log__ = log.iter( 'elem', self )
    for elem in __log__:
      pos, neg = elem.trim( levelset=levelset, maxrefine=maxrefine, numer=numer )
      if pos:
        poselems.append( pos )
      if neg:
        negelems.append( neg )
    return TrimmedTopology( self, poselems ), \
           TrimmedTopology( self, negelems )

  def elem_project( self, funcs, degree, ischeme=None, check_exact=False ):

    single_arg = not isinstance( funcs, (list,tuple) )
    if single_arg:
      funcs = funcs,

    if ischeme is None:
      ischeme = 'gauss%d' % (degree*2)

    blocks = function.Tuple([ function.Tuple([ function.Tuple( ind_f )
      for ind_f in function.blocks( func ) ])
        for func in funcs ])

    bases = {}
    extractions = [ [] for ifunc in range(len(funcs) ) ]

    __log__ = log.iter( 'elem', self )
    for elem in __log__:

      try:
        points, projector, basis = bases[ elem.reference ]
      except KeyError:
        points, weights = elem.reference.getischeme( ischeme )
        basis = elem.reference.stdfunc(degree).eval( points )
        npoints, nfuncs = basis.shape
        A = numeric.dot( weights, basis[:,:,_] * basis[:,_,:] )
        projector = numpy.linalg.solve( A, basis.T * weights )
        bases[ elem.reference ] = points, projector, basis

      for ifunc, ind_val in enumerate( blocks.eval( elem, points ) ):

        if len(ind_val) == 1:
          (allind, sumval), = ind_val
        else:
          allind, where = zip( *[ numpy.unique( [ i for ind, val in ind_val for i in ind[iax] ], return_inverse=True ) for iax in range( funcs[ifunc].ndim ) ] )
          sumval = numpy.zeros( [ len(n) for n in (points,) + allind ] )
          for ind, val in ind_val:
            I, where = zip( *[ ( w[:len(n)], w[len(n):] ) for w, n in zip( where, ind ) ] )
            sumval[ numpy.ix_( range(len(points)), *I ) ] += val
          assert not any( where )

        ex = numeric.dot( projector, sumval )
        if check_exact:
          numpy.testing.assert_almost_equal( sumval, numeric.dot( basis, ex ), decimal=15 )

        extractions[ifunc].append(( allind, ex ))

    if single_arg:
      extractions, = extractions

    return extractions


def UnstructuredTopology( elems, ndims ):
  return Topology( elems )

class StructuredTopology( Topology ):
  'structured topology'

  def __init__( self, structure, periodic=() ):
    'constructor'

    structure = numpy.asarray(structure)
    self.structure = structure
    self.periodic = tuple(periodic)
    Topology.__init__( self, structure.flat )

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

    boundaries = {}
    for idim in range(self.ndims):
      if idim in self.periodic:
        continue
      for iside in range(2):
        iedge = 2 * idim + iside
        s = [ slice(None) ] * self.ndims
        s[idim] = iside-1
        s = tuple(s)
        belems = numpy.frompyfunc( lambda elem: elem.edge( iedge ) if elem is not None else None, 1, 1 )( self.structure[s] )
        periodic = [ d - (d>idim) for d in self.periodic if d != idim ] # TODO check that dimensions are correct for ndim > 2
        name = ( 'right', 'left', 'top', 'bottom', 'back', 'front' )[iedge]
        boundaries[name] = StructuredTopology( belems, periodic=periodic )

    allbelems = [ belem for boundary in boundaries.values() for belem in boundary.structure.flat if belem is not None ]
    topo = Topology( allbelems, self.ndims-1 )
    for name, btopo in boundaries.items():
      topo[name] = btopo

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
      trans1 = transform.affine( A, b[:-1], isflipped=False )
      trans2 = transform.affine( A, b[1:], isflipped=True )
      edge = element.SimplexReference(1)**(self.ndims-1)
      for elem1, elem2 in numpy.broadcast( self.structure[t1], self.structure[t2] ):
        assert elem1.transform == elem1.opposite
        assert elem2.transform == elem2.opposite
        ielem = element.Element( edge, elem1.transform << trans1, elem2.transform << trans2 )
        interfaces.append( ielem )
    return Topology( interfaces, self.ndims-1 )

  def basis_spline( self, degree, neumann=(), knots=None, periodic=None, closed=False, removedofs=None ):
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
          funcmap[elem.transform] = (std,None),
        else:
          assert mask.any()
          dofmap[elem.transform] = dofs[mask]
          funcmap[elem.transform] = (std,mask),

    if hasnone:
      touched = numpy.zeros( dofcount, dtype=bool )
      for dofs in dofmap.itervalues():
        touched[ dofs ] = True
      renumber = touched.cumsum()
      dofcount = int(renumber[-1])
      dofmap = dict( ( trans, renumber[dofs]-1 ) for trans, dofs in dofmap.items() )

    return function.function( funcmap, dofmap, dofcount, self.ndims )

  def basis_bspline( self, degree, knotvalues=None, knotmultiplicities=None, periodic=None ):
    'Bspline from vertices'
    
    if periodic is None:
      periodic = self.periodic

    if isinstance( degree, int ):
      degree = [degree]*self.ndims

    assert len(degree)==self.ndims

    if knotvalues is None:
      knotvalues = [None]*self.ndims
    
    if knotmultiplicities is None:
      knotmultiplicities = [None]*self.ndims

    vertex_structure = numpy.array( 0 )
    dofcount = 1
    slices = []
    cache = {}
    for idim in range( self.ndims ):
      p = degree[idim]
      n = self.structure.shape[idim]
      isperiodic = idim in periodic

      k = knotvalues[idim]
      if k is None: #Defaults to uniform spacing
        k = numpy.arange(n+1)
      else:
        k = numpy.asarray( k )

      m = knotmultiplicities[idim]
      if m is None: #Defaults to open spline without internal repetitions
        m = numpy.array([p+1]+[1]*(n-1)+[p+1]) if not isperiodic else numpy.ones(n+1,dtype=int)
      else:
        m = numpy.array(m) #Make copy to prevent overwriting of data

      assert min(m)>0 and max(m)<=p+1, 'Incorrect multiplicity encountered'
      assert len(k)==len(m), 'Length mismatch between knots vector and knot multiplicities vector'
      assert len(k)==n+1, 'Knot vector size does not match the topology size'

      if not isperiodic:
        nd = sum(m)-p-1
        npre  = p+1-m[0]  #Number of knots to be appended to front
        npost = p+1-m[-1] #Number of knots to be appended to rear
        m[0] = m[-1] = p+1
      else:
        assert m[0]==m[-1], 'Periodic spline multiplicity expected'
        assert m[0]<p+1, 'Endpoint multiplicity for periodic spline should be p or smaller'

        nd = sum(m[:-1])
        npre = npost = 0
        k = numpy.concatenate( [ k[-p-1:-1]+k[0]-k[-1], k, k[1:1+p]-k[0]+k[-1]] )
        m = numpy.concatenate( [ m[-p-1:-1], m, m[1:1+p] ] )

      km = numpy.array([ki for ki,mi in zip(k,m) for cnt in range(mi)],dtype=float)
      assert len(km)==sum(m)
      assert nd>0, 'No basis functions defined. Knot vector too short.'

      stdelems_i = []
      slices_i = []
      offsets = numpy.cumsum(m[:-1])-p
      if isperiodic:
        offsets = offsets[p:-p]
      offset0 = offsets[0]+npre

      for offset in offsets:
        start = max(offset0-offset,0) #Zero unless prepending influence
        stop  = p+1-max(offset-offsets[-1]+npost,0) #Zero unless appending influence
        slices_i.append( slice(offset-offset0+start,offset-offset0+stop) )
        lknots  = km[offset:offset+2*p] - km[offset] #Copy operation required
        if p: #Normalize for optimized caching
          lknots /= lknots[-1]
        key = ( tuple(numeric.round(lknots*numpy.iinfo(numpy.int32).max)), p )
        try:
          coeffs = cache[key]
        except KeyError:
          coeffs = self._localsplinebasis( lknots, p )
          cache[key] = coeffs
        poly = element.PolyLine( coeffs[:,start:stop] )
        stdelems_i.append( poly )
      stdelems = stdelems[...,_]*stdelems_i if idim else numpy.array(stdelems_i)

      numbers = numpy.arange(nd)
      if isperiodic:
        numbers = numpy.concatenate([numbers,numbers[:p]])
      vertex_structure = vertex_structure[...,_]*nd+numbers
      dofcount*=nd
      slices.append(slices_i)

    #Cache effectivity
    log.debug( 'Local knot vector cache effectivity: %d' % (100*(1.-len(cache)/float(sum(self.structure.shape)))) )

    dofmap = {}
    funcmap = {}
    for item in numpy.broadcast( self.structure, stdelems, *numpy.ix_(*slices) ):
      elem = item[0]
      std = item[1]
      S = item[2:]
      dofs = vertex_structure[S].ravel()
      dofmap[elem.transform] = dofs
      funcmap[elem.transform] = (std,None),

    return function.function( funcmap, dofmap, dofcount, self.ndims )

  @staticmethod
  def _localsplinebasis ( lknots, p ):
  
    assert isinstance(lknots,numpy.ndarray), 'Local knot vector should be numpy array'
    assert len(lknots)==2*p, 'Expected 2*p local knots'
  
    #Based on Algorithm A2.2 Piegl and Tiller
    N    = [None]*(p+1)
    N[0] = numpy.poly1d([1.])
  
    if p > 0:
  
      assert (lknots[:-1]-lknots[1:]<numpy.spacing(1)).all(), 'Local knot vector should be non-decreasing'
      assert lknots[p]-lknots[p-1]>numpy.spacing(1), 'Element size should be positive'
      
      lknots = lknots.astype(float)
  
      xi = numpy.poly1d([lknots[p]-lknots[p-1],lknots[p-1]])
  
      left  = [None]*p
      right = [None]*p
  
      for i in range(p):
        left[i] = xi - lknots[p-i-1]
        right[i] = -xi + lknots[p+i]
        saved = 0.
        for r in range(i+1):
          temp = N[r]/(lknots[p+r]-lknots[p+r-i-1])
          N[r] = saved+right[r]*temp
          saved = left[i-r]*temp
        N[i+1] = saved

    assert all(Ni.order==p for Ni in N)

    return numpy.array([Ni.coeffs for Ni in N]).T[::-1]

  def linearfunc( self ):
    'linears'

  def basis_discont( self, degree ):
    'discontinuous shape functions'

    if isinstance( degree, int ):
      degree = (degree,) * self.ndims
    assert len(degree) == self.ndims
    assert all( p >= 0 for p in degree )

    stdfunc = util.product( element.PolyLine( element.PolyLine.bernstein_poly(p) ) for p in degree )
      
    fmap = {}
    nmap = {}
    ndofs = 0
    for elem in self:
      fmap[elem.transform] = (stdfunc,None),
      nmap[elem.transform] = ndofs + numpy.arange(stdfunc.nshapes)
      ndofs += stdfunc.nshapes
    return function.function( fmap=fmap, nmap=nmap, ndofs=ndofs, ndims=self.ndims )

  def basis_std( self, degree, removedofs=None ):
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
          dofmap[ elem.transform ] = dofs
          funcmap[ elem.transform ] = (stdelem,None),
        elif mask.any():
          dofmap[ elem.transform ] = dofs[mask]
          funcmap[ elem.transform ] = (stdelem,mask),

    if hasnone:
      touched = numpy.zeros( dofcount, dtype=bool )
      for dofs in dofmap.itervalues():
        touched[ dofs ] = True
      renumber = touched.cumsum()
      dofcount = int(renumber[-1])
      dofmap = { trans: renumber[dofs]-1 for trans, dofs in dofmap.items() }

    return function.function( funcmap, dofmap, dofcount, self.ndims )

  @cache.property
  def refined( self ):
    'refine non-uniformly'

    structure = numpy.array( [ elem.children if elem is not None else [None]*(2**self.ndims) for elem in self.structure.flat ] )
    structure = structure.reshape( self.structure.shape + (2,)*self.ndims )
    structure = structure.transpose( sum( [ ( i, self.ndims+i ) for i in range(self.ndims) ], () ) )
    structure = structure.reshape( numpy.array(self.structure.shape) * 2 )
    refined = StructuredTopology( structure )
    for group in self.groupnames:
      refined[group] = self[group].refined
    return refined

  def __str__( self ):
    'string representation'

    return '%s(%s)' % ( self.__class__.__name__, 'x'.join( str(n) for n in self.structure.shape ) )

  @cache.property
  def multiindex( self ):
    'Inverse map of self.structure: given an element find its location in the structure.'
    return dict( (self.structure[alpha], alpha) for alpha in numpy.ndindex( self.structure.shape ) )

class HierarchicalTopology( Topology ):
  'collection of nested topology elments'

  def __init__( self, basetopo, elements ):
    'constructor'

    self.basetopo = basetopo if not isinstance( basetopo, HierarchicalTopology ) else basetopo.basetopo
    Topology.__init__( self, elements )

  @cache.property
  def refined( self ):
    elements = [ child for elem in self for child in elem.children ]
    return HierarchicalTopology( self.basetopo, elements )

  def __getitem__( self, item ):
    itemtopo = self.basetopo[item]
    elems = []
    for topo in itemtopo.refine_iter:
      elems.extend( elem for elem in topo if elem in self )
      if topo.transrange[0] >= self.transrange[1]:
        break
    return HierarchicalTopology( itemtopo, elems )

  @cache.property
  def boundary( self ):
    'boundary elements'

    boundarytopo = self.basetopo.boundary
    elems = []
    for topo in boundarytopo.refine_iter:
      elems.extend( elem for elem in topo if elem.transform[:-1] in self.edict )
      if topo.transrange[0] - 1 >= self.transrange[1]:
        break
    return HierarchicalTopology( boundarytopo, elems )

  @cache.property
  def interfaces( self ):
    'interface elements & groups'

    raise NotImplementedError( 'awaiting reimplementation' )
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
    return Topology( allinterfaces, self.ndims-1 )

  @log.title
  def basis( self, name, *args, **kwargs ):
    'build hierarchical function space'

    collect = {}
    ndofs = 0 # total number of dofs of new function object
    remaining = len(self) # element count down (know when to stop)

    for topo in self.basetopo.refine_iter:
      assert topo.transrange[0] <= self.transrange[1]

      funcsp = topo.basis( name, *args, **kwargs ) # shape functions for current level
      supported = numpy.ones( funcsp.shape[0], dtype=bool ) # True if dof is fully contained in self or parents
      touchtopo = numpy.zeros( funcsp.shape[0], dtype=bool ) # True if dof touches at least one elem in self
      myelems = [] # all top-level or parent elements in current level

      for trans, idofs, stds in function._unpack( funcsp ):
        mytrans = trans.lookup( self.edict )
        if mytrans == trans: # trans is in domain
          remaining -= 1
          touchtopo[idofs] = True
          myelems.append(( trans, idofs, stds ))
        elif mytrans: # trans is finer than domain
          supported[idofs] = False
        else: # trans is coarser than domain
          myelems.append(( trans, idofs, stds ))
  
      keep = numpy.logical_and( supported, touchtopo ) # THE refinement law
      renumber = (ndofs-1) + keep.cumsum()

      for trans, idofs, stds in myelems: # loop over all top-level or parent elements in current level
        (std,origkeep), = stds
        assert origkeep is None
        mykeep = keep[idofs]
        if mykeep.all():
          newstds = (std,None),
          newdofs = renumber[idofs]
        elif mykeep.any():
          newstds = (std,mykeep),
          newdofs = renumber[idofs[mykeep]]
        else:
          newstds = (None,None),
          newdofs = numpy.zeros( [0], dtype=int )
        if topo != self.basetopo:
          olddofs, oldstds = collect[ trans[:-1] ] # dofs, stds of all underlying 'broader' shapes
          newstds += oldstds
          newdofs = numpy.hstack([ newdofs, olddofs ])
        collect[ trans ] = newdofs, newstds # add result to IEN mapping of new function object
  
      ndofs += int( keep.sum() ) # update total number of dofs
      if not remaining:
        break

    nmap = {}
    fmap = {}
    check = numpy.zeros( ndofs, dtype=bool )
    for elem in self:
      dofs, stds = collect[ elem.transform ]
      nmap[ elem.transform ] = dofs
      fmap[ elem.transform ] = stds
      check[dofs] = True
    assert check.all()

    return function.function( fmap=fmap, nmap=nmap, ndofs=ndofs, ndims=self.ndims )

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

class TrimmedTopology( Topology ):
  'trimmed'

  def __init__( self, basetopo, elements, trimmed=[] ):
    self.basetopo = basetopo
    self.trimmed = tuple(trimmed)
    Topology.__init__( self, elements )

  @cache.property
  def refined( self ):
    elements = [ child for elem in self for child in elem.children ]
    trimmed = [ child for elem in self.trimmed for child in elem.children ]
    return TrimmedTopology( self.basetopo.refined, elements, trimmed )

  @cache.property
  def boundary( self ):
    warnings.warn( 'warning: boundaries of trimmed topologies are not trimmed' )
    belems = list( self.trimmed ) + [ belem for belem in self.basetopo.boundary if belem.transform.lookup(self.edict) ]
    boundary = TrimmedTopology( self.basetopo.boundary, belems )
    if self.trimmed:
      boundary['trimmed'] = Topology( self.trimmed )
    return boundary

  def __getitem__( self, key ):
    try:
      itemtopo = Topology.__getitem__( self, key )
    except KeyError:
      elements = []
      keytopo = self.basetopo[key]
      for elem in keytopo:
        trimelem = self.edict.get(elem.transform)
        if trimelem is not None:
          elements.append( trimelem )
      itemtopo = TrimmedTopology( keytopo, elements )
    return itemtopo

  @log.title
  def basis( self, name, *args, **kwargs ):
    funcsp = self.basetopo.basis( name, *args, **kwargs )
    ((dofaxis,),func), = function.blocks( funcsp )
    nmap = {}
    fmap = {}
    renumber = {}
    for elem in self:
      trans = elem.transform
      dofs = []
      for dof in dofaxis.dofmap[trans]:
        newdof = renumber.get(dof)
        if newdof is None:
          newdof = len(renumber)
          renumber[dof] = newdof
        dofs.append( newdof )
      nmap[trans] = numpy.array(dofs)
      fmap[trans] = func.stdmap[trans]
    return function.function( fmap=fmap, nmap=nmap, ndofs=len(renumber), ndims=self.ndims )


# vim:shiftwidth=2:foldmethod=indent:foldnestmax=2
