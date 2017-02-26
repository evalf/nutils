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
from .index import IndexedArray
import warnings, functools, collections, itertools, functools, operator

_identity = lambda x: x

class Topology( object ):
  'topology base class'

  # subclass needs to implement: .elements

  def __init__( self, ndims ):
    'constructor'

    assert numeric.isint( ndims ) and ndims >= 0
    self.ndims = ndims

  def __str__( self ):
    'string representation'

    return '%s(#%s)' % ( self.__class__.__name__, len(self) )

  def __len__( self ):
    return len( self.elements )

  def __iter__( self ):
    return iter( self.elements )

  def getitem( self, item ):
    return EmptyTopology( self.ndims )

  def __getitem__( self, item ):
    if not isinstance( item, tuple ):
      item = item,
    if all( it in (...,slice(None)) for it in item ):
      return self
    topo = self.getitem(item) if len(item) != 1 or not isinstance(item[0],str) \
       else functools.reduce( operator.or_, map( self.getitem, item[0].split(',') ), EmptyTopology(self.ndims) )
    if not topo:
      raise KeyError( item )
    return topo

  def __invert__( self ):
    return OppositeTopology( self )

  def __or__( self, other ):
    assert isinstance( other, Topology ) and other.ndims == self.ndims
    return other if not self \
      else self if not other \
      else NotImplemented if isinstance( other, UnionTopology ) \
      else UnionTopology( (self,other) )

  __ror__ = lambda self, other: self.__or__( other )

  def __add__( self, other ):
    return self | other

  def __contains__( self, element ):
    ielem = self.edict.get(element.transform)
    return ielem is not None and self.elements[ielem] == element

  def __sub__( self, other ):
    assert isinstance( other, Topology ) and other.ndims == self.ndims
    return other.__rsub__( self )

  def __rsub__( self, other ):
    assert isinstance( other, Topology ) and other.ndims == self.ndims
    return other - other.subset( self, newboundary=getattr(self,'boundary',None) )

  def __mul__( self, other ):
    return ProductTopology( self, other )

  @cache.property
  def edict( self ):
    '''transform -> ielement mapping'''
    return { elem.transform: ielem for ielem, elem in enumerate(self) }

  @cache.property
  def border_transforms( self ):
    border_transforms = set()
    for belem in self.boundary:
      try:
        ielem, tail = belem.transform.lookup_item( self.edict )
      except KeyError:
        pass
      else:
        border_transforms.add( self.elements[ielem].transform )
    return border_transforms

  @property
  def refine_iter( self ):
    topo = self
    for irefine in log.count( 'refinement level' ):
      yield topo
      topo = topo.refined

  stdfunc     = lambda self, *args, **kwargs: self.basis( 'std', *args, **kwargs )
  linearfunc  = lambda self, *args, **kwargs: self.basis( 'std', degree=1, *args, **kwargs )
  splinefunc  = lambda self, *args, **kwargs: self.basis( 'spline', *args, **kwargs )
  bubblefunc  = lambda self, *args, **kwargs: self.basis( 'bubble', *args, **kwargs )
  discontfunc = lambda self, *args, **kwargs: self.basis( 'discont', *args, **kwargs )

  def basis( self, name, *args, **kwargs ):
    if self.ndims == 0:
      return function.asarray( [1] )
    f = getattr( self, 'basis_' + name )
    return f( *args, **kwargs )

  @log.title
  @core.single_or_multiple
  def elem_eval( self, funcs, ischeme, separate=False, geometry=None, asfunction=False, edit=_identity ):
    'element-wise evaluation'

    fcache = cache.WrapperCache()

    assert not separate or not asfunction, '"separate" and "asfunction" are mutually exclusive'
    if geometry:
      iwscale = function.J( geometry, self.ndims )
      npoints = len(self)
      slices = range(npoints)
    else:
      iwscale = 1
      slices = []
      npoints = 0
      for elem in log.iter( 'elem', self ):
        ipoints, iweights = ischeme[elem] if isinstance(ischeme,dict) else fcache[elem.reference.getischeme]( ischeme )
        np = len( ipoints )
        slices.append( slice(npoints,npoints+np) )
        npoints += np

    nprocs = min( core.getprop( 'nprocs', 1 ), len(self) )
    zeros = parallel.shzeros if nprocs > 1 else numpy.zeros
    retvals = []
    idata = []
    for ifunc, func in enumerate( funcs ):
      if isinstance( func, IndexedArray ):
        func = func.unwrap( geometry )
      func = function.asarray( edit( func * iwscale ) )
      retval = zeros( (npoints,)+func.shape, dtype=func.dtype )
      if function.isarray( func ):
        for ind, f in function.blocks( func ):
          idata.append( function.Tuple([ ifunc, ind, f ]) )
      else:
        idata.append( function.Tuple([ ifunc, (), func ]) )
      retvals.append( retval )
    idata = function.Tuple( idata )

    if core.getprop( 'dot', False ):
      idata.graphviz()

    for ielem, elem in parallel.pariter( log.enumerate( 'elem', self ), nprocs=nprocs ):
      ipoints, iweights = ischeme[elem] if isinstance(ischeme,dict) else fcache[elem.reference.getischeme]( ischeme )
      s = slices[ielem],
      for ifunc, index, data in idata.eval( elem, ipoints, fcache ):
        retvals[ifunc][s+numpy.ix_(*[ ind for (ind,) in index ])] += numeric.dot(iweights,data) if geometry else data

    log.debug( 'cache', fcache.stats )
    log.info( 'created', ', '.join( '%s(%s)' % ( retval.__class__.__name__, ','.join( str(n) for n in retval.shape ) ) for retval in retvals ) )

    if asfunction:
      if geometry:
        retvals = [ function.elemwise( { elem.transform: value for elem, value in zip( self, retval ) }, shape=retval.shape[1:] ) for retval in retvals ]
      else:
        tsp = [ ( elem.transform, s, fcache[elem.reference.getischeme](ischeme)[0] ) for elem, s in zip( self, slices ) ]
        retvals = [ function.sampled( { trans: (retval[s],points) for trans, s, points in tsp }, self.ndims ) for retval in retvals ]
    elif separate:
      retvals = [ [ retval[s] for s in slices ] for retval in retvals ]

    return retvals

  @log.title
  @core.single_or_multiple
  def elem_mean( self, funcs, geometry, ischeme ):
    'element-wise average'

    retvals = self.elem_eval( (1,)+funcs, geometry=geometry, ischeme=ischeme )
    return [ v / retvals[0][(slice(None),)+(_,)*(v.ndim-1)] for v in retvals[1:] ]

  def _integrate( self, funcs, ischeme, fcache=None ):

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

    if fcache is None:
      fcache = cache.WrapperCache()

    # To allocate (shared) memory for all block data we evaluate indexfunc to
    # build an nblocks x nelems+1 offset array, and nblocks index lists of
    # length nelems.

    offsets = numpy.zeros( ( len(blocks), len(self)+1 ), dtype=int )
    indices = [ [] for i in range( len(blocks) ) ]

    for ielem, elem in enumerate( self ):
      for iblock, index in enumerate( indexfunc.eval( elem, None, fcache ) ):
        n = util.product( len(ind) for (ind,) in index ) if index else 1
        offsets[iblock,ielem+1] = offsets[iblock,ielem] + n
        indices[iblock].append([ ind for (ind,) in index ])

    # Since several blocks may belong to the same function, we post process the
    # offsets to form consecutive intervals in longer arrays. The length of
    # these arrays is captured in the nfuncs-array nvals.

    nvals = numpy.zeros( len(funcs), dtype=int )
    for iblock, ifunc in enumerate( block2func ):
      offsets[iblock] += nvals[ifunc]
      nvals[ifunc] = offsets[iblock,-1]

    # The data_index list contains shared memory index and value arrays for
    # each function argument.

    nprocs = min( core.getprop( 'nprocs', 1 ), len(self) )
    empty = parallel.shzeros if nprocs > 1 else numpy.empty
    data_index = [
      ( empty( n, dtype=float ),
        empty( (funcs[ifunc].ndim,n), dtype=int ) )
            for ifunc, n in enumerate(nvals) ]

    # In a second, parallel element loop, valuefunc is evaluated to fill the
    # data part of data_index using the offsets array for location. Each
    # element has its own location so no locks are required. The index part of
    # data_index is filled in the same loop. It does not use valuefunc data but
    # benefits from parallel speedup.

    for ielem, elem in parallel.pariter( log.enumerate( 'elem', self ), nprocs=nprocs ):
      ipoints, iweights = ischeme[elem] if isinstance(ischeme,dict) else fcache[elem.reference.getischeme]( ischeme )
      assert iweights is not None, 'no integration weights found'
      for iblock, intdata in enumerate( valuefunc.eval( elem, ipoints, fcache ) ):
        s = slice(*offsets[iblock,ielem:ielem+2])
        data, index = data_index[ block2func[iblock] ]
        w_intdata = numeric.dot( iweights, intdata )
        data[s] = w_intdata.ravel()
        si = (slice(None),) + (_,) * (w_intdata.ndim-1)
        for idim, ii in enumerate( indices[iblock][ielem] ):
          index[idim,s].reshape(w_intdata.shape)[...] = ii[si]
          si = si[:-1]

    log.debug( 'cache', fcache.stats )

    return data_index

  @log.title
  @core.single_or_multiple
  def integrate( self, funcs, ischeme, geometry=None, force_dense=False, fcache=None, edit=_identity ):
    'integrate'

    iwscale = function.J( geometry, self.ndims ) if geometry else 1
    funcs = [ func.unwrap( geometry ) if isinstance( func, IndexedArray ) else func for func in funcs ]
    integrands = [ function.asarray( edit( func * iwscale ) ) for func in funcs ]
    data_index = self._integrate( integrands, ischeme, fcache )
    return [ matrix.assemble( data, index, integrand.shape, force_dense ) for integrand, (data,index) in zip( integrands, data_index ) ]

  @log.title
  @core.single_or_multiple
  def integrate_symm( self, funcs, ischeme, geometry=None, force_dense=False, fcache=None, edit=_identity ):
    'integrate a symmetric integrand on a product domain' # TODO: find a proper home for this

    iwscale = function.J( geometry, self.ndims ) if geometry else 1
    funcs = [ func.unwrap( geometry ) if isinstance( func, IndexedArray ) else func for func in funcs ]
    integrands = [ function.asarray( edit( func * iwscale ) ) for func in funcs ]
    assert all( integrand.ndim == 2 for integrand in integrands )
    diagelems = []
    trielems = []
    for elem in self:
      head1 = elem.transform[:-1]
      head2 = elem.opposite[:-1]
      if head1 == head2:
        diagelems.append( elem )
      elif head1 < head2:
        trielems.append( elem )
    diag_data_index = UnstructuredTopology( self.ndims, diagelems )._integrate( integrands, ischeme, fcache=fcache )
    tri_data_index = UnstructuredTopology( self.ndims, trielems )._integrate( integrands, ischeme, fcache=fcache )
    retvals = []
    for integrand, (diagdata,diagindex), (tridata,triindex) in zip( integrands, diag_data_index, tri_data_index ):
      data = numpy.concatenate( [ diagdata, tridata, tridata ], axis=0 )
      index = numpy.concatenate( [ diagindex, triindex, triindex[::-1] ], axis=1 )
      retvals.append( matrix.assemble( data, index, integrand.shape, force_dense ) )
    return retvals

  def projection( self, fun, onto, geometry, **kwargs ):
    'project and return as function'

    weights = self.project( fun, onto, geometry, **kwargs )
    return onto.dot( weights )

  @log.title
  def project( self, fun, onto, geometry, tol=0, ischeme=None, droptol=1e-12, exact_boundaries=False, constrain=None, verify=None, ptype='lsqr', precon='diag', edit=_identity, **solverargs ):
    'L2 projection of function onto function space'

    log.debug( 'projection type:', ptype )

    if isinstance( fun, IndexedArray ):
      fun = fun.unwrap( geometry )
    if isinstance( onto, IndexedArray ):
      onto = onto.unwrap( geometry )

    if constrain is None:
      constrain = util.NanVec( onto.shape[0] )
    else:
      constrain = constrain.copy()
    if exact_boundaries:
      constrain |= self.boundary.project( fun, onto, geometry, constrain=constrain, title='boundaries', ischeme=ischeme, tol=tol, droptol=droptol, ptype=ptype, edit=edit )
    assert isinstance( constrain, util.NanVec )
    assert constrain.shape == onto.shape[:1]

    avg_error = None # setting this depends on projection type

    if ptype == 'lsqr':
      assert ischeme is not None, 'please specify an integration scheme for lsqr-projection'
      fun2 = function.asarray( fun )**2
      if len( onto.shape ) == 1:
        Afun = function.outer( onto )
        bfun = onto * fun
      elif len( onto.shape ) == 2:
        Afun = function.outer( onto ).sum( 2 )
        bfun = function.sum( onto * fun, -1 )
        if fun2.ndim:
          fun2 = fun2.sum(-1)
      else:
        raise Exception
      assert fun2.ndim == 0
      A, b, f2, area = self.integrate( [Afun,bfun,fun2,1], geometry=geometry, ischeme=ischeme, edit=edit, title='building system' )
      N = A.rowsupp(droptol)
      if numpy.all( b == 0 ):
        constrain[~constrain.where&N] = 0
        avg_error = 0.
      else:
        solvecons = constrain.copy()
        solvecons[~(constrain.where|N)] = 0
        u = A.solve( b, solvecons, tol=tol, symmetric=True, precon=precon, **solverargs )
        constrain[N] = u[N]
        err2 = f2 - numpy.dot( 2*b-A.matvec(u), u ) # can be negative ~zero due to rounding errors
        avg_error = numpy.sqrt( err2 ) / area if err2 > 0 else 0

    elif ptype == 'convolute':
      assert ischeme is not None, 'please specify an integration scheme for convolute-projection'
      if len( onto.shape ) == 1:
        ufun = onto * fun
        afun = onto
      elif len( onto.shape ) == 2:
        ufun = function.sum( onto * fun, axis=-1 )
        afun = function.norm2( onto )
      else:
        raise Exception
      u, scale = self.integrate( [ ufun, afun ], geometry=geometry, ischeme=ischeme, edit=edit )
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
      data = function.Tuple( function.Tuple([ fun, onto_f, onto_ind ]) for onto_ind, onto_f in function.blocks( onto ) )
      for elem in self:
        for fun_, onto_f_, onto_ind_ in data.eval( elem, 'bezier2' ):
          onto_f_ = onto_f_.swapaxes(0,1) # -> dof axis, point axis, ...
          indfun_ = fun_[ (slice(None),)+numpy.ix_(*onto_ind_[1:]) ]
          assert onto_f_.shape[0] == len(onto_ind_[0])
          assert onto_f_.shape[1:] == indfun_.shape
          W[onto_ind_[0]] += onto_f_.reshape(onto_f_.shape[0],-1).sum(1)
          F[onto_ind_[0]] += ( onto_f_ * indfun_ ).reshape(onto_f_.shape[0],-1).sum(1)
          I[onto_ind_[0]] = True

      I[constrain.where] = False
      constrain[I] = F[I] / W[I]

    else:
      raise Exception( 'invalid projection %r' % ptype )

    numcons = constrain.where.sum()
    info = 'constrained {}/{} dofs'.format( numcons, constrain.size )
    if avg_error is not None:
      info += ', error {:.2e}/area'.format( avg_error )
    log.info( info )
    if verify is not None:
      assert numcons == verify, 'number of constraints does not meet expectation: %d != %d' % ( numcons, verify )

    return constrain

  @cache.property
  def simplex( self ):
    simplices = [ simplex for elem in self for simplex in elem.simplices ]
    return UnstructuredTopology( self.ndims, simplices )

  def refined_by( self, refine ):
    'create refined space by refining dofs in existing one'

    refine = set( item.transform if isinstance(item,element.Element) else item for item in refine )
    refined = []
    for elem in self:
      if elem.transform in refine:
        refined.extend( elem.children )
      else:
        refined.append( elem )
    return self.hierarchical( refined, precise=True )

  def hierarchical( self, refined, precise=False ):
    return HierarchicalTopology( self, refined, precise )

  @property
  def refined( self ):
    return RefinedTopology( self )

  def refine( self, n ):
    'refine entire topology n times'

    if numpy.iterable( n ):
      assert len(n) == self.ndims
      assert all( ni == n[0] for ni in n )
      n = n[0]
    return self if n <= 0 else self.refined.refine( n-1 )

  @log.title
  def trim( self, levelset, maxrefine, ndivisions=8, name='trimmed', leveltopo=None ):
    'trim element along levelset'

    fcache = cache.WrapperCache()
    if leveltopo is None:
      ischeme = 'vertex{}'.format(maxrefine)
      refs = [ elem.reference.trim( levelset.eval(elem,ischeme,fcache), maxrefine=maxrefine, ndivisions=ndivisions ) for elem in log.iter( 'elem', self ) ]
    else:
      log.info( 'collecting leveltopo elements' )
      bins = [ [] for ielem in range(len(self)) ]
      for elem in leveltopo:
        ielem, tail = elem.transform.lookup_item( self.edict )
        bins[ielem].append( tail )
      refs = []
      for elem, ctransforms in log.zip( 'elem', self, bins ):
        levels = numpy.empty( elem.reference.nvertices_by_level(maxrefine) )
        cover = list( fcache[elem.reference.vertex_cover]( sorted(ctransforms), maxrefine ) )
        # confirm cover and greedily optimize order
        mask = numpy.ones( len(levels), dtype=bool )
        while mask.any():
          imax = numpy.argmax([ mask[indices].sum() for trans, points, indices in cover ])
          trans, points, indices = cover.pop( imax )
          levels[indices] = levelset.eval( elem.transform << trans, points, fcache )
          mask[indices] = False
        refs.append( elem.reference.trim( levels, maxrefine=maxrefine, ndivisions=ndivisions ) )
    log.debug( 'cache', fcache.stats )
    return SubsetTopology( self, refs, newboundary=name )

  def subset( self, elements, newboundary=None, strict=False ):
    'intersection'
    refs = [ element.EmptyReference(self.ndims) ] * len(self)
    for elem in elements:
      try:
        ielem = self.edict[ elem.transform ]
      except KeyError:
        assert not strict, 'elements do not form a strict subset'
      else:
        ref = self.elements[ielem].reference & elem.reference
        if strict:
          assert ref == elem.reference, 'elements do not form a strict subset'
        refs[ielem] = ref
    if not any( refs ):
      return EmptyTopology( self.ndims )
    return SubsetTopology( self, refs, newboundary )

  def withgroups( self, vgroups={}, bgroups={}, igroups={}, pgroups={} ):
    return WithGroupsTopology( self, vgroups, bgroups, igroups, pgroups ) if vgroups or bgroups or igroups or pgroups else self

  withsubdomain  = lambda self, **kwargs: self.withgroups( vgroups=kwargs )
  withboundary   = lambda self, **kwargs: self.withgroups( bgroups=kwargs )
  withinterfaces = lambda self, **kwargs: self.withgroups( igroups=kwargs )
  withpoints     = lambda self, **kwargs: self.withgroups( pgroups=kwargs )

  @log.title
  @core.single_or_multiple
  def elem_project( self, funcs, degree, ischeme=None, check_exact=False ):

    if ischeme is None:
      ischeme = 'gauss%d' % (degree*2)

    blocks = function.Tuple([ function.Tuple([ function.Tuple( ind_f )
      for ind_f in function.blocks( func ) ])
        for func in funcs ])

    bases = {}
    extractions = [ [] for ifunc in range(len(funcs) ) ]

    for elem in log.iter( 'elem', self ):

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

    return extractions

  @log.title
  def volume( self, geometry, ischeme='gauss1' ):
    return self.integrate( 1, geometry=geometry, ischeme=ischeme )

  @log.title
  def volume_check( self, geometry, ischeme='gauss1', decimal=15 ):
    volume = self.volume( geometry, ischeme )
    zeros, volumes = self.boundary.integrate( [ geometry.normal(), geometry * geometry.normal() ], geometry=geometry, ischeme=ischeme )
    numpy.testing.assert_almost_equal( zeros, 0., decimal=decimal )
    numpy.testing.assert_almost_equal( volumes, volume, decimal=decimal )
    return volume

  def indicator( self ):
    return function.elemwise( { elem.transform: 1. for elem in self }, (), default=0. )

  def select( self, indicator, ischeme='bezier2' ):
    values = self.elem_eval( indicator, ischeme, separate=True )
    selected = [ elem for elem, value in zip( self, values ) if numpy.any( value > 0 ) ]
    return UnstructuredTopology( self.ndims, selected )

  def prune_basis( self, basis ):
    used = numpy.zeros( len(basis), dtype=bool )
    for axes, func in function.blocks( basis ):
      dofmap = axes[0]
      for elem in self:
        dofs = dofmap.eval( elem )
        used[dofs] = True
    if isinstance( basis, function.Inflate ) and isinstance( basis.func, function.Function ) and isinstance( basis.dofmap, function.DofMap ):
      renumber = used.cumsum()-1
      nmap = { trans: renumber[dofs[used[dofs]]] for trans, dofs in basis.dofmap.dofmap.items() }
      return function.function( fmap=basis.func.stdmap, nmap=nmap, ndofs=used.sum() )
    return function.mask( basis, used )

  def locate( self, geom, points, ischeme='vertex', scale=1, tol=1e-12, eps=0, maxiter=100 ):
    if geom.ndim == 0:
      geom = geom[_]
      points = points[...,_]
    assert geom.shape == (self.ndims,)
    points = numpy.asarray( points, dtype=float )
    assert points.ndim == 2 and points.shape[1] == self.ndims
    vertices = self.elem_eval( geom, ischeme=ischeme, separate=True )
    bboxes = numpy.array([ numpy.mean(v,axis=0) * (1-scale) + numpy.array([ numpy.min(v,axis=0), numpy.max(v,axis=0) ]) * scale
      for v in vertices ]) # nelems x {min,max} x ndims
    vref = element.getsimplex(0)
    pelems = []
    for point in points:
      ielems, = ((point >= bboxes[:,0,:]) & (point <= bboxes[:,1,:])).all(axis=-1).nonzero()
      for ielem in sorted( ielems, key=lambda i: numpy.linalg.norm(bboxes[i].mean(0)-point) ):
        converged = False
        elem = self.elements[ielem]
        xi, w = elem.reference.getischeme( 'gauss1' )
        xi = ( numpy.dot(w,xi) / w.sum() )[_] if len(xi) > 1 else xi.copy()
        J = function.localgradient( geom, self.ndims )
        geom_J = function.Tuple(( geom, J ))
        for iiter in range( maxiter ):
          point_xi, J_xi = geom_J.eval( elem, xi )
          err = numpy.linalg.norm( point - point_xi )
          if err < tol:
            converged = True
            break
          if iiter and err > prev_err:
            break
          prev_err = err
          xi += numpy.linalg.solve( J_xi, point - point_xi )
        if converged and elem.reference.inside( xi, eps=eps ):
          break
      else:
        raise Exception( 'failed to locate point', point )
      trans = transform.affine( linear=1, offset=xi[0] )
      for idim in range(self.ndims,0,-1): # transcend dimensions one by one to produce valid transformation
        trans <<= transform.affine( linear=numpy.eye(idim)[:,:-1], offset=numpy.zeros(idim), isflipped=False )
      pelems.append( element.Element( vref, elem.transform << trans, elem.opposite and elem.opposite << trans, oriented=True ) )
    return UnstructuredTopology( 0, pelems )

  def supp( self, basis, mask=None ):
    if mask is None:
      mask = numpy.ones( len(basis), dtype=bool )
    elif isinstance( mask, list ) or isinstance( mask, numpy.ndarray ) and mask.dtype == int:
      tmp = numpy.zeros( len(basis), dtype=bool )
      tmp[mask] = True
      mask = tmp
    else:
      assert isinstance( mask, numpy.ndarray ) and mask.dtype == bool and mask.shape == basis.shape[:1]
    indfunc = function.Tuple([ ind[0] for ind, f in basis.blocks ])
    subset = []
    for elem in self:
      try:
        ind, = numpy.concatenate( indfunc.eval(elem), axis=1 )
      except function.EvaluationError:
        pass
      else:
        if mask[ind].any():
          subset.append( elem )
    if not subset:
      return EmptyTopology( self.ndims )
    return self.subset( subset, newboundary='supp', strict=True )

  def revolved( self, geom ):
    assert geom.ndim == 1
    revdomain = self * RevolutionTopology()
    angle, = function.rootcoords(1)
    geom, angle = function.bifurcate( geom, angle )
    revgeom = function.concatenate([ geom[0] * function.trignormal(angle), geom[1:] ])
    simplify = lambda arg: 0 if arg is angle else function.edit( arg, simplify )
    return revdomain, revgeom, simplify

  def extruded( self, geom, nelems, periodic=False, bnames=('front','back') ):
    assert geom.ndim == 1
    root = transform.roottrans( 'extrude', shape=[ nelems if periodic else 0 ] )
    extopo = self * StructuredLine( root, i=0, j=nelems, periodic=periodic, bnames=bnames )
    exgeom = function.concatenate( function.bifurcate( geom, function.rootcoords(1) ) )
    return extopo, exgeom

class WithGroupsTopology( Topology ):
  'item topology'

  def __init__( self, basetopo, vgroups={}, bgroups={}, igroups={}, pgroups={} ):
    assert vgroups or bgroups or igroups or pgroups
    self.basetopo = basetopo
    self.vgroups = vgroups.copy()
    self.bgroups = bgroups.copy()
    self.igroups = igroups.copy()
    self.pgroups = pgroups.copy()
    Topology.__init__( self, basetopo.ndims )
    if core.getprop( 'selfcheck', False ):
      if self.vgroups:
        for topo in self.vgroups.values():
          if topo is not Ellipsis and not isinstance( topo, str ):
            assert isinstance( topo, Topology )
            assert topo.ndims == basetopo.ndims
            assert set(self.basetopo.edict).issuperset(topo.edict)
      if self.bgroups:
        self.boundary
      if self.igroups:
        self.interfaces
      if self.pgroups:
        self.points

  def withgroups( self, vgroups={}, bgroups={}, igroups={}, pgroups={} ):
    args = []
    for groups, newgroups in (self.vgroups,vgroups), (self.bgroups,bgroups), (self.igroups,igroups), (self.pgroups,pgroups):
      groups = groups.copy()
      groups.update( newgroups )
      args.append( groups )
    return WithGroupsTopology( self.basetopo, *args )

  def __iter__( self ):
    return iter( self.basetopo )

  def __len__( self ):
    return len( self.basetopo )

  def getitem( self, item ):
    if not isinstance( item, str ):
      itemtopo = self.basetopo.getitem(item)
    elif item in self.vgroups:
      itemtopo = self.vgroups[item]
      if itemtopo is Ellipsis:
        return self.basetopo
      if not isinstance( itemtopo, Topology ):
        itemtopo = self.basetopo[itemtopo]
    else:
      itemtopo = self.basetopo.getitem(item)
    assert isinstance( itemtopo, Topology ) and itemtopo.ndims == self.ndims
    return self.subset( itemtopo, strict=True )

  @property
  def edict( self ):
    return self.basetopo.edict

  @property
  def border_transforms( self ):
    return self.basetopo.border_transforms

  @property
  def connectivity( self ):
    return self.basetopo.connectivity

  @property
  def structure( self ):
    return self.basetopo.structure

  @property
  def elements( self ):
    return self.basetopo.elements

  @property
  def boundary( self ):
    return self.basetopo.boundary.withgroups( self.bgroups )

  @property
  def interfaces( self ):
    baseitopo = self.basetopo.interfaces
    # last minute orientation fix
    igroups = { name: UnstructuredTopology( self.ndims-1, [ elem if elem.transform in baseitopo.edict else elem.flipped for elem in elems ] ) for name, elems in self.igroups.items() }
    return baseitopo.withgroups( igroups )

  @property
  def points( self ):
    return self.basetopo.points.withgroups( self.pgroups )

  def basis( self, name, *args, **kwargs ):
    return self.basetopo.basis( name, *args, **kwargs )

  @cache.property
  def refined( self ):
    groups = [ { name: topo.refined if isinstance(topo,Topology) else topo for name, topo in groups.items() } for groups in (self.vgroups,self.bgroups,self.igroups,self.pgroups) ]
    return self.basetopo.refined.withgroups( *groups )

class OppositeTopology( Topology ):
  'opposite topology'

  def __init__( self, basetopo ):
    self.basetopo = basetopo
    Topology.__init__( self, basetopo.ndims )

  def getitem( self, item ):
    return ~( self.basetopo.getitem(item) )

  def __iter__( self ):
    return ( elem.flipped for elem in self.basetopo )

  def __len__( self ):
    return len( self.basetopo )

  @cache.property
  def elements( self ):
    return tuple( self )

  def __invert__( self ):
    return self.basetopo

class EmptyTopology( Topology ):
  'empty topology'

  def __iter__( self ):
    return iter([])

  def __len__( self ):
    return 0

  def __or__( self, other ):
    assert self.ndims == other.ndims
    return other

  def __rsub__( self, other ):
    return other

  @property
  def elements( self ):
    return ()

class Point( Topology ):
  'point'

  def __init__( self, trans, opposite=None ):
    assert isinstance( trans, transform.TransformChain ) and trans.fromdims == 0
    self.elem = element.Element( element.getsimplex(0), trans, opposite, oriented=True )
    Topology.__init__( self, ndims=0 )

  def __iter__( self ):
    yield self.elem

  @property
  def elements( self ):
    return self.elem,

class StructuredLine( Topology ):
  'structured topology'

  def __init__( self, root, i, j, periodic=False, bnames=None ):
    'constructor'

    assert isinstance(i,int) and isinstance(j,int) and j > i
    assert not bnames or len(bnames) == 2 and all( isinstance(bname,str) for bname in bnames )
    assert isinstance( root, transform.TransformChain )
    self.root = root
    self.i = i
    self.j = j
    self.periodic = periodic
    self.bnames = bnames or ()
    Topology.__init__( self, ndims=1 )

  @cache.property
  def _transforms( self ):
    # one extra left and right for opposites, even if periodic=True
    return tuple( self.root << transform.affine( linear=1, offset=[offset] ) for offset in range( self.i-1, self.j+1 ) )

  def __iter__( self ):
    reference = element.getsimplex(1)
    return ( element.Element( reference, trans ) for trans in self._transforms[1:-1] )

  def __len__( self ):
    return self.j - self.i

  @cache.property
  def elements( self ):
    return tuple( self )

  @cache.property
  def boundary( self ):
    if self.periodic:
      return EmptyTopology( ndims=0 )
    transforms = self._transforms
    left = transform.affine( numpy.zeros((1,0)), offset=[0], isflipped=True )
    right = transform.affine( numpy.zeros((1,0)), offset=[1], isflipped=False )
    bnd = Point( transforms[1] << left, transforms[0] << right ), Point( transforms[-2] << right, transforms[-1] << left )
    return UnionTopology( bnd, self.bnames )

  @cache.property
  def interfaces( self ):
    transforms = self._transforms
    left = transform.affine( numpy.zeros((1,0)), offset=[0], isflipped=True )
    right = transform.affine( numpy.zeros((1,0)), offset=[1], isflipped=False )
    points = [ Point( trans << left, opp << right ) for trans, opp in zip( transforms[2:-1], transforms[1:-2] ) ]
    if self.periodic:
      points.append( Point( transforms[1] << left, transforms[-2] << right ) )
    return UnionTopology( points )

  def basis_spline( self, degree, periodic=None, removedofs=None ):
    'spline from vertices'

    if periodic is None:
      periodic = self.periodic

    if numpy.iterable( degree ):
      degree, = degree

    if numpy.iterable( removedofs ):
      removedofs, = removedofs

    ndofs = len(self) + degree
    dofs = numpy.arange( ndofs )

    if periodic and degree > 0:
      assert ndofs >= 2 * degree
      dofs[-degree:] = dofs[:degree]
      ndofs -= overlap

    fmap = dict( zip( self._transforms[1:-1], element.PolyLine.spline( degree=degree, nelems=len(self), periodic=periodic ) ) )
    nmap = { trans: dofs[i:i+degree+1] for i, trans in enumerate(self._transforms[1:-1]) }
    func = function.function( fmap, nmap, ndofs )

    if not removedofs:
      return func

    mask = numpy.ones( ndofs, dtype=bool )
    mask[list(removedofs)] = False
    return function.mask( func, mask )

  def basis_discont( self, degree ):
    'discontinuous shape functions'

    fmap = dict.fromkeys( self._transforms[1:-1], element.PolyLine( element.PolyLine.bernstein_poly(degree) ) )
    nmap = dict( zip( self._transforms[1:-1], numpy.arange(len(self)*(degree+1)).reshape(len(self),degree+1) ) )
    return function.function( fmap=fmap, nmap=nmap, ndofs=len(self)*(degree+1) )

  def basis_std( self, degree, periodic=None, removedofs=None ):
    'spline from vertices'

    if periodic is None:
      periodic = self.periodic

    ndofs = len(self) * degree + 1
    dofs = numpy.arange( ndofs )

    if periodic and degree > 0:
      dofs[-1] = dofs[0]
      ndofs -= 1

    fmap = dict.fromkeys( self._transforms[1:-1], element.PolyLine( element.PolyLine.bernstein_poly(degree) ) )
    nmap = { trans: dofs[i*degree:(i+1)*degree+1] for i, trans in enumerate(self._transforms[1:-1]) }
    func = function.function( fmap, nmap, ndofs )
    if not removedofs:
      return func

    mask = numpy.ones( ndofs, dtype=bool )
    mask[list(removedofs)] = False
    return function.mask( func, mask )

  def __str__( self ):
    'string representation'

    return '{}({}:{})'.format( self.__class__.__name__, self.i, self.j )

class StructuredTopology( Topology ):
  'structured topology'

  def __init__( self, root, axes, nrefine=0, bnames=None ):
    'constructor'

    self.root = root
    self.axes = tuple(axes)
    self.nrefine = nrefine
    self.shape = tuple( axis.j - axis.i for axis in self.axes if axis.isdim )
    if bnames is None:
      assert len(self.axes) <= 3
      bnames = ('left', 'right'), ('bottom', 'top'), ('front', 'back')
      bnames = itertools.chain.from_iterable( n for axis, n in zip( self.axes, bnames ) if axis.isdim and not axis.isperiodic )
    self._bnames = tuple( bnames )
    assert len(self._bnames) == sum( 2 for axis in self.axes if axis.isdim and not axis.isperiodic )
    assert all( isinstance(bname,str) for bname in self._bnames )
    Topology.__init__( self, len(self.shape) )

  def __iter__( self ):
    reference = element.getsimplex(1)**self.ndims
    return ( element.Element( reference, trans, opp, oriented=True ) for trans, opp in zip( self._transform.flat, self._opposite.flat ) )

  def __len__( self ):
    return numpy.prod( self.shape, dtype=int )

  def getitem( self, item ):
    if not isinstance( item, tuple ):
      return EmptyTopology( self.ndims )
    assert all( isinstance(it,slice) for it in item ) and len(item) <= self.ndims
    if all( it == slice(None) for it in item ): # shortcut
      return self
    axes = []
    idim = 0
    for axis in self.axes:
      if axis.isdim and idim < len(item):
        s = item[idim]
        start, stop, stride = s.indices( axis.j - axis.i )
        assert stride == 1
        assert stop > start
        if start > 0 or stop < axis.j - axis.i:
          axis = DimAxis( axis.i+start, axis.i+stop, isperiodic=False )
        idim += 1
      axes.append( axis )
    return StructuredTopology( self.root, axes, self.nrefine, bnames=self._bnames )

  @cache.property
  def elements( self ):
    return tuple( self )

  @property
  def periodic( self ):
    return tuple( idim for idim, axis in enumerate(self.axes) if axis.isdim and axis.isperiodic )

  @staticmethod
  def mktransforms( axes, root, nrefine ):
    assert nrefine >= 0

    updim = transform.identity
    ndims = len(axes)
    active = numpy.ones( ndims, dtype=bool )
    for order, side, idim in sorted( (axis.ibound,axis.side,idim) for idim, axis in enumerate(axes) if not axis.isdim ):
      where = (numpy.arange(len(active))[active]==idim)
      matrix = numpy.eye(ndims)[:,~where]
      offset = where.astype(float) if side else numpy.zeros(ndims)
      updim <<= transform.affine(matrix,offset,isflipped=(idim%2==1)==side)
      ndims -= 1
      active[idim] = False

    grid = [ numpy.arange(axis.i>>nrefine,((axis.j-1)>>nrefine)+1) if axis.isdim else numpy.array([(axis.i-1 if axis.side else axis.j)>>nrefine]) for axis in axes ]
    indices = numeric.broadcast( *numeric.ix(grid) )
    transforms = numeric.asobjvector( transform.affine(1,index) for index in log.iter( 'elem', indices, indices.size ) ).reshape( indices.shape )

    if nrefine:
      shifts = numeric.broadcast( *numeric.ix( [0,.5] for axis in axes ) )
      scales = numeric.asobjvector( transform.affine( .5, shift ) for shift in shifts ).reshape( shifts.shape )
      for irefine in log.range( 'level', nrefine-1, -1, -1 ):
        offsets = numpy.array([ r[0] for r in grid ])
        grid = [ numpy.arange(axis.i>>irefine,((axis.j-1)>>irefine)+1) if axis.isdim else numpy.array([(axis.i-1 if axis.side else axis.j)>>irefine]) for axis in axes ]
        A = transforms[ numpy.broadcast_arrays( *numeric.ix( r//2-o for r, o in zip( grid, offsets ) ) ) ]
        B = scales[ numpy.broadcast_arrays( *numeric.ix( r%2 for r in grid ) ) ]
        transforms = A << B
      
    shape = tuple( axis.j - axis.i for axis in axes if axis.isdim )
    return numeric.asobjvector( ( root << trans << updim ).canonical for trans in log.iter( 'canonical', transforms.flat ) ).reshape( shape )

  @cache.property
  @log.title
  def _transform( self ):
    return self.mktransforms( self.axes, self.root, self.nrefine )

  @cache.property
  @log.title
  def _opposite( self ):
    nbounds = len( self.axes ) - self.ndims
    if nbounds == 0:
      return self._transform
    axes = [ BndAxis( axis.i, axis.j, axis.ibound, not axis.side ) if not axis.isdim and axis.ibound==nbounds-1 else axis for axis in self.axes ]
    return self.mktransforms( axes, self.root, self.nrefine )

  @property
  def structure( self ):
    warnings.warn( 'topology.structure will be removed in future', DeprecationWarning )
    reference = element.getsimplex(1)**self.ndims
    return numeric.asobjvector( element.Element( reference, trans, opp, oriented=True ) for trans, opp in numpy.broadcast( self._transform, self._opposite ) ).reshape( self.shape )

  @cache.property
  def connectivity( self ):
    connectivity = numpy.empty( self.shape+(self.ndims,2), dtype=int )
    connectivity[...] = -1
    ielems = numpy.arange( len(self) ).reshape( self.shape )
    for idim in range( self.ndims ):
      s = (slice(None),)*idim
      s1 = s + (slice(1,None),)
      s2 = s + (slice(0,-1),)
      connectivity[s2+(...,idim,0)] = ielems[s1]
      connectivity[s1+(...,idim,1)] = ielems[s2]
      if idim in self.periodic:
        connectivity[s+(-1,...,idim,0)] = ielems[s+(0,)]
        connectivity[s+(0,...,idim,1)] = ielems[s+(-1,)]
    return connectivity.reshape( len(self), self.ndims*2 )

  @cache.property
  def boundary( self ):
    'boundary'

    nbounds = len(self.axes) - self.ndims
    btopo = EmptyTopology( self.ndims-1 )
    jdim = 0
    for idim, axis in enumerate( self.axes ):
      if not axis.isdim or axis.isperiodic:
        continue
      btopos = [
        StructuredTopology(
          root=self.root,
          axes=self.axes[:idim] + (BndAxis(n,n if not axis.isperiodic else 0,nbounds,side),) + self.axes[idim+1:],
          nrefine=self.nrefine,
          bnames=self._bnames[:jdim*2]+self._bnames[jdim*2+2:] )
        for side, n in enumerate((axis.i,axis.j)) ]
      btopo |= UnionTopology( btopos, self._bnames[jdim*2:jdim*2+2] )
      jdim += 1
    return btopo

  @cache.property
  def interfaces( self ):
    'interfaces'

    assert self.ndims > 0, 'zero-D topology has no interfaces'
    itopos = []
    nbounds = len(self.axes) - self.ndims
    for idim, axis in enumerate(self.axes):
      if not axis.isdim:
        continue
      bndprops = [ BndAxis( i, i, ibound=nbounds, side=True ) for i in range( axis.i+1, axis.j ) ]
      if axis.isperiodic:
        assert axis.i == 0
        bndprops.append( BndAxis( axis.j, 0, ibound=nbounds, side=True ) )
      itopos.append( EmptyTopology( self.ndims-1 ) if not bndprops
                else UnionTopology( StructuredTopology( self.root, self.axes[:idim] + (axis,) + self.axes[idim+1:], self.nrefine ) for axis in bndprops ) )
    assert len(itopos) == self.ndims
    return UnionTopology( itopos, names=[ 'dir{}'.format(idim) for idim in range(self.ndims) ] )

  def basis_spline( self, degree, neumann=(), knots=None, periodic=None, closed=False, removedofs=None ):
    'spline from vertices'

    if periodic is None:
      periodic = self.periodic

    if numeric.isint( degree ):
      degree = ( degree, ) * self.ndims

    if removedofs == None:
      removedofs = [None] * self.ndims
    else:
      assert len(removedofs) == self.ndims

    dofshape = []
    slices = []
    vertex_structure = numpy.array( 0 )
    for idim in range( self.ndims ):
      periodic_i = idim in periodic
      n = self.shape[idim]
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
      numbers = numpy.arange( nd )
      if periodic_i and p > 0:
        overlap = p
        assert len(numbers) >= 2 * overlap
        numbers[ -overlap: ] = numbers[ :overlap ]
        nd -= overlap
      vertex_structure = vertex_structure[...,_] * nd + numbers
      dofshape.append( nd )
      slices.append( [ slice(i,i+p+1) for i in range(n) ] )

    dofmap = {}
    funcmap = {}
    for trans, std, *S in numpy.broadcast( self._transform, stdelems, *numpy.ix_(*slices) ):
      dofmap[trans] = vertex_structure[S].ravel()
      funcmap[trans] = std
    func = function.function( funcmap, dofmap, numpy.product(dofshape) )
    if not any( removedofs ):
      return func

    mask = numpy.ones( (), dtype=bool )
    for idofs, ndofs in zip( removedofs, dofshape ):
      mask = mask[...,_].repeat( ndofs, axis=-1 )
      if idofs:
        mask[...,[ numeric.normdim(ndofs,idof) for idof in idofs ]] = False
    assert mask.shape == tuple(dofshape)
    return function.mask( func, mask.ravel() )

  def basis_bspline( self, degree, knotvalues=None, knotmultiplicities=None, periodic=None ):
    'Bspline from vertices'
    
    if periodic is None:
      periodic = self.periodic

    if numeric.isint( degree ):
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
      n = self.shape[idim]
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
    log.debug( 'Local knot vector cache effectivity: %d' % (100*(1.-len(cache)/float(sum(self.shape)))) )

    dofmap = {}
    funcmap = {}
    for item in numpy.broadcast( self._transform, stdelems, *numpy.ix_(*slices) ):
      trans = item[0]
      std = item[1]
      S = item[2:]
      dofs = vertex_structure[S].ravel()
      dofmap[trans] = dofs
      funcmap[trans] = std

    return function.function( funcmap, dofmap, dofcount )

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

  def basis_discont( self, degree ):
    'discontinuous shape functions'

    if numeric.isint( degree ):
      degree = (degree,) * self.ndims
    assert len(degree) == self.ndims
    assert all( p >= 0 for p in degree )

    stdfunc = util.product( element.PolyLine( element.PolyLine.bernstein_poly(p) ) for p in degree )
      
    fmap = {}
    nmap = {}
    ndofs = 0
    for elem in self:
      fmap[elem.transform] = stdfunc
      nmap[elem.transform] = ndofs + numpy.arange(stdfunc.nshapes)
      ndofs += stdfunc.nshapes
    return function.function( fmap=fmap, nmap=nmap, ndofs=ndofs )

  def basis_std( self, degree, removedofs=None, periodic=None ):
    'spline from vertices'

    if periodic is None:
      periodic = self.periodic

    if numeric.isint( degree ):
      degree = ( degree, ) * self.ndims

    if removedofs == None:
      removedofs = [None] * self.ndims
    else:
      assert len(removedofs) == self.ndims

    dofshape = []
    slices = []
    vertex_structure = numpy.array( 0 )
    for idim in range( self.ndims ):
      periodic_i = idim in periodic
      n = self.shape[idim]
      p = degree[idim]
      nd = n * p + 1
      numbers = numpy.arange( nd )
      if periodic_i and p > 0:
        numbers[-1] = numbers[0]
        nd -= 1
      vertex_structure = vertex_structure[...,_] * nd + numbers
      dofshape.append( nd )
      slices.append( [ slice(p*i,p*i+p+1) for i in range(n) ] )

    funcmap = dict.fromkeys( self._transform.flat, util.product( element.PolyLine( element.PolyLine.bernstein_poly(d) ) for d in degree ) )
    dofmap = { trans: vertex_structure[S].ravel() for trans, *S in numpy.broadcast( self._transform, *numpy.ix_(*slices) ) }
    func = function.function( funcmap, dofmap, numpy.product(dofshape) )
    if not any( removedofs ):
      return func

    mask = numpy.ones( (), dtype=bool )
    for idofs, ndofs in zip( removedofs, dofshape ):
      mask = mask[...,_].repeat( ndofs, axis=-1 )
      if idofs:
        mask[...,[ numeric.normdim(ndofs,idof) for idof in idofs ]] = False
    assert mask.shape == tuple(dofshape)
    return function.mask( func, mask.ravel() )

  @property
  def refined( self ):
    'refine non-uniformly'

    axes = [ DimAxis(i=axis.i*2,j=axis.j*2,isperiodic=axis.isperiodic) if axis.isdim
        else BndAxis(i=axis.i*2,j=axis.j*2,ibound=axis.ibound,side=axis.side) for axis in self.axes ]
    return StructuredTopology( self.root, axes, self.nrefine+1, bnames=self._bnames )

  def __str__( self ):
    'string representation'

    return '%s(%s)' % ( self.__class__.__name__, 'x'.join( str(n) for n in self.shape ) )

class UnstructuredTopology( Topology ):
  'unstructured topology'

  def __init__( self, ndims, elements ):
    self.elements = tuple(elements)
    assert all( elem.ndims == ndims for elem in self.elements )
    Topology.__init__( self, ndims )

  @cache.property
  @log.title
  def connectivity( self ):
    edges = {}
    connectivity = []
    for ielem, elem in log.enumerate( 'elem', self ):
      connectivity.append( -numpy.ones( elem.nedges, dtype=int ) )
      for iedge, belem in enumerate( elem.edges ):
        belemcoords = belem.vertices
        edgekey = tuple( sorted( belemcoords ) )
        try:
          jelem, jedge = edges.pop( edgekey )
        except KeyError:
          edges[edgekey] = ielem, iedge
        else:
          # TODO assert transformation equivalence
          connectivity[ielem][iedge] = jelem
          connectivity[jelem][jedge] = ielem
    return tuple( connectivity )

  @cache.property
  def boundary( self ):
    elements = [ elem.edge(iedge) for elem, ioppelems in zip( self, self.connectivity ) for iedge in numpy.where( ioppelems == -1 )[0] ]
    return UnstructuredTopology( self.ndims-1, elements )

  @cache.property
  def interfaces( self ):
    seen = set()
    elements = []
    for ielem, ioppelems in enumerate( self.connectivity ):
      elem = self.elements[ielem]
      for iedge, ioppelem in enumerate( ioppelems ):
        if ioppelem == -1:
          continue
        try:
          seen.remove(( ielem, iedge ))
        except KeyError:
          ioppedge = tuple(self.connectivity[ioppelem]).index(ielem)
          oppelem = self.elements[ioppelem]
          elements.append( elem.edge(iedge).withopposite( oppelem.edge(ioppedge), oriented=False ) )
          seen.add(( ioppelem, ioppedge ))
    assert not seen
    return UnstructuredTopology( self.ndims-1, elements )

  def basis_std( self, degree=1 ):
    'std from vertices'

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
    return function.function( fmap=fmap, nmap=nmap, ndofs=len(dofmap) )

  def basis_bubble( self ):
    'bubble from vertices'

    assert self.ndims == 2
    dofmap = {}
    fmap = {}
    nmap = {}
    stdfunc = element.BubbleTriangle()
    for ielem, elem in enumerate(self):
      assert isinstance( elem.reference, element.TriangleReference )
      dofs = numpy.empty( elem.nverts+1, dtype=int )
      for i, v in enumerate( elem.vertices ):
        dof = dofmap.get(v)
        if dof is None:
          dof = len(self) + len(dofmap)
          dofmap[v] = dof
        dofs[i] = dof
      dofs[ elem.nverts ] = ielem
      fmap[elem.transform] = stdfunc
      nmap[elem.transform] = dofs
    return function.function( fmap=fmap, nmap=nmap, ndofs=len(self)+len(dofmap) )

  def basis_spline( self, degree ):
    assert degree == 1
    return self.basis( 'std', degree )

  def basis_discont( self, degree ):
    'discontinuous shape functions'

    assert numeric.isint( degree ) and degree >= 0
    fmap = {}
    nmap = {}
    ndofs = 0
    for elem in self:
      stdfunc = elem.reference.stdfunc(degree)
      fmap[elem.transform] = stdfunc
      nmap[elem.transform] = ndofs + numpy.arange(stdfunc.nshapes)
      ndofs += stdfunc.nshapes
    return function.function( fmap=fmap, nmap=nmap, ndofs=ndofs )

class UnionTopology( Topology ):
  'grouped topology'

  def __init__( self, topos, names=() ):
    self._topos = tuple(topos)
    assert all( isinstance( topo, Topology ) for topo in self._topos )
    self._names = tuple(names)[:len(self._topos)]
    assert all( isinstance(name,str) for name in self._names )
    assert len(set(self._names)) == len(self._names), 'duplicate name'
    ndims = self._topos[0].ndims
    assert all( topo.ndims == ndims for topo in self._topos )
    Topology.__init__( self, ndims )

  def getitem( self, item ):
    topos = [ topo if name == item else topo.getitem(item) for topo, name in itertools.zip_longest( self._topos, self._names ) ]
    return functools.reduce( operator.or_, topos, EmptyTopology(self.ndims) )

  def __or__( self, other ):
    if not isinstance( other, UnionTopology ):
      return UnionTopology( self._topos + (other,), self._names )
    return UnionTopology( self._topos[:len(self._names)] + other._topos + self._topos[len(self._names):], self._names + other._names )

  @cache.property
  def elements( self ):
    topos = util.OrderedDict()
    for itopo, topo in enumerate(self._topos):
      for elem in topo:
        topos.setdefault( elem.transform, [] ).append( elem )
    elements = []
    for trans, elems in topos.items():
      if len(elems) == 1:
        elements.append( elems[0] )
      else:
        refs = [ elem.reference for elem in elems ]
        while len(refs) > 1: # sweep all possible unions until a single reference is left
          nrefs = len(refs)
          iref = 0
          while iref < len(refs)-1:
            for jref in range( iref+1, len(refs) ):
              try:
                unionref = refs[iref] | refs[jref]
              except TypeError:
                pass
              else:
                refs[iref] = unionref
                del refs[jref]
                break
            iref += 1
          assert len(refs) < nrefs, 'incompatible elements in union'
        unionref, = refs
        opposite = elems[0].opposite
        assert all( elem.opposite == opposite for elem in elems[1:] )
        elements.append( element.Element( unionref, trans, opposite, oriented=True ) )
    return elements

  @property
  def refined( self ):
    return UnionTopology( [ topo.refined for topo in self._topos ], self._names )

class SubsetTopology( Topology ):
  'trimmed'

  def __init__( self, basetopo, refs, newboundary=None ):
    if newboundary is not None:
      assert isinstance( newboundary, str ) or isinstance( newboundary, Topology ) and newboundary.ndims == basetopo.ndims-1
    assert len(refs) == len(basetopo)
    self.refs = tuple(refs)
    self.basetopo = basetopo
    self.newboundary = newboundary
    Topology.__init__( self, basetopo.ndims )

  def getitem( self, item ):
    return self.basetopo.getitem(item).subset( self.elements, strict=False )

  def __rsub__( self, other ):
    assert self.basetopo == other
    refs = [ elem.reference - ref for elem, ref in zip( self.basetopo, self.refs ) ]
    return SubsetTopology( self.basetopo, refs, ~self.newboundary if isinstance(self.newboundary,Topology) else self.newboundary )

  def __or__( self, other ):
    if not isinstance( other, SubsetTopology ) or self.basetopo != other.basetopo:
      return super().__or__( other )
    refs = [ ref1 | ref2 for ref1, ref2 in zip( self.refs, other.refs ) ]
    if all( elem.reference == ref for elem, ref in zip( self.basetopo, refs ) ):
      return self.basetopo
    return SubsetTopology( self.basetopo, refs ) # TODO boundary

  @cache.property
  def connectivity( self ):
    mask = numpy.array([ bool(ref) for ref in self.refs ] + [False] ) # trailing false serves to map -1 to -1
    renumber = numpy.cumsum(mask)-1
    renumber[~mask] = -1
    connectivity = tuple( tuple( renumber[ioppelems[iedge]] if edgeref and iedge < len(ioppelems) else -1 for iedge, edgeref in enumerate(ref.edge_refs) )
      for ref, ioppelems in zip( self.refs, self.basetopo.connectivity ) if ref )
    return connectivity

  @cache.property
  def elements( self ):
    return tuple( element.Element( ref, elem.transform, elem.opposite ) for elem, ref in zip( self.basetopo, self.refs ) if ref )

  @property
  def refined( self ):
    elems = [ child for elem in self for child in elem.children if child ]
    return self.basetopo.refined.subset( elems, self.newboundary.refined if isinstance(self.newboundary,Topology) else self.newboundary, strict=True )

  @cache.property
  @log.title
  def boundary( self ):
    brefs = [ element.EmptyReference(self.ndims-1) ] * len(self.basetopo.boundary) # subset of original boundary
    newbelems = [] # newly formed boundary elements
    connectivity = self.basetopo.connectivity
    elements = self.basetopo.elements
    baseboundary = self.basetopo.boundary
    for ielem, ioppelems in enumerate(connectivity):
      ref = self.refs[ielem]
      if not ref:
        continue
      elem = elements[ielem]
      newbelems.extend( element.Element( edge, elem.transform << etrans, elem.transform << etrans.flipped ) for etrans, edge in ref.edges[elem.reference.nedges:] )
      for iedge, ioppelem in enumerate( ioppelems ):
        bref = ref.edge_refs[iedge]
        if not bref:
          continue
        if ioppelem == -1:
          index = baseboundary.edict[ (elem.transform<<ref.edge_transforms[iedge]).canonical ]
          brefs[index] = bref # by construction, bref must be equal or subset of original
        else:
          ioppedge = tuple(connectivity[ioppelem]).index(ielem)
          oppref = self.refs[ioppelem]
          if oppref:
            bref -= oppref.edge_refs[ioppedge]
          if bref:
            newbelems.append( element.Element( bref, elem.transform<<ref.edge_transforms[iedge], elements[ioppelem].edge(ioppedge).transform ) )
    origboundary = SubsetTopology( baseboundary, brefs )
    trimboundary = OrientedGroupsTopology( self.newboundary if isinstance(self.newboundary,Topology) else self.basetopo.interfaces, newbelems )
    return UnionTopology([ trimboundary, origboundary ], names=[ self.newboundary ] if isinstance(self.newboundary,str) else [] )

  @cache.property
  @log.title
  def interfaces( self ):
    irefs = [ element.EmptyReference(self.ndims-1) ] * len(self.basetopo.interfaces) # subset of original interfaces
    connectivity = self.basetopo.connectivity
    elements = self.basetopo.elements
    baseinterfaces = self.basetopo.interfaces
    for ielem, ioppelems in enumerate(connectivity):
      ref = self.refs[ielem]
      if not ref:
        continue # edge is empty
      elem = elements[ielem]
      for iedge, ioppelem in enumerate( ioppelems ):
        if ioppelem == -1:
          continue # edge lies on the boundary
        oppref = self.refs[ioppelem]
        if not oppref:
          continue # edge is empty
        index = baseinterfaces.edict.get( (elem.transform<<ref.edge_transforms[iedge]).canonical )
        if index is None:
          continue # edge is not oriented with an interface; rely on connectivity to also yield flipped element
        ioppedge = tuple( connectivity[ioppelem] ).index( ielem )
        irefs[index] = ref.edge_refs[iedge] & oppref.edge_refs[ioppedge]
    return SubsetTopology( baseinterfaces, irefs )

  @log.title
  def basis( self, name, *args, **kwargs ):
    if isinstance( self.basetopo, HierarchicalTopology ):
      warnings.warn( 'basis may be linearly dependent; a linearly indepent basis is obtained by trimming first, then creating hierarchical refinements' )
    basis = self.basetopo.basis( name, *args, **kwargs )
    return self.prune_basis( basis )

class OrientedGroupsTopology( UnstructuredTopology ):
  'unstructured topology with undirected semi-overlapping basetopology'

  def __init__( self, basetopo, elems ):
    self.basetopo = basetopo
    super().__init__( basetopo.ndims, elems )

  def getitem( self, item ):
    elements = []
    for elem in self.basetopo.getitem(item):
      try:
        ielem = self.edict[elem.transform]
      except KeyError:
        elem = elem.flipped
        try:
          ielem = self.edict[elem.transform]
        except KeyError:
          continue
      ref = self.elements[ielem].reference & elem.reference
      elements.append( element.Element( ref, elem.transform, elem.opposite, oriented=True ) )
    return UnstructuredTopology( self.ndims, elements )

class RefinedTopology( Topology ):
  'refinement'

  def __init__( self, basetopo ):
    self.basetopo = basetopo
    Topology.__init__( self, basetopo.ndims )

  def getitem( self, item ):
    return self.basetopo.getitem(item).refined

  @cache.property
  def elements( self ):
    return tuple([ child for elem in self.basetopo for child in elem.children ])

  @cache.property
  def boundary( self ):
    return self.basetopo.boundary.refined

class TrimmedTopologyItem( Topology ):
  'trimmed topology item'

  def __init__( self, basetopo, refdict ):
    self.basetopo = basetopo
    self.refdict = refdict
    Topology.__init__( self, basetopo.ndims )

  @cache.property
  def elements( self ):
    elements = []
    for elem in self.basetopo:
      ref = elem.reference & self.refdict[elem.transform]
      if ref:
        elements.append( element.Element( ref, elem.transform, elem.opposite, oriented=True ) )
    return elements

class TrimmedTopologyBoundaryItem( Topology ):
  'trimmed topology boundary item'

  def __init__( self, btopo, trimmed, othertopo ):
    self.btopo = btopo
    self.trimmed = trimmed
    self.othertopo = othertopo
    Topology.__init__( self, btopo.ndims )

  @cache.property
  def elements( self ):
    belems = [ elem for elem in self.trimmed if elem.opposite in self.btopo.edict ]
    if self.othertopo:
      belems.extend( self.othertopo )
    return belems

class HierarchicalTopology( Topology ):
  'collection of nested topology elments'

  def __init__( self, basetopo, allelements, precise ):
    'constructor'

    assert not isinstance( basetopo, HierarchicalTopology )
    self.basetopo = basetopo
    self.allelements = tuple(allelements)
    if precise:
      self.elements = self.allelements
    Topology.__init__( self, basetopo.ndims )

  def getitem( self, item ):
    return self.basetopo.getitem(item).hierarchical( self.allelements, precise=False )

  def hierarchical( self, elements, precise=False ):
    return self.basetopo.hierarchical( elements, precise )

  @cache.property
  def elements( self ):
    itemelems = []
    for elem in self.allelements:
      try:
        ielem, tail = elem.transform.lookup_item( self.basetopo.edict )
      except KeyError:
        continue
      itemelem = self.basetopo.elements[ielem]
      ref = itemelem.reference
      for trans in tail:
        index = ref.child_transforms.index((trans,))
        ref = ref.child_refs[ index ]
        if not ref:
          break
      else:
        ref &= elem.reference
        if ref:
          itemelems.append( element.Element( ref, elem.transform, elem.opposite, oriented=True ) )
    return itemelems

  @cache.property
  @log.title
  def levels( self ):
    levels = [ self.basetopo ]
    for elem in self:
      try:
        ielem, tail = elem.transform.lookup_item( self.basetopo.edict )
      except KeyError:
        raise Exception( 'element is not a refinement of basetopo' )
      else:
        nrefine = len(tail)
        while nrefine >= len(levels):
          levels.append( levels[-1].refined )
        assert elem.transform in levels[nrefine].edict, 'element is not a refinement of basetopo'
    return tuple(levels)

  @cache.property
  def refined( self ):
    elements = [ child for elem in self for child in elem.children ]
    return self.basetopo.hierarchical( elements, precise=True )

  @cache.property
  @log.title
  def boundary( self ):
    'boundary elements'

    basebtopo = self.basetopo.boundary
    edgepool = itertools.chain.from_iterable( elem.edges for elem in self if elem.transform.lookup( self.basetopo.border_transforms ) )
    belems = []
    for edge in edgepool: # superset of boundary elements
      try:
        iedge, tail = edge.transform.lookup_item( basebtopo.edict )
      except KeyError:
        pass
      else:
        opptrans = basebtopo.elements[iedge].opposite << tail
        belems.append( element.Element( edge.reference, edge.transform, opptrans, oriented=True ) )
    return basebtopo.hierarchical( belems, precise=True )

  @cache.property
  @log.title
  def interfaces( self ):
    'interfaces'

    # Build a lookup table for level and element indices given elements in this
    # topology.
    elem_index_level = {
      elem: (ielem, ilevel)
      for ilevel, level in enumerate( self.levels )
      for ielem, elem in enumerate( level )
    }
    oriented = isinstance( self.basetopo, StructuredTopology )
    edict = self.edict
    interfaces = []
    for elem in log.iter( 'elem', self ):
      # Get `level`, element number at `level` of `elem`.
      ielem, ilevel = elem_index_level[elem]
      level = self.levels[ilevel]
      # Loop over neighbours of `elem`.
      for ielemedge, ineighbor in enumerate( level.connectivity[ielem] ):
        if ineighbor < 0:
          # Not an interface.
          continue
        neighbor = level.elements[ineighbor]
        # Lookup `neighbor` (from the same `level` as `elem`) in this topology.
        head, tail = neighbor.transform.lookup( edict ) or (None, None)
        if not head:
          # `neighbor` not found, hence refinements of `neighbor` are present.
          # The interface of this edge will be added when we encounter the
          # refined elements.
          continue
        # Find the edge of `neighbor` between `neighbor` and `elem`.
        ineighboredge = numpy.where( level.connectivity[ineighbor] == ielem )[0][0]
        if not tail and (ielem, ielemedge) > (ineighbor, ineighboredge):
          # `neighbor` itself, not a parent of, exists in this topology (`tail`
          # is empty).  To make sure we add this interface only once we
          # continue here if the current element has a higher index (in
          # `level`) than the neighbor (or a higher edge number if the elements
          # are equal, which might occur when there is only one element in a
          # periodic dimension).
          continue
        # Create and add the interface between `elem` and `neighbor`.
        elemedge = elem.edges[ielemedge]
        neighboredge = neighbor.edges[ineighboredge]
        interfaces.append( element.Element( elemedge.reference, elemedge.transform, neighboredge.transform, oriented=oriented ) )
    return UnstructuredTopology( self.ndims-1, interfaces )

  @log.title
  def basis( self, name, *args, **kwargs ):
    'build hierarchical function space'

    # The law: a basis function is retained if all elements of self can
    # evaluate it through cascade, and at least one element of self can
    # evaluate it directly.

    # Procedure: per refinement level, track which basis functions have at
    # least one supporting element coinsiding with self ('touched') and no
    # supporting element finer than self ('supported').

    funcs = []
    dofmaps = []
    supports = []
    length = 0

    for topo in log.iter( 'level', self.levels ):

      basis = topo.basis( name, *args, **kwargs ) # shape functions for current level

      supported = numpy.ones( len(basis), dtype=bool ) # True if dof is fully contained in self or parents
      touchtopo = numpy.zeros( len(basis), dtype=bool ) # True if dof touches at least one elem in self

      (axes,func), = function.blocks( basis )
      dofmap, = axes
      for elem in topo:
        trans = elem.transform
        idofs, = dofmap.eval( elem )
        if trans in self.edict:
          touchtopo[idofs] = True
        elif trans.lookup( self.edict ):
          supported[idofs] = False

      funcs.append( func )
      dofmaps.append( dofmap + length )
      supports.append( supported & touchtopo )
      length += len( supported )

    funcs = function.concatenate( funcs, axis=0 )
    dofmaps = function.concatenate( dofmaps, axis=0 )
    supports = numpy.concatenate( supports, axis=0 )
    return function.mask( function.inflate( funcs, dofmaps, length, 0 ), supports )

class ProductTopology( Topology ):
  'product topology'

  def __init__( self, topo1, topo2 ):
    self.topo1 = topo1
    self.topo2 = topo2
    Topology.__init__( self, topo1.ndims+topo2.ndims )

  def __len__( self ):
    return len(self.topo1) * len(self.topo2)

  @cache.property
  def structure( self ):
    return self.topo1.structure[(...,)+(_,)*self.topo2.ndims] * self.topo2.structure

  @cache.property
  def elements( self ):
    return ( numpy.array( self.topo1.elements, dtype=object )[:,_] * numpy.array( self.topo2.elements, dtype=object )[_,:] ).ravel()

  def __iter__( self ):
    return self.elements.flat

  @property
  def refined( self ):
    return self.topo1.refined * self.topo2.refined

  def refine( self, n ):
    if numpy.iterable( n ):
      assert len(n) == self.ndims
    else:
      n = (n,)*self.ndims
    return self.topo1.refine( n[:self.topo1.ndims] ) * self.topo2.refine( n[self.topo1.ndims:] )

  def getitem( self, item ):
    return self.topo1.getitem(item) * self.topo2 | self.topo1 * self.topo2.getitem(item) if isinstance( item, str ) \
      else self.topo1[item[:self.topo1.ndims]] * self.topo2[item[self.topo1.ndims:]]

  def basis( self, name, *args, **kwargs ):
    def _split( arg ):
      if isinstance( arg, (list,tuple) ):
        assert len(arg) == self.ndims
        return arg[:self.topo1.ndims], arg[self.topo1.ndims:]
      else:
        return arg, arg
    splitargs = [ _split(arg) for arg in args ]
    splitkwargs = [ (name,)+_split(arg) for name, arg in kwargs.items() ]
    basis1, basis2 = function.bifurcate(
      self.topo1.basis( name, *[ arg1 for arg1, arg2 in splitargs ], **{ name: arg1 for name, arg1, arg2 in splitkwargs } ),
      self.topo2.basis( name, *[ arg2 for arg1, arg2 in splitargs ], **{ name: arg2 for name, arg1, arg2 in splitkwargs } ) )
    return function.ravel( function.outer(basis1,basis2), axis=0 )

  @cache.property
  def boundary( self ):
    return self.topo1 * self.topo2.boundary + self.topo1.boundary * self.topo2

  @cache.property
  def interfaces( self ):
    return self.topo1 * self.topo2.interfaces + self.topo1.interfaces * self.topo2

class RevolutionTopology( Topology ):
  'topology consisting of a single revolution element'

  def __init__( self ):
    self.elements = element.Element( element.RevolutionReference(), transform.roottrans('angle',(1,)) ),
    self.boundary = EmptyTopology( ndims=0 )
    Topology.__init__( self, ndims=1 )

  def basis( self, name, *args, **kwargs ):
    return function.asarray( [1] )

# UTILITY FUNCTIONS

DimAxis = collections.namedtuple( 'DimAxis', ['i','j','isperiodic'] )
DimAxis.isdim = True
BndAxis = collections.namedtuple( 'BndAxis', ['i','j','ibound','side'] )
BndAxis.isdim = False

def common_refine( topo1, topo2 ):
  assert topo1.ndims == topo2.ndims
  elements = []
  select2 = numpy.ones( len(topo2), dtype=bool )
  for elem1 in topo1:
    try:
      ielem2, tail = elem1.transform.lookup_item( topo2.edict )
    except KeyError:
      pass
    else:
      elements.append( elem1 )
      select2[ielem2] = False
  elements.extend( elem for ielem, elem in enumerate(topo2) if select2[ielem] )
  return UnstructuredTopology( topo1.ndims, elements )

# vim:shiftwidth=2:softtabstop=2:expandtab:foldmethod=indent:foldnestmax=2
