# Copyright (c) 2014 Evalf
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

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

from . import element, function, util, numpy, parallel, matrix, log, core, numeric, cache, transform, _
import warnings, functools, collections.abc, itertools, functools, operator

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

  def __and__(self, other):
    # Strategy: loop over combined elements sorted by .transform while keeping
    # track of the origin (mine=True for self, mine=False for other), and
    # select an element if it is equal to or a refinement of the previous
    # (hold) element and it originates from the other topology (mine == need).
    # Hold is not updated in case of a match because it might match multiple
    # children.
    elems = []
    need = None
    for elem, mine in sorted([(elem, True) for elem in self] + [(elem, False) for elem in other], key=lambda v: v[0].transform):
      if mine == need and elem.transform[:len(hold.transform)] == hold.transform:
        assert elem.opposite[:len(hold.opposite)] == hold.opposite
        elems.append(elem)
      else:
        hold = elem
        need = not mine
    return UnstructuredTopology(self.ndims, elems)

  __rand__ = lambda self, other: self.__and__(other)

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
        ielem, tail = transform.lookup_item(belem.transform, self.edict)
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
  def elem_eval( self, funcs, ischeme, separate=False, geometry=None, asfunction=False, edit=_identity, *, arguments=None ):
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
        ipoints, iweights = ischeme[elem] if isinstance(ischeme,collections.abc.Mapping) else fcache[elem.reference.getischeme]( ischeme )
        np = len( ipoints )
        slices.append( slice(npoints,npoints+np) )
        npoints += np

    nprocs = min( core.getprop( 'nprocs', 1 ), len(self) )
    zeros = parallel.shzeros if nprocs > 1 else numpy.zeros
    retvals = []
    idata = []
    for ifunc, func in enumerate( funcs ):
      func = function.asarray( edit( func * iwscale ) )
      func = function.zero_argument_derivatives(func)
      retval = zeros( (npoints,)+func.shape, dtype=func.dtype )
      idata.extend( function.Tuple([ifunc, function.Tuple(ind), f.simplified]) for ind, f in function.blocks(func) )
      retvals.append( retval )
    idata = function.Tuple( idata )

    if core.getprop( 'dot', False ):
      idata.graphviz()

    if arguments is None:
      arguments = {}

    for ielem, elem in parallel.pariter( log.enumerate( 'elem', self ), nprocs=nprocs ):
      ipoints, iweights = ischeme[elem] if isinstance(ischeme,collections.abc.Mapping) else fcache[elem.reference.getischeme]( ischeme )
      s = slices[ielem],
      try:
        for ifunc, index, data in idata.eval(_transforms=(elem.transform, elem.opposite), _points=ipoints, _cache=fcache, **arguments):
          numpy.add.at(retvals[ifunc], s+numpy.ix_(*[ ind for (ind,) in index ]), numeric.dot(iweights,data) if geometry else data)
      except function.EvaluationError:
        warnings.warn('not all functions evaluated successfully')
        for ifunc, indexfunc, datafunc in idata:
          index = indexfunc.eval(_transforms=(elem.transform, elem.opposite), _points=ipoints, _cache=fcache, **arguments)
          try:
            data = datafunc.eval(_transforms=(elem.transform, elem.opposite), _points=ipoints, _cache=fcache, **arguments)
          except function.EvaluationError:
            retvals[ifunc][s+numpy.ix_(*[ ind for (ind,) in index ])] = numpy.nan
          else:
            numpy.add.at(retvals[ifunc], s+numpy.ix_(*[ ind for (ind,) in index ]), numeric.dot(iweights,data) if geometry else data)

    log.debug( 'cache', fcache.stats )
    log.info( 'created', ', '.join( '%s(%s)' % ( retval.__class__.__name__, ','.join( str(n) for n in retval.shape ) ) for retval in retvals ) )

    if asfunction:
      if geometry:
        retvals = [ function.elemwise( { elem.transform: value for elem, value in zip( self, retval ) }, shape=retval.shape[1:] ) for retval in retvals ]
      else:
        tsp = [ ( elem.transform, s, fcache[elem.reference.getischeme](ischeme)[0] ) for elem, s in zip( self, slices ) ]
        retvals = [ function.sampled( { trans: (numeric.const(retval[s], copy=False), points) for trans, s, points in tsp }, self.ndims ) for retval in retvals ]
    elif separate:
      retvals = [ [ retval[s] for s in slices ] for retval in retvals ]

    return retvals

  @log.title
  @core.single_or_multiple
  def elem_mean( self, funcs, geometry, ischeme, *, arguments=None ):
    'element-wise average'

    retvals = self.elem_eval( (1,)+funcs, geometry=geometry, ischeme=ischeme, arguments=arguments )
    return [ v / retvals[0][(slice(None),)+(_,)*(v.ndim-1)] for v in retvals[1:] ]

  def _integrate( self, funcs, ischeme, fcache=None, arguments=None ):

    if arguments is None:
      arguments = {}

    # Functions may consist of several blocks, such as originating from
    # chaining. Here we make a list of all blocks consisting of triplets of
    # argument id, evaluable index, and evaluable values.

    blocks = [(ifunc, function.Tuple(ind), f.simplified)
      for ifunc, func in enumerate(funcs)
        for ind, f in function.blocks(function.zero_argument_derivatives(func))]

    block2func, indices, values = zip( *blocks ) if blocks else ([],[],[])

    log.debug( 'integrating %s distinct blocks' % '+'.join(
      str(block2func.count(ifunc)) for ifunc in range(len(funcs)) ) )

    if core.getprop( 'dot', False ):
      function.Tuple(values).graphviz()

    if fcache is None:
      fcache = cache.WrapperCache()

    # To allocate (shared) memory for all block data we evaluate indexfunc to
    # build an nblocks x nelems+1 offset array, and nblocks index lists of
    # length nelems.

    offsets = numpy.zeros((len(blocks), len(self)+1), dtype=int)
    if blocks:
      sizefunc = function.stack([f.size for ifunc, ind, f in blocks]).simplified
      for ielem, elem in enumerate(self):
        n, = sizefunc.eval(_transforms=(elem.transform, elem.opposite), _cache=fcache, **arguments)
        offsets[:,ielem+1] = offsets[:,ielem] + n

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

    valueindexfunc = function.Tuple(function.Tuple([value]+list(index)) for value, index in zip(values, indices))
    for ielem, elem in parallel.pariter( log.enumerate( 'elem', self ), nprocs=nprocs ):
      ipoints, iweights = ischeme[elem] if isinstance(ischeme,collections.abc.Mapping) else fcache[elem.reference.getischeme]( ischeme )
      assert iweights is not None, 'no integration weights found'
      for iblock, (intdata, *indices) in enumerate(valueindexfunc.eval(_transforms=(elem.transform, elem.opposite), _points=ipoints, _cache=fcache, **arguments)):
        s = slice(*offsets[iblock,ielem:ielem+2])
        data, index = data_index[ block2func[iblock] ]
        w_intdata = numeric.dot( iweights, intdata )
        data[s] = w_intdata.ravel()
        si = (slice(None),) + (_,) * (w_intdata.ndim-1)
        for idim, (ii,) in enumerate(indices):
          index[idim,s].reshape(w_intdata.shape)[...] = ii[si]
          si = si[:-1]

    log.debug( 'cache', fcache.stats )

    return data_index

  @log.title
  @core.single_or_multiple
  def integrate( self, funcs, ischeme='gauss', degree=None, geometry=None, force_dense=False, fcache=None, edit=_identity, *, arguments=None ):
    'integrate'

    if degree is not None:
      ischeme += str(degree)
    iwscale = function.J( geometry, self.ndims ) if geometry else 1
    integrands = [ function.asarray( edit( func * iwscale ) ) for func in funcs ]
    data_index = self._integrate( integrands, ischeme, fcache, arguments )
    return [ matrix.assemble( data, index, integrand.shape, force_dense ) for integrand, (data,index) in zip( integrands, data_index ) ]

  @log.title
  def integral(self, func, ischeme='gauss', degree=None, geometry=None, edit=_identity):
    'integral'

    if degree is not None:
      ischeme += str(degree)
    iwscale = function.J(geometry, self.ndims) if geometry else 1
    integrand = edit(func * iwscale)
    from . import solver
    return solver.Integral([((self, ischeme), integrand)])

  def projection( self, fun, onto, geometry, **kwargs ):
    'project and return as function'

    weights = self.project( fun, onto, geometry, **kwargs )
    return onto.dot( weights )

  @log.title
  def project( self, fun, onto, geometry, tol=0, ischeme='gauss', degree=None, droptol=1e-12, exact_boundaries=False, constrain=None, verify=None, ptype='lsqr', precon='diag', edit=_identity, *, arguments=None, **solverargs ):
    'L2 projection of function onto function space'

    log.debug( 'projection type:', ptype )

    if degree is not None:
      ischeme += str(degree)
    if constrain is None:
      constrain = util.NanVec( onto.shape[0] )
    else:
      constrain = constrain.copy()
    if exact_boundaries:
      constrain |= self.boundary.project( fun, onto, geometry, constrain=constrain, title='boundaries', ischeme=ischeme, tol=tol, droptol=droptol, ptype=ptype, edit=edit, arguments=arguments )
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
      A, b, f2, area = self.integrate( [Afun,bfun,fun2,1], geometry=geometry, ischeme=ischeme, edit=edit, arguments=arguments, title='building system' )
      N = A.rowsupp(droptol)
      if numpy.equal(b, 0).all():
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
      u, scale = self.integrate( [ ufun, afun ], geometry=geometry, ischeme=ischeme, edit=edit, arguments=arguments )
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
      fun = function.zero_argument_derivatives(function.asarray( fun ))
      data = function.Tuple(function.Tuple([fun, onto_f.simplified, function.Tuple(onto_ind)]) for onto_ind, onto_f in function.blocks(function.zero_argument_derivatives(onto)))
      for elem in self:
        ipoints, iweights = elem.getischeme('bezier2')
        for fun_, onto_f_, onto_ind_ in data.eval(_transforms=(elem.transform, elem.opposite), _points=ipoints, **arguments or {}):
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
  def trim( self, levelset, maxrefine, ndivisions=8, name='trimmed', leveltopo=None, *, arguments=None ):
    'trim element along levelset'

    if arguments is None:
      arguments = {}

    fcache = cache.WrapperCache()
    levelset = function.zero_argument_derivatives(levelset).simplified
    if leveltopo is None:
      ischeme = 'vertex{}'.format(maxrefine)
      refs = [elem.reference.trim(levelset.eval(_transforms=(elem.transform, elem.opposite), _points=fcache[elem.reference.getischeme](ischeme)[0], _cache=fcache, **arguments), maxrefine=maxrefine, ndivisions=ndivisions) for elem in log.iter('elem', self)]
    else:
      log.info( 'collecting leveltopo elements' )
      bins = [ [] for ielem in range(len(self)) ]
      for elem in leveltopo:
        ielem, tail = transform.lookup_item(elem.transform, self.edict)
        bins[ielem].append( tail )
      refs = []
      for elem, ctransforms in log.zip( 'elem', self, bins ):
        levels = numpy.empty( elem.reference.nvertices_by_level(maxrefine) )
        cover = list(fcache[elem.reference.vertex_cover](tuple(sorted(ctransforms)), maxrefine))
        # confirm cover and greedily optimize order
        mask = numpy.ones( len(levels), dtype=bool )
        while mask.any():
          imax = numpy.argmax([ mask[indices].sum() for trans, points, indices in cover ])
          trans, points, indices = cover.pop( imax )
          levels[indices] = levelset.eval(_transforms=(elem.transform + trans,), _points=points, _cache=fcache, **arguments)
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
  def elem_project( self, funcs, degree, ischeme=None, check_exact=False, *, arguments=None ):

    if arguments is None:
      arguments = {}

    if ischeme is None:
      ischeme = 'gauss%d' % (degree*2)

    blocks = function.Tuple([function.Tuple([function.Tuple((function.Tuple(ind), f.simplified))
      for ind, f in function.blocks(function.zero_argument_derivatives(func))])
        for func in funcs])

    bases = {}
    extractions = [ [] for ifunc in range(len(funcs) ) ]

    for elem in log.iter( 'elem', self ):

      try:
        points, projector, basis = bases[ elem.reference ]
      except KeyError:
        points, weights = elem.reference.getischeme( ischeme )
        coeffs = elem.reference.get_poly_coeffs('bernstein', degree=degree)
        basis = function.Polyval(coeffs, function.POINTS, points.shape[1]).eval(_points=points)
        npoints, nfuncs = basis.shape
        A = numeric.dot( weights, basis[:,:,_] * basis[:,_,:] )
        projector = numpy.linalg.solve( A, basis.T * weights )
        bases[ elem.reference ] = points, projector, basis

      for ifunc, ind_val in enumerate(blocks.eval(_transforms=(elem.transform, elem.opposite), _points=points, **arguments)):

        if len(ind_val) == 1:
          (allind, sumval), = ind_val
        else:
          allind, where = zip( *[ numpy.unique( [ i for ind, val in ind_val for i in ind[iax] ], return_inverse=True ) for iax in range( funcs[ifunc].ndim ) ] )
          sumval = numpy.zeros( [ len(n) for n in (points,) + allind ] )
          for ind, val in ind_val:
            I, where = zip( *[ ( w[:len(n)], w[len(n):] ) for w, n in zip( where, ind ) ] )
            numpy.add.at(sumval, numpy.ix_(range(len(points)), *I), val)
          assert not any( where )

        ex = numeric.dot( projector, sumval )
        if check_exact:
          numpy.testing.assert_almost_equal( sumval, numeric.dot( basis, ex ), decimal=15 )

        extractions[ifunc].append(( allind, ex ))

    return extractions

  @log.title
  def volume( self, geometry, ischeme='gauss1', *, arguments=None ):
    return self.integrate( 1, geometry=geometry, ischeme=ischeme, arguments=arguments )

  @log.title
  def volume_check( self, geometry, ischeme='gauss1', decimal=15, *, arguments=None ):
    volume = self.volume( geometry, ischeme, arguments=arguments )
    zeros, volumes = self.boundary.integrate( [ geometry.normal(), geometry * geometry.normal() ], geometry=geometry, ischeme=ischeme, arguments=arguments )
    numpy.testing.assert_almost_equal( zeros, 0., decimal=decimal )
    numpy.testing.assert_almost_equal( volumes, volume, decimal=decimal )
    return volume

  def indicator(self, subtopo):
    if isinstance(subtopo, str):
      subtopo = self[subtopo]
    transforms = tuple(sorted(elem.transform for elem in self))
    values = numeric.const([int(trans in subtopo.edict) for trans in transforms])
    assert len(subtopo) == values.sum(0), '{} is not a proper subtopology of {}'.format(subtopo, self)
    return function.Get(values, axis=0, item=function.FindTransform(transforms, function.Promote(self.ndims, trans=function.TRANS)))

  def select( self, indicator, ischeme='bezier2', *, arguments=None ):
    values = self.elem_eval( indicator, ischeme, separate=True, arguments=arguments )
    selected = [elem for elem, value in zip( self, values ) if numpy.greater(value, 0).any()]
    return UnstructuredTopology( self.ndims, selected )

  def prune_basis( self, basis ):
    used = numpy.zeros( len(basis), dtype=bool )
    for axes, func in function.blocks( basis ):
      dofmap = axes[0]
      for elem in self:
        dofs = dofmap.eval(_transforms=(elem.transform, elem.opposite))
        used[dofs] = True
    return function.mask( basis, used )

  def locate( self, geom, points, ischeme='vertex', scale=1, tol=1e-12, eps=0, maxiter=100, *, arguments=None ):
    nprocs = min( core.getprop( 'nprocs', 1 ), len(self) )
    if arguments is None:
      arguments = {}
    if geom.ndim == 0:
      geom = geom[_]
      points = points[...,_]
    assert geom.shape == (self.ndims,)
    points = numpy.asarray( points, dtype=float )
    assert points.ndim == 2 and points.shape[1] == self.ndims
    vertices = self.elem_eval( geom, ischeme=ischeme, separate=True, arguments=arguments )
    bboxes = numpy.array([ numpy.mean(v,axis=0) * (1-scale) + numpy.array([ numpy.min(v,axis=0), numpy.max(v,axis=0) ]) * scale
      for v in vertices ]) # nelems x {min,max} x ndims
    vref = element.getsimplex(0)
    ielems = parallel.shzeros(len(points), dtype=int)
    xis = parallel.shzeros((len(points),len(geom)), dtype=float)
    for ipoint, point in parallel.pariter(log.enumerate('point', points), nprocs=nprocs):
      ielemcandidates, = numpy.logical_and(numpy.greater_equal(point, bboxes[:,0,:]), numpy.less_equal(point, bboxes[:,1,:])).all(axis=-1).nonzero()
      for ielem in sorted( ielemcandidates, key=lambda i: numpy.linalg.norm(bboxes[i].mean(0)-point) ):
        converged = False
        elem = self.elements[ielem]
        xi, w = elem.reference.getischeme( 'gauss1' )
        xi = ( numpy.dot(w,xi) / w.sum() )[_] if len(xi) > 1 else xi.copy()
        J = function.localgradient( geom, self.ndims )
        geom_J = function.Tuple(( function.zero_argument_derivatives(geom), function.zero_argument_derivatives(J) )).simplified
        for iiter in range( maxiter ):
          point_xi, J_xi = geom_J.eval(_transforms=(elem.transform, elem.opposite), _points=xi, **arguments)
          err = numpy.linalg.norm( point - point_xi )
          if err < tol:
            converged = True
            break
          if iiter and err > prev_err:
            break
          prev_err = err
          xi += numpy.linalg.solve( J_xi, point - point_xi )
        if converged and elem.reference.inside( xi[0], eps=eps ):
          ielems[ipoint] = ielem
          xis[ipoint], = xi
          break
      else:
        raise LocateError( 'failed to locate point: {}'.format(point) )
    
    pelems = []
    for ielem, xi in zip(ielems, xis):
      elem = self.elements[ielem]
      trans = transform.Shift(xi),
      for idim in range(self.ndims,0,-1): # transcend dimensions one by one to produce valid transformation
        trans += transform.Updim(linear=numpy.eye(idim)[:,:-1], offset=numpy.zeros(idim), isflipped=False),
      pelems.append( element.Element( vref, elem.transform + trans, elem.opposite and elem.opposite + trans, oriented=True ) )
    return UnstructuredTopology(0, pelems)

  def supp( self, basis, mask=None ):
    if mask is None:
      mask = numpy.ones( len(basis), dtype=bool )
    elif isinstance(mask, list) or numeric.isarray(mask) and mask.dtype == int:
      tmp = numpy.zeros( len(basis), dtype=bool )
      tmp[mask] = True
      mask = tmp
    else:
      assert numeric.isarray(mask) and mask.dtype == bool and mask.shape == basis.shape[:1]
    indfunc = function.Tuple([ind[0] for ind, f in function.blocks(basis)])
    subset = []
    for elem in self:
      try:
        ind, = numpy.concatenate(indfunc.eval(_transforms=(elem.transform, elem.opposite)), axis=1)
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
    simplify = cache.replace(lambda op: function.zeros(()) if op is angle else None)
    return revdomain, revgeom, simplify

  def extruded( self, geom, nelems, periodic=False, bnames=('front','back') ):
    assert geom.ndim == 1
    root = transform.RootTrans( 'extrude', shape=[ nelems if periodic else 0 ] )
    extopo = self * StructuredLine( root, i=0, j=nelems, periodic=periodic, bnames=bnames )
    exgeom = function.concatenate( function.bifurcate( geom, function.rootcoords(1) ) )
    return extopo, exgeom

class LocateError( Exception ):
  pass

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
      return self.basetopo.getitem(item)
    try:
      itemtopo = self.vgroups[item]
    except KeyError:
      return self.basetopo.getitem(item)
    else:
      return itemtopo if isinstance( itemtopo, Topology ) else self.basetopo[itemtopo]

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
    assert trans[-1].fromdims == 0
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
    assert isinstance( root, transform.TransformItem )
    self.root = root
    self.i = i
    self.j = j
    self.periodic = periodic
    self.bnames = bnames or ()
    Topology.__init__( self, ndims=1 )

  @cache.property
  def _transforms( self ):
    # one extra left and right for opposites, even if periodic=True
    return tuple((self.root, transform.Shift([float(offset)])) for offset in range(self.i-1, self.j+1))

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
    left = transform.Updim(numpy.zeros((1,0)), offset=[0.], isflipped=True),
    right = transform.Updim(numpy.zeros((1,0)), offset=[1.], isflipped=False),
    bnd = Point( transforms[1] + left, transforms[0] + right ), Point( transforms[-2] + right, transforms[-1] + left )
    return UnionTopology( bnd, self.bnames )

  @cache.property
  def interfaces( self ):
    transforms = self._transforms
    left = transform.Updim(numpy.zeros((1,0)), offset=[0.], isflipped=True),
    right = transform.Updim(numpy.zeros((1,0)), offset=[1.], isflipped=False),
    points = [ Point( trans + left, opp + right ) for trans, opp in zip( transforms[2:-1], transforms[1:-2] ) ]
    if self.periodic:
      points.append( Point( transforms[1] + left, transforms[-2] + right ) )
    return UnionTopology( points )

  @classmethod
  def _bernstein_poly(cls, degree):
    'bernstein polynomial coefficients'


  @classmethod
  def _spline_coeffs(cls, p, n):
    'spline polynomial coefficients'

    assert p >= 0, 'invalid polynomial degree %d' % p
    if p == 0:
      assert n == -1
      return numpy.array([[[1.]]])

    assert 1 <= n < 2*p
    extractions = numpy.empty((n, p+1, p+1))
    extractions[0] = numpy.eye(p+1)
    for i in range(1, n):
      extractions[i] = numpy.eye(p+1)
      for j in range(2, p+1):
        for k in reversed(range(j, p+1)):
          alpha = 1. / min(2+k-j, n-i+1)
          extractions[i-1,:,k] = alpha * extractions[i-1,:,k] + (1-alpha) * extractions[i-1,:,k-1]
        extractions[i,-j-1:-1,-j-1] = extractions[i-1,-j:,-1]

    # magic bernstein triangle
    poly = numpy.zeros([p+1,p+1], dtype=int)
    for k in range(p//2+1):
      poly[k,k] = root = (-1)**p if k == 0 else (poly[k-1,k] * (k*2-1-p)) / k
      for i in range(k+1,p+1-k):
        poly[i,k] = poly[k,i] = root = (root * (k+i-p-1)) / i
    poly = poly[::-1].astype(float)

    return numeric.const(numeric.contract(extractions[:,_,:,:], poly[_,:,_,:], axis=-1).transpose(0,2,1), copy=False)

  def basis_spline(self, degree, periodic=None, removedofs=None):
    'spline from vertices'

    if periodic is None:
      periodic = self.periodic

    if numpy.iterable( degree ):
      degree, = degree

    if numpy.iterable( removedofs ):
      removedofs, = removedofs

    strides = 1, 1
    shape = len(self), degree+1
    ndofs = sum(s*(n-1) for s, n in zip(strides, shape))+1
    dofs = numpy.arange(ndofs)
    if periodic and degree > 0:
      assert ndofs >= 2 * degree
      dofs[-degree:] = dofs[:degree]
      ndofs -= degree
    dofs = numpy.lib.stride_tricks.as_strided(dofs, shape=shape, strides=tuple(s*dofs.strides[0] for s in strides))
    dofs = numeric.const(dofs, copy=False)

    p = degree
    n = 2*p-1
    nelems = len(self)
    if periodic:
      if nelems == 1: # periodicity on one element can only mean a constant
        coeffs = [self._spline_coeffs(0, n)]
        dofs = numeric.const([[0]], copy=False)
      else:
        coeffs = list(self._spline_coeffs(p, n)[p-1:p]) * nelems
    else:
      coeffs = list(self._spline_coeffs(p, min(nelems,n)))
      if len(coeffs) < nelems:
        coeffs = coeffs[:p-1] + coeffs[p-1:p] * (nelems-2*(p-1)) + coeffs[p:]
    coeffs = numeric.const(coeffs, copy=False)

    func = function.polyfunc(coeffs, dofs, ndofs, self._transforms[1:-1], issorted=False)
    if not removedofs:
      return func

    mask = numpy.ones( ndofs, dtype=bool )
    mask[list(removedofs)] = False
    return function.mask( func, mask )

  def basis_discont( self, degree ):
    'discontinuous shape functions'

    ref = element.LineReference()
    coeffs = [ref.get_poly_coeffs('bernstein', degree=degree)]*len(self)
    ndofs = ref.get_ndofs(degree)
    dofs = numeric.const(numpy.arange(ndofs*len(self), dtype=int).reshape(len(self), ndofs), copy=False)
    return function.polyfunc(coeffs, dofs, ndofs*len(self), self._transforms[1:-1], issorted=False)

  def basis_std( self, degree, periodic=None, removedofs=None ):
    'spline from vertices'

    if periodic is None:
      periodic = self.periodic

    strides = max(1, degree), 1
    shape = len(self), degree+1
    ndofs = sum(s*(n-1) for s, n in zip(strides, shape))+1
    dofs = numpy.arange(ndofs)
    if periodic and degree > 0:
      dofs[-1] = dofs[0]
      ndofs -= 1
    dofs = numpy.lib.stride_tricks.as_strided(dofs, shape=shape, strides=tuple(s*dofs.strides[0] for s in strides))
    dofs = numeric.const(dofs, copy=False)

    coeffs = [element.LineReference().get_poly_coeffs('bernstein', degree=degree)]*len(self)
    func = function.polyfunc(coeffs, dofs, ndofs, self._transforms[1:-1], issorted=False)
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
    dimaxes = ( axis for axis in self.axes if axis.isdim )
    return tuple( idim for idim, axis in enumerate(dimaxes) if axis.isdim and axis.isperiodic )

  @staticmethod
  def mktransforms( axes, root, nrefine ):
    assert nrefine >= 0

    updim = []
    ndims = len(axes)
    active = numpy.ones( ndims, dtype=bool )
    for order, side, idim in sorted( (axis.ibound,axis.side,idim) for idim, axis in enumerate(axes) if not axis.isdim ):
      where = (numpy.arange(len(active))[active]==idim)
      matrix = numpy.eye(ndims)[:,~where]
      offset = where.astype(float) if side else numpy.zeros(ndims)
      updim.append(transform.Updim(matrix, offset, isflipped=(idim%2==1)==side))
      ndims -= 1
      active[idim] = False

    grid = [ numpy.arange(axis.i>>nrefine, ((axis.j-1)>>nrefine)+1) if axis.isdim else numpy.array([(axis.i-1 if axis.side else axis.j)>>nrefine]) for axis in axes ]
    indices = numeric.broadcast( *numeric.ix(grid) )
    transforms = numeric.asobjvector([transform.Shift(numpy.array(index, dtype=float))] for index in log.iter('elem', indices, indices.size)).reshape( indices.shape)

    if nrefine:
      shifts = numeric.broadcast( *numeric.ix( [0,.5] for axis in axes ) )
      scales = numeric.asobjvector( [transform.Scale(.5, shift)] for shift in shifts ).reshape( shifts.shape )
      for irefine in log.range( 'level', nrefine-1, -1, -1 ):
        offsets = numpy.array([ r[0] for r in grid ])
        grid = [ numpy.arange(axis.i>>irefine,((axis.j-1)>>irefine)+1) if axis.isdim else numpy.array([(axis.i-1 if axis.side else axis.j)>>irefine]) for axis in axes ]
        A = transforms[ numpy.broadcast_arrays( *numeric.ix( r//2-o for r, o in zip( grid, offsets ) ) ) ]
        B = scales[ numpy.broadcast_arrays( *numeric.ix( r%2 for r in grid ) ) ]
        transforms = A + B
      
    shape = tuple( axis.j - axis.i for axis in axes if axis.isdim )
    return numeric.asobjvector(transform.canonical([root] + trans + updim) for trans in log.iter('canonical', transforms.flat)).reshape(shape)

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

  def _basis_spline( self, degree, knotvalues=None, knotmultiplicities=None, periodic=None ):
    'spline with structure information'
    
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
    stdelems = []
    dofshape = []
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
          coeffs = numeric.const(self._localsplinebasis(lknots, p).T, copy=False)
          cache[key] = coeffs
        stdelems_i.append(coeffs[start:stop])
      stdelems.append(stdelems_i)

      numbers = numpy.arange(nd)
      if isperiodic:
        numbers = numpy.concatenate([numbers,numbers[:p]])
      vertex_structure = vertex_structure[...,_]*nd+numbers
      dofshape.append( nd )
      slices.append(slices_i)

    #Cache effectivity
    log.debug( 'Local knot vector cache effectivity: %d' % (100*(1.-len(cache)/float(sum(self.shape)))) )

    # deduplicate stdelems and compute tensorial products `unique` with indices `index`
    # such that unique[index[i,j]] == poly_outer_product(stdelems[0][i], stdelems[1][j])
    index = numpy.array(0)
    for stdelems_i in stdelems:
      unique_i = tuple(set(stdelems_i))
      unique = unique_i if not index.ndim \
        else [numeric.poly_outer_product(a, b) for a in unique for b in unique_i]
      index = index[...,_] * len(unique_i) + tuple(map(unique_i.index, stdelems_i))

    coeffs = [unique[i] for i in index.flat]
    dofmap = [numeric.const(vertex_structure[S].ravel(), copy=False) for S in itertools.product(*slices)]
    return coeffs, dofmap, dofshape

  def basis_spline( self, degree, knotvalues=None, knotmultiplicities=None, periodic=None, removedofs=None ):
    'spline basis'

    if removedofs == None:
      removedofs = [None] * self.ndims
    else:
      assert len(removedofs) == self.ndims

    coeffs, dofmap, dofshape = self._basis_spline(degree=degree, knotvalues=knotvalues, knotmultiplicities=knotmultiplicities, periodic=periodic)
    func = function.polyfunc(coeffs, dofmap, util.product(dofshape), (elem.transform for elem in self), issorted=False)
    if not any( removedofs ):
      return func

    mask = numpy.ones( (), dtype=bool )
    for idofs, ndofs in zip( removedofs, dofshape ):
      mask = mask[...,_].repeat( ndofs, axis=-1 )
      if idofs:
        mask[...,[ numeric.normdim(ndofs,idof) for idof in idofs ]] = False
    assert mask.shape == tuple(dofshape)
    return function.mask( func, mask.ravel() )

  def basis_bspline( self, *args, **kwargs ):
    warnings.warn( 'basis "bspline" has been merged with "spline"', DeprecationWarning )
    return self.basis_spline( *args, **kwargs )

  @staticmethod
  def _localsplinebasis ( lknots, p ):
  
    assert numeric.isarray(lknots), 'Local knot vector should be numpy array'
    assert len(lknots)==2*p, 'Expected 2*p local knots'
  
    #Based on Algorithm A2.2 Piegl and Tiller
    N    = [None]*(p+1)
    N[0] = numpy.poly1d([1.])
  
    if p > 0:
  
      assert numpy.less(lknots[:-1]-lknots[1:], numpy.spacing(1)).all(), 'Local knot vector should be non-decreasing'
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

    return numeric.const([Ni.coeffs for Ni in N]).T[::-1]

  def basis_discont(self, degree):
    'discontinuous shape functions'

    ref = util.product([element.LineReference()]*self.ndims)
    coeffs = [ref.get_poly_coeffs('bernstein', degree=degree)]*len(self)
    ndofs = ref.get_ndofs(degree)
    dofs = numeric.const(numpy.arange(ndofs*len(self), dtype=int).reshape(len(self), ndofs), copy=False)
    return function.polyfunc(coeffs, dofs, ndofs*len(self), (elem.transform for elem in self), issorted=False)

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

    lineref = element.LineReference()
    coeffs = [functools.reduce(numeric.poly_outer_product, (lineref.get_poly_coeffs('bernstein', degree=p) for p in degree))]*len(self)
    dofs = [numeric.const(vertex_structure[S].ravel(), copy=False) for S in numpy.broadcast(*numpy.ix_(*slices))]
    func = function.polyfunc(coeffs, dofs, numpy.product(dofshape), self._transform.ravel(), issorted=False)
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

  def basis_bubble( self ):
    'bubble from vertices'

    assert self.ndims == 2
    coeffs = numeric.const([[[  1, -1,  0,  0], [ -1,  0,  0,  0], [  0,  0,  0,  0], [  0,  0,  0,  0]],
                            [[  0,  0,  0,  0], [  1,  0,  0,  0], [  0,  0,  0,  0], [  0,  0,  0,  0]],
                            [[  0,  1,  0,  0], [  0,  0,  0,  0], [  0,  0,  0,  0], [  0,  0,  0,  0]],
                            [[  0,  0,  0,  0], [  0, 27,-27,  0], [  0,-27,  0,  0], [  0,  0,  0,  0]]])

    nmap = []
    dofmap = {}
    for ielem, elem in enumerate(self):
      assert isinstance( elem.reference, element.TriangleReference )
      assert elem.nverts == 3
      dofs = numpy.empty(4, dtype=int)
      for i, v in enumerate( elem.vertices ):
        dof = dofmap.get(v)
        if dof is None:
          dof = len(self) + len(dofmap)
          dofmap[v] = dof
        dofs[i] = dof
      dofs[3] = ielem
      nmap.append(numeric.const(dofs, copy=False))
    ndofs = len(self)+len(dofmap)

    return function.polyfunc([coeffs]*len(self), nmap, ndofs, (elem.transform for elem in self), issorted=False)

  def basis_spline( self, degree ):
    assert degree == 1
    return self.basis( 'std', degree )

  def basis_discont( self, degree ):
    'discontinuous shape functions'

    assert numeric.isint(degree) and degree >= 0
    coeffs = []
    nmap = []
    ndofs = 0
    for elem in self:
      elemcoeffs = elem.reference.get_poly_coeffs('bernstein', degree=degree)
      coeffs.append(elemcoeffs)
      nmap.append(numeric.const(ndofs + numpy.arange(len(elemcoeffs)), copy=False))
      ndofs += len(elemcoeffs)
    degrees = set(n-1 for c in coeffs for n in c.shape[1:])
    return function.polyfunc(coeffs, nmap, ndofs, (elem.transform for elem in self), issorted=False)

  def _basis_c0_structured(self, name, degree):
    'C^0-continuous shape functions with lagrange stucture'

    assert numeric.isint(degree) and degree >= 0

    if degree == 0:
      raise ValueError('Cannot build a C^0-continuous basis of degree 0.  Use basis \'discont\' instead.')

    nlocaldofs = 0
    elem_slices = []
    coeffs = []
    for elem in self:
      elem_coeffs = elem.reference.get_poly_coeffs(name, degree=degree)
      coeffs.append(elem_coeffs)
      elem_slices.append(slice(nlocaldofs, nlocaldofs+len(elem_coeffs)))
      nlocaldofs += len(elem_coeffs)
    dofmap = -numpy.ones([nlocaldofs], dtype=int)

    for ielem, elem in enumerate(self):
      # Loop over all neighbors of elem and merge dofs.
      for iedge, jelem in enumerate(self.connectivity[ielem]):
        if jelem < 0:
          continue
        jedge = self.connectivity[jelem].tolist().index(ielem)
        #if ielem < jelem or ielem == jelem and iedge < jedge:
        #  continue
        idofs = elem.reference.get_edge_dofs(degree, iedge)
        verts = tuple(elem.edge(iedge).vertices)
        neighbor = self.elements[jelem]
        neighbor_verts = tuple(neighbor.edge(jedge).vertices)
        jdofs = neighbor.reference.get_edge_dofs(degree, jedge)
        #jdofs = jdofs[neighbor.reference.edges[jedge][1].get_dof_transpose_map(degree, tuple(map(neighbor_verts.index, verts)))]
        foo = neighbor.reference.edges[jedge][1].get_dof_transpose_map(degree, tuple(map(neighbor_verts.index, verts)))
        for idof, j in zip(idofs, foo):
          ridof = idof = elem_slices[ielem].start+idof
          # Resolve idof.
          while dofmap[ridof] >= 0:
            ridof = dofmap[ridof]
          # Resolve jdof.
          rjdof = jdof = elem_slices[jelem].start+jdofs[j]
          while dofmap[rjdof] >= 0:
            rjdof = dofmap[rjdof]
          if ridof != rjdof:
            dofmap[max(ridof,rjdof)] = min(ridof,rjdof)
          if ridof != idof:
            dofmap[idof] = min(ridof,rjdof)
    # Assign dof numbers.
    ndofs = 0
    for i in range(len(dofmap)):
      if dofmap[i] < 0:
        dofmap[i] = ndofs
        ndofs += 1
      else:
        dofmap[i] = dofmap[dofmap[i]]

    dofs = tuple(numeric.const(dofmap[s]) for s in elem_slices)
    return function.polyfunc(coeffs, dofs, ndofs, (elem.transform for elem in self), issorted=False)

  def basis_lagrange(self, degree):
    'lagrange shape functions'
    return self._basis_c0_structured('lagrange', degree)

  def basis_bernstein(self, degree):
    'bernstein shape functions'
    return self._basis_c0_structured('bernstein', degree)

  basis_std = basis_bernstein

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
    if self.basetopo == other:
      refs = [ elem.reference - ref for elem, ref in zip( self.basetopo, self.refs ) ]
      return SubsetTopology( self.basetopo, refs, ~self.newboundary if isinstance(self.newboundary,Topology) else self.newboundary )
    return super().__rsub__(other)

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
      newbelems.extend( element.Element(edge, elem.transform + (etrans,), elem.transform + (etrans.flipped,)) for etrans, edge in ref.edges[elem.reference.nedges:] )
      for iedge, ioppelem in enumerate( ioppelems ):
        bref = ref.edge_refs[iedge]
        if not bref:
          continue
        if ioppelem == -1:
          index = baseboundary.edict[transform.canonical(elem.transform + (ref.edge_transforms[iedge],))]
          brefs[index] = bref # by construction, bref must be equal or subset of original
        else:
          ioppedge = tuple(connectivity[ioppelem]).index(ielem)
          oppref = self.refs[ioppelem]
          if oppref:
            bref -= oppref.edge_refs[ioppedge]
          if bref:
            newbelems.append(element.Element(bref, elem.transform + (ref.edge_transforms[iedge],), elements[ioppelem].edge(ioppedge).transform))
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
        index = baseinterfaces.edict.get(transform.canonical(elem.transform + (ref.edge_transforms[iedge],)))
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
        ielem, tail = transform.lookup_item(elem.transform, self.edict)
      except KeyError:
        elem = elem.flipped
        try:
          ielem, tail = transform.lookup_item(elem.transform, self.edict)
        except KeyError:
          continue
      if tail:
        raise NotImplementedError
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
        ielem, tail = transform.lookup_item(elem.transform, self.basetopo.edict)
      except KeyError:
        continue
      itemelem = self.basetopo.elements[ielem]
      ref = itemelem.reference
      for trans in tail:
        index = ref.child_transforms.index(trans)
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
        ielem, tail = transform.lookup_item(elem.transform, self.basetopo.edict)
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
    edgepool = [edge for elem in self if transform.lookup(elem.transform, self.basetopo.border_transforms) for edge in elem.edges if edge is not None]
    belems = []
    for edge in edgepool: # superset of boundary elements
      try:
        iedge, tail = transform.lookup_item(edge.transform, basebtopo.edict)
      except KeyError:
        pass
      else:
        opptrans = basebtopo.elements[iedge].opposite + tail
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
        head, tail = transform.lookup(neighbor.transform, edict) or (None, None)
        if not head:
          # `neighbor` not found, hence refinements of `neighbor` are present.
          # The interface of this edge will be added when we encounter the
          # refined elements.
          continue
        # Find the edge of `neighbor` between `neighbor` and `elem`.
        ineighboredge = tuple(level.connectivity[ineighbor]).index(ielem)
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

    dofs_coeffs = []
    renumber = []
    supports = []
    length = 0

    for topo in log.iter( 'level', self.levels ):

      basis = topo.basis( name, *args, **kwargs ) # shape functions for current level

      supported = numpy.ones( len(basis), dtype=bool ) # True if dof is fully contained in self or parents
      touchtopo = numpy.zeros( len(basis), dtype=bool ) # True if dof touches at least one elem in self

      (axes,func), = function.blocks( basis )
      dofmap, = axes
      if isinstance(func, function.Polyval):
        coeffs = func.coeffs
        assert coeffs.ndim == 1+self.ndims
      elif func.isconstant:
        assert func.ndim == 1
        coeffs = func[(slice(None),*(_,)*self.ndims)]
      else:
        raise ValueError

      for elem in topo:
        trans = elem.transform
        idofs, = dofmap.eval(_transforms=(elem.transform, elem.opposite))
        if trans in self.edict:
          touchtopo[idofs] = True
        elif transform.lookup(trans, self.edict):
          supported[idofs] = False

      support = supported & touchtopo
      supports.append(support)
      cumsum_support = numpy.cumsum(support)
      renumber.append(cumsum_support+(length-1))
      length += cumsum_support[-1]
      dofs_coeffs.append(function.Tuple((dofmap, coeffs)))

    dofs = []
    coeffs = []
    transforms = tuple(sorted(elem.transform for elem in self))
    for trans in transforms:
      hcoeffs = []
      hdofs = []
      ibase, tail = transform.lookup_item(trans, self.basetopo.edict)
      for ilevel in range(len(tail)+1):
        (idofs,), (icoeffs,) = dofs_coeffs[ilevel].eval(_transforms=(trans,))
        isupport = supports[ilevel][idofs]
        if not isupport.any():
          continue
        hdofs.extend(map(renumber[ilevel].__getitem__, idofs[isupport]))
        hcoeffs.extend(transform.transform_poly(tail[ilevel:], icoeffs[isupport]))
      dofs.append(numeric.const(hdofs))
      coeffs.append(numeric.poly_stack(hcoeffs))

    return function.polyfunc(coeffs, dofs, length, transforms, issorted=True)

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
    self.elements = element.Element(element.RevolutionReference(), [transform.RootTrans('angle',(1,))]),
    self.boundary = EmptyTopology( ndims=0 )
    Topology.__init__( self, ndims=1 )

  def basis( self, name, *args, **kwargs ):
    return function.asarray( [1] )

class MultipatchTopology( Topology ):
  'multipatch topology'

  Patch = collections.namedtuple( 'Patch', [ 'topo', 'verts', 'boundaries'] )
  Patch.__qualname__ = 'MultipatchTopology.Patch'

  @staticmethod
  def build_boundarydata( connectivity ):
    'build boundary data based on connectivity'

    boundarydata = []
    for patch in connectivity:
      ndims = len( patch.shape )
      patchboundarydata = []
      for dim, side in itertools.product( range( ndims ), [-1, 0] ):
        # ignore vertices at opposite face
        verts = numpy.array( patch )
        opposite = tuple( {0:-1, -1:0}[side] if i == dim else slice(None) for i in range( ndims ) )
        verts[opposite] = verts.max()+1
        if len( set( verts.flat ) ) != 2**(ndims-1)+1:
          raise NotImplementedError( 'Cannot compute canonical boundary if vertices are used more than once.' )
        # reverse axes such that lowest vertex index is at first position
        reverse = tuple( slice(None, None, -1) if i else slice(None) for i in numpy.unravel_index(verts.argmin(), verts.shape) )
        verts = verts[reverse]
        # transpose such that second lowest vertex connects to lowest vertex in first dimension, third in second dimension, et cetera
        k = [ verts[tuple( 1 if i == j else 0 for j in range( ndims ) )] for i in range( ndims ) ]
        transpose = tuple( sorted( range( ndims ), key=k.__getitem__ ) )
        verts = verts.transpose( transpose )
        # boundarid
        boundaryid = tuple( verts[...,0].flat )
        patchboundarydata.append( (boundaryid,dim,side,reverse,transpose) )
      boundarydata.append( tuple( patchboundarydata ) )

    # TODO: boundary sanity checks

    return boundarydata

  def __init__( self, patches ):
    'constructor'

    self.patches = tuple( self.Patch(*patch) for patch in patches )

    self._patchinterfaces = {}
    for patch in self.patches:
      for boundaryid, dim, side, reverse, transpose in patch.boundaries:
        self._patchinterfaces.setdefault( boundaryid, [] ).append(( patch.topo, dim, side, reverse, transpose ))
    self._patchinterfaces = {
      boundaryid: tuple( data )
      for boundaryid, data in self._patchinterfaces.items()
      if len( data ) > 1
    }

    super().__init__( self.patches[0].topo.ndims )

  @cache.property
  def elements( self ):
    return tuple( itertools.chain.from_iterable( patch.topo for patch in self.patches ) )

  def getitem( self, key ):
    for i in range( len( self.patches ) ):
      if key == 'patch{}'.format(i):
        return self.patches[i].topo
    else:
      return UnionTopology( patch.topo.getitem(key) for patch in self.patches )

  def basis_spline( self, degree, patchcontinuous=True, knotvalues=None, knotmultiplicities=None ):
    '''spline from vertices

    Create a spline basis with degree ``degree`` per patch.  If
    ``patchcontinuous``` is true the basis is $C^0$-continuous at patch
    interfaces.
    '''

    if knotvalues is None:
      knotvalues = {None: None}
    else:
      knotvalues, _knotvalues = {}, knotvalues
      for edge, k in _knotvalues.items():
        if k is None:
          rk = None
        else:
          k = tuple(k)
          rk = k[::-1]
        if edge is None:
          knotvalues[edge] = k
        else:
          l, r = edge
          assert (l,r) not in knotvalues
          assert (r,l) not in knotvalues
          knotvalues[(l,r)] = k
          knotvalues[(r,l)] = rk

    if knotmultiplicities is None:
      knotmultiplicities = {None: None}
    else:
      knotmultiplicities, _knotmultiplicities = {}, knotmultiplicities
      for edge, k in _knotmultiplicities.items():
        if k is None:
          rk = None
        else:
          k = tuple(k)
          rk = k[::-1]
        if edge is None:
          knotmultiplicities[edge] = k
        else:
          l, r = edge
          assert (l,r) not in knotmultiplicities
          assert (r,l) not in knotmultiplicities
          knotmultiplicities[(l,r)] = k
          knotmultiplicities[(r,l)] = rk

    missing = object()

    coeffs = []
    dofmap = []
    transforms = []
    dofcount = 0
    commonboundarydofs = {}
    for ipatch, patch in enumerate( self.patches ):
      transforms.extend(elem.transform for elem in patch.topo)
      # build structured spline basis on patch `patch.topo`
      patchknotvalues = []
      patchknotmultiplicities = []
      for idim in range( self.ndims ):
        left = tuple( 0 if j == idim else slice(None) for j in range( self.ndims ) )
        right = tuple( 1 if j == idim else slice(None) for j in range( self.ndims ) )
        dimknotvalues = set()
        dimknotmultiplicities = set()
        for edge in zip( patch.verts[left].flat, patch.verts[right].flat ):
          v = knotvalues.get( edge, knotvalues.get( None, missing ) )
          m = knotmultiplicities.get( edge, knotmultiplicities.get( None, missing ) )
          if v is missing:
            raise 'missing edge'
          dimknotvalues.add(v)
          if m is missing:
            raise 'missing edge'
          dimknotmultiplicities.add(m)
        if len(dimknotvalues) != 1:
          raise 'ambiguous knot values for patch {}, dimension {}'.format( ipatch, idim )
        if len(dimknotmultiplicities) != 1:
          raise 'ambiguous knot multiplicities for patch {}, dimension {}'.format( ipatch, idim )
        patchknotvalues.append(next(iter(dimknotvalues)))
        patchknotmultiplicities.append(next(iter(dimknotmultiplicities)))
      patchcoeffs, patchdofmap, patchdofcount = patch.topo._basis_spline(degree, knotvalues=patchknotvalues, knotmultiplicities=patchknotmultiplicities)
      coeffs.extend(patchcoeffs)
      dofmap.extend(numeric.const(dofs+dofcount, copy=False) for dofs in patchdofmap)
      if patchcontinuous:
        # reconstruct multidimensional dof structure
        dofs = dofcount + numpy.arange( numpy.prod( patchdofcount ), dtype=int ).reshape( patchdofcount )
        for boundaryid, dim, side, reverse, transpose in patch.boundaries:
          # get patch boundary dofs and reorder to canonical form
          boundarydofs = dofs[reverse].transpose(transpose)[...,0].ravel()
          # append boundary dofs to list (in increasing order, automatic by outer loop and dof increment)
          commonboundarydofs.setdefault( boundaryid, [] ).append( boundarydofs )
      dofcount += numpy.prod( patchdofcount )

    if patchcontinuous:
      # build merge mapping: merge common boundary dofs (from low to high)
      pairs = itertools.chain(*( zip( *dofs ) for dofs in commonboundarydofs.values() if len( dofs ) > 1 ))
      merge = {}
      for dofs in sorted( pairs ):
        dst = merge.get( dofs[0], dofs[0] )
        for src in dofs[1:]:
          merge[src] = dst
      # build renumber mapping: renumber remaining dofs consecutively, starting at 0
      remainder = set( merge.get( dof, dof ) for dof in range( dofcount ) )
      renumber = dict( zip( sorted( remainder ), range( len( remainder ) ) ) )
      # apply mappings
      dofmap = tuple(numeric.const(tuple(renumber[merge.get(dof, dof)] for dof in v.flat), dtype=int).reshape(v.shape) for v in dofmap)
      dofcount = len(remainder)

    return function.polyfunc(coeffs, dofmap, dofcount, transforms, issorted=False)

  def basis_discont(self, degree):
    'discontinuous shape functions'

    bases = [patch.topo.basis('discont', degree=degree) for patch in self.patches]
    coeffs = []
    dofs = []
    ndofs = 0
    for patch in self.patches:
      basis = patch.topo.basis('discont', degree=degree)
      (axes,func), = function.blocks(basis)
      patch_dofmap, = axes
      if isinstance(func, function.Polyval):
        patch_coeffs = func.coeffs
        assert patch_coeffs.ndim == 1+self.ndims
      elif func.isconstant:
        assert func.ndim == 1
        patch_coeffs = func[(slice(None),*(_,)*self.ndims)]
      else:
        raise ValueError
      patch_coeffs_dofs = function.Tuple((patch_coeffs, patch_dofmap))
      for elem in patch.topo:
        (elem_coeffs,), (elem_dofs,) = patch_coeffs_dofs.eval(_transforms=(elem.transform,))
        coeffs.append(elem_coeffs)
        dofs.append(numeric.const(ndofs+elem_dofs, copy=False))
      ndofs += len(basis)
    return function.polyfunc(coeffs, dofs, ndofs, (elem.transform for patch in self.patches for elem in patch.topo), issorted=False)

  def basis_patch(self):
    'degree zero patchwise discontinuous basis'

    npatches = len(self.patches)
    coeffs = [numeric.const(1, dtype=int).reshape(1, *(1,)*self.ndims)]*npatches
    dofs = numeric.const(range(npatches), dtype=int)[:,_]
    return function.polyfunc(coeffs, dofs, npatches, ((patch.topo.root,) for patch in self.patches), issorted=False)

  @cache.property
  def boundary( self ):
    'boundary'

    subtopos = []
    subnames = []
    for i, patch in enumerate( self.patches ):
      names = dict( zip( itertools.product( range( self.ndims ), [0,-1] ), patch.topo._bnames ) )
      for boundaryid, dim, side, reverse, transpose in patch.boundaries:
        if boundaryid in self._patchinterfaces:
          continue
        subtopos.append( patch.topo.boundary[names[dim,side]] )
        subnames.append( 'patch{}-{}'.format( i, names[dim,side] ) )
    if len( subtopos ) == 0:
      return EmptyTopology( self.ndims-1 )
    else:
      return UnionTopology( subtopos, subnames )

  @cache.property
  def interfaces( self ):
    '''interfaces

    Return a topology with all element interfaces.  The patch interfaces are
    accessible via the group ``'interpatch'`` and the interfaces *inside* a
    patch via ``'intrapatch'``.
    '''

    intrapatchtopo = EmptyTopology( self.ndims-1 ) if not self.patches else \
      UnionTopology( patch.topo.interfaces for patch in self.patches )

    btopos = []
    bconnectivity = []
    for boundaryid, patchdata in self._patchinterfaces.items():
      if len( patchdata ) > 2:
        raise ValueError( 'Cannot create interfaces of multipatch topologies with more than two interface connections.' )
      pairs = []
      for topo, dim, side, reverse, transpose in patchdata:
        names = dict( zip( itertools.product( range( self.ndims ), [0,-1] ), topo._bnames ) )
        # get structured set of boundary elements
        elems = topo.boundary[names[dim, side]].structure
        # add singleton axis
        elems = elems[tuple( _ if i == dim else slice( None ) for i in range( self.ndims ) )]
        # apply canonical transformation
        elems = elems[reverse].transpose(transpose)[..., 0]
        shape = elems.shape
        pairs.append( elems.flat )
      # join element pairs
      elems = [
        element.Element( elem_a.reference, elem_a.transform, elem_b.transform, oriented=True )
        for elem_a, elem_b in zip( *pairs )
      ]
      # create structured topology of joined element pairs
      bpatch = numpy.array( boundaryid ).reshape( (2,)*(self.ndims-1) )
      #btopos.append( StructuredTopology( numpy.array( elems ).reshape( shape ) ) )
      btopos.append( UnstructuredTopology( self.ndims-1, elems ) )
      bconnectivity.append( bpatch )
    # create multipatch topology of interpatch boundaries
    interpatchtopo = MultipatchTopology( zip( btopos, bconnectivity, self.build_boundarydata( bconnectivity ) ) )

    return UnionTopology( (intrapatchtopo, interpatchtopo), ('intrapatch', 'interpatch') )

  @cache.property
  def refined( self ):
    'refine'

    return MultipatchTopology( (patch.topo.refined, patch.verts, patch.boundaries) for patch in self.patches )

# UTILITY FUNCTIONS

DimAxis = collections.namedtuple( 'DimAxis', ['i','j','isperiodic'] )
DimAxis.isdim = True
BndAxis = collections.namedtuple( 'BndAxis', ['i','j','ibound','side'] )
BndAxis.isdim = False

def common_refine(topo1, topo2):
  warnings.warn('common_refine(a, b) will be removed in future; use a & b instead', DeprecationWarning)
  return topo1 & topo2

# vim:shiftwidth=2:softtabstop=2:expandtab:foldmethod=indent:foldnestmax=2
