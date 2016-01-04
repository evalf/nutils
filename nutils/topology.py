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
import warnings, functools, collections, itertools

_identity = lambda x: x

class Topology( object ):
  'topology base class'

  # subclass needs to implement: .elements

  def __init__( self, ndims ):
    'constructor'

    assert numeric.isint( ndims ) and ndims >= 0
    self.ndims = ndims

  def __len__( self ):
    return len( self.elements )

  def __iter__( self ):
    return iter( self.elements )

  def __getitem__( self, item ):
    raise KeyError( item )

  def __invert__( self ):
    return OppositeTopology( self )

  def __or__( self, other ):
    assert isinstance( other, Topology ) and other.ndims == self.ndims
    return other if not self \
      else self if not other \
      else NotImplemented if isinstance( other, (ItemTopology,UnionTopology) ) \
      else UnionTopology( (self,other) )

  def __ror__( self, other ):
    return other.__or__( self )

  def __add__( self, other ):
    return self | other

  def __contains__( self, element ):
    ielem = self.edict.get(element.transform)
    return ielem is not None and self.elements[ielem] == element

  def __sub__( self, other ):
    assert isinstance( other, Topology ) and other.ndims == self.ndims
    return self if not other \
      else NotImplemented if isinstance( other, ItemTopology ) \
      else SubtractionTopology( self, other )

  def __mul__( self, other ):
    'element products'

    quad = element.getsimplex(1)**2
    ndims = self.ndims + other.ndims
    eye = numpy.eye( ndims, dtype=int )
    self_trans = transform.affine(eye[:self.ndims], numpy.zeros(self.ndims), isflipped=False )
    other_trans = transform.affine(eye[self.ndims:], numpy.zeros(other.ndims), isflipped=False )

    if any( elem.reference != quad for elem in self ) or any( elem.reference != quad for elem in other ):
      return UnstructuredTopology( self.ndims+other.ndims, [ element.Element( elem1.reference * elem2.reference, elem1.transform << self_trans, elem2.transform << other_trans )
        for elem1 in self for elem2 in other ] )

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
    return UnstructuredTopology( self.ndims+other.ndims, elements )

  @cache.property
  def edict( self ):
    '''transform -> ielement mapping'''
    return { elem.transform: ielem for ielem, elem in enumerate(self) }

  def outward_from( self, outward, inward=None ):
    'direct interface elements to evaluate in topo first'

    directed = []
    for iface in self:
      if not iface.transform.lookup( outward.edict ):
        assert iface.opposite.lookup( outward.edict ), 'interface not adjacent to outward topo'
        iface = element.Element( iface.reference, iface.opposite, iface.transform )
      if inward:
        assert iface.opposite.lookup( inward.edict ), 'interface no adjacent to inward topo'
      directed.append( iface )
    return UnstructuredTopology( self.ndims, directed )

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
    f = getattr( self, 'basis_' + name )
    return f( *args, **kwargs )

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
      fmap[elem.transform] = (stdfunc,None),
      nmap[elem.transform] = dofs
    return function.function( fmap=fmap, nmap=nmap, ndofs=len(dofmap), ndims=self.ndims )

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
      fmap[elem.transform] = (stdfunc,None),
      nmap[elem.transform] = dofs
    return function.function( fmap=fmap, nmap=nmap, ndofs=len(self)+len(dofmap), ndims=self.ndims )

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
      fmap[elem.transform] = (stdfunc,None),
      nmap[elem.transform] = ndofs + numpy.arange(stdfunc.nshapes)
      ndofs += stdfunc.nshapes
    return function.function( fmap=fmap, nmap=nmap, ndofs=ndofs, ndims=self.ndims )

  @log.title
  @core.single_or_multiple
  def elem_eval( self, funcs, ischeme, separate=False, geometry=None, asfunction=False, edit=_identity ):
    'element-wise evaluation'

    assert not separate or not asfunction, '"separate" and "asfunction" are mutually exclusive'
    if geometry:
      iwscale = function.J( geometry, self.ndims )
      npoints = len(self)
      slices = range(npoints)
    else:
      iwscale = 1
      slices = []
      pointshape = function.PointShape()
      npoints = 0
      for elem in log.iter( 'elem', self ):
        np, = pointshape.eval( elem, ischeme )
        slices.append( slice(npoints,npoints+np) )
        npoints += np

    retvals = []
    idata = []
    for ifunc, func in enumerate( funcs ):
      func = function.asarray( edit( func * iwscale ) )
      retval = parallel.shzeros( (npoints,)+func.shape, dtype=func.dtype )
      if function._isfunc( func ):
        for ind, f in function.blocks( func ):
          idata.append( function.Tuple([ ifunc, ind, f ]) )
      else:
        idata.append( function.Tuple([ ifunc, (), func ]) )
      retvals.append( retval )
    idata = function.Tuple( idata )

    fcache = cache.WrapperCache()
    for ielem, elem in parallel.pariter( log.enumerate( 'elem', self ) ):
      ipoints, iweights = fcache[elem.reference.getischeme]( ischeme )
      s = slices[ielem],
      for ifunc, index, data in idata.eval( elem, ipoints, fcache ):
        retvals[ifunc][s+numpy.ix_(*index)] += numeric.dot(iweights,data) if geometry else data

    log.debug( 'cache', fcache.stats )
    log.info( 'created', ', '.join( '%s(%s)' % ( retval.__class__.__name__, ','.join( str(n) for n in retval.shape ) ) for retval in retvals ) )

    if asfunction:
      if geometry:
        retvals = [ function.Elemwise( { elem.transform: value for elem, value in zip( self, retval ) }, shape=retval.shape[1:] ) for retval in retvals ]
      else:
        tsp = [ ( elem.transform, s, fcache[elem.reference.getischeme](ischeme)[0] ) for elem, s in zip( self, slices ) ]
        retvals = [ function.Sampled({ trans: (retval[s],points) for trans, s, points in tsp }) for retval in retvals ]
    elif separate:
      retvals = [ [ retval[s] for s in slices ] for retval in retvals ]

    return retvals

  @log.title
  @core.single_or_multiple
  def elem_mean( self, funcs, geometry, ischeme ):
    'element-wise average'

    retvals = self.elem_eval( (1,)+funcs, geometry=geometry, ischeme=ischeme )
    return [ v / retvals[0][(slice(None),)+(_,)*(v.ndim-1)] for v in retvals[1:] ]

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

    fcache = cache.WrapperCache()

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

    for ielem, elem in parallel.pariter( log.enumerate( 'elem', self ) ):
      ipoints, iweights = fcache[elem.reference.getischeme]( ischeme[elem] if isinstance(ischeme,dict) else ischeme )
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
  def integrate( self, funcs, ischeme, geometry=None, force_dense=False, edit=_identity ):
    'integrate'

    iwscale = function.J( geometry, self.ndims ) if geometry else 1
    integrands = [ function.asarray( edit( func * iwscale ) ) for func in funcs ]
    data_index = self._integrate( integrands, ischeme )
    return [ matrix.assemble( data, index, integrand.shape, force_dense ) for integrand, (data,index) in zip( integrands, data_index ) ]

  @log.title
  @core.single_or_multiple
  def integrate_symm( self, funcs, ischeme, geometry=None, force_dense=False, edit=_identity ):
    'integrate a symmetric integrand on a product domain' # TODO: find a proper home for this

    iwscale = function.J( geometry, self.ndims ) if geometry else 1
    integrands = [ function.asarray( edit( func * iwscale ) ) for func in funcs ]
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
    diag_data_index = UnstructuredTopology( self.ndims, diagelems )._integrate( integrands, ischeme )
    tri_data_index = UnstructuredTopology( self.ndims, trielems )._integrate( integrands, ischeme )
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

    if exact_boundaries:
      constrain |= self.boundary.project( fun, onto, geometry, constrain=constrain, title='boundaries', ischeme=ischeme, tol=tol, droptol=droptol, ptype=ptype, edit=edit )
    elif constrain is None:
      constrain = util.NanVec( onto.shape[0] )
    else:
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

    return self if n <= 0 else self.refined.refine( n-1 )

  @log.title
  def trim( self, levelset, maxrefine, ndivisions=8, name='trimmed' ):
    'trim element along levelset'

    fcache = cache.WrapperCache()
    elements = []
    for elem in log.iter( 'elem', self ):
      ref = elem.trim( levelset=levelset, maxrefine=maxrefine, ndivisions=ndivisions, fcache=fcache )
      if ref:
        elements.append( element.Element( ref, elem.transform, elem.opposite ) )
    log.debug( 'cache', fcache.stats )
    return self.subset( elements, name, precise=True )

  def subset( self, elements, boundaryname=None, precise=False ):
    return SubsetTopology( self, elements, boundaryname, precise )

  def withsubs( self, subtopos={} ):
    return ItemTopology( self, subtopos )

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
    return function.Elemwise( { elem.transform: 1. for elem in self }, (), default=0. )

  def select( self, indicator, ischeme='bezier2' ):
    values = self.elem_eval( indicator, ischeme, separate=True )
    selected = [ elem for elem, value in zip( self, values ) if numpy.any( value > 0 ) ]
    return UnstructuredTopology( self.ndims, selected )

  def prune_basis( self, basis ):
    used = numpy.zeros( len(basis), dtype=bool )
    for axes, func in function.blocks( basis ):
      dofmap = axes[0]
      for elem in self:
        used[ dofmap.dofmap[elem.transform] + dofmap.offset ] = True
    return basis[used]

class ItemTopology( Topology ):
  'item topology'

  def __init__( self, basetopo, subtopos ):
    Topology.__init__( self, basetopo.ndims )
    self.basetopo = basetopo
    self.subtopos = {}
    if subtopos and core.getprop( 'selfcheck', False ):
      for name, subtopo in subtopos.items():
        self[name] = subtopo
    else:
      self.subtopos = subtopos.copy()

  def __iter__( self ):
    return iter( self.basetopo )

  def __len__( self ):
    return len( self.basetopo )

  def __or__( self, other ):
    other = other.withsubs()
    subtopos = self.subtopos.copy()
    for name, topo in other.subtopos.items():
      if name in subtopos:
        subtopos[name] |= topo
      else:
        subtopos[name] = topo
    return ( self.basetopo | other.basetopo ).withsubs( subtopos )

  def __sub__( self, other ):
    othertopo = other.basetopo if isinstance( other, ItemTopology ) else other
    subtopos = { name: topo - othertopo for name, topo in self.subtopos.items() }
    return ( self.basetopo - othertopo ).withsubs( subtopos )

  def __rsub__( self, other ):
    return other - self.basetopo

  def __getitem__( self, item ):
    'subtopology'

    if not isinstance( item, str ):
      return self.basetopo.__getitem__( item )
    topo = EmptyTopology( self.ndims )
    for it in item.split( ',' ):
      topo |= self.subtopos[it]
    return topo

  def __setitem__( self, item, topo ):
    assert isinstance( topo, Topology )
    assert topo.ndims == self.ndims, 'wrong dimension: got %d, expected %d' % ( topo.ndims, self.ndims )
    assert all( elem.transform in self.basetopo.edict for elem in topo ), 'group %r is not a subtopology' % item
    self.subtopos[item] = topo.withsubs()

  def __invert__( self ):
    subtopos = { name[1:] if name[0] == '~' else '~'+name: ~topo for name, topo in self.subtopos.items() }
    return ( ~self.basetopo ).withsubs( subtopos )

  def withsubs( self, subtopos={} ):
    if subtopos and core.getprop( 'selfcheck', False ):
      for name, topo in subtopos.items():
        self[name] = topo
    else:
      self.subtopos.update( subtopos )
    return self

  @property
  def edict( self ):
    return self.basetopo.edict

  @property
  def elements( self ):
    return self.basetopo.elements

  @property
  def boundary( self ):
    return self.basetopo.boundary

  @boundary.setter
  def boundary( self, value ):
    self.basetopo.boundary = value.withsubs()

  @property
  def interfaces( self ):
    return self.basetopo.interfaces

  @interfaces.setter
  def interfaces( self, value ):
    self.basetopo.interfaces = value.withsubs()

  @property
  def points( self ):
    return self.basetopo.points

  @points.setter
  def points( self, value ):
    self.basetopo.points = value.withsubs()

  def basis( self, name, *args, **kwargs ):
    return self.basetopo.basis( name, *args, **kwargs )

  @cache.property
  def refined( self ):
    subtopos = { name: topo.refined for name, topo in self.subtopos.items() }
    return self.basetopo.refined.withsubs( subtopos )

  def subset( self, elements, boundaryname=None, precise=False ):
    subtopos = { name: topo.subset( elements, boundaryname, precise=False ) for name, topo in self.subtopos.items() }
    return self.basetopo.subset( elements, boundaryname, precise ).withsubs( subtopos )

  def hierarchical( self, elements, precise=False ):
    subtopos = { name: topo.hierarchical( elements, precise=False ) for name, topo in self.subtopos.items() }
    return self.basetopo.hierarchical( elements, precise ).withsubs( subtopos )

class OppositeTopology( Topology ):
  'opposite topology'

  def __init__( self, basetopo ):
    self.basetopo = basetopo
    Topology.__init__( self, basetopo.ndims )

  def __iter__( self ):
    return ( element.Element( elem.reference, elem.opposite, elem.transform ) for elem in self.basetopo )

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

  @property
  def elements( self ):
    return ()

class StructuredTopology( Topology ):
  'structured topology'

  def __init__( self, root, axes, nrefine=0 ):
    'constructor'

    self.root = root
    self.axes = tuple(axes)
    self.nrefine = nrefine
    self.shape = tuple( axis.j - axis.i for axis in self.axes if axis.isdim )
    Topology.__init__( self, len(self.shape) )

  def __iter__( self ):
    reference = element.getsimplex(1)**self.ndims
    izip = getattr( itertools, 'izip', zip ) # results in iterator zip for both python2 and python3
    return ( element.Element( reference, trans, opp ) for trans, opp in izip( self._transform.flat, self._opposite.flat ) )

  def __len__( self ):
    return numpy.prod( self.shape, dtype=int )

  def __getitem__( self, item ):
    items = (item,) if not isinstance( item, tuple ) else item
    if not all( isinstance(it,slice) for it in items ):
      return Topology.__getitem__( self, item )
    assert len(items) <= self.ndims
    axes = []
    idim = 0
    for axis in self.axes:
      if axis.isdim and idim < len(items):
        s = items[idim]
        start, stop, stride = s.indices( axis.j - axis.i )
        assert stride == 1
        assert stop > start
        if start > 0 or stop < axis.j - axis.i:
          axis = DimAxis( axis.i+start, axis.i+stop, isperiodic=False )
        idim += 1
      axes.append( axis )
    return StructuredTopology( self.root, axes, self.nrefine )

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
    transforms = numeric.asobjvector( transform.affine(0,index) for index in log.iter( 'elem', indices, indices.size ) ).reshape( indices.shape )

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
    return numeric.asobjvector( element.Element( reference, trans, opp ) for trans, opp in numpy.broadcast( self._transform, self._opposite ) ).reshape( self.shape )

  @cache.property
  def boundary( self ):
    'boundary'

    nbounds = len(self.axes) - self.ndims
    union = EmptyTopology( self.ndims-1 )
    subtopos = []
    for idim, axis in enumerate( self.axes ):
      for side, n in enumerate( (axis.i,axis.j) if axis.isdim and not axis.isperiodic else () ):
        topo = StructuredTopology( self.root, self.axes[:idim] + (BndAxis(n,n if not axis.isperiodic else 0,nbounds,side),) + self.axes[idim+1:], self.nrefine )
        subtopos.append( topo )
        union |= topo
    subtopos = dict( zip( ('left','right','bottom','top','front','back'), subtopos ) )
    return union.withsubs( subtopos )

  @cache.property
  def interfaces( self ):
    'interfaces'

    topos = []
    nbounds = len(self.axes) - self.ndims
    for idim, axis in enumerate(self.axes):
      if not axis.isdim:
        continue
      bndprops = [ BndAxis( i, i, ibound=nbounds, side=True ) for i in range( axis.i+1, axis.j ) ]
      if axis.isperiodic:
        assert axis.i == 0
        bndprops.append( BndAxis( axis.j, 0, ibound=nbounds, side=True ) )
      itopo = EmptyTopology( self.ndims-1 ) if not bndprops \
         else UnionTopology( StructuredTopology( self.root, self.axes[:idim] + (axis,) + self.axes[idim+1:], self.nrefine ) for axis in bndprops )
      topos.append( itopo )
    return UnionTopology( topos ).withsubs()

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

    vertex_structure = numpy.array( 0 )
    dofcount = 1
    slices = []

    for idim in range( self.ndims ):
      periodic_i = idim in periodic
      n = self.shape[idim]
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
    for item in numpy.broadcast( self._transform, stdelems, *numpy.ix_(*slices) ):
      trans = item[0]
      std = item[1]
      S = item[2:]
      dofs = vertex_structure[S].ravel()
      mask = dofs >= 0
      if mask.all():
        dofmap[trans] = dofs
        funcmap[trans] = (std,None),
      else:
        assert mask.any()
        dofmap[trans] = dofs[mask]
        funcmap[trans] = (std,mask),

    return function.function( funcmap, dofmap, dofcount, self.ndims )

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
      funcmap[trans] = (std,None),

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
      fmap[elem.transform] = (stdfunc,None),
      nmap[elem.transform] = ndofs + numpy.arange(stdfunc.nshapes)
      ndofs += stdfunc.nshapes
    return function.function( fmap=fmap, nmap=nmap, ndofs=ndofs, ndims=self.ndims )

  def basis_std( self, degree, removedofs=None ):
    'spline from vertices'

    if numeric.isint( degree ):
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
      n = self.shape[idim]
      p = degree[idim]

      nd = n * p + 1
      numbers = numpy.arange( nd )
      if idim in self.periodic and p > 0:
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
    for item in numpy.broadcast( self._transform, *numpy.ix_(*slices) ):
      trans = item[0]
      S = item[1:]
      dofs = vertex_structure[S].ravel()
      mask = dofs >= 0
      if mask.all():
        dofmap[ trans ] = dofs
        funcmap[ trans ] = (stdelem,None),
      elif mask.any():
        dofmap[ trans ] = dofs[mask]
        funcmap[ trans ] = (stdelem,mask),

    return function.function( funcmap, dofmap, dofcount, self.ndims )

  @property
  def refined( self ):
    'refine non-uniformly'

    axes = [ DimAxis(i=axis.i*2,j=axis.j*2,isperiodic=axis.isperiodic) if axis.isdim
        else BndAxis(i=axis.i*2,j=axis.j*2,ibound=axis.ibound,side=axis.side) for axis in self.axes ]
    return StructuredTopology( self.root, axes, self.nrefine+1 )

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
  def edge_search( self ):
    edges = {}
    ifaces = []
    for elem in log.iter( 'elem', self ):
      elemcoords = elem.vertices
      for iedge, iverts in enumerate( elem.reference.edge2vertex ):
        edgekey = tuple( sorted( c for c, n in zip( elemcoords, iverts ) if n ) )
        edge = elem.edge(iedge)
        try:
          oppedge = edges.pop( edgekey )
        except KeyError:
          edges[edgekey] = edge
        else:
          assert edge.reference == oppedge.reference
          ifaces.append(( edge, oppedge ))
    return tuple(edges.values()), tuple(ifaces)

  @cache.property
  def boundary( self ):
    edges, interfaces = self.edge_search
    return UnstructuredTopology( self.ndims-1, edges )

  @cache.property
  def interfaces( self ):
    edges, interfaces = self.edge_search
    return UnstructuredTopology( self.ndims-1, [ element.Element( edge.reference, edge.transform,
      oppedge.transform << transform.solve( oppedge.transform, edge.transform ) )
        for edge, oppedge in interfaces ])

class UnionTopology( Topology ):
  'grouped topology'

  def __init__( self, topos ):
    self._topos = tuple(topos)
    ndims = self._topos[0].ndims
    assert all( topo.ndims == ndims for topo in self._topos )
    Topology.__init__( self, ndims )

  def __or__( self, other ):
    if isinstance( other, UnionTopology ):
      return UnionTopology( self._topos + other._topos )
    return UnionTopology( self._topos + (other,) )

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
        raise NotImplementedError
    return elements

class SubtractionTopology( Topology ):
  'subtraction topology'

  def __init__( self, basetopo, subtopo ):
    assert basetopo.ndims == subtopo.ndims
    self.basetopo = basetopo
    self.subtopo = subtopo
    Topology.__init__( self, basetopo.ndims )

  def __getitem__( self, item ):
    return self.basetopo[item] - self.subtopo

  def __or__( self, other ):
    if other == self.subtopo:
      return self.basetopo
    return Topology.__or__( self, other )

  @cache.property
  def elements( self ):
    elements = []
    edict = self.subtopo.edict
    for elem in self.basetopo:
      try:
        index = edict[elem.transform]
      except KeyError:
        elements.append( elem )
      else:
        ref = elem.reference - self.subtopo.elements[index].reference
        if ref:
          elements.append( element.Element( ref, elem.transform, elem.opposite ) )
    return elements

  @property
  def refined( self ):
    return self.basetopo.refined - self.subtopo.refined

  @cache.property
  @log.title
  def boundary( self ):
    'boundary'

    btopo = EmptyTopology( self.ndims-1 )
    for boundary in ~self.subtopo.boundary, self.basetopo.boundary:
      belems = [] # trimmed boundary elements
      for belem in boundary:
        btrans = belem.transform.promote( self.ndims )
        head = btrans.lookup( self.edict )
        if head:
          ref = self.elements[self.edict[head]].reference
          try:
            index = ref.edge_transforms.index( btrans.slicefrom(len(head)) )
          except ValueError: # e.g. when belem is opposite side of intersected element
            pass
          else:
            edge = ref.edge_refs[index] & belem.reference
            if edge:
              belems.append( element.Element( edge, belem.transform, belem.opposite ) )
      btopo |= boundary.subset( belems, boundaryname=None, precise=True )
    return btopo

  @cache.property
  @log.title
  def interfaces( self ):
    'boundary'

    ielems = []
    subinterfaces = self.subtopo.interfaces
    for ielem in self.basetopo.interfaces:
      ref = ielem.reference
      head = ielem.transform.lookup( subinterfaces.edict )
      if head:
        if head != ielem.transform:
          raise NotImplementedError # wait for unit test to implement this using ref.transform
        subielem = subinterfaces.elements[subinterfaces.edict[head]]
        assert subielem.opposite == ielem.opposite
        ref -= subielem.reference
      else:
        head = ielem.opposite.lookup( subinterfaces.edict )
        if head:
          if head != ielem.opposite:
            raise NotImplementedError # wait for unit test to implement this using ref.transform
          subielem = subinterfaces.elements[subinterfaces.edict[head]]
          assert subielem.transform == ielem.opposite
          ref -= subielem.reference
      if ref:
        ielems.append( element.Element( ref, ielem.transform, ielem.opposite ) )
    return UnstructuredTopology( self.ndims-1, ielems )

  @log.title
  def basis( self, name, *args, **kwargs ):
    basis = self.basetopo.basis( name, *args, **kwargs )
    return self.prune_basis( basis )
    
class SubsetTopology( Topology ):
  'trimmed'

  def __init__( self, basetopo, elements, boundaryname, precise ):
    assert not boundaryname or isinstance( boundaryname, str )
    self.allelements = tuple(elements)
    if precise:
      self.elements = self.allelements
    self.basetopo = basetopo
    self.boundaryname = boundaryname
    Topology.__init__( self, basetopo.ndims )

  def __getitem__( self, item ):
    return self.basetopo[item].subset( self.allelements, self.boundaryname, precise=False )

  @cache.property
  def elements( self ):
    edict = self.basetopo.edict
    elements = []
    for elem in self.allelements:
      index = edict.get( elem.transform )
      if index is not None:
        ref = self.basetopo.elements[index].reference & elem.reference
        if ref:
          elements.append( element.Element( ref, elem.transform, elem.opposite ) )
    return elements

  @property
  def refined( self ):
    elems = [ child for elem in self for child in elem.children if child ]
    return self.basetopo.refined.subset( elems, self.boundaryname, precise=True )

  @cache.property
  @log.title
  def search_interfaces( self ):
    edict = self.edict
    empty = element.EmptyReference( self.ndims-1 )
    ielems = []
    belems = []
    for iface in log.iter( 'elem', self.basetopo.interfaces ):
      btrans = iface.transform.promote( self.ndims )
      head = btrans.lookup( edict )
      if head:
        ref = self.elements[edict[head]].reference
        index = ref.edge_transforms.index( btrans.slicefrom( len(head) ) )
        edge = ref.edge_refs[ index ]
      else:
        edge = empty
      btrans = iface.opposite.promote( self.ndims )
      head = btrans.lookup( edict )
      if head:
        ref = self.elements[edict[head]].reference
        tail = btrans.slicefrom( len(head) )
        tail2head = tail.lookup( ref.edge_transforms ) # strip optional adjustment transformation (temporary?)
        index = ref.edge_transforms.index( tail2head )
        oppedge = ref.edge_refs[ index ].transform( tail2head.slicefrom( len(tail2head) ) )
      else:
        oppedge = empty
      if iface.reference == edge == oppedge: # chortcut
        ielems.append( iface )
      else:
        ifaceref = edge & oppedge
        if iface:
          ielems.append( element.Element( ifaceref, iface.transform, iface.opposite ) )
        edgeref = edge - oppedge
        if edgeref:
          belems.append( element.Element( edgeref, iface.transform, iface.opposite ) )
        edgeref = oppedge - edge
        if edgeref:
          belems.append( element.Element( edgeref, iface.opposite, iface.transform ) )
    return tuple(belems), tuple(ielems)

  @cache.property
  @log.title
  def interfaces( self ):
    belems, ielems = self.search_interfaces
    return self.basetopo.interfaces.subset( ielems, boundaryname=None, precise=True )

  @cache.property
  @log.title
  def boundary( self ):
    'boundary'

    belems, ielems = self.search_interfaces
    trimmed = list( belems )
    for elem in self: # cheap search for intersected elements
      index = self.basetopo.edict[ elem.transform ]
      n = self.basetopo.elements[index].reference.nedges
      trimmed.extend( element.Element( edge, elem.transform<<trans, elem.transform<<trans.flipped ) for trans, edge in elem.reference.edges[n:] )
    trimboundarybase = UnstructuredTopology( self.ndims-1, trimmed )
    trimboundary = trimboundarybase.withsubs( { self.boundaryname: trimboundarybase.withsubs() } if self.boundaryname else {} )

    belems = [] # prior boundary elements (reduced)
    basebtopo = self.basetopo.boundary
    for belem in log.iter( 'element', basebtopo ):
      btrans = belem.transform.promote( self.ndims )
      head = btrans.lookup( self.edict )
      if head:
        ielem = self.edict[head]
        ref = self.elements[ielem].reference
        iedge = ref.edge_transforms.index( btrans.slicefrom(len(head)) )
        edge = ref.edge_refs[iedge]
        if edge:
          belems.append( element.Element( edge, belem.transform, belem.opposite ) )
    origboundary = basebtopo.subset( belems, self.boundaryname, precise=True )

    return trimboundary | origboundary

  @log.title
  def basis( self, name, *args, **kwargs ):
    basis = self.basetopo.basis( name, *args, **kwargs )
    return self.prune_basis( basis )

class RefinedTopology( Topology ):
  'refinement'

  def __init__( self, basetopo ):
    self.basetopo = basetopo
    Topology.__init__( self, basetopo.ndims )

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
        elements.append( element.Element( ref, elem.transform, elem.opposite ) )
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

  def __getitem__( self, item ):
    return self.basetopo[item].hierarchical( self.allelements, precise=False )

  def hierarchical( self, elements, precise=False ):
    return self.basetopo.hierarchical( elements, precise )

  @cache.property
  def elements( self ):
    itemelems = []
    for elem in self.allelements:
      head = elem.transform.lookup(self.basetopo.edict)
      if not head:
        continue
      itemelem = self.basetopo.elements[self.basetopo.edict[head]]
      ref = itemelem.reference
      tail = elem.transform.slicefrom( len(head) )
      while tail:
        index = ref.child_transforms.index( tail.sliceto(1) )
        ref = ref.child_refs[ index ]
        if not ref:
          break
        tail = tail.slicefrom(1)
      else:
        ref &= elem.reference
        if ref:
          itemelems.append( element.Element( ref, elem.transform, elem.opposite ) )
    return itemelems

  @cache.property
  @log.title
  def levels( self ):
    levels = [ self.basetopo ]
    for elem in self:
      trans = elem.transform.lookup( self.basetopo.edict )
      assert trans, 'element is not a refinement of basetopo'
      nrefine = len(elem.transform) - len(trans)
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
    belems = []
    for belem in log.iter( 'elem', basebtopo ):
      transform = belem.transform.promote( self.ndims )
      if transform.lookup( self.edict ):
        # basetopo boundary element is as fine or finer than element found in
        # self; this may happen for example when basetopo is a trimmed topology
        belems.append( belem )
      else:
        trans, updim = transform.split( self.ndims-1, after=False )
        elems = [ elem for elem in self if elem.transform.sliceto(len(trans)) == trans ]
        belems.extend( element.Element( edge.reference, edge.transform, belem.opposite << edge.transform.slicefrom(len(belem.transform)) )
          for elem in elems for edge in elem.edges if edge.transform.sliceto(len(belem.transform)) == belem.transform )
    return basebtopo.hierarchical( belems, precise=True )

  @cache.property
  def interfaces( self ):
    'interface elements'

    ielems = [ ielem for topo in log.iter( 'level', self.levels ) for ielem in topo.interfaces
      if ielem.transform.promote( self.ndims ).sliceto(-1) in self.edict and ielem.opposite.promote( self.ndims ).lookup( self.edict )
      or ielem.opposite.promote( self.ndims ).sliceto(-1) in self.edict and ielem.transform.promote( self.ndims ).lookup( self.edict ) ]
    baseitopo = self.basetopo.interfaces
    return baseitopo.hierarchical( ielems, precise=True )

  @log.title
  def basis( self, name, *args, **kwargs ):
    'build hierarchical function space'

    # The law: a basis function is retained if all elements of self can
    # evaluate it through cascade, and at least one element of self can
    # evaluate it directly.

    # Procedure: per refinement level, track which basis functions have at
    # least one supporting element coinsiding with self ('touched') and no
    # supporting element finer than self ('supported').

    collect = {}
    ndofs = 0 # total number of dofs of new function object
    remaining = len(self) # element count down (know when to stop)

    for topo in log.iter( 'level', self.levels ):

      funcsp = topo.basis( name, *args, **kwargs ) # shape functions for current level
      supported = numpy.ones( funcsp.shape[0], dtype=bool ) # True if dof is fully contained in self or parents
      touchtopo = numpy.zeros( funcsp.shape[0], dtype=bool ) # True if dof touches at least one elem in self
      myelems = [] # all top-level or parent elements in current level

      (axes,func), = function.blocks( funcsp )
      dofmap = axes[0].dofmap
      stdmap = func.stdmap
      for elem in topo:
        trans = elem.transform
        idofs = dofmap[trans]
        stds = stdmap[trans]
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

  def subset( self, elements, boundaryname=None, precise=False ):
    return self.basetopo.hierarchical( elements )

class RevolvedTopology( Topology ):
  'revolved'

  def __init__( self, basetopo ):
    self.basetopo = basetopo
    Topology.__init__( self, basetopo.ndims )

  def __iter__( self ):
    return iter( self.basetopo )

  def __len__( self ):
    return len( self.basetopo )

  def __getitem__( self, item ):
    return RevolvedTopology( self.basetopo[item] )

  @property
  def elements( self ):
    return self.basetopo.elements

  @cache.property
  def boundary( self ):
    return RevolvedTopology( self.basetopo.boundary )

  @log.title
  @core.single_or_multiple
  def integrate( self, funcs, ischeme, geometry, force_dense=False, edit=_identity ):
    iwscale = function.jacobian( geometry, self.ndims+1 ) * function.Iwscale(self.ndims)
    integrands = [ function.asarray( edit( func * iwscale ) ) for func in funcs ]
    data_index = self._integrate( integrands, ischeme )
    return [ matrix.assemble( data, index, integrand.shape, force_dense ) for integrand, (data,index) in zip( integrands, data_index ) ]

  def basis( self, name, *args, **kwargs ):
    return function.revolved( self.basetopo.basis( name, *args, **kwargs ) )

  def elem_eval( self, *args, **kwargs ):
    return self.basetopo.elem_eval( *args, **kwargs )

  def refined_by( self, refine ):
    return RevolvedTopology( self.basetopo.refined_by(refine) )


# UTILITY FUNCTIONS

DimAxis = collections.namedtuple( 'DimAxis', ['i','j','isperiodic'] )
DimAxis.isdim = True
BndAxis = collections.namedtuple( 'BndAxis', ['i','j','ibound','side'] )
BndAxis.isdim = False

def common_refine( topo1, topo2 ):
  assert topo1.ndims == topo2.ndims
  elements = []
  topo2trans = { elem.transform: elem for elem in topo2 }
  for elem1 in topo1:
    head = elem1.transform.lookup( topo2trans )
    if head:
      elements.append( elem1 )
      topo2trans[ head ] = None
  elements.extend( elem for elem in topo2trans.values() if elem is not None )
  return UnstructuredTopology( topo1.ndims, elements )

# vim:shiftwidth=2:softtabstop=2:expandtab:foldmethod=indent:foldnestmax=2
