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

  # subclass needs to implement:
  # __iter__
  # __len__
  # getelem

  def __init__( self, ndims, groups={} ):
    'constructor'

    assert isinstance( ndims, int ) and ndims >= 0
    self.ndims = ndims
    self.groups = groups.copy()

  @property
  def elements( self ):
    warnings.warn( 'topology.elements will be removed in future; please use tuple(topology) instead', DeprecationWarning )
    return tuple( self )

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
    return Topology( edges )

  @cache.property
  def interfaces( self ):
    edges, interfaces = self.edge_search
    return UnstructuredTopology( self.ndims-1, [ element.Element( edge.reference, edge.transform,
      oppedge.transform << transform.solve( oppedge.transform, edge.transform ) )
        for edge, oppedge in interfaces ])

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
  def groupnames( self ):
    return self.groups.keys()

  def __contains__( self, element ):
    ielem = self.edict.get(element.transform)
    return ielem is not None and self.getelem(ielem) == element

  def __add__( self, other ):
    'add topologies'

    assert self.ndims == other.ndims
    return UnstructuredTopology( self.ndims, set(self) | set(other) )

  def _sub( self, other ):
    assert isinstance( other, Topology )
    assert self.ndims == other.ndims
    refmap = { trans: other.getelem(index).reference for trans, index in other.edict.items() }
    refs = [ elem.reference - refmap.pop(elem.transform,elem.reference.empty) for elem in self ]
    assert not refmap, 'subtracted topology is not a subtopology'
    return TrimmedTopology( self, refs )

  __sub__ = lambda self, other: self._sub( other ) if isinstance( other, Topology ) and not isinstance( other, TrimmedTopology ) else NotImplemented
  __rsub__ = lambda self, other: other._sub( self ) if isinstance( other, Topology ) else NotImplemented

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

  def __getitem__( self, item ):
    'subtopology'

    if not isinstance( item, str ):
      raise KeyError( str(item) )
    return util.sum( self.groups[it] for it in item.split( ',' ) )

  def __setitem__( self, item, topo ):
    assert isinstance( topo, Topology ), 'wrong type: got %s, expected Topology' % type(topo)
    assert topo.ndims == self.ndims, 'wrong dimension: got %d, expected %d' % ( topo.ndims, self.ndims )
    assert all( elem.transform in self.edict for elem in topo ), 'group %r is not a subtopology' % item
    self.groups[item] = topo

  @cache.property
  def edict( self ):
    '''transform -> ielement mapping'''
    return { elem.transform: ielem for ielem, elem in enumerate(self) }

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
    return HierarchicalTopology( self, refined )

  @log.title
  @core.single_or_multiple
  def elem_eval( self, funcs, ischeme, separate=False, geometry=None, edit=_identity ):
    'element-wise evaluation'

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

    if separate:
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
    refs = [ elem.trim( levelset=levelset, maxrefine=maxrefine, ndivisions=ndivisions, fcache=fcache ) for elem in log.iter( 'elem', self ) ]
    return TrimmedTopology( self, refs, name )
    log.debug( 'cache', fcache.stats )

#   for key in log.iter( 'remaining', extras ):
#     ielems = functools.reduce( numpy.intersect1d, [ self.v2elem[vert] for vert in key ] )
#     if len(ielems) == 1: # vertices lie on boundary
#       continue # if the interface coincides with the boundary, the boundary wins
#     assert len(ielems) == 2
#     value = ()
#     for ielem in ielems:
#       posneg = elems[ielem]
#       elem = self.elements[ielem]
#       mask = numpy.array([ vtx in key for vtx in elem.vertices ])
#       (iedge,), = numpy.where(( elem.reference.edge2vertex == mask ).all( axis=1 ))
#       trans = elem.transform << elem.reference.edge_transforms[iedge]
#       value += posneg, iedge, trans
#     pairs.append( value )

#   trims = []
#   for (pos,neg), iedge, trans, (_pos,_neg), _iedge, _trans in pairs:

#     posref = pos and pos.edge_refs[iedge]
#     negref = neg and neg.edge_refs[iedge]
#     _posref = _pos and _pos.edge_refs[_iedge]
#     _negref = _neg and _neg.edge_refs[_iedge]

#     ref = (posref or _posref) and posref^_posref
#     _ref = (negref or _negref) and negref^_negref
#     if transform.equivalent( trans, _trans, flipped=True ):
#       assert ref == _ref
#     else:
#       # edges are rotated; in this case we have no way of asserting
#       # identity, we can only make the transformation matching and hope
#       # we didn't make any mistakes
#       _trans <<= transform.solve(_trans,trans)
#     if ref:
#       trims.append( element.Element( ref, trans, _trans ) if ref & posref
#                else element.Element( _ref, _trans, trans ) )

#   return TrimmedTopology( self, [ pos for pos, neg in elems ], name, trims, ndims=self.ndims ), \
#          TrimmedTopology( self, [ neg for pos, neg in elems ], name, [ trim.flipped for trim in trims ], ndims=self.ndims )

  @cache.property
  @log.title
  def v2elem( self ):
    v2elem = {}
    for ielem, elem in log.enumerate( 'elem', self ):
      for vert in elem.vertices:
        v2elem.setdefault( vert, [] ).append( ielem )
    return v2elem

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

class EmptyTopology( Topology ):
  'empty topology'

  def __iter__( self ):
    return iter([])

  def __len__( self ):
    return 0

  def getelem( self, index ):
    raise IndexError( 'out of bounds' )

class UnstructuredTopology( Topology ):
  'unstructured topology'

  def __init__( self, ndims, elements, groups={} ):
    self._elements = tuple(elements)
    assert all( elem.ndims == ndims for elem in self._elements )
    Topology.__init__( self, ndims, groups )

  def __iter__( self ):
    return iter( self._elements )

  def __len__( self ):
    return len( self._elements )

  def getelem( self, index ):
    assert isinstance( index, int )
    return self._elements[index]

class StructuredTopology( Topology ):
  'structured topology'

  def __init__( self, root, axes, nrefine=0, groups={} ):
    'constructor'

    self.root = root
    self.axes = tuple(axes)
    self.nrefine = nrefine
    self._shape = tuple( axis.j - axis.i for axis in self.axes if axis.isdim )
    Topology.__init__( self, len(self._shape), groups=groups )

  def __iter__( self ):
    reference = element.getsimplex(1)**self.ndims
    return ( element.Element( reference, trans, opp ) for trans, opp in itertools.izip( self._transform.flat, self._opposite.flat ) )

  def __len__( self ):
    return numpy.prod( self._shape )

  def getelem( self, index ):
    assert isinstance( index, int )
    reference = element.getsimplex(1)**self.ndims
    return element.Element( reference, self._transform.flat[index], self._opposite.flat[index] )

  def __getitem__( self, item ):
    'subtopology'

    if isinstance( item, str ):
      return Topology.__getitem__( self, item )
    if not isinstance( item, tuple ):
      item = item,
    assert len(item) <= self.ndims
    axes = []
    idim = 0
    for axis in self.axes:
      if axis.isdim and idim < len(item):
        s = item[idim]
        assert isinstance( s, slice )
        start, stop, stride = s.indices( axis.j - axis.i )
        assert stride == 1
        assert stop > start
        if start > 0 or stop < axis.j - axis.i:
          axis = DimAxis( axis.i+start, axis.i+stop, isperiodic=False )
        idim += 1
      axes.append( axis )
    return StructuredTopology( self.root, axes, self.nrefine )

  @property
  def periodic( self ):
    return tuple( idim for idim, axis in enumerate(self.axes) if axis.isdim and axis.isperiodic )

  @staticmethod
  def mktransforms( axes, root, nrefine ):
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

    @numeric.broadcasted
    def mktrans( *index ):
      index = numpy.array( index )
      trans = transform.identity
      for irefine in range( nrefine ):
        index, offset = divmod( index, 2 )
        trans >>= transform.affine( .5, .5*offset )
      trans >>= transform.affine( 0, index )
      return root << trans << updim

    indices = numpy.ix_( *[ numpy.arange(axis.i,axis.j) if axis.isdim else [axis.i-1 if axis.side else axis.j] for axis in axes ] )
    return mktrans( *indices )

  @cache.property
  def _transform( self ):
    return self.mktransforms( self.axes, self.root, self.nrefine )

  @cache.property
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
    @numeric.broadcasted
    def mkelem( trans, opp ):
      return element.Element( reference, trans, opp )
    return mkelem( self._transform, self._opposite )

  @cache.property
  def boundary( self ):
    'boundary'

    nbounds = len(self.axes) - self.ndims
    names = ('left','right'), ('bottom','top'), ('front','back')
    groups = { names[idim][side]: StructuredTopology( self.root, self.axes[:idim] + (BndAxis(n,n if not axis.isperiodic else 0,nbounds,side),) + self.axes[idim+1:], self.nrefine )
      for idim, axis in enumerate(self.axes)
        for side, n in enumerate( (axis.i,axis.j) if axis.isdim and not axis.isperiodic else () ) }
    return GroupedTopology( named=groups )

  @cache.property
  def interfaces( self ):
    'interfaces'

    groups = {}
    nbounds = len(self.axes) - self.ndims
    for idim, axis in enumerate(self.axes):
      if not axis.isdim:
        continue
      bndprops = [ BndAxis( i, i, ibound=nbounds, side=True ) for i in range( axis.i+1, axis.j ) ]
      if axis.isperiodic:
        assert axis.i == 0
        bndprops.append( BndAxis( axis.j, 0, ibound=nbounds, side=True ) )
      itopo = EmptyTopology( self.ndims-1 ) if not bndprops \
         else GroupedTopology( unnamed=[ StructuredTopology( self.root, self.axes[:idim] + (axis,) + self.axes[idim+1:] ) for axis in bndprops ] )
      groups[ 'dir{}'.format(len(groups)) ] = itopo
    return GroupedTopology( named=groups )

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
      n = self._shape[idim]
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
      n = self._shape[idim]
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
    log.debug( 'Local knot vector cache effectivity: %d' % (100*(1.-len(cache)/float(sum(self._shape)))) )

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
      n = self._shape[idim]
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

  @cache.property
  def refined( self ):
    'refine non-uniformly'

    axes = [ DimAxis(i=axis.i*2,j=axis.j*2,isperiodic=axis.isperiodic) if axis.isdim
        else BndAxis(i=axis.i*2,j=axis.j*2,ibound=axis.ibound,side=axis.side) for axis in self.axes ]
    groups = { name: topo.refined for name, topo in self.groups.items() }
    return StructuredTopology( self.root, axes, self.nrefine+1, groups )

  def __str__( self ):
    'string representation'

    return '%s(%s)' % ( self.__class__.__name__, 'x'.join( str(n) for n in self._shape ) )

class GroupedTopology( Topology ):
  'grouped topology'

  def __init__( self, unnamed=[], named={} ):
    assert unnamed or named
    self._topos = tuple(unnamed) + tuple(named.itervalues())
    ndims = self._topos[0].ndims
    assert all( topo.ndims == ndims for topo in self._topos )
    Topology.__init__( self, ndims, named )

  def __iter__( self ):
    return ( elem for topo in self._topos for elem in topo )

  def __len__( self ):
    return sum( len(topo) for topo in self._topos )

  def getelem( self, index ):
    for topo in self._topos:
      nelems = len(topo)
      if index < nelems:
        return topo.getelem(index)
      index -= nelems
    raise Exception, 'index out of bounds'

class HierarchicalTopology( UnstructuredTopology ):
  'collection of nested topology elments'

  def __init__( self, basetopo, elements, groups={} ):
    'constructor'

    self.basetopo = basetopo if not isinstance( basetopo, HierarchicalTopology ) else basetopo.basetopo
    UnstructuredTopology.__init__( self, basetopo.ndims, elements, groups={} )

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
    return HierarchicalTopology( self.basetopo, elements )

  def __getitem__( self, item ):
    itemtopo = self.basetopo[item]
    elems = [ elem for elem in self if elem.transform.lookup(itemtopo.edict) ]
    return HierarchicalTopology( itemtopo, elems )

  @cache.property
  def boundary( self ):
    'boundary elements'

    belems = [ belem for topo in log.iter( 'level', self.levels ) for belem in topo.boundary if belem.transform.promote( self.ndims ).sliceto(-1) in self.edict ]
    return HierarchicalTopology( self.basetopo.boundary, belems )

  @cache.property
  def interfaces( self ):
    'interface elements & groups'

    ielems = [ ielem for topo in log.iter( 'level', self.levels ) for ielem in topo.interfaces
      if ielem.transform.promote( self.ndims ).sliceto(-1) in self.edict and ielem.opposite.promote( self.ndims ).lookup( self.edict )
      or ielem.opposite.promote( self.ndims ).sliceto(-1) in self.edict and ielem.transform.promote( self.ndims ).lookup( self.edict ) ]
    return HierarchicalTopology( self.basetopo.interfaces, ielems )

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

  def trim( self, *args, **kwargs ):
    elems = Topology.trim( self, *args, **kwargs )
    return HierarchicalTopology( self.basetopo, elems )

class RefinedTopology( Topology ):
  'refinement'

  def __init__( self, basetopo, groups={} ):
    self.basetopo = basetopo
    Topology.__init__( self, basetopo.ndims, groups=groups )

  @cache.property
  def _elements( self ):
    return tuple([ child for elem in self.basetopo for child in elem.children ])

  def __iter__( self ):
    return iter( self._elements )

  def __len__( self ):
    return len( self._elements )

  def getelem( self, index ):
    assert isinstance( index, int )
    return self._elements[ index ]

  def __getitem__( self, item ):
    if isinstance( item, int ):
      return self._elements[ item ]
    return self.basetopo[item].refined

  @cache.property
  def boundary( self ):
    return self.basetopo.boundary.refined

class TrimmedTopology( Topology ):
  'trimmed'

  def __init__( self, basetopo, refs, trimname='trimmed', groups={} ):
    assert len(refs) == len(basetopo)
    assert all( isinstance(ref,element.Reference) for ref in refs )
    self.__refs = refs
    self._indices = numpy.array( [ index for index, ref in enumerate(self.__refs) if ref ], dtype=int )
    self.basetopo = basetopo
    self.trimname = trimname
    Topology.__init__( self, basetopo.ndims, groups=groups )

  def __iter__( self ):
    return ( element.Element( ref, elem.transform, elem.opposite ) for elem, ref in zip( self.basetopo, self.__refs ) if ref )

  def __len__( self ):
    return len( self._indices )

  def getelem( self, index ):
    assert isinstance( index, int )
    origindex = self._indices[ index ]
    origelem = self.basetopo.getelem( origindex )
    return element.Element( self.__refs[origindex], origelem.transform, origelem.opposite )

  @property
  def _inverse( self ):
    refs = [ elem.reference - ref for elem, ref in zip( self.basetopo, self.__refs ) ]
    return TrimmedTopology( self.basetopo, refs, self.trimname )

  __sub__ = lambda self, other: other._inverse if isinstance( other, TrimmedTopology ) and other.basetopo == self else Topology.__sub__( self, other )
  __rsub__ = lambda self, other: self._inverse if other == self.basetopo else Topology.__rsub__( self, other )

  @cache.property
  def refined( self ):
    elems = [ child for elem in self for child in elem.children ]
    edict = { elem.transform: elem.reference for elem in elems }
    basetopo = self.basetopo.refined
    refs = [ edict.pop(elem.transform,elem.reference.empty) for elem in basetopo ]
    assert not edict, 'leftover elements'
    groups = { name: topo.refined for name, topo in self.groups.items() }
    return TrimmedTopology( basetopo, refs, self.trimname, groups=groups )

  @cache.property
  @log.title
  def trimmed_edges( self ):
    refs = []
    for iface in log.iter( 'elem', self.basetopo.interfaces ):
      btrans1 = iface.transform.promote( self.ndims )
      head1 = btrans1.lookup( self.basetopo.edict )
      tail1 = btrans1.slicefrom( len(head1) )
      btrans2 = iface.opposite.promote( self.ndims )
      head2 = btrans2.lookup( self.basetopo.edict )
      tail2 = btrans2.slicefrom( len(head2) )
      r1 = self.__refs[ self.basetopo.edict[head1] ]
      if r1:
        e1 = r1.edge_refs[ r1.edge_transforms.index(tail1) ]
      else:
        e1 = iface.reference.empty
      r2 = self.__refs[ self.basetopo.edict[head2] ]
      if r2:
        tail2head = tail2.lookup( r2.edge_transforms ) # strip optional opposite adjustment transformation (temporary?)
        tail2tail = tail2.slicefrom( len(tail2head) )
        e2 = r2.edge_refs[ r2.edge_transforms.index(tail2head) ].transform( tail2tail )
      else:
        e2 = iface.reference.empty
      refs.append(( iface, e1, e2 ))
    return refs

  @cache.property
  def interfaces( self ):
    return TrimmedTopology( self.basetopo.interfaces, [ ref1 & ref2 for iface, ref1, ref2 in self.trimmed_edges ] )

  @cache.property
  def boundary( self ):
    trimmed = []
    for iface, ref1, ref2 in self.trimmed_edges:
      edge = ref1 - ref2
      oppedge = ref2 - ref1
      if edge:
        trimmed.append( element.Element( edge, iface.transform, iface.opposite ) )
      if oppedge:
        trimmed.append( element.Element( oppedge, iface.opposite, iface.transform ) )
    for elem, ref in zip( self.basetopo, self.__refs ):
      if ref:
        n = elem.reference.nedges
        trimmed.extend( element.Element( edge, elem.transform<<trans, elem.transform<<trans.flipped ) for trans, edge in ref.edges[n:] )

    belems = []
    basebtopo = self.basetopo.boundary
    for belem in log.iter( 'element', basebtopo ):
      btrans = belem.transform.promote( self.ndims )
      head = btrans.lookup( self.basetopo.edict )
      ielem = self.basetopo.edict[head]
      ref = self.__refs[ielem]
      if ref:
        tail = btrans.slicefrom(len(head))
        iedge = ref.edge_transforms.index(tail)
        edge = ref.edge_refs[iedge]
        if edge:
          belems.append( element.Element( edge, belem.transform, belem.opposite ) )

    boundary = UnstructuredTopology( self.ndims-1, trimmed + belems )
    for name, basebgroup in basebtopo.groups.items():
      refs = []
      for basebelem in basebgroup:
        ibelem = boundary.edict.get(basebelem.transform)
        refs.append( boundary.getelem(ibelem).reference if ibelem is not None else basebelem.reference.empty )
      if any( refs ):
        boundary[name] = TrimmedTopology( basebgroup, refs )

    if trimmed:
      oldtrimtopo = boundary.groups.get( self.trimname )
      newtrimtopo = UnstructuredTopology( self.ndims-1, trimmed )
      boundary[self.trimname] = oldtrimtopo + newtrimtopo if oldtrimtopo else newtrimtopo

    return boundary

  def __getitem__( self, key ):
    if isinstance(key,str) and key in self.groups:
      return self.groups[key]
    keytopo = self.basetopo[key]
    refs = [ self.__refs[ self.basetopo.edict[elem.transform] ] for elem in keytopo ]
    topo = TrimmedTopology( keytopo, refs, self.trimname )
    if isinstance(key,str):
      self.groups[key] = topo
    return topo

  def prune_basis( self, basis ):
    used = numpy.zeros( len(basis), dtype=bool )
    for axes, func in function.blocks( basis ):
      dofmap = axes[0]
      for elem in self:
        used[ dofmap.dofmap[elem.transform] + dofmap.offset ] = True
    return basis[used]

  @log.title
  def basis( self, name, *args, **kwargs ):
    basis = self.basetopo.basis( name, *args, **kwargs )
    return self.prune_basis( basis )

class RevolvedTopology( Topology ):
  'revolved'

  def __init__( self, basetopo ):
    self.basetopo = basetopo
    Topology.__init__( basetopo.ndims )

  def __iter__( self ):
    return iter( self.basetopo )

  def __len__( self ):
    return len( self.basetopo )

  def getelem( self, index ):
    return self.basetopo.getelem( index )

  @cache.property
  def boundary( self ):
    return RevolvedTopology(self.basetopo.boundary)

  def __getitem__( self, item ):
    return RevolvedTopology(self.basetopo[item])

  def __setitem__( self, item, topo ):
    assert isinstance( topo, RevolvedTopology )
    self.basetopo.__setitem__( item, topo.basetopo )

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
  isrevolved = isinstance( topo1, RevolvedTopology )
  assert isinstance( topo2, RevolvedTopology ) == isrevolved
  if isrevolved:
    topo1 = topo1.basetopo
    topo2 = topo2.basetopo
  commonelem = []
  topo2trans = { elem.transform: elem for elem in topo2 }
  for elem1 in topo1:
    head = elem1.transform.lookup( topo2trans )
    if head:
      commonelem.append( elem1 )
      topo2trans[ head ] = None
  commonelem.extend( elem for elem in topo2trans.values() if elem is not None )
  basetopo = topo1.basetopo if isinstance( topo1, HierarchicalTopology ) else topo1
  commontopo = HierarchicalTopology( basetopo, commonelem )
  if isrevolved:
    commontopo = RevolvedTopology( commontopo )
  return commontopo

# vim:shiftwidth=2:softtabstop=2:expandtab:foldmethod=indent:foldnestmax=2
