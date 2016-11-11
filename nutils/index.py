# Module INDEX
#
# Part of Nutils: open source numerical utilities for Python. Jointly developed
# by HvZ Computational Engineering, TU/e Multiscale Engineering Fluid Dynamics,
# and others. More info at http://nutils.org <info@nutils.org>. (c) 2015

from __future__ import print_function, division
from . import function, numpy
import sys, collections, operator, numbers, itertools


class IndexedArray:
  '''wrapper of `Array` with index notation and Einstein summation convention

  This wrapper adds indices to the axes of the wrapped `Array` which is used
  to align the `Array` with other wrapped `Array`s when doing simple
  arithmetic operations.  In addition, repeated indices are summed (Einstein
  summation convention).

  Passing an index string to `a[...]`, with `a` an `Array`, wraps the array
  func.  The number of indices must match the number of axes of `a`.  The index
  string may contain only lower case latin characters.  A wrapped array can be
  unwrapped via the `unwrap` method.

  The index string may contain a `,` or `;` followed by any strict positive
  number of indices.  For all indices following the comma or semicolon
  respectively a gradient or surface gradient is computed with respect to the
  geometry specified in the `unwrap` method.

  Examples.  Let `a` and `b` be `Array`s with shape `(n,n)` and `(n,)`.  The
  following pairs are equivalent when unwrapped with geometry `geom`:

      a['ij']*b['j']
      (a*b[_,:]).sum(1)

      a['ij']+a['ji']
      a+transpose(a)

      a['ij']+b['i']*b['j']
      a+b[:,_]*b[_,:]

      a['ij,k']
      a.grad(geom)

      a['ij,j']
      trace( a.grad(geom), 1, 2 )

      (b['i']*b['j'])[,j']
      trace( (b[:,_]*b[_,:]).grad(geom), 1, 2 )

      b['i;j']
      b.grad(geom, -1)

      a['i0,1']
      a[:,0].grad(geom)[:,1]
  '''

  def __init__( self, shape, linked_lengths, op, args ):
    'constructor'

    self._shape = collections.OrderedDict( shape )
    self._op = op
    self._args = tuple( args )

    # join overlapping sets in `linked_lengths`
    linked_lengths = set( linked_lengths )
    cache = { k: g for g in linked_lengths for k in g }
    for g in linked_lengths:
      linked = set( cache[k] for k in g )
      if len( linked ) == 1:
        continue
      g = frozenset( itertools.chain( *linked ) )
      cache.update( (k, g) for k in g )
    # verify linked lengths
    for g in linked_lengths:
      if len( set(k for k in g if isinstance(g, numbers.Integral)) ) > 1:
        raise ValueError( 'axes have different lengths' )
    # update shape with numbers if possible
    for k, v in self._shape.items():
      if not isinstance( v, numbers.Integral ):
        for i in cache.get( v, [] ):
          if isinstance( i, numbers.Integral ):
            self._shape[k] = i
    self._linked_lengths = frozenset( cache.values() )

    self.indices = ''.join( self._shape )

    assert all( 'a' <= index <= 'z' for index in self.indices ), 'invalid index'
    assert all( isinstance( arg, IndexedArray ) for arg in self._args ), 'incompatible argument'

  @property
  def ndim( self ):
    return len( self._shape )

  @property
  def shape( self ):
    return self._shape.keys()

  def unwrap( self, geometry=None, indices=None ):
    '''unwrap the `Array` aligned according to `indices`

    Parameters
    ----------
    geometry : Array, optional
        The geometry used when computing gradients.  This argument is mandatory
        when this `IndexedArray` contains gradients, otherwise this argument is
        ignored.
    indices : str, optional
        This indicates the order of the axes of the unwrapped `Array`.
        `indices` must contain all indices of this `IndexedArray`, may not have
        repeated indices and may not have indices other than those of this
        `IndexedArray`.

    Returns
    -------
    Array
    '''

    if indices is None:
      indices = tuple( sorted( self.indices ) )
    else:
      if not isinstance( indices, str ):
        raise ValueError( 'expected a `str`, got {!r}'.format( indices ) )
      if len( set( indices ) ) != len( indices ):
        raise ValueError( 'indices may not be repeated when unwrapping' )
      if set( indices ) != set( self.indices ):
        raise ValueError( 'invalid indices: expected {!r} (any order), got {!r}'.format( self.indices, indices ) )

    delayed_lengths = {}
    for g in self._linked_lengths:
      if geometry is not None:
        g = frozenset( len(geometry) if i == 'geom' else i for i in g )
      g_ints = set( i for i in g if isinstance(i, numbers.Integral) )
      if len( g_ints ) > 1:
        raise ValueError( 'axes have different lengths' )
      if len( g_ints ) == 0:
        continue
      i = next( iter( g_ints ) )
      for j in g:
        if isinstance( j, numbers.Integral ):
          continue
        delayed_lengths[j] = i

    return function.align(
      self._unwrap_tree( geometry, delayed_lengths ),
      tuple( map( indices.index, self.indices ) ),
      len( self.indices ) )

  def _unwrap_tree( self, geom, delayed_lengths ):
    return self._op( geom, delayed_lengths, *( arg._unwrap_tree( geom, delayed_lengths ) for arg in self._args ) )

  @staticmethod
  def _array_grad( geom, delayed_lengths, array ):
    if geom is None:
      raise ValueError( '`geom` is required for unwrapping this `IndexedArray`' )
    return array.grad( geom )

  @staticmethod
  def _array_surfgrad( geom, delayed_lengths, array ):
    if geom is None:
      raise ValueError( '`geom` is required for unwrapping this `IndexedArray`' )
    return array.grad( geom, -1 )

  def __getitem__( self, item ):
    if not isinstance( item, str ):
      raise ValueError( 'expected a `str`, got {!r}'.format( item ) )
    if item.startswith( ',' ):
      grad = self._array_grad
    elif item.startswith( ';' ):
      grad = self._array_surfgrad
    else:
      # reindex, similar to `self.unwrap()[item]`, but without requiring a geometry
      return wrap( self, item )
    if not all( 'a' <= index <= 'z' or '0' <= index <= '9' for index in item[1:] ):
      raise ValueError( 'invalid index, only lower case latin characters and numbers are allowed' )
    if not all( 1 <= c <= 2 for index, c in collections.Counter( self.indices + item[1:] ).items() if not ('0' <= index <= '9') ):
      raise ValueError( 'indices may not be repeated more than once' )
    shape = collections.OrderedDict( self._shape )
    linked_lengths = set( self._linked_lengths )
    for index in item[1:]:
      if '0' <= index <= '9':
        # `index` is a number
        # get element `index` of the last axis
        op = lambda geom, delayed_lengths, array: grad( geom, delayed_lengths, array )[..., int(index)]
      elif index in shape:
        # `index` is a repeated index
        # find positions of `index`
        indices = tuple( shape )
        ax1 = indices.index( index )
        ax2 = len( shape )
        # link lengths of `ax1` and `ax2`
        linked_lengths.add( frozenset([ shape[index], 'geom' ]) )
        # drop `index` from shape
        del shape[index]
        op = lambda geom, delayed_lengths, array: function.trace( grad( geom, delayed_lengths, array ), ax1, ax2 )
      else:
        # `index` is 'new', append to shape
        shape[index] = 'geom'
        op = grad
      self = IndexedArray( shape, linked_lengths, op, ( self, ) )
    return self

  def _get_element( self, axis, index ):
    '''get element `index` from axis `axis`'''

    i = self.indices.index(axis)
    return IndexedArray(
      ( (k, v) for k, v in self._shape.items() if k != axis ),
      self._linked_lengths,
      lambda geom, delayed_lengths, array: array[(slice(None),)*i + (index,)],
      [self])

  def _trace( self, axis1, axis2 ):
    '''get the trace along `axis1` and `axis2`'''

    i1 = self.indices.index(axis1)
    i2 = self.indices.index(axis2)
    return IndexedArray(
      ( (k, v) for k, v in self._shape.items() if k not in (axis1, axis2) ),
      self._linked_lengths | frozenset([ frozenset([ self._shape[axis1], self._shape[axis2] ]) ]),
      lambda geom, delayed_lengths, array: function.trace( array, i1, i2 ),
      [self])

  def __neg__( self ):
    return IndexedArray( self._shape, self._linked_lengths, lambda *args: -self._op( *args ), self._args )

  @classmethod
  def _apply_add_sub( cls, op, left, right ):
    # convert all args to `IndexedArray`s, if possible
    try:
      left = asindexedarray( left )
      right = asindexedarray( right )
    except ValueError:
      return NotImplemented
    # check indices of left and right operands
    if set( left.indices ) != set( right.indices ):
      raise ValueError( 'left and right operand have different indices' )
    # apply `op`
    align_right = tuple( map( left.indices.index, right.indices ) )
    return IndexedArray(
      left._shape,
      left._linked_lengths | right._linked_lengths | frozenset([ frozenset([ left._shape[k], right._shape[k] ]) for k in left.indices ]),
      lambda geom, delayed_lengths, left_array, right_array:
        op( left_array, function.align( right_array, align_right, left.ndim ) ),
      ( left, right ) )

  def __add__( self, other ):
    return self._apply_add_sub( operator.add, self, other )

  def __radd__( self, other ):
    return self._apply_add_sub( operator.add, other, self )

  def __sub__( self, other ):
    return self._apply_add_sub( operator.sub, self, other )

  def __rsub__( self, other ):
    return self._apply_add_sub( operator.sub, other, self )

  def __mul__( self, other ):
    try:
      other = asindexedarray( other )
    except ValueError:
      return NotImplemented
    common_indices = set( self.indices ) & set( other.indices )
    # collect all indices, common indices at the end, ordered on first appearance
    indices = [ i for i in itertools.chain( self.indices, other.indices ) if i not in common_indices ]
    n_remaining = len( indices )
    for i in itertools.chain( self.indices, other.indices ):
      if i not in indices:
        indices.append( i )
    # alignment
    align_self = tuple( map( indices.index, self.indices ) )
    align_other = tuple( map( indices.index, other.indices ) )
    if common_indices:
      # dot the common axes
      return IndexedArray(
        ( (k, self._shape.get(k, other._shape.get(k, None))) for k in indices[:n_remaining] ),
        self._linked_lengths | other._linked_lengths | frozenset(frozenset([self._shape[k], other._shape[k]]) for k in common_indices),
        lambda geom, delayed_lengths, self_array, other_array: function.dot(
          function.align( self_array, align_self, len( indices ) ),
          function.align( other_array, align_other, len( indices ) ),
          range( -len(common_indices), 0 ) ),
        ( self, other ) )
    else:
      # no common axes, multiply `self` and `other`
      return IndexedArray(
        itertools.chain( self._shape.items(), other._shape.items() ),
        self._linked_lengths | other._linked_lengths,
        lambda geom, delayed_lengths, self_array, other_array:
          function.align( self_array, align_self, len( indices ) )
          * function.align( other_array, align_other, len( indices ) ),
        ( self, other ) )

  __rmul__ = __mul__

  def __truediv__( self, other ):
    try:
      other = asindexedarray( other )
    except ValueError:
      return NotImplemented
    if len( other.indices ) != 0:
      raise ValueError( 'cannot divide by an array, only a scalar' )
    return IndexedArray(
      self._shape,
      self._linked_lengths,
      lambda geom, delayed_lengths, self_array, other_array: operator.truediv( self_array, other_array ),
      ( self, other ) )

def asindexedarray( arg ):
  'convert `arg` to an `IndexedArray` if possible'

  if isinstance( arg, IndexedArray ):
    return arg
  elif isinstance( arg, (function.Array, numpy.ndarray) ) and len( arg.shape ) == 0:
    return IndexedArray( (), (), lambda geom, delayed_lengths: arg, () )
  elif isinstance( arg, (numbers.Number, numpy.generic) ):
    return IndexedArray( (), (), lambda geom, delayed_lengths: numpy.array( arg ), () )
  else:
    raise ValueError( 'cannot convert {!r} to a `IndexedArray`'.format( arg ) )

def wrap( array, indices ):
  '''wrap a scalar, numpy array or `Array` in an `IndexedArray`

  Parameters
  ----------
  array : scalar, numpy.ndarray or function.Array
  indices : str

  Returns
  -------
  IndexedArray
  '''

  if not isinstance( array, IndexedArray ):
    array = function.asarray( array )
  if not isinstance( indices, str ):
    raise ValueError( 'expected a `str`, got {!r}'.format( indices ) )
  # separate gradient indices from array indices
  if ',' in indices:
    indices, grad_indices = indices.split( ',', 1 )
    grad_indices = ','+grad_indices
  elif ';' in indices:
    indices, grad_indices = indices.split( ';', 1 )
    grad_indices = ';'+grad_indices
  else:
    grad_indices = ''
  if len( indices ) != array.ndim:
    raise ValueError( 'expected {} indices, got {}'.format( array.ndim, len( indices ) ) )
  if not all( 'a' <= index <= 'z' or '0' <= index <= '9' for index in indices + grad_indices[1:] ):
    raise ValueError( 'invalid index, only lower case latin characters and numbers are allowed' )
  if not all( 1 <= c <= 2 for index, c in collections.Counter( indices + grad_indices[1:] ).items() if not ('0' <= index <= '9') ):
    raise ValueError( 'indices may not be repeated more than once' )
  # apply numbered indices (skip gradient indices, will be processed in `self[grad_indices]` later)
  for i, index in reversed( tuple( enumerate( indices ) ) ):
    if '0' <= index <= '9':
      # get element `index` of axis `i` from `array`
      if isinstance( array, IndexedArray ):
        array = array._get_element( sorted(array.indices)[i], int(index) )
      else:
        array = array[(slice(None),)*i+(int(index),)+(slice(None),)*(len(indices)-i-1)]
      # remove this number from `indices`
      indices = indices[:i] + indices[i+1:]
  # sum repeated indices (skip gradient indices, will be processed in `self[grad_indices]` later)
  for repeated in sorted( ( i for i, c in collections.Counter( indices ).items() if c == 2 ), key = indices.index ):
    ax1 = indices.index( repeated )
    ax2 = indices.index( repeated, ax1+1 )
    indices = indices[:ax1] + indices[ax1+1:ax2] + indices[ax2+1:]
    if isinstance( array, IndexedArray ):
      sorted_array_indices = sorted( array.indices )
      array = array._trace( sorted_array_indices[ax1], sorted_array_indices[ax2] )
    else:
      array = function.trace( array, ax1, ax2 )
  if isinstance( array, IndexedArray ):
    # sort `indices` such that the original order matches `array.indices` sorted alphabetically
    shape = ( (indices[i], array._shape[array.indices[i]]) for i in sorted(range(array.ndim), key=lambda item: array.indices[item]) )
    self = IndexedArray( shape, array._linked_lengths, array._op, array._args )
  else:
    self = IndexedArray( zip( indices, array.shape ), (), lambda geom, delayed_lengths: array, () )
  # apply gradients, if any
  if grad_indices:
    self = self[grad_indices]
  return self


# vim:shiftwidth=2:softtabstop=2:expandtab:foldmethod=indent:foldnestmax=2
