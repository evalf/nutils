# Module INDEX
#
# Part of Nutils: open source numerical utilities for Python. Jointly developed
# by HvZ Computational Engineering, TU/e Multiscale Engineering Fluid Dynamics,
# and others. More info at http://nutils.org <info@nutils.org>. (c) 2015

from __future__ import print_function, division
from . import function, numpy
import sys, collections, operator, numbers


class IndexedArray:
  '''wrapper of `ArrayFunc` with index notation and Einstein summation convention

  This wrapper adds indices to the axes of the wrapped `ArrayFunc` which is used
  to align the `ArrayFunc` with other wrapped `ArrayFunc`s when doing simple
  arithmetic operations.  In addition, repeated indices are summed (Einstein
  summation convention).

  Passing an index string to `a[...]`, with `a` an `ArrayFunc`, wraps the array
  func.  The number of indices must match the number of axes of `a`.  The index
  string may contain only lower case latin characters.  A wrapped array can be
  unwrapped via the `unwrap` method.

  The index string may contain a `,` or `;` followed by any strict positive
  number of indices.  For all indices following the comma or semicolon
  respectively a gradient or surface gradient is computed with respect to the
  geometry specified in the `unwrap` method.

  Examples.  Let `a` and `b` be `ArrayFunc`s with shape `(n,n)` and `(n,)`.  The
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
  '''

  def __init__( self, indices, op, args ):
    'constructor'

    self.indices = indices
    self._op = op
    self._args = tuple( args )

    assert all( 'a' <= index <= 'z' for index in self.indices ), 'invalid index'
    assert len( set( self.indices ) ) == len( self.indices ), 'repeated index'
    assert all( isinstance( arg, IndexedArray ) for arg in self._args ), 'incompatible argument'

  @property
  def ndim( self ):
    return len( self.indices )

  def unwrap( self, geometry=None, indices=None ):
    '''unwrap the `ArrayFunc`

    Returns the wrapped `ArrayFunc` aligned according to `indices`.  If
    `indices` is `None`, the `ArrayFunc` is aligned alphabetically.  Otherwise,
    `indices` must contain all indices of this wrapped array and may not contain
    repeated indices.  The optional argument `geometry` must be specified when
    the wrapped array contains gradients.
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

    return function.align(
      self._unwrap_tree( geometry ),
      tuple( map( indices.index, self.indices ) ),
      len( self.indices ) )

  def _unwrap_tree( self, geom ):
    return self._op( geom, *( arg._unwrap_tree( geom ) for arg in self._args ) )

  @staticmethod
  def _array_grad( geom, array ):
    if geom is None:
      raise ValueError( '`geom` is required for unwrapping this `IndexedArray`' )
    return array.grad( geom )

  @staticmethod
  def _array_surfgrad( geom, array ):
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
      raise ValueError( 'additional indices are not allowed, only derivatives' )
    if not all( 'a' <= index <= 'z' for index in item[1:] ):
      raise ValueError( 'invalid index, only lower case latin characters are allowed' )
    if not all( 1 <= c <= 2 for c in collections.Counter( self.indices + item[1:] ).values() ):
      raise ValueError( 'indices may not be repeated more than once' )
    indices = self.indices + item[1]
    if item[1] in self.indices:
      # `item[1]` is a repeated index
      ax1 = indices.index( item[1] )
      ax2 = indices.index( item[1], ax1+1 )
      indices = indices[:ax1] + indices[ax1+1:ax2] + indices[ax2+1:]
      op = lambda geom, array: function.trace( grad( geom, array ), ax1, ax2 )
    else:
      # `item[1]` is not a repeated index
      op = grad
    result = IndexedArray( indices, op, ( self, ) )
    # apply remaining gradients
    if len( item ) > 2:
      result = result[item[0]+item[2:]]
    return result

  def __neg__( self ):
    return IndexedArray( self.indices, lambda *args: -self._op( *args ), self._args )

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
      left.indices,
      lambda geom, left_array, right_array:
        op( left_array, function.align( right_array, align_right, len( left.indices ) ) ),
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
    # collect all indices, common indices at the end, ordered on first appearance
    indices = [ i for i in self.indices if i not in other.indices ]
    indices.extend( i for i in other.indices if i not in self.indices )
    n = len( indices )
    indices.extend( i for i in self.indices if i in other.indices )
    indices = ''.join( indices )
    n_common = len( indices ) - n
    # alignment
    align_self = tuple( map( indices.index, self.indices ) )
    align_other = tuple( map( indices.index, other.indices ) )
    if n_common > 0:
      # dot last `n_common` axes
      return IndexedArray(
        indices[:n],
        lambda geom, self_array, other_array: function.dot(
          function.align( self_array, align_self, len( indices ) ),
          function.align( other_array, align_other, len( indices ) ),
          range( -n_common, 0 ) ),
        ( self, other ) )
    else:
      # no common axes, multiply `self` and `other`
      return IndexedArray(
        indices,
        lambda geom, self_array, other_array:
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
      self.indices,
      lambda geom, self_array, other_array: operator.truediv( self_array, other_array ),
      ( self, other ) )

  def __div__( self, other ):
    try:
      other = asindexedarray( other )
    except ValueError:
      return NotImplemented
    if len( other.indices ) != 0:
      raise ValueError( 'cannot divide by an array, only a scalar' )
    return IndexedArray(
      self.indices,
      lambda geom, self_array, other_array: operator.div( self_array, other_array ),
      ( self, other ) )

def asindexedarray( arg ):
  if isinstance( arg, IndexedArray ):
    return arg
  elif isinstance( arg, (function.ArrayFunc, numpy.ndarray) ) and len( arg.shape ) == 0:
    return IndexedArray( '', lambda geom: arg, () )
  elif isinstance( arg, (numbers.Number, numpy.generic) ):
    return IndexedArray( '', lambda geom: numpy.array( arg ), () )
  else:
    raise ValueError( 'cannot convert {!r} to a `IndexedArray`'.format( arg ) )

def wrap( array, indices ):
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
  if len( indices ) != len( array.shape ):
    raise ValueError( 'expected {} indices, got {}'.format( len( array.shape ), len( indices ) ) )
  if not all( 'a' <= index <= 'z' for index in indices + grad_indices[1:] ):
    raise ValueError( 'invalid index, only lower case latin characters are allowed' )
  if not all( 1 <= c <= 2 for c in collections.Counter( indices + grad_indices[1:] ).values() ):
    raise ValueError( 'indices may not be repeated more than once' )
  # sum repeated indices (skip gradient indices, will be processed in `self[grad_indices]` later)
  for repeated in sorted( ( i for i, c in collections.Counter( indices ).items() if c == 2 ), key = indices.index ):
    ax1 = indices.index( repeated )
    ax2 = indices.index( item[1], ax1+1 )
    indices = indices[:ax1] + indices[ax1+1:ax2] + indices[ax2+1:]
    array = function.trace( array, ax1, ax2 )
  self = IndexedArray( indices, lambda geom: array, () )
  # apply gradients, if any
  if grad_indices:
    self = self[grad_indices]
  return self


# vim:shiftwidth=2:softtabstop=2:expandtab:foldmethod=indent:foldnestmax=2
