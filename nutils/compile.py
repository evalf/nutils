# Module FUNCTIONALS
#
# Part of Nutils: open source numerical utilities for Python. Jointly developed
# by HvZ Computational Engineering, TU/e Multiscale Engineering Fluid Dynamics,
# and others. More info at http://nutils.org <info@nutils.org>. (c) 2016

from . import topology, matrix, function, index, numpy, util, core
import inspect


class ArgumentNotUnwrappedError( Exception ): pass


class TopologyArgument:
  'Topology argument'

  def __init__( self, arg, ndims, get=() ):
    self._arg = arg
    self.ndims = ndims
    self._get = tuple( get )

  def __hash__( self ):
    return hash( ( self._arg, self.ndims, self._get ) )

  def __eq__( self, other ):
    return (
      self.__class__ == other.__class__ and self._arg == other._arg and
      self.ndims == other.ndims and self._get == other._get
    )

  @core.single_or_multiple
  def integrate( self, integrands, *, ischeme, geometry=None, edit=topology._identity ):
    integrands = topology.Topology._integrate_prepare( integrands, geometry, edit, self.ndims )
    return [
      Integrals( { ( self._arg, self._get, ischeme ): integrand } )
      for integrand in integrands
    ]
#   if isinstance( integrand, index.IndexedArray ):
#     integrand = integrand.unwrap( geometry )
#   if geometry:
#     integrand = integrand * function.J( geometry, self.ndims )
#   return Integrals( { ( self._arg, self._get, ischeme ): integrand } )

  def __getitem__( self, item ):
    return TopologyArgument( self.ndims, self._get+(('__getitem__', item),) )

  @property
  def boundary( self ):
    return TopologyArgument( self.ndims-1, self._get+(('boundary', None),) )

  @property
  def interfaces( self ):
    return TopologyArgument( self.ndims-1, self._get+(('interfaces', None),) )


class ArrayFuncArgument( function.ArrayFunc ):
  'ArrayFunc argument'

  def __init__( self, arg, shape ):
    self._arg = arg
    function.ArrayFunc.__init__( self, args=[], shape=shape )

  def __hash__( self ):
    return hash( self._arg )

  def __eq__( self, other ):
    return self.__class__ == other.__class__ and self._arg == other._arg

  def evalf( self ):
    raise ArgumentNotUnwrappedError( 'unwrap {!r} before evaluation'.format( self ) )

  def _edit( self, op ):
    return self

  def _derivative( self, var, shape, seen ):
    #return function._zeros( self.shape + shape )
    if isinstance( var, tuple ) and len( var ) == 2 and var[0] == self:
      axes = var[1]
      assert shape == tuple( self.shape[axis] for axis in axes )
      result = 1
      for i, axis in enumerate( axes ):
        result *= function.align( function.eye( self.shape[axis] ), ( axis, self.ndim+i ), self.ndim+len(axes) )
      return result
    else:
      return function._zeros( self.shape + shape )


class Integrals:
  'integrals'

  def __init__( self, integrals ):
    self._integrals = util.OrderedDict( integrals )

  @property
  def shape( self ):
    return next( iter( self._integrals.values() ) ).shape

  def __mul__( self, other ):
    if isinstance( other, numbers.Number ):
      return Integrals( ( k, v * other ) for k, v in self._integrals.items() )
    else:
      return NotImplemented

  __rmul__ = __mul__

  def __truediv__( self, other ):
    if isinstance( other, numbers.Number ):
      return Integrals( ( k, v / other ) for k, v in self._integrals.items() )
    else:
      return NotImplemented

  def __add__( self, other ):
    if isinstance( other, ( function.ArrayFunc, matrix.Matrix ) ):
      other = Integrals( [( None, other )] )
    if isinstance( other, Integrals ):
      integrals = util.OrderedDict( self._integrals )
      for k, v in other._integrals.items():
        if k in integrals:
          integrals[k] += v
        else:
          integrals[k] = v
      return Integrals( integrals )
    elif other == 0:
      return self
    else:
      return NotImplemented

  __radd__ = __add__

  def __sub__( self, other ):
    if isinstance( other, ( Integrals, function.ArrayFunc, matrix.Matrix ) ):
      return self + (-other)
    elif other == 0:
      return self
    else:
      return NotImplemented

  def __rsub__( self, other ):
    if isinstance( other, ( Integrals, function.ArrayFunc, matrix.Matrix ) ):
      return (-self) + other
    elif other == 0:
      return -self
    else:
      return NotImplemented

  def __neg__( self ):
    return Integrals( ( k, -v ) for k, v in self._integrals.items() )

  def _derivative( self, var, shape, seen ):
    if var == 'localcoords':
      return 0
    else:
      return Integrals(
        ( k, function.derivative( v, var, shape, seen ) )
        for k, v in self._integrals.items()
      )

  def _edit( self, op ):

    return Integrals( ( k, op( v ) ) for k, v in self._integrals )

  def _eval( self, wrapped_args ):
    def _replace( a ):
      try:
        return wrapped_args[a]
      except ( KeyError, TypeError ):
        return a
    replace = lambda f: function.edit( _replace( f ), replace )
    result = None
    for key, integrand in self._integrals.items():
      integrand = replace( integrand )
      if key:
        topology, get, ischeme = key
        topology = wrapped_args.get( topology, topology )
        for item, arg in get:
          if item == '__getitem__':
            topology = topology[arg]
          elif item == 'boundary':
            topology = topology.boundary
          elif item == 'interfaces':
            topology = topology.interfaces
          else:
            raise ValueError
        partial_result = topology.integrate( integrand, ischeme=ischeme )
        if result is None:
          result = partial_result
        else:
          result += partial_result
      else:
        result += integrand
    return result


class Compiled:
  'compiled function class'

  def __init__( self, func, derivative=(), cache=None, func_key=None ):
    self.__wrapped__ = func
    self._derivative = tuple(derivative)
    self._cache = {} if cache is None else cache
    self._func_key = object() if func_key is None else func_key

  def J( self, arg_key, axes ):
    sig = inspect.signature( self )
    if isinstance( arg_key, str ) and arg_key in sig.parameters:
      # convert `arg_key` to index if possible
      param = sig.parameters[arg_key]
      if param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD):
        for i, (n, p) in enumerate( self._signature.parameters.items() ):
          if p.kind not in (p.POSITIONAL, p.POSITIONAL_OR_KEYWORD):
            break
          if n == arg_key:
            arg_key = i
            break
    return Compiled( self.__wrapped__, self._derivative+((arg_key, tuple(axes)),), self._cache, self._func_key )

  def _wrap_arg( self, arg, name, wrapped_args ):
    key = self._func_key, name
    if isinstance( arg, ( numpy.ndarray, function.ArrayFunc ) ):
      wrapped_arg = ArrayFuncArgument( key, arg.shape )
      key = wrapped_arg
    elif isinstance( arg, topology.Topology ):
      wrapped_arg = TopologyArgument( key, arg.ndims )
    else:
      return arg
    wrapped_args[key] = arg
    return wrapped_arg

  def __call__( *args, **kwargs ):
    self, *args = args
    sig = inspect.signature( self.__wrapped__ )
    ba = sig.bind( *args, **kwargs )
    # apply defaults
    for param in sig.parameters.values():
      if (param.name not in ba.arguments and param.default is not param.empty):
        ba.arguments[param.name] = param.default
    # wrap ArrayFunc, Topology arguments
    wrapped_args = {}
    key = [self._derivative]
    for param in sig.parameters.values():
      arg = ba.arguments[param.name]
      if param.kind == param.VAR_POSITIONAL:
        arg = tuple(
          ( i, self._wrap_arg( arg_i, (param.name,i), wrapped_args ) )
          for i, arg_i in sorted( args.items() )
        )
        key.append( arg )
        arg = dict( arg )
      elif param.kind == param.VAR_KEYWORD:
        arg = tuple(
          self._wrap_arg( arg_i, (param.name,i), wrapped_args )
          for i, arg_i in enumerate(args)
        )
        key.append( arg )
      else:
        arg = self._wrap_arg( arg, param.name, wrapped_args )
        key.append( arg )
      ba.arguments[param.name] = arg
    key = tuple( key )
    # get from cache or create compiled function
    try:
      result = self._cache[key]
    except KeyError:
      #f = self.__wrapped__
      #for arg, axes in self._derivative:
      #  f = function.partial_derivative( f, arg, axes )
      #result = f( *ba.args, **ba.kwargs )

      result = self.__wrapped__( *ba.args, **ba.kwargs )
      for arg_key, arg_axes in self._derivative:

        # get derivative argument
        if isinstance(arg_key, int):
          arg = ba.args[arg_key]
        elif arg_key in ba.kwargs:
          arg = ba.kwargs[arg_key]
        else:
          raise ValueError( 'argument not found: {!r}'.format( arg_key ) )

        # apply derivative
        shape = tuple( arg.shape[i] for i in arg_axes )
        result = function.derivative( result, ( arg, arg_axes ), shape )

      self._cache[key] = result
    except TypeError:
      # unhashable key
      raise ValueError( 'cannot compile for given arguments' )
    # evaluate result
    if isinstance( result, Integrals ):
      result = result._eval( wrapped_args )
    return result


def compile( func ):
  if not isinstance( func, Compiled ):
    func = Compiled( func )
  return func


# vim:shiftwidth=2:softtabstop=2:expandtab:foldmethod=indent:foldnestmax=2
