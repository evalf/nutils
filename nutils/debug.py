# -*- coding: utf8 -*-
#
# Module DEBUG
#
# Part of Nutils: open source numerical utilities for Python. Jointly developed
# by HvZ Computational Engineering, TU/e Multiscale Engineering Fluid Dynamics,
# and others. More info at http://nutils.org <info@nutils.org>. (c) 2014

"""
The debug module provides a collection of tools to facilitate debugging.
"""

from . import numeric, log
import re, numpy

def base64_enc( obj, nsig, ndec ):
  import zlib, binascii, re
  serialized = __serialize( obj, nsig, ndec )
  binary = zlib.compress( '1,({},{}),{}'.format( nsig, ndec, serialized ).encode(), 9 )
  return re.sub( '(.{80})', r'\1\n', binascii.b2a_base64( binary ).decode() )

def base64_dec( base64 ):
  import zlib, binascii, re
  binary = binascii.a2b_base64( re.sub( '\s+', '', base64 ) )
  serialized = zlib.decompress( binary )
  proto, args, obj = eval( serialized, numpy.__dict__ )
  return proto, args, obj

def __serialize( obj, nsig, ndec ):
  if isinstance(obj, (int,str) ):
    return repr(obj)
  if isinstance(obj,dict):
    return '{%s,}' % ','.join( repr(n) + ':' + __serialize(o,nsig,ndec) for n, o in obj.items() )
  if not isinstance( obj, float ): # assume iterable
    return '(%s,)' % ','.join( __serialize(o,nsig,ndec) for o in obj )
  if not numpy.isfinite(obj): # nan, inf
    return str(obj)
  if not obj:
    return '0.'
  n = numeric.floor( numpy.log10( abs(obj) ) )
  N = max( n-(nsig-1), -ndec )
  i = int( obj * 10.**-N + ( .5 if obj >= 0 else -.5 ) )
  return '%de%d' % (i,N) if i and N else '%d.' % i

@log.title
def __compare( verify_obj, obj, nsig, ndec ):
  if isinstance(verify_obj,tuple):
    if not isinstance(obj,(tuple,list,numpy.ndarray)):
      log.error( 'non matching types: {} != {}'.format( type(verify_obj), type(obj) ) )
    elif len(verify_obj) != len(obj):
      log.error( 'non matching lenghts: {} != {}'.format( len(verify_obj), len(obj) ) )
    else:
      return all([ __compare( vo, o, nsig, ndec, title='#%d' % i )
        for i, (vo,o) in enumerate( zip( verify_obj, obj ) ) ])
  elif isinstance(verify_obj,dict):
    if not isinstance(obj,dict):
      log.error( 'non matching types: {} != {}'.format( type(verify_obj), type(obj) ) )
    elif len(verify_obj) != len(obj):
      log.error( 'non matching lenghts: {} != {}'.format( len(verify_obj), len(obj) ) )
    else:
      obj_keys, obj_vals = zip(*sorted(obj.items()))
      verify_obj_keys, verify_obj_vals = zip(*sorted(verify_obj.items()))
      if obj_keys != verify_obj_keys:
        log.error( 'non matching keys' )
      else:
        return all([ __compare( vo, o, nsig, ndec, title='#%d' % i )
          for i, (vo,o) in enumerate( zip( verify_obj_vals, obj_vals ) ) ])
  elif not isinstance(verify_obj,float):
    if type(verify_obj) != type(obj):
      log.error( 'non equal object types: {} != {}'.format( type(obj), type(verify_obj) ) )
    elif verify_obj != obj:
      log.error( 'non equal objects: {} != {}'.format( obj, verify_obj ) )
    else:
      return True
  elif numpy.isnan(verify_obj):
    if numpy.isnan(obj):
      return True
    log.error( 'expected nan: %s' % obj )
  elif numpy.isinf(verify_obj):
    if numpy.isinf(obj):
      return True
    log.error( 'expected inf: %s' % obj )
  else:
    if verify_obj:
      n = numeric.floor( numpy.log10( abs(verify_obj) ) )
      N = max( n-(nsig-1), -ndec )
    else:
      N = -ndec
    maxerr = .5 * 10.**N
    if abs(verify_obj-obj) <= maxerr:
      return True
    log.error( 'non equal to %s digits: %e != %e' % ( nsig, obj, verify_obj ) )
  return False

@log.title
def checkdata( obj, base64 ):
  try:
    proto, args, verify_obj = base64_dec( base64 )
    assert proto == 1, 'unsupported protocol version %s' % proto
    nsig, ndec = args
  except Exception as e:
    log.error( 'failed to decode base64 data: %s' % e )
    equal = False
    nsig = 4
    ndec = 15
  else:
    log.debug( 'checking %d significant digits up to %d decimals' % ( nsig, ndec ) )
    equal = __compare( verify_obj, obj, nsig, ndec, title='compare' )
  if not equal:
    s = base64_enc( obj, nsig, ndec )
    log.warning( 'objects are not equal; if this is expected replace base64 data with:\n%s' % s )
  return equal

def trace_uncollected( exhaustive=False ):
  import gc
  gc.set_debug( gc.DEBUG_SAVEALL )
  ncollect = gc.collect()
  if not ncollect:
    log.info( 'found no uncollected objects' )
    return
  log.warning( 'found %d uncollected objects' % ncollect )
  if not exhaustive:
    return
  garbage = { id(obj): obj for obj in gc.garbage }
  pointers = {}
  for n, obj in garbage.items():
    indices = [ id(refobj) for refobj in gc.get_referents(obj) if id(refobj) in garbage ]
    if indices:
      pointers[n] = indices
  changed = True
  while changed:
    changed = False
    for n, refs in list(pointers.items()):
      newrefs = [ i for i in refs if i in pointers ]
      if newrefs != refs:
        changed = True
        if not newrefs:
          pointers.pop( n )
        elif newrefs != refs:
          pointers[n] = newrefs
  strpointers = {}
  discard = []
  for n, refs in pointers.items():
    obj = garbage[n]
    if hasattr( obj, '__dict__' ) and id(obj.__dict__) in pointers:
      discard.append( id(obj.__dict__) )
      strpointers[n] = [ ( '.%s' % attr, id(refobj) ) for attr, refobj in obj.__dict__.items() if id(refobj) in pointers ]
    else:
      ids = { id(getattr(obj,attr)): '.%s' % attr for attr in obj.__slots__ } if hasattr( obj, '__slots__' ) and isinstance( obj.__slots__, tuple ) \
       else { id(item): '#%d' % iitem for iitem, item in enumerate(obj) } if isinstance( obj, (list,tuple) ) \
       else { id(item): repr(key) for key, item in obj.items() } if isinstance( obj, dict ) \
       else {}
      strpointers[n] = [ ( ids.get(i,'?'), i ) for i in refs ]
  for i in discard:
    strpointers.pop( i )
  if strpointers:
    log.warning( 'of which %d in circular reference:' % len(strpointers) )
    renumber = { n: i+1 for i, n in enumerate( strpointers ) }
    for n, refs in strpointers.items():
      refstr = ', '.join( s + '->%d' % renumber[i] for s, i in refs )
      log.warning( '%3d. %s: %s' % ( renumber[n], type(garbage[n]).__name__, refstr ) )


# vim:shiftwidth=2:softtabstop=2:expandtab:foldmethod=indent:foldnestmax=2
