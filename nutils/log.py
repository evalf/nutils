# -*- coding: utf8 -*-
#
# Module LOG
#
# Part of Nutils: open source numerical utilities for Python. Jointly developed
# by HvZ Computational Engineering, TU/e Multiscale Engineering Fluid Dynamics,
# and others. More info at http://nutils.org <info@nutils.org>. (c) 2014

"""
The log module provides print methods ``debug``, ``info``, ``user``,
``warning``, and ``error``, in increasing order of priority. Output is sent to
stdout as well as to an html formatted log file if so configured.
"""

from __future__ import print_function, division
import sys, time, os, warnings, re, functools
from . import core

warnings.showwarning = lambda message, category, filename, lineno, *args: \
  warning( '%s: %s\n  In %s:%d' % ( category.__name__, message, filename, lineno ) )

LEVELS = 'path', 'error', 'warning', 'user', 'progress', 'info', 'debug'


## STREAMS

class Stream( object ):
  '''File like object with a .write and .writelines method.'''

  def writelines( self, lines ):
    for line in lines:
      self.write( line )

  def flush( self ):
    pass

  def close( self ):
    self.flush()

  def write( self, text ):
    raise NotImplementedError( 'write method must be overloaded' )

class ColorStream( Stream ):
  '''Wraps all text in unix terminal escape sequences to select color and
  weight.'''

  _colors = 'black', 'red', 'green', 'yellow', 'blue', 'purple', 'cyan', 'white'

  def __init__( self, color, bold=False ):
    colorid = self._colors.index( color )
    boldid = 1 if bold else 0
    self.fmt = '\033[%d;3%dm%%s\033[0m' % ( boldid, colorid )

  def flush( self ):
    sys.stdout.flush()

  def write( self, text ):
    sys.stdout.write( self.fmt % text )

class HtmlStream( Stream ):
  '''Buffers all text, and sends it to html stream upon destruction wrapped in
  level-dependent html tag.'''

  def __init__( self, level, context, html ):
    self.level = level
    if context:
      self.buf = ' &middot; '
      self.head = self.buf.join( context )
    else:
      self.buf = None
      self.head = ''
    self.body = ''
    self.html = html
    self.isopen = True

  def write( self, text ):
    assert self.isopen
    if not text:
      return
    if self.buf:
      if text != '\n':
        self.head += self.buf
      self.buf = None
    self.body += text.replace( '<', '&lt;' ).replace( '>', '&gt;' )

  @staticmethod
  def _path2href( match ):
    whitelist = ['.jpg','.png','.svg','.txt'] + core.getprop( 'plot_extensions', [] )
    filename = match.group(0)
    ext = match.group(1)
    return '<a href="%s">%s</a>' % (filename,filename) if ext not in whitelist \
      else '<a href="%s" name="%s" class="plot">%s</a>' % (filename,filename,filename)

  def close( self ):
    assert self.isopen
    self.isopen = False

    body = self.body
    if self.level == 'path':
      body = re.sub( r'\b\w+([.]\w+)\b', self._path2href, body )
    if self.level:
      body = '<span class="%s">%s</span>' % ( self.level, body )
    line = '<span class="line">%s</span>' % ( self.head + body )

    self.html.write( line )
    self.html.flush()

  def __del__( self ):
    if self.isopen:
      self.close()

class PostponedStream( Stream ):
  '''Send postponedtext to postponedstream upon first written character, unless
  the first character is a carriage return. In that case postponedtext is
  dropped.'''

  def __init__( self, postponedstream, postponedtext, stream ):
    self.postponedstream = postponedstream
    self.postponedtext = postponedtext
    self.stream = stream

  def flush( self ):
    self.stream.flush()

  def write( self, text ):
    if not text:
      return
    if self.postponedtext:
      if text != '\n':
        self.postponedstream.write( self.postponedtext )
      self.postponedtext = None
    self.stream.write( text )

class Tee( Stream ):
  '''Duplicates output to several stream.'''

  def __init__( self, *streams ):
    self.streams = streams

  def flush( self ):
    for stream in self.streams:
      stream.flush()

  def close( self ):
    for stream in self.streams:
      stream.close()

  def write( self, text ):
    for stream in self.streams:
      stream.write( text )

class DevNull( Stream ):
  '''Discards all input.'''

  def write( self, text ):
    pass

class CaptureStream( Stream ):
  '''Append all output silently to chunks member.'''

  def __init__( self, chunks ):
    self.__chunks = chunks

  def write( self, text ):
    self.__chunks.append( text )


## STREAM FACTORIES

class StreamFactory( object ):
  '''Callable object that return a stream for given level and context.'''
  
  def __call__( self, level, context ):
    raise NotImplementedError( '__call__ method must be overloaded' )

class StdoutStreamFactory( StreamFactory ):
  '''Produces stdout stream, optionally with color depending on the richoutput
  property.'''

  def __init__( self ):
    if core.getprop( 'richoutput', False ):
      self.contextstream = ColorStream('black',True)
      self.sep = sep = ' · '
      self.streams = {
        'path': ColorStream( 'green', True ),
        'error': ColorStream( 'red', True ),
        'warning': ColorStream( 'red' ),
        'user': ColorStream( 'yellow', True ),
        'progress': self.contextstream }
    else:
      self.contextstream = sys.stdout
      self.sep = ' > '
      self.streams = {}
    StreamFactory.__init__( self )

  def __call__( self, level, context ):
    stream = self.streams.get( level, sys.stdout )
    if not context:
      return stream
    self.contextstream.write( self.sep.join(context) )
    return PostponedStream( self.contextstream, self.sep, stream )

class HtmlStreamFactory( StreamFactory ):
  '''Produces an html stream.'''

  def __init__( self, html ):
    self.html = html
    StreamFactory.__init__( self )

  def __call__( self, level, context ):
    return HtmlStream( level, context, self.html )

class TeeStreamFactory( StreamFactory ):
  '''Combines multiple factory output into a Tee stream'''

  def __init__( self, *factories ):
    self.factories = factories
    StreamFactory.__init__( self )

  def __call__( self, level, context ):
    return Tee( *[ factory(level,context) for factory in self.factories ] )
    
class ProgressStreamFactory( StreamFactory ):
  '''Factory wrapper that writes log level indication characters to a second stream.'''

  def __init__( self, stream, factory ):
    stream.flush()
    self.stream = stream
    self.factory = factory
    StreamFactory.__init__( self )

  def __call__( self, level, context ):
    self.stream.write( level[0] )
    self.stream.flush()
    return self.factory( level, context )

class CaptureStreamFactory( StreamFactory ):
  '''Capture all stream output silently in a 'captured' member.'''

  def __init__( self ):
    self.chunks = []
    StreamFactory.__init__( self )

  @property
  def captured( self ):
    return ''.join( self.chunks )

  def __call__( self, level, context ):
    sep = ' > '
    self.chunks.append( sep.join(context) )
    stream = CaptureStream( self.chunks )
    return PostponedStream( stream, sep, stream )


## LOG

class Log( object ):
  '''The log object is what is stored in the __log__ property. It contains the
  streamfactory and a mutable context list, to which items can be added via the
  .append method. Context items can be anything with a string representation,
  and will be ignored and purged if nonzero tests False. The log can be cloned
  with the .clone method.'''

  def __init__( self, streamfactory, context=() ):
    assert isinstance( streamfactory, StreamFactory )
    self.streamfactory = streamfactory
    self.context = context

  def _print( self, level, *args, **kw ):
    print( *args, file=self.getstream(level), **kw )

  def __getattr__( self, attr ):
    assert attr in LEVELS
    return functools.partial( self._print, attr )

  def clone( self ):
    return Log( self.streamfactory, self.context )

  def append( self, newitem ):
    self.context = [ item for item in self.context if item ]
    self.context.append( newitem )

  def getstream( self, level, verbosity=None ):
    if verbosity is None:
      verbosity = core.getprop('verbose',9)
    if level in LEVELS[ verbosity: ]:
      return DevNull()
    context = [ str(item) for item in self.context if item ]
    return self.streamfactory( level, context )


## INTERNAL FUNCTIONS

# references to objects that are going to be redefined
_range = range
_iter = iter
_zip = zip
_enumerate = enumerate

class _PrintableIterator( object ):
  'iterable context logger that updates progress info'

  def __init__( self, text, iterator, length=None ):
    self.__text = text
    self.__length = length
    self.__iterator = iterator
    self.__index = 0
    self.__alive = True
    
    # clock
    self.__dt = core.getprop( 'progress_interval', 1. )
    self.__dtexp = core.getprop( 'progress_interval_scale', 2 )
    self.__dtmax = core.getprop( 'progress_interval_max', 0 )
    self.__tnext = time.time() + self.__dt

    append( self )

  def __str__( self ):
    self.__tnext = time.time() + self.__dt
    return '%s %d' % ( self.__text, self.__index ) if self.__length is None \
      else '%s %d/%d (%d%%)' % ( self.__text, self.__index, self.__length, (self.__index-.5) * 100. / self.__length )

  def __nonzero__( self ): # python2
    return self.__alive

  def __bool__( self ): # python3
    return self.__alive

  def __iter__( self ):
    try:
      for item in self.__iterator:
        self.__index += 1
        now = time.time()
        if self.__alive and now > self.__tnext:
          if self.__dtexp != 1:
            self.__dt *= self.__dtexp
            if self.__dt > self.__dtmax > 0:
              self.__dt = self.__dtmax
          progress()
          self.__tnext = now + self.__dt
        yield item
    finally:
      self.__alive = False

def _len( iterable ):
  '''Return length if available, otherwise None'''

  try:
    return len(iterable)
  except:
    return None


## MODULE-ACCESIBLE LOG METHODS

__methods__ = LEVELS + ( 'getstream', 'clone', 'append' )

def _logmethod( attr ):
  def wrapper( *args, **kwargs ):
    log = core.getprop( 'log', None )
    if not isinstance( log, Log ):
      if log is not None:
        warnings.warn( '''Invalid logger object found: {!r}
          This is usually caused by manually setting the __log__ variable.'''.format(log), stacklevel=2 )
      log = Log( StdoutStreamFactory() )
    method = getattr( log, attr )
    return method( *args, **kwargs )
  wrapper.__name__ = attr
  return wrapper

locals().update({ name: _logmethod(name) for name in __methods__ })


## MODULE-ONLY METHODS

def range( title, *args ):
  '''Progress logger identical to built in range'''

  items = _range( *args )
  return _PrintableIterator( title, _iter(items), len(items) )

def iter( title, iterable, length=None ):
  '''Progress logger identical to built in iter'''

  return _PrintableIterator( title, _iter(iterable), length or _len(iterable) )

def enumerate( title, iterable, length=None ):
  '''Progress logger identical to built in enumerate'''

  return _PrintableIterator( title, _enumerate(iterable), length or _len(iterable) )

def zip( title, *iterables ):
  '''Progress logger identical to built in enumerate'''

  return _PrintableIterator( title, _zip(*iterables), None )

def count( title, start=0 ):
  '''Progress logger identical to itertools.count'''

  from itertools import count
  return _PrintableIterator( title, count(start), None )
    
def stack( msg ):
  '''Print stack trace'''

  from . import debug
  if isinstance( msg, tuple ):
    exc_type, exc_value, tb = msg
    msg = repr( exc_value )
    frames = debug.frames_from_traceback( tb )
  else:
    frames = debug.frames_from_callstack( depth=2 )
  print( msg, *reversed(frames), sep='\n', file=getstream( 'error' ) )

def title( f ): # decorator
  '''Decorator, adds title argument with default value equal to the name of the
  decorated function, unless argument already exists. The title value is used
  in a static log context that is destructed with the function frame.'''

  assert getattr( f, '__self__', None ) is None, 'cannot decorate bound instance method'
  default = f.__name__
  argnames = f.__code__.co_varnames[:f.__code__.co_argcount]
  if 'title' in argnames:
    index = argnames.index( 'title' )
    if index >= len(argnames) - len(f.__defaults__ or []):
      default = f.__defaults__[ index-len(argnames) ]
    gettitle = lambda args, kwargs: args[index] if index < len(args) else kwargs.get('title',default)
  else:
    gettitle = lambda args, kwargs: kwargs.pop('title',default)
  @functools.wraps(f)
  def wrapped( *args, **kwargs ):
    __log__ = clone()
    __log__.append( gettitle(args,kwargs) )
    return f( *args, **kwargs )
  return wrapped


# vim:shiftwidth=2:foldmethod=indent:foldnestmax=2
