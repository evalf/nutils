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

from __future__ import print_function
import sys, time, os, warnings, re
from . import core

warnings.showwarning = lambda message, category, filename, lineno, *args: \
  warning( '%s: %s\n  In %s:%d' % ( category.__name__, message, filename, lineno ) )

LEVELS = 'path', 'error', 'warning', 'user', 'info', 'progress', 'debug'


# references to objects that are going to be redefined
_range = range
_iter = iter
_enumerate = enumerate


## STREAMS


class Stream( object ):

  def writelines( self, lines ):
    for line in lines:
      self.write( line )

    
class ColorStream( Stream ):

  # color = 0:black, 1:red, 2:green, 3:yellow, 4:blue, 5:purple, 6:cyan, 7:white
  # bold = 0:no, 1:yes

  def __init__( self, color, bold ):
    self.fmt = '\033[%d;3%dm%%s\033[0m' % ( bold, color )

  def write( self, text ):
    sys.stdout.write( self.fmt % text )


class HtmlStream( Stream ):
  'html line stream'

  def __init__( self, level, contexts, html ):
    'constructor'

    self.out = stdlog.getstream( level, *contexts )
    self.level = level
    self.head = ''.join( '%s &middot; ' % context for context in contexts )
    self.body = ''
    self.html = html

  def write( self, text ):
    'write to out and buffer for html'

    self.out.write( text )
    self.body += text.replace( '<', '&lt;' ).replace( '>', '&gt;' )

  @staticmethod
  def _path2href( match ):
    whitelist = ['.jpg','.png','.svg','.txt'] + core.getprop( 'plot_extensions', [] )
    filename = match.group(0)
    ext = match.group(1)
    return '<a href="%s">%s</a>' % (filename,filename) if ext not in whitelist \
      else '<a href="%s" name="%s" class="plot">%s</a>' % (filename,filename,filename)

  def __del__( self ):
    'postprocess buffer and write to html'

    body = self.body
    if self.level == 'path':
      body = re.sub( r'\b\w+([.]\w+)\b', self._path2href, body )
    if self.level:
      body = '<span class="%s">%s</span>' % ( self.level, body )
    line = '<span class="line">%s</span>' % ( self.head + body )

    self.html.write( line )
    self.html.flush()


class DevNull( Stream ):

  def write( self, text ):
    pass

# instances
devnull = DevNull()
boldgreen = ColorStream(2,1)
boldred = ColorStream(1,1)
red = ColorStream(1,0)
boldblue = ColorStream(4,1)
boldyellow = ColorStream(3,1)
gray = ColorStream(0,1)

colorpicker = {
  'path': boldgreen,
  'error': boldred,
  'warning': red,
  'user': boldyellow,
  'progress': gray,
}


## LOGGERS


class Log( object ):

  def kill( self ):
    raise Exception, 'primary loggers are immortable'

  @property
  def living( self ):
    return self

  @property
  def alive( self ):
    return True

  @property
  def parent( self ):
    raise Exception, 'primary loggers have no parents'


class StdLog( Log ):

  def getstream( self, level, *contexts ):
    verbosity = core.getprop( 'verbose', 6 )
    if level in LEVELS[ verbosity: ]:
      stream = devnull
    elif core.getprop( 'richoutput', False ):
      gray.writelines( '%s · ' % context for context in contexts )
      stream = colorpicker.get(level,sys.stdout)
    else:
      sys.stdout.writelines( '%s > ' % context for context in contexts )
      stream = sys.stdout
    return stream


class HtmlLog( Log ):
  'html log'

  def __init__( self, html ):
    Log.__init__( self )
    self.__html = html

  def getstream( self, level, *contexts ):
    return HtmlStream( level, contexts, self.__html )


class CaptureLog( Log ):
  'capture output without printing'

  def __init__( self ):
    Log.__init__( self )
    self.__buf = ''

  def __nonzero__( self ):
    return bool( self.__buf )

  def __str__( self ):
    return self.__buf

  def getstream( self, level, *contexts ):
    for context in contexts:
      self.__buf += '%s > ' % context
    return self

  def write( self, text ):
    self.__buf += text


class ContextLog( Log ):

  def __init__( self ):
    self.__parent = _getlog()
    self.__alive = True
    self.__awake = True

  @property
  def living( self ):
    return self if self.__alive else self.parent

  def kill( self ):
    self.__alive = False

  def sleep( self ):
    self.__awake = False

  def wake( self ):
    self.__awake = True

  @property
  def alive( self ):
    return self.__alive

  @property
  def awake( self ):
    return self.__alive

  @property
  def parent( self ):
    if not self.__parent.alive:
      self.__parent = self.__parent.parent
    return self.__parent

  def getstream( self, level, *contexts ):
    assert self.__alive, 'logging from a dead object'
    return self.parent.getstream( level, self.getcontext(), *contexts ) if self.awake \
      else self.parent.getstream( level, *contexts )

  def getcontext( self ):
    return '<no context>'


class StaticLog( ContextLog ):
  'static text with parent'

  def __init__( self, context ):
    ContextLog.__init__( self )
    self.__context = context

  def getcontext( self ):
    return self.__context


class IterLog( ContextLog ):
  'iterable context logger that updates progress info'

  def __init__( self, context, iterator, length=None ):
    ContextLog.__init__( self )

    self.__context = context
    self.__length = length
    self.__iterator = iterator
    self.__index = 0

    # clock
    self.__dt = core.getprop( 'progress_interval', 1. )
    self.__dtexp = core.getprop( 'progress_interval_scale', 2 )
    self.__dtmax = core.getprop( 'progress_interval_max', 0 )
    self.__tnext = time.time() + self.__dt

    self.sleep()

  def getcontext( self ):
    self.__tnext = time.time() + self.__dt
    return '%s %d' % ( self.__context, self.__index ) if self.__length is None \
      else '%s %d/%d (%d%%)' % ( self.__context, self.__index, self.__length, (self.__index-.5) * 100. / self.__length )

  def __iter__( self ):
    try:
      self.wake()
      for item in self.__iterator:
        self.__index += 1
        if time.time() > self.__tnext:
          if self.__dtexp != 1:
            self.__dt *= self.__dtexp
            if self.__dt > self.__dtmax > 0:
              self.__dt = self.__dtmax
          self.parent.getstream( 'progress' ).write( self.getcontext() + '\n' )
        yield item
    finally:
      self.kill()


# instances

stdlog = StdLog()

# helper functions

_getlog = lambda: core.getprop( 'log', stdlog ).living
_getstream = lambda level: _getlog().getstream( level )
_mklog = lambda level: lambda *args, **kw: print( *args, file=_getstream(level), **kw )


## MODULE METHODS


locals().update({ level: _mklog(level) for level in LEVELS })

def range( title, *args ):
  items = _range( *args )
  return IterLog( title, _iter(items), len(items) )

def iter( title, iterable, length=None ):
  if length is None:
    try:
      length = len(iterable)
    except:
      pass
  return IterLog( title, _iter(iterable), length )

def enumerate( title, iterable, length=None ):
  if length is None:
    try:
      length = len(iterable)
    except:
      pass
  return IterLog( title, _enumerate(iterable), length )

def count( title, start=0 ):
  from itertools import count
  return IterLog( title, count(start), None )
    
def stack( msg ):
  'print stack trace'

  from . import debug
  if isinstance( msg, tuple ):
    exc_type, exc_value, tb = msg
    msg = repr( exc_value )
    frames = debug.frames_from_traceback( tb )
  else:
    frames = debug.frames_from_callstack( depth=2 )
  print( msg, *reversed(frames), sep='\n', file=_getstream( 'error' ) )

def title( f ): # decorator
  def wrapped( *args, **kwargs ):
    __log__ = StaticLog( kwargs.pop( 'title', f.func_name ) )
    return f( *args, **kwargs )
  return wrapped


# vim:shiftwidth=2:foldmethod=indent:foldnestmax=2
