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

  pass


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
    self.html = html

  def getstream( self, level, *contexts ):
    return HtmlStream( level, contexts, self.html )


class ContextLog( Log ):
  'static text with parent'

  def __init__( self, title, parent=None ):
    self.title = title
    self.parent = parent or _getlog()

  def getstream( self, level, *contexts ):
    return self.parent.getstream( level, self.title, *contexts )


class IterLog( Log ):
  'iterable context logger that updates progress info'

  def __init__( self, title, iterator, length=None, parent=None ):
    self.title = title
    self.parent = parent or _getlog()
    self.length = length
    self.iterator = iterator
    self.index = -1

    # clock
    self.dt = core.getprop( 'progress_interval', 1. )
    self.dtexp = core.getprop( 'progress_interval_scale', 2 )
    self.dtmax = core.getprop( 'progress_interval_max', 0 )
    self.tnext = time.time() + self.dt

  def mktitle( self ):
    self.tnext = time.time() + self.dt
    return '%s %d' % ( self.title, self.index ) if self.length is None \
      else '%s %d/%d (%d%%)' % ( self.title, self.index, self.length, (self.index-.5) * 100. / self.length )

  def __iter__( self ):
    self.index = 0
    return self

  def next( self ):
    if time.time() > self.tnext:
      if self.dtexp != 1:
        self.dt *= self.dtexp
        if self.dt > self.dtmax > 0:
          self.dt = self.dtmax
      self.parent.getstream( 'progress' ).write( self.mktitle() + '\n' )
    self.index += 1
    try:
      return self.iterator.next()
    except:
      self.index = -1
      raise

  def getstream( self, level, *contexts ):
    return self.parent.getstream( level, self.mktitle(), *contexts ) if self.index >= 0 \
      else self.parent.getstream( level, *contexts )


class CaptureLog( Log ):
  'capture output without printing'

  def __init__( self ):
    self.buf = ''

  def __nonzero__( self ):
    return bool( self.buf )

  def __str__( self ):
    return self.buf

  def getstream( self, level, *contexts ):
    for context in contexts:
      self.buf += '%s > ' % context
    return self

  def write( self, text ):
    self.buf += text


# instances
stdlog = StdLog()

# helper functions
_getlog = lambda: core.getprop( 'log', stdlog )
_getstream = lambda level: _getlog().getstream( level )
_mklog = lambda level: lambda *args, **kw: print( *args, file=_getstream(level), **kw )


## MODULE METHODS


locals().update({ level: _mklog(level) for level in LEVELS })

def range( title, *args ):
  items = _range( *args )
  return IterLog( title, _iter(items), len(items) )

def iter( title, iterable, length=None, parent=None ):
  return IterLog( title, _iter(iterable), len(iterable) if length is None else length, parent )

def enumerate( title, iterable, length=None, parent=None ):
  return IterLog( title, _enumerate(iterable), len(iterable) if length is None else length, parent )

def count( title, start=0, parent=None ):
  from itertools import count
  return IterLog( title, count(start), None, parent )
    
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
    __log__ = ContextLog( kwargs.pop( 'title', f.func_name ) )
    return f( *args, **kwargs )
  return wrapped


# vim:shiftwidth=2:foldmethod=indent:foldnestmax=2
