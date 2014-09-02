# -*- coding: utf8 -*-
#
# Module DEBUG
#
# Part of Nutils: open source numerical utilities for Python. Jointly developed
# by HvZ Computational Engineering, TU/e Multiscale Engineering Fluid Dynamics,
# and others. More info at http://nutils.org <info@nutils.org>. (c) 2014

"""
The debug module provides code inspection tools and the "traceback explorer"
interactive shell environment. Access to these components is primarily via
:func:`breakpoint` and an exception handler in :func:`nutils.util.run`.
"""

from . import core, cache, numeric, log
import sys, cmd, re, os, linecache, numpy

class Frame( object ):
  'frame info'

  def __init__( self, frame, lineno=None ):
    'constructor'

    self.frame = frame
    self.lineno = lineno or self.frame.f_lineno

  @staticmethod
  def _name( frame ):
    # If frame is a class method try to add the class name
    # http://stackoverflow.com/questions/2203424/python-how-to-retrieve-class-information-from-a-frame-object/15704609#15704609
    name = frame.f_code.co_name
    for classname, obj in frame.f_globals.iteritems():
      try:
        assert obj.__dict__[name].func_code is frame.f_code
      except:
        pass
      else:
        return '%s.%s' % ( classname, name )
    return name

  def getline( self, lineno ):
    return linecache.getline( self.frame.f_code.co_filename, lineno )

  @cache.property
  def where( self ):
    relpath = os.path.relpath( self.frame.f_code.co_filename )
    name = self._name( self.frame )
    return 'File "%s", line %d, in %s' % ( relpath, self.lineno, name )

  @cache.property
  def context( self ):
    return '  %s\n    %s' % ( self.where,self.getline( self.lineno ).strip() )

  @cache.property
  def source( self ):
    path = self.frame.f_code.co_filename
    lineno = self.frame.f_code.co_firstlineno
    line = self.getline( lineno )
    if not line:
      return '<not avaliable>'
    indent = len(line) - len(line.lstrip())
    source = line[indent:]
    while line[indent] == '@':
      lineno += 1
      line = self.getline( lineno )
      source += line[indent:]
    linebreak = False
    while True:
      lineno += 1
      line = linecache.getline( path, lineno )
      if line[:indent+1].lstrip().startswith('#'):
        line = ' ' * indent + line.lstrip()
      elif not line or line[:indent+1].strip() and not linebreak:
        break
      source += '>' + line[indent+1:] if lineno == self.lineno else line[indent:]
      linebreak = line.rstrip().endswith( '\\' )
    return source.rstrip() # strip trailing empty lines

  def __str__( self ):
    'string representation'

    return self.context

class Explorer( cmd.Cmd ):
  'traceback explorer'

  def __init__( self, msg, frames, intro ):
    'constructor'

    cmd.Cmd.__init__( self, completekey='tab' )

    lines = [ 'WELCOME TO TRACEBACK EXPLORER.' ]
    maxlen = 44
    nextline = ''
    for word in intro.split():
      if not nextline or len(nextline) + 1 + len(word) > maxlen:
        lines.append( nextline )
        nextline = word
      else:
        nextline += ' ' + word
    lines.append( nextline )
    if core.getprop( 'richoutput', False ):
      ul = '┌'
      ur = '┐'
      ll = '└'
      lr = '┘'
      hh = '─'
      vv = '│'
    else:
      ul = ur = ll = lr = '+'
      hh = '-'
      vv = '|'
    self.intro = '\n'.join(
        [ ul + hh * (maxlen+2) + ur ]
      + [ vv + (' '+line).ljust(maxlen+2) + vv for line in lines ]
      + [ ll + hh * (maxlen+2) + lr ] )

    self.msg = msg
    self.frames = frames
    self.index = len(frames) - 1
    self.prompt = '\n>>> '

  def show_context( self ):
    'show traceback up to index'

    for i, f in enumerate(self.frames):
      print ' *'[i == self.index] + f.context[1:]
    print ' ', self.msg

  def do_s( self, arg ):
    '''Show source code of the currently focussed frame.'''

    for f in self.frames[:self.index+1]:
      print f.where
    print f.source

  def do_l( self, arg ):
    '''List the stack and exception type'''

    self.show_context()

  def do_q( self, arg ):
    '''Quit traceback exploror.'''

    print 'quit.'
    return True

  def do_u( self, arg ):
    '''Shift focus to the frame above the current one.'''

    if self.index > 0:
      self.index -= 1
      self.show_context()

  def do_d( self, arg ):
    '''Shift focus to the frame below the current one.'''

    if self.index < len(self.frames)-1:
      self.index += 1
      self.show_context()

  def do_w( self, arg ):
    '''Show overview of local variables.'''

    frame = self.frames[self.index].frame
    maxlen = max( len(name) for name in frame.f_locals )
    fmt = '  %' + str(maxlen) + 's : %s'
    for item in frame.f_locals.iteritems():
      print fmt % item

  def do_p( self, arg ):
    '''Print local of global variable, or function evaluation.'''

    frame = self.frames[self.index].frame
    print eval(arg,frame.f_globals,frame.f_locals)

  def onecmd( self, text ):
    'wrap command handling to avoid a second death'

    try:
      return cmd.Cmd.onecmd( self, text )
    except Exception, e:
      print '%s in %r:' % ( e.__class__.__name__, text ), e

  def do_pp( self, arg ):
    '''Pretty-print local of global variable, or function evaluation.'''

    import pprint
    frame = self.frames[self.index].frame
    pprint.pprint( eval(arg,frame.f_globals,frame.f_locals) )

  def completedefault( self, text, line, begidx, endidx ):
    'complete object names'

    frame = self.frames[self.index].frame

    objs = {}
    objs.update( frame.f_globals )
    objs.update( frame.f_locals )

    base = ''
    while '.' in text:
      objname, attr = text.split( '.', 1 )
      try:
        obj = objs[ objname ]
      except KeyError:
        return []
      objs = {}
      for attrname in dir(obj):
        try:
          objs[attrname] = getattr(obj,attrname)
        except:
          pass
      base += objname + '.'
      text = attr

    return [ base+name for name in objs if name.startswith(text) ]

def format_exc():
  exc_type, exc_value, tb = sys.exc_info()
  return '\n'.join( [ repr(exc_value) ] + [ str(f) for f in frames_from_traceback( tb ) ] )

def frames_from_traceback( tb ):
  frames = []
  while tb:
    frames.append( Frame( tb.tb_frame, tb.tb_lineno ) )
    tb = tb.tb_next
  return frames

def frames_from_callstack( depth=1 ):
  try:
    frame = sys._getframe( depth )
  except ValueError:
    return []
  frames = []
  while frame:
    frames.append( Frame( frame ) )
    frame = frame.f_back
  return frames[::-1]

def write_html( out, exc_info ):
  'write exception info to html file'

  exc_type, exc_value, tb = exc_info
  frames = frames_from_traceback( tb )
  out.write( '<span class="info">' )
  out.write( '\n<hr/>' )
  out.write( '<b>EXHAUSTIVE POST-MORTEM DUMP FOLLOWS</b>\n' )
  out.write( '\n'.join( [ repr(exc_value) ] + [ str(f) for f in frames ] ) )
  out.write( '<hr/>\n' )
  for f in reversed(frames):
    out.write( f.context.splitlines()[0] + '\n' )
    for line in f.source.splitlines():
      if line.startswith( '>' ):
        fmt = '<span class="error"> %s</span>\n'
        line = line[1:]
      else:
        fmt = '%s\n'
      line = re.sub( r'\b(def|if|elif|else|for|while|with|in|return)\b', r'<b>\1</b>', line.replace('<','&lt;').replace('>','&gt;') )
      out.write( fmt % line )
    out.write( '\n\n' )
    out.write( '<table border="1" style="border:none; margin:0px; padding:0px;">\n' )
    for key, val in f.frame.f_locals.iteritems():
      try:
        val = str(val).replace('<','&lt;').replace('>','&gt;')
      except:
        val = 'ERROR'
      out.write( '<tr><td>%s</td><td>%s</td></tr>\n' % ( key, val ) )
    out.write( '</table>\n' )
    out.write( '<hr/>\n' )
  out.write( '</span>' )
  out.flush()

def traceback_explorer( exc_info ):
  exc_type, exc_value, tb = exc_info
  Explorer( repr(exc_value), frames_from_traceback(tb), '''
    Your program has died. The traceback exporer allows you to examine its
    post-mortem state to figure out why this happened. Type 'help' for an
    overview of commands to get going.''' ).cmdloop()

def breakpoint():
  Explorer( 'Suspended.', frames_from_callstack(2), '''
    Your program is suspended. The traceback explorer allows you to examine
    its current state and even alter it. Closing the explorer will resume
    program execution.''' ).cmdloop()

def base64_enc( obj, nsig, ndec ):
  import zlib, binascii, re
  serialized = __serialize( obj, nsig, ndec )
  binary = zlib.compress( '1,(%d,%d),%s' % ( nsig, ndec, serialized ), 9 )
  return re.sub( '(.{80})', r'\1\n', binascii.b2a_base64( binary ) )

def base64_dec( base64 ):
  import zlib, binascii, re
  binary = binascii.a2b_base64( re.sub( '\s+', '', base64 ) )
  serialized = zlib.decompress( binary )
  proto, args, obj = eval( serialized, numpy.__dict__ )
  return proto, args, obj

def __serialize( obj, nsig, ndec ):
  if isinstance(obj,int):
    return str(obj)
  if isinstance(obj,str):
    return repr(obj)
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
    if len(verify_obj) == len(obj):
      return all([ __compare( vo, o, nsig, ndec, title='#%d' % i )
        for i, (vo,o) in enumerate( zip( verify_obj, obj ) ) ])
    log.error( 'non matching lenghts: %d != %d' % ( len(verify_obj), len(obj) ) )
  elif not isinstance(verify_obj,float):
    if verify_obj == obj:
      return True
    log.error( 'non equal: %s != %d' % ( obj, verify_obj ) )
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
  import zlib, binascii, re
  try:
    proto, args, verify_obj = base64_dec( base64 )
    assert proto == 1, 'unsupported protocol version %s' % proto
    nsig, ndec = args
  except Exception, e:
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

# vim:shiftwidth=2:foldmethod=indent:foldnestmax=2
