import os, sys, traceback
from nutils import log

status = [ 0, 0, 0 ] # ok, failed, error

class TestCaptureLog( log.CaptureLog ):
  def __call__( self, level, *contexts ):
    sys.stdout.write( '.' if level == 'progress' else level[0] )
    return log.CaptureLog.__call__( self, level, *contexts )
  def __str__( self ):
    return '---captured output---\n%s---------------------' % log.CaptureLog.__str__(self)
    
def testgroup( func ):
  def wrapped( *args, **kwargs ):
    firstarg = func.func_code.co_varnames[0]
    sys.stdout.write( ' * %s=%s [' % ( firstarg, args[0] if args else kwargs[firstarg] ) )
    sys.stdout.flush()
    __logger__ = TestCaptureLog()
    try:
      retval = func( *args, **kwargs )
    except:
      print '\n## ERROR IN SETUP CODE'
      if __logger__:
        print __logger__
      raise
    sys.stdout.write( ']\n' )
    return retval
  return wrapped

def unittest( func ):
  __logger__ = TestCaptureLog()
  try:
    func()
  except KeyboardInterrupt:
    raise
  except AssertionError, e:
    status[1] += 1
    sys.stdout.write( 'F' )
    print '\n##', func.func_name, 'FAILED(%d):' % status[1], e
    if __logger__:
      print __logger__
  except:
    status[2] += 1
    sys.stdout.write( 'E' )
    print '\n##', func.func_name, 'ERROR(%d)' % status[2]
    if __logger__:
      print __logger__
    traceback.print_exc()
  else:
    status[0] += 1
    sys.stdout.write( '&' )
    sys.stdout.flush()
  return func

if __name__ == '__main__':
  path, myname = os.path.split( sys.argv[0] )
  count = 0
  for item in sorted( os.listdir( path ) ):
    name, ext = os.path.splitext( item )
    if item != myname and ext == '.py':
      count += 1
      print '%d.' % count, name.upper()
      __import__( name )
      print
  print '%d tests successful, %d failed, %d errors' % tuple( status )
  raise SystemExit( status[1] + status[2] )
