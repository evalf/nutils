import os, sys, traceback

status = [ 0, 0, 0 ] # ok, failed, error

def unittest( func ):
  try:
    func()
  except KeyboardInterrupt:
    raise
  except AssertionError, e:
    status[1] += 1
    sys.stdout.write( 'F' )
    print
    print '##', func.func_name, 'FAILED(%d):' % status[1], e
  except:
    status[2] += 1
    sys.stdout.write( 'E' )
    print
    print '##', func.func_name, 'ERROR(%d)' % status[2]
    traceback.print_exc()
  else:
    status[0] += 1
    sys.stdout.write( '.' )
    sys.stdout.flush()
  return func

if __name__ == '__main__':
  path, myname = os.path.split( sys.argv[0] )
  count = 0
  for item in sorted( os.listdir( path ) ):
    name, ext = os.path.splitext( item )
    if item != myname and ext == '.py':
      count += 1
      print '%d.' % count, name.upper(), '|',
      __import__( name )
      print
  print '%d tests successful, %d failed, %d errors' % tuple( status )
  raise SystemExit( status[1] + status[2] )
