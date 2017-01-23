from . import register, unittest
import sys, os, tempfile, io
import nutils.cli, nutils.log

def main(
  iarg: 'integer' = 1,
  farg: 'float' = 1.,
  sarg: 'string' = 'foo',
  parg: 'path' = nutils.cli.Path() ):
  assert isinstance( iarg, int ), 'n should be int, got {}'.format( type(iarg) )
  assert isinstance( farg, float ), 'f should be float, got {}'.format( type(farg) )
  assert isinstance( sarg, str ), 'f should be str, got {}'.format( type(sarg) )
  assert isinstance( parg, nutils.cli.Path ), 'f should be Path, got {}'.format( type(parg) )
  print( 'all OK' )

@register
def good():

  with tempfile.TemporaryDirectory() as outrootdir:

    _savestreams = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = stringio = io.StringIO()
    try:
      nutils.cli.run( main, args=['--outrootdir',outrootdir,'--pdb=false','--symlink=xyz','main','--iarg=1','--farg=1','--sarg=1','--parg=1'] )
    except SystemExit as e:
      status = e
    else:
      status = None
    finally:
      sys.stdout, sys.stderr = _savestreams

    @unittest
    def outdir():
      assert os.path.isdir( os.path.join(outrootdir,'__main__.py') ), 'output directory not found'

    @unittest
    def symlink():
      assert os.path.islink( os.path.join(outrootdir,'xyz') ), 'first symlink not found'
      assert os.path.islink( os.path.join(outrootdir,'__main__.py','xyz') ), 'second symlink not found'

    @unittest
    def argparse():
      output = stringio.getvalue()
      nutils.log.info( output )
      assert 'all OK' in output

    @unittest
    def exitstatus():
      assert status and status.code == 0, 'expected SystemExit 0, got {}'.format(status)

@register
def bad():

  with tempfile.TemporaryDirectory() as outrootdir:

    _savestreams = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = stringio = io.StringIO()
    try:
      nutils.cli.run( main, args=['--outrootdir',outrootdir,'--pdb=false','--symlink=xyz','main','--iarg=1','--farg=x','--sarg=1','--parg=1'] )
    except SystemExit as e:
      status = e
    else:
      status = None
    finally:
      sys.stdout, sys.stderr = _savestreams

    @unittest
    def outdir():
      assert not os.path.isdir( os.path.join(outrootdir,'__main__.py') ), 'outdir directory found'

    @unittest
    def argparse():
      output = stringio.getvalue()
      nutils.log.info( output )
      assert 'all OK' not in output

    @unittest
    def exitstatus():
      assert status and status.code == 2, 'expected SystemExit 2, got {}'.format(status)
