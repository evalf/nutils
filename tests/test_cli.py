import sys, os, tempfile, io
import nutils.cli, nutils.log
from . import *

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

@parametrize
class run(ContextTestCase):

  scriptname = 'test.py'

  def setUpContext(self, stack):
    super().setUpContext(stack)
    self.outrootdir = stack.enter_context(tempfile.TemporaryDirectory())

  def test_good(self):
    _savestreams = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = stringio = io.StringIO()
    try:
      if self.method == 'run':
        nutils.cli.run(main, args=['--outrootdir',self.outrootdir,'--pdb=false','--symlink=xyz','--iarg=1','--farg=1','--sarg=1','--parg=1'], scriptname=self.scriptname)
      else:
        nutils.cli.choose(main, args=['--outrootdir',self.outrootdir,'--pdb=false','--symlink=xyz','main','--iarg=1','--farg=1','--sarg=1','--parg=1'], scriptname=self.scriptname)
    except SystemExit as e:
      status = e
    else:
      status = None
    finally:
      sys.stdout, sys.stderr = _savestreams

    with self.subTest('outdir'):
      print(os.listdir(self.outrootdir))
      self.assertTrue(os.path.isdir(os.path.join(self.outrootdir,self.scriptname)), 'output directory not found')

    with self.subTest('first-symlink'):
      self.assertTrue(os.path.islink(os.path.join(self.outrootdir,'xyz')), 'first symlink not found')
    with self.subTest('second-symlink'):
      self.assertTrue(os.path.islink(os.path.join(self.outrootdir,self.scriptname,'xyz')), 'second symlink not found')

    with self.subTest('argparse'):
      output = stringio.getvalue()
      nutils.log.info( output )
      self.assertIn('all OK', output)

    with self.subTest('exitstatus'):
      self.assertIsNotNone(status)
      self.assertEqual(status.code, 0)

  def test_bad(self):
    _savestreams = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = stringio = io.StringIO()
    try:
      if self.method == 'run':
        nutils.cli.run(main, args=['--outrootdir',self.outrootdir,'--pdb=false','--symlink=xyz','--iarg=1','--farg=x','--sarg=1','--parg=1'])
      else:
        nutils.cli.choose(main, args=['--outrootdir',self.outrootdir,'--pdb=false','--symlink=xyz','main','--iarg=1','--farg=x','--sarg=1','--parg=1'])
    except SystemExit as e:
      status = e
    else:
      status = None
    finally:
      sys.stdout, sys.stderr = _savestreams

    with self.subTest('outdir'):
      self.assertFalse(os.path.isdir(os.path.join(self.outrootdir,self.scriptname)), 'outdir directory found')

    with self.subTest('argparse'):
      output = stringio.getvalue()
      nutils.log.info( output )
      self.assertNotIn('all OK', output)

    with self.subTest('exitstatus'):
      self.assertIsNotNone(status)
      self.assertEquals(status.code, 2)

run(method='run')
run(method='choose')
