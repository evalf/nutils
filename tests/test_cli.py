import tempfile
from nutils import cli, testing


class method(testing.TestCase):

    def setUp(self):
        super().setUp()
        self.outrootdir = self.enter_context(tempfile.TemporaryDirectory())

    def main(
            self,
            iarg: int = 1,
            farg: float = 1.,
            sarg: str = 'foo'):
        '''Dummy function to test argument parsing.'''

        self.assertIsInstance(iarg, int)
        self.assertIsInstance(farg, float)
        self.assertIsInstance(sarg, str)

    def test_good(self):
        self._cli('iarg=1', 'farg=1', 'sarg=1')

    def test_badarg(self):
        with self.assertRaisesRegex(SystemExit, "Error: invalid argument 'bla'"):
            self._cli('iarg=1', 'bla')

    def test_badvalue(self):
        with self.assertRaisesRegex(SystemExit, "Error: invalid value 'x' for farg: could not convert string to float: 'x'"):
            self._cli('iarg=1', 'farg=x', 'sarg=1')

    def test_help(self):
        for arg in '-h', '--help':
            with self.assertRaisesRegex(SystemExit, self.usage):
                self._cli(arg)


class run(method):

    def _cli(self, *args):
        argv = ['test.py', *args, 'pdb=no', 'outrootdir='+self.outrootdir]
        return cli.run(self.main, argv=argv)

    usage = '''\
USAGE: test.py \[iarg=I\] \[farg=F\] \[sarg=S\] \[pdb=P\] \[gracefulexit=G\]
  \[outrootdir=O\] \[outrooturi=O\] \[scriptname=S\] \[outdir=O\] \[outuri=O\]
  \[richoutput=R\] \[verbose=V\] \[matrix=M\] \[nprocs=N\] \[cache=C\]
  \[cachedir=C\]

Dummy function to test argument parsing.'''


class choose(method):

    def other(*_):
        self.fail("wrong function")

    def _cli(self, *args, funcname='main'):
        argv = ['test.py', funcname, *args, 'pdb=no', 'outrootdir='+self.outrootdir]
        return cli.choose(self.main, self.other, argv=argv)

    def test_badchoice(self):
        with self.assertRaisesRegex(SystemExit, r'USAGE: test.py main|other \[...\]'):
            self._cli(funcname='bla')

    usage = '''\
USAGE: test.py main \[iarg=I\] \[farg=F\] \[sarg=S\] \[pdb=P\]
  \[gracefulexit=G\] \[outrootdir=O\] \[outrooturi=O\] \[scriptname=S\]
  \[outdir=O\] \[outuri=O\] \[richoutput=R\] \[verbose=V\] \[matrix=M\]
  \[nprocs=N\] \[cache=C\] \[cachedir=C\]

Dummy function to test argument parsing.'''


del method # hide base class from unittest discovery
