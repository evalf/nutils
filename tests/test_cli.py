import tempfile
from nutils import cli, testing


def main(
        iarg: int = 1,
        farg: float = 1.,
        sarg: str = 'foo'):
    '''Dummy function to test argument parsing.'''

    assert isinstance(iarg, int), 'n should be int, got {}'.format(type(iarg))
    assert isinstance(farg, float), 'f should be float, got {}'.format(type(farg))
    assert isinstance(sarg, str), 'f should be str, got {}'.format(type(sarg))
    return f'received iarg={iarg} <{type(iarg).__name__}>, farg={farg} <{type(farg).__name__}>, sarg={sarg} <{type(sarg).__name__}>'


class method(testing.TestCase):

    def setUp(self):
        super().setUp()
        self.outrootdir = self.enter_context(tempfile.TemporaryDirectory())
        self.method = getattr(cli, self.__class__.__name__)

    def assertEndsWith(self, s, suffix):
        self.assertEqual(s[-len(suffix):], suffix)

    def _cli(self, *args, funcname='main'):
        argv = ['test.py', *args, 'pdb=no', 'outrootdir='+self.outrootdir]
        if self.method is cli.choose:
            argv.insert(1, funcname)
        try:
            return self.method(main, argv=argv)
        except SystemExit as e:
            return e.code

    def test_good(self):
        retval = self._cli('iarg=1', 'farg=1', 'sarg=1')
        self.assertEqual(retval, 'received iarg=1 <int>, farg=1.0 <float>, sarg=1 <str>')

    def test_badarg(self):
        retval = self._cli('bla')
        self.assertEndsWith(retval, "Error: invalid argument 'bla'")

    def test_badvalue(self):
        retval = self._cli('iarg=1', 'farg=x', 'sarg=1')
        self.assertEndsWith(retval, "Error: invalid value 'x' for farg: could not convert string to float: 'x'")

    def test_help(self):
        for arg in '-h', '--help':
            retval = self._cli(arg)
            self.assertEndsWith(retval, 'Dummy function to test argument parsing.')


class run(method):
    pass


class choose(method):

    def test_badchoice(self):
        retval = self._cli(funcname='bla')
        self.assertEqual(retval, 'USAGE: test.py main [...]')


del method # hide base class from unittest discovery
