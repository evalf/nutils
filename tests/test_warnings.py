from nutils import testing, warnings
import warnings as py_warnings


class via(testing.TestCase):

    def test(self):

        printed = []
        with warnings.via(printed.append):
            py_warnings.warn('via on')
        py_warnings.warn('via off')

        self.assertEqual(len(printed), 1)
        self.assertTrue(printed[0].startswith('UserWarning: via on\n  In'))
