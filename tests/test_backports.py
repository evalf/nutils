from nutils.testing import TestCase
import nutils._backports as bp


class cached_property(TestCase):

    def test(self):
        class A:
            def __init__(self):
                self.counter = 0
            @bp.cached_property
            def x(self):
                self.counter += 1
                return 'x'
        a = A()
        self.assertEqual(a.x, 'x')
        self.assertEqual(a.x, 'x')
        self.assertEqual(a.counter, 1)


class comb(TestCase):

    def test(self):

        self.assertEqual(bp.comb(0,0), 1)

        self.assertEqual(bp.comb(1,0), 1)
        self.assertEqual(bp.comb(1,1), 1)

        self.assertEqual(bp.comb(2,0), 1)
        self.assertEqual(bp.comb(2,1), 2)
        self.assertEqual(bp.comb(2,2), 1)

        self.assertEqual(bp.comb(9,4), 126)
