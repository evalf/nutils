from nutils import SI

import numpy
import pickle
import typing
import unittest


class Dimension(unittest.TestCase):

    def test_multiply(self):
        self.assertEqual(SI.Velocity * SI.Time, SI.Length)

    def test_divide(self):
        self.assertEqual(SI.Length / SI.Time, SI.Velocity)

    def test_power(self):
        self.assertEqual(SI.Length**2, SI.Area)
        self.assertEqual(SI.Area**.5, SI.Length)

    def test_name(self):
        self.assertEqual(SI.Force.__name__, '[M*L/T2]')
        self.assertEqual((SI.Force**.5).__name__, '[M_2*L_2/T]')
        self.assertEqual((SI.Force**1.5).__name__, '[M3_2*L3_2/T3]')

    def test_fromname(self):
        self.assertEqual(getattr(SI.Quantity, '[M*L/T2]'), SI.Force)
        self.assertEqual(getattr(SI.Quantity, '[M_2*L_2/T]'), SI.Force**.5)
        self.assertEqual(getattr(SI.Quantity, '[M3_2*L3_2/T3]'), SI.Force**1.5)

    def test_typing(self):
        self.assertEqual(SI.Length | None, typing.Optional[SI.Length])
        self.assertEqual(None | SI.Length | SI.Time, typing.Optional[typing.Union[SI.Time, SI.Length]])

    def test_pickle(self):
        T = SI.Length / SI.Time
        s = pickle.dumps(T)
        self.assertEqual(pickle.loads(s), T)


class Quantity(unittest.TestCase):

    def test_fromstring(self):
        F = SI.parse('5kN')
        self.assertEqual(type(F), SI.Force)
        self.assertEqual(F / 'N', 5000)
        v = SI.parse('-864km/24h')
        self.assertEqual(type(v), SI.Velocity)
        self.assertEqual(v / 'm/s', -10)
        v = SI.parse('2m/5cm')
        self.assertEqual(v, 40)

    def test_fromvalue(self):
        F = SI.Force('10N')
        self.assertEqual(type(F), SI.Force)
        self.assertEqual(F / SI.Force('2N'), 5)

    def test_getitem(self):
        F = SI.units.N * numpy.arange(6).reshape(2, 3)
        self.assertEqual(F[0, 0], SI.Force('0N'))
        self.assertEqual(F[0, 1], SI.Force('1N'))
        self.assertEqual(F[0, 2], SI.Force('2N'))
        self.assertEqual(F[1, 0], SI.Force('3N'))
        self.assertEqual(F[1, 1], SI.Force('4N'))
        self.assertEqual(F[1, 2], SI.Force('5N'))

    def test_setitem(self):
        F = SI.units.N * numpy.zeros(3)
        F[0] = SI.Force('1N')
        F[1] = SI.Force('2N')
        with self.assertRaisesRegex(TypeError, r'cannot assign \[L2\] to \[M\*L/T2\]'):
            F[2] = SI.Area('10m2')
        F[2] = SI.Force('3N')
        self.assertTrue(numpy.all(F == SI.units.N * numpy.array([1, 2, 3])))

    def test_iter(self):
        F = SI.units.N * numpy.arange(6).reshape(2, 3)
        for i, Fi in enumerate(F):
            for j, Fij in enumerate(Fi):
                self.assertEqual(Fij, SI.units.N * (i*3+j))

    def test_multiply(self):
        self.assertEqual(SI.Mass('2kg') * SI.Acceleration('10m/s2'), SI.Force('20N'))
        self.assertEqual(2 * SI.Acceleration('10m/s2'), SI.Acceleration('20m/s2'))
        self.assertEqual(SI.Mass('2kg') * 10, SI.Mass('20kg'))
        self.assertEqual(SI.Time('2s') * SI.Frequency('10/s'), 20)
        self.assertEqual(numpy.multiply(SI.Mass('2kg'), SI.Acceleration('10m/s2')), SI.Force('20N'))

    def test_matmul(self):
        self.assertEqual((SI.units.kg * numpy.array([2, 3])) @ (SI.parse('m/s2') * numpy.array([5, -3])), SI.Force('1N'))

    def test_divide(self):
        self.assertEqual(SI.Length('2m') / SI.Time('10s'), SI.Velocity('.2m/s'))
        self.assertEqual(2 / SI.Time('10s'), SI.Frequency('.2/s'))
        self.assertEqual(SI.Length('2m') / 10, SI.Length('.2m'))
        self.assertEqual(SI.Density('2kg/m3') / SI.Density('10kg/m3'), .2)
        self.assertEqual(numpy.divide(SI.Length('2m'), SI.Time('10s')), SI.Velocity('.2m/s'))

    def test_power(self):
        self.assertEqual(SI.Length('3m')**2, SI.Area('9m2'))
        self.assertEqual(SI.Length('3m')**0, 1)
        self.assertEqual(numpy.power(SI.Length('3m'), 2), SI.Area('9m2'))

    def test_add(self):
        self.assertEqual(SI.Mass('2kg') + SI.Mass('3kg'), SI.Mass('5kg'))
        self.assertEqual(numpy.add(SI.Mass('2kg'), SI.Mass('3kg')), SI.Mass('5kg'))
        with self.assertRaisesRegex(TypeError, r'incompatible arguments for add: \[M\], \[L\]'):
            SI.Mass('2kg') + SI.Length('3m')

    def test_sub(self):
        self.assertEqual(SI.Mass('2kg') - SI.Mass('3kg'), SI.Mass('-1kg'))
        self.assertEqual(numpy.subtract(SI.Mass('2kg'), SI.Mass('3kg')), SI.Mass('-1kg'))
        with self.assertRaisesRegex(TypeError, r'incompatible arguments for sub: \[M\], \[L\]'):
            SI.Mass('2kg') - SI.Length('3m')

    def test_hypot(self):
        self.assertEqual(numpy.hypot(SI.Mass('3kg'), SI.Mass('4kg')), SI.Mass('5kg'))
        with self.assertRaisesRegex(TypeError, r'incompatible arguments for hypot: \[M\], \[L\]'):
            numpy.hypot(SI.Mass('3kg'), SI.Length('4m'))

    def test_neg(self):
        self.assertEqual(-SI.Mass('2kg'), SI.Mass('-2kg'))
        self.assertEqual(numpy.negative(SI.Mass('2kg')), SI.Mass('-2kg'))

    def test_pos(self):
        self.assertEqual(+SI.Mass('2kg'), SI.Mass('2kg'))
        self.assertEqual(numpy.positive(SI.Mass('2kg')), SI.Mass('2kg'))

    def test_abs(self):
        self.assertEqual(numpy.abs(SI.Mass('-2kg')), SI.Mass('2kg'))

    def test_sqrt(self):
        self.assertEqual(numpy.sqrt(SI.Area('4m2')), SI.Length('2m'))

    def test_sum(self):
        self.assertTrue(numpy.all(numpy.sum(SI.units.kg * numpy.arange(6).reshape(2, 3), 0) == SI.units.kg * numpy.array([3, 5, 7])))
        self.assertTrue(numpy.all(numpy.sum(SI.units.kg * numpy.arange(6).reshape(2, 3), 1) == SI.units.kg * numpy.array([3, 12])))

    def test_mean(self):
        self.assertTrue(numpy.all(numpy.mean(SI.units.kg * numpy.arange(6).reshape(2, 3), 0) == SI.units.kg * numpy.array([1.5, 2.5, 3.5])))
        self.assertTrue(numpy.all(numpy.mean(SI.units.kg * numpy.arange(6).reshape(2, 3), 1) == SI.units.kg * numpy.array([1, 4])))

    def test_broadcast_to(self):
        v = numpy.array([1, 2, 3])
        A = SI.units.kg * v
        B = numpy.broadcast_to(A, (2, 3))
        self.assertEqual(B.unwrap().shape, (2, 3))
        self.assertEqual(B[1, 1], SI.Mass('2kg'))

    def test_trace(self):
        A = SI.units.kg * numpy.arange(18).reshape(3, 2, 3)
        self.assertTrue(numpy.all(numpy.trace(A, axis1=0, axis2=2) == SI.units.kg * numpy.array([21, 30])))

    def test_ptp(self):
        A = SI.units.kg * numpy.array([2, -10, 5, 0])
        self.assertEqual(numpy.ptp(A), SI.Mass('15kg'))

    def test_min(self):
        A = SI.units.kg * numpy.array([2, -10, 5, 0])
        self.assertEqual(numpy.max(A), SI.Mass('5kg'))

    def test_max(self):
        A = SI.units.kg * numpy.array([2, -10, 5, 0])
        self.assertEqual(numpy.min(A), SI.Mass('-10kg'))

    def test_cmp(self):
        A = SI.Mass('2kg')
        B = SI.Mass('3kg')
        self.assertTrue(A < B)
        self.assertTrue(numpy.less(A, B))
        self.assertTrue(A <= B)
        self.assertTrue(numpy.less_equal(A, B))
        self.assertFalse(A > B)
        self.assertFalse(numpy.greater(A, B))
        self.assertFalse(A >= B)
        self.assertFalse(numpy.greater_equal(A, B))
        self.assertFalse(A == B)
        self.assertFalse(numpy.equal(A, B))
        self.assertTrue(A != B)
        self.assertTrue(numpy.not_equal(A, B))

    def test_shape(self):
        A = SI.Mass('2kg')
        self.assertEqual(numpy.shape(A), ())
        A = SI.units.kg * numpy.arange(3)
        self.assertEqual(numpy.shape(A), (3,))
        self.assertEqual(A.unwrap().shape, (3,))

    def test_ndim(self):
        A = SI.Mass('2kg')
        self.assertEqual(numpy.ndim(A), 0)
        A = SI.units.kg * numpy.arange(3)
        self.assertEqual(numpy.ndim(A), 1)
        self.assertEqual(A.unwrap().ndim, 1)

    def test_size(self):
        A = SI.Mass('2kg')
        self.assertEqual(numpy.size(A), 1)
        A = SI.units.kg * numpy.arange(3)
        self.assertEqual(numpy.size(A), 3)
        self.assertEqual(A.unwrap().size, 3)

    def test_isnan(self):
        self.assertTrue(numpy.isnan(SI.units.kg * float('nan')))
        self.assertFalse(numpy.isnan(SI.Mass('2kg')))

    def test_isfinite(self):
        self.assertFalse(numpy.isfinite(SI.units.kg * float('nan')))
        self.assertFalse(numpy.isfinite(SI.units.kg * float('inf')))
        self.assertTrue(numpy.isfinite(SI.Mass('2kg')))

    def test_stack(self):
        A = SI.Mass('2kg')
        B = SI.Mass('3kg')
        C = SI.Mass('4kg')
        D = SI.Time('5s')
        self.assertTrue(numpy.all(numpy.stack([A, B, C]) == SI.units.kg * numpy.array([2, 3, 4])))
        with self.assertRaisesRegex(TypeError, r'incompatible arguments for stack: \[M\], \[M\], \[M\], \[T\]'):
            numpy.stack([A, B, C, D])

    def test_concatenate(self):
        A = SI.units.kg * numpy.array([1, 2])
        B = SI.units.kg * numpy.array([3, 4])
        C = SI.units.s * numpy.array([5, 6])
        self.assertTrue(numpy.all(numpy.concatenate([A, B]) == SI.units.kg * numpy.array([1, 2, 3, 4])))
        with self.assertRaisesRegex(TypeError, r'incompatible arguments for concatenate: \[M\], \[M\], \[T\]'):
            numpy.concatenate([A, B, C])

    def test_format(self):
        s = 'velocity: {:.1m/s}'.format(SI.parse('9km/h'))
        self.assertEqual(s, 'velocity: 2.5m/s')

    def test_pickle(self):
        v = SI.Velocity('2m/s')
        s = pickle.dumps(v)
        self.assertEqual(pickle.loads(s), v)

    def test_string_representation(self):
        F = numpy.array([1.,2.]) * SI.units.N
        self.assertEqual(str(F), '[1. 2.][M*L/T2]')
        self.assertEqual(repr(F), 'array([1., 2.])[M*L/T2]')

    def test_wrap_unwrap(self):
        T = SI.Length / SI.Time
        v = T.wrap(5.)
        self.assertIsInstance(v, T)
        self.assertEqual(v.unwrap(), 5)

    def test_hash(self):
        v = SI.Velocity('2m/s')
        h = hash(v)
