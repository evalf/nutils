from nutils import testing, unit, types, warnings
import stringly


class Unit(testing.TestCase):

    def setUp(self):
        super().setUp()
        self.U = unit.create(m=1, s=1, g=1e-3,
                             Pa='N/m2', N='kg*m/s2', lb='453.59237g', h='3600s', **{'in': '.0254m'})

    def check(self, *args, **powers):
        s, v = args
        u = self.U(s)
        U = type(u)
        self.assertEqual(u, v)
        self.assertEqual(self.U._parse(s).powers, powers)
        self.assertEqual(stringly.dumps(U, u), s)
        self.assertEqual(stringly.loads(U, s), u)

    def test_length(self):
        self.check('1m', 1, m=1)
        self.check('10in', .254, m=1)
        self.check('10000000000000000m', 1e16, m=1)  # str(1e16) has no decimal point

    def test_mass(self):
        self.check('1kg', 1, g=1)
        self.check('1lb', .45359237, g=1)

    def test_time(self):
        self.check('1s', 1, s=1)
        self.check('0.5h', 1800, s=1)

    def test_velocity(self):
        self.check('1m/s', 1, m=1, s=-1)
        self.check('1km/h', 1/3.6, m=1, s=-1)

    def test_force(self):
        self.check('1N', 1, g=1, m=1, s=-2)

    def test_pressure(self):
        self.check('1Pa', 1, g=1, m=-1, s=-2)

    def test_bind(self):
        T = self.U['m']
        self.assertEqual(T.__name__, 'unit:m')
        stringly.loads(T, '2in')
        with self.assertRaises(ValueError):
            stringly.loads(T, '2kg')

    def test_invalid(self):
        with self.assertRaises(ValueError):
            self.U('2foo')

    def test_loads_dumps(self):
        U = self.U['Pa*mm2']
        for s in '123456789Pa*mm2', '12.34Pa*mm2', '0Pa*mm2', '0.000012345Pa*mm2':
            v = stringly.loads(U, s)
            self.assertEqual(s, stringly.dumps(U, v))
        with self.assertRaises(ValueError):
            stringly.dumps(U, 'foo')


class DeprecatedUnit(testing.TestCase):

    def test_types_unit(self):
        with self.assertWarns(warnings.NutilsDeprecationWarning):
            U = types.unit(m=1, s=1, g=1e-3)
        self.assertIsInstance(U, unit._Unbound)

    def test_types_unit_create(self):
        with self.assertWarns(warnings.NutilsDeprecationWarning):
            U = types.unit.create(m=1, s=1, g=1e-3)
        self.assertIsInstance(U, unit._Unbound)

# vim:sw=2:sts=2:et
