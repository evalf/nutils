import functools
import numpy
import operator
from nutils.testing import TestCase
from nutils import expression_v2, function, mesh, sample, SI


class SerializedOps:

    def from_int(self, v): return '{}i'.format(v)
    def from_float(self, v): return '{}f'.format(v)
    def scope(self, array): return 'scope({})'.format(array)
    def mean(self, array): return 'mean({})'.format(array)
    def jump(self, array): return 'jump({})'.format(array)
    def add(self, *args): return 'add({})'.format(', '.join(('neg({})' if negate else '{}').format(arg) for negate, arg in args))
    def multiply(self, *args): return 'mul({})'.format(', '.join(map(str, args)))
    def divide(self, numerator, denominator): return 'div({}, {})'.format(numerator, denominator)
    def power(self, base, exponent): return 'pow({}, {})'.format(base, exponent)
    def get_element(self, array, axis, item): return 'get({}, {}, {})'.format(array, axis, item)
    def transpose(self, array, axes): return 'transpose({})'.format(', '.join(map(str, (array, *axes))))
    def trace(self, array, axis1, axis2): return 'trace({}, {}, {})'.format(array, axis1, axis2)

    def get_variable(self, name, ndim):
        if name.startswith('a') and all('0' <= i <= '9' for i in name[1:]):
            if len(name) != ndim + 1:
                return expression_v2._InvalidDimension(len(name) - 1)
            else:
                return name, tuple(map(int, name[1:]))

    def call(self, name, ngenerates, arg):
        if not name.startswith('f') or not all('0' <= i <= '9' for i in name[1:]):
            return None
        gen_sh = tuple(map(int, name[1:]))
        if len(gen_sh) != ngenerates:
            return expression_v2._InvalidDimension(len(gen_sh))
        return 'call({}, {})'.format(name, str(arg)), gen_sh


class Parser(TestCase):

    def setUp(self):
        super().setUp()
        self.parser = expression_v2._Parser(SerializedOps())

    def mkasserts(self, parse):

        def assertParses(expression, desired_result, desired_indices, *desired_shape):
            with self.subTest('without-spaces'):
                s_expression = expression_v2._Substring(expression)
                actual_result, actual_shape, actual_indices, summed_indices = parse(s_expression)
                self.assertEqual(actual_result, desired_result)
                self.assertEqual(actual_shape, desired_shape)
                self.assertEqual(actual_indices, desired_indices)
            with self.subTest('with-spaces'):
                s_expression = expression_v2._Substring(' ' + expression + ' ')
                actual_result, actual_shape, actual_indices, summed_indices = parse(s_expression)
                self.assertEqual(actual_result, desired_result)
                self.assertEqual(actual_shape, desired_shape)
                self.assertEqual(actual_indices, desired_indices)

        def assertRaises(message, expression, markers, check_trim=True):
            s_expression = expression_v2._Substring(expression)
            with self.assertRaises(expression_v2.ExpressionSyntaxError) as cm:
                parse(s_expression)
            self.assertEqual(str(cm.exception), message + '\n' + expression + '\n' + markers)
            if check_trim:
                expression_spaces = ' ' + expression + ' '
                s_expression = expression_v2._Substring(expression_spaces)
                with self.assertRaises(expression_v2.ExpressionSyntaxError) as cm:
                    parse(s_expression)
                self.assertEqual(str(cm.exception), message + '\n' + expression_spaces + '\n ' + markers)

        return assertParses, assertRaises

    def test_parse_expression(self):
        assertParses, assertRaises = self.mkasserts(self.parser.parse_expression)
        assertParses('1', '1i', '')
        assertParses('-1 + 2', 'add(neg(1i), 2i)', '')
        assertParses('- 1    + a2_i a2_i +   2', 'add(neg(1i), trace(mul(a2, a2), 0, 1), 2i)', '')
        assertParses('a2_i + a23_ij a3_j + a2_i', 'add(a2, trace(mul(a23, a3), 1, 2), a2)', 'i', 2)
        assertParses('a2_i + a23_ij a3_j + a2_i', 'add(a2, trace(mul(a23, a3), 1, 2), a2)', 'i', 2)
        assertParses('a012_ijk + a021_ikj + a102_jik + a120_jki + a201_kij + a210_kji', 'add(a012, transpose(a021, 0, 2, 1), transpose(a102, 1, 0, 2), transpose(a120, 2, 0, 1), transpose(a201, 1, 2, 0), transpose(a210, 2, 1, 0))', 'ijk', 0, 1, 2)
        assertParses('-2^2', 'add(neg(pow(2i, 2i)))', '')  # https://en.wikipedia.org/wiki/Order_of_operations#Unary_minus_sign
        assertRaises(
            'Index i of the first term [^] is missing in the third term [~].',
            'a2_i + a2_i + 3 + a2_i',
            '^^^^          ~')
        assertRaises(
            'Index i of the second term [~] is missing in the first term [^].',
            '1 + a2_i + 3',
            '^   ~~~~')
        assertRaises(
            'Index i has length 2 in the first term [^] but length 3 in the fourth term [~].',
            'a23_ij + a23_ij + a23_ij + a32_ij',
            '^^^^^^                     ~~~~~~')

    def test_parse_fraction(self):
        assertParses, assertRaises = self.mkasserts(self.parser.parse_fraction)
        assertParses('1 / 2', 'div(1i, 2i)', '')
        assertParses('2 a2_i / 2 a2_j a2_j', 'div(mul(2i, a2), trace(mul(2i, a2, a2), 0, 1))', 'i', 2)
        assertRaises(
            'Repeated fractions are not allowed. Use parentheses if necessary.',
            '1 / 2 / 3',
            '^^^^^^^^^')
        assertRaises(
            'The denominator must have dimension zero.',
            '1 / a2_i',
            '    ^^^^')
        assertRaises(
            'Index i occurs more than twice.',
            'a2_i / a22_ii',
            '^^^^^^^^^^^^^')
        assertRaises(
            'Index i occurs more than twice.',
            'a22_ii / a22_ii',
            '^^^^^^^^^^^^^^^')

    def test_parse_term(self):
        assertParses, assertRaises = self.mkasserts(self.parser.parse_term)
        assertParses('1 a2_i a2_j', 'mul(1i, a2, a2)', 'ij', 2, 2)
        assertParses('a2_i a23_ij a3_j', 'trace(trace(mul(a2, a23, a3), 0, 1), 0, 1)', '')
        assertParses('a2_i a3_j a3_j', 'trace(mul(a2, a3, a3), 1, 2)', 'i', 2)
        assertRaises(
            'Numbers are only allowed at the start of a term.',
            '1 1',
            '  ^')
        assertRaises(
            'Index i is assigned to axes with different lengths: 2 and 3.',
            '1 a2_i a3_i a',
            '^^^^^^^^^^^^^')
        assertRaises(
            'Index i occurs more than twice.',
            '1 a22_ii a2_i a',
            '^^^^^^^^^^^^^^^')
        assertRaises(
            'Index i occurs more than twice.',
            '1 a22_ii a22_ii a',
            '^^^^^^^^^^^^^^^^^')

    def test_parse_power_number(self):
        assertParses, assertRaises = self.mkasserts(lambda s: self.parser.parse_power(s, allow_number=True))
        assertParses('1^2', 'pow(1i, 2i)', '')
        assertParses('1^-2', 'pow(1i, -2i)', '')
        assertParses('a2_i^2', 'pow(a2, 2i)', 'i', 2)
        assertRaises(
            'The exponent must have dimension zero.',
            'a^(a2_i)',
            '  ^^^^^^')
        assertRaises(
            'Index i occurs more than twice.',
            'a2_i^(a22_ii)',
            '^^^^^^^^^^^^^')
        assertRaises(
            'Index i occurs more than twice.',
            'a2_i^(a22_ii)',
            '^^^^^^^^^^^^^')
        assertRaises(
            'Unexpected whitespace before `^`.',
            'a ^2',
            ' ^')
        assertRaises(
            'Unexpected whitespace after `^`.',
            'a^ 2',
            '  ^')
        assertRaises(
            'Expected a number, variable, scope, mean, jump or function call.',
            '^2',
            '^')

    def test_parse_power_nonumber(self):
        assertParses, assertRaises = self.mkasserts(lambda s: self.parser.parse_power(s, allow_number=False))
        assertParses('a2_i^2', 'pow(a2, 2i)', 'i', 2)
        assertParses('a23_ij^-2', 'pow(a23, -2i)', 'ij', 2, 3)
        assertRaises(
            'The exponent must have dimension zero.',
            'a^(a2_i)',
            '  ^^^^^^')
        assertRaises(
            'Unexpected whitespace before `^`.',
            'a ^2',
            ' ^')
        assertRaises(
            'Unexpected whitespace after `^`.',
            'a^ 2',
            '  ^')
        assertRaises(
            'Expected a variable, scope, mean, jump or function call.',
            '^2',
            '^')
        assertRaises(
            'Expected an int.',
            'a^2_i',
            '  ^^^')
        assertRaises(
            'Expected an int or scoped expression.',
            'a^',
            '  ^')
        assertRaises(
            'Expected an int or scoped expression.',
            'a^a2_i',
            '  ^^^^')
        assertRaises(
            'Repeated powers are not allowed. Use parentheses if necessary.',
            'a^a^a',
            '^^^^^')

    def test_parse_variable(self):
        assertParses, assertRaises = self.mkasserts(lambda s: self.parser.parse_item(s, allow_number=False))
        assertParses('a22_ij', 'a22', 'ij', 2, 2)
        assertParses('a222_iji', 'trace(a222, 0, 2)', 'j', 2)
        assertParses('a2_0', 'get(a2, 0, 0)', '')
        assertParses('a23_1i', 'get(a23, 0, 1)', 'i', 3)
        assertRaises(
            'No such variable: `unknown`.',
            'unknown_i',
            '^^^^^^^')
        assertRaises(
            'Expected 1 index for variable `a2` but got 2.',
            'a2_ij',
            '^^^^^')
        assertRaises(
            'Expected 2 indices for variable `a22` but got 1.',
            'a22_i',
            '^^^^^')
        assertRaises(
            'Index i occurs more than twice.',
            'a222_iii',
            '^^^^^^^^')
        assertRaises(
            'Index of axis with length 2 out of range.',
            'a23_3i',
            '    ^')
        assertRaises(
            'Symbol `$` is not allowed as index.',
            'a234_i$j',
            '      ^')

    def test_parse_call(self):
        assertParses, assertRaises = self.mkasserts(lambda s: self.parser.parse_item(s, allow_number=False))
        assertParses('f(a2_i + a2_i)', 'call(f, add(a2, a2))', 'i', 2)
        assertParses('f(a2_i (a3_j + a3_j))', 'call(f, mul(a2, scope(add(a3, a3))))', 'ij', 2, 3)
        assertParses('f62_mi(a256_ilm)', 'trace(trace(call(f62, a256), 2, 3), 0, 2)', 'l', 5)
        assertParses('f42_ij(a34_ki)', 'trace(call(f42, a34), 1, 2)', 'kj', 3, 2)
        assertParses('f32_ij(a34_ik)', 'trace(call(f32, a34), 0, 2)', 'kj', 4, 2)
        assertParses('f23_i0(a2_k)', 'get(call(f23, a2), 2, 0)', 'ki', 2, 2)
        assertParses('f23_1j(a2_k)', 'get(call(f23, a2), 1, 1)', 'kj', 2, 3)
        assertRaises(
            'Expected a number, variable, scope, mean, jump or function call.',
            'f()',
            '  ^')
        assertRaises(
            'No such function: `g`.',
            'g(a)',
            '^')
        assertRaises(
            'Index i occurs more than twice.',
            'f2_i(a22_ii)',
            '^^^^^^^^^^^^')
        assertRaises(
            'Index i occurs more than twice.',
            'f22_ii(a2_i)',
            '^^^^^^^^^^^^')
        assertRaises(
            'Index i occurs more than twice.',
            'f22_ii(a22_ii)',
            '^^^^^^^^^^^^^^')
        assertRaises(
            'Index of axis with length 2 out of range.',
            'f2_2(a)',
            '   ^')
        assertRaises(
            'Expected 2 indices for axes generated by function `f23` but got 1.',
            'f23_j(a4_i)',
            '^^^^^^^^^^^')
        assertRaises(
            'Symbol `$` is not allowed as index.',
            'f234_i$j(a)',
            '      ^')

    def test_parse_item_number(self):
        assertParses, assertRaises = self.mkasserts(lambda s: self.parser.parse_item(s, allow_number=True))
        assertRaises(
            'Expected a number, variable, scope, mean, jump or function call.',
            '   ',
            '^^^', check_trim=False)
        assertRaises(
            'Expected a number, variable, scope, mean, jump or function call.',
            '1a',
            '^^')
        assertRaises(
            'Expected a number, variable, scope, mean, jump or function call. '
            'Hint: the operators `+`, `-` and `/` must be surrounded by spaces.',
            '1+a',
            '^^^')
        assertRaises(
            'Expected a number, variable, scope, mean, jump or function call. '
            'Hint: the operators `+`, `-` and `/` must be surrounded by spaces.',
            '1-a',
            '^^^')
        assertRaises(
            'Expected a number, variable, scope, mean, jump or function call. '
            'Hint: the operators `+`, `-` and `/` must be surrounded by spaces.',
            '1/a',
            '^^^')

    def test_parse_item_nonumber(self):
        assertParses, assertRaises = self.mkasserts(lambda s: self.parser.parse_item(s, allow_number=False))
        assertRaises(
            'Expected a variable, scope, mean, jump or function call.',
            '   ',
            '^^^', check_trim=False)
        assertRaises(
            'Numbers are only allowed at the start of a term.',
            '1',
            '^')
        assertRaises(
            'Numbers are only allowed at the start of a term.',
            '1a',
            '^^')
        assertRaises(
            'Expected a variable, scope, mean, jump or function call.',
            'f[a]',
            '^^^^')
        assertRaises(
            'Expected a variable, scope, mean, jump or function call.',
            'f{a}',
            '^^^^')
        assertRaises(
            'Expected a variable, scope, mean, jump or function call.',
            'f<a>',
            '^^^^')
        assertRaises(
            'Expected a variable, scope, mean, jump or function call.',
            '<a>',
            '^^^')

    def test_parse_scope(self):
        assertParses, assertRaises = self.mkasserts(lambda s: self.parser.parse_item(s, allow_number=False))
        assertParses('(1)', 'scope(1i)', '')
        assertParses('(1 + a)', 'scope(add(1i, a))', '')
        assertRaises(
            'Unclosed `(`.',
            '(1',
            '^ ~', check_trim=False)
        assertRaises(
            'Parenthesis `(` closed by `]`.',
            '(1]',
            '^ ~')
        assertRaises(
            'Parenthesis `(` closed by `]`.',
            '(1])',
            '^ ~')
        assertRaises(
            'Unexpected symbols after scope.',
            '(1)spam',
            '   ^^^^')

    def test_parse_mean(self):
        assertParses, assertRaises = self.mkasserts(lambda s: self.parser.parse_item(s, allow_number=False))
        assertParses('{1 + 2}', 'mean(add(1i, 2i))', '')
        assertParses('{(a2_i)}', 'mean(scope(a2))', 'i', 2)

    def test_parse_jump(self):
        assertParses, assertRaises = self.mkasserts(lambda s: self.parser.parse_item(s, allow_number=False))
        assertParses('[1 + 2]', 'jump(add(1i, 2i))', '')
        assertParses('[(a2_i)]', 'jump(scope(a2))', 'i', 2)

    def test_parse_signed_int(self):
        assertParses, assertRaises = self.mkasserts(self.parser.parse_signed_int)
        assertParses('1', '1i', '')
        assertParses('-1', '-1i', '')
        assertRaises(
            'Expected an int.',
            '',
            '^', check_trim=False)
        assertRaises(
            'Expected an int.',
            '   ',
            '^^^', check_trim=False)
        assertRaises(
            'Expected an int.',
            'a',
            '^')

    def test_parse_unsigned_int(self):
        assertParses, assertRaises = self.mkasserts(self.parser.parse_unsigned_int)
        assertParses('1', '1i', '')
        assertParses('2', '2i', '')
        assertParses('34', '34i', '')
        assertRaises(
            'Expected an int.',
            '',
            '^', check_trim=False)
        assertRaises(
            'Expected an int.',
            '   ',
            '^^^', check_trim=False)
        assertRaises(
            'Expected an int.',
            'a',
            '^')
        assertRaises(
            'Expected an int.',
            '-1',
            '^^')

    def test_parse_unsigned_float(self):
        assertParses, assertRaises = self.mkasserts(self.parser.parse_unsigned_float)
        assertParses('1', '1.0f', '')
        assertParses('1.0', '1.0f', '')
        assertParses('1.', '1.0f', '')
        assertParses('0.1', '0.1f', '')
        assertParses('1e-1', '0.1f', '')
        assertParses('1.0e-1', '0.1f', '')
        assertParses('.1e-1', '0.01f', '')
        assertRaises(
            'Expected a float.',
            '',
            '^', check_trim=False)
        assertRaises(
            'Expected a float.',
            '   ',
            '^^^', check_trim=False)
        assertRaises(
            'Expected a float.',
            'a',
            '^')
        assertRaises(
            'Expected a float.',
            '-1.2',
            '^^^^')


class Namespace(TestCase):

    def test_set_int(self):
        ns = expression_v2.Namespace()
        ns.a = 1

    def test_set_float(self):
        ns = expression_v2.Namespace()
        ns.a = 1.2

    def test_set_complex(self):
        ns = expression_v2.Namespace()
        ns.a = 1+1j

    def test_set_numpy(self):
        ns = expression_v2.Namespace()
        ns.a = numpy.array([1, 2])

    def test_set_numpy_indices(self):
        ns = expression_v2.Namespace()
        a = numpy.array([1, 2])
        with self.assertRaisesRegex(AttributeError, '^Cannot assign an array to an attribute with an underscore.'):
            ns.a_i = a

    def test_set_expression_0d(self):
        ns = expression_v2.Namespace()
        ns.a = '1.2'

    def test_set_expression_1d(self):
        ns = expression_v2.Namespace()
        ns.a = numpy.array([1, 2])
        ns.b_i = '2 a_i'

    def test_set_expression_2d(self):
        ns = expression_v2.Namespace()
        ns.a = numpy.array([1, 2])
        ns.b_ij = 'a_i a_j'
        ns.c_ji = 'a_i a_j'

    def test_set_expression_invalid_indices(self):
        ns = expression_v2.Namespace()
        ns.a = numpy.array([1, 2])
        with self.assertRaisesRegex(AttributeError, '^Only lower case latin characters are allowed as indices.'):
            ns.b_α = 'a_α'

    def test_set_expression_duplicate_indices(self):
        ns = expression_v2.Namespace()
        ns.a = numpy.array([1, 2])
        with self.assertRaisesRegex(AttributeError, '^All indices must be unique.'):
            ns.b_ii = 'a_i a_i'

    def test_set_expression_missing_indices(self):
        ns = expression_v2.Namespace()
        ns.a = numpy.array([1, 2])
        with self.assertRaisesRegex(AttributeError, '^Index i of the expression is missing in the namespace attribute.'):
            ns.b_j = 'a_i a_j'
        with self.assertRaisesRegex(AttributeError, '^Index j of the namespace attribute is missing in the expression.'):
            ns.b_ij = 'a_i'

    def test_set_function(self):
        ns = expression_v2.Namespace()
        ns.f = lambda a: a**2
        self.assertAlmostEqual(('f(2)' @ ns).eval(), 4)

    def test_set_function_unexpected_indices(self):
        ns = expression_v2.Namespace()
        with self.assertRaisesRegex(AttributeError, '^Cannot assign a function to an attribute with an underscore.'):
            ns.f_i = lambda a: numpy.stack([a, a], axis=-1)

    def test_set_other(self):
        ns = expression_v2.Namespace()
        with self.assertRaisesRegex(AttributeError, '^Cannot assign an object of type'):
            ns.a = object()

    def test_eval(self):
        ns = expression_v2.Namespace()
        ns.a = numpy.array([1, 2])
        ns.b = numpy.array([1, 2, 3])
        self.assertEqual(('1' @ ns).eval().tolist(), 1)
        self.assertEqual(('a_i' @ ns).eval().tolist(), [1, 2])
        self.assertEqual(('a_i b_j' @ ns).eval().export('dense').tolist(), [[1, 2, 3], [2, 4, 6]])
        self.assertEqual(('b_j a_i' @ ns).eval().export('dense').tolist(), [[1, 2, 3], [2, 4, 6]])

    def test_eval_tuple_list(self):
        ns = expression_v2.Namespace()
        self.assertEqual(function.eval(('1', '2', '3') @ ns), (1, 2, 3))
        self.assertEqual(function.eval(['1', '2', '3'] @ ns), (1, 2, 3))

    def test_define_for_0d(self):
        ns = expression_v2.Namespace()
        topo, ns.t = mesh.line(numpy.linspace(0, 1, 3), bnames=('past', 'future'))
        ns.define_for('t', gradient='∂t', normal='nt', jacobians=['dt', 'dtb'])
        ns.basis = topo.basis('spline', degree=1)
        self.assertAlmostEqual(topo.integral('dt' @ ns, degree=2).eval(), 1)
        self.assertAlmostEqual(topo.integral('∂t(t^2) dt' @ ns, degree=2).eval(), 1)
        self.assertAlmostEqual(topo.boundary['future'].integral('nt dtb' @ ns, degree=2).eval(), 1)
        self.assertAlmostEqual(topo.boundary['past'].integral('nt dtb' @ ns, degree=2).eval(), -1)
        with self.assertRaisesRegex(ValueError, 'The curl can only be defined for a geometry with shape'):
            ns.define_for('t', curl='curl')

    def test_define_for_1d(self):
        ns = expression_v2.Namespace()
        topo, ns.x = mesh.rectilinear([numpy.linspace(0, 1, 3), numpy.linspace(0, 1, 5)])
        ns.define_for('x', gradient='∇', normal='n', jacobians=['dV', 'dS'])
        self.assertAlmostEqual(topo.integral('dV' @ ns, degree=2).eval(), 1)
        self.assertAlmostEqual(topo.integral('∇_i(x_i^2) dV' @ ns, degree=2).eval(), 2)
        self.assertAlmostEqual(topo.boundary['right'].integral('n_0 dS' @ ns, degree=2).eval(), 1)
        self.assertAlmostEqual(topo.boundary['bottom'].integral('n_1 dS' @ ns, degree=2).eval(), -1)
        with self.assertRaisesRegex(ValueError, 'The curl can only be defined for a geometry with shape'):
            ns.define_for('x', curl='curl')

    def test_define_for_3d(self):
        ns = expression_v2.Namespace()
        topo, ns.X = mesh.rectilinear([numpy.linspace(-1, 1, 3)]*3)
        ns.x, ns.y, ns.z = ns.X
        ns.define_for('X', gradient='∇', curl='curl')
        ns.δ = function.eye(3)
        ns.ε = function.levicivita(3)
        ns.f = function.Array.cast([['x', '-z', 'y'], ['0', 'x z', '0']] @ ns)
        smpl = topo.sample('gauss', 5)
        assertEvalAlmostEqual = lambda *args: self.assertAllAlmostEqual(*(smpl(f).as_evaluable_array.simplified.eval() for f in args))
        assertEvalAlmostEqual('curl_ij(y δ_j0 - x δ_j1 + z δ_j2)' @ ns, '-2 δ_i2' @ ns)
        assertEvalAlmostEqual('curl_ij(-x^2 δ_j1)' @ ns, '-2 x δ_i2' @ ns)
        assertEvalAlmostEqual('curl_ij((x δ_j0 - z δ_j1 + y δ_j2) δ_k0 + x z δ_j1 δ_k1)' @ ns, '2 δ_i0 δ_k0 - x δ_i0 δ_k1 + z δ_i2 δ_k1' @ ns)
        assertEvalAlmostEqual('curl_ij(∇_j(x y + z))' @ ns, function.zeros((3,)))

    def test_add_single_field(self):
        ns = expression_v2.Namespace()
        ns.add_field('u', numpy.array([1,2,3]))
        self.assertEquals(ns.u.argshapes, dict(u=(3,)))
        self.assertEquals(ns.u.shape, ())

    def test_add_multiple_fields(self):
        ns = expression_v2.Namespace()
        ns.add_field(('u', 'v'), numpy.array([1,2,3]))
        self.assertEquals(ns.u.argshapes, dict(u=(3,)))
        self.assertEquals(ns.u.shape, ())
        self.assertEquals(ns.v.argshapes, dict(v=(3,)))
        self.assertEquals(ns.v.shape, ())

    def test_add_single_field_multiple_bases(self):
        ns = expression_v2.Namespace()
        ns.add_field('u', numpy.array([1,2,3]), numpy.array([4,5,6,7]))
        self.assertEquals(ns.u.argshapes, dict(u=(3,4)))
        self.assertEquals(ns.u.shape, ())

    def test_add_single_field_with_shape(self):
        ns = expression_v2.Namespace()
        ns.add_field('u', numpy.array([1,2,3]), shape=(2,))
        self.assertEquals(ns.u.argshapes, dict(u=(3,2)))
        self.assertEquals(ns.u.shape, (2,))

    def test_copy(self):
        ns1 = expression_v2.Namespace()
        ns1.a = 1
        ns2 = ns1.copy_()
        self.assertAlmostEqual(ns2.a.eval(), 1)

    def test_copy_replace_array(self):
        ns1 = expression_v2.Namespace()
        ns1.a = function.Argument('x', ())
        ns2 = ns1.copy_(x=1.)
        self.assertAlmostEqual(ns2.a.eval(), 1)

    def test_copy_replace_SI_array(self):
        ns1 = expression_v2.Namespace()
        ns1.a = function.Argument('x', ()) * SI.Length('1m')
        ns2 = ns1.copy_(x=1.)
        self.assertIsInstance(ns2.a, SI.Length)
        self.assertAlmostEqual((ns2.a / 'm').eval(), 1)
