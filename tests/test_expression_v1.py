from nutils import evaluable, expression_v1, function, mesh, warnings
from nutils.testing import TestCase
import pickle
import numpy

_ = lambda arg: (None, arg)


class Array:
    def __init__(self, text, shape):
        self.text = text
        self.shape = tuple(shape)
        self.ndim = len(self.shape)

    def __len__(self):
        return self.shape[0]

    def __str__(self):
        return self.text

    def __repr__(self):
        return self.text

    def __eq__(self, other):
        return type(self) == type(other) and self.text == other.text

    def __hash__(self):
        return hash(self.text)


class Variables:
    def __init__(self, x, altgeom, funcoverride):
        self.x = x
        self.altgeom = altgeom
        self.funcoverride = funcoverride
        self._lengths = {str(i): i for i in range(10)}

    def __getitem__(self, name):
        if not name.startswith('_'):
            try:
                return getattr(self, name)
            except AttributeError:
                pass
        raise KeyError(name)

    def __contains__(self, name):
        try:
            self[name]
            return True
        except KeyError:
            return False

    def __getattr__(self, name):
        if name.startswith('_'):
            return _(getattr(self, name[1:]))
        elif name.startswith('a'):
            return Array(name, tuple(self._lengths.get(i, expression_v1._Length(ord(i))) for i in name[1:]))
        else:
            raise AttributeError(name)

    def get(self, name, default):
        try:
            return self[name]
        except KeyError:
            return default


v = Variables(x=Array('x', [2]), altgeom=Array('altgeom', [3]), funcoverride=Array('funcoverride', []))


class parse(TestCase):

    def assert_ast(self, expression, indices, ast, variables=None, **parse_kwargs):
        if variables is None:
            variables = v
        self.assertEqual(expression_v1.parse(expression, variables, indices, **parse_kwargs)[0], ast)

    def assert_syntax_error(self, msg, expression, indices, highlight, arg_shapes={}, fixed_lengths=None, exccls=expression_v1.ExpressionSyntaxError):
        with self.assertRaises(exccls) as cm:
            expression_v1.parse(expression, v, indices, arg_shapes, fixed_lengths=fixed_lengths)
        self.assertEqual(str(cm.exception), msg + '\n' + expression + '\n' + highlight)

    # OTHER

    def test_no_indices_0(self): self.assert_ast('a', None, v._a)
    def test_no_indices_1(self): self.assert_ast('a2_i', None, v._a2)

    def test_ambiguous_alignment(self):
        self.assert_syntax_error(
            "Cannot unambiguously align the array because the array has more than one dimension.",
            "a23_ij", None,
            "^^^^^^",
            exccls=expression_v1.AmbiguousAlignmentError)

    def test_mul_2(self):
        self.assert_ast('a2_i a3_j', 'ij',
                        ('mul',
                         ('append_axis', v._a2, _(3)),
                            ('transpose',
                             ('append_axis', v._a3, _(2)),
                             _((1, 0)))))

    def test_mul_add_sub(self):
        self.assert_ast('1_j a2_i + 1_i a3_j - a23_ij', 'ij',
                        ('transpose',
                         ('sub',
                          ('add',
                           ('mul',
                            ('append_axis', ('append_axis', _(1), _(3)), _(2)),
                               ('transpose', ('append_axis', v._a2, _(3)), _((1, 0)))),
                              ('transpose',
                               ('mul',
                                ('append_axis', ('append_axis', _(1), _(2)), _(3)),
                                   ('transpose', ('append_axis', v._a3, _(2)), _((1, 0)))),
                               _((1, 0)))),
                             ('transpose', v._a23, _((1, 0)))),
                            _((1, 0))))

    def test_mul_reduce_2(self):
        self.assert_ast('a2_i a23_ij a3_j', '',
                        ('sum',
                         ('mul',
                          ('sum',
                           ('mul',
                            ('append_axis', v._a2, _(3)),
                               v._a23),
                              _(0)),
                             v._a3),
                            _(0)))

    def test_dupl_indices_1(self):
        self.assert_syntax_error(
            "Index 'i' occurs more than twice.",
            "a3_j a2_i,ii a2_k", "i",
            "     ^^^^^^^")

    def test_missing_indices_1(self):
        self.assert_syntax_error(
            "Expected 1 index, got 2.",
            "a2_i a3_ij a2_j", "",
            "     ^^^^^")

    def test_missing_indices_2(self):
        self.assert_syntax_error(
            "Expected 2 indices, got 0.",
            "a2_i a23 a2_j", "ij",
            "     ^^^")

    def test_wrap_array_trace(self): self.assert_ast('a222_ijj', 'i', ('trace', v._a222, _(1), _(2)))
    def test_div_const_scalar(self): self.assert_ast('a2_i / 1', 'i', ('truediv', v._a2, _(1)))
    def test_div_scalar_variable(self): self.assert_ast('a2_i / a', 'i', ('truediv', v._a2, v._a))
    def test_div_scalar_sum(self): self.assert_ast('a2_i / 2 a3_j a3_j', 'i', ('truediv', v._a2, ('sum', ('mul', ('mul', ('append_axis', _(2), _(3)), v._a3), v._a3), _(0))))

    def test_array_denominator(self):
        self.assert_syntax_error(
            "A denominator must have dimension 0.",
            "a2_i / a3_j", "ij",
            "       ^^^^")

    def test_duplicate_indices_numerator_denominator(self):
        self.assert_syntax_error(
            "Index 'i' occurs more than twice.",
            "1 + a2_i a2_i / a3_i a3_i", "",
            "    ^^^^^^^^^^^^^^^^^^^^^")

    def test_duplicate_indices_2(self):
        self.assert_syntax_error(
            "Index 'i' occurs more than twice.",
            "a2_i (a3_i a23_ji)", "ij",
            "^^^^^^^^^^^^^^^^^^")

    def test_duplicate_indices_3(self):
        self.assert_syntax_error(
            "Index 'i' occurs more than twice.",
            "a2_i (a3_i a23_ji)", "ij",
            "^^^^^^^^^^^^^^^^^^")

    def test_duplicate_indices_4(self):
        self.assert_syntax_error(
            "Index 'i' occurs more than twice.",
            "a222_iii", "",
            "^^^^^^^^")

    def test_leading_zeros_int(self):
        self.assert_syntax_error(
            "Leading zeros are forbidden.",
            "1 + 01", "",
            "    ^^")

    def test_leading_zeros_float(self):
        self.assert_syntax_error(
            "Leading zeros are forbidden.",
            "1 + 01.0", "",
            "    ^^^^")

    def test_missing_indices_3(self):
        self.assert_syntax_error(
            "Missing indices.",
            "a22_ij + a2_", "ij",
            "            ^")

    def test_missing_gradient_indices_1(self):
        self.assert_syntax_error(
            "Missing indices.",
            "a22_ij + a2_,", "ij",
            "            ^")

    def test_missing_whitespace_add_right(self):
        self.assert_syntax_error(
            "Missing whitespace.",
            "a2_i +a2_i", "i",
            "      ^")

    def test_missing_whitespace_add_left(self):
        self.assert_syntax_error(
            "Missing whitespace.",
            "a2_i+ a2_i", "i",
            "    ^")

    def test_missing_whitespace_sub_right(self):
        self.assert_syntax_error(
            "Missing whitespace.",
            "a2_i -a2_i", "i",
            "      ^")

    def test_missing_whitespace_sub_left(self):
        self.assert_syntax_error(
            "Missing whitespace.",
            "a2_i- a2_i", "i",
            "    ^")

    def test_int(self):
        self.assert_ast('1', '', _(1))

    def test_float(self):
        for f in '10', '1', '1.', '.1', '1.2', '0.01', '10.0':
            self.assert_ast(f, '', _(float(f)))

    def test_scientific(self):
        for base in '0', '1', '10', '1.', '.1', '.01', '1.2':
            for exp in '-1', '0', '1', '10':
                self.assert_ast(base+'e'+exp, '', _(float(base+'e'+exp)))

    def test_jump_mean(self):
        self.assert_ast('[a2_i,i] + {a2_j,j}', '',
                        ('add',
                         ('jump', ('trace', ('grad', v._a2, v._x), _(0), _(1))),
                            ('mean', ('trace', ('grad', v._a2, v._x), _(0), _(1)))))

    def test_jump_normal(self):
        with self.assertWarns(warnings.NutilsDeprecationWarning):
            self.assert_ast('[a]_i', 'i', ('mul', ('append_axis', ('jump', v._a), _(2)), ('normal', v._x)))

    def test_jump_normal_altgeom(self):
        with self.assertWarns(warnings.NutilsDeprecationWarning):
            self.assert_ast('[a]_altgeom_i', 'i', ('mul', ('append_axis', ('jump', v._a), _(3)), ('normal', v._altgeom)))

    def test_laplace_of_group(self):
        self.assert_ast('(2 a2_i)_,jj', 'i',
                        ('trace',
                         ('grad',
                          ('grad',
                           ('group',
                            ('mul',
                             ('append_axis', _(2), _(2)),
                                v._a2)),
                              v._x),
                             v._x),
                            _(1), _(2)))

    def test_indices_on_group(self):
        self.assert_syntax_error(
            "Indices can only be specified for variables, e.g. 'a_ij', not for groups, e.g. '(a+b)_ij'.",
            "1 + (a2_i)_j + 1", "ij",
            "           ^")

    def test_unknown_symbol(self):
        self.assert_syntax_error(
            "Unknown symbol: '#'.",
            "1 + # + 1", "",
            "    ^")

    def test_invalid_group_end_partial_expression(self):
        self.assert_syntax_error(
            "Expected a variable, group or function call.",
            "1 + (2 + )", "",
            "         ^")

    def test_invalid_group_end_wrong_bracket_no_whitespace(self):
        self.assert_syntax_error(
            "Expected ')'.",
            "1 + (2 + 3] + 4", "",
            "          ^")

    def test_invalid_group_end_wrong_bracket_whitespace(self):
        self.assert_syntax_error(
            "Expected ')'.",
            "1 + (2 + 3 ] + 4", "",
            "           ^")

    def test_invalid_group_end_eof(self):
        self.assert_syntax_error(
            "Expected ')'.",
            "1 + (2 + 3", "",
            "          ^")

    def test_expected_EOF(self):
        self.assert_syntax_error(
            "Unexpected symbol at end of expression.",
            "1 ) 1", "",
            "  ^")

    def test_shape_mismatch(self):
        self.assert_syntax_error(
            "Shapes at index 'i' differ: 2, 4.",
            "1_j + a234_iji + 1_j", "j",
            "      ^^^^^^^^")

    def test_unknown_variable(self):
        self.assert_syntax_error(
            "Unknown variable: 'b'.",
            "1 + b + 1", "",
            "    ^")

    def test_const_numeric_indices(self):
        self.assert_syntax_error(
            "Numeric indices are not allowed on constant values.",
            "1 + 1_i0 + 1", "",
            "    ^^^^")

    def test_const_repeated_indices(self):
        self.assert_syntax_error(
            "Indices of a constant value may not be repeated.",
            "1 + 1_ii + 1", "",
            "    ^^^^")

    def test_const_index_pos(self):
        self.assert_syntax_error(
            "Length of axis cannot be determined from the expression.",
            "1_i", "i",
            "  ^")

    # NEG

    def test_neg_no_whitspace(self): self.assert_ast('-a2_i', 'i', ('neg', v._a2))
    def test_neg_whitespace(self): self.assert_ast('- a2_i', 'i', ('neg', v._a2))
    def test_neg_in_group(self): self.assert_ast('(- a2_i)', 'i', ('group', ('neg', v._a2)))

    # ADD SUB

    def test_add_sub_unmatched_indices(self):
        self.assert_syntax_error(
            "Cannot add arrays with unmatched indices: 'i', 'j'.",
            "a22_ij + (a2_i + a2_j + a2_ij)", "ij",
            "          ^^^^^^^^^^^")

    # POW

    def test_array_pow_pos(self): self.assert_ast('a2_i^2', 'i', ('pow', v._a2, _(2)))
    def test_array_pow_neg(self): self.assert_ast('a2_i^-2', 'i', ('pow', v._a2, ('neg', _(2))))
    def test_array_pow_scientific(self): self.assert_ast('a2_i^1e1', 'i', ('pow', v._a2, _(1e1)))
    def test_array_pow_scalar_expr(self): self.assert_ast('a2_i^(1 / 3)', 'i', ('pow', v._a2, ('truediv', _(1), _(3))))
    def test_scalar_pow_pos(self): self.assert_ast('2^3', '', ('pow', _(2), _(3)))
    def test_scalar_pow_neg(self): self.assert_ast('2^-3', '', ('pow', _(2), ('neg', _(3))))
    def test_scalar_pow_scalar_expr(self): self.assert_ast('2^(1 / 3)', '', ('pow', _(2), ('truediv', _(1), _(3))))

    def test_array_pow_nonconst(self):
        self.assert_syntax_error(
            "Expected a number.",
            "a2_i + a2_i^a + a2_i", "i",
            "            ^")

    def test_array_pow_vector_expr(self):
        self.assert_syntax_error(
            "An exponent must have dimension 0.",
            "1_i + a2_i^(a2_j) + 1_i", "ij",
            "      ^^^^^^^^^^^")

    def test_array_pow_repeated_indices(self):
        self.assert_syntax_error(
            "Index 'i' occurs more than twice.",
            "1_i + a2_i^(a22_ii) + 1_i", "i",
            "      ^^^^^^^^^^^^^")

    # NUMERIC INDEX

    def test_numeric_index(self): self.assert_ast('a23_i0', 'i', ('getitem', v._a23, _(1), _(0)))
    def test_numeric_index_grad(self): self.assert_ast('a2_i,1', 'i', ('getitem', ('grad', v._a2, v._x), _(1), _(1)))

    def test_numeric_index_out_of_range(self):
        self.assert_syntax_error(
            "Index of dimension 1 with length 4 out of range.",
            "1 + a343_i4i + 1", "",
            "    ^^^^^^^^")

    def test_numeric_index_out_of_range_grad(self):
        self.assert_syntax_error(
            "Index of dimension 0 with length 2 out of range.",
            "1 + a2_1,2 + 1", "",
            "    ^^^^^^")

    # EYE

    def test_single_eye(self):
        for eye in 'δ$':
            with self.subTest(eye=eye):
                self.assert_ast('a2_i {}_ij'.format(eye), 'j', ('sum', ('mul', ('append_axis', v._a2, _(2)), ('eye', _(2))), _(0)))

    def test_multiple_eye(self): self.assert_ast('δ_ij δ_jk a2_i a2_k', '',
                                                 ('sum',
                                                  ('mul',
                                                   ('sum',
                                                    ('mul',
                                                     ('sum',
                                                      ('mul',
                                                       ('append_axis', ('eye', _(2)), _(2)),
                                                          ('transpose', ('append_axis', ('eye', _(2)), _(2)), _((2, 0, 1)))),
                                                         _(1)),
                                                        ('append_axis', v._a2, _(2))),
                                                       _(0)),
                                                      v._a2),
                                                     _(0)))

    def test_eye_missing_indices(self):
        self.assert_syntax_error(
            "Expected 2 indices, got 0.",
            "1 + δ + 1", "",
            "    ^")

    def test_eye_invalid_number_of_indices(self):
        self.assert_syntax_error(
            "Expected 2 indices, got 3.",
            "1 + δ_ijk + 1", "",
            "    ^^^^^")

    def test_eye_same_index(self):
        self.assert_syntax_error(
            "Length of axis cannot be determined from the expression.",
            "1 + δ_ii + 1", "",
            "      ^")

    def test_eye_shape_mismatch(self):
        self.assert_syntax_error(
            "Shapes at index 'k' differ: 2, 3.",
            "1 + δ_ij δ_jk a2_i a3_k + 1", "",
            "    ^^^^^^^^^^^^^^^^^^^")

    def test_variable_startswith_eye(self):
        b = Array('b', [2, 2])
        δx = Array('δx', [2, 2])
        self.assert_ast('b_ij + δx_ij', 'ij', ('add', _(b), _(δx)), variables=dict(b=b, δx=δx))

    def test_variable_startswith_dollar_eye(self):
        self.assert_syntax_error(
            "Expected 2 indices, got 0.",
            "a22_ij + $x_ij", "ij",
            "         ^")

    # GRAD

    def test_gradient_default(self): self.assert_ast('a2_i,j', 'ij', ('grad', v._a2, v._x))
    def test_gradient_other_default(self): self.assert_ast('a2_i,j', 'ij', ('grad', v._a2, v._altgeom), default_geometry_name='altgeom')
    def test_gradient_default_trace(self): self.assert_ast('a2_i,i', '', ('trace', ('grad', v._a2, v._x), _(0), _(1)))
    def test_gradient_default_double_trace(self): self.assert_ast('a422_ijk,jk', 'i', ('trace', ('grad', ('trace', ('grad', v._a422, v._x), _(1), _(3)), v._x), _(1), _(2)))

    def test_gradient_altgeom(self):
        with self.assertWarns(warnings.NutilsDeprecationWarning):
            self.assert_ast('a3_i,altgeom_j', 'ij', ('grad', v._a3, v._altgeom))

    def test_gradient_altgeom_trace(self):
        with self.assertWarns(warnings.NutilsDeprecationWarning):
            self.assert_ast('a3_i,altgeom_i', '', ('trace', ('grad', v._a3, v._altgeom), _(0), _(1)))

    def test_gradient_altgeom_double_trace(self):
        with self.assertWarns(warnings.NutilsDeprecationWarning):
            self.assert_ast('a433_ijk,altgeom_jk', 'i', ('trace', ('grad', ('trace', ('grad', v._a433, v._altgeom), _(1), _(3)), v._altgeom), _(1), _(2)))

    def test_surfgrad_default(self): self.assert_ast('a2_i;j', 'ij', ('surfgrad', v._a2, v._x))
    def test_surfgrad_default_trace(self): self.assert_ast('a2_i;i', '', ('trace', ('surfgrad', v._a2, v._x), _(0), _(1)))

    def test_gradient_invalid_geom_0dim(self):
        with self.assertWarns(warnings.NutilsDeprecationWarning):
            self.assert_syntax_error(
                "Invalid geometry: expected 1 dimension, but 'a' has 0.",
                "1 + a2_i,a_i + 1", "",
                "    ^^^^^^^^")

    def test_gradient_invalid_geom_2dim(self):
        with self.assertWarns(warnings.NutilsDeprecationWarning):
            self.assert_syntax_error(
                "Invalid geometry: expected 1 dimension, but 'a22' has 2.",
                "1 + a2_i,a22_i + 1", "",
                "    ^^^^^^^^^^")

    def test_gradient_const_scalar(self):
        self.assert_syntax_error(
            "Taking a derivative of a constant is not allowed.",
            "1_i + 1_,i + 1_i", "i",
            "      ^^^^")

    def test_gradient_const_array(self):
        self.assert_syntax_error(
            "Taking a derivative of a constant is not allowed.",
            "1 + 1_i,i + 1", "",
            "    ^^^^^")

    # NEW GRAD

    def test_newgradient(self):
        with self.assertWarns(warnings.NutilsDeprecationWarning):
            self.assert_ast('dx_j:a2_i', 'ij', ('grad', v._a2, v._x))

    def test_newgradient_trace(self):
        with self.assertWarns(warnings.NutilsDeprecationWarning):
            self.assert_ast('dx_i:a2_i', '', ('trace', ('grad', v._a2, v._x), _(0), _(1)))

    def test_newgradient_double_trace(self):
        with self.assertWarns(warnings.NutilsDeprecationWarning):
            self.assert_ast('dx_k:(dx_j:a422_ijk)', 'i', ('trace', ('grad', ('group', ('trace', ('grad', v._a422, v._x), _(1), _(3))), v._x), _(1), _(2)))

    # DERIVATIVE

    def test_derivative0(self):
        with self.assertWarns(warnings.NutilsDeprecationWarning):
            self.assert_ast('(2 ?arg + 1)_,?arg', '', ('derivative', ('group', ('add', ('mul', _(2), ('arg', _('arg'))), _(1))), ('arg', _('arg'))))

    def test_derivative1(self):
        with self.assertWarns(warnings.NutilsDeprecationWarning):
            self.assert_ast('(a2_i + ?arg_i)_,?arg_j', 'ij', ('derivative', ('group', ('add', v._a2, ('arg', _('arg'), _(2)))), ('arg', _('arg'), _(2))))

    def test_derivative2(self):
        with self.assertWarns(warnings.NutilsDeprecationWarning):
            self.assert_ast('(a23_ij + ?arg_ij)_,?arg_kj', 'ik', ('trace', ('derivative', ('group', ('add', v._a23, ('arg', _('arg'), _(2), _(3)))), ('arg', _('arg'), _(2), _(3))), _(1), _(3)))

    # NEW DERIVATIVE

    def test_newderivative0(self):
        with self.assertWarns(warnings.NutilsDeprecationWarning):
            self.assert_ast('d?arg:(2 ?arg + 1)', '', ('derivative', ('group', ('add', ('mul', _(2), ('arg', _('arg'))), _(1))), ('arg', _('arg'))))

    def test_newderivative1(self):
        with self.assertWarns(warnings.NutilsDeprecationWarning):
            self.assert_ast('d?arg_j:(a2_i + ?arg_i)', 'ij', ('derivative', ('group', ('add', v._a2, ('arg', _('arg'), _(2)))), ('arg', _('arg'), _(2))))

    def test_newderivative2(self):
        with self.assertWarns(warnings.NutilsDeprecationWarning):
            self.assert_ast('d?arg_kj:(a23_ij + ?arg_ij)', 'ik', ('trace', ('derivative', ('group', ('add', v._a23, ('arg', _('arg'), _(2), _(3)))), ('arg', _('arg'), _(2), _(3))), _(1), _(3)))

    # NORMAL

    def test_normal_default(self): self.assert_ast('n_i', 'i', ('normal', v._x))
    def test_normal_default_grad_default(self): self.assert_ast('n_i,j', 'ij', ('grad', ('normal', v._x), v._x))

    def test_normal(self):
        with self.assertWarns(warnings.NutilsDeprecationWarning):
            self.assert_ast('n:x_i', 'i', ('normal', v._x))

    def test_normal_altgeom(self):
        with self.assertWarns(warnings.NutilsDeprecationWarning):
            self.assert_ast('n_altgeom_i', 'i', ('normal', v._altgeom))

    def test_normal_altgeom_grad_default(self):
        with self.assertWarns(warnings.NutilsDeprecationWarning):
            self.assert_ast('n_altgeom_i,x_j', 'ij', ('grad', ('normal', v._altgeom), v._x))

    def test_normal_altgeom_grad_altgeom(self):
        with self.assertWarns(warnings.NutilsDeprecationWarning):
            self.assert_ast('n_altgeom_i,altgeom_j', 'ij', ('grad', ('normal', v._altgeom), v._altgeom))

    def test_normal_altgeom_grad_nogeom(self):
        self.assert_syntax_error(
            "Missing geometry, e.g. ',altgeom_i' or ',x_i'.",
            "1 + n_altgeom_i,i + 1", "",
            "               ^")

    def test_normal_missing_indices(self):
        self.assert_syntax_error(
            "Expected 1 index, got 0.",
            "1 + n + 1", "",
            "    ^")

    def test_normal_too_many_indices(self):
        self.assert_syntax_error(
            "Expected 1 index, got 2.",
            "1 + n_ij + 1", "",
            "    ^^^^")

    def test_normal_invalid_geom_0dim(self):
        with self.assertWarns(warnings.NutilsDeprecationWarning):
            self.assert_syntax_error(
                "Invalid geometry: expected 1 dimension, but 'a' has 0.",
                "1 + n_a_i + 1", "",
                "    ^^^^")

    def test_normal_invalid_geom_2dim(self):
        with self.assertWarns(warnings.NutilsDeprecationWarning):
            self.assert_syntax_error(
                "Invalid geometry: expected 1 dimension, but 'a22' has 2.",
                "1 + n_a22_i + 1", "",
                "    ^^^^^^")

    def test_variable_startswith_normal(self):
        nx = Array('nx', [2])
        self.assert_ast('nx_i', 'i', _(nx), variables=dict(nx=nx))

    # JACOBIAN

    def test_jacobian(self): self.assert_ast('J:x', '', ('jacobian', v._x, _(2)))
    def test_jacobian_boundary(self): self.assert_ast('J^:x', '', ('jacobian', v._x, _(1)))
    def test_jacobian_double_boundary(self): self.assert_ast('J^^:x', '', ('jacobian', v._x, _(0)))
    def test_old_jacobian(self): self.assert_ast('d:x', '', ('jacobian', v._x, _(None)))

    # VARIABLE LENGTH TESTS

    def test_variable_lengths_shape_mismatch1(self):
        self.assert_syntax_error(
            "Axes have different lengths: 2, 3.",
            "aXY_ii + aX2_ii + aY3_ii", "",
            "^^^^^^^^^^^^^^^^^^^^^^^^")

    def test_variable_lengths_shape_mismatch2(self):
        self.assert_syntax_error(
            "Shapes at index 'j' differ: 2, 3.",
            "aX2X3_iijj", "",
            "^^^^^^^^^^")

    # FIXED LENGTHS

    def test_fixed_lengths(self): self.assert_ast('δ_ij', 'ij', ('eye', _(3)), fixed_lengths=dict(i=3))

    def test_fixed_lengths_invalid(self):
        self.assert_syntax_error(
            'Length of index i is fixed at 3 but the expression has length 2.',
            'a2_i', 'i',
            '   ^',
            fixed_lengths=dict(i=3))

    def test_fixed_lengths_invalid_linked(self):
        self.assert_syntax_error(
            'Axes have different lengths: 2, 3.',
            'a2_i δ_ij', 'j',
            '^^^^^^^^^',
            fixed_lengths=dict(j=3))

    # FALLBACK LENGHT

    def test_fallback_length(self): self.assert_ast('1_i', 'i', ('append_axis', _(1), _(2)), fallback_length=2)

    # ARG

    def test_arg0(self): self.assert_ast('a ?coeffs', '', ('mul', v._a, ('arg', _('coeffs'))))
    def test_arg1(self): self.assert_ast('a2_i ?coeffs_i', '', ('sum', ('mul', v._a2, ('arg', _('coeffs'), _(2))), _(0)))
    def test_arg2(self): self.assert_ast('a23_ij ?coeffs_ij', '', ('sum', ('sum', ('mul', v._a23, ('arg', _('coeffs'), _(2), _(3))), _(1)), _(0)))

    def test_arg_reshape(self):
        self.assert_syntax_error(
            "Argument 'arg' previously defined with 1 axis instead of 2.",
            "a2_i (a2_j + ?arg_j) + ?arg_ij", "ij",
            "                       ^^^^^^^")

    def test_arg_shape_mismatch(self):
        self.assert_syntax_error(
            "Axes have different lengths: 2, 3.",
            "1 + a2_i ?arg_i + a3_j ?arg_j + 1", "",
            "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")

    def test_arg_reshape_external(self):
        self.assert_syntax_error(
            "Argument 'arg' previously defined with 2 axes instead of 1.",
            "1 + a3_j ?arg_j + 1", "",
            "         ^^^^^^",
            {'arg': (2, 3)})

    def test_arg_shape_mismatch_external(self):
        self.assert_syntax_error(
            "Shapes at index 'j' differ: 3, 2.",
            "1 + a3_j ?arg_j + 1", "",
            "    ^^^^^^^^^^^",
            {'arg': (2,)})

    def test_arg_index_pos1(self):
        self.assert_syntax_error(
            "Length of axis cannot be determined from the expression.",
            "?arg_n", "n",
            "     ^")

    def test_arg_index_pos2(self):
        with self.assertWarns(warnings.NutilsDeprecationWarning):
            self.assert_syntax_error(
                "Length of axis cannot be determined from the expression.",
                "?foo_,?bar_n", "n",
                "           ^")

    # SUBSTITUTE

    def test_arg_subs_0d_const(self):
        with self.assertWarns(warnings.NutilsDeprecationWarning):
            self.assert_ast('?arg_,?arg(arg=1)', '', ('substitute', ('derivative', ('arg', _('arg')), ('arg', _('arg'))), ('arg', _('arg')), _(1)))

    def test_arg_subs_0d_var(self):
        with self.assertWarns(warnings.NutilsDeprecationWarning):
            self.assert_ast('?arg_,?arg(arg=a )', '', ('substitute', ('derivative', ('arg', _('arg')), ('arg', _('arg'))), ('arg', _('arg')), v._a))

    def test_arg_subs_1d_var(self):
        with self.assertWarns(warnings.NutilsDeprecationWarning):
            self.assert_ast('?arg_i,?arg_j(arg_i = a2_i)', 'ij', ('substitute', ('derivative', ('arg', _('arg'), _(2)), ('arg', _('arg'), _(2))), ('arg', _('arg'), _(2)), v._a2))

    def test_arg_subs_2d_var(self):
        with self.assertWarns(warnings.NutilsDeprecationWarning):
            self.assert_ast('?arg_ij,?arg_kl( arg_ij =a23_ji)', 'ijkl', ('substitute', ('derivative', ('arg', _('arg'), _(3), _(2)), ('arg', _('arg'), _(3), _(2))), ('arg', _('arg'), _(3), _(2)), ('transpose', v._a23, _((1, 0)))))

    def test_arg_multisubs(self): self.assert_ast('(1 + ?x + ?y)(x=1 + a, y=2)', '', ('substitute', ('group', ('add', ('add', _(1), ('arg', _('x'))), ('arg', _('y')))), ('arg', _('x')), ('add', _(1), v._a), ('arg', _('y')), _(2)))

    def test_arg_subs_missing_equals(self):
        self.assert_syntax_error(
            "Expected '='.",
            "(1 + ?x)(x + 2)", "",
            "           ^")

    def test_arg_subs_unmatched_indices(self):
        self.assert_syntax_error(
            "Left and right hand side should have the same indices, got 'kl' and 'jk'.",
            "a23_ij + ?x_ij(x_kl=a23_jk) + a23_ij", "ij",
            "               ^^^^^^^^^^^")

    def test_arg_subs_lhs_repeated_index(self):
        self.assert_syntax_error(
            "Repeated indices are not allowed on the left hand side.",
            "a23_ij + ?x_ij(x_kk=a23_jk) + 2", "ij",
            "               ^^^^")

    def test_arg_subs_lhs_numeric_index(self):
        self.assert_syntax_error(
            "Numeric indices are not allowed on the left hand side.",
            "a23_ij + ?x_ij(x_k0=a23_0k) + 2", "ij",
            "               ^^^^")

    def test_arg_subs_lhs_with_questionmark(self):
        self.assert_syntax_error(
            "The argument name at the left hand side of a substitution must not be prefixed by a '?'.",
            "?x_ij(?x_ij=1_ij)", "ij",
            "      ^^^")

    def test_arg_subs_lhs_not_an_argument(self):
        self.assert_syntax_error(
            "Expected an argument, e.g. 'argname'.",
            "?x(1=2)", "",
            "   ^")

    def test_arg_subs_double_occurence(self):
        self.assert_syntax_error(
            "Argument 'x' occurs more than once.",
            "?x(x=1, x=2)", "",
            "        ^")

    def test_arg_subs_zero(self):
        self.assert_syntax_error(
            "Zero substitutions are not allowed.",
            "?x_ij()", "ij",
            "^^^^^^^")

    # TRANSPOSE

    def test_transpose_duplicate_indices(self):
        self.assert_syntax_error(
            "Cannot transpose from 'ij' to 'jii': duplicate indices.",
            "a23_ij", "jii",
            "^^^^^^")

    def test_transpose_indices_differ(self):
        self.assert_syntax_error(
            "Cannot transpose from 'ij' to 'jk': indices differ.",
            "a23_ij", "jk",
            "^^^^^^")

    # STACK

    def test_stack_1_0d(self): self.assert_ast('<a>_i', 'i', ('append_axis', v._a, _(1)))
    def test_stack_2_0d_0d(self): self.assert_ast('<a, a>_i', 'i', ('concatenate', ('append_axis', v._a, _(1)), ('append_axis', v._a, _(1))))

    def test_stack_no_indices(self):
        self.assert_syntax_error(
            "Expected 1 index.",
            "<1, a2_i> + a3_i", "i",
            "         ^")

    def test_stack_too_many_indices(self):
        self.assert_syntax_error(
            "Expected 1 index, got 2.",
            "<1, a2_i>_ij + a3_i", "i",
            "          ^^")

    def test_stack_numeric_index(self):
        self.assert_syntax_error(
            "Expected a non-numeric index, got '1'.",
            "<1, a2_i>_1 + a3_i", "i",
            "          ^")

    def test_stack_0(self):
        self.assert_syntax_error(
            "Cannot stack 0 arrays.",
            "1_i + <>_i + 1_i", "i",
            "      ^^^^")

    def test_stack_unmatched_indices(self):
        self.assert_syntax_error(
            "Cannot stack arrays with unmatched indices (excluding the stack index 'i'): j, ijk.",
            "1_ij + <a2_j, a222_ijk>_i + 1_ij", "ij",
            "       ^^^^^^^^^^^^^^^^^^")

    def test_stack_whitespace_left(self): self.assert_ast('< a, a>_i', 'i', ('concatenate', ('append_axis', v._a, _(1)), ('append_axis', v._a, _(1))))
    def test_stack_whitespace_right(self): self.assert_ast('<a, a >_i', 'i', ('concatenate', ('append_axis', v._a, _(1)), ('append_axis', v._a, _(1))))
    def test_stack_whitespace_before_comma(self): self.assert_ast('<a , a>_i', 'i', ('concatenate', ('append_axis', v._a, _(1)), ('append_axis', v._a, _(1))))

    def test_stack_1_1di_1(self):
        with self.assertWarnsRegex(warnings.NutilsDeprecationWarning, 'Concatenating arrays .* is deprecated.$'):
            self.assert_ast('<a2_i>_i', 'i', v._a2)

    def test_stack_2_1di_0d(self):
        with self.assertWarnsRegex(warnings.NutilsDeprecationWarning, 'Concatenating arrays .* is deprecated.$'):
            self.assert_ast('<a2_i, a>_i', 'i', ('concatenate', v._a2, ('append_axis', v._a, _(1))))

    def test_stack_3_2di_1d_1d(self):
        with self.assertWarnsRegex(warnings.NutilsDeprecationWarning, 'Concatenating arrays .* is deprecated.$'):
            self.assert_ast('<a23_ij, a3_j, 1_j>_i', 'ij', ('concatenate', v._a23, ('transpose',  ('append_axis', v._a3, _(1)), _((1, 0))), ('transpose', ('append_axis', ('append_axis', _(1), _(3)), _(1)), _((1, 0)))))

    def test_stack_undetermined_length(self):
        with self.assertWarnsRegex(warnings.NutilsDeprecationWarning, 'Concatenating arrays .* is deprecated.$'):
            self.assert_syntax_error(
                "Cannot determine the length of the stack axis, because the length at 12 is unknown.",
                "1_i + <a, 1_i>_i + 1_i", "i",
                "            ^")

    # FUNCTION

    def test_function(self): self.assert_ast('func1(a)', '', ('call', _('func1'), _(0), _(0), v._a))
    def test_function_1d(self): self.assert_ast('func1(a2_i)', 'i', ('call', _('func1'), _(0), _(0), v._a2))
    def test_function_2d(self): self.assert_ast('func1(a23_ij)', 'ij', ('call', _('func1'), _(0), _(0), v._a23))
    def test_function_0d_0d(self): self.assert_ast('func2(a, a)', '', ('call', _('func2'), _(0), _(0), v._a, v._a))
    def test_function_1d_1d(self): self.assert_ast('func2(a2_i, a2_j)', 'ij', ('call', _('func2'), _(0), _(0), v._a2, v._a2))
    def test_function_1d_1d_trace(self): self.assert_ast('func2(a2_i, a2_i)', '', ('trace', ('call', _('func2'), _(0), _(0), v._a2, v._a2), _(0), _(1)))
    def test_function_2d_2d(self): self.assert_ast('func2(a23_ij, a32_kl)', 'ijkl', ('call', _('func2'), _(0), _(0), v._a23, v._a32))
    def test_function_1d_1d_2d(self): self.assert_ast('func3(a2_i, a2_j, a23_kl)', 'ijkl', ('call', _('func3'), _(0), _(0), v._a2, v._a2, v._a23))
    def test_function_generates(self): self.assert_ast('func_j(a2_i)', 'ij', ('call', _('func'), _(1), _(0), v._a2), fallback_length=2)
    def test_function_generates_trace(self): self.assert_ast('func_i(a2_i)', '', ('trace', ('call', _('func'), _(1), _(0), v._a2), _(0), _(1)))
    def test_function_consumes(self): self.assert_ast('sum:i(a2_i)', '', ('call', _('sum'), _(0), _(1), v._a2))
    def test_function_consumes_transpose(self): self.assert_ast('sum:i(a23_ij)', 'j', ('call', _('sum'), _(0), _(1), ('transpose', v._a23, _((1, 0)))))
    def test_function_consumes_omitted(self): self.assert_ast('sum(a2)', '', ('call', _('sum'), _(0), _(1), v._a2))

    def test_function_triple_index(self):
        self.assert_syntax_error(
            "Index 'i' occurs more than twice.",
            "1_i + func(a2_i, a2_i, a2_i) + 1_i", "i",
            "      ^^^^^^^^^^^^^^^^^^^^^^")

    def test_function_unmatched_shape(self):
        self.assert_syntax_error(
            "Shapes at index 'i' differ: 2, 3.",
            "1 + func2(a23_ij, a33_ij) + 1", "",
            "    ^^^^^^^^^^^^^^^^^^^^^")

    def test_function_override(self):
        self.assert_syntax_error(
            "Expected '='.",
            "1_ij + funcoverride(a23_ij) + 1_ij", "ij",
            "                          ^")

    def test_function_consumes_missing_index(self):
        self.assert_syntax_error(
            "All axes to be consumed (i) must be present in all arguments.",
            "1 + sum:i(a) + 1", "",
            "    ^^^^^^^^")

    def test_function_consumes_omitted_shape_mismatch(self):
        self.assert_syntax_error(
            "All arguments should have the same shape.",
            "1 + f(a2, a3) + 1", None,
            "    ^^^^^^^^^")

    # OMITTED INDICES

    def test_omitted_add(self): self.assert_ast('a2 + a2', None, ('add', v._a2, v._a2))
    def test_omitted_sub(self): self.assert_ast('a2 - a2', None, ('sub', v._a2, v._a2))
    def test_omitted_truediv(self): self.assert_ast('a2 / a', None, ('truediv', v._a2, v._a))
    def test_omitted_neg(self): self.assert_ast('-a2', None, ('neg', v._a2))
    def test_omitted_pow(self): self.assert_ast('a2^2', None, ('pow', v._a2, _(2)))
    def test_omitted_group(self): self.assert_ast('(a2)', None, ('group', v._a2))
    def test_omitted_jump(self): self.assert_ast('[a2]', None, ('jump', v._a2))
    def test_omitted_mean(self): self.assert_ast('{a2}', None, ('mean', v._a2))
    def test_omitted_function(self): self.assert_ast('sum(a2)', None, ('call', _('sum'), _(0), _(1), v._a2))
    def test_omitted_normal(self): self.assert_ast('n', None, ('normal', v._x))

    def test_omitted_add_shape_mismatch(self):
        with self.assertRaises(expression_v1.ExpressionSyntaxError):
            expression_v1.parse("a2 + a3", v, None)

    def test_omitted_sub_shape_mismatch(self):
        with self.assertRaises(expression_v1.ExpressionSyntaxError):
            expression_v1.parse("a2 - a3", v, None)

    def test_omitted_mul(self):
        with self.assertRaises(expression_v1.ExpressionSyntaxError):
            expression_v1.parse("a2 a3", v, None)

    def test_omitted_truediv_nonscalar_denominator(self):
        with self.assertRaises(expression_v1.ExpressionSyntaxError):
            expression_v1.parse("a2 / a3", v, None)

    def test_omitted_pow_nonscalar_exponent(self):
        with self.assertRaises(expression_v1.ExpressionSyntaxError):
            expression_v1.parse("a2^(a3)", v, None)


class namespace(TestCase):

    def test_set_scalar(self):
        ns = expression_v1.Namespace()
        ns.scalar = 1

    def test_set_array(self):
        ns = expression_v1.Namespace()
        ns.array = function.zeros([2, 3])

    def test_set_scalar_expression(self):
        ns = expression_v1.Namespace()
        ns.scalar = '1'

    def test_set_array_expression(self):
        ns = expression_v1.Namespace()
        ns.foo = function.zeros([3, 3])
        ns.array_ij = 'foo_ij + foo_ji'

    def test_set_readonly(self):
        ns = expression_v1.Namespace()
        with self.assertRaises(AttributeError):
            ns._foo = None

    def test_set_readonly_internal(self):
        ns = expression_v1.Namespace()
        with self.assertRaises(AttributeError):
            ns._attributes = None

    def test_del_existing(self):
        ns = expression_v1.Namespace()
        ns.foo = function.zeros([2, 3])
        del ns.foo

    def test_del_readonly_internal(self):
        ns = expression_v1.Namespace()
        with self.assertRaises(AttributeError):
            del ns._attributes

    def test_del_nonexisting(self):
        ns = expression_v1.Namespace()
        with self.assertRaises(AttributeError):
            del ns.foo

    def test_get_nonexisting(self):
        ns = expression_v1.Namespace()
        with self.assertRaises(AttributeError):
            ns.foo

    def test_invalid_default_geometry_no_str(self):
        with self.assertRaises(ValueError):
            expression_v1.Namespace(default_geometry_name=None)

    def test_invalid_default_geometry_no_variable(self):
        with self.assertRaises(ValueError):
            expression_v1.Namespace(default_geometry_name='foo_bar')

    def assertEqualLowered(self, actual, desired, *, topo=None):
        if topo:
            smpl = topo.sample('gauss', 2)
            lower = lambda f: evaluable.asarray(smpl(f))
        else:
            lower = evaluable.asarray
        return self.assertEqual(lower(actual), lower(desired))

    def test_default_geometry_property(self):
        ns = expression_v1.Namespace()
        ns.x = 1
        self.assertEqualLowered(ns.default_geometry, ns.x)
        ns = expression_v1.Namespace(default_geometry_name='y')
        ns.y = 2
        self.assertEqualLowered(ns.default_geometry, ns.y)

    def test_copy(self):
        ns = expression_v1.Namespace()
        ns.foo = function.zeros([2, 3])
        ns = ns.copy_()
        self.assertTrue(hasattr(ns, 'foo'))

    def test_copy_change_geom(self):
        ns1 = expression_v1.Namespace()
        domain, ns1.y = mesh.rectilinear([2, 2])
        ns1.basis = domain.basis('spline', degree=2)
        ns2 = ns1.copy_(default_geometry_name='y')
        self.assertEqual(ns2.default_geometry_name, 'y')
        self.assertEqualLowered(ns2.eval_ni('basis_n,i'), ns2.basis.grad(ns2.y), topo=domain)

    def test_copy_preserve_geom(self):
        ns1 = expression_v1.Namespace(default_geometry_name='y')
        domain, ns1.y = mesh.rectilinear([2, 2])
        ns1.basis = domain.basis('spline', degree=2)
        ns2 = ns1.copy_()
        self.assertEqual(ns2.default_geometry_name, 'y')
        self.assertEqualLowered(ns2.eval_ni('basis_n,i'), ns2.basis.grad(ns2.y), topo=domain)

    def test_copy_fixed_lengths(self):
        ns = expression_v1.Namespace(length_i=2)
        ns = ns.copy_()
        self.assertEqual(ns.eval_ij('δ_ij').shape, (2, 2))

    def test_copy_fallback_length(self):
        ns = expression_v1.Namespace(fallback_length=2)
        ns = ns.copy_()
        self.assertEqual(ns.eval_ij('δ_ij').shape, (2, 2))

    def test_eval(self):
        ns = expression_v1.Namespace()
        ns.foo = function.zeros([3, 3])
        ns.eval_ij('foo_ij + foo_ji')

    def test_eval_fixed_lengths(self):
        ns = expression_v1.Namespace(length_i=2)
        self.assertEqual(ns.eval_ij('δ_ij').shape, (2, 2))

    def test_eval_fixed_lengths_multiple(self):
        ns = expression_v1.Namespace(length_jk=2)
        self.assertEqual(ns.eval_ij('δ_ij').shape, (2, 2))
        self.assertEqual(ns.eval_ik('δ_ik').shape, (2, 2))

    def test_eval_fallback_length(self):
        ns = expression_v1.Namespace(fallback_length=2)
        self.assertEqual(ns.eval_ij('δ_ij').shape, (2, 2))

    def test_matmul_0d(self):
        ns = expression_v1.Namespace()
        ns.foo = 2
        self.assertEqualLowered('foo' @ ns, ns.foo)

    def test_matmul_1d(self):
        ns = expression_v1.Namespace()
        ns.foo = function.zeros([2])
        self.assertEqualLowered('foo_i' @ ns, ns.foo)

    def test_matmul_2d(self):
        ns = expression_v1.Namespace()
        ns.foo = function.zeros([2, 3])
        with self.assertRaises(ValueError):
            'foo_ij' @ ns

    def test_matmul_nostr(self):
        ns = expression_v1.Namespace()
        with self.assertRaises(TypeError):
            1 @ ns

    def test_matmul_fixed_lengths(self):
        ns = expression_v1.Namespace(length_i=2)
        self.assertEqual(('1_i δ_ij' @ ns).shape, (2,))

    def test_matmul_fallback_length(self):
        ns = expression_v1.Namespace(fallback_length=2)
        self.assertEqual(('1_i δ_ij' @ ns).shape, (2,))

    def test_replace(self):
        ns = expression_v1.Namespace(default_geometry_name='y')
        ns.foo = function.Argument('arg', [2, 3])
        ns.bar_ij = 'sin(foo_ij) + cos(2 foo_ij)'
        ns = ns(arg=function.zeros([2, 3]))
        self.assertEqualLowered(ns.foo, function.zeros([2, 3]))
        self.assertEqual(ns.default_geometry_name, 'y')

    def test_pickle(self):
        orig = expression_v1.Namespace()
        domain, geom = mesh.unitsquare(2, 'square')
        orig.x = geom
        orig.v = numpy.stack([1, geom[0], geom[0]**2], 0)
        orig.u = 'v_n ?lhs_n'
        orig.f = 'cosh(x_0)'
        pickled = pickle.loads(pickle.dumps(orig))
        for attr in ('x', 'v', 'u', 'f'):
            self.assertEqualLowered(getattr(pickled, attr), getattr(orig, attr), topo=domain)
        self.assertEqual(pickled.arg_shapes['lhs'], orig.arg_shapes['lhs'])

    def test_pickle_default_geometry_name(self):
        orig = expression_v1.Namespace(default_geometry_name='g')
        pickled = pickle.loads(pickle.dumps(orig))
        self.assertEqual(pickled.default_geometry_name, orig.default_geometry_name)

    def test_pickle_fixed_lengths(self):
        orig = expression_v1.Namespace(length_i=2)
        pickled = pickle.loads(pickle.dumps(orig))
        self.assertEqual(pickled.eval_ij('δ_ij').shape, (2, 2))

    def test_pickle_fallback_length(self):
        orig = expression_v1.Namespace(fallback_length=2)
        pickled = pickle.loads(pickle.dumps(orig))
        self.assertEqual(pickled.eval_ij('δ_ij').shape, (2, 2))

    def test_duplicate_fixed_lengths(self):
        with self.assertRaisesRegex(ValueError, '^length of index i specified more than once$'):
            expression_v1.Namespace(length_ii=2)

    def test_unexpected_keyword_argument(self):
        with self.assertRaisesRegex(TypeError, r"^__init__\(\) got an unexpected keyword argument 'test'$"):
            expression_v1.Namespace(test=2)

    def test_d_geom(self):
        ns = expression_v1.Namespace()
        topo, ns.x = mesh.rectilinear([1])
        self.assertEqualLowered(ns.eval_ij('d(x_i, x_j)'), function.grad(ns.x, ns.x), topo=topo)

    def test_d_arg(self):
        ns = expression_v1.Namespace()
        ns.a = '?a'
        self.assertEqual(ns.eval_('d(2. ?a + 1., ?a)').as_evaluable_array.simplified, function.asarray(2.).as_evaluable_array.simplified)

    def test_n(self):
        ns = expression_v1.Namespace()
        topo, ns.x = mesh.rectilinear([1])
        self.assertEqualLowered(ns.eval_i('n(x_i)'), function.normal(ns.x), topo=topo.boundary)

    def test_functions(self):
        def sqr(a):
            return a**2

        def mul(*args):
            if len(args) == 2:
                return args[0][(...,)+(None,)*args[1].ndim] * args[1][(None,)*args[0].ndim]
            else:
                return mul(mul(args[0], args[1]), *args[2:])
        ns = expression_v1.Namespace(functions=dict(sqr=sqr, mul=mul))
        ns.a = numpy.array([1, 2, 3])
        ns.b = numpy.array([4, 5])
        ns.A = numpy.array([[6, 7, 8], [9, 10, 11]])
        l = lambda f: f.as_evaluable_array.simplified
        self.assertEqual(l(ns.eval_i('sqr(a_i)')), l(sqr(ns.a)))
        self.assertEqual(l(ns.eval_ij('mul(a_i, b_j)')), l(ns.eval_ij('a_i b_j')))
        self.assertEqual(l(ns.eval_('mul(b_i, A_ij, a_j)')), l(ns.eval_('b_i A_ij a_j')))

    def test_builtin_functions(self):
        ns = expression_v1.Namespace()
        ns.a = numpy.array([1, 2, 3])
        ns.A = numpy.array([[6, 7, 8], [9, 10, 11]])
        l = lambda f: f.as_evaluable_array.simplified
        self.assertEqual(l(ns.eval_('norm2(a)')), l(numpy.linalg.norm(ns.a)))
        self.assertEqual(l(ns.eval_i('sum:j(A_ij)')), l(numpy.sum(ns.A, 1)))

    def test_builtin_jacobian_vector(self):
        ns = expression_v1.Namespace()
        domain, ns.x = mesh.rectilinear([1]*2)
        l = lambda f: evaluable.asarray(domain.sample('gauss', 2)(f)).simplified
        self.assertEqual(l(ns.eval_('J(x)')), l(function.jacobian(ns.x)))

    def test_builtin_jacobian_scalar(self):
        ns = expression_v1.Namespace()
        domain, (ns.t,) = mesh.rectilinear([1])
        l = lambda f: evaluable.asarray(domain.sample('gauss', 2)(f)).simplified
        self.assertEqual(l(ns.eval_('J(t)')), l(function.jacobian(ns.t[None])))

    def test_builtin_jacobian_matrix(self):
        ns = expression_v1.Namespace()
        ns.x = numpy.array([[1, 2], [3, 4]])
        with self.assertRaises(ValueError):
            ns.eval_('J(x)')

    def test_builtin_jacobian_vectorization(self):
        with self.assertRaises(NotImplementedError):
            expression_v1._J_expr(function.Array.cast([[1, 2], [3, 4]]), consumes=1)


class eval_ast(TestCase):

    def setUp(self):
        super().setUp()
        self.domain, x = mesh.rectilinear([2, 2])
        self.ns = expression_v1.Namespace()
        self.ns.x = x
        self.ns.altgeom = numpy.concatenate([self.ns.x, [0]], 0)
        self.ns.basis = self.domain.basis('spline', degree=2)
        self.ns.a = 2
        self.ns.a2 = numpy.array([1, 2.])
        self.ns.a3 = numpy.array([1, 2, 3])
        self.ns.a22 = numpy.array([[1, 2], [3, 4]])
        self.ns.a32 = numpy.array([[1, 2], [3, 4], [5, 6]])
        self.x = function.Argument('x', ())

    def assertEqualLowered(self, s, f, *, topo=None, indices=None):
        if topo is None:
            topo = self.domain
        smpl = topo.sample('gauss', 2)
        lower = lambda g: evaluable.asarray(smpl(g)).simplified
        if indices:
            evaluated = getattr(self.ns, 'eval_'+indices)(s)
        else:
            evaluated = s @ self.ns
        self.assertEqual(lower(evaluated), lower(f))

    def test_group(self): self.assertEqualLowered('(a)', self.ns.a)
    def test_arg(self): self.assertEqualLowered('a2_i ?x_i', numpy.matmul(self.ns.a2, function.Argument('x', [2])))
    def test_substitute(self): self.assertEqualLowered('(?x_i^2)(x_i=a2_i)', self.ns.a2**2)
    def test_multisubstitute(self): self.assertEqualLowered('(a2_i + ?x_i + ?y_i)(x_i=?y_i, y_i=?x_i)', self.ns.a2 + function.Argument('y', [2]) + function.Argument('x', [2]))
    def test_call(self): self.assertEqualLowered('sin(a)', numpy.sin(self.ns.a))
    def test_call2(self): self.assertEqualLowered('arctan2(a2_i, a3_j)', numpy.arctan2(self.ns.a2[:, None], self.ns.a3[None, :]), indices='ij')
    def test_eye(self): self.assertEqualLowered('δ_ij a2_i', numpy.matmul(function.eye(2), self.ns.a2))
    def test_normal(self): self.assertEqualLowered('n_i', self.ns.x.normal(), topo=self.domain.boundary)
    def test_getitem(self): self.assertEqualLowered('a2_0', self.ns.a2[0])
    def test_trace(self): self.assertEqualLowered('a22_ii', numpy.trace(self.ns.a22, axis1=0, axis2=1))
    def test_sum(self): self.assertEqualLowered('a2_i a2_i', numpy.sum(self.ns.a2 * self.ns.a2, axis=0))
    def test_concatenate(self): self.assertEqualLowered('<a, a>_i', numpy.concatenate([self.ns.a[None], self.ns.a[None]], axis=0))
    def test_grad(self): self.assertEqualLowered('basis_n,0', self.ns.basis.grad(self.ns.x)[:, 0])
    def test_surfgrad(self): self.assertEqualLowered('surfgrad(basis_0, altgeom_i)', function.grad(self.ns.basis[0], self.ns.altgeom, len(self.ns.altgeom)-1))
    def test_derivative(self): self.assertEqualLowered('d(exp(?x), ?x)', function.derivative(numpy.exp(self.x), self.x))
    def test_append_axis(self): self.assertEqualLowered('a a2_i', self.ns.a[None]*self.ns.a2)
    def test_transpose(self): self.assertEqualLowered('a22_ij a22_ji', numpy.sum(self.ns.a22 * self.ns.a22.T, [0,1]))
    def test_jump(self): self.assertEqualLowered('[a]', function.jump(self.ns.a))
    def test_mean(self): self.assertEqualLowered('{a}', function.mean(self.ns.a))
    def test_neg(self): self.assertEqualLowered('-a', -self.ns.a)
    def test_add(self): self.assertEqualLowered('a + ?x', self.ns.a + self.x)
    def test_sub(self): self.assertEqualLowered('a - ?x', self.ns.a - self.x)
    def test_mul(self): self.assertEqualLowered('a ?x', self.ns.a * self.x)
    def test_truediv(self): self.assertEqualLowered('a / ?x', self.ns.a / self.x)
    def test_pow(self): self.assertEqualLowered('a^2', self.ns.a**2)

    def test_unknown_opcode(self):
        with self.assertRaises(ValueError):
            expression_v1._eval_ast(('invalid-opcode',), {})

    def test_call_invalid_shape(self):
        with self.assertRaisesRegex(ValueError, '^expected an array with shape'):
            expression_v1._eval_ast(('call', (None, 'f'), (None, 0), (None, 0), (None, function.zeros((2,), float)), (None, function.zeros((3,), float))),
                                    dict(f=lambda a, b: a[None, :] * b[:, None]))  # result is transposed

    def test_surfgrad_deprecated(self):
        with self.assertWarns(warnings.NutilsDeprecationWarning):
            self.assertEqualLowered('basis_n;altgeom_0', function.grad(self.ns.basis, self.ns.altgeom, len(self.ns.altgeom)-1)[:, 0])

    def test_derivative_deprecated(self):
        with self.assertWarns(warnings.NutilsDeprecationWarning):
            self.assertEqualLowered('exp(?x)_,?x', function.derivative(numpy.exp(self.x), self.x))

# vim:shiftwidth=2:softtabstop=2:expandtab:foldmethod=indent:foldnestmax=2
