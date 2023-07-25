'''
The SI module provides a framework for working with physical units in Python.
It has no dependencies beyond Python itself, yet is fully inter-operable with
Numpy's API as well as Nutils' own function arrays.


Usage
-----

The SI module defines all base units and derived units of the International
System of Units (SI) are predefined, as well as the full set of metric
prefixes. Dimensional values are generated primarily by instantiating the
Quantity type with a string value.

    >>> v = SI.parse('7μN*5h/6g')

The Quantity constructor recognizes the multiplication (\*) and division (/)
operators to separate factors. Every factor can be prefixed with a scale and
suffixed with a power. The remainder must be either a unit, or else a unit with
a metric prefix.

In this example, the resulting object is of type "L/T", i.e. length over time,
which is a subtype of Quantity that stores the powers L=1 and T=-1. Many
subtypes are readily defined by their physical names; others can be created
through manipulation.

    >>> type(v) == SI.Velocity == SI.Length / SI.Time
    True

While Quantity can instantiate any subtype, we could have created the same
object by instantiating Velocity directly, which has the advantage of verifying
that the specified quantity is indeed of the desired dimension.

    >>> w = SI.Velocity('8km')
    Traceback (most recent call last):
         ...
    TypeError: expected [L/T], got [L]

Explicit subtypes can also be used in function annotations:

    >>> def f(size: SI.Length, load: SI.Force): pass

The Quantity type acts as an opaque container. As long as a quantity has a
physical dimension, its value is inaccessible. The value can only be retrieved
by dividing out a reference quantity, so that the result becomes dimensionless
and the Quantity wrapper falls away.

    >>> v / SI.parse('m/s')
    21.0

To simplify this fairly common situation, any operation involving a Quantity
and a string is handled by parsing the latter automatically.

    >>> v / 'm/s'
    21.0

A value can also be retrieved as textual output via string formatting. The
syntax is similar to that of floating point values, with the desired unit
taking the place of the 'f' suffix.

    >>> f'velocity: {v:.1m/s}'
    'velocity: 21.0m/s'

A Quantity container can hold an object of any type that supports arithmetical
manipulation. Though an object can technically be wrapped directly, the
idiomatic way is to rely on multiplication so as not to depend on the specifics
of the internal reference system.

    >>> import numpy
    >>> F = numpy.array([1,2,3]) * SI.parse('N')

No Numpy specific methods or attributes are defined. Array manipulations must
be performed via Numpy's API, which is supported via the array protocol ([NEP
18](https://numpy.org/neps/nep-0018-array-function-protocol.html)).

    >>> f'total force: {numpy.sum(F):.1N}'
    'total force: 6.0N'


Extension
---------

In case the predefined set of dimensions and units are insufficient, both can
be extended. For instance, though it is not part of the official SI system, it
might be desirable to add an angular dimension. This is done by creating a new
Dimension instance, using a symbol that avoids the existing symbols T, L, M, I,
Θ, N and J:

    >>> Angle = SI.Dimension.create('Φ')

At this point, the dimension is not very useful yet as it lacks units. To
rectify this we define the radian by its abbreviation 'rad' in terms of the
provided reference quantity, and assign it to the global table of units:

    >>> SI.units.rad = Angle.wrap(1.)

Additional units can be defined by relating them to pre-existing ones:

    >>> import math
    >>> SI.units.deg = math.pi / 180 * SI.units.rad

Alternatively, units can be defined using the same string syntax that is used
by the Quantity constructor. Nevertheless, the following statement fails as we
cannot define the same unit twice.

    >>> SI.units.deg = '0.017453292519943295rad'
    Traceback (most recent call last):
         ...
    ValueError: cannot define 'deg': unit is already defined

Having defined the new units we can directly use them:

    >>> angle = SI.parse('30deg')

Any function that accepts angular values will expect to receive them in a
specific unit. The new Angle dimension makes this unit explicit:

    >>> math.sin(angle / 'rad')
    0.49999999999999994
'''

import fractions
import operator
import typing
import functools


class Dimension(type):

    __cache = {} # subtypes

    @classmethod
    def create(mcls, arg):
        if not isinstance(arg, str):
            raise ValueError(f'create requires a string, got {type(arg).__name__}')
        if next(_split_factors(arg))[0] != arg:
            raise ValueError(f'invalid dimension string {arg!r}')
        if arg in mcls.__cache:
            raise ValueError(f'dimension {arg!r} is already in use')
        return mcls.from_powers({arg: fractions.Fraction(1)})

    @classmethod
    def from_powers(mcls, arg):
        if not isinstance(arg, dict):
            raise ValueError(f'from_powers requires a dict, got {type(arg).__name__}')
        if not all(isinstance(base, str) for base in arg.keys()):
            raise ValueError('all keys must be of type str')
        if not all(isinstance(power, fractions.Fraction) for power in arg.values()):
            raise ValueError('all values must be of type Fraction')
        powers = {base: power for base, power in arg.items() if power}
        name = ''.join(('*' if power > 0 else '/') + base
                     + (str(abs(power.numerator)) if abs(power.numerator) != 1 else '')
                     + ('_'+str(abs(power.denominator)) if abs(power.denominator) != 1 else '')
            for base, power in sorted(powers.items(), key=lambda item: item[::-1], reverse=True)).lstrip('*')
        try:
            cls = mcls.__cache[name]
        except KeyError:
            cls = mcls(f'[{name}]', (Quantity,), {})
            cls.__powers = powers
            cls.__qualname__ = 'Quantity.' + cls.__name__
            mcls.__cache[name] = cls
        return cls

    def __getattr__(cls, attr):
        if attr.startswith('[') and attr.endswith(']'):
            # this, together with __qualname__, is what makes pickle work
            return Dimension.from_powers({base: fractions.Fraction(power if isnumer else -power)
              for base, power, isnumer in _split_factors(attr[1:-1]) if power})
        raise AttributeError(attr)

    def __bool__(cls) -> bool:
        return bool(cls.__powers)

    def __or__(cls, other):
        return typing.Union[cls, other]

    def __ror__(cls, other):
        return typing.Union[other, cls]

    @staticmethod
    def _binop(op, a, b):
        return Dimension.from_powers({base: op(a.get(base, 0), b.get(base, 0)) for base in set(a) | set(b)})

    def __mul__(cls, other):
        if not isinstance(other, Dimension):
            return NotImplemented
        return cls._binop(operator.add, cls.__powers, other.__powers)

    def __truediv__(cls, other):
        if not isinstance(other, Dimension):
            return NotImplemented
        return cls._binop(operator.sub, cls.__powers, other.__powers)

    def __pow__(cls, other):
        try:
            # Fraction supports only a fixed set of input types, so to extend
            # this we first see if we can convert the argument to integer.
            other = other.__index__()
        except:
            pass
        return Dimension.from_powers({base: power*fractions.Fraction(other) for base, power in cls.__powers.items()})

    def __stringly_loads__(cls, s):
        return cls(s)

    def __stringly_dumps__(cls, v):
        try:
            return v._parsed_from
        except AttributeError:
            raise NotImplementedError

    def __call__(cls, value):
        if cls is Quantity:
            raise Exception('Quantity base class cannot be instantiated')
        if isinstance(value, cls):
            return value
        if not isinstance(value, str):
            raise ValueError(f'expected a str, got {type(value).__name__}')
        q = parse(value)
        expect = float if not cls.__powers else cls
        if type(q) != expect:
            raise TypeError(f'expected {expect.__name__}, got {type(q).__name__}')
        return q

    def wrap(cls, value):
        '''Wrap a numerical value in a Quantity container.

        The value must represent a multiple of the relevant reference quantity.
        See :func:`Quantity.unwrap` for the reverse operation.

        .. warning::
            Wrap and unwrap are advanced methods that allow you to bypass unit
            checks. Use with caution!
        '''

        if not cls.__powers:
            return value
        return super().__call__(value)


def parse(s):
    if not isinstance(s, str):
        raise ValueError(f'expected a str, received {type(s).__name__}')
    tail = s.lstrip('+-0123456789.')
    q = float(s[:len(s)-len(tail)] or 1)
    for expr, power, isnumer in _split_factors(tail):
        u = expr.lstrip('+-0123456789.')
        try:
            v = float(expr[:len(expr)-len(u)] or 1) * getattr(units, u)**power
        except (ValueError, AttributeError):
            raise ValueError(f'invalid (sub)expression {expr!r}') from None
        q = q * v if isnumer else q / v
    if isinstance(q, Quantity):
        q._parsed_from = s
    return q


class Quantity(metaclass=Dimension):

    def __init__(self, value):
        self.__value = value

    def __getnewargs__(self):
        return self.__value,

    def unwrap(self):
        '''Unwrap a numerical value inside a Quantity container.

        The value represents a multiply of the relevant reference quantity.
        See :func:`Dimension.wrap` for the reverse operation. For any quantity
        ``q`` it holds that ``q == type(q).wrap(q.unwrap())``.

        .. warning::
            Wrap and unwrap are advanced methods that allow you to bypass unit
            checks. Use with caution!
        '''

        return self.__value

    def __bool__(self):
        return bool(self.__value)

    def __len__(self):
        return len(self.__value)

    def __iter__(self):
        return map(type(self).wrap, self.__value)

    def __format__(self, format_spec):
        if not format_spec:
            return repr(self)
        n = len(format_spec) - len(format_spec.lstrip('0123456789.,'))
        v = self / type(self)(format_spec[n:])
        return v.__format__(format_spec[:n]+'f') + format_spec[n:]

    def __repr__(self):
        return repr(self.__value) + type(self).__name__

    def __str__(self):
        return str(self.__value) + type(self).__name__

    def __hash__(self):
        return hash((type(self), self.__value))

    @staticmethod
    def __unpack(*args):
        unpacked_any = False
        for arg in args:
            if isinstance(arg, Quantity):
                yield type(arg), arg.__value
                unpacked_any = True
            else:
                yield Dimensionless, arg
        assert unpacked_any, 'no dimensional quantities found'

    __DISPATCH_TABLE = {}

    ## POPULATE DISPATCH TABLE

    def register(*names, __table=__DISPATCH_TABLE):
        assert not any(name in __table for name in names)
        return lambda f: __table.update(dict.fromkeys(names, f))

    @register('neg', 'negative', 'pos', 'positive', 'abs', 'absolute', 'sum',
              'trace', 'ptp', 'amax', 'amin', 'max', 'min', 'mean', 'take',
              'broadcast_to', 'transpose', 'getitem', 'opposite', 'jump',
              'replace_arguments', 'linearize', 'derivative', 'integral',
              'sample', 'scatter', 'kronecker')
    def __unary(op, *args, **kwargs):
        (dim0, arg0), = Quantity.__unpack(args[0])
        return dim0.wrap(op(arg0, *args[1:], **kwargs))

    @register('add', 'sub', 'subtract', 'hypot', 'mod', 'maximum', 'minimum')
    def __add_like(op, *args, **kwargs):
        (dim0, arg0), (dim1, arg1) = Quantity.__unpack(args[0], args[1])
        if dim0 != dim1:
            raise TypeError(f'incompatible arguments for {op.__name__}: {dim0.__name__}, {dim1.__name__}')
        return dim0.wrap(op(arg0, arg1, *args[2:], **kwargs))

    @register('mul', 'multiply', 'matmul')
    def __mul_like(op, *args, **kwargs):
        (dim0, arg0), (dim1, arg1) = Quantity.__unpack(args[0], args[1])
        return (dim0 * dim1).wrap(op(arg0, arg1, *args[2:], **kwargs))

    @register('truediv', 'true_divide', 'divide', 'grad', 'surfgrad', 'div',
              'curl')
    def __div_like(op, *args, **kwargs):
        (dim0, arg0), (dim1, arg1) = Quantity.__unpack(args[0], args[1])
        return (dim0 / dim1).wrap(op(arg0, arg1, *args[2:], **kwargs))

    @register('laplace')
    def __laplace(op, *args, **kwargs):
        (dim0, arg0), (dim1, arg1) = Quantity.__unpack(args[0], args[1])
        return (dim0 / dim1**2).wrap(op(arg0, arg1, *args[2:], **kwargs))

    @register('sqrt')
    def __sqrt(op, *args, **kwargs):
        (dim0, arg0), = Quantity.__unpack(args[0])
        return (dim0**fractions.Fraction(1,2)).wrap(op(arg0, *args[1:], **kwargs))

    @register('setitem')
    def __setitem(op, *args, **kwargs):
        (dim0, arg0), (dim2, arg2) = Quantity.__unpack(args[0], args[2])
        if dim0 != dim2:
            raise TypeError(f'cannot assign {dim2.__name__} to {dim0.__name__}')
        return dim0.wrap(op(arg0, args[1], arg2, *args[3:], **kwargs))

    @register('pow', 'power', 'jacobian')
    def __pow_like(op, *args, **kwargs):
        (dim0, arg0), = Quantity.__unpack(args[0])
        return (dim0**args[1]).wrap(op(arg0, *args[1:], **kwargs))

    @register('isfinite', 'isnan', 'shape', 'ndim', 'size', 'normal', 'normalized')
    def __unary_drop(op, *args, **kwargs):
        (_dim0, arg0), = Quantity.__unpack(args[0])
        return op(arg0, *args[1:], **kwargs)

    @register('lt', 'le', 'eq', 'ne', 'gt', 'ge', 'equal', 'not_equal', 'less',
              'less_equal', 'greater', 'greater_equal')
    def __binary_drop(op, *args, **kwargs):
        (dim0, arg0), (dim1, arg1) = Quantity.__unpack(args[0], args[1])
        if dim0 != dim1:
            raise TypeError(f'incompatible arguments for {op.__name__}: {dim0.__name__}, {dim1.__name__}')
        return op(arg0, arg1, *args[2:], **kwargs)

    @register('stack', 'concatenate')
    def __stack_like(op, *args, **kwargs):
        dims, arg0 = zip(*Quantity.__unpack(*args[0]))
        if any(dim != dims[0] for dim in dims[1:]):
            raise TypeError(f'incompatible arguments for {op.__name__}: ' + ', '.join(dim.__name__ for dim in dims))
        return dims[0].wrap(op(arg0, *args[1:], **kwargs))

    @register('curvature')
    def __evaluate(op, *args, **kwargs):
        (dim0, arg0), = Quantity.__unpack(args[0])
        return (dim0**-1).wrap(op(*args, **kwargs))

    @register('evaluate')
    def __evaluate(op, *args, **kwargs):
        dims, args = zip(*Quantity.__unpack(*args))
        return tuple(dim.wrap(ret) for (dim, ret) in zip(dims, op(*args, **kwargs)))

    @register('dotarg')
    def __dotarg(op, *args, **kwargs):
        dims, args = zip(*Quantity.__unpack(*args)) # we abuse the fact that unpack str returns dimensionless
        return functools.reduce(operator.mul, dims).wrap(op(*args, **kwargs))

    del register

    ## DEFINE OPERATORS

    def op(name, with_reverse=False, *, __table=__DISPATCH_TABLE):
        dispatch = __table[name]
        op = getattr(operator, name)
        ret = lambda *args: dispatch(op, *args)
        if with_reverse:
            ret = ret, lambda self, other: dispatch(op, other, self)
        return ret

    __getitem__ = op('getitem')
    __setitem__ = op('setitem')
    __neg__ = op('neg')
    __pos__ = op('pos')
    __abs__ = op('abs')
    __lt__ = op('lt')
    __le__ = op('le')
    __eq__ = op('eq')
    __ne__ = op('ne')
    __gt__ = op('gt')
    __ge__ = op('ge')
    __add__, __radd__ = op('add', True)
    __sub__, __rsub__ = op('sub', True)
    __mul__, __rmul__ = op('mul', True)
    __matmul__, __rmatmul__ = op('matmul', True)
    __truediv, __rtruediv__ = op('truediv', True)
    __mod__, __rmod__ = op('mod', True)
    __pow__, __rpow__ = op('pow', True)

    def __truediv__(self, other):
        if type(other) is str:
            return self.__value / self.__class__(other).__value
        return self.__truediv(other)

    del op

    ## DISPATCH THIRD PARTY CALLS

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method != '__call__' or ufunc.__name__ not in self.__DISPATCH_TABLE:
            return NotImplemented
        return self.__DISPATCH_TABLE[ufunc.__name__](ufunc, *inputs, **kwargs)

    def __array_function__(self, func, types, args, kwargs):
        if func.__name__ not in self.__DISPATCH_TABLE:
            return NotImplemented
        return self.__DISPATCH_TABLE[func.__name__](func, *args, **kwargs)

    @classmethod
    def __nutils_dispatch__(cls, func, args, kwargs):
        if func.__name__ not in cls.__DISPATCH_TABLE:
            return NotImplemented
        return cls.__DISPATCH_TABLE[func.__name__](func, *args, **kwargs)


class Units(dict):

    __prefix = dict(Y=1e24, Z=1e21, E=1e18, P=1e15, T=1e12, G=1e9, M=1e6, k=1e3, h=1e2,
        d=1e-1, c=1e-2, m=1e-3, μ=1e-6, n=1e-9, p=1e-12, f=1e-15, a=1e-18, z=1e-21, y=1e-24)

    def __setattr__(self, name, value):
        if not isinstance(value, Quantity):
            if not isinstance(value, str):
                raise TypeError(f'can only assign Quantity or str, got {type(value).__name__}')
            value = parse(value)
        if name in self:
            raise ValueError(f'cannot define {name!r}: unit is already defined')
        scaled_units = {p + name: value * s for p, s in self.__prefix.items()}
        collisions = set(scaled_units) & set(self)
        if collisions:
            raise ValueError(f'cannot define {name!r}: unit collides with ' + ', '.join(collisions))
        self[name] = value
        self.update(scaled_units)

    def __getattr__(self, name):
        if name not in self:
            raise AttributeError(name)
        return self[name]


def _split_factors(s):
    for parts in s.split('*'):
        isnumer = True
        for factor in parts.split('/'):
            if factor:
                base = factor.rstrip('0123456789_')
                numer, sep, denom = factor[len(base):].partition('_')
                power = fractions.Fraction(int(numer or 1), int(denom or 1))
                yield base, power, isnumer
            isnumer = False


## SI DIMENSIONS

Dimensionless = Dimension.from_powers({})

Time = Dimension.create('T')
Length = Dimension.create('L')
Mass = Dimension.create('M')
ElectricCurrent = Dimension.create('I')
Temperature = Dimension.create('θ')
AmountOfSubstance = Dimension.create('N')
LuminousFlux = LuminousIntensity = Dimension.create('J')

Area = Length**2
Volume = Length**3
WaveNumber = Vergence = Length**-1
Velocity = Speed = Length / Time
Acceleration = Velocity / Time
Force = Weight = Mass * Acceleration
Pressure = Stress = Force / Area
Tension = Force / Length
Energy = Work = Heat = Force * Length
Power = Energy / Time
Density = Mass / Volume
SpecificVolume = MassConcentration = Density**-1
SurfaceDensity = Mass / Area
Viscosity = Pressure * Time
Frequency = Radioactivity = Time**-1
CurrentDensity = ElectricCurrent / Area
MagneticFieldStrength = ElectricCurrent / Length
Charge = ElectricCurrent * Time
ElectricPotential = Power / ElectricCurrent
Capacitance = Charge / ElectricPotential
Resistance = Impedance = Reactance = ElectricPotential / ElectricCurrent
Conductance = Resistance**-1
MagneticFlux = ElectricPotential * Time
MagneticFluxDensity = MagneticFlux / Area
Inductance = MagneticFlux / ElectricCurrent
Llluminance = LuminousFlux / Area
AbsorbedDose = EquivalentDose = Energy / Mass
Concentration = AmountOfSubstance / Volume
CatalyticActivity = AmountOfSubstance / Time


## SI UNITS

units = Units()

units.m = Length.wrap(1.)
units.s = Time.wrap(1.)
units.g = Mass.wrap(1e-3)
units.A = ElectricCurrent.wrap(1.)
units.K = Temperature.wrap(1.)
units.mol = AmountOfSubstance.wrap(1.)
units.cd = LuminousIntensity.wrap(1.)

units.N = 'kg*m/s2' # newton
units.Pa = 'N/m2' # pascal
units.J = 'N*m' # joule
units.W = 'J/s' # watt
units.Hz = '/s' # hertz
units.C = 'A*s' # coulomb
units.V = 'J/C' # volt
units.F = 'C/V' # farad
units.Ω = 'V/A' # ohm
units.S = '/Ω' # siemens
units.Wb = 'V*s' # weber
units.T = 'Wb/m2' # tesla
units.H = 'Wb/A' # henry
units.lm = 'cd' # lumen
units.lx = 'lm/m2' # lux
units.Bq = '/s' # becquerel
units.Gy = 'J/kg' # gray
units.Sv = 'J/kg' # sievert
units.kat = 'mol/s' # katal

units.min = '60s' # minute
units.h = '60min' # hour
units.day = '24h' # day
units.au = '149597870700m' # astronomical unit
units.ha = 'hm2' # hectare
units.L = 'dm3' # liter
units.t = '1000kg' # ton
units.Da = '1.66053904020yg' # dalton
units.eV = '.1602176634aJ' # electronvolt
units['in'] = 25.4 * units.mm # inch (no prefixes)
