'''
Framework for physical units.

The unit class provides a basic framework for specifying values with physical
units using readable notation such as ``2.5km/h``. The system ensures that
values are consistent with a measurement system derived from base units, but
it does impose or even preload one such system. Instead, a derived class,
created using either :func:`create`, should specify the units and scales
relevant for the situation to which it is applied.

Once units are defined, the formal syntax for instantiating a quantity is:

.. code:: BNF

    <quantity> ::= <number> <units> | <number> <operator> <units>
    <number>   ::= "" | <integer> | <integer> "." <integer>
                ; Numerical value, allowing for decimal fractions but not
                ; scientific notation. An empty number is equivalent to 1.
    <units>    ::= <unit> | <unit> <operator> <units>
    <unit>     ::= <prefix> <name> <power>
    <prefix>   ::= "" | "h" | "k" | "M" | "G" | "T" | "P" | "E" | "Z" | "Y"
                | "d" | "c" | "m" | "μ" | "n" | "p" | "f" | "a" | "z" | "y"
                ; Single character prefix to indicate a multiple or fraction
                ; of the unit. All SI prefixes are supported except for deca.
                ; An empty prefix signifies no scaling.
    <name>     ::= <string>
                ; One of the defined units, case sensitive, containing Latin
                ; or Greek symbols.
    <power>    ::= "" | <integer>
                ; Integer power to which to raise the unit. An empty power is
                ; equivalent to 1.
    <operator> ::= "*" | "/"
                ; Multiplication or division.

With the prefix and unit name sharing an alphabet there is potential for
ambiguities (is it mol or micro-ol?). These are resolved using the simple
logic that the first character is considered part of the unit if this unit
exists; otherwise it is considered a prefix.
'''

import re


def create(_typename='unit', **units):
    '''
    Create new unit type.

    The unit system is defined via variable keyword arguments, with every unit
    specified either as a direct numerical value or as a string referencing
    other units using the standard expression syntax. Ultimately every unit
    should be resolvable to a numerical value by tracing its dependencies.

    The following example defines a subset of the SI system. Note that we
    cannot use prefixes on the receiving end of a definition for reasons of
    ambiguity, hence the definition of a gram as 1/1000:

    >>> SI = create(m=1, s=1, g=1e-3, N='kg*m/s2', Pa='N/m2')
    >>> SI('2km')
    2000.0
    >>> SI('2g')
    0.002

    Args
    ----
    name : :class:`str` (optional, positional only)
        Name of the new class object.
    **units :
        Unit definitions.

    Returns
    -------
    :
        The newly created (uninitiated) unit class.
    '''

    return _Unbound(_typename, (float,), dict(_parse=_Units(units).parse))


# INTERNAL HELPER FUNCTIONS


class _Unbound(type):
    'metaclass for unbound unit types'

    def __call__(cls, s):
        return cls[s.lstrip('1234567890.*')](s)

    def __getitem__(cls, s):
        if s.startswith('1234567890.*'):
            raise ValueError('unit cannot start with a numeral')
        return _Bound('{}:{}'.format(cls.__name__, s), (float,), dict(_parse=cls._parse, _unit=s))


class _Bound(type):
    'metaclass for bound unit types'

    def __call__(cls, s):
        return super().__call__(cls.__stringly_loads__(s))

    def __stringly_loads__(cls, s):
        q = cls._parse(s)
        powers = cls._parse(cls._unit).powers
        if q.powers != powers:
            raise ValueError('invalid unit: expected {}, got {}'.format(powers, q.powers))
        return q.value

    def __stringly_dumps__(cls, v):
        if not isinstance(v, (int, float)):
            raise ValueError('can only dump numerical values as unit, got {!r}'.format(type(v)))
        return _f2s(v / cls._parse(cls._unit).value) + cls._unit


class _Units:
    'minimal supporting object representing a collection of units'

    _words = re.compile('([a-zA-Zα-ωΑ-Ω]+)')
    _prefix = dict(Y=1e24, Z=1e21, E=1e18, P=1e15, T=1e12, G=1e9, M=1e6, k=1e3, h=1e2,
                   d=1e-1, c=1e-2, m=1e-3, μ=1e-6, n=1e-9, p=1e-12, f=1e-15, a=1e-18, z=1e-21, y=1e-24)

    def __init__(self, units):
        seen = {}

        def depth(name):
            if name not in units:
                name = name[1:]  # strip prefix
            if name not in seen:
                value = units.get(name)
                seen[name] = isinstance(value, str) and sum(map(depth, self._words.findall(value)), 1)
            return seen[name]
        self.quantities = {}
        for name in sorted(units, key=depth):  # sort by dependency depth to establish resolve order
            value = units[name]
            self.quantities[name] = self.parse(value) if isinstance(value, str) else _Quantity(value, {name: 1})

    def parse(self, s):
        parts = self._words.split(s)
        q = _Quantity(parts[0].rstrip('*/') or 1)
        for i in range(1, len(parts), 2):
            s = int(parts[i+1].rstrip('*/') or 1)
            if parts[i-1].endswith('/'):
                s = -s
            name = parts[i]
            if name not in self.quantities:
                if name[0] not in self._prefix or name[1:] not in self.quantities:
                    raise ValueError('unknown unit: {}'.format(name))
                q *= _Quantity(self._prefix[name[0]]**s)
                name = name[1:]
            q *= self.quantities[name]**s
        return q


class _Quantity:
    'minimal supporting object representing a dimensional number'

    def __init__(self, value, powers=()):
        self.value = float(value)
        self.powers = dict(powers)
        assert all(self.powers.values()), 'powers may not contain zeros'

    def __pow__(self, n):
        if not isinstance(n, int):
            return NotImplemented
        return self if n == 1 \
            else _Quantity(1) if n == 0 \
            else _Quantity(self.value**n, {k: v*n for k, v in self.powers.items()})

    def __imul__(self, other):
        if not isinstance(other, _Quantity):
            return NotImplemented
        self.value *= other.value
        for key, value in other.powers.items():
            value += self.powers.pop(key, 0)
            if value:
                self.powers[key] = value
        return self

    def __str__(self):
        return str(self.value) + ''.join(k + str(v) for k, v in sorted(self.powers.items()))


def _f2s(v):
    'convert float to string without scientific notation'

    s, sep, e = str(v).partition('e')
    a, sep, b = s.partition('.')
    pos = len(a) + int(e or 0)
    s = (a + b).rstrip('0')
    return s.ljust(pos, '0') if pos >= len(s) \
        else '0.' + '0' * -pos + s if pos <= 0 \
        else s[:pos] + '.' + s[pos:]


# vim:sw=4:sts=4:et
