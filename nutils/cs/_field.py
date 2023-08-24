from typing import Tuple, Union, Protocol, runtime_checkable, NewType
from dataclasses import dataclass


# STUB FOR RELEVANT GMSH API

Tag = NewType('Tag', int)

class FieldFactory(Protocol):
    def add(self, name: str) -> Tag:
        ...
    def setString(self, tag: Tag, option: str, value: str) -> None:
        ...
    def setNumber(self, tag: Tag, option: str, value: Union[int, float]) -> None:
        ...
    def setAsBackgroundMesh(self, tag: Tag) -> None:
        ...

# END STUB


AsField = Union['Field',int,float]


@runtime_checkable
class Field(Protocol):
    'A scalar field over a coordinate system.'

    def getexpr(self, ff: FieldFactory) -> str:
        ...
    def __add__(self, other: AsField) -> 'Field':
        return FieldOp(self, as_field(other), '+')
    def __radd__(self, other: AsField) -> 'Field':
        return FieldOp(as_field(other), self, '+')
    def __sub__(self, other: AsField) -> 'Field':
        return FieldOp(self, as_field(other), '-')
    def __rsub__(self, other: AsField) -> 'Field':
        return FieldOp(as_field(other), self, '-')
    def __mul__(self, other: AsField) -> 'Field':
        return FieldOp(self, as_field(other), '*')
    def __rmul__(self, other: AsField) -> 'Field':
        return FieldOp(as_field(other), self, '*')
    def __truediv__(self, other: AsField) -> 'Field':
        return FieldOp(self, as_field(other), '/')


def as_field(f: AsField) -> Field:
    if isinstance(f, Field):
        return f
    if isinstance(f, (int, float)):
        return Constant(f)
    raise ValueError(f'cannot interpret {f!r} as a field')


@dataclass(frozen=True)
class FieldOp(Field):

    a: Field
    b: Field
    op: str

    def getexpr(self, ff: FieldFactory) -> str:
        a = self.a.getexpr(ff)
        b = self.b.getexpr(ff)
        return f'({a}){self.op}({b})'


@dataclass(frozen=True)
class Constant(Field):
    'Constant element size'

    size: float

    def getexpr(self, ff: FieldFactory) -> str:
        return str(self.size)


@dataclass(frozen=True)
class Coord(Field):

    coord: str

    def getexpr(self, ff: FieldFactory) -> str:
        return self.coord

x = Coord('x')
y = Coord('y')
z = Coord('z')


@dataclass(frozen=True)
class LocalRefinement(Field):
    'Refine elements according to a gaussian bell.'

    center: Tuple[float,float] = (0., 0.)
    radius: float = 1.
    factor: float = 2.

    def getexpr(self, ff: FieldFactory) -> str:
        x, y = self.center
        s = 1 - 1/self.factor
        return f'1-{s}*exp(-((x-{x})^2+(y-{y})^2)/({self.radius})^2)'


@dataclass(frozen=True)
class Ball(Field):
    'Refine elements uniformly inside a circle'

    center: Tuple[float,float]
    radius: float
    inside: float
    outside: float
    thickness: float = 0.

    def getexpr(self, ff: FieldFactory) -> str:
        tag = ff.add('Ball')
        ff.setNumber(tag, 'Radius', self.radius)
        ff.setNumber(tag, 'Thickness', self.thickness)
        ff.setNumber(tag, 'VIn', self.inside)
        ff.setNumber(tag, 'VOut', self.outside)
        ff.setNumber(tag, 'XCenter', self.center[0])
        ff.setNumber(tag, 'YCenter', self.center[1])
        ff.setNumber(tag, 'ZCenter', 0)
        return 'F{tag}'


@dataclass(frozen=True)
class Min(Field):

    f1: AsField
    f2: AsField

    def getexpr(self, ff: FieldFactory) -> str:
        s1 = as_field(self.f1).getexpr(ff)
        s2 = as_field(self.f2).getexpr(ff)
        return f'min({s1}, {s2})'


@dataclass(frozen=True)
class Max(Field):

    f1: AsField
    f2: AsField

    def getexpr(self, ff: FieldFactory) -> str:
        s1 = as_field(self.f1).getexpr(ff)
        s2 = as_field(self.f2).getexpr(ff)
        return f'max({s1}, {s2})'


def set_background(ff: FieldFactory, elemsize: AsField) -> None:
    F = as_field(elemsize).getexpr(ff)
    tag = ff.add('MathEval')
    ff.setString(tag, 'F', F)
    ff.setAsBackgroundMesh(tag)


# vim:sw=4:sts=4:et
