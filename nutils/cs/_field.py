from typing import Tuple, Union, Protocol, runtime_checkable, NewType
import math
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

    def gettag(self, ff: FieldFactory, fragments) -> str:
        ...
    def __add__(self, other: AsField) -> 'Field':
        return MathEval('({})+({})', self, as_field(other))
    def __radd__(self, other: AsField) -> 'Field':
        return MathEval('({})+({})', as_field(other), self)
    def __sub__(self, other: AsField) -> 'Field':
        return MathEval('({})-({})', self, as_field(other))
    def __rsub__(self, other: AsField) -> 'Field':
        return MathEval('({})-({})', as_field(other), self)
    def __mul__(self, other: AsField) -> 'Field':
        return MathEval('({})*({})', as_field(other), self)
    def __rmul__(self, other: AsField) -> 'Field':
        return MathEval('({})*({})', self, as_field(other))
    def __truediv__(self, other: AsField) -> 'Field':
        return MathEval('({})/({})', as_field(other), self)
    def __pow__(self, other: AsField) -> 'Field':
        return MathEval('({})^({})', as_field(other), self)


def as_field(f: AsField) -> Field:
    if isinstance(f, Field):
        return f
    if isinstance(f, (int, float)):
        return MathEval(str(f))
    raise ValueError(f'cannot interpret {f!r} as a field')


class MathEval(Field):

    def __init__(self, fmt, *args):
        self.fmt = fmt
        self.args = args

    def getexpr(self, ff: FieldFactory, fragments) -> str:
        return self.fmt.format(*[arg.getexpr(ff, fragments) if isinstance(arg, MathEval) else f'F{arg.gettag(ff, fragments)}' for arg in self.args])

    def gettag(self, ff: FieldFactory, fragments):
        expr = self.getexpr(ff, fragments)
        tag = ff.add('MathEval')
        print(tag, '->', expr)
        ff.setString(tag, 'F', expr)
        return tag


x = MathEval('x')
y = MathEval('y')
z = MathEval('z')


@dataclass(frozen=True)
class LocalRefinement(Field):
    'Refine elements according to a gaussian bell.'

    distance: Field
    radius: float
    factor: float = 2.

    # d = 0 -> X
    # d = radius -> Y
    # d = inf -> 1

    # scale = 1 - (1-X) * ((1-X)/(1-Y))^(-d^2/radius^2)

    # X = 1/factor
    # Y = 2/(factor+1)

    # (1-X)/(1-Y) = (1-1/factor)/(1-2/(factor+1)) = (factor+1)/factor = 1+1/factor

    # scale = 1 - (1-1/factor) * (1+1/factor)^(-d^2/radius^2)

    def gettag(self, ff: FieldFactory, fragments) -> str:
        c = 1 / self.factor
        d = self.distance.gettag(ff, fragments)
        expr = f'1-{1-c}*{1+c}^(-(({d})/{self.radius})^2)'
        #expr = f'1-.5*exp(-({d})^2)'
        #s = 1 - 1/self.factor
        #expr = f'1-{s}*exp(-(({d})/({self.radius}))^2)'
        print(expr)
        return expr


class Distance(Field):
    'Refine elements uniformly inside a circle'

    def __init__(self, *entities, sampling: int = 20):
        self.entities = entities
        self.sampling = sampling

    def gettag(self, ff: FieldFactory, fragments) -> str:
        tag = ff.add('Distance')
        ff.setNumber(tag, 'NumPointsPerCurve', round(self.sampling)) # gmsh 34a4d3c613
        #ff.setNumber(tag, 'Sampling', round(self.sampling))
        surfaces = set()
        curves = set()
        points = set()
        for entity in self.entities:
            tags = entity.select(fragments)
            if entity.ndims == 2:
                surfaces.update(tags)
            elif entity.ndims == 1:
                curves.update(tags)
            elif entity.ndims == 0:
                points.update(tags)
            else:
                bla
        if surfaces:
            ff.setNumbers(tag, 'SurfacesList', sorted(surfaces))
        if curves:
            ff.setNumbers(tag, 'CurvesList', sorted(curves))
        if points:
            ff.setNumbers(tag, 'PointsList', sorted(points))
        return tag


@dataclass
class Threshold(Field):

    d: Field
    dmin: float
    dmax: float
    vmin: float
    vmax: float
    sigmoid: bool = False

    def gettag(self, ff: FieldFactory, fragments):
        tag = ff.add('Threshold')
        ff.setNumber(tag, 'InField', self.d.gettag(ff, fragments))
        ff.setNumber(tag, 'DistMin', self.dmin)
        ff.setNumber(tag, 'DistMax', self.dmax)
        ff.setNumber(tag, 'SizeMin', self.vmin)
        ff.setNumber(tag, 'SizeMax', self.vmax)
        ff.setNumber(tag, 'Sigmoid', self.sigmoid)
        return tag


@dataclass(frozen=True)
class Ball(Field):
    'Refine elements uniformly inside a circle'

    center: Tuple[float,float]
    radius: float
    inside: float
    outside: float
    thickness: float = 0.

    def gettag(self, ff: FieldFactory, fragments) -> str:
        tag = ff.add('Ball')
        ff.setNumber(tag, 'Radius', self.radius)
        ff.setNumber(tag, 'Thickness', self.thickness)
        ff.setNumber(tag, 'VIn', self.inside)
        ff.setNumber(tag, 'VOut', self.outside)
        ff.setNumber(tag, 'XCenter', self.center[0])
        ff.setNumber(tag, 'YCenter', self.center[1])
        ff.setNumber(tag, 'ZCenter', 0)
        return tag


class Min(Field):

    def __init__(self, *fields):
        self.fields = fields

    def gettag(self, ff: FieldFactory, fragments) -> str:
        tags = [as_field(f).gettag(ff, fragments) for f in self.fields]
        tag = ff.add('Min')
        ff.setNumbers(tag, 'FieldsList', tags)
        return tag


class Max(Field):

    def __init__(self, *fields):
        self.fields = fields

    def gettag(self, ff: FieldFactory, fragments) -> str:
        tags = [as_field(f).gettag(ff, fragments) for f in self.fields]
        tag = ff.add('Max')
        ff.setNumbers(tag, 'FieldsList', tags)
        return tag


def set_background(ff: FieldFactory, elemsize: AsField, fragments) -> None:
    tag = as_field(elemsize).gettag(ff, fragments)
    ff.setAsBackgroundMesh(tag)


# vim:sw=4:sts=4:et
