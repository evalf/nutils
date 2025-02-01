from ._gmsh_stub import OCC, Mesh, Model, Tag, DimTags, Affine
from ._axes import Orientation
from ._util import overdimensioned
from ._field import set_background
from typing import Tuple, Dict, Optional, Iterable, Sequence, List, Mapping, Set
from abc import ABC, abstractmethod
import numpy, re


Tags = Set[Tag]
XY = Tuple[float,float]
XYZ = Tuple[float,float,float]
Fragments = Dict['Shape', Tuple[Tags, Dict[str, Tags]]]
Entities = Mapping[str,'Entity']


class Entity(ABC):

    def __init__(self, ndims: int):
        self.ndims = ndims

    @abstractmethod
    def get_shapes(self) -> Iterable['Shape']:
        ...

    @abstractmethod
    def select(self, fragments: Fragments) -> Tags:
        ...


class Shape(Entity):

    def __init__(self, ndims: int, periodicity: Iterable[Tuple[str,str,Affine]] = ()):
        self._periodicity = tuple(periodicity)
        super().__init__(ndims)

    def get_shapes(self):
        yield self

    def __sub__(self, other: 'Shape') -> 'BinaryOp':
        return BinaryOp(self, other, 'cut')

    def __and__(self, other: 'Shape') -> 'BinaryOp':
        return BinaryOp(self, other, 'intersect')

    def __or__(self, other: 'Shape') -> 'BinaryOp':
        return BinaryOp(self, other, 'fuse')

    def extruded(self, segments: Sequence[Tuple[float,float,float]], **orientation_kwargs) -> 'Pipe':
        '''Extruded 2D shape along 3D wire.

        The 2D `shape` is positioned in 3D space by translating to `origin` and
        rotating in the directions of `xaxis` and `yaxis`. The shape is
        subsequently extruded via a number of sections, each of which defines a
        length, an x-curvature and a y-curvature. If both curvatures are zero then
        the shape is linearly extruded over a distance of `length`. Otherwise, the
        vector (xcurv, ycurv) defines the axis of rotation in the 2D plane, and its
        length the curvature. Rotation follows the right-hand rule.'''

        orientation_kwargs.setdefault('origin', numpy.zeros(self.ndims+1))
        return Pipe(self, segments, Orientation(**orientation_kwargs))

    def revolved(self, angle: float = 360., **orientation_kwargs) -> 'Revolved':
        ''''Revolve 2D shape.

        The 2D `shape` is positioned in 3D space by translating to `origin` and
        rotating in the directions of `xaxis` and `yaxis`. The shape is
        subsequently rotated over its y-axis to form a (partially) revolved body.
        In case the rotation `angle` is less than 360 degrees then boundary groups
        'front' and 'back' are added to the 2D shape's existing boundaries;
        otherwise the revolved shape defines only the 2D boundary groups.'''

        orientation_kwargs.setdefault('origin', numpy.zeros(self.ndims+1))
        return Revolved(self, angle, Orientation(**orientation_kwargs))

    def select(self, fragments: Fragments) -> Tags:
        vtags, btags = fragments[self]
        return vtags

    def make_periodic(self, mesh: Mesh, btags: Dict[str, Tags]) -> None:
        for a, b, affinetrans in self._periodicity:
            mesh.setPeriodic(self.ndims-1, sorted(btags[a]), sorted(btags[b]), affinetrans)

    @property
    def boundary(self) -> 'Boundary':
        return Boundary(self)

    @abstractmethod
    def add_to(self, occ: OCC) -> DimTags:
        ...

    @abstractmethod
    def bnames(self, n: int) -> Iterable[str]:
        ...


class Interval(Shape):

    def __init__(self, left: Optional[float] = None, right: Optional[float] = None, center: Optional[float] = None, length: Optional[float] = None, periodic: bool = False):
        self.left, self.right, self.center, self.length = overdimensioned((left, 1, 0), (right, 0, 1), (center, .5, .5), (length, -1, 1))
        if self.length <= 0:
            raise ValueError('negative interval')
        self.periodic = periodic
        super().__init__(ndims=1)

    def bnames(self, n):
        assert n == 2
        return 'left', 'right'

    def add_to(self, occ: OCC) -> DimTags:
        p1, p2 = [occ.addPoint(x, 0, 0) for x in (self.left, self.right)]
        return (1, occ.addLine(p1, p2)),


class Point(Shape):

    def __init__(self, *p):
        self.p = p
        super().__init__(ndims=0)

    def bnames(self, n: int) -> Iterable[str]:
        assert n == 0
        return ()

    def add_to(self, occ: OCC) -> DimTags:
        p = occ.addPoint(*self.p, *[0]*(3-len(self.p)))
        return (0, p),


class Line(Shape):

    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2
        assert len(p1) == len(p2)
        super().__init__(ndims=1)

    def bnames(self, n):
        assert n == 2
        return 'left', 'right'

    def add_to(self, occ: OCC) -> DimTags:
        p1, p2 = [occ.addPoint(*p, *[0]*(3-len(p))) for p in (self.p1, self.p2)]
        return (self.ndims, occ.addLine(p1, p2)),


class Rectangle(Shape):
    'Rectangular domain'

    def __init__(self, x: Interval = Interval(0, 1), y: Interval = Interval(0, 1)):
        self.x = x.left
        self.dx = x.length
        self.y = y.left
        self.dy = y.length
        periodicity = [(b, a, Affine.shift(*d*iv.length)) for d, (a, iv, b) in zip(numpy.eye(3),
            [('left', x, 'right'), ('bottom', y, 'top')]) if iv.periodic]
        super().__init__(ndims=2, periodicity=periodicity)

    def bnames(self, n):
        assert n == 4
        return 'bottom', 'right', 'top', 'left'

    def add_to(self, occ: OCC) -> DimTags:
        return (2, occ.addRectangle(x=self.x, y=self.y, z=0., dx=self.dx, dy=self.dy)),


class Circle(Shape):
    'Circular domain'

    def __init__(self, center: XY = (0., 0.), radius: float = 1.):
        self.center = center
        self.radius = radius
        super().__init__(ndims=2)

    def bnames(self, n):
        assert n == 1
        yield 'wall'

    def add_to(self, occ: OCC) -> DimTags:
        return (2, occ.addDisk(*self.center, 0., rx=self.radius, ry=self.radius)),


class Ellipse(Shape):
    'Ellipsoidal domain'

    def __init__(self, center: XY = (0., 0.), width: float = 1., height: float = .5, angle: float = 0.):
        self.center = center
        self.width = width
        self.height = height
        self.angle = angle
        super().__init__(ndims=2)

    def bnames(self, n):
        assert n == 1
        yield 'wall'

    def add_to(self, occ: OCC) -> DimTags:
        height, width, angle = (self.height, self.width, self.angle) if self.width > self.height \
                          else (self.width, self.height, self.angle + 90)
        tag = occ.addDisk(*self.center, 0., rx=width/2, ry=height/2)
        occ.rotate([(2, tag)], *self.center, 0., ax=0., ay=0., az=1., angle=angle*numpy.pi/180)
        return (2, tag),


class Path(Shape):
    'Arbitrarily shaped domain enclosed by straight and curved boundary segments'

    def __init__(self, vertices: Tuple[XY,...], curvatures: Optional[Tuple[float,...]] = None):
        self.vertices = tuple(vertices)
        assert all(len(v) == 2 for v in self.vertices)
        self.curvatures = numpy.array(curvatures) if curvatures else numpy.zeros(len(self.vertices))
        assert len(self.curvatures) == len(self.vertices)
        super().__init__(ndims=2)

    def bnames(self, n):
        assert n == len(self.vertices)
        return [f'segment{i}' for i in range(n)]

    def add_to(self, occ: OCC) -> DimTags:
        points = [(v, occ.addPoint(*v, 0.)) for v in self.vertices]
        points.append(points[0])
        lines = [occ.addLine(p1, p2) if k == 0
            else occ.addCircleArc(p1, occ.addPoint(*self._center(v1, v2, 1/k), 0.), p2)
                for k, (v1, p1), (v2, p2) in zip(self.curvatures, points[:-1], points[1:])]
        loop = occ.addCurveLoop(lines)
        return (2, occ.addPlaneSurface([loop])),

    @staticmethod
    def _center(v1: XY, v2: XY, r: float, nudge: float = 1e-7) -> XY:
        cx, cy = numpy.add(v1, v2) / 2
        dx, dy = numpy.subtract(v1, v2) / 2
        r2 = dx**2 + dy**2
        D2 = r**2 / r2 - 1 + nudge
        assert D2 > 0, f'invalid arc radius: {r} < {numpy.sqrt(r2)}'
        D = numpy.copysign(numpy.sqrt(D2), r)
        return cx + dy * D, cy - dx * D,


class Box(Shape):
    'Box'

    def __init__(self, x: Interval = Interval(0, 1), y: Interval = Interval(0, 1), z: Interval = Interval(0, 1)):
        self.x = x.left
        self.dx = x.length
        self.y = y.left
        self.dy = y.length
        self.z = z.left
        self.dz = z.length
        periodicity = [(b, a, Affine.shift(*d*iv.length)) for d, (a, iv, b) in zip(numpy.eye(3),
            [('left', x, 'right'), ('bottom', y, 'top'), ('front', z, 'back')]) if iv.periodic]
        super().__init__(ndims=3, periodicity=periodicity)

    def bnames(self, n):
        assert n == 6
        return 'left', 'right', 'bottom', 'top', 'front', 'back'

    def add_to(self, occ: OCC) -> DimTags:
        return (3, occ.addBox(x=self.x, y=self.y, z=self.z, dx=self.dx, dy=self.dy, dz=self.dz)),


class Sphere(Shape):
    'Sphere'

    def __init__(self, center: XYZ = (0., 0., 0., ), radius: float = 1.):
        self.center = center
        self.radius = radius
        super().__init__(ndims=3)

    def bnames(self, n):
        assert n == 1
        return 'wall',

    def add_to(self, occ: OCC) -> DimTags:
        return (3, occ.addSphere(*self.center, self.radius)),


class Cylinder(Shape):
    'Cylinder'

    def __init__(self, front: XYZ = (0.,0.,0.), back: XYZ = (0.,0.,1.), radius: float = 1., periodic: bool = False):
        self.center = front
        self.axis = back[0] - front[0], back[1] - front[1], back[2] - front[2]
        self.radius = radius
        super().__init__(ndims=3, periodicity=[('back', 'front', Affine.shift(*self.axis))] if periodic else ())

    def bnames(self, n):
        assert n == 3
        return 'side', 'back', 'front'

    def add_to(self, occ: OCC) -> DimTags:
        return (3, occ.addCylinder(*self.center, *self.axis, self.radius)),


class BinaryOp(Shape):

    def __init__(self, shape1: Shape, shape2: Shape, op: str):
        self.shape1 = shape1
        self.shape2 = shape2
        self.op = op
        assert shape2.ndims == shape1.ndims
        super().__init__(shape1.ndims)

    def bnames(self, n):
        return [f'section{i}' for i in range(n)]

    def add_to(self, occ: OCC) -> DimTags:
        op = getattr(occ, self.op)
        return op(objectDimTags=self.shape1.add_to(occ), toolDimTags=self.shape2.add_to(occ))[0]


class Revolved(Shape):

    def __init__(self, shape: Shape, angle: float, orientation: Orientation):
        assert orientation.ndims == shape.ndims + 1
        self.shape = shape
        self.front = orientation
        self.angle = float(angle) * numpy.pi / 180
        super().__init__(ndims=orientation.ndims)

    def bnames(self, n):
        partial = self.angle < 2 * numpy.pi
        if partial:
            yield 'back'
            n -= 2
        for bname in self.shape.bnames(n):
            yield f'side-{bname}'
        if partial:
            yield 'front'

    def add_to(self, occ: OCC) -> DimTags:
        front = self.shape.add_to(occ)
        self.front.orient(occ, front)
        axes, origin = self.front.axes.as_3(self.front.origin)
        iaxis = {2: 2, 3: 1}[self.ndims] # TODO: allow variation of revolution axis in 3D
        dimtags = occ.revolve(front, *origin, *axes[iaxis], -self.angle)
        return [(dim, tag) for dim, tag in dimtags if dim == self.ndims]


class Pipe(Shape):

    def __init__(self, shape: Shape, segments: Sequence[Tuple[float,float,float]], orientation: Orientation):
        assert orientation.ndims == shape.ndims + 1
        self.shape = shape
        self.nsegments = len(segments)

        self.front = orientation
        vertices = [orientation.origin]
        midpoints: List[Optional[XYZ]] = []
        for length, *curvature in segments:
            if not any(curvature):
                midpoints.append(None)
                orientation = Orientation(orientation.origin + length * orientation.axes[-1], axes=orientation.axes)
            else:
                if shape.ndims == 1:
                    kx, = curvature
                    radius = numpy.array([-1/kx])
                    rotation = kx * length
                elif shape.ndims == 2:
                    kx, ky = curvature
                    radius = numpy.divide([ky, -kx], kx**2 + ky**2)
                    rotation = numpy.multiply(curvature, length) @ orientation.axes[:-1]
                else:
                    raise NotImplementedError
                center = orientation.origin + radius @ orientation.axes[:-1]
                midpoints.append(center)
                axes = orientation.axes.rotate(rotation)
                orientation = Orientation(center - radius @ axes[:-1], axes=axes)
            vertices.append(orientation.origin)
        self.midpoints = tuple(midpoints)
        self.vertices = tuple(vertices)
        self.back = orientation

        super().__init__(ndims=orientation.ndims)

    def bnames(self, n):
        nb, rem = divmod(n-2, self.nsegments)
        assert not rem
        if self.ndims == 2:
            bnames = [f'segment{i}-{bname}' for i in range(self.nsegments) for bname in self.shape.bnames(nb)]
            bnames.insert(1, 'front')
            bnames.append('back')
        elif self.ndims == 3:
            bnames = [f'segment{i}-{bname}' for bname in self.shape.bnames(nb) for i in range(self.nsegments)]
            bnames.insert(0, 'front')
            bnames.append('back')
        else:
            raise NotImplementedError
        return bnames

    def add_to(self, occ: OCC) -> DimTags:
        z = (0,) * (3 - self.ndims)
        points = [occ.addPoint(*v, *z) for v in self.vertices]
        segments = [occ.addLine(p1, p2) if v is None
               else occ.addCircleArc(p1, occ.addPoint(*v, *z), p2) for p1, v, p2 in zip(points, self.midpoints, points[1:])]
        wire_tag = occ.addWire(segments)
        front = self.shape.add_to(occ)
        self.front.orient(occ, front)
        return occ.addPipe(front, wire_tag)


class Boundary(Entity):

    def __init__(self, parent: Shape, patterns = ()):
        self.parent = parent
        self.patterns = patterns
        super().__init__(parent.ndims - 1)

    def __getitem__(self, item: str) -> 'Boundary':
        return Boundary(self.parent, (*self.patterns, re.compile(item)))

    def get_shapes(self):
        return self.parent.get_shapes()

    def select(self, fragments: Fragments) -> Tags:
        vtags, btags = fragments[self.parent]
        s = set.union(*[items for bname, items in btags.items() if all(pattern.fullmatch(bname) for pattern in self.patterns)])
        if not s:
            raise ValueError(f'{self.parent} does not have a boundary {", ".join(p.pattern for p in self.patterns)}')
        return s


class Skeleton(Entity):

    def get_shapes(self):
        return ()

    def select(self, fragments: Fragments) -> Tags:
        return set.union(*[items for vtags, btags in fragments.values() for items in btags.values()])


def generate_mesh(model: Model, mapping: Entities, elemsize) -> None:

    shapes = tuple(dict.fromkeys(shape for entity in mapping.values() for shape in entity.get_shapes())) # stable unique via dict
    shape_tags = [shape.add_to(model.occ) for shape in shapes] # create all shapes before sync

    model.occ.synchronize() # required for getBoundary

    objectDimTags: List[Tuple[int, Tag]] = []
    slices = []
    a = 0
    for dimtags in shape_tags:
        objectDimTags.extend(dimtags)
        b = len(objectDimTags)
        vslice = slice(a, b)
        objectDimTags.extend(model.getBoundary(dimtags, oriented=False))
        a = len(objectDimTags)
        bslice = slice(b, a)
        slices.append((vslice, bslice))
    _, fragment_map = model.occ.fragment(objectDimTags=objectDimTags, toolDimTags=[], removeObject=False)
    assert len(fragment_map) == a

    model.occ.synchronize()

    # setting fragment's removeObject=True has a tendency to remove (boundary)
    # entities that are still in use, so we remove unused entities manually
    # instead
    remove = set(objectDimTags)
    for dimtags in fragment_map:
        remove.difference_update(dimtags)
    if remove:
        model.removeEntities(sorted(remove))

    fragments = {}
    for shape, (vslice, bslice) in zip(shapes, slices):

        vtags = _tags([dimtag for dimtags in fragment_map[vslice] for dimtag in dimtags], shape.ndims)
        btagslist = [_tags(dimtags, shape.ndims-1) for dimtags in fragment_map[bslice]]

        btags = dict(zip(shape.bnames(len(btagslist)), btagslist))
        shape.make_periodic(model.mesh, btags)

        fragments[shape] = vtags, btags

    for name, entity in mapping.items():
        tag = model.addPhysicalGroup(entity.ndims, sorted(entity.select(fragments)))
        model.setPhysicalName(dim=entity.ndims, tag=tag, name=name)

    set_background(model.mesh.field, elemsize, fragments)

    model.mesh.generate(max(shape.ndims for shape in shapes))


def _tags(dimtags: DimTags, expect_dim: int) -> Tags:
    assert all(dim == expect_dim for dim, tag in dimtags)
    return {tag for dim, tag in dimtags}


# vim:sw=4:sts=4:et
