from typing import Union, Tuple, Set, Dict, Optional, Callable, Protocol, Iterable, NewType, NamedTuple, Sequence, List, Mapping, KeysView, cast
import numpy, re
from abc import ABC, abstractmethod

class Affine(NamedTuple):
    xx: float = 1.; xy: float = 0.; xz: float = 0.; dx: float = 0.
    yx: float = 0.; yy: float = 1.; yz: float = 0.; dy: float = 0.
    zx: float = 0.; zy: float = 0.; zz: float = 1.; dz: float = 0.
    u1: float = 0.; u2: float = 0.; u3: float = 0.; u4: float = 1.


# STUB FOR RELEVANT GMSH API

Tag = NewType('Tag', int)
DimTags = Sequence[Tuple[int, Tag]]

class OCC(Protocol):
    def addRectangle(self, x: float, y: float, z: float, dx: float, dy: float, tag: Tag = Tag(-1), roundedRadius: float = 0.) -> Tag:
        ...
    def addDisk(self, xc: float, yc: float, zc: float, rx: float, ry: float) -> Tag:
        ...
    def addPoint(self, x: float, y: float, z: float, meshSize: float = 0., tag: Tag = Tag(-1)) -> Tag:
        ...
    def addLine(self, startTag: Tag, endTag: Tag, tag: Tag = Tag(-1)) -> Tag:
        ...
    def addCircleArc(self, startTag: Tag, centerTag: Tag, endTag: Tag, tag: Tag = Tag(-1), nx: float = 0., ny: float = 0., nz: float = 0.) -> Tag:
        ...
    def addCurveLoop(self, curveTags: Sequence[Tag], tag: Tag = Tag(-1)) -> Tag:
        ...
    def addPlaneSurface(self, wireTags: Sequence[Tag], tag: Tag = Tag(-1)) -> Tag:
        ...
    def addBox(self, x: float, y: float, z: float, dx: float, dy: float, dz: float) -> Tag:
        ...
    def addCylinder(self, x: float, y: float, z: float, dx: float, dy: float, dz: float, r: float) -> Tag:
        ...
    def rotate(self, dimTags: DimTags, x: float, y: float, z: float, ax: float, ay: float, az: float, angle: float) -> Tag:
        ...
    def fuse(self, objectDimTags: DimTags, toolDimTags: DimTags, tag: Tag = Tag(-1), removeObject: bool = True, removeTool: bool = True) -> Tuple[DimTags, Sequence[DimTags]]:
        ...
    def revolve(self, dimTags: DimTags, x: float, y: float, z: float, ax: float, ay: float, az: float, angle: float) -> DimTags:
        ...
    def fragment(self, objectDimTags: DimTags, toolDimTags: Sequence[Tag], tag: Tag = Tag(-1), removeObject: bool = True, removeTool: bool = True) -> Tuple[Sequence[Tag],Sequence[DimTags]]:
        ...
    def synchronize(self) -> None:
        ...

class Mesh(Protocol):
    def setPeriodic(self, dim: int, tags: Sequence[Tag], tagsMaster: Sequence[Tag], affineTransform: Affine) -> None:
        ...
    def generate(self, dim: int = 3) -> None:
        ...

class Model(Protocol):
    occ: OCC
    mesh: Mesh
    def addPhysicalGroup(self, dim: int, tags: Sequence[Tag], tag: Tag = Tag(-1), name: str = '') -> Tag:
        ...
    def setPhysicalName(self, dim: int, tag: Tag, name: str) -> None:
        ...
    def getBoundary(self, dimTags: DimTags, combined: bool = True, oriented: bool = True, recursive: bool = False) -> DimTags:
        ...

# END STUB


Tags = Set[Tag]
Fragments = Dict[Union['PrimaryShape', Tuple['PrimaryShape',int]], Tags]
XY = Tuple[float,float]
XYZ = Tuple[float,float,float]
PrimaryShapes = Iterable['PrimaryShape']

def _shift(dx: float, dy: float, dz: float) -> Affine:
    return Affine(dx=dx, dy=dy, dz=dz)


class Shape(ABC):
    'Geometric shape with dimensions and positioned in a coordinate system.'

    def __init__(self, ndims: int):
        self.ndims = ndims
    def __sub__(self, other: 'Shape') -> 'Shape':
        return SetOp(self, other, set.__sub__)
    def __and__(self, other: 'Shape') -> 'Shape':
        return SetOp(self, other, set.__and__)
    def __or__(self, other: 'Shape') -> 'Shape':
        return SetOp(self, other, set.__or__)
    @abstractmethod
    def primary_shapes(self) -> PrimaryShapes:
        ...
    @abstractmethod
    def select(self, fragments: Fragments) -> Tags:
        ...


class SetOp(Shape):
    def __init__(self, a: Shape, b: Shape, op: Callable[[Tags, Tags], Tags]):
        assert a.ndims == b.ndims
        self.a = a
        self.b = b
        self.op = op
        super().__init__(a.ndims)
    def primary_shapes(self) -> PrimaryShapes:
        yield from self.a.primary_shapes()
        yield from self.b.primary_shapes()
    def select(self, fragments: Fragments) -> Tags:
        return self.op(self.a.select(fragments), self.b.select(fragments))


class PrimaryShape(Shape):
    'A CS building block that does not depend on other shapes.'

    def __init__(self, ndims: int, periodicity: Iterable[Tuple[str,str,Affine]] = ()):
        self._periodicity = tuple(periodicity)
        super().__init__(ndims)
    def primary_shapes(self) -> PrimaryShapes:
        yield self
    def select(self, fragments: Fragments) -> Tags:
        vtags, btags = fragments[self]
        return set(vtags)
    @property
    def boundary(self) -> 'Boundary':
        return Boundary(self)
    @abstractmethod
    def add_to(self, occ: OCC) -> DimTags:
        ...
    @abstractmethod
    def bnames(self, n: int) -> Iterable[str]:
        ...
    def make_periodic(self, mesh, btags):
        for a, b, affinetrans in self._periodicity:
            mesh.setPeriodic(self.ndims-1, btags[a], btags[b], affinetrans)


class Boundary(Shape):
    def __init__(self, parent: PrimaryShape, patterns = ()):
        self.parent = parent
        self.patterns = patterns
        super().__init__(parent.ndims-1)
    def __getitem__(self, item: str) -> Shape:
        return Boundary(self.parent, (*self.patterns, re.compile(item)))
    def primary_shapes(self) -> PrimaryShapes:
        yield from self.parent.primary_shapes()
    def select(self, fragments: Fragments) -> Tags:
        vtags, btags = fragments[self.parent]
        s = set()
        for bname, items in btags.items():
            if all(pattern.fullmatch(bname) for pattern in self.patterns):
                s.update(items)
        if not s:
            raise ValueError(f'{self.parent} does not have a boundary {", ".join(p.pattern for p in self.patterns)}')
        return s


def _overdimensioned(values, injection):
    selected = [i for i, v in enumerate(values) if v is not None]
    injection = numpy.array(injection)
    assert injection.ndim == 2 and injection.shape[0] == len(values) and injection.shape[1] < injection.shape[0]
    if len(selected) != injection.shape[1]:
        raise ValueError(f'exactly {len(injection)} arguments should be specified')
    return injection @ numpy.linalg.solve(injection[selected], [values[i] for i in selected])


class Interval(PrimaryShape):

    def __init__(self, left: float = None, right: float = None, center: float = None, length: float = None, periodic: bool = False):
        self.left, self.right, self.center, self.length = _overdimensioned([left, right, center, length], [[1, 0], [0, 1], [.5, .5], [-1, 1]])
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


class Rectangle(PrimaryShape):
    'Rectangular domain'

    def __init__(self, x: Interval = Interval(0, 1), y: Interval = Interval(0, 1)):
        self.x = x.left
        self.dx = x.length
        self.y = y.left
        self.dy = y.length
        periodicity = [(b, a, _shift(*d*iv.length)) for d, (a, iv, b) in zip(numpy.eye(3),
            [('left', x, 'right'), ('bottom', y, 'top')]) if iv.periodic]
        super().__init__(ndims=2, periodicity=periodicity)

    def bnames(self, n):
        assert n == 4
        return 'bottom', 'right', 'top', 'left'

    def add_to(self, occ: OCC) -> DimTags:
        return (2, occ.addRectangle(x=self.x, y=self.y, z=0., dx=self.dx, dy=self.dy)),


class Circle(PrimaryShape):
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


class Ellipse(PrimaryShape):
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


class Path(PrimaryShape):
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
            else occ.addCircleArc(p1, occ.addPoint(*_center(v1, v2, 1/k), 0.), p2)
                for k, (v1, p1), (v2, p2) in zip(self.curvatures, points[:-1], points[1:])]
        loop = occ.addCurveLoop(lines)
        return (2, occ.addPlaneSurface([loop])),


class Axes2:
    'representation of 2D, positive, orthonormal axes'

    @classmethod
    def eye(cls):
        return cls(0.)

    @classmethod
    def from_x(cls, xaxis):
        cosθ, sinθ = xaxis
        return cls(numpy.arctan2(sinθ, cosθ))

    def __init__(self, rotation: float):
        self._θ = rotation

    def __len__(self):
        return 2

    def __getitem__(self, s):
        sin = numpy.sin(self._θ)
        cos = numpy.cos(self._θ)
        return numpy.array([[cos, sin], [-sin, cos]][s])

    def rotate(self, angle):
        return Axes2(self._θ + angle)

    def as_3(self, origin):
        return Axes3.from_rotation_vector((0, 0, self._θ)), numpy.array((*origin, 0))


class Axes3:
    'representation of 3D, positive, orthonormal axes'
    # immutable representation of a unit quaternion, see https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation

    @classmethod
    def eye(cls):
        return cls(1., numpy.zeros(3))

    @classmethod
    def from_rotation_vector(cls, v: XYZ):
        v = numpy.asarray(v)
        θ = numpy.linalg.norm(v) / 2
        return cls(numpy.cos(θ), v * (-.5 * numpy.sinc(θ/numpy.pi)))

    @classmethod
    def from_xy(cls, xaxis, yaxis):
        Q = numpy.array([xaxis, yaxis, numpy.cross(xaxis, yaxis)])

        K = numpy.empty([4,4])
        K[1:,1:] = Q + Q.T
        K[0,1:] = K[1:,0] = Q[[2,0,1],[1,2,0]] - Q[[1,2,0],[2,0,1]]
        K[0,0] = 2 * numpy.trace(Q)

        eigval, eigvec = [v.T[-1] for v in numpy.linalg.eigh(K)]
        # is xaxis and yaxis are orthonormal then (eigval-K[0,0]/2)/3 == 1

        return cls(eigvec[0], eigvec[1:])

    def __init__(self, w: float, v: XYZ):
        self._w = w
        self._v = numpy.array(v)

    def __len__(self):
        return 3

    def __getitem__(self, s):
        I = numpy.eye(3)[s]
        return 2 * ((.5 - self._v @ self._v) * I + self._v[s, numpy.newaxis] * self._v + numpy.cross(I, self._w * self._v))

    def rotate(self, rotvec):
        other = Axes3.from_rotation_vector(rotvec)
        # hamilton product
        return Axes3(self._w * other._w - self._v @ other._v, self._w * other._v + other._w * self._v + numpy.cross(self._v, other._v))

    def as_3(self, origin):
        return self, origin

    @property
    def rotation_axis(self):
        return self._v

    @property
    def rotation_angle(self):
        return -2 * numpy.arctan2(numpy.linalg.norm(self._v), self._w)

    @property
    def rotation_vector(self):
        n = numpy.linalg.norm(self._v)
        return self._v * (n and -2 * numpy.arctan2(n, self._w) / n)


class Orientation:

    def __init__(self, origin, **kwargs):
        ndims = len(origin)
        args = ', '.join(sorted(kwargs))
        if args == 'axes':
            axes = kwargs['axes']
            assert len(axes) == ndims
        elif ndims == 2 and not args:
            axes = Axes2.eye()
        elif ndims == 3 and not args:
            axes = Axes3.eye()
        elif ndims == 2 and args == 'rotation':
            axes = Axes2(kwargs['rotation'])
        elif ndims == 2 and args == 'xaxis':
            axes = Axes2.from_x(kwargs['xaxis'])
        elif ndims == 3 and args == 'xaxis, yaxis':
            axes = Axes3.from_xy(kwargs['xaxis'], kwargs['yaxis'])
        elif ndims == 3 and args == 'rotvec':
            axes = Axes3.from_rotation_vector(kwargs['rotvec'])
        else:
            raise ValueError(f'cannot create {ndims}D orientation based on arguments {args}')
        self.origin = numpy.array(origin)
        self.ndims = ndims
        self.axes = axes

    def orient(self, occ: OCC, dimtags: DimTags):
        'position a 2D shape in 3D space'
        axes, origin = self.axes.as_3(self.origin)
        if origin.any():
            occ.translate(dimtags, *origin)
        if axes.rotation_angle:
            occ.rotate(dimtags, *origin, *axes.rotation_axis, -axes.rotation_angle)


class Box(PrimaryShape):
    'Box'

    def __init__(self, x: Interval = Interval(0, 1), y: Interval = Interval(0, 1), z: Interval = Interval(0, 1)):
        self.x = x.left
        self.dx = x.length
        self.y = y.left
        self.dy = y.length
        self.z = z.left
        self.dz = z.length
        periodicity = [(b, a, _shift(*d*iv.length)) for d, (a, iv, b) in zip(numpy.eye(3),
            [('left', x, 'right'), ('bottom', y, 'top'), ('front', z, 'back')]) if iv.periodic]
        super().__init__(ndims=3, periodicity=periodicity)

    def bnames(self, n):
        assert n == 6
        return 'left', 'right', 'bottom', 'top', 'front', 'back'

    def add_to(self, occ: OCC) -> DimTags:
        return (3, occ.addBox(x=self.x, y=self.y, z=self.z, dx=self.dx, dy=self.dy, dz=self.dz)),


class Sphere(PrimaryShape):
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


class Cylinder(PrimaryShape):
    'Cylinder'

    def __init__(self, front: XYZ = (0.,0.,0.), back: XYZ = (0.,0.,1.), radius: float = 1., periodic: bool = False):
        self.center = front
        self.axis = back[0] - front[0], back[1] - front[1], back[2] - front[2]
        self.radius = radius
        super().__init__(ndims=3, periodicity=[('back', 'front', _shift(*self.axis))] if periodic else ())

    def bnames(self, n):
        assert n == 3
        return 'side', 'back', 'front'

    def add_to(self, occ: OCC) -> DimTags:
        return (3, occ.addCylinder(*self.center, *self.axis, self.radius)),


class Fused(PrimaryShape):
    'Fuse multiple shapes into one'

    def __init__(self, *shapes: PrimaryShape):
        self.shapes = shapes
        ndims = shapes[0].ndims
        assert all(shape.ndims == ndims for shape in shapes[1:])
        super().__init__(ndims)

    def bnames(self, n):
        return [f'section{i}' for i in range(n)]

    def add_to(self, occ: OCC) -> DimTags:
        dimtags = []
        for shape in self.shapes:
            dimtags.extend(shape.add_to(occ))
        return occ.fuse(objectDimTags=dimtags[:1], toolDimTags=dimtags[1:])[0]


class Revolved(PrimaryShape):
    ''''Revolve 2D shape.

    The 2D `shape` is positioned in 3D space by translating to `origin` and
    rotating in the directions of `xaxis` and `yaxis`. The shape is
    subsequently rotated over its y-axis to form a (partially) revolved body.
    In case the rotation `angle` is less than 360 degrees then boundary groups
    'front' and 'back' are added to the 2D shape's existing boundaries;
    otherwise the revolved shape defines only the 2D boundary groups.'''

    def __init__(self, shape: PrimaryShape, angle: float = 360., orientation: Optional[Orientation] = None):
        if orientation is None:
            orientation = Orientation(numpy.zeros(shape.ndims+1))
        else:
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


class Pipe(PrimaryShape):
    '''Extruded 2D shape along 3D wire.

    The 2D `shape` is positioned in 3D space by translating to `origin` and
    rotating in the directions of `xaxis` and `yaxis`. The shape is
    subsequently extruded via a number of sections, each of which defines a
    length, an x-curvature and a y-curvature. If both curvatures are zero then
    the shape is linearly extruded over a distance of `length`. Otherwise, the
    vector (xcurv, ycurv) defines the axis of rotation in the 2D plane, and its
    length the curvature. Rotation follows the right-hand rule.'''

    def __init__(self, shape: PrimaryShape, segments: Sequence[Tuple[float,float,float]], orientation: Optional[Orientation] = None):
        if orientation is None:
            orientation = Orientation(numpy.zeros(shape.ndims+1))
        else:
            assert orientation.ndims == shape.ndims + 1
        self.shape = shape
        self.nsegments = len(segments)

        self.front = orientation
        vertices = [orientation.origin]
        for length, *curvature in segments:
            if not any(curvature):
                vertices.append(None)
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
                vertices.append(center)
                axes = orientation.axes.rotate(rotation)
                orientation = Orientation(center - radius @ axes[:-1], axes=axes)
            vertices.append(orientation.origin)
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
        points = [occ.addPoint(*v, *z) if v is not None else None for v in self.vertices]
        segments = [occ.addLine(p1, p2) if pc is None else occ.addCircleArc(p1, pc, p2) for p1, pc, p2 in zip(points[0::2], points[1::2], points[2::2])]
        wire_tag = occ.addWire(segments)
        front = self.shape.add_to(occ)
        self.front.orient(occ, front)
        return occ.addPipe(front, wire_tag)


Shapes = Mapping[str,Shape]


def generate_mesh(model: Model, shapes: Shapes) -> None:

    dim, fragments = _get_fragments(model, [pshape for shape in shapes.values() for pshape in shape.primary_shapes()])

    for name, shape in shapes.items():
        tag = model.addPhysicalGroup(shape.ndims, sorted(shape.select(fragments)))
        model.setPhysicalName(dim=shape.ndims, tag=tag, name=name)

    model.mesh.generate(dim)


# HELPER FUNCTIONS


def _get_fragments(model: Model, shapes: PrimaryShapes) -> Tuple[int, Fragments]:

    unique_shapes = tuple(dict.fromkeys(shapes))
    ndims, = {shape.ndims for shape in unique_shapes}
    shape_tags = [shape.add_to(model.occ) for shape in unique_shapes] # create all shapes before sync

    model.occ.synchronize()

    objectDimTags = []
    vslices = []
    bslices = []
    a = 0
    for dimtags in shape_tags:
        assert all(dim == ndims for dim, tag in dimtags)
        objectDimTags.extend(dimtags)
        b = len(objectDimTags)
        vslices.append(slice(a, b))
        objectDimTags.extend(model.getBoundary(dimtags, oriented=False))
        a = len(objectDimTags)
        bslices.append(slice(b, a))
    _, mapping = model.occ.fragment(objectDimTags=objectDimTags, toolDimTags=[], removeObject=False)
    assert len(mapping) == a

    model.occ.synchronize()

    # setting fragment's removeObject=True has a tendency to remove (boundary)
    # entities that are still in use, so we remove unused entities manually
    # instead
    remove = set(objectDimTags)
    for dimtags in mapping:
        remove.difference_update(dimtags)
    if remove:
        model.removeEntities(sorted(remove))

    shapevtags = []
    for s in vslices:
        dims, tags = zip(*[dimtag for dimtags in mapping[s] for dimtag in dimtags])
        assert all(d == ndims for d in dims)
        shapevtags.append(tags)

    shapebtags = []
    for s in bslices:
        dimslist, tagslist = zip(*[zip(*dimtags) for dimtags in mapping[s]])
        assert all(d == ndims-1 for dims in dimslist for d in dims)
        shapebtags.append(tagslist)

    fragments = {}
    for shape, vtags, btagslist in zip(unique_shapes, shapevtags, shapebtags):
        btags = dict(zip(shape.bnames(len(btagslist)), btagslist))
        shape.make_periodic(model.mesh, btags)
        fragments[shape] = vtags, btags

    return ndims, fragments


def _center(v1: XY, v2: XY, r: float, nudge: float = 1e-7) -> XY:
    cx, cy = numpy.add(v1, v2) / 2
    dx, dy = numpy.subtract(v1, v2) / 2
    r2 = dx**2 + dy**2
    D2 = r**2 / r2 - 1 + nudge
    assert D2 > 0, f'invalid arc radius: {r} < {numpy.sqrt(r2)}'
    D = numpy.copysign(numpy.sqrt(D2), r)
    return cx + dy * D, cy - dx * D,


# vim:sw=4:sts=4:et
