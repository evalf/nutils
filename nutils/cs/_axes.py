from ._gmsh_stub import OCC, DimTags
from typing import Tuple, Any, Union
import numpy
import math

XY = Tuple[float,float]
XYZ = Tuple[float,float,float]


class Axes2:
    'representation of 2D, positive, orthonormal axes'

    @classmethod
    def eye(cls) -> 'Axes2':
        return cls(0.)

    @classmethod
    def from_x(cls, xaxis: XY) -> 'Axes2':
        cosθ, sinθ = xaxis
        return cls(math.atan2(sinθ, cosθ))

    def __init__(self, rotation: float):
        self._θ = rotation

    def __len__(self) -> int:
        return 2

    def __getitem__(self, s):
        sin = numpy.sin(self._θ)
        cos = numpy.cos(self._θ)
        return numpy.array([[cos, sin], [-sin, cos]][s])

    def rotate(self, angle: float) -> 'Axes2':
        return Axes2(self._θ + angle)

    def as_3(self, origin: XY) -> Tuple['Axes3', XYZ]:
        return Axes3.from_rotation_vector((0, 0, self._θ)), (*origin, 0)


class Axes3:
    'representation of 3D, positive, orthonormal axes'
    # immutable representation of a unit quaternion, see https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation

    @classmethod
    def eye(cls) -> 'Axes3':
        return cls(1., (0., 0., 0.))

    @classmethod
    def from_rotation_vector(cls, v: XYZ) -> 'Axes3':
        v_ = numpy.asarray(v)
        θ = numpy.linalg.norm(v_) / 2
        return cls(numpy.cos(θ), v_ * (-.5 * numpy.sinc(θ/numpy.pi)))

    @classmethod
    def from_xy(cls, xaxis: XYZ, yaxis: XYZ) -> 'Axes3':
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

    def __len__(self) -> int:
        return 3

    def __getitem__(self, s):
        I = numpy.eye(3)[s]
        return 2 * ((.5 - self._v @ self._v) * I + self._v[s, numpy.newaxis] * self._v + numpy.cross(I, self._w * self._v))

    def rotate(self, rotvec: XYZ) -> 'Axes3':
        other = Axes3.from_rotation_vector(rotvec)
        # hamilton product
        return Axes3(self._w * other._w - self._v @ other._v, self._w * other._v + other._w * self._v + numpy.cross(self._v, other._v))

    def as_3(self, origin: XYZ) -> Tuple['Axes3', XYZ]:
        return self, origin

    @property
    def rotation_axis(self) -> XYZ:
        return tuple(self._v)

    @property
    def rotation_angle(self) -> float:
        return -2 * math.atan2(numpy.linalg.norm(self._v), self._w)

    @property
    def rotation_vector(self) -> XYZ:
        n = numpy.linalg.norm(self._v)
        return self._v * (n and -2 * math.atan2(n, self._w) / n)


class Orientation:

    def __init__(self, origin: Union[XY, XYZ], **kwargs: Any):
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

    def orient(self, occ: OCC, dimtags: DimTags) -> None:
        'position a shape in 3D space'

        axes, origin = self.axes.as_3(self.origin)
        if any(origin):
            occ.translate(dimtags, *origin)
        if axes.rotation_angle:
            occ.rotate(dimtags, *origin, *axes.rotation_axis, -axes.rotation_angle)
