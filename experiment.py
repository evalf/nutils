from __future__ import annotations
from typing import Final
from dataclasses import dataclass
import numpy

@dataclass
class Transform:
  linear: numpy.ndarray
  offset: numpy.ndarray

class Shape:

  def __init__(self, dim: int) -> None:
    self.dim: Final[int] = dim

  @property
  def children(self) -> tuple[tuple[Shape, Transform], ...]:
    raise NotImplementedError

  @property
  def edges(self) -> tuple[tuple[Shape, Transform], ...]:
    raise NotImplementedError

  @property
  def connectivity(self) -> tuple[tuple[int, ...], ...]:
    raise NotImplementedError

class Point(Shape):

  def __init__(self) -> None:
    super().__init__(0)

  @property
  def children(self) -> tuple[tuple[Shape, Transform], ...]:
    return (self, Transform(numpy.eye(0), numpy.zeros((0,)))),

  @property
  def edges(self) -> tuple[tuple[Shape, Transform], ...]:
    raise ValueError("Point does not have edges.")

  @property
  def connectivity(self) -> tuple[tuple[int, ...], ...]:
    raise ValueError("Point does not have edges.")

class Line(Shape):

  def __init__(self) -> None:
    super().__init__(1)

  @property
  def children(self) -> tuple[tuple[Shape, Transform], ...]:
    return (self, Transform(numpy.eye(1) / 2, numpy.array([0.0]))), (self, Transform(numpy.eye(1) / 2, numpy.array([0.5])))

  @property
  def edges(self) -> tuple[tuple[Shape, Transform], ...]:
    return (Point(), Transform(numpy.zeros((1,0), float), numpy.array([0.0]))), (Point(), Transform(numpy.zeros((1,0), float), numpy.array([1.0])))

  @property
  def connectivity(self) -> tuple[tuple[int, ...], ...]:
    return (-1, 1), (0, -2)

def Simplex(dim: int) -> Shape:
  return (Point, Line)[dim]()

def _block_diagonal(a, b):
  r = numpy.zeros((a.shape[0] + b.shape[0], a.shape[1] + b.shape[1]), float)
  r[:a.shape[0], :a.shape[1]] = a
  r[a.shape[0]:, a.shape[1]:] = b
  return r

class ProductShape(Shape):

  def __init__(self, shape1: Shape, shape2: Shape) -> None:
    self._shape1 = shape1
    self._shape2 = shape2
    super().__init__(shape1.dim + shape2.dim)

  @property
  def children(self) -> tuple[tuple[Shape, Transform], ...]:
    return tuple((ProductShape(child1, child2), Transform(_block_diagonal(trans1.linear, trans2.linear), numpy.stack([trans1.offset, trans2.offset])))
                 for child1, trans1 in self._shape1.children
                 for child2, trans2 in self._shape1.children)

  @property
  def edges(self) -> tuple[tuple[Shape, Transform], ...]:
    return tuple((ProductShape(s1, s2), Transform(_block_diagonal(t1.linear, t2.linear), numpy.stack([t1.offset, t2.offset])))
                 for i in range(2)
                 for s1, t1 in (self._shape1.children if i == 0 else self._shape1.edge)
                 for s2, t2 in (self._shape2.edges if i == 0 else self._shape2.children))

class Root:

  def __init__(self, shapes: tuple[Shape, ...]) -> None:
    self.dim: Final[int] = shapes[0].dim
    self.shapes: Final[tuple[Shape, ...]] = shapes

  def __hash__(self) -> int:
    return hash(self.shapes)

  def __eq__(self, other) -> bool:
    return type(self) == type(other) and self.shapes == other.shapes

  def __len__(self) -> int:
    return len(self.shapes)

class Operator:

  def __init__(self, shapes: tuple[Shape, ...]) -> None:
    self.dim: Final[int] = shapes[0].dim
    self.shapes: Final[tuple[Shape, ...]] = shapes

  def __len__(self) -> int:
    return len(self.shapes)

  def get_parent_index_transform(self, index: int) -> tuple[int, Transform]:
    raise NotImplementedError

  def remove_head(self, operators: tuple[Operator, ...]) -> tuple[Operator, ...]:
    if operators and operators[0] == self:
      return operators[1:]
    else:
      raise ValueError(f'Cannot remove {self} from {operators}.')

class Children(Operator):

  def __init__(self, parent_shapes: tuple[Shape, ...]) -> None:
    self._parent_shapes = parent_shapes
    self._split_index = tuple((iparent, ichild) for iparent, shape in enumerate(parent_shapes) for ichild in range(len(shape.children)))
    super().__init__(tuple(child for shape in parent_shapes for child, transform in shape.children))

  def __hash__(self) -> int:
    return hash(self._parent_shapes)

  def __eq__(self, other) -> bool:
    return type(self) == type(other) and self._parent_shapes == other._parent_shapes

  def __repr__(self) -> str:
    return 'Children'

  def get_parent_index_transform(self, index: int) -> tuple[int, Transform]:
    iparent, ichild = self._split_index[index]
    return iparent, self._parent_shapes[iparent].children[ichild][1]

  def remove_head(self, operators: tuple[Operator, ...]) -> tuple[Operator, ...]:
    if operators and operators[0] == self:
      return operators[1:]
    elif len(operators) >= 2 and isinstance(operators[0], Edges) and isinstance(operators[1], Children):
      indices = []
      ichildedge = 0
      for shape in self._parent_shapes:
        local_indices = []
        for ichild, edges in enumerate(shape.connectivity):
          for i in edges:
            if i < 0:
              local_indices.append((-1 - i, ichildedge))
            ichildedge += 1
        local_indices.sort()
        assert tuple(iec for iec, ice in local_indices) == tuple(range(sum(len(edge.children) for edge, transform in shape.edges)))
        indices.extend(ice for iec, ice in local_indices)
      return Edges(self.shapes), Get(Edges(self.shapes).shapes, indices), *operators[2:]
    else:
      raise ValueError(f'Cannot remove {self} from {operators}.')

class Edges(Operator):

  def __init__(self, parent_shapes: tuple[Shape, ...]) -> None:
    self._parent_shapes = parent_shapes
    self._split_index = tuple((iparent, iedge) for iparent, shape in enumerate(parent_shapes) for iedge in range(len(shape.edges)))
    super().__init__(tuple(edge for shape in parent_shapes for edge, transform in shape.edges))

  def __hash__(self) -> int:
    return hash(self._parent_shapes)

  def __eq__(self, other) -> bool:
    return type(self) == type(other) and self._parent_shapes == other._parent_shapes

  def __repr__(self) -> str:
    return 'Edges'

  def get_parent_index_transform(self, index: int) -> tuple[int, Transform]:
    iparent, iedge = self._split_index[index]
    return iparent, self._parent_shapes[iparent].edges[iedge][1]

class Get(Operator):

  def __init__(self, parent_shapes: tuple[Shape, ...], indices: numpy.ndarray) -> None:
    self._parent_shapes = parent_shapes
    self._indices = indices
    super().__init__(tuple(parent_shapes[index] for index in indices))

  def __hash__(self) -> int:
    return hash((self._parent_shapes, self._indices))

  def __eq__(self, other) -> bool:
    return type(self) == type(other) and self._parent_shapes == other._parent_shapes and self._indices == other._indices

  def __repr__(self) -> str:
    return f'Get{self._indices}'

  def get_parent_index_transform(self, index: int) -> tuple[int, Transform]:
    return self._indices[index], Transform(numpy.eye(self.dim), numpy.zeros((self.dim,)))

class Sequence:

  def __init__(self, root: Root, *operators: Operator) -> None:
    self.root: Final[Root] = root
    self.operators: Final[tuple[Operator, ...]] = operators

  @property
  def shapes(self) -> tuple[Shape, ...]:
    if self.operators:
      return self.operators[-1].shapes
    else:
      return self.root.shapes

  def get_tail(self, reference: Sequence) -> tuple[Operators, ...]:
    if self.root != reference.root:
      raise ValueError
    tail = self.operators
    for head in reference.operators:
      tail = head.remove_head(tail)
    return tail

  @property
  def children(self) -> Sequence:
    return Sequence(self.root, *self.operators, Children(self.shapes))

  @property
  def edges(self) -> Sequence:
    return Sequence(self.root, *self.operators, Edges(self.shapes))

if __name__ == '__main__':
  root = Sequence(Root((Line(),)*4))
  print(root.edges.children.get_tail(root.children))
  print(root.edges.children.get_tail(root.children.edges))
