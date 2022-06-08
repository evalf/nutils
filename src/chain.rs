use crate::simplex::Simplex;
use crate::types::Dim;
use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::iter;
use std::ops::Mul;

pub(crate) trait UnsizedSequence: Clone {
    /// Returns the difference between the dimension of the input and the output sequence.
    fn delta_dim(&self) -> Dim;
    fn delta_len(&self) -> (usize, usize);
    fn increment_offset(&mut self, amount: Dim);
    /// Increment the offset of the coordinate transformation, if applicable.
    fn decrement_offset(&mut self, amount: Dim);
    /// Map the index.
    fn apply_index(&self, index: usize) -> usize;
    /// Returns all indices that map to the given indices. TODO: examples with take and children?
    fn unapply_indices(&self, indices: &[usize]) -> Vec<usize>;
    /// Map the index and coordinate of an element in the output sequence to
    /// the input sequence. The index is returned, the coordinate is adjusted
    /// in-place. If the coordinate dimension of the input sequence is larger
    /// than that of the output sequence, the [Operator::delta_dim()] last
    /// elements of the coordinate are discarded.
    fn apply_one_inplace(&self, index: usize, coordinate: &mut [f64]) -> usize;
    /// Map the index and multiple coordinates of an element in the output
    /// sequence to the input sequence. The index is returned, the coordinates
    /// are adjusted in-place.
    fn apply_many_inplace(&self, index: usize, coordinates: &mut [f64], dim: Dim) -> usize {
        let dim = dim as usize;
        let mut result_index = 0;
        for i in 0..coordinates.len() / dim {
            result_index = self.apply_one_inplace(index, &mut coordinates[i * dim..(i + 1) * dim]);
        }
        result_index
    }
    /// Map the index and coordinate of an element in the output sequence to
    /// the input sequence.
    fn apply_one(&self, index: usize, coordinate: &[f64]) -> (usize, Vec<f64>) {
        let delta_dim = self.delta_dim() as usize;
        let to_dim = coordinate.len() + delta_dim;
        let mut result = Vec::with_capacity(to_dim);
        result.extend_from_slice(coordinate);
        result.extend(iter::repeat(0.0).take(delta_dim));
        (self.apply_one_inplace(index, &mut result), result)
    }
    /// Map the index and multiple coordinates of an element in the output
    /// sequence to the input sequence.
    fn apply_many(&self, index: usize, coordinates: &[f64], dim: Dim) -> (usize, Vec<f64>) {
        assert_eq!(coordinates.len() % dim as usize, 0);
        let ncoords = coordinates.len() / dim as usize;
        let delta_dim = self.delta_dim();
        let to_dim = dim + delta_dim;
        let mut result = Vec::with_capacity(ncoords * to_dim as usize);
        for coord in coordinates.chunks(dim as usize) {
            result.extend_from_slice(coord);
            result.extend(iter::repeat(0.0).take(delta_dim as usize));
        }
        (self.apply_many_inplace(index, &mut result, to_dim), result)
    }
}

#[derive(Debug, Clone, PartialEq)]
enum OperatorKind {
    Index(usize, usize),
    Coordinate(Dim, Dim, Dim, usize),
    Other,
}

trait DescribeOperator {
    /// Returns the kind of operation the operator applies to its parent sequence.
    fn operator_kind(&self) -> OperatorKind;
    #[inline]
    fn as_children(&self) -> Option<&Children> {
        None
    }
    #[inline]
    fn as_children_mut(&mut self) -> Option<&mut Children> {
        None
    }
    #[inline]
    fn as_edges(&self) -> Option<&Edges> {
        None
    }
    #[inline]
    fn as_edges_mut(&mut self) -> Option<&mut Edges> {
        None
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Offset(usize);

impl Offset {
    #[inline]
    pub fn new(offset: usize) -> Self {
        Self(offset)
    }
}

impl UnsizedSequence for Offset {
    #[inline]
    fn delta_dim(&self) -> Dim {
        0
    }
    #[inline]
    fn delta_len(&self) -> (usize, usize) {
        (1, 1)
    }
    #[inline]
    fn increment_offset(&mut self, _amount: Dim) {}
    #[inline]
    fn decrement_offset(&mut self, _amount: Dim) {}
    #[inline]
    fn apply_index(&self, index: usize) -> usize {
        index + self.0
    }
    #[inline]
    fn unapply_indices(&self, indices: &[usize]) -> Vec<usize> {
        indices.iter().filter_map(|i| i.checked_sub(self.0)).collect()
    }
    #[inline]
    fn apply_one_inplace(&self, index: usize, _coordinate: &mut [f64]) -> usize {
        self.apply_index(index)
    }
    #[inline]
    fn apply_many_inplace(&self, index: usize, _coordinate: &mut [f64], _dim: Dim) -> usize {
        self.apply_index(index)
    }
}

impl DescribeOperator for Offset {
    #[inline]
    fn operator_kind(&self) -> OperatorKind {
        OperatorKind::Other
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Transpose {
    len1: usize,
    len2: usize,
}

impl Transpose {
    #[inline]
    pub fn new(len1: usize, len2: usize) -> Self {
        Self { len1, len2 }
    }
}

impl UnsizedSequence for Transpose {
    #[inline]
    fn delta_dim(&self) -> Dim {
        0
    }
    #[inline]
    fn delta_len(&self) -> (usize, usize) {
        (1, 1)
    }
    #[inline]
    fn increment_offset(&mut self, _amount: Dim) {}
    #[inline]
    fn decrement_offset(&mut self, _amount: Dim) {}
    #[inline]
    fn apply_index(&self, index: usize) -> usize {
        let low2 = index % self.len2;
        let index = index / self.len2;
        let low1 = index % self.len1;
        let index = index / self.len1;
        index * (self.len1 * self.len2) + low2 * self.len1 + low1
    }
    #[inline]
    fn unapply_indices(&self, indices: &[usize]) -> Vec<usize> {
        indices.iter().map(|index| {
            let low1 = index % self.len1;
            let index = index / self.len1;
            let low2 = index % self.len2;
            let index = index / self.len2;
            index * (self.len1 * self.len2) + low1 * self.len2 + low2
        }).collect()
    }
    #[inline]
    fn apply_one_inplace(&self, index: usize, _coordinate: &mut [f64]) -> usize {
        self.apply_index(index)
    }
    #[inline]
    fn apply_many_inplace(&self, index: usize, _coordinate: &mut [f64], _dim: Dim) -> usize {
        self.apply_index(index)
    }
}

impl DescribeOperator for Transpose {
    #[inline]
    fn operator_kind(&self) -> OperatorKind {
        OperatorKind::Index(self.len1 * self.len2, self.len1 * self.len2)
    }
}

// TODO: add StrictMonotonicIncreasingTake

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Take {
    indices: Box<[usize]>,
    len: usize,
}

impl Take {
    #[inline]
    pub fn new(indices: impl Into<Box<[usize]>>, len: usize) -> Self {
        Self {
            indices: indices.into(),
            len,
        }
    }
}

impl UnsizedSequence for Take {
    #[inline]
    fn delta_dim(&self) -> Dim {
        0
    }
    #[inline]
    fn delta_len(&self) -> (usize, usize) {
        (self.indices.len(), self.len)
    }
    #[inline]
    fn increment_offset(&mut self, _amount: Dim) {}
    #[inline]
    fn decrement_offset(&mut self, _amount: Dim) {}
    #[inline]
    fn apply_index(&self, index: usize) -> usize {
        let nindices = self.indices.len();
        let low = index % nindices;
        let high = index / nindices;
        high * self.len + self.indices[low]
    }
    #[inline]
    fn unapply_indices(&self, indices: &[usize]) -> Vec<usize> {
        indices.iter().filter_map(|index| {
            let low = index % self.len;
            let high = index / self.len;
            self.indices.iter().position(|i| *i == low).map(|i| i + high * self.indices.len())
        }).collect()
    }
    #[inline]
    fn apply_one_inplace(&self, index: usize, _coordinate: &mut [f64]) -> usize {
        self.apply_index(index)
    }
    #[inline]
    fn apply_many_inplace(&self, index: usize, _coordinate: &mut [f64], _dim: Dim) -> usize {
        self.apply_index(index)
    }
}

impl DescribeOperator for Take {
    #[inline]
    fn operator_kind(&self) -> OperatorKind {
        OperatorKind::Index(self.len, self.indices.len())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Children {
    simplex: Simplex,
    offset: Dim,
}

impl Children {
    #[inline]
    pub fn new(simplex: Simplex, offset: Dim) -> Self {
        Self { simplex, offset }
    }
}

impl UnsizedSequence for Children {
    #[inline]
    fn delta_dim(&self) -> Dim {
        0
    }
    #[inline]
    fn delta_len(&self) -> (usize, usize) {
        (self.simplex.nchildren(), 1)
    }
    #[inline]
    fn increment_offset(&mut self, amount: Dim) {
        self.offset += amount;
    }
    #[inline]
    fn decrement_offset(&mut self, amount: Dim) {
        self.offset -= amount;
    }
    #[inline]
    fn apply_index(&self, index: usize) -> usize {
        self.simplex.apply_child_index(index)
    }
    #[inline]
    fn unapply_indices(&self, indices: &[usize]) -> Vec<usize> {
        self.simplex.unapply_child_indices(indices)
    }
    #[inline]
    fn apply_one_inplace(&self, index: usize, coordinate: &mut [f64]) -> usize {
        self.simplex
            .apply_child_inplace(index, &mut coordinate[self.offset as usize..])
    }
}

impl DescribeOperator for Children {
    #[inline]
    fn operator_kind(&self) -> OperatorKind {
        OperatorKind::Coordinate(
            self.offset,
            self.simplex.dim(),
            self.simplex.dim(),
            self.simplex.nchildren(),
        )
    }
    #[inline]
    fn as_children(&self) -> Option<&Children> {
        Some(self)
    }
    #[inline]
    fn as_children_mut(&mut self) -> Option<&mut Children> {
        Some(self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Edges {
    simplex: Simplex,
    offset: Dim,
}

impl Edges {
    #[inline]
    pub fn new(simplex: Simplex, offset: Dim) -> Self {
        Self { simplex, offset }
    }
}

impl UnsizedSequence for Edges {
    #[inline]
    fn delta_dim(&self) -> Dim {
        1
    }
    #[inline]
    fn delta_len(&self) -> (usize, usize) {
        (self.simplex.nedges(), 1)
    }
    #[inline]
    fn increment_offset(&mut self, amount: Dim) {
        self.offset += amount;
    }
    #[inline]
    fn decrement_offset(&mut self, amount: Dim) {
        self.offset -= amount;
    }
    #[inline]
    fn apply_index(&self, index: usize) -> usize {
        self.simplex.apply_edge_index(index)
    }
    #[inline]
    fn unapply_indices(&self, indices: &[usize]) -> Vec<usize> {
        self.simplex.unapply_edge_indices(indices)
    }
    #[inline]
    fn apply_one_inplace(&self, index: usize, coordinate: &mut [f64]) -> usize {
        self.simplex
            .apply_edge_inplace(index, &mut coordinate[self.offset as usize..])
    }
}

impl DescribeOperator for Edges {
    #[inline]
    fn operator_kind(&self) -> OperatorKind {
        OperatorKind::Coordinate(
            self.offset,
            self.simplex.dim(),
            self.simplex.edge_dim(),
            self.simplex.nedges(),
        )
    }
    #[inline]
    fn as_edges(&self) -> Option<&Edges> {
        Some(self)
    }
    #[inline]
    fn as_edges_mut(&mut self) -> Option<&mut Edges> {
        Some(self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(transparent)]
struct FiniteF64(pub f64);

impl Eq for FiniteF64 {}

impl PartialOrd for FiniteF64 {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl Ord for FiniteF64 {
    fn cmp(&self, other: &Self) -> Ordering {
        if let Some(ord) = self.0.partial_cmp(&other.0) {
            ord
        } else {
            panic!("not finite");
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct UniformPoints {
    points: Box<[FiniteF64]>,
    point_dim: Dim,
    offset: Dim,
}

impl UniformPoints {
    pub fn new(points: Box<[f64]>, point_dim: Dim, offset: Dim) -> Self {
        // TODO: assert that the points are actually finite.
        let points: Box<[FiniteF64]> = unsafe { std::mem::transmute(points) };
        Self {
            points,
            point_dim,
            offset,
        }
    }
    #[inline]
    pub const fn npoints(&self) -> usize {
        self.points.len() / self.point_dim as usize
    }
}

impl UnsizedSequence for UniformPoints {
    #[inline]
    fn delta_dim(&self) -> Dim {
        self.point_dim
    }
    #[inline]
    fn delta_len(&self) -> (usize, usize) {
        (self.npoints(), 1)
    }
    #[inline]
    fn increment_offset(&mut self, amount: Dim) {
        self.offset += amount;
    }
    #[inline]
    fn decrement_offset(&mut self, amount: Dim) {
        self.offset -= amount;
    }
    #[inline]
    fn apply_index(&self, index: usize) -> usize {
        index / self.npoints()
    }
    #[inline]
    fn unapply_indices(&self, indices: &[usize]) -> Vec<usize> {
        let mut point_indices = Vec::with_capacity(indices.len() * self.npoints());
        for index in indices.iter() {
            for ipoint in 0..self.npoints() {
                point_indices.push(index * self.npoints() + ipoint);
            }
        }
        point_indices
    }
    fn apply_one_inplace(&self, index: usize, coordinate: &mut [f64]) -> usize {
        let point_dim = self.point_dim as usize;
        let coordinate = &mut coordinate[self.offset as usize..];
        coordinate.copy_within(..coordinate.len() - point_dim, point_dim);
        let ipoint = index % self.npoints();
        let offset = ipoint as usize * point_dim;
        let points: &[f64] =
            unsafe { std::mem::transmute(&self.points[offset..offset + point_dim]) };
        coordinate[..point_dim].copy_from_slice(points);
        index / self.npoints()
    }
}

impl DescribeOperator for UniformPoints {
    #[inline]
    fn operator_kind(&self) -> OperatorKind {
        OperatorKind::Coordinate(self.offset, self.point_dim, 0, self.npoints())
    }
}

/// An operator that maps a sequence of elements to another sequence of elements.
///
/// Given a sequence of elements an [`Operator`] defines a new sequence. For
/// example [`Operator::Children`] gives the sequence of child elements and
/// [`Operator::Take`] gives a subset of the input sequence.
///
/// All variants of [`Operator`] apply some operation to either every element of
/// the parent sequence, variants [`Operator::Children`], [`Operator::Edges`]
/// and [`Operator::UniformPoints`], or to consecutive chunks of the input
/// sequence, in which case the size of the chunks is included in the variant
/// and the input sequence is assumed to be a multiple of the chunk size long.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum Operator {
    Offset(Offset),
    /// The transpose of a sequence: the input sequence is reshaped to `(_,
    /// len1, len2)`, the last two axes are swapped and the result is
    /// flattened.
    Transpose(Transpose),
    /// A subset of a sequence: the input sequence is reshaped to `(_, len)`,
    /// the given `indices` are taken from the last axis and the result is
    /// flattened.
    Take(Take),
    /// The children of a every element of a sequence.
    Children(Children),
    /// The edges of a every element of a sequence.
    Edges(Edges),
    UniformPoints(UniformPoints),
}

macro_rules! impl_from_for_operator {
    ($($Variant:ident),*) => {$(
        impl From<$Variant> for Operator {
            fn from(variant: $Variant) -> Self {
                Self::$Variant(variant)
            }
        }
    )*}
}

impl_from_for_operator! {Offset, Transpose, Take, Children, Edges, UniformPoints}

impl Operator {
    pub fn new_offset(offset: usize) -> Self {
        Offset::new(offset).into()
    }
    /// Construct a new operator that transposes a sequence of elements.
    pub fn new_transpose(len1: usize, len2: usize) -> Self {
        Transpose::new(len1, len2).into()
    }
    /// Construct a new operator that takes a subset of a sequence of elements.
    pub fn new_take(indices: impl Into<Box<[usize]>>, len: usize) -> Self {
        Take::new(indices, len).into()
    }
    /// Construct a new operator that maps a sequence of elements to its children.
    pub fn new_children(simplex: Simplex, offset: Dim) -> Self {
        Children::new(simplex, offset).into()
    }
    /// Construct a new operator that maps a sequence of elements to its edges.
    pub fn new_edges(simplex: Simplex, offset: Dim) -> Self {
        Edges::new(simplex, offset).into()
    }
    /// Construct a new operator that adds points to every element of a sequence.
    pub fn new_uniform_points(points: Box<[f64]>, point_dim: Dim, offset: Dim) -> Self {
        UniformPoints::new(points, point_dim, offset).into()
    }
    pub fn swap(&self, other: &Self) -> Option<Vec<Self>> {
        let mut other = other.clone();
        swap(self, &mut other).map(|tail| iter::once(other).chain(tail.into_iter()).collect())
    }
}

macro_rules! dispatch {
    ($vis:vis fn $fn:ident(&$self:ident $(, $arg:ident: $ty:ty)*) $($ret:tt)*) => {
        #[inline]
        $vis fn $fn(&$self $(, $arg: $ty)*) $($ret)* {
            match $self {
                Operator::Offset(var) => var.$fn($($arg),*),
                Operator::Transpose(var) => var.$fn($($arg),*),
                Operator::Take(var) => var.$fn($($arg),*),
                Operator::Children(var) => var.$fn($($arg),*),
                Operator::Edges(var) => var.$fn($($arg),*),
                Operator::UniformPoints(var) => var.$fn($($arg),*),
            }
        }
    };
    ($vis:vis fn $fn:ident(&mut $self:ident $(, $arg:ident: $ty:ty)*) $($ret:tt)*) => {
        #[inline]
        $vis fn $fn(&mut $self $(, $arg: $ty)*) $($ret)* {
            match $self {
                Operator::Offset(var) => var.$fn($($arg),*),
                Operator::Transpose(var) => var.$fn($($arg),*),
                Operator::Take(var) => var.$fn($($arg),*),
                Operator::Children(var) => var.$fn($($arg),*),
                Operator::Edges(var) => var.$fn($($arg),*),
                Operator::UniformPoints(var) => var.$fn($($arg),*),
            }
        }
    };
}

impl UnsizedSequence for Operator {
    dispatch! {fn delta_dim(&self) -> Dim}
    dispatch! {fn delta_len(&self) -> (usize, usize)}
    dispatch! {fn increment_offset(&mut self, amount: Dim)}
    dispatch! {fn decrement_offset(&mut self, amount: Dim)}
    dispatch! {fn apply_index(&self, index: usize) -> usize}
    dispatch! {fn unapply_indices(&self, indices: &[usize]) -> Vec<usize>}
    dispatch! {fn apply_one_inplace(&self, index: usize, coordinate: &mut [f64]) -> usize}
    dispatch! {fn apply_many_inplace(&self, index: usize, coordinates: &mut [f64], dim: Dim) -> usize}
    dispatch! {fn apply_one(&self, index: usize, coordinate: &[f64]) -> (usize, Vec<f64>)}
    dispatch! {fn apply_many(&self, index: usize, coordinates: &[f64], dim: Dim) -> (usize, Vec<f64>)}
}

impl DescribeOperator for Operator {
    dispatch! {fn operator_kind(&self) -> OperatorKind}
    dispatch! {fn as_children(&self) -> Option<&Children>}
    dispatch! {fn as_children_mut(&mut self) -> Option<&mut Children>}
    dispatch! {fn as_edges(&self) -> Option<&Edges>}
    dispatch! {fn as_edges_mut(&mut self) -> Option<&mut Edges>}
}

impl std::fmt::Debug for Operator {
    dispatch! {fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result}
}

fn swap<L, R>(l: &L, r: &mut R) -> Option<Vec<Operator>>
where
    L: DescribeOperator + UnsizedSequence + Into<Operator>,
    R: DescribeOperator + UnsizedSequence,
{
    if let (Some(edges), Some(children)) = (l.as_edges(), r.as_children_mut()) {
        if edges.offset == children.offset && edges.simplex.edge_dim() == children.simplex.dim() {
            let simplex = edges.simplex;
            let indices = simplex.swap_edges_children_map();
            let take = Operator::new_take(indices, simplex.nchildren() * simplex.nedges());
            children.simplex = simplex;
            return Some(vec![l.clone().into(), take]);
        }
    }
    use OperatorKind::*;
    match (l.operator_kind(), r.operator_kind()) {
        (Index(1, 1), Coordinate(_, _, _, _)) => Some(vec![l.clone().into()]),
        (Index(l_nout, l_nin), Coordinate(_, _, _, r_gen)) => Some(vec![
            Operator::new_transpose(r_gen, l_nout),
            l.clone().into(),
            Operator::new_transpose(l_nin, r_gen),
        ]),
        (Coordinate(l_off, _, l_nin, l_gen), Coordinate(r_off, r_nout, _, r_gen)) => {
            if l_off + l_nin <= r_off {
                r.increment_offset(l.delta_dim());
                Some(vec![
                    l.clone().into(),
                    Operator::new_transpose(l_gen, r_gen),
                ])
            } else if l_off >= r_off + r_nout {
                let mut l = l.clone();
                l.decrement_offset(r.delta_dim());
                Some(vec![l.into(), Operator::new_transpose(l_gen, r_gen)])
            } else {
                None
            }
        }
        _ => None,
    }
}

/// A chain of [`Operator`]s.
#[derive(Debug, Clone, PartialEq)]
pub struct UnsizedChain {
    rev_operators: Vec<Operator>,
}

impl UnsizedChain {
    #[inline]
    pub fn new<Operators>(operators: Operators) -> Self
    where
        Operators: IntoIterator<Item = Operator>,
        Operators::IntoIter: DoubleEndedIterator,
    {
        UnsizedChain {
            rev_operators: operators.into_iter().rev().collect(),
        }
    }
    #[inline]
    pub fn empty() -> Self {
        UnsizedChain {
            rev_operators: Vec::new(),
        }
    }
    pub fn push(&mut self, operator: impl Into<Operator>) {
        self.rev_operators.insert(0, operator.into())
    }
    /// Returns a clone of this [`UnsizedChain`] with the given `operator` appended.
    #[inline]
    pub fn clone_and_push(&self, operator: Operator) -> Self {
        Self::new(
            self.rev_operators
                .iter()
                .rev()
                .cloned()
                .chain(iter::once(operator)),
        )
    }
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = &Operator> + DoubleEndedIterator {
        self.rev_operators.iter().rev()
    }
    fn split_heads(&self) -> BTreeMap<Operator, Vec<Operator>> {
        let mut heads = BTreeMap::new();
        'a: for (i, head) in self.rev_operators.iter().enumerate() {
            let mut rev_tail: Vec<_> = self.rev_operators.iter().take(i).cloned().collect();
            let mut head = head.clone();
            for op in self.rev_operators.iter().skip(i + 1) {
                if let Some(ops) = swap(op, &mut head) {
                    rev_tail.extend(ops.into_iter().rev());
                } else {
                    continue 'a;
                }
            }
            heads.insert(head, rev_tail);
        }
        'b: for (i, op) in self.rev_operators.iter().enumerate() {
            if let Operator::Edges(Edges {
                simplex: Simplex::Line,
                offset,
            }) = op
            {
                let simplex = Simplex::Line;
                let mut rev_tail: Vec<_> = self.rev_operators.iter().take(i).cloned().collect();
                let mut head = Operator::new_children(simplex, *offset);
                let indices = simplex.swap_edges_children_map();
                let take = Operator::new_take(indices, simplex.nchildren() * simplex.nedges());
                rev_tail.push(take);
                rev_tail.push(op.clone());
                for op in self.rev_operators.iter().skip(i + 1) {
                    if let Some(ops) = swap(op, &mut head) {
                        rev_tail.extend(ops.into_iter().rev());
                    } else {
                        continue 'b;
                    }
                }
                heads.insert(head, rev_tail);
            }
        }
        heads
    }
    /// Remove and return the common prefix of two chains, transforming either if necessary.
    pub fn remove_common_prefix(&self, other: &Self) -> (Self, Self, Self) {
        let mut common = Vec::new();
        let mut rev_a = self.rev_operators.clone();
        let mut rev_b = other.rev_operators.clone();
        let mut i = 0;
        while !rev_a.is_empty() && !rev_b.is_empty() {
            i += 1;
            if i > 10 {
                break;
            }
            if rev_a.last() == rev_b.last() {
                common.push(rev_a.pop().unwrap());
                rev_b.pop();
                continue;
            }
            let heads_a = UnsizedChain::new(rev_a.iter().rev().cloned()).split_heads();
            let heads_b = UnsizedChain::new(rev_b.iter().rev().cloned()).split_heads();
            if let Some((head, a, b)) = heads_a
                .iter()
                .filter_map(|(h, a)| heads_b.get(h).map(|b| (h, a, b)))
                .min_by_key(|(_, a, b)| std::cmp::max(a.len(), b.len()))
            {
                common.push(head.clone());
                rev_a = a.clone();
                rev_b = b.clone();
                continue;
            }
            break;
        }
        let common = if rev_a.is_empty()
            && (!rev_b.is_empty() || self.rev_operators.len() <= other.rev_operators.len())
        {
            self.clone()
        } else if rev_b.is_empty() {
            other.clone()
        } else {
            Self::new(common)
        };
        (
            common,
            Self::new(rev_a.into_iter().rev()),
            Self::new(rev_b.into_iter().rev()),
        )
    }
}

impl UnsizedSequence for UnsizedChain {
    #[inline]
    fn delta_dim(&self) -> Dim {
        self.rev_operators.iter().map(|op| op.delta_dim()).sum()
    }
    #[inline]
    fn delta_len(&self) -> (usize, usize) {
        self.rev_operators.iter().rfold((1, 1), |(n, d), op| {
            let (opn, opd) = op.delta_len();
            (n * opn, d * opd)
        })
    }
    #[inline]
    fn increment_offset(&mut self, amount: Dim) {
        for op in self.rev_operators.iter_mut() {
            op.increment_offset(amount);
        }
    }
    #[inline]
    fn decrement_offset(&mut self, amount: Dim) {
        for op in self.rev_operators.iter_mut() {
            op.decrement_offset(amount);
        }
    }
    #[inline]
    fn apply_index(&self, index: usize) -> usize {
        self.rev_operators
            .iter()
            .fold(index, |index, op| op.apply_index(index))
    }
    #[inline]
    fn unapply_indices(&self, indices: &[usize]) -> Vec<usize> {
        let indices = indices.iter().cloned().collect();
        self.rev_operators
            .iter()
            .rev()
            .fold(indices, |indices, op| op.unapply_indices(&indices))
    }
    #[inline]
    fn apply_one_inplace(&self, index: usize, coordinate: &mut [f64]) -> usize {
        self.rev_operators
            .iter()
            .fold(index, |index, op| op.apply_one_inplace(index, coordinate))
    }
    #[inline]
    fn apply_many_inplace(&self, index: usize, coordinates: &mut [f64], dim: Dim) -> usize {
        self.rev_operators.iter().fold(index, |index, op| {
            op.apply_many_inplace(index, coordinates, dim)
        })
    }
}

impl IntoIterator for UnsizedChain {
    type Item = Operator;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.rev_operators.into_iter()
    }
}

impl FromIterator<Operator> for UnsizedChain {
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = Operator>,
    {
        let ops: Vec<_> = iter.into_iter().collect();
        UnsizedChain::new(ops)
    }
}

//#[derive(Debug, Clone)]
//struct AbsoluteChains {
//    tip_dim: Dim,
//    delta_dim: Dim,
//    chains: Vec<(usize, Chain)>,
//}
//
//#[derive(Debug, Clone, PartialEq)]
//enum NewAbsoluteChainsError {
//    DeltaDimensionMismatch,
//}
//
//impl std::error::Error for NewAbsoluteChainsError {}
//
//impl std::fmt::Display for NewAbsoluteChainsError {
//    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//        std::fmt::Debug::fmt(self, f)
//    }
//}
//
//impl AbsoluteChains {
//    pub fn new(
//        tip_dim: Dim,
//        delta_dim: Dim,
//        chains: Vec<(usize, Chain)>,
//    ) -> Result<Self, NewAbsoluteChainsError> {
//        if chains
//            .iter()
//            .any(|(_, chain)| chain.delta_dim() != delta_dim)
//        {
//            Err(NewAbsoluteChainsError::DeltaDimensionMismatch)
//        } else {
//            Ok(Self {
//                tip_dim,
//                delta_dim,
//                chains,
//            })
//        }
//    }
//    #[inline]
//    pub fn len(&self) -> usize {
//        self.chains.iter().map(|(len, _)| len).sum()
//    }
//    #[inline]
//    pub fn root_dim(&self) -> Dim {
//        self.tip_dim + self.delta_dim
//    }
//    #[inline]
//    pub fn tip_dim(&self) -> Dim {
//        self.tip_dim
//    }
//    fn get_chain(&self, mut index: usize) -> Option<(&Chain, usize)> {
//        for (len, chain) in self.chains.iter() {
//            if index < *len {
//                return Some((&chain, index));
//            }
//            index -= len;
//        }
//        None
//    }
//    #[inline]
//    pub fn apply_inplace(&self, index: usize, coordinate: &mut [f64]) -> Option<usize> {
//        self.get_chain(index)
//            .map(|(chain, index)| chain.apply_many_inplace(index, coordinate, self.root_dim()))
//    }
//    #[inline]
//    pub fn apply(&self, index: usize, coordinates: &[f64]) -> Option<(usize, Vec<f64>)> {
//        self.get_chain(index)
//            .map(|(chain, index)| chain.apply_many(index, coordinates, self.tip_dim()))
//    }
//}
//
//pub struct AbsoluteChain {
//    tip_dim: Dim,
//    delta_dim: Dim,
//    tip_len: usize,
//    chain: Chain,
//}
//
//pub struct RelativeChain {
//    tip_dim: Dim,
//    delta_dim: Dim,
//    order: Vec<usize>,
//    chains: Vec<usize, AbsoluteChain>,
//}
//
//impl RelativeChain {
//    pub fn get_chain(&self, index: usize) -> Option<(Chain, usize, usize)> {
//        for (offset, indices,
//    pub fn apply_inplace(&self, index: usize, coordinates: &mut [f64]) -> Option<usize> {
//        self.get_chain(index).map(|(chain, offset, index)| {
//            offset + chain.apply_many_inplace(index, coordinates, self.tip_dim())
//        })
//    }
//    pub fn apply(&self, index: usize, coordinates: &[f64]) -> Option<(usize, Vec<f64>)> {
//        self.get_chain(index).map(|(chain, offset, index)| {
//            offset + chain.apply_many(index, coordinates, self.tip_dim())
//        })
//    }
//}

#[derive(Debug, Clone)]
pub struct Topology {
    transforms: UnsizedChain,
    dim: Dim,
    root_len: usize,
    len: usize,
}

impl Topology {
    pub fn new(dim: Dim, len: usize) -> Self {
        Self {
            transforms: UnsizedChain::new([]),
            dim,
            root_len: len,
            len,
        }
    }
    pub fn derive(&self, operator: Operator) -> Self {
        let (n, d) = operator.delta_len();
        Self {
            root_len: self.root_len,
            len: self.len * n / d,
            dim: self.dim - operator.delta_dim(),
            transforms: self.transforms.clone_and_push(operator),
        }
    }
}

impl Mul for &Topology {
    type Output = Topology;

    fn mul(self, other: &Topology) -> Topology {
        Topology {
            transforms: UnsizedChain::new(
                iter::once(Operator::new_transpose(other.root_len, self.root_len))
                    .chain(self.transforms.iter().cloned())
                    .chain(iter::once(Operator::new_transpose(
                        self.len,
                        other.root_len,
                    )))
                    .chain(other.transforms.iter().map(|op| {
                        let mut op = op.clone();
                        op.increment_offset(self.dim);
                        op
                    })),
            ),
            dim: self.dim + other.dim,
            root_len: self.root_len * other.root_len,
            len: self.len * other.len,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use Simplex::*;

    #[test]
    fn unapply_indices_offset() {
        let offset = Offset::new(4);
        assert_eq!(offset.unapply_indices(&[3, 4, 6]), vec![0, 2]);
        assert_eq!(offset.unapply_indices(&[3]), vec![]);
    }

    #[test]
    fn unapply_indices_transpose() {
        let transpose = Transpose::new(2, 3);
        assert_eq!(transpose.unapply_indices(&[0, 1, 3, 8, 13, 14]), vec![0, 3, 4, 7, 15, 13]);
    }

    #[test]
    fn unapply_indices_take() {
        let take = Take::new([4, 1], 5);
        assert_eq!(take.unapply_indices(&[0, 1, 4, 5, 8, 9]), vec![1, 0, 2]);
        assert_eq!(take.unapply_indices(&[0]), vec![]);
    }

    #[test]
    fn unapply_indices_children() {
        let children = Children::new(Triangle, 0);
        assert_eq!(children.unapply_indices(&[0, 2]), vec![0, 1, 2, 3, 8, 9, 10, 11]);
    }

    #[test]
    fn unapply_indices_edges() {
        let edges = Edges::new(Triangle, 0);
        assert_eq!(edges.unapply_indices(&[0, 2]), vec![0, 1, 2, 6, 7, 8]);
    }

    #[test]
    fn unapply_indices_uniform_points() {
        let points = UniformPoints::new(Box::new([0.0, 0.0, 1.0, 0.0, 0.0, 1.0]), 2, 0);
        assert_eq!(points.unapply_indices(&[0, 2]), vec![0, 1, 2, 6, 7, 8]);
    }

    macro_rules! assert_eq_op_apply {
        ($op:expr, $ii:expr, $ic:expr, $oi:expr, $oc:expr) => {{
            let ic = $ic;
            let oc = $oc;
            let mut work = oc.clone();
            for i in 0..ic.len() {
                work[i] = ic[i];
            }
            for i in ic.len()..oc.len() {
                work[i] = 0.0;
            }
            assert_eq!($op.apply_one_inplace($ii, &mut work), $oi);
            assert_abs_diff_eq!(work[..], oc[..]);
        }};
    }

    #[test]
    fn apply_children_line() {
        let op = Operator::new_children(Line, 0);
        assert_eq_op_apply!(op, 0 * 2 + 0, [0.0], 0, [0.0]);
        assert_eq_op_apply!(op, 1 * 2 + 0, [1.0], 1, [0.5]);
        assert_eq_op_apply!(op, 2 * 2 + 1, [0.0], 2, [0.5]);
        assert_eq_op_apply!(op, 3 * 2 + 1, [1.0], 3, [1.0]);
        assert_eq_op_apply!(op, 0, [0.0, 2.0], 0, [0.0, 2.0]);
        assert_eq_op_apply!(op, 1, [0.0, 3.0, 4.0], 0, [0.5, 3.0, 4.0]);
        let op = Operator::new_children(Line, 1);
        assert_eq_op_apply!(op, 1, [2.0, 0.0], 0, [2.0, 0.5]);
        assert_eq_op_apply!(op, 1, [3.0, 0.0, 4.0], 0, [3.0, 0.5, 4.0]);
    }

    #[test]
    fn apply_edges_line() {
        let op = Operator::new_edges(Line, 0);
        assert_eq_op_apply!(op, 0, [], 0, [1.0]);
        assert_eq_op_apply!(op, 3, [], 1, [0.0]);
        assert_eq_op_apply!(op, 4, [], 2, [1.0]);
        assert_eq_op_apply!(op, 7, [], 3, [0.0]);
        assert_eq_op_apply!(op, 0, [2.0], 0, [1.0, 2.0]);
        assert_eq_op_apply!(op, 1, [3.0, 4.0], 0, [0.0, 3.0, 4.0]);
        let op = Operator::new_edges(Line, 1);
        assert_eq_op_apply!(op, 0, [2.0], 0, [2.0, 1.0]);
        assert_eq_op_apply!(op, 0, [3.0, 4.0], 0, [3.0, 1.0, 4.0]);
    }

    // #[test]
    // fn apply_edges_square() {
    //     let op = Operator::Edges {
    //         simplices: Box::new([Line, Line]),
    //         offset: 0,
    //     };
    //     assert_eq!(op.apply(0 * 4 + 0, &[0.0]), (0, vec![1.0, 0.0]));
    //     assert_eq!(op.apply(1 * 4 + 0, &[1.0]), (1, vec![1.0, 1.0]));
    //     assert_eq!(op.apply(2 * 4 + 1, &[0.0]), (2, vec![0.0, 0.0]));
    //     assert_eq!(op.apply(3 * 4 + 1, &[1.0]), (3, vec![0.0, 1.0]));
    //     assert_eq!(op.apply(4 * 4 + 2, &[0.0]), (4, vec![0.0, 1.0]));
    //     assert_eq!(op.apply(5 * 4 + 2, &[1.0]), (5, vec![1.0, 1.0]));
    //     assert_eq!(op.apply(6 * 4 + 3, &[0.0]), (6, vec![0.0, 0.0]));
    //     assert_eq!(op.apply(7 * 4 + 3, &[1.0]), (7, vec![1.0, 0.0]));
    //     assert_eq!(op.apply(0, &[0.0, 2.0]), (0, vec![1.0, 0.0, 2.0]));
    //     assert_eq!(op.apply(1, &[0.0, 3.0, 4.0]), (0, vec![0.0, 0.0, 3.0, 4.0]));
    // }

    #[test]
    fn apply_transpose_index() {
        let op = Operator::new_transpose(2, 3);
        for i in 0..3 {
            for j in 0..2 {
                for k in 0..3 {
                    assert_eq!(
                        op.apply_one((i * 2 + j) * 3 + k, &[]),
                        ((i * 3 + k) * 2 + j, vec![])
                    );
                }
            }
        }
    }

    #[test]
    fn apply_take_all() {
        let op = Operator::new_take([3, 2, 0, 4, 1], 5); // inverse: [2, 4, 1, 0, 3]
        assert_eq_op_apply!(op, 0, [], 3, []);
        assert_eq_op_apply!(op, 6, [1.0], 7, [1.0]);
        assert_eq_op_apply!(op, 12, [2.0, 3.0], 10, [2.0, 3.0]);
        assert_eq_op_apply!(op, 18, [], 19, []);
        assert_eq_op_apply!(op, 24, [], 21, []);
    }

    #[test]
    fn apply_take_some() {
        let op = Operator::new_take([4, 0, 1], 5); // inverse: [1, 2, x, x, 0]
        assert_eq_op_apply!(op, 0, [], 4, []);
        assert_eq_op_apply!(op, 4, [1.0], 5, [1.0]);
        assert_eq_op_apply!(op, 8, [2.0, 3.0], 11, [2.0, 3.0]);
    }

    #[test]
    fn apply_uniform_points() {
        let op = Operator::new_uniform_points(Box::new([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]), 2, 0);
        assert_eq_op_apply!(op, 0, [], 0, [0.0, 1.0]);
        assert_eq_op_apply!(op, 4, [6.0], 1, [2.0, 3.0, 6.0]);
        assert_eq_op_apply!(op, 8, [7.0, 8.0], 2, [4.0, 5.0, 7.0, 8.0]);
    }

    #[test]
    fn mul_topo() {
        let xtopo = Topology::new(1, 2).derive(Operator::new_children(Line, 0));
        let ytopo = Topology::new(1, 3).derive(Operator::new_children(Line, 0));
        let xytopo = &xtopo * &ytopo;
        assert_eq!(xtopo.len, 4);
        assert_eq!(ytopo.len, 6);
        assert_eq!(xytopo.len, 24);
        assert_eq!(xytopo.root_len, 6);
        for i in 0..4 {
            for j in 0..6 {
                let x = xtopo.transforms.apply_many(i, &[0.0, 0.0, 1.0, 1.0], 1).1;
                let y = ytopo.transforms.apply_many(j, &[0.0, 1.0, 0.0, 1.0], 1).1;
                let mut xy = Vec::with_capacity(8);
                for k in 0..4 {
                    xy.push(x[k]);
                    xy.push(y[k]);
                }
                assert_eq!(
                    xytopo.transforms.apply_many(
                        i * 6 + j,
                        &[0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
                        2
                    ),
                    ((i / 2) * 3 + j / 2, xy),
                );
            }
        }
    }

    macro_rules! assert_equiv_topo {
        ($topo1:expr, $topo2:expr$(, $simplex:ident)*) => {
            #[allow(unused_mut)]
            let mut topo1 = $topo1.clone();
            #[allow(unused_mut)]
            let mut topo2 = $topo2.clone();
            assert_eq!(topo1.dim, topo2.dim, "topos have different dim");
            assert_eq!(topo1.len, topo2.len, "topos have different len");
            assert_eq!(topo1.root_len, topo2.root_len, "topos have different root_len");
            let from_dim = 0$(+$simplex.dim())*;
            assert_eq!(topo1.dim, from_dim, "dimension of topo differs from dimension of given simplices");
            let nelems = topo1.len;
            $(
                let points = Operator::new_uniform_points(
                    $simplex.vertices().into(),
                    $simplex.dim(),
                    0,
                );
                topo1 = topo1.derive(points.clone());
                topo2 = topo2.derive(points.clone());
            )*
            let npoints = topo1.len;
            let mut coord1: Vec<_> = iter::repeat(0.0).take((topo1.dim + topo1.transforms.delta_dim()) as usize).collect();
            let mut coord2 = coord1.clone();
            for i in 0..topo1.len {
                let ielem = i / (npoints / nelems);
                assert_eq!(
                    topo1.transforms.apply_one_inplace(i, &mut coord1),
                    topo2.transforms.apply_one_inplace(i, &mut coord2),
                    "topo1 and topo2 map element {ielem} to different root elements"
                );
                assert_abs_diff_eq!(coord1[..], coord2[..]);
            }
        };
    }

    #[test]
    fn swap_edges_children_1d() {
        let topo1 = Topology::new(1, 3).derive(Operator::new_edges(Line, 0));
        let topo2 = Topology::new(1, 3)
            .derive(Operator::new_children(Line, 0))
            .derive(Operator::new_edges(Line, 0))
            .derive(Operator::new_take([2, 1], 4));
        assert_equiv_topo!(topo1, topo2);
    }

    #[test]
    fn swap_take_children() {
        let take = Operator::new_take([2, 3, 1], 5);
        let children = Operator::new_children(Line, 0);
        let swapped = vec![
            children.clone(),
            Operator::new_transpose(2, 5),
            take.clone(),
            Operator::new_transpose(3, 2),
        ];
        let base = Topology::new(1, 5);
        assert_eq!(take.swap(&children), Some(swapped.clone()));
        assert_equiv_topo!(
            base.derive(take).derive(children),
            swapped
                .iter()
                .cloned()
                .fold(base.clone(), |t, o| t.derive(o)),
            Line
        );
    }

    #[test]
    fn swap_take_edges() {
        let take = Operator::new_take([2, 3, 1], 5);
        let edges = Operator::new_edges(Line, 0);
        let swapped = vec![
            edges.clone(),
            Operator::new_transpose(2, 5),
            take.clone(),
            Operator::new_transpose(3, 2),
        ];
        let base = Topology::new(1, 5);
        assert_eq!(take.swap(&edges), Some(swapped.clone()));
        assert_equiv_topo!(
            base.derive(take).derive(edges),
            swapped
                .iter()
                .cloned()
                .fold(base.clone(), |t, o| t.derive(o))
        );
    }

    macro_rules! fn_test_operator_swap {
        ($name:ident, $len:expr $(, $simplex:ident)*; $op1:expr, $op2:expr,) => {
            #[test]
            fn $name() {
                let op1: Operator = $op1;
                let op2: Operator = $op2;
                let swapped = op1.swap(&op2).expect("not swapped");
                println!("op1: {op1:?}");
                println!("op2: {op2:?}");
                println!("swapped: {swapped:?}");
                let root_dim = op1.delta_dim() + op2.delta_dim() $(+ $simplex.dim())*;
                let base = Topology::new(root_dim, 1);
                let topo1 = [op1, op2].iter().fold(base.clone(), |t, o| t.derive(o.clone()));
                let topo2 = swapped.iter().fold(base, |t, o| t.derive(o.clone()));
                let len = $len;
                assert_eq!(topo1.len, len, "unswapped topo has unexpected length");
                assert_eq!(topo2.len, len, "swapped topo has unexpected length");
                assert_equiv_topo!(topo1, topo2 $(, $simplex)*);
            }
        }
    }

    fn_test_operator_swap! {
        swap_edges_children_triangle1, 6, Line, Line;
        Operator::new_edges(Triangle, 0),
        Operator::new_children(Line, 0),
    }

    fn_test_operator_swap! {
        swap_unoverlapping_children_lt_children, 8, Triangle, Line;
        Operator::new_children(Triangle, 0),
        Operator::new_children(Line, 2),
    }

    fn_test_operator_swap! {
        swap_unoverlapping_children_gt_children, 8, Line, Triangle;
        Operator::new_children(Line, 2),
        Operator::new_children(Triangle, 0),
    }

    fn_test_operator_swap! {
        swap_unoverlapping_edges_lt_children, 6, Line, Line;
        Operator::new_edges(Triangle, 0),
        Operator::new_children(Line, 1),
    }

    fn_test_operator_swap! {
        swap_unoverlapping_edges_gt_children, 6, Line, Line;
        Operator::new_edges(Triangle, 1),
        Operator::new_children(Line, 0),
    }

    fn_test_operator_swap! {
        swap_unoverlapping_children_lt_edges, 6, Line, Line;
        Operator::new_children(Line, 0),
        Operator::new_edges(Triangle, 1),
    }

    fn_test_operator_swap! {
        swap_unoverlapping_children_gt_edges, 6, Line, Line;
        Operator::new_children(Line, 2),
        Operator::new_edges(Triangle, 0),
    }

    fn_test_operator_swap! {
        swap_unoverlapping_edges_lt_edges, 6, Line;
        Operator::new_edges(Line, 0),
        Operator::new_edges(Triangle, 0),
    }

    fn_test_operator_swap! {
        swap_unoverlapping_edges_gt_edges, 6, Line;
        Operator::new_edges(Line, 2),
        Operator::new_edges(Triangle, 0),
    }

    #[test]
    fn split_heads() {
        let chain = UnsizedChain::new([
            Operator::new_edges(Triangle, 1),
            Operator::new_children(Line, 0),
            Operator::new_edges(Line, 2),
            Operator::new_children(Line, 1),
            Operator::new_children(Line, 0),
        ]);
        let desired = chain
            .iter()
            .cloned()
            .fold(Topology::new(4, 1), |topo, op| topo.derive(op));
        for (head, tail) in chain.split_heads().into_iter() {
            let actual = iter::once(head)
                .chain(tail.into_iter().rev())
                .fold(Topology::new(4, 1), |topo, op| topo.derive(op));
            assert_equiv_topo!(actual, desired, Line, Line);
        }
    }

    #[test]
    fn remove_common_prefix() {
        let a = UnsizedChain::new([
            Operator::new_children(Line, 0),
            Operator::new_children(Line, 0),
        ]);
        let b = UnsizedChain::new([Operator::new_edges(Line, 0)]);
        assert_eq!(
            a.remove_common_prefix(&b),
            (
                UnsizedChain::new([
                    Operator::new_children(Line, 0),
                    Operator::new_children(Line, 0)
                ]),
                UnsizedChain::new([]),
                UnsizedChain::new([
                    Operator::new_edges(Line, 0),
                    Operator::new_take([2, 1], 4),
                    Operator::new_take([2, 1], 4)
                ]),
            )
        );
    }
}
