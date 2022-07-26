use super::{AddOffset, Error, Map, UnapplyIndicesData};
use crate::finite_f64::FiniteF64;
use crate::simplex::Simplex;
use num::Integer as _;
use std::ops::{Deref, DerefMut};
use std::sync::Arc;

/// An interface for an unbounded coordinate and index map.
pub trait UnboundedMap {
    /// Minimum dimension of the input coordinate. If the dimension of the input
    /// coordinate of [`UnboundedMap::apply_inplace()`] is larger than the minimum, then
    /// the map of the surplus is the identity map.
    fn dim_in(&self) -> usize;
    /// Minimum dimension of the output coordinate.
    fn dim_out(&self) -> usize {
        self.dim_in() + self.delta_dim()
    }
    /// Difference in dimension of the output and input coordinate.
    fn delta_dim(&self) -> usize;
    /// Modulus of the input index. The map repeats itself at index `mod_in`
    /// and the output index is incremented with `in_index / mod_in * mod_out`.
    fn mod_in(&self) -> usize;
    /// Modulus if the output index.
    fn mod_out(&self) -> usize;
    fn apply_mod_out_to_in(&self, n: usize) -> Option<usize> {
        let (i, rem) = n.div_rem(&self.mod_out());
        (rem == 0).then(|| i * self.mod_in())
    }
    fn apply_mod_in_to_out(&self, n: usize) -> Option<usize> {
        let (i, rem) = n.div_rem(&self.mod_in());
        (rem == 0).then(|| i * self.mod_out())
    }
    /// Apply the given index and coordinate, the latter in-place.
    fn apply_inplace(
        &self,
        index: usize,
        coords: &mut [f64],
        stride: usize,
        offset: usize,
    ) -> usize;
    /// Apply the index.
    fn apply_index(&self, index: usize) -> usize;
    /// Apply a sequence of indices in-place.
    fn apply_indices_inplace(&self, indices: &mut [usize]) {
        for index in indices.iter_mut() {
            *index = self.apply_index(*index);
        }
    }
    /// Unapply a sequence of indices.
    fn unapply_indices<T: UnapplyIndicesData>(&self, indices: &[T]) -> Vec<T>;
    /// Returns true if this is the identity map.
    fn is_identity(&self) -> bool {
        self.mod_in() == 1 && self.mod_out() == 1 && self.dim_out() == 0
    }
    /// Returns true if this map manipulates indices only.
    fn is_index_map(&self) -> bool {
        self.dim_out() == 0
    }
    fn update_basis(&self, index: usize, basis: &mut [f64], dim_out: usize, dim_in: &mut usize, offset: usize) -> usize;
    fn basis_is_constant(&self) -> bool;
}

fn coords_iter_mut(
    flat: &mut [f64],
    stride: usize,
    offset: usize,
    dim_out: usize,
    dim_in: usize,
) -> impl Iterator<Item = &mut [f64]> {
    flat.chunks_mut(stride).map(move |coord| {
        let coord = &mut coord[offset..];
        let delta = dim_out - dim_in;
        if delta != 0 {
            coord.copy_within(..coord.len() - delta, delta);
        }
        &mut coord[..dim_out]
    })
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Identity;

impl UnboundedMap for Identity {
    #[inline]
    fn dim_in(&self) -> usize {
        0
    }
    #[inline]
    fn delta_dim(&self) -> usize {
        0
    }
    #[inline]
    fn mod_in(&self) -> usize {
        1
    }
    #[inline]
    fn mod_out(&self) -> usize {
        1
    }
    #[inline]
    fn apply_inplace(
        &self,
        index: usize,
        _coords: &mut [f64],
        _stride: usize,
        _offset: usize,
    ) -> usize {
        index
    }
    #[inline]
    fn apply_index(&self, index: usize) -> usize {
        index
    }
    #[inline]
    fn apply_indices_inplace(&self, _indices: &mut [usize]) {}
    #[inline]
    fn unapply_indices<T: UnapplyIndicesData>(&self, indices: &[T]) -> Vec<T> {
        indices.to_vec()
    }
    #[inline]
    fn is_identity(&self) -> bool {
        true
    }
    #[inline]
    fn update_basis(&self, index: usize, _basis: &mut [f64], _dim_out: usize, _dim_in: &mut usize, _offset: usize) -> usize {
        index
    }
    #[inline]
    fn basis_is_constant(&self) -> bool {
        true
    }
}

impl AddOffset for Identity {
    fn add_offset(&mut self, _offset: usize) {}
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Offset<M: UnboundedMap>(pub M, pub usize);

impl<M: UnboundedMap> UnboundedMap for Offset<M> {
    #[inline]
    fn dim_in(&self) -> usize {
        if self.0.dim_out() > 0 {
            self.0.dim_in() + self.1
        } else {
            0
        }
    }
    #[inline]
    fn delta_dim(&self) -> usize {
        self.0.delta_dim()
    }
    #[inline]
    fn mod_in(&self) -> usize {
        self.0.mod_in()
    }
    #[inline]
    fn mod_out(&self) -> usize {
        self.0.mod_out()
    }
    #[inline]
    fn apply_inplace(
        &self,
        index: usize,
        coords: &mut [f64],
        stride: usize,
        offset: usize,
    ) -> usize {
        self.0.apply_inplace(index, coords, stride, offset + self.1)
    }
    #[inline]
    fn apply_index(&self, index: usize) -> usize {
        self.0.apply_index(index)
    }
    #[inline]
    fn apply_indices_inplace(&self, indices: &mut [usize]) {
        self.0.apply_indices_inplace(indices);
    }
    #[inline]
    fn unapply_indices<T: UnapplyIndicesData>(&self, indices: &[T]) -> Vec<T> {
        self.0.unapply_indices(indices)
    }
    #[inline]
    fn is_identity(&self) -> bool {
        self.0.is_identity()
    }
    #[inline]
    fn update_basis(&self, index: usize, basis: &mut [f64], dim_out: usize, dim_in: &mut usize, offset: usize) -> usize {
        self.0.update_basis(index, basis, dim_out, dim_in, offset + self.1)
    }
    #[inline]
    fn basis_is_constant(&self) -> bool {
        self.0.basis_is_constant()
    }
}

impl<M: UnboundedMap> AddOffset for Offset<M> {
    fn add_offset(&mut self, offset: usize) {
        self.1 += offset;
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Transpose(usize, usize);

impl Transpose {
    #[inline]
    pub const fn new(len1: usize, len2: usize) -> Self {
        Self(len1, len2)
    }
    #[inline]
    pub fn reverse(&mut self) {
        std::mem::swap(&mut self.0, &mut self.1);
    }
}

impl UnboundedMap for Transpose {
    fn dim_in(&self) -> usize {
        0
    }
    fn delta_dim(&self) -> usize {
        0
    }
    fn mod_in(&self) -> usize {
        if self.0 != 1 && self.1 != 1 {
            self.0 * self.1
        } else {
            1
        }
    }
    fn mod_out(&self) -> usize {
        self.mod_in()
    }
    fn apply_inplace(
        &self,
        index: usize,
        _coords: &mut [f64],
        _stride: usize,
        _offset: usize,
    ) -> usize {
        self.apply_index(index)
    }
    fn apply_index(&self, index: usize) -> usize {
        let (j, k) = index.div_rem(&self.1);
        let (i, j) = j.div_rem(&self.0);
        (i * self.1 + k) * self.0 + j
    }
    fn unapply_indices<T: UnapplyIndicesData>(&self, indices: &[T]) -> Vec<T> {
        indices
            .iter()
            .map(|index| {
                let (j, k) = index.get().div_rem(&self.0);
                let (i, j) = j.div_rem(&self.1);
                index.set((i * self.0 + k) * self.1 + j)
            })
            .collect()
    }
    #[inline]
    fn update_basis(&self, index: usize, _basis: &mut [f64], _dim_out: usize, _dim_in: &mut usize, _offset: usize) -> usize {
        self.apply_index(index)
    }
    #[inline]
    fn basis_is_constant(&self) -> bool {
        true
    }
}

impl AddOffset for Transpose {
    fn add_offset(&mut self, _offset: usize) {}
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Take {
    indices: Arc<[usize]>,
    nindices: usize,
    len: usize,
}

impl Take {
    pub fn new(indices: impl Into<Arc<[usize]>>, len: usize) -> Self {
        // TODO: return err if indices.is_empty()
        let indices = indices.into();
        assert!(!indices.is_empty());
        let nindices = indices.len();
        Take {
            indices,
            nindices,
            len,
        }
    }
    pub fn get_indices(&self) -> Arc<[usize]> {
        self.indices.clone()
    }
}

impl UnboundedMap for Take {
    fn dim_in(&self) -> usize {
        0
    }
    fn delta_dim(&self) -> usize {
        0
    }
    fn mod_in(&self) -> usize {
        self.nindices
    }
    fn mod_out(&self) -> usize {
        self.len
    }
    fn apply_inplace(
        &self,
        index: usize,
        _coords: &mut [f64],
        _stride: usize,
        _offset: usize,
    ) -> usize {
        self.apply_index(index)
    }
    fn apply_index(&self, index: usize) -> usize {
        self.indices[index % self.nindices] + index / self.nindices * self.len
    }
    fn unapply_indices<T: UnapplyIndicesData>(&self, indices: &[T]) -> Vec<T> {
        indices
            .iter()
            .filter_map(|index| {
                let (j, iout) = index.get().div_rem(&self.len);
                let offset = j * self.nindices;
                self.indices
                    .iter()
                    .position(|i| *i == iout)
                    .map(|iin| index.set(offset + iin))
            })
            .collect()
    }
    #[inline]
    fn update_basis(&self, index: usize, _basis: &mut [f64], _dim_out: usize, _dim_in: &mut usize, _offset: usize) -> usize {
        self.apply_index(index)
    }
    #[inline]
    fn basis_is_constant(&self) -> bool {
        true
    }
}

impl AddOffset for Take {
    fn add_offset(&mut self, _offset: usize) {}
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Slice {
    start: usize,
    len_in: usize,
    len_out: usize,
}

impl Slice {
    pub fn new(start: usize, len_in: usize, len_out: usize) -> Self {
        assert!(len_in > 0);
        assert!(len_out >= start + len_in);
        Slice {
            start,
            len_in,
            len_out,
        }
    }
    #[inline]
    fn is_identity(&self) -> bool {
        self.start == 0 && self.len_in == self.len_out
    }
}

impl UnboundedMap for Slice {
    fn dim_in(&self) -> usize {
        0
    }
    fn delta_dim(&self) -> usize {
        0
    }
    fn mod_in(&self) -> usize {
        self.len_in
    }
    fn mod_out(&self) -> usize {
        self.len_out
    }
    fn apply_inplace(
        &self,
        index: usize,
        _coords: &mut [f64],
        _stride: usize,
        _offset: usize,
    ) -> usize {
        self.apply_index(index)
    }
    fn apply_index(&self, index: usize) -> usize {
        self.start + index % self.len_in + index / self.len_in * self.len_out
    }
    fn unapply_indices<T: UnapplyIndicesData>(&self, indices: &[T]) -> Vec<T> {
        indices
            .iter()
            .filter_map(|index| {
                let (j, i) = index.get().div_rem(&self.len_out);
                (self.start..self.start + self.len_in)
                    .contains(&i)
                    .then(|| index.set(i - self.start + j * self.len_in))
            })
            .collect()
    }
    #[inline]
    fn update_basis(&self, index: usize, _basis: &mut [f64], _dim_out: usize, _dim_in: &mut usize, _offset: usize) -> usize {
        self.apply_index(index)
    }
    #[inline]
    fn basis_is_constant(&self) -> bool {
        true
    }
}

impl AddOffset for Slice {
    fn add_offset(&mut self, _offset: usize) {}
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Children(Simplex);

impl Children {
    pub fn new(simplex: Simplex) -> Self {
        Self(simplex)
    }
}

impl UnboundedMap for Children {
    fn dim_in(&self) -> usize {
        self.0.dim()
    }
    fn delta_dim(&self) -> usize {
        0
    }
    fn mod_in(&self) -> usize {
        self.0.nchildren()
    }
    fn mod_out(&self) -> usize {
        1
    }
    fn apply_inplace(
        &self,
        index: usize,
        coords: &mut [f64],
        stride: usize,
        offset: usize,
    ) -> usize {
        self.0.apply_child(index, coords, stride, offset)
    }
    fn apply_index(&self, index: usize) -> usize {
        self.0.apply_child_index(index)
    }
    fn apply_indices_inplace(&self, indices: &mut [usize]) {
        self.0.apply_child_indices_inplace(indices)
    }
    fn unapply_indices<T: UnapplyIndicesData>(&self, indices: &[T]) -> Vec<T> {
        indices
            .iter()
            .flat_map(|i| (0..self.mod_in()).map(move |j| i.set(i.get() * self.mod_in() + j)))
            .collect()
    }
    fn update_basis(&self, index: usize, basis: &mut [f64], dim_out: usize, dim_in: &mut usize, offset: usize) -> usize {
        self.0.update_child_basis(index, basis, dim_out, dim_in, offset)
    }
    #[inline]
    fn basis_is_constant(&self) -> bool {
        match self.0 {
            Simplex::Line => true,
            _ => false,
        }
    }
}

impl From<Simplex> for Children {
    fn from(simplex: Simplex) -> Children {
        Children::new(simplex)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Edges(pub Simplex);

impl Edges {
    pub fn new(simplex: Simplex) -> Self {
        Self(simplex)
    }
}

impl UnboundedMap for Edges {
    fn dim_in(&self) -> usize {
        self.0.edge_dim()
    }
    fn delta_dim(&self) -> usize {
        1
    }
    fn mod_in(&self) -> usize {
        self.0.nedges()
    }
    fn mod_out(&self) -> usize {
        1
    }
    fn apply_inplace(
        &self,
        index: usize,
        coords: &mut [f64],
        stride: usize,
        offset: usize,
    ) -> usize {
        self.0.apply_edge(index, coords, stride, offset)
    }
    fn apply_index(&self, index: usize) -> usize {
        self.0.apply_edge_index(index)
    }
    fn apply_indices_inplace(&self, indices: &mut [usize]) {
        self.0.apply_edge_indices_inplace(indices)
    }
    fn unapply_indices<T: UnapplyIndicesData>(&self, indices: &[T]) -> Vec<T> {
        indices
            .iter()
            .flat_map(|i| (0..self.mod_in()).map(move |j| i.set(i.get() * self.mod_in() + j)))
            .collect()
    }
    fn update_basis(&self, index: usize, basis: &mut [f64], dim_out: usize, dim_in: &mut usize, offset: usize) -> usize {
        self.0.update_edge_basis(index, basis, dim_out, dim_in, offset)
    }
    #[inline]
    fn basis_is_constant(&self) -> bool {
        false
    }
}

impl From<Simplex> for Edges {
    fn from(simplex: Simplex) -> Edges {
        Edges::new(simplex)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct UniformPoints {
    points: Arc<[FiniteF64]>,
    npoints: usize,
    point_dim: usize,
}

impl UniformPoints {
    pub fn new(points: impl Into<Arc<[f64]>>, point_dim: usize) -> Self {
        let points: Arc<[FiniteF64]> = unsafe { std::mem::transmute(points.into()) };
        assert_eq!(points.len() % point_dim, 0);
        assert_ne!(point_dim, 0);
        let npoints = points.len() / point_dim;
        UniformPoints {
            points,
            npoints,
            point_dim,
        }
    }
}

impl UnboundedMap for UniformPoints {
    fn dim_in(&self) -> usize {
        0
    }
    fn delta_dim(&self) -> usize {
        self.point_dim
    }
    fn mod_in(&self) -> usize {
        self.npoints
    }
    fn mod_out(&self) -> usize {
        1
    }
    fn apply_inplace(
        &self,
        index: usize,
        coords: &mut [f64],
        stride: usize,
        offset: usize,
    ) -> usize {
        let points: &[f64] = unsafe { std::mem::transmute(&self.points[..]) };
        let point = &points[(index % self.npoints) * self.point_dim..][..self.point_dim];
        for coord in coords_iter_mut(coords, stride, offset, self.point_dim, 0) {
            coord.copy_from_slice(point);
        }
        index / self.npoints
    }
    fn apply_index(&self, index: usize) -> usize {
        index / self.npoints
    }
    fn apply_indices_inplace(&self, indices: &mut [usize]) {
        indices.iter_mut().for_each(|i| *i /= self.npoints)
    }
    fn unapply_indices<T: UnapplyIndicesData>(&self, indices: &[T]) -> Vec<T> {
        indices
            .iter()
            .flat_map(|i| (0..self.mod_in()).map(move |j| i.set(i.get() * self.mod_in() + j)))
            .collect()
    }
    fn update_basis(&self, index: usize, basis: &mut [f64], dim_out: usize, dim_in: &mut usize, offset: usize) -> usize {
        assert!(offset <= *dim_in);
        assert!(*dim_in + self.point_dim < dim_out);
        // Shift rows `offset..dim_in` `self.point_dim` rows down.
        for i in (offset..*dim_in).into_iter().rev() {
            for j in 0..*dim_in {
                basis[(i + self.point_dim) * dim_out + j] = basis[i * dim_out + j];
            }
        }
        for i in offset..offset + self.point_dim {
            for j in 0..*dim_in {
                basis[i * self.point_dim + j] = 0.0;
            }
        }
        // Append identity columns.
        for i in 0..*dim_in + self.point_dim {
            for j in 0..self.point_dim {
                basis[i * dim_out + j] = 0.0;
            }
        }
        for i in 0..self.point_dim {
            basis[(i + offset) * dim_out + *dim_in + i] = 1.0;
        }
        *dim_in += self.point_dim;
        index / self.npoints
    }
    #[inline]
    fn basis_is_constant(&self) -> bool {
        true
    }
}

/// An enum of primitive maps.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Primitive {
    Transpose(Transpose),
    Take(Take),
    Slice(Slice),
    Children(Offset<Children>),
    Edges(Offset<Edges>),
    UniformPoints(Offset<UniformPoints>),
}

impl Primitive {
    #[inline]
    pub fn new_transpose(len1: usize, len2: usize) -> Self {
        Transpose::new(len1, len2).into()
    }
    #[inline]
    pub fn new_take(indices: impl Into<Arc<[usize]>>, len: usize) -> Self {
        Take::new(indices, len).into()
    }
    #[inline]
    pub fn new_slice(start: usize, len_in: usize, len_out: usize) -> Self {
        Slice::new(start, len_in, len_out).into()
    }
    #[inline]
    pub fn new_children(simplex: Simplex) -> Self {
        Children::new(simplex).into()
    }
    #[inline]
    pub fn new_edges(simplex: Simplex) -> Self {
        Edges::new(simplex).into()
    }
    #[inline]
    pub fn new_uniform_points(points: impl Into<Arc<[f64]>>, point_dim: usize) -> Self {
        UniformPoints::new(points, point_dim).into()
    }
    #[inline]
    fn offset_mut(&mut self) -> Option<&mut usize> {
        match self {
            Self::Children(Offset(_, ref mut offset)) => Some(offset),
            Self::Edges(Offset(_, ref mut offset)) => Some(offset),
            Self::UniformPoints(Offset(_, ref mut offset)) => Some(offset),
            _ => None,
        }
    }
    pub fn with_offset(mut self, offset: usize) -> Self {
        if let Some(self_offset) = self.offset_mut() {
            *self_offset = offset
        }
        self
    }
    #[inline]
    const fn is_transpose(&self) -> bool {
        matches!(self, Self::Transpose(_))
    }
    pub fn as_transpose(&self) -> Option<&Transpose> {
        match self {
            Self::Transpose(transpose) => Some(transpose),
            _ => None,
        }
    }
    pub fn as_transpose_mut(&mut self) -> Option<&mut Transpose> {
        match self {
            Self::Transpose(ref mut transpose) => Some(transpose),
            _ => None,
        }
    }
}

macro_rules! dispatch {
    (
        $vis:vis fn $fn:ident$(<$genarg:ident: $genpath:path>)?(
            &$self:ident $(, $arg:ident: $ty:ty)*
        ) $(-> $ret:ty)?
    ) => {
        #[inline]
        $vis fn $fn$(<$genarg: $genpath>)?(&$self $(, $arg: $ty)*) $(-> $ret)? {
            dispatch!(@match $self; $fn; $($arg),*)
        }
    };
    ($vis:vis fn $fn:ident(&mut $self:ident $(, $arg:ident: $ty:ty)*) $(-> $ret:ty)?) => {
        #[inline]
        $vis fn $fn(&mut $self $(, $arg: $ty)*) $(-> $ret)? {
            dispatch!(@match $self; $fn; $($arg),*)
        }
    };
    (@match $self:ident; $fn:ident; $($arg:ident),*) => {
        match $self {
            Self::Transpose(var) => var.$fn($($arg),*),
            Self::Take(var) => var.$fn($($arg),*),
            Self::Slice(var) => var.$fn($($arg),*),
            Self::Children(var) => var.$fn($($arg),*),
            Self::Edges(var) => var.$fn($($arg),*),
            Self::UniformPoints(var) => var.$fn($($arg),*),
        }
    };
}

impl UnboundedMap for Primitive {
    dispatch! {fn dim_in(&self) -> usize}
    dispatch! {fn delta_dim(&self) -> usize}
    dispatch! {fn mod_in(&self) -> usize}
    dispatch! {fn mod_out(&self) -> usize}
    dispatch! {fn apply_inplace(&self, index: usize, coords: &mut[f64], stride: usize, offset: usize) -> usize}
    dispatch! {fn apply_index(&self, index: usize) -> usize}
    dispatch! {fn apply_indices_inplace(&self, indices: &mut [usize])}
    dispatch! {fn unapply_indices<T: UnapplyIndicesData>(&self, indices: &[T]) -> Vec<T>}
    dispatch! {fn is_identity(&self) -> bool}
    dispatch! {fn is_index_map(&self) -> bool}
    dispatch! {fn update_basis(&self, index: usize, basis: &mut [f64], dim_out: usize, dim_in: &mut usize, offset: usize) -> usize}
    dispatch! {fn basis_is_constant(&self) -> bool}
}

impl AddOffset for Primitive {
    dispatch! {fn add_offset(&mut self, offset: usize)}
}

impl std::fmt::Debug for Primitive {
    dispatch! {fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result}
}

impl From<Transpose> for Primitive {
    fn from(transpose: Transpose) -> Self {
        Self::Transpose(transpose)
    }
}

impl From<Take> for Primitive {
    fn from(take: Take) -> Self {
        Self::Take(take)
    }
}

impl From<Slice> for Primitive {
    fn from(slice: Slice) -> Self {
        Self::Slice(slice)
    }
}

impl From<Children> for Primitive {
    fn from(children: Children) -> Self {
        Self::Children(Offset(children, 0))
    }
}

impl From<Edges> for Primitive {
    fn from(edges: Edges) -> Self {
        Self::Edges(Offset(edges, 0))
    }
}

impl From<UniformPoints> for Primitive {
    fn from(uniform_points: UniformPoints) -> Self {
        Self::UniformPoints(Offset(uniform_points, 0))
    }
}

#[inline]
fn comp_dim_out_in<M: UnboundedMap>(map: &M, dim_out: usize, dim_in: usize) -> (usize, usize) {
    let n = map.dim_in().checked_sub(dim_out).unwrap_or(0);
    (dim_out + map.delta_dim() + n, dim_in + n)
}

#[inline]
fn comp_mod_out_in<M: UnboundedMap>(map: &M, mod_out: usize, mod_in: usize) -> (usize, usize) {
    let n = mod_out.lcm(&map.mod_in());
    (n / map.mod_in() * map.mod_out(), mod_in * n / mod_out)
}

impl<M, Array> UnboundedMap for Array
where
    M: UnboundedMap,
    Array: Deref<Target = [M]>,
{
    #[inline]
    fn dim_in(&self) -> usize {
        self.iter()
            .rev()
            .fold((0, 0), |(o, i), map| comp_dim_out_in(map, o, i))
            .1
    }
    #[inline]
    fn delta_dim(&self) -> usize {
        self.iter().map(|map| map.delta_dim()).sum()
    }
    #[inline]
    fn mod_in(&self) -> usize {
        self.iter()
            .rev()
            .fold((1, 1), |(o, i), map| comp_mod_out_in(map, o, i))
            .1
    }
    #[inline]
    fn mod_out(&self) -> usize {
        self.iter()
            .rev()
            .fold((1, 1), |(o, i), map| comp_mod_out_in(map, o, i))
            .0
    }
    #[inline]
    fn apply_inplace(
        &self,
        index: usize,
        coords: &mut [f64],
        stride: usize,
        offset: usize,
    ) -> usize {
        self.iter().rev().fold(index, |index, map| {
            map.apply_inplace(index, coords, stride, offset)
        })
    }
    #[inline]
    fn apply_index(&self, index: usize) -> usize {
        self.iter()
            .rev()
            .fold(index, |index, map| map.apply_index(index))
    }
    #[inline]
    fn apply_indices_inplace(&self, indices: &mut [usize]) {
        self.iter()
            .rev()
            .for_each(|map| map.apply_indices_inplace(indices));
    }
    #[inline]
    fn unapply_indices<T: UnapplyIndicesData>(&self, indices: &[T]) -> Vec<T> {
        self.iter().fold(indices.to_vec(), |indices, map| {
            map.unapply_indices(&indices)
        })
    }
    #[inline]
    fn is_identity(&self) -> bool {
        self.iter().all(|map| map.is_identity())
    }
    #[inline]
    fn update_basis(&self, index: usize, basis: &mut [f64], dim_out: usize, dim_in: &mut usize, offset: usize) -> usize {
        self.iter().rev().fold(index, |index, map| map.update_basis(index, basis, dim_out, dim_in, offset))
    }
    #[inline]
    fn basis_is_constant(&self) -> bool {
        if self.mod_in() == 1 {
            true
        } else {
            self.iter().all(|map| map.basis_is_constant())
        }
    }
}

impl<M, Array> AddOffset for Array
where
    M: UnboundedMap + AddOffset,
    Array: Deref<Target = [M]> + DerefMut,
{
    fn add_offset(&mut self, offset: usize) {
        self.iter_mut().for_each(|map| map.add_offset(offset));
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WithBounds<M: UnboundedMap> {
    map: M,
    dim_in: usize,
    delta_dim: usize,
    len_in: usize,
    len_out: usize,
}

impl<M: UnboundedMap> WithBounds<M> {
    pub fn from_input(map: M, dim_in: usize, len_in: usize) -> Result<Self, Error> {
        if dim_in < map.dim_in() {
            Err(Error::DimensionMismatch)
        } else if len_in % map.mod_in() != 0 {
            Err(Error::LengthMismatch)
        } else {
            Ok(Self::new_unchecked(map, dim_in, len_in))
        }
    }
    pub fn from_output(map: M, dim_out: usize, len_out: usize) -> Result<Self, Error> {
        if dim_out < map.dim_out() {
            Err(Error::DimensionMismatch)
        } else if len_out % map.mod_out() != 0 {
            Err(Error::LengthMismatch)
        } else {
            let dim_in = dim_out - map.delta_dim();
            let len_in = len_out / map.mod_out() * map.mod_in();
            Ok(Self::new_unchecked(map, dim_in, len_in))
        }
    }
    pub fn new_unchecked(map: M, dim_in: usize, len_in: usize) -> Self {
        let delta_dim = map.delta_dim();
        let len_out = len_in / map.mod_in() * map.mod_out();
        Self {
            map,
            dim_in,
            delta_dim,
            len_in,
            len_out,
        }
    }
    pub fn unbounded(&self) -> &M {
        &self.map
    }
}

impl<M: UnboundedMap> Map for WithBounds<M> {
    #[inline]
    fn dim_in(&self) -> usize {
        self.dim_in
    }
    #[inline]
    fn delta_dim(&self) -> usize {
        self.delta_dim
    }
    #[inline]
    fn len_in(&self) -> usize {
        self.len_in
    }
    #[inline]
    fn len_out(&self) -> usize {
        self.len_out
    }
    #[inline]
    fn apply_inplace_unchecked(
        &self,
        index: usize,
        coords: &mut [f64],
        stride: usize,
        offset: usize,
    ) -> usize {
        self.map.apply_inplace(index, coords, stride, offset)
    }
    #[inline]
    fn apply_index_unchecked(&self, index: usize) -> usize {
        self.map.apply_index(index)
    }
    #[inline]
    fn apply_indices_inplace_unchecked(&self, indices: &mut [usize]) {
        self.map.apply_indices_inplace(indices)
    }
    #[inline]
    fn unapply_indices_unchecked<T: UnapplyIndicesData>(&self, indices: &[T]) -> Vec<T> {
        self.map.unapply_indices(indices)
    }
    #[inline]
    fn is_identity(&self) -> bool {
        self.map.is_identity()
    }
    #[inline]
    fn is_index_map(&self) -> bool {
        self.map.is_index_map()
    }
    #[inline]
    fn update_basis(&self, index: usize, basis: &mut [f64], dim_out: usize, dim_in: &mut usize, offset: usize) -> usize {
        self.map.update_basis(index, basis, dim_out, dim_in, offset)
    }
    #[inline]
    fn basis_is_constant(&self) -> bool {
        self.map.basis_is_constant()
    }
}

impl<M> AddOffset for WithBounds<M>
where
    M: UnboundedMap + AddOffset,
{
    fn add_offset(&mut self, offset: usize) {
        self.dim_in += offset;
        self.map.add_offset(offset);
    }
}

/// An interface for swapping a composition of `Self` with a [`Primitive`].
pub trait SwapPrimitiveComposition {
    type Output;

    /// Returns a [`Primitive`'] and a map such that the composition of those is equivalent to the composition of `self` with `inner`.
    fn swap_primitive_composition(
        &self,
        inner: &Primitive,
        stride: usize,
    ) -> Option<((Primitive, usize), Self::Output)>;
}

impl SwapPrimitiveComposition for [Primitive] {
    type Output = Vec<Primitive>;

    fn swap_primitive_composition(
        &self,
        inner: &Primitive,
        stride: usize,
    ) -> Option<((Primitive, usize), Self::Output)> {
        if self.is_empty() {
            return Some(((inner.clone(), stride), Vec::new()));
        } else if inner.is_transpose() {
            return None;
        }
        let mut target = inner.clone();
        let mut shifted_items: Vec<Primitive> = Vec::new();
        let mut queue: Vec<Primitive> = Vec::new();
        let mut stride_out = stride;
        let mut stride_in = stride;
        for mut item in self.iter().rev().cloned() {
            // Swap matching edges and children at the same offset.
            if let Primitive::Edges(Offset(Edges(esimplex), eoffset)) = &item {
                if let Primitive::Children(Offset(Children(ref mut csimplex), coffset)) =
                    &mut target
                {
                    if eoffset == coffset && esimplex.edge_dim() == csimplex.dim() {
                        if stride_in != 1 && inner.mod_in() != 1 {
                            shifted_items.push(Primitive::new_transpose(stride_in, inner.mod_in()));
                        }
                        shifted_items.append(&mut queue);
                        if stride_out != 1 && inner.mod_in() != 1 {
                            shifted_items
                                .push(Primitive::new_transpose(inner.mod_in(), stride_out));
                        }
                        shifted_items.push(Primitive::new_take(
                            esimplex.swap_edges_children_map(),
                            esimplex.nedges() * esimplex.nchildren(),
                        ));
                        shifted_items.push(Primitive::Edges(Offset(Edges(*esimplex), *eoffset)));
                        *csimplex = *esimplex;
                        stride_in = 1;
                        stride_out = 1;
                        continue;
                    }
                }
            }
            // Update strides.
            if inner.mod_in() == 1 && inner.mod_out() == 1 {
            } else if inner.mod_out() == 1 {
                let n = stride_out.gcd(&item.mod_in());
                stride_out = stride_out / n * item.mod_out();
                stride_in *= item.mod_in() / n;
            } else if let Some(Transpose(ref mut m, ref mut n)) = item.as_transpose_mut() {
                if stride_out % (*m * *n) == 0 {
                } else if stride_out % *n == 0 && (*m * *n) % (stride_out * inner.mod_out()) == 0 {
                    stride_out /= *n;
                    *m = *m / inner.mod_out() * inner.mod_in();
                } else if *n % stride_out == 0 && *n % (stride_out * inner.mod_out()) == 0 {
                    stride_out *= *m;
                    *n = *n / inner.mod_out() * inner.mod_in();
                } else {
                    return None;
                }
            } else if stride_out % item.mod_in() == 0 {
                stride_out = stride_out / item.mod_in() * item.mod_out();
            } else {
                return None;
            }
            // Update offsets.
            let item_delta_dim = item.delta_dim();
            let target_delta_dim = target.delta_dim();
            let item_dim_in = item.dim_in();
            let target_dim_out = target.dim_out();
            if let (Some(item_offset), Some(target_offset)) =
                (item.offset_mut(), target.offset_mut())
            {
                if item_dim_in <= *target_offset {
                    *target_offset += item_delta_dim;
                } else if target_dim_out <= *item_offset {
                    *item_offset -= target_delta_dim;
                } else {
                    return None;
                }
            }
            if !item.is_identity() {
                queue.push(item);
            }
        }
        if stride_in != 1 && target.mod_in() != 1 {
            shifted_items.push(Primitive::new_transpose(stride_in, target.mod_in()));
        }
        shifted_items.extend(queue);
        if stride_out != 1 && target.mod_in() != 1 {
            shifted_items.push(Primitive::new_transpose(target.mod_in(), stride_out));
        }
        if target.mod_out() == 1 {
            stride_out = 1;
        }
        shifted_items.reverse();
        Some(((target, stride_out), shifted_items))
    }
}

impl SwapPrimitiveComposition for Vec<Primitive> {
    type Output = Self;

    #[inline]
    fn swap_primitive_composition(
        &self,
        inner: &Primitive,
        stride: usize,
    ) -> Option<((Primitive, usize), Self::Output)> {
        (&self[..]).swap_primitive_composition(inner, stride)
    }
}

impl<M: UnboundedMap + SwapPrimitiveComposition> SwapPrimitiveComposition for WithBounds<M>
where
    M: UnboundedMap + SwapPrimitiveComposition,
    M::Output: UnboundedMap,
{
    type Output = WithBounds<M::Output>;

    fn swap_primitive_composition(
        &self,
        inner: &Primitive,
        stride: usize,
    ) -> Option<((Primitive, usize), Self::Output)> {
        self.unbounded()
            .swap_primitive_composition(inner, stride)
            .map(|(outer, slf)| {
                let dim_in = self.dim_in() - inner.delta_dim();
                let len_in = self.len_in() / inner.mod_out() * inner.mod_in();
                (outer, WithBounds::new_unchecked(slf, dim_in, len_in))
            })
    }
}

/// Return type of [`AllPrimitiveDecompositions::all_primitive_decompositions()`].
pub type PrimitiveDecompositionIter<'a, T> = Box<dyn Iterator<Item = ((Primitive, usize), T)> + 'a>;

/// An interface for iterating over all possible decompositions into [`Primitive`] and `Self`.
pub trait AllPrimitiveDecompositions: Sized {
    /// Return an iterator over all possible decompositions into `Self` and an [`Primitive`].
    ///
    /// # Examples
    ///
    /// ```
    /// use nutils_test::simplex::Simplex::*;
    /// use nutils_test::primitive::{Primitive, AllPrimitiveDecompositions as _};
    /// use nutils_test::prim_comp;
    /// let map = prim_comp![Triangle*2 <- Edges <- Children];
    /// let mut iter = map.all_primitive_decompositions();
    /// assert_eq!(
    ///     iter.next(),
    ///     Some(((Primitive::new_edges(Triangle), 1), prim_comp![Line*6 <- Children])));
    /// assert_eq!(
    ///     iter.next(),
    ///     Some((
    ///         (Primitive::new_children(Triangle), 1),
    ///         prim_comp![Triangle*8 <- Edges <- Take([3, 6, 1, 7, 2, 5], 12)],
    ///     ))
    /// );
    /// assert_eq!(iter.next(), None);
    /// ```
    fn all_primitive_decompositions<'a>(&'a self) -> PrimitiveDecompositionIter<'a, Self>;
    fn as_transposes(&self) -> Option<Vec<Transpose>>
    where
        Self: Map,
    {
        let mut transposes = Vec::new();
        if self.is_identity() {
            return Some(Vec::new());
        }
        let mut next = |m: &Self| {
            if let Some(((Primitive::Transpose(transpose), stride), rhs)) =
                m.all_primitive_decompositions().next()
            {
                let len = transpose.mod_out();
                if stride != 1 {
                    unimplemented! {}
                    transposes.push(Transpose::new(stride, len));
                }
                transposes.push(transpose);
                if stride != 1 {
                    unimplemented! {}
                    transposes.push(Transpose::new(len, stride));
                }
                Some(rhs)
            } else {
                None
            }
        };
        let mut rhs = if let Some(rhs) = next(self) {
            rhs
        } else {
            return None;
        };
        while !rhs.is_identity() {
            rhs = if let Some(rhs) = next(&rhs) {
                rhs
            } else {
                return None;
            };
        }
        Some(transposes)
    }
}

impl AllPrimitiveDecompositions for Vec<Primitive> {
    fn all_primitive_decompositions<'a>(&'a self) -> PrimitiveDecompositionIter<'a, Self> {
        let mut splits = Vec::new();
        for (i, item) in self.iter().enumerate() {
            if let Some((outer, mut inner)) = (&self[..i]).swap_primitive_composition(item, 1) {
                inner.extend(self[i + 1..].iter().cloned());
                splits.push((outer, inner));
            }
            if let Primitive::Edges(Offset(Edges(Simplex::Line), offset)) = item {
                let mut children = Primitive::new_children(Simplex::Line);
                children.add_offset(*offset);
                if let Some((outer, mut inner)) =
                    (&self[..i]).swap_primitive_composition(&children, 1)
                {
                    inner.push(item.clone());
                    inner.push(Primitive::new_take(
                        Simplex::Line.swap_edges_children_map(),
                        Simplex::Line.nedges() * Simplex::Line.nchildren(),
                    ));
                    inner.extend(self[i + 1..].iter().cloned());
                    splits.push((outer, inner));
                }
            }
        }
        Box::new(splits.into_iter())
    }
}

impl<M> AllPrimitiveDecompositions for WithBounds<M>
where
    M: UnboundedMap + AllPrimitiveDecompositions,
{
    fn all_primitive_decompositions<'a>(&'a self) -> PrimitiveDecompositionIter<'a, Self> {
        Box::new(
            self.unbounded()
                .all_primitive_decompositions()
                .into_iter()
                .map(|(outer, unbounded)| {
                    (
                        outer,
                        WithBounds::new_unchecked(unbounded, self.dim_in(), self.len_in()),
                    )
                }),
        )
    }
}

/// Create a bounded composition of primitive maps.
///
/// # Syntax
///
/// The arguments of the macro are separated by `<-`, indicating the direction
/// of the map. The first argument is a simplex (`Triangle`, `Line`) or
/// `Point`, multiplied with the output length of the map. The remaining arguments
/// are primitive maps: `Children`, `Edges`, `Transpose(len1, len2)` or `Take(indices, len)`.
///
/// # Examples
///
/// ```
/// use nutils_test::prim_comp;
/// use nutils_test::simplex::Simplex::*;
/// prim_comp![Line*2 <- Children <- Edges];
/// ```
#[macro_export]
macro_rules! prim_comp {
    (Point*$len_out:literal $($tail:tt)*) => {{
        use $crate::map::primitive::{Primitive, WithBounds};
        #[allow(unused_mut)]
        let mut comp: Vec<Primitive> = Vec::new();
        $crate::prim_comp!{@adv comp, Point; $($tail)*}
        WithBounds::from_output(comp, 0, $len_out).unwrap()
    }};
    ($simplex:tt*$len_out:literal $($tail:tt)*) => {{
        use $crate::map::primitive::{Primitive, WithBounds};
        #[allow(unused_mut)]
        let mut comp: Vec<Primitive> = Vec::new();
        $crate::prim_comp!{@adv comp, $simplex; $($tail)*}
        let dim_out = $crate::prim_comp!(@dim $simplex);
        WithBounds::from_output(comp, dim_out, $len_out).unwrap()
    }};
    (@dim Point) => {0};
    (@dim Line) => {1};
    (@dim Triangle) => {2};
    (@adv $comp:ident, $simplex:tt;) => {};
    (@adv $comp:ident, $simplex:tt; <- Children $($tail:tt)*) => {{
        $comp.push(Primitive::new_children($crate::simplex::Simplex::$simplex));
        $crate::prim_comp!{@adv $comp, $simplex; $($tail)*}
    }};
    (@adv $comp:ident, Triangle; <- Edges $($tail:tt)*) => {{
        $comp.push(Primitive::new_edges($crate::simplex::Simplex::Triangle));
        $crate::prim_comp!{@adv $comp, Line; $($tail)*}
    }};
    (@adv $comp:ident, Line; <- Edges $($tail:tt)*) => {{
        $comp.push(Primitive::new_edges($crate::simplex::Simplex::Line));
        $crate::prim_comp!{@adv $comp, Point; $($tail)*}
    }};
    (@adv $comp:ident, $simplex:tt; <- Transpose($len1:expr, $len2:expr) $($tail:tt)*) => {{
        $comp.push(Primitive::new_transpose($len1, $len2));
        $crate::prim_comp!{@adv $comp, $simplex; $($tail)*}
    }};
    (@adv $comp:ident, $simplex:tt; <- Take($indices:expr, $len:expr) $($tail:tt)*) => {{
        $comp.push(Primitive::new_take($indices.to_vec(), $len));
        $crate::prim_comp!{@adv $comp, $simplex; $($tail)*}
    }};
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use std::iter;
    use Simplex::*;

    macro_rules! assert_map_apply {
        ($item:expr, $inidx:expr, $incoords:expr, $outidx:expr, $outcoords:expr) => {{
            use std::borrow::Borrow;
            let item = $item.borrow();
            let incoords = $incoords;
            let outcoords = $outcoords;
            assert_eq!(incoords.len(), outcoords.len());
            let stride;
            let mut work: Vec<_>;
            if incoords.len() == 0 {
                stride = item.dim_out();
                work = Vec::with_capacity(0);
            } else {
                stride = outcoords[0].len();
                work = iter::repeat(-1.0).take(outcoords.len() * stride).collect();
                for (work, incoord) in iter::zip(work.chunks_mut(stride), incoords.iter()) {
                    work[..incoord.len()].copy_from_slice(incoord);
                }
            }
            assert_eq!(item.apply_inplace($inidx, &mut work, stride, 0), $outidx);
            for (actual, desired) in iter::zip(work.chunks(stride), outcoords.iter()) {
                assert_abs_diff_eq!(actual[..], desired[..]);
            }
        }};
        ($item:expr, $inidx:expr, $outidx:expr) => {{
            use std::borrow::Borrow;
            let item = $item.borrow();
            let mut work = Vec::with_capacity(0);
            assert_eq!(
                item.apply_inplace($inidx, &mut work, item.dim_out(), 0),
                $outidx
            );
        }};
    }

    #[test]
    fn apply_transpose() {
        let item = Primitive::new_transpose(3, 2);
        assert_map_apply!(item, 0, 0);
        assert_map_apply!(item, 1, 3);
        assert_map_apply!(item, 2, 1);
        assert_map_apply!(item, 3, 4);
        assert_map_apply!(item, 4, 2);
        assert_map_apply!(item, 5, 5);
        assert_map_apply!(item, 6, 6);
        assert_map_apply!(item, 7, 9);
    }

    #[test]
    fn apply_take() {
        let item = Primitive::new_take(vec![4, 1, 2], 5);
        assert_map_apply!(item, 0, 4);
        assert_map_apply!(item, 1, 1);
        assert_map_apply!(item, 2, 2);
        assert_map_apply!(item, 3, 9);
        assert_map_apply!(item, 4, 6);
        assert_map_apply!(item, 5, 7);
    }

    #[test]
    fn apply_children_line() {
        let mut item = Primitive::new_children(Line);
        assert_map_apply!(item, 0, [[0.0], [1.0]], 0, [[0.0], [0.5]]);
        assert_map_apply!(item, 1, [[0.0], [1.0]], 0, [[0.5], [1.0]]);
        assert_map_apply!(item, 2, [[0.0], [1.0]], 1, [[0.0], [0.5]]);

        item.add_offset(1);
        assert_map_apply!(
            item,
            3,
            [[0.2, 0.0], [0.3, 1.0]],
            1,
            [[0.2, 0.5], [0.3, 1.0]]
        );
    }

    #[test]
    fn apply_edges_line() {
        let mut item = Primitive::new_edges(Line);
        assert_map_apply!(item, 0, [[]], 0, [[1.0]]);
        assert_map_apply!(item, 1, [[]], 0, [[0.0]]);
        assert_map_apply!(item, 2, [[]], 1, [[1.0]]);

        item.add_offset(1);
        assert_map_apply!(item, 0, [[0.2]], 0, [[0.2, 1.0]]);
    }

    #[test]
    fn apply_uniform_points() {
        let mut item = Primitive::new_uniform_points(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2);
        assert_map_apply!(item, 0, [[]], 0, [[1.0, 2.0]]);
        assert_map_apply!(item, 1, [[]], 0, [[3.0, 4.0]]);
        assert_map_apply!(item, 2, [[]], 0, [[5.0, 6.0]]);
        assert_map_apply!(item, 3, [[]], 1, [[1.0, 2.0]]);

        item.add_offset(1);
        assert_map_apply!(item, 0, [[7.0]], 0, [[7.0, 1.0, 2.0]]);
    }

    macro_rules! assert_unapply {
        ($item:expr) => {{
            let item = $item;
            let nin = 2 * item.mod_in();
            let nout = 2 * item.mod_out();
            assert!(nout > 0);
            let mut map: Vec<Vec<usize>> = (0..nout).map(|_| Vec::new()).collect();
            let mut work = Vec::with_capacity(0);
            for i in 0..nin {
                map[item.apply_inplace(i, &mut work, item.dim_out(), 0)].push(i);
            }
            for (j, desired) in map.into_iter().enumerate() {
                let mut actual = item.unapply_indices(&[j]);
                actual.sort();
                assert_eq!(actual, desired);
            }
        }};
    }

    #[test]
    fn unapply_indices_transpose() {
        assert_unapply!(Primitive::new_transpose(3, 2));
    }

    #[test]
    fn unapply_indices_take() {
        assert_unapply!(Primitive::new_take(vec![4, 1], 5));
    }

    #[test]
    fn unapply_indices_children() {
        assert_unapply!(Primitive::new_children(Triangle));
    }

    #[test]
    fn unapply_indices_edges() {
        assert_unapply!(Primitive::new_edges(Triangle));
    }

    #[test]
    fn unapply_indices_uniform_points() {
        assert_unapply!(Primitive::new_uniform_points(
            vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
            2
        ));
    }

    macro_rules! assert_equiv_maps {
        ($a:expr, $b:expr $(, $simplex:ident)*) => {{
            let a: &[Primitive] = &$a;
            let b: &[Primitive] = &$b;
            let dim_in = 0 $(+ $simplex.dim())*;
            let dim_out = a.dim_out();
            println!("a: {a:?}");
            println!("b: {b:?}");
            assert_eq!(a.mod_in(), b.mod_in());
            assert_eq!(a.mod_out(), b.mod_out());
            assert_eq!(a.dim_in(), dim_in);
            assert_eq!(b.dim_in(), dim_in);
            assert_eq!(a.delta_dim(), b.delta_dim());
            // Build coords: the outer product of the vertices of the given simplices, zero-padded
            // to the out dimension.
            let coords = iter::once([]);
            $(
                let coords = coords.flat_map(|coord| {
                    $simplex
                        .vertices()
                        .chunks($simplex.dim())
                        .map(move |vert| [&coord, vert].concat())
                    });
            )*
            let pad: Vec<f64> = iter::repeat(0.0).take(a.delta_dim()).collect();
            let coords: Vec<f64> = coords.flat_map(|coord| [&coord[..], &pad].concat()).collect();
            // Test if every tip index and coordinate maps to the same out index and coordinate.
            for iin in 0..2 * a.mod_in() {
                let mut crds_a = coords.clone();
                let mut crds_b = coords.clone();
                let iout_a = a.apply_inplace(iin, &mut crds_a, dim_out, 0);
                let iout_b = b.apply_inplace(iin, &mut crds_b, dim_out, 0);
                assert_eq!(iout_a, iout_b, "iin={iin}");
                assert_abs_diff_eq!(crds_a[..], crds_b[..]);
            }
        }};
    }

    macro_rules! assert_swap_primitive_composition {
        ($($item:expr),*; $($simplex:ident),*) => {{
            let unshifted = [$(Primitive::from($item),)*];
            let ((litem, lstride), lchain) = (&unshifted[..unshifted.len() - 1]).swap_primitive_composition(&unshifted.last().unwrap(), 1).unwrap();
            let mut shifted: Vec<Primitive> = Vec::new();
            if lstride != 1 {
                shifted.push(Primitive::new_transpose(lstride, litem.mod_out()));
            }
            shifted.push(litem);
            shifted.extend(lchain.into_iter());
            assert_equiv_maps!(&shifted[..], &unshifted[..] $(, $simplex)*);
        }};
    }

    #[test]
    fn swap_primitive_composition() {
        assert_swap_primitive_composition!(
            Transpose::new(4, 3), Primitive::new_take(vec![0, 1], 3);
        );
        assert_swap_primitive_composition!(
            Transpose::new(3, 5), Transpose::new(5, 4*3), Primitive::new_take(vec![0, 1], 3);
        );
        assert_swap_primitive_composition!(
            Transpose::new(5, 4), Transpose::new(5*4, 3), Primitive::new_take(vec![0, 1], 3);
        );
        assert_swap_primitive_composition!(
            {let mut elem = Primitive::new_children(Line); elem.add_offset(1); elem},
            Primitive::new_children(Line);
            Line, Line
        );
        assert_swap_primitive_composition!(
            Primitive::new_edges(Line), Primitive::new_children(Line);
            Line
        );
        assert_swap_primitive_composition!(
            Primitive::new_children(Line),
            {let mut elem = Primitive::new_edges(Line); elem.add_offset(1); elem};
            Line
        );
        assert_swap_primitive_composition!(
            Primitive::new_edges(Triangle), Primitive::new_children(Line);
            Line
        );
    }
}
