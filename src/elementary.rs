use crate::finite_f64::FiniteF64;
use crate::simplex::Simplex;
use crate::{AddOffset, Error, Map, UnapplyIndicesData};
use num::Integer as _;
use std::ops::{Deref, DerefMut};
use std::rc::Rc;

pub trait UnboundedMap {
    // Minimum dimension of the input coordinate. If the dimension of the input
    // coordinate of [UnboundedMap::apply_inplace()] is larger than the minimum, then
    // the map of the surplus is the identity map.
    fn dim_in(&self) -> usize;
    // Minimum dimension of the output coordinate.
    fn dim_out(&self) -> usize {
        self.dim_in() + self.delta_dim()
    }
    // Difference in dimension of the output and input coordinate.
    fn delta_dim(&self) -> usize;
    // Modulus of the input index. The map repeats itself at index `mod_in`
    // and the output index is incremented with `in_index / mod_in * mod_out`.
    fn mod_in(&self) -> usize;
    // Modulus if the output index.
    fn mod_out(&self) -> usize;
    fn apply_mod_out_to_in(&self, n: usize) -> Option<usize> {
        let (i, rem) = n.div_rem(&self.mod_out());
        (rem == 0).then(|| i * self.mod_in())
    }
    fn apply_mod_in_to_out(&self, n: usize) -> Option<usize> {
        let (i, rem) = n.div_rem(&self.mod_in());
        (rem == 0).then(|| i * self.mod_out())
    }
    fn apply_inplace(
        &self,
        index: usize,
        coords: &mut [f64],
        stride: usize,
        offset: usize,
    ) -> usize;
    fn apply_index(&self, index: usize) -> usize;
    fn apply_indices_inplace(&self, indices: &mut [usize]) {
        for index in indices.iter_mut() {
            *index = self.apply_index(*index);
        }
    }
    fn unapply_indices<T: UnapplyIndicesData>(&self, indices: &[T]) -> Vec<T>;
    fn is_identity(&self) -> bool {
        self.mod_in() == 1 && self.mod_out() == 1 && self.dim_out() == 0
    }
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

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Identity;

impl UnboundedMap for Identity {
    fn dim_in(&self) -> usize {
        0
    }
    fn delta_dim(&self) -> usize {
        0
    }
    fn mod_in(&self) -> usize {
        1
    }
    fn mod_out(&self) -> usize {
        1
    }
    fn apply_inplace(
        &self,
        index: usize,
        _coords: &mut [f64],
        _stride: usize,
        _offset: usize,
    ) -> usize {
        index
    }
    fn apply_index(&self, index: usize) -> usize {
        index
    }
    fn apply_indices_inplace(&self, _indices: &mut [usize]) {}
    fn unapply_indices<T: UnapplyIndicesData>(&self, indices: &[T]) -> Vec<T> {
        indices.to_vec()
    }
    fn is_identity(&self) -> bool {
        true
    }
}

impl AddOffset for Identity {
    fn add_offset(&mut self, _offset: usize) {}
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Offset<M: UnboundedMap>(M, usize);

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
}

impl<M: UnboundedMap> AddOffset for Offset<M> {
    fn add_offset(&mut self, offset: usize) {
        self.1 += offset;
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
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
}

impl AddOffset for Transpose {
    fn add_offset(&mut self, _offset: usize) {}
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Take {
    indices: Rc<[usize]>,
    nindices: usize,
    len: usize,
}

impl Take {
    pub fn new(indices: impl Into<Rc<[usize]>>, len: usize) -> Self {
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
    pub fn get_indices(&self) -> Rc<[usize]> {
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
}

impl AddOffset for Take {
    fn add_offset(&mut self, _offset: usize) {}
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
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
}

impl AddOffset for Slice {
    fn add_offset(&mut self, _offset: usize) {}
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
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
}

impl From<Simplex> for Children {
    fn from(simplex: Simplex) -> Children {
        Children::new(simplex)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
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
}

impl From<Simplex> for Edges {
    fn from(simplex: Simplex) -> Edges {
        Edges::new(simplex)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct UniformPoints {
    points: Rc<[FiniteF64]>,
    npoints: usize,
    point_dim: usize,
}

impl UniformPoints {
    pub fn new(points: impl Into<Rc<[f64]>>, point_dim: usize) -> Self {
        let points: Rc<[FiniteF64]> = unsafe { std::mem::transmute(points.into()) };
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
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum Elementary {
    Transpose(Transpose),
    Take(Take),
    Slice(Slice),
    Children(Offset<Children>),
    Edges(Offset<Edges>),
    UniformPoints(Offset<UniformPoints>),
}

impl Elementary {
    #[inline]
    pub fn new_transpose(len1: usize, len2: usize) -> Self {
        Transpose::new(len1, len2).into()
    }
    #[inline]
    pub fn new_take(indices: impl Into<Rc<[usize]>>, len: usize) -> Self {
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
    pub fn new_uniform_points(points: impl Into<Rc<[f64]>>, point_dim: usize) -> Self {
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
    pub fn shift_inner_to_outer(
        &self,
        items: &[Self],
    ) -> Option<(Option<Transpose>, Self, Vec<Self>)> {
        if self.is_transpose() {
            return None;
        }
        let mut target = self.clone();
        let mut shifted_items: Vec<Self> = Vec::new();
        let mut queue: Vec<Self> = Vec::new();
        let mut stride_out = 1;
        let mut stride_in = 1;
        for mut item in items.iter().cloned() {
            // Swap matching edges and children at the same offset.
            if let Self::Edges(Offset(Edges(esimplex), eoffset)) = &item {
                if let Self::Children(Offset(Children(ref mut csimplex), coffset)) = &mut target {
                    if eoffset == coffset && esimplex.edge_dim() == csimplex.dim() {
                        if stride_in != 1 && self.mod_in() != 1 {
                            shifted_items.push(Self::new_transpose(stride_in, self.mod_in()));
                        }
                        shifted_items.append(&mut queue);
                        if stride_out != 1 && self.mod_in() != 1 {
                            shifted_items.push(Self::new_transpose(self.mod_in(), stride_out));
                        }
                        shifted_items.push(Self::new_take(
                            esimplex.swap_edges_children_map(),
                            esimplex.nedges() * esimplex.nchildren(),
                        ));
                        shifted_items.push(Self::Edges(Offset(Edges(*esimplex), *eoffset)));
                        *csimplex = *esimplex;
                        stride_in = 1;
                        stride_out = 1;
                        continue;
                    }
                }
            }
            // Update strides.
            if self.mod_in() == 1 && self.mod_out() == 1 {
            } else if self.mod_out() == 1 {
                let n = stride_out.gcd(&item.mod_in());
                stride_out = stride_out / n * item.mod_out();
                stride_in *= item.mod_in() / n;
            } else if let Some(Transpose(ref mut m, ref mut n)) = item.as_transpose_mut() {
                if stride_out % (*m * *n) == 0 {
                } else if stride_out % *n == 0 && (*m * *n) % (stride_out * self.mod_out()) == 0 {
                    stride_out /= *n;
                    *m = *m / self.mod_out() * self.mod_in();
                } else if *n % stride_out == 0 && *n % (stride_out * self.mod_out()) == 0 {
                    stride_out *= *m;
                    *n = *n / self.mod_out() * self.mod_in();
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
        if stride_in != 1 && self.mod_in() != 1 {
            shifted_items.push(Self::new_transpose(stride_in, self.mod_in()));
        }
        shifted_items.extend(queue);
        if stride_out != 1 && self.mod_in() != 1 {
            shifted_items.push(Self::new_transpose(self.mod_in(), stride_out));
        }
        let outer_transpose = if self.mod_out() == 1 || stride_out == 1 {
            None
        } else {
            Some(Transpose::new(stride_out, self.mod_out()))
        };
        Some((outer_transpose, target, shifted_items))
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
            Elementary::Transpose(var) => var.$fn($($arg),*),
            Elementary::Take(var) => var.$fn($($arg),*),
            Elementary::Slice(var) => var.$fn($($arg),*),
            Elementary::Children(var) => var.$fn($($arg),*),
            Elementary::Edges(var) => var.$fn($($arg),*),
            Elementary::UniformPoints(var) => var.$fn($($arg),*),
        }
    };
}

impl UnboundedMap for Elementary {
    dispatch! {fn dim_in(&self) -> usize}
    dispatch! {fn delta_dim(&self) -> usize}
    dispatch! {fn mod_in(&self) -> usize}
    dispatch! {fn mod_out(&self) -> usize}
    dispatch! {fn apply_inplace(&self, index: usize, coords: &mut[f64], stride: usize, offset: usize) -> usize}
    dispatch! {fn apply_index(&self, index: usize) -> usize}
    dispatch! {fn apply_indices_inplace(&self, indices: &mut [usize])}
    dispatch! {fn unapply_indices<T: UnapplyIndicesData>(&self, indices: &[T]) -> Vec<T>}
    dispatch! {fn is_identity(&self) -> bool}
}

impl AddOffset for Elementary {
    dispatch! {fn add_offset(&mut self, offset: usize)}
}

impl std::fmt::Debug for Elementary {
    dispatch! {fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result}
}

impl From<Transpose> for Elementary {
    fn from(transpose: Transpose) -> Self {
        Self::Transpose(transpose)
    }
}

impl From<Take> for Elementary {
    fn from(take: Take) -> Self {
        Self::Take(take)
    }
}

impl From<Slice> for Elementary {
    fn from(slice: Slice) -> Self {
        Self::Slice(slice)
    }
}

impl From<Children> for Elementary {
    fn from(children: Children) -> Self {
        Self::Children(Offset(children, 0))
    }
}

impl From<Edges> for Elementary {
    fn from(edges: Edges) -> Self {
        Self::Edges(Offset(edges, 0))
    }
}

impl From<UniformPoints> for Elementary {
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
            .fold((1, 1), |(o, i), map| comp_mod_out_in(map, o, i))
            .1
    }
    #[inline]
    fn mod_out(&self) -> usize {
        self.iter()
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
        self.iter().fold(index, |index, map| {
            map.apply_inplace(index, coords, stride, offset)
        })
    }
    #[inline]
    fn apply_index(&self, index: usize) -> usize {
        self.iter().fold(index, |index, map| map.apply_index(index))
    }
    #[inline]
    fn apply_indices_inplace(&self, indices: &mut [usize]) {
        self.iter()
            .for_each(|map| map.apply_indices_inplace(indices));
    }
    #[inline]
    fn unapply_indices<T: UnapplyIndicesData>(&self, indices: &[T]) -> Vec<T> {
        let mut iter = self.iter().rev();
        if let Some(map) = iter.next() {
            iter.fold(map.unapply_indices(indices), |indices, map| {
                map.unapply_indices(&indices)
            })
        } else {
            Vec::new()
        }
    }
    #[inline]
    fn is_identity(&self) -> bool {
        self.iter().all(|map| map.is_identity())
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
        let item = Elementary::new_transpose(3, 2);
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
        let item = Elementary::new_take(vec![4, 1, 2], 5);
        assert_map_apply!(item, 0, 4);
        assert_map_apply!(item, 1, 1);
        assert_map_apply!(item, 2, 2);
        assert_map_apply!(item, 3, 9);
        assert_map_apply!(item, 4, 6);
        assert_map_apply!(item, 5, 7);
    }

    #[test]
    fn apply_children_line() {
        let mut item = Elementary::new_children(Line);
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
        let mut item = Elementary::new_edges(Line);
        assert_map_apply!(item, 0, [[]], 0, [[1.0]]);
        assert_map_apply!(item, 1, [[]], 0, [[0.0]]);
        assert_map_apply!(item, 2, [[]], 1, [[1.0]]);

        item.add_offset(1);
        assert_map_apply!(item, 0, [[0.2]], 0, [[0.2, 1.0]]);
    }

    #[test]
    fn apply_uniform_points() {
        let mut item = Elementary::new_uniform_points(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2);
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
        assert_unapply!(Elementary::new_transpose(3, 2));
    }

    #[test]
    fn unapply_indices_take() {
        assert_unapply!(Elementary::new_take(vec![4, 1], 5));
    }

    #[test]
    fn unapply_indices_children() {
        assert_unapply!(Elementary::new_children(Triangle));
    }

    #[test]
    fn unapply_indices_edges() {
        assert_unapply!(Elementary::new_edges(Triangle));
    }

    #[test]
    fn unapply_indices_uniform_points() {
        assert_unapply!(Elementary::new_uniform_points(
            vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
            2
        ));
    }

    macro_rules! assert_equiv_maps {
        ($a:expr, $b:expr $(, $simplex:ident)*) => {{
            let a: &[Elementary] = &$a;
            let b: &[Elementary] = &$b;
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

    macro_rules! assert_shift_inner_to_outer {
        ($($item:expr),*; $($simplex:ident),*) => {{
            let unshifted = [$(Elementary::from($item),)*];
            let (ltrans, litem, lchain) = unshifted.first().unwrap().shift_inner_to_outer(&unshifted[1..]).unwrap();
            let mut shifted: Vec<Elementary> = Vec::new();
            shifted.extend(lchain.into_iter());
            shifted.push(litem);
            if let Some(ltrans) = ltrans {
                shifted.push(ltrans.into());
            }
            assert_equiv_maps!(&shifted[..], &unshifted[..] $(, $simplex)*);
        }};
    }

    #[test]
    fn shift_inner_to_outer() {
        assert_shift_inner_to_outer!(
            Elementary::new_take(vec![0, 1], 3), Transpose::new(4, 3);
        );
        assert_shift_inner_to_outer!(
            Elementary::new_take(vec![0, 1], 3), Transpose::new(5, 4*3), Transpose::new(3, 5);
        );
        assert_shift_inner_to_outer!(
            Elementary::new_take(vec![0, 1], 3), Transpose::new(5*4, 3), Transpose::new(5, 4);
        );
        assert_shift_inner_to_outer!(
            Elementary::new_children(Line),
            {let mut elem = Elementary::new_children(Line); elem.add_offset(1); elem};
            Line, Line
        );
        assert_shift_inner_to_outer!(
            Elementary::new_children(Line), Elementary::new_edges(Line);
            Line
        );
        assert_shift_inner_to_outer!(
            Elementary::new_children(Line),
            {let mut elem = Elementary::new_edges(Line); elem.add_offset(1); elem};
            Line
        );
        assert_shift_inner_to_outer!(
            Elementary::new_children(Line), Elementary::new_edges(Triangle);
            Line
        );
    }
}
