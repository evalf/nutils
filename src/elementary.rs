use crate::finite_f64::FiniteF64;
use crate::simplex::Simplex;
use crate::{AddOffset, UnapplyIndicesData, UnboundedMap};
use num::Integer as _;
use std::rc::Rc;

#[inline]
const fn divmod(x: usize, y: usize) -> (usize, usize) {
    (x / y, x % y)
}

fn coordinates_iter_mut(
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

#[derive(Debug, Clone, Copy, PartialEq)]
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
        _coordinates: &mut [f64],
        _stride: usize,
        _offset: usize,
    ) -> usize {
        index
    }
    fn apply_index(&self, index: usize) -> usize {
        index
    }
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
        _coordinates: &mut [f64],
        _stride: usize,
        _offset: usize,
    ) -> usize {
        self.apply_index(index)
    }
    fn apply_index(&self, index: usize) -> usize {
        let (j, k) = divmod(index, self.1);
        let (i, j) = divmod(j, self.0);
        (i * self.1 + k) * self.0 + j
    }
    fn unapply_indices<T: UnapplyIndicesData>(&self, indices: &[T]) -> Vec<T> {
        indices
            .iter()
            .map(|index| {
                let (j, k) = divmod(index.last(), self.0);
                let (i, j) = divmod(j, self.1);
                index.push((i * self.0 + k) * self.1 + j)
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
        let indices = indices.into();
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
        _coordinates: &mut [f64],
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
                let (j, iout) = divmod(index.last(), self.len);
                let offset = j * self.nindices;
                self.indices
                    .iter()
                    .position(|i| *i == iout)
                    .map(|iin| index.push(offset + iin))
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
        _coordinates: &mut [f64],
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
                let (j, i) = divmod(index.last(), self.len_out);
                (self.start..self.start + self.len_in)
                    .contains(&i)
                    .then(|| index.push(i - self.start + j * self.len_in))
            })
            .collect()
    }
}

impl AddOffset for Slice {
    fn add_offset(&mut self, _offset: usize) {}
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Children(Simplex, usize);

impl Children {
    pub fn new(simplex: Simplex) -> Self {
        Self(simplex, 0)
    }
}

impl UnboundedMap for Children {
    fn dim_in(&self) -> usize {
        self.0.dim() + self.1
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
        coordinates: &mut [f64],
        stride: usize,
        offset: usize,
    ) -> usize {
        self.0
            .apply_child(index, coordinates, stride, offset + self.1)
    }
    fn apply_index(&self, index: usize) -> usize {
        self.0.apply_child_index(index)
    }
    fn unapply_indices<T: UnapplyIndicesData>(&self, indices: &[T]) -> Vec<T> {
        indices
            .iter()
            .flat_map(|i| (0..self.mod_in()).map(move |j| i.push(i.last() * self.mod_in() + j)))
            .collect()
    }
}

impl AddOffset for Children {
    fn add_offset(&mut self, offset: usize) {
        self.1 += offset;
    }
}

impl From<Simplex> for Children {
    fn from(simplex: Simplex) -> Children {
        Children::new(simplex)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Edges(pub Simplex, pub usize);

impl Edges {
    pub fn new(simplex: Simplex) -> Self {
        Self(simplex, 0)
    }
}

impl UnboundedMap for Edges {
    fn dim_in(&self) -> usize {
        self.0.edge_dim() + self.1
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
        coordinates: &mut [f64],
        stride: usize,
        offset: usize,
    ) -> usize {
        self.0
            .apply_edge(index, coordinates, stride, offset + self.1)
    }
    fn apply_index(&self, index: usize) -> usize {
        self.0.apply_edge_index(index)
    }
    fn unapply_indices<T: UnapplyIndicesData>(&self, indices: &[T]) -> Vec<T> {
        indices
            .iter()
            .flat_map(|i| (0..self.mod_in()).map(move |j| i.push(i.last() * self.mod_in() + j)))
            .collect()
    }
}

impl AddOffset for Edges {
    fn add_offset(&mut self, offset: usize) {
        self.1 += offset;
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
    offset: usize,
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
            offset: 0,
        }
    }
}

impl UnboundedMap for UniformPoints {
    fn dim_in(&self) -> usize {
        self.offset
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
        for coord in coordinates_iter_mut(coords, stride, offset + self.offset, self.point_dim, 0) {
            coord.copy_from_slice(point);
        }
        index / self.npoints
    }
    fn apply_index(&self, index: usize) -> usize {
        index / self.npoints
    }
    fn unapply_indices<T: UnapplyIndicesData>(&self, indices: &[T]) -> Vec<T> {
        indices
            .iter()
            .flat_map(|i| (0..self.mod_in()).map(move |j| i.push(i.last() * self.mod_in() + j)))
            .collect()
    }
}

impl AddOffset for UniformPoints {
    fn add_offset(&mut self, offset: usize) {
        self.offset += offset;
    }
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum Elementary {
    Transpose(Transpose),
    Take(Take),
    Slice(Slice),
    Children(Children),
    Edges(Edges),
    UniformPoints(UniformPoints),
}

impl Elementary {
    #[inline]
    pub fn new_transpose(len1: usize, len2: usize) -> Self {
        Self::Transpose(Transpose::new(len1, len2))
    }
    #[inline]
    pub fn new_take(indices: impl Into<Rc<[usize]>>, len: usize) -> Self {
        Self::Take(Take::new(indices, len))
    }
    #[inline]
    pub fn new_slice(start: usize, len_in: usize, len_out: usize) -> Self {
        Self::Slice(Slice::new(start, len_in, len_out))
    }
    #[inline]
    pub fn new_children(simplex: Simplex) -> Self {
        Self::Children(Children::new(simplex))
    }
    #[inline]
    pub fn new_edges(simplex: Simplex) -> Self {
        Self::Edges(Edges::new(simplex))
    }
    #[inline]
    pub fn new_uniform_points(points: impl Into<Rc<[f64]>>, point_dim: usize) -> Self {
        Self::UniformPoints(UniformPoints::new(points, point_dim))
    }
    #[inline]
    fn offset_mut(&mut self) -> Option<&mut usize> {
        match self {
            Self::Children(Children(_, ref mut offset)) => Some(offset),
            Self::Edges(Edges(_, ref mut offset)) => Some(offset),
            Self::UniformPoints(UniformPoints { ref mut offset, .. }) => Some(offset),
            _ => None,
        }
    }
    #[inline]
    fn set_offset(&mut self, new_offset: usize) {
        self.add_offset(new_offset);
    }
    #[inline]
    pub fn with_offset(mut self, new_offset: usize) -> Self {
        self.set_offset(new_offset);
        self
    }
    //pub fn swap(&mut self, other: &Self) -> Option<Vec<Self>> {
    //    if self.mod_out() == 1 {
    //
    //    } else {
    //        None
    //    }
    //}
    pub fn shift_left(&self, items: &[Self]) -> Option<(Option<Transpose>, Self, Vec<Self>)> {
        if self.is_transpose() {
            return None;
        }
        let mut target = self.clone();
        let mut shifted_items: Vec<Self> = Vec::new();
        let mut queue: Vec<Self> = Vec::new();
        let mut stride_out = 1;
        let mut stride_in = 1;
        for mut item in items.iter().rev().cloned() {
            // Swap matching edges and children at the same offset.
            if let Self::Edges(Edges(esimplex, eoffset)) = &item {
                if let Self::Children(Children(ref mut csimplex, coffset)) = &mut target {
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
                        shifted_items.push(Self::Edges(Edges(*esimplex, *eoffset)));
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
        let leading_transpose = if self.mod_out() == 1 || stride_out == 1 {
            None
        } else {
            Some(Transpose::new(stride_out, self.mod_out()))
        };
        shifted_items.reverse();
        Some((leading_transpose, target, shifted_items))
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
    dispatch! {fn apply_inplace(&self, index: usize, coordinates: &mut[f64], stride: usize, offset: usize) -> usize}
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
        Self::Children(children)
    }
}

impl From<Edges> for Elementary {
    fn from(edges: Edges) -> Self {
        Self::Edges(edges)
    }
}

impl From<UniformPoints> for Elementary {
    fn from(uniform_points: UniformPoints) -> Self {
        Self::UniformPoints(uniform_points)
    }
}

pub trait PushElementary {
    fn push_elementary(&mut self, map: &Elementary);
    fn clone_and_push_elementary(&self, map: &Elementary) -> Self
    where
        Self: Clone,
    {
        let mut cloned = self.clone();
        cloned.push_elementary(map);
        cloned
    }
}

impl PushElementary for Vec<Elementary> {
    fn push_elementary(&mut self, map: &Elementary) {
        self.push(map.clone());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use std::iter;
    use Simplex::*;

    macro_rules! assert_eq_apply {
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
        assert_eq_apply!(item, 0, 0);
        assert_eq_apply!(item, 1, 3);
        assert_eq_apply!(item, 2, 1);
        assert_eq_apply!(item, 3, 4);
        assert_eq_apply!(item, 4, 2);
        assert_eq_apply!(item, 5, 5);
        assert_eq_apply!(item, 6, 6);
        assert_eq_apply!(item, 7, 9);
    }

    #[test]
    fn apply_take() {
        let item = Elementary::new_take(vec![4, 1, 2], 5);
        assert_eq_apply!(item, 0, 4);
        assert_eq_apply!(item, 1, 1);
        assert_eq_apply!(item, 2, 2);
        assert_eq_apply!(item, 3, 9);
        assert_eq_apply!(item, 4, 6);
        assert_eq_apply!(item, 5, 7);
    }

    #[test]
    fn apply_children_line() {
        let item = Elementary::new_children(Line);
        assert_eq_apply!(item, 0, [[0.0], [1.0]], 0, [[0.0], [0.5]]);
        assert_eq_apply!(item, 1, [[0.0], [1.0]], 0, [[0.5], [1.0]]);
        assert_eq_apply!(item, 2, [[0.0], [1.0]], 1, [[0.0], [0.5]]);

        let item = item.with_offset(1);
        assert_eq_apply!(
            item,
            3,
            [[0.2, 0.0], [0.3, 1.0]],
            1,
            [[0.2, 0.5], [0.3, 1.0]]
        );
    }

    #[test]
    fn apply_edges_line() {
        let item = Elementary::new_edges(Line);
        assert_eq_apply!(item, 0, [[]], 0, [[1.0]]);
        assert_eq_apply!(item, 1, [[]], 0, [[0.0]]);
        assert_eq_apply!(item, 2, [[]], 1, [[1.0]]);

        let item = item.with_offset(1);
        assert_eq_apply!(item, 0, [[0.2]], 0, [[0.2, 1.0]]);
    }

    #[test]
    fn apply_uniform_points() {
        let item = Elementary::new_uniform_points(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2);
        assert_eq_apply!(item, 0, [[]], 0, [[1.0, 2.0]]);
        assert_eq_apply!(item, 1, [[]], 0, [[3.0, 4.0]]);
        assert_eq_apply!(item, 2, [[]], 0, [[5.0, 6.0]]);
        assert_eq_apply!(item, 3, [[]], 1, [[1.0, 2.0]]);

        let item = item.with_offset(1);
        assert_eq_apply!(item, 0, [[7.0]], 0, [[7.0, 1.0, 2.0]]);
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
            println!("a: {a:?}");
            println!("b: {b:?}");
            let tip_dim = 0 $(+ $simplex.dim())*;
            // Determine the length of the sequence and the root dimension for sequence `a`.
            let mut tip_len = 1;
            let mut root_len = 1;
            let mut root_dim = tip_dim;
            for item in a.iter().rev() {
                let i = (1..)
                    .into_iter()
                    .find(|i| (root_len * i) % item.mod_in() == 0)
                    .unwrap();
                tip_len *= i;
                root_len *= i;
                root_len = root_len / item.mod_in() * item.mod_out();
                assert!(item.dim_in() <= root_dim);
                root_dim += item.delta_dim();
            }
            assert!(tip_len > 0);
            // Verify the length and the root dimension for sequence `b`.
            let mut root_len_b = tip_len;
            let mut root_dim_b = tip_dim;
            for item in b.iter().rev() {
                assert_eq!(root_len_b % item.mod_in(), 0);
                root_len_b = root_len_b / item.mod_in() * item.mod_out();
                assert!(item.dim_in() <= root_dim_b);
                root_dim_b += item.delta_dim();
            }
            assert_eq!(root_len_b, root_len);
            assert_eq!(root_dim_b, root_dim);
            // Build coords: the outer product of the vertices of the given simplices, zero-padded
            // to the dimension of the root.
            let coords = iter::once([]);
            $(
                let coords = coords.flat_map(|coord| {
                    $simplex
                        .vertices()
                        .chunks($simplex.dim())
                        .map(move |vert| [&coord, vert].concat())
                    });
            )*
            let pad: Vec<f64> = iter::repeat(0.0).take(root_dim - tip_dim).collect();
            let coords: Vec<f64> = coords.flat_map(|coord| [&coord[..], &pad].concat()).collect();
            // Test if every tip index and coordinate maps to the same root index and coordinate.
            for itip in 0..2 * tip_len {
                let mut crds_a = coords.clone();
                let mut crds_b = coords.clone();
                let iroot_a = a.iter().rev().fold(itip, |i, item| item.apply_inplace(i, &mut crds_a, root_dim, 0));
                let iroot_b = b.iter().rev().fold(itip, |i, item| item.apply_inplace(i, &mut crds_b, root_dim, 0));
                assert_eq!(iroot_a, iroot_b, "itip={itip}");
                assert_abs_diff_eq!(crds_a[..], crds_b[..]);
            }
        }};
    }

    macro_rules! assert_shift_left {
        ($($item:expr),*; $($simplex:ident),*) => {{
            let unshifted = [$(Elementary::from($item),)*];
            let (ltrans, litem, lchain) = unshifted.last().unwrap().shift_left(&unshifted[..unshifted.len()-1]).unwrap();
            let mut shifted: Vec<Elementary> = Vec::new();
            if let Some(ltrans) = ltrans {
                shifted.push(ltrans.into());
            }
            shifted.push(litem);
            shifted.extend(lchain.into_iter());
            assert_equiv_maps!(&shifted[..], &unshifted[..] $(, $simplex)*);
        }};
    }

    #[test]
    fn shift_left() {
        assert_shift_left!(
            Transpose::new(4, 3), Elementary::new_take(vec![0, 1], 3);
        );
        assert_shift_left!(
            Transpose::new(3, 5), Transpose::new(5, 4*3), Elementary::new_take(vec![0, 1], 3);
        );
        assert_shift_left!(
            Transpose::new(5, 4), Transpose::new(5*4, 3), Elementary::new_take(vec![0, 1], 3);
        );
        assert_shift_left!(
            Elementary::new_children(Line).with_offset(1), Elementary::new_children(Line);
            Line, Line
        );
        assert_shift_left!(
            Elementary::new_edges(Line), Elementary::new_children(Line);
            Line
        );
        assert_shift_left!(
            Elementary::new_edges(Line).with_offset(1), Elementary::new_children(Line);
            Line
        );
        assert_shift_left!(
            Elementary::new_edges(Triangle), Elementary::new_children(Line);
            Line
        );
    }
}
