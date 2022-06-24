use crate::elementary::{Elementary, PushElementary};
use crate::{BoundedMap, UnapplyIndicesData, UnboundedMap};
use num::Integer as _;
use std::ops::{Deref, DerefMut};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WithBoundsError {
    DimensionTooSmall,
    LengthNotAMultipleOfRepetition,
}

impl std::error::Error for WithBoundsError {}

impl std::fmt::Display for WithBoundsError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::DimensionTooSmall => write!(f, "The dimension of the sized map is smaller than the minimum dimension of the unsized map."),
            Self::LengthNotAMultipleOfRepetition => write!(f, "The length of the sized map is not a multiple of the repetition length of the unsized map."),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct WithBounds<M: UnboundedMap> {
    map: M,
    delta_dim: usize,
    dim_in: usize,
    len_out: usize,
    len_in: usize,
}

impl<M: UnboundedMap + std::fmt::Debug> WithBounds<M> {
    pub fn new(map: M, dim_out: usize, len_out: usize) -> Result<Self, WithBoundsError> {
        if dim_out < map.dim_out() {
            Err(WithBoundsError::DimensionTooSmall)
        } else if len_out % map.mod_out() != 0 {
            Err(WithBoundsError::LengthNotAMultipleOfRepetition)
        } else {
            Ok(Self::new_unchecked(map, dim_out, len_out))
        }
    }
    pub(crate) fn new_unchecked(map: M, dim_out: usize, len_out: usize) -> Self {
        let delta_dim = map.delta_dim();
        let dim_in = dim_out - delta_dim;
        let len_in = len_out / map.mod_out() * map.mod_in();
        Self {
            map,
            delta_dim,
            dim_in,
            len_out,
            len_in,
        }
    }
    pub fn get_unbounded(&self) -> &M {
        &self.map
    }
    pub fn into_unbounded(self) -> M {
        self.map
    }
}

impl<M: UnboundedMap> BoundedMap for WithBounds<M> {
    fn len_in(&self) -> usize {
        self.len_in
    }
    fn len_out(&self) -> usize {
        self.len_out
    }
    fn dim_in(&self) -> usize {
        self.dim_in
    }
    fn delta_dim(&self) -> usize {
        self.delta_dim
    }
    fn add_offset(&mut self, offset: usize) {
        self.map.add_offset(offset);
        self.dim_in += offset;
    }
    fn apply_inplace_unchecked(
        &self,
        index: usize,
        coordinates: &mut [f64],
        stride: usize,
    ) -> usize {
        self.map.apply_inplace(index, coordinates, stride)
    }
    fn apply_index_unchecked(&self, index: usize) -> usize {
        self.map.apply_index(index)
    }
    fn apply_indices_inplace_unchecked(&self, indices: &mut [usize]) {
        self.map.apply_indices_inplace(indices)
    }
    fn unapply_indices_unchecked<T: UnapplyIndicesData>(&self, indices: &[T]) -> Vec<T> {
        self.map.unapply_indices(indices)
    }
    fn is_identity(&self) -> bool {
        self.map.is_identity()
    }
}

impl<M: UnboundedMap + PushElementary> PushElementary for WithBounds<M> {
    fn push_elementary(&mut self, item: &Elementary) {
        // TODO: return an error if we push something that causes `dim_in` to drop below zero
        // or with incompatible length.
        assert!(self.dim_in >= item.delta_dim());
        self.delta_dim += item.delta_dim();
        self.dim_in -= item.delta_dim();
        if item.mod_out() > 0 {
            assert_eq!(self.len_in % item.mod_out(), 0);
            self.len_in = self.len_in / item.mod_out() * item.mod_in();
        } else {
            self.len_in = 0;
        }
        self.map.push_elementary(item);
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ComposeCodomainDomainMismatch;

impl std::error::Error for ComposeCodomainDomainMismatch {}

impl std::fmt::Display for ComposeCodomainDomainMismatch {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "The codomain of the first map doesn't match the domain of the second map,"
        )
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Composition<Inner, Outer> {
    inner: Inner,
    outer: Outer,
}

impl<Inner: BoundedMap, Outer: BoundedMap> Composition<Inner, Outer> {
    pub fn new(inner: Inner, outer: Outer) -> Result<Self, ComposeCodomainDomainMismatch> {
        if inner.len_out() == outer.len_in() && inner.dim_out() == outer.dim_in() {
            Ok(Self { inner, outer })
        } else {
            Err(ComposeCodomainDomainMismatch)
        }
    }
}

impl<Inner: BoundedMap, Outer: BoundedMap> BoundedMap for Composition<Inner, Outer> {
    fn len_in(&self) -> usize {
        self.inner.len_in()
    }
    fn len_out(&self) -> usize {
        self.outer.len_out()
    }
    fn dim_in(&self) -> usize {
        self.inner.dim_in()
    }
    fn delta_dim(&self) -> usize {
        self.inner.delta_dim() + self.outer.delta_dim()
    }
    fn add_offset(&mut self, offset: usize) {
        self.inner.add_offset(offset);
        self.outer.add_offset(offset);
    }
    fn apply_inplace_unchecked(
        &self,
        index: usize,
        coordinates: &mut [f64],
        stride: usize,
    ) -> usize {
        let index = self
            .inner
            .apply_inplace_unchecked(index, coordinates, stride);
        self.outer
            .apply_inplace_unchecked(index, coordinates, stride)
    }
    fn apply_index_unchecked(&self, index: usize) -> usize {
        let index = self.inner.apply_index_unchecked(index);
        self.outer.apply_index_unchecked(index)
    }
    fn apply_indices_inplace_unchecked(&self, indices: &mut [usize]) {
        self.inner.apply_indices_inplace_unchecked(indices);
        self.outer.apply_indices_inplace_unchecked(indices);
    }
    fn unapply_indices_unchecked<T: UnapplyIndicesData>(&self, indices: &[T]) -> Vec<T> {
        let indices = self.outer.unapply_indices_unchecked(indices);
        self.inner.unapply_indices_unchecked(&indices)
    }
    fn is_identity(&self) -> bool {
        self.inner.is_identity() && self.outer.is_identity()
    }
}

impl<Inner: UnboundedMap, Outer: UnboundedMap> UnboundedMap for Composition<Inner, Outer> {
    fn mod_in(&self) -> usize {
        update_mod_out_in(&self.outer, self.inner.mod_out(), self.inner.mod_in()).1
    }
    fn mod_out(&self) -> usize {
        update_mod_out_in(&self.outer, self.inner.mod_out(), self.inner.mod_in()).0
    }
    fn dim_in(&self) -> usize {
        update_dim_out_in(&self.outer, self.inner.dim_out(), self.inner.dim_in()).0
    }
    fn delta_dim(&self) -> usize {
        self.inner.delta_dim() + self.outer.delta_dim()
    }
    fn add_offset(&mut self, offset: usize) {
        self.inner.add_offset(offset);
        self.outer.add_offset(offset);
    }
    fn apply_inplace(&self, index: usize, coordinates: &mut [f64], stride: usize) -> usize {
        let index = self.inner.apply_inplace(index, coordinates, stride);
        self.outer.apply_inplace(index, coordinates, stride)
    }
    fn apply_index(&self, index: usize) -> usize {
        let index = self.inner.apply_index(index);
        self.outer.apply_index(index)
    }
    fn apply_indices_inplace(&self, indices: &mut [usize]) {
        self.inner.apply_indices_inplace(indices);
        self.outer.apply_indices_inplace(indices);
    }
    fn unapply_indices<T: UnapplyIndicesData>(&self, indices: &[T]) -> Vec<T> {
        let indices = self.outer.unapply_indices(indices);
        self.inner.unapply_indices(&indices)
    }
    fn is_identity(&self) -> bool {
        self.inner.is_identity() && self.outer.is_identity()
    }
}

pub trait Compose: BoundedMap + Sized {
    fn compose<Rhs: BoundedMap>(
        self,
        rhs: Rhs,
    ) -> Result<Composition<Self, Rhs>, ComposeCodomainDomainMismatch> {
        Composition::new(self, rhs)
    }
}

impl<M: BoundedMap> Compose for M {}

#[derive(Debug, Clone, PartialEq)]
pub struct Concatenation<Item: BoundedMap>(Vec<Item>);

impl<Item: BoundedMap> Concatenation<Item> {
    pub fn new(items: Vec<Item>) -> Self {
        // TODO: Return `Result<Self, ...>`.
        let first = items.first().unwrap();
        let dim_in = first.dim_in();
        let delta_dim = first.delta_dim();
        let len_out = first.len_out();
        for item in items.iter() {
            assert_eq!(item.dim_in(), dim_in);
            assert_eq!(item.delta_dim(), delta_dim);
            assert_eq!(item.len_out(), len_out);
        }
        Self(items)
    }
    fn resolve_item_unchecked(&self, mut index: usize) -> (&Item, usize) {
        for item in self.0.iter() {
            if index < item.len_in() {
                return (item, index);
            }
            index -= item.len_in();
        }
        panic!("index out of range");
    }
    pub fn iter(&self) -> impl Iterator<Item = &Item> {
        self.0.iter()
    }
    pub fn into_vec(self) -> Vec<Item> {
        self.0
    }
}

impl<Item: BoundedMap> BoundedMap for Concatenation<Item> {
    fn dim_in(&self) -> usize {
        self.0.first().unwrap().dim_in()
    }
    fn delta_dim(&self) -> usize {
        self.0.first().unwrap().delta_dim()
    }
    fn len_out(&self) -> usize {
        self.0.first().unwrap().len_out()
    }
    fn len_in(&self) -> usize {
        self.iter().map(|item| item.len_in()).sum()
    }
    fn add_offset(&mut self, offset: usize) {
        for item in &mut self.0 {
            item.add_offset(offset);
        }
    }
    fn apply_inplace_unchecked(
        &self,
        index: usize,
        coordinates: &mut [f64],
        stride: usize,
    ) -> usize {
        let (item, index) = self.resolve_item_unchecked(index);
        item.apply_inplace_unchecked(index, coordinates, stride)
    }
    fn apply_index_unchecked(&self, index: usize) -> usize {
        let (item, index) = self.resolve_item_unchecked(index);
        item.apply_index_unchecked(index)
    }
    fn unapply_indices_unchecked<T: UnapplyIndicesData>(&self, indices: &[T]) -> Vec<T> {
        let mut result = Vec::new();
        let mut offset = 0;
        for item in &self.0 {
            result.extend(
                item.unapply_indices_unchecked(indices)
                    .into_iter()
                    .map(|i| i.push(i.last() + offset)),
            );
            offset += item.len_in();
        }
        result
    }
    fn is_identity(&self) -> bool {
        false
    }
}

impl<M> PushElementary for Concatenation<M>
where
    M: BoundedMap + PushElementary,
{
    fn push_elementary(&mut self, map: &Elementary) {
        match map {
            Elementary::Children(_) | Elementary::Edges(_) => {
                for item in &mut self.0 {
                    item.push_elementary(map);
                }
            }
            Elementary::Take(take) => {
                let mut offset = 0;
                let indices = take.get_indices();
                let mut indices = indices.iter().cloned().peekable();
                for item in &mut self.0 {
                    let len = item.len_in();
                    let mut item_indices = Vec::new();
                    while let Some(index) = indices.next_if(|&i| i < offset + len) {
                        if index < offset {
                            unimplemented! {"take of concatenation with unordered indices"};
                        }
                        item_indices.push(index - offset);
                    }
                    item.push_elementary(&Elementary::new_take(item_indices, len));
                    offset += len;
                }
            }
            _ => unimplemented! {},
        }
    }
}

#[inline]
fn update_dim_out_in<M: UnboundedMap>(map: &M, dim_out: usize, dim_in: usize) -> (usize, usize) {
    if let Some(n) = map.dim_in().checked_sub(dim_out) {
        (map.dim_out(), dim_in + n)
    } else {
        (dim_out, dim_in)
    }
}

#[inline]
fn update_mod_out_in<M: UnboundedMap>(map: &M, mod_out: usize, mod_in: usize) -> (usize, usize) {
    let n = mod_out.lcm(&map.mod_in());
    (n / map.mod_in() * map.mod_out(), mod_in * n / mod_out)
}

/// Composition.
impl<Item, Array> UnboundedMap for Array
where
    Item: UnboundedMap,
    Array: Deref<Target = [Item]> + DerefMut + std::fmt::Debug,
{
    fn dim_in(&self) -> usize {
        self.deref()
            .iter()
            .rev()
            .fold((0, 0), |(dim_out, dim_in), item| {
                update_dim_out_in(item, dim_out, dim_in)
            })
            .1
    }
    fn delta_dim(&self) -> usize {
        self.iter().map(|item| item.delta_dim()).sum()
    }
    fn add_offset(&mut self, offset: usize) {
        for item in self.iter_mut() {
            item.add_offset(offset);
        }
    }
    fn mod_in(&self) -> usize {
        self.deref()
            .iter()
            .rev()
            .fold((1, 1), |(mod_out, mod_in), item| {
                update_mod_out_in(item, mod_out, mod_in)
            })
            .1
    }
    fn mod_out(&self) -> usize {
        self.deref()
            .iter()
            .rev()
            .fold((1, 1), |(mod_out, mod_in), item| {
                update_mod_out_in(item, mod_out, mod_in)
            })
            .0
    }
    fn apply_inplace(&self, index: usize, coordinates: &mut [f64], stride: usize) -> usize {
        self.iter().rev().fold(index, |index, item| {
            item.apply_inplace(index, coordinates, stride)
        })
    }
    fn apply_index(&self, index: usize) -> usize {
        self.iter()
            .rev()
            .fold(index, |index, item| item.apply_index(index))
    }
    fn apply_indices_inplace(&self, indices: &mut [usize]) {
        for item in self.iter().rev() {
            item.apply_indices_inplace(indices);
        }
    }
    fn unapply_indices<T: UnapplyIndicesData>(&self, indices: &[T]) -> Vec<T> {
        self.iter().fold(indices.to_vec(), |indices, item| {
            item.unapply_indices(&indices)
        })
    }
}
