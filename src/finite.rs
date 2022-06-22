use crate::infinite::{UnboundedMap, Elementary};
use crate::UnapplyIndicesData;

pub trait BoundedMap {
    fn len_out(&self) -> usize;
    fn len_in(&self) -> usize;
    fn dim_out(&self) -> usize {
        self.dim_in() + self.delta_dim()
    }
    fn dim_in(&self) -> usize;
    fn delta_dim(&self) -> usize;
    fn add_offset(&mut self, offset: usize);
    fn apply_inplace_unchecked(&self, index: usize, coordinates: &mut [f64]) -> usize;
    fn apply_inplace(&self, index: usize, coordinates: &mut [f64]) -> Option<usize> {
        if index < self.len_in() {
            Some(self.apply_inplace_unchecked(index, coordinates))
        } else {
            None
        }
    }
    fn apply_index_unchecked(&self, index: usize) -> usize;
    fn apply_index(&self, index: usize) -> Option<usize> {
        if index < self.len_in() {
            Some(self.apply_index_unchecked(index))
        } else {
            None
        }
    }
    fn apply_indices_inplace_unchecked(&self, indices: &mut [usize]) {
        for index in indices.iter_mut() {
            *index = self.apply_index_unchecked(*index);
        }
    }
    fn apply_indices(&self, indices: &[usize]) -> Option<Vec<usize>> {
        if indices.iter().all(|index| *index < self.len_in()) {
            let mut indices = indices.to_vec();
            self.apply_indices_inplace_unchecked(&mut indices);
            Some(indices)
        } else {
            None
        }
    }
    fn unapply_indices_unchecked<T: UnapplyIndicesData>(&self, indices: &[T]) -> Vec<T>;
    fn unapply_indices<T: UnapplyIndicesData>(&self, indices: &[T]) -> Option<Vec<T>> {
        if indices.iter().all(|index| index.last() < self.len_out()) {
            Some(self.unapply_indices_unchecked(indices))
        } else {
            None
        }
    }
    fn is_identity(&self) -> bool;
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
pub struct Composition<M1: BoundedMap, M2: BoundedMap>(M1, M2);

impl<M1: BoundedMap, M2: BoundedMap> Composition<M1, M2> {
    pub fn new(map1: M1, map2: M2) -> Result<Self, ComposeCodomainDomainMismatch> {
        if map1.len_out() == map2.len_in() && map1.dim_out() == map2.dim_in() {
            Ok(Self(map1, map2))
        } else {
            Err(ComposeCodomainDomainMismatch)
        }
    }
}

impl<M1: BoundedMap, M2: BoundedMap> BoundedMap for Composition<M1, M2> {
    fn len_in(&self) -> usize {
        self.0.len_in()
    }
    fn len_out(&self) -> usize {
        self.1.len_out()
    }
    fn dim_in(&self) -> usize {
        self.0.dim_in()
    }
    fn delta_dim(&self) -> usize {
        self.0.delta_dim() + self.1.delta_dim()
    }
    fn add_offset(&mut self, offset: usize) {
        self.0.add_offset(offset);
        self.1.add_offset(offset);
    }
    fn apply_inplace_unchecked(&self, index: usize, coordinates: &mut [f64]) -> usize {
        let index = self.0.apply_inplace_unchecked(index, coordinates);
        self.1.apply_inplace_unchecked(index, coordinates)
    }
    fn apply_index_unchecked(&self, index: usize) -> usize {
        let index = self.0.apply_index_unchecked(index);
        self.1.apply_index_unchecked(index)
    }
    fn apply_indices_inplace_unchecked(&self, indices: &mut [usize]) {
        self.0.apply_indices_inplace_unchecked(indices);
        self.1.apply_indices_inplace_unchecked(indices);
    }
    fn unapply_indices_unchecked<T: UnapplyIndicesData>(&self, indices: &[T]) -> Vec<T> {
        let indices = self.1.unapply_indices_unchecked(indices);
        self.0.unapply_indices_unchecked(&indices)
    }
    fn is_identity(&self) -> bool {
        self.0.is_identity() && self.1.is_identity()
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
pub struct Concatenation<Item: BoundedMap> {
    dim_in: usize,
    delta_dim: usize,
    len_out: usize,
    len_in: usize,
    items: Vec<Item>,
}

impl<Item: BoundedMap> Concatenation<Item> {
    pub fn new(items: Vec<Item>) -> Self {
        // TODO: Return `Result<Self, ...>`.
        let first = items.first().unwrap();
        let dim_in = first.dim_in();
        let delta_dim = first.delta_dim();
        let len_out = first.len_out();
        let mut len_in = 0;
        for item in items.iter() {
            assert_eq!(item.dim_in(), dim_in);
            assert_eq!(item.delta_dim(), delta_dim);
            assert_eq!(item.len_out(), len_out);
            len_in += item.len_in();
        }
        Self {
            dim_in,
            delta_dim,
            len_out,
            len_in,
            items,
        }
    }
    fn resolve_item_unchecked(&self, mut index: usize) -> (&Item, usize) {
        for item in self.items.iter() {
            if index < item.len_in() {
                return (item, index);
            }
            index -= item.len_in();
        }
        panic!("index out of range");
    }
    pub fn iter(&self) -> impl Iterator<Item = &Item> {
        self.items.iter()
    }
}

impl<Item: BoundedMap> BoundedMap for Concatenation<Item> {
    fn dim_in(&self) -> usize {
        self.dim_in
    }
    fn delta_dim(&self) -> usize {
        self.delta_dim
    }
    fn len_out(&self) -> usize {
        self.len_out
    }
    fn len_in(&self) -> usize {
        self.len_in
    }
    fn add_offset(&mut self, offset: usize) {
        for item in self.items.iter_mut() {
            item.add_offset(offset);
        }
        self.dim_in += offset;
    }
    fn apply_inplace_unchecked(&self, index: usize, coordinates: &mut [f64]) -> usize {
        let (item, index) = self.resolve_item_unchecked(index);
        item.apply_inplace_unchecked(index, coordinates)
    }
    fn apply_index_unchecked(&self, index: usize) -> usize {
        let (item, index) = self.resolve_item_unchecked(index);
        item.apply_index_unchecked(index)
    }
    fn unapply_indices_unchecked<T: UnapplyIndicesData>(&self, indices: &[T]) -> Vec<T> {
        unimplemented! {}
    }
    fn is_identity(&self) -> bool {
        false
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NewWithBoundsError {
    DimensionTooSmall,
    LengthNotAMultipleOfRepetition,
}

impl std::error::Error for NewWithBoundsError {}

impl std::fmt::Display for NewWithBoundsError {
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

impl<M: UnboundedMap> WithBounds<M> {
    pub fn new(map: M, dim_out: usize, len_out: usize) -> Result<Self, NewWithBoundsError> {
        if dim_out < map.dim_out() {
            Err(NewWithBoundsError::DimensionTooSmall)
        } else if len_out % map.mod_out() != 0 {
            Err(NewWithBoundsError::LengthNotAMultipleOfRepetition)
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
    pub fn get_infinite(&self) -> &M {
        &self.map
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
    fn apply_inplace_unchecked(&self, index: usize, coordinates: &mut [f64]) -> usize {
        self.map.apply_inplace(index, coordinates, self.dim_out())
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
