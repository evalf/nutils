use crate::infinite::InfiniteMapping;

pub trait Mapping {
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
    fn unapply_indices_unchecked(&self, indices: &[usize]) -> Vec<usize>;
    fn unapply_indices(&self, indices: &[usize]) -> Option<Vec<usize>> {
        if indices.iter().all(|index| *index < self.len_out()) {
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
            "The codomain of the first maping doesn't match the domain of the second mapping,"
        )
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Composition<M1: Mapping, M2: Mapping>(M1, M2);

impl<M1: Mapping, M2: Mapping> Composition<M1, M2> {
    pub fn new(mapping1: M1, mapping2: M2) -> Result<Self, ComposeCodomainDomainMismatch> {
        if mapping1.len_out() == mapping2.len_in() && mapping1.dim_out() == mapping2.dim_in() {
            Ok(Self(mapping1, mapping2))
        } else {
            Err(ComposeCodomainDomainMismatch)
        }
    }
}

impl<M1: Mapping, M2: Mapping> Mapping for Composition<M1, M2> {
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
    fn unapply_indices_unchecked(&self, indices: &[usize]) -> Vec<usize> {
        let indices = self.1.unapply_indices_unchecked(indices);
        self.0.unapply_indices_unchecked(&indices)
    }
    fn is_identity(&self) -> bool {
        self.0.is_identity() && self.1.is_identity()
    }
}

pub trait Compose: Mapping + Sized {
    fn compose<Rhs: Mapping>(self, rhs: Rhs) -> Result<Composition<Self, Rhs>, ComposeCodomainDomainMismatch> {
        Composition::new(self, rhs)
    }
}

impl<M: Mapping> Compose for M {}

#[derive(Debug, Clone, PartialEq)]
pub struct Concatenation<Item: Mapping> {
    dim_in: usize,
    delta_dim: usize,
    len_out: usize,
    len_in: usize,
    items: Vec<Item>,
}

impl<Item: Mapping> Concatenation<Item> {
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

impl<Item: Mapping> Mapping for Concatenation<Item> {
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
    fn unapply_indices_unchecked(&self, indices: &[usize]) -> Vec<usize> {
        unimplemented! {}
    }
    fn is_identity(&self) -> bool {
        false
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NewBoundedError {
    DimensionTooSmall,
    LengthNotAMultipleOfRepetition,
}

impl std::error::Error for NewBoundedError {}

impl std::fmt::Display for NewBoundedError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::DimensionTooSmall => write!(f, "The dimension of the sized mapping is smaller than the minimum dimension of the unsized mapping."),
            Self::LengthNotAMultipleOfRepetition => write!(f, "The length of the sized mapping is not a multiple of the repetition length of the unsized mapping."),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Bounded<M: InfiniteMapping> {
    mapping: M,
    delta_dim: usize,
    dim_in: usize,
    len_out: usize,
    len_in: usize,
}

impl<M: InfiniteMapping> Bounded<M> {
    pub fn new(mapping: M, dim_out: usize, len_out: usize) -> Result<Self, NewBoundedError> {
        if dim_out < mapping.dim_out() {
            Err(NewBoundedError::DimensionTooSmall)
        } else if len_out % mapping.mod_out() != 0 {
            Err(NewBoundedError::LengthNotAMultipleOfRepetition)
        } else {
            let delta_dim = mapping.delta_dim();
            let dim_in = dim_out - delta_dim;
            let len_in = len_out / mapping.mod_out() * mapping.mod_in();
            Ok(Self {
                mapping,
                delta_dim,
                dim_in,
                len_out,
                len_in,
            })
        }
    }
    pub fn get_unsized(&self) -> &M {
        &self.mapping
    }
}

impl<M: InfiniteMapping> Mapping for Bounded<M> {
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
        self.mapping.add_offset(offset);
        self.dim_in += offset;
    }
    fn apply_inplace_unchecked(&self, index: usize, coordinates: &mut [f64]) -> usize {
        self.mapping
            .apply_inplace(index, coordinates, self.dim_out())
    }
    fn apply_index_unchecked(&self, index: usize) -> usize {
        self.mapping.apply_index(index)
    }
    fn apply_indices_inplace_unchecked(&self, indices: &mut [usize]) {
        self.mapping.apply_indices_inplace(indices)
    }
    fn unapply_indices_unchecked(&self, indices: &[usize]) -> Vec<usize> {
        self.mapping.unapply_indices(indices)
    }
    fn is_identity(&self) -> bool {
        self.mapping.is_identity()
    }
}
