use crate::{UnboundedMap, BoundedMap, UnapplyIndicesData, AddOffset};
use crate::ops::{Concatenation, WithBoundsError};
use crate::elementary::{Elementary, PushElementary};
use crate::simplex::Simplex;

#[derive(Debug, Clone, PartialEq)]
pub struct WithShape<M: UnboundedMap> {
    map: M,
    shapes: Vec<Simplex>,
    dim_in: usize,
    delta_dim: usize,
    len_out: usize,
    len_in: usize,
}

impl<M: UnboundedMap + PushElementary + Clone> WithShape<M> {
    pub fn new(map: M, shapes: Vec<Simplex>, len_in: usize) -> Result<Self, WithBoundsError> {
        let dim_in: usize = shapes.iter().map(|simplex| simplex.dim()).sum();
        if dim_in < map.dim_in() {
            Err(WithBoundsError::DimensionTooSmall)
        } else if len_in != 0 && len_in.checked_rem(map.mod_in()) != Some(0) {
            Err(WithBoundsError::LengthNotAMultipleOfRepetition)
        } else {
            Ok(Self::new_unchecked(map, shapes, len_in))
        }
    }
    pub(crate) fn new_unchecked(map: M, shapes: Vec<Simplex>, len_in: usize) -> Self {
        let dim_in: usize = shapes.iter().map(|simplex| simplex.dim()).sum();
        let delta_dim = map.delta_dim();
        let len_out = if len_in == 0 {
            0
        } else {
            len_in / map.mod_in() * map.mod_out()
        };
        Self {
            map,
            shapes,
            dim_in,
            delta_dim,
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
    pub fn children(&self) -> Self {
        let mut map = self.map.clone();
        let mut offset = 0;
        let mut len_in = self.len_in;
        for simplex in &self.shapes {
            let mut children = Elementary::new_children(*simplex);
            children.add_offset(offset);
            map.push_elementary(&children);
            len_in *= simplex.nchildren();
        }
        Self::new_unchecked(map, self.shapes.clone(), len_in)
    }
    pub fn edges(&self) -> Self {
        let mut map = self.map.clone();
        let mut offset = 0;
        let mut len_in = self.len_in;
        let mut shapes = Vec::new();
        for simplex in &self.shapes {
            let mut edges = Elementary::new_edges(*simplex);
            edges.add_offset(offset);
            map.push_elementary(&edges);
            len_in *= simplex.nedges();
            if let Some(edge_simplex) = simplex.edge_simplex() {
                shapes.push(edge_simplex);
            }
        }
        Self::new_unchecked(map, shapes, len_in)
    }
}

impl<M: UnboundedMap> BoundedMap for WithShape<M> {
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

#[derive(Debug, Clone, PartialEq)]
pub struct Tesselation(Concatenation<WithShape<Vec<Elementary>>>);

impl Tesselation {
    pub fn new_identity(shapes: Vec<Simplex>, len: usize) -> Self {
        Self(Concatenation::new(vec![WithShape::new_unchecked(vec![], shapes, len)]))
    }
    pub fn iter(&self) -> impl Iterator<Item = &WithShape<Vec<Elementary>>> {
        self.0.iter()
    }
    pub fn into_vec(self) -> Vec<WithShape<Vec<Elementary>>> {
        self.0.into_vec()
    }
    pub fn take(&self, indices: &[usize]) -> Self {
        unimplemented!{}
    }
    pub fn children(&self) -> Result<Self, String> {
        unimplemented!{}
    }
    pub fn edges(&self) -> Result<Self, String> {
        unimplemented!{}
    }
    pub fn internal_edges_of_children(&self) -> Result<Self, String> {
        unimplemented!{}
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
            $self.0.$fn($($arg),*)
        }
    };
    ($vis:vis fn $fn:ident(&mut $self:ident $(, $arg:ident: $ty:ty)*) $(-> $ret:ty)?) => {
        #[inline]
        $vis fn $fn(&mut $self $(, $arg: $ty)*) $(-> $ret)? {
            $self.0.$fn($($arg),*)
        }
    };
}

impl BoundedMap for Tesselation {
    dispatch! {fn len_out(&self) -> usize}
    dispatch! {fn len_in(&self) -> usize}
    dispatch! {fn dim_out(&self) -> usize}
    dispatch! {fn dim_in(&self) -> usize}
    dispatch! {fn delta_dim(&self) -> usize}
    dispatch! {fn apply_inplace_unchecked(&self, index: usize, coordinates: &mut [f64], stride: usize) -> usize}
    dispatch! {fn apply_inplace(&self, index: usize, coordinates: &mut [f64], stride: usize) -> Option<usize>}
    dispatch! {fn apply_index_unchecked(&self, index: usize) -> usize}
    dispatch! {fn apply_index(&self, index: usize) -> Option<usize>}
    dispatch! {fn apply_indices_inplace_unchecked(&self, indices: &mut [usize])}
    dispatch! {fn apply_indices(&self, indices: &[usize]) -> Option<Vec<usize>>}
    dispatch! {fn unapply_indices_unchecked<T: UnapplyIndicesData>(&self, indices: &[T]) -> Vec<T>}
    dispatch! {fn unapply_indices<T: UnapplyIndicesData>(&self, indices: &[T]) -> Option<Vec<T>>}
    dispatch! {fn is_identity(&self) -> bool}
}

