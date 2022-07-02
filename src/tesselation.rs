use crate::elementary::{Elementary, PushElementary};
use crate::ops::{Concatenation, WithBoundsError};
use crate::simplex::Simplex;
use crate::{AddOffset, BoundedMap, UnapplyIndicesData, UnboundedMap};

struct UniformTesselation {
    shapes: Vec<Simplex>,
    map: WithBounds<Vec<Elementary>>,
}

impl UniformTesselation {
    fn identity(shapes: Vec<Simplex>, len: usize) -> Self {
        let dim = shapes.iter().map(|simplex| simplex.dim()).sum();
        let map = WithBounds::from_output(vec![], dim, len);
        Self { shapes, map }
    }
    fn children(&self) -> Self {
        let mut map = self.map.clone();
        let mut offset = 0;
        for shape in &self.shapes {
            let item = Elementary::new_children(shape);
            item.add_offset(offset);
            map.push(item);
            offset += shape.dim();
        }
        Self { shapes: self.shapes.clone(), map }
    }
    fn edges(&self) -> Result<Self, String> {
        if self.shapes().is_empty() {
            return Err("dimension zero");
        }
        let mut map = self.map.clone();
        let mut shapes = Vec::with_capacity(self.shapes.len());
        let mut offset = 0;
        for shape in &self.shapes {
            let item = Elementary::new_edges(shape);
            item.add_offset(offset);
            map.push(item);
            offset += shape.edge_dim();
            if let Some(edge_shape) = shape.edge_shape() {
                shapes.push(edge_shape);
            }
        }
        Ok(Self { shapes, map })
    }
}

impl Deref for UniformTesselation {
    type Target = WithBounds<Vec<Elementary>>;

    fn deref(&self) -> Self::Target {
        self.map
    }
}

pub struct Tesselation(UniformConcat<UniformTesselation>, Vec<usize>>);

impl Tesselation {
    pub fn identity(shapes: Vec<Simplex>, len: usize) -> Self {
        Self(UniformConcat::new_unchecked(vec![UniformTesselation::identity(shapes, len)]))
    }
    pub fn len(&self) -> Self {
        self.0.len_in()
    }
    pub fn dim(&self) -> Self {
        self.0.dim_in()
    }
    pub fn product(&self, other: &Self) -> Self {
        unimplemented!{}
    }
    pub fn concatenate(&self, other: &Self) -> Self {
        unimplemented!{}
    }
    pub fn take(&self, indices: &[usize]) -> Self {
        unimplemented!{}
    }
    pub fn children(&self) -> Self {
        Self(UniformConcat::new_unchecked(self.0.iter().map(|item| item.children()).collect()))
    }
    pub fn edges(&self) -> Result<Self, String> {
        let edges = self.0.iter().map(|item| item.edges()).collect::<Vec<Result<_, _>>()?;
        Self(UniformConcat::new_unchecked(edges))
    }
    pub fn centroids(&self) -> Self {
        unimplemented!{}
    }
    pub fn vertices(&self) -> Self {
        unimplemented!{}
    }
    pub fn apply_inplace(&self, mut index: usize, coords: &mut [f64], stride: usize) -> Option<index> {
        self.0.apply_inplace(index, coords, stride, 0)
    }
    pub fn unapply_indices<T: UnapplyIndicesData>(&self, indices: &[T]) -> Vec<T> {
        self.0.unapply_indices(indices)
    }
}

impl Deref for Tesselation {
    type Target = OptionReorder<UniformConcat<UniformTesselation>>, Vec<usize>>;

    fn deref(&self) -> Self::Target {
        self.0
    }
}






enum Tesselation {
    Uniform(UniformTesselation),
    Concatenation(Vec<Tesselation>),
    Product(Vec<Tesselation>),
    Reordered(Box<Tesselation>, Vec<usize>),
}


impl RelativeTo<Tesselation> for Tesselation {
    fn relative_to(&self, target: &Tesselation) -> ... {
        // convert product
    }
}






struct UniformTesselation {
    shapes: Vec<Simplex>
    delta_dim: usize,
    len_out: usize,
    maps: Vec<(usize, Vec<Elementary>)>,
    reorder: Option<Vec<usize>>,
}

impl Tesselation {
    pub fn new_identity(shapes: Vec<Simplex>, len: usize) -> Self {
        Self {
            shapes,
            delta_dim: 0,
            len_out: len,
            maps: vec![(len, Vec::new()],
            reorder: None,
        }
    }
    pub fn product(&self, other: &Self) -> Self {
        unimplemented!{}
    }
    pub fn concatenate(&self, other: &Self) -> Self {
        unimplemented!{}
    }
    pub fn take(&self, indices: &[usize]) -> Self {
        unimplemented!{}
    }
    pub fn children(&self) -> Self {
        unimplemented!{}
    }
    pub fn edges(&self) -> Self {
        unimplemented!{}
    }
    pub fn centroids(&self) -> Self {
        unimplemented!{}
    }
    pub fn vertices(&self) -> Self {
        unimplemented!{}
    }
    pub fn len(&self) -> usize {
        self.maps.iter().map(|(n, _)| n).sum()
    }
    pub fn dim(&self) -> usize {
        self.shapes.iter().map(|shape| shape.dim()).sum()
    }
    pub fn apply_inplace(&self, mut index: usize, coords: &mut [f64], stride: usize) -> Option<index> {
        for (len, map) in &self.maps {
            if index < len {
                return map.apply_inplace(index, coords, stride, 0);
            }
            index -= len;
        }
        None
    }
    pub fn unapply_indices<T: UnapplyIndicesData>(&self, indices: &[T]) -> Vec<T> {
        unimplemented!{}
    }
}

// struct ProductTesselation(Vec<Tesselation>);

// struct Relative




//#[derive(Debug, Clone, PartialEq)]
//pub struct WithShape<M: UnboundedMap> {
//    map: M,
//    shapes: Vec<Simplex>,
//    dim_in: usize,
//    delta_dim: usize,
//    len_out: usize,
//    len_in: usize,
//}
//
//impl<M: UnboundedMap + PushElementary + Clone> WithShape<M> {
//    pub fn new(map: M, shapes: Vec<Simplex>, len_in: usize) -> Result<Self, WithBoundsError> {
//        let dim_in: usize = shapes.iter().map(|simplex| simplex.dim()).sum();
//        if dim_in < map.dim_in() {
//            Err(WithBoundsError::DimensionTooSmall)
//        } else if len_in != 0 && len_in.checked_rem(map.mod_in()) != Some(0) {
//            Err(WithBoundsError::LengthNotAMultipleOfRepetition)
//        } else {
//            Ok(Self::new_unchecked(map, shapes, len_in))
//        }
//    }
//    pub(crate) fn new_unchecked(map: M, shapes: Vec<Simplex>, len_in: usize) -> Self {
//        let dim_in: usize = shapes.iter().map(|simplex| simplex.dim()).sum();
//        let delta_dim = map.delta_dim();
//        let len_out = if len_in == 0 {
//            0
//        } else {
//            len_in / map.mod_in() * map.mod_out()
//        };
//        Self {
//            map,
//            shapes,
//            dim_in,
//            delta_dim,
//            len_out,
//            len_in,
//        }
//    }
//    pub fn get_unbounded(&self) -> &M {
//        &self.map
//    }
//    pub fn into_unbounded(self) -> M {
//        self.map
//    }
//    pub fn children(&self) -> Self {
//        let mut map = self.map.clone();
//        let mut offset = 0;
//        let mut len_in = self.len_in;
//        for simplex in &self.shapes {
//            let mut children = Elementary::new_children(*simplex);
//            children.add_offset(offset);
//            map.push_elementary(&children);
//            len_in *= simplex.nchildren();
//        }
//        Self::new_unchecked(map, self.shapes.clone(), len_in)
//    }
//    pub fn edges(&self) -> Self {
//        let mut map = self.map.clone();
//        let mut offset = 0;
//        let mut len_in = self.len_in;
//        let mut shapes = Vec::new();
//        for simplex in &self.shapes {
//            let mut edges = Elementary::new_edges(*simplex);
//            edges.add_offset(offset);
//            map.push_elementary(&edges);
//            len_in *= simplex.nedges();
//            if let Some(edge_simplex) = simplex.edge_simplex() {
//                shapes.push(edge_simplex);
//            }
//        }
//        Self::new_unchecked(map, shapes, len_in)
//    }
//}
//
//impl<M: UnboundedMap> BoundedMap for WithShape<M> {
//    fn len_in(&self) -> usize {
//        self.len_in
//    }
//    fn len_out(&self) -> usize {
//        self.len_out
//    }
//    fn dim_in(&self) -> usize {
//        self.dim_in
//    }
//    fn delta_dim(&self) -> usize {
//        self.delta_dim
//    }
//    fn apply_inplace_unchecked(
//        &self,
//        index: usize,
//        coordinates: &mut [f64],
//        stride: usize,
//        offset: usize,
//    ) -> usize {
//        self.map.apply_inplace(index, coordinates, stride, offset)
//    }
//    fn apply_index_unchecked(&self, index: usize) -> usize {
//        self.map.apply_index(index)
//    }
//    fn apply_indices_inplace_unchecked(&self, indices: &mut [usize]) {
//        self.map.apply_indices_inplace(indices)
//    }
//    fn unapply_indices_unchecked<T: UnapplyIndicesData>(&self, indices: &[T]) -> Vec<T> {
//        self.map.unapply_indices(indices)
//    }
//    fn is_identity(&self) -> bool {
//        self.map.is_identity()
//    }
//}
//
//#[derive(Debug, Clone, PartialEq)]
//pub struct Tesselation(Concatenation<WithShape<Vec<Elementary>>>);
//
//impl Tesselation {
//    pub fn new_identity(shapes: Vec<Simplex>, len: usize) -> Self {
//        Self(Concatenation::new(vec![WithShape::new_unchecked(
//            vec![],
//            shapes,
//            len,
//        )]))
//    }
//    pub fn iter(&self) -> impl Iterator<Item = &WithShape<Vec<Elementary>>> {
//        self.0.iter()
//    }
//    pub fn into_vec(self) -> Vec<WithShape<Vec<Elementary>>> {
//        self.0.into_vec()
//    }
//    pub fn take(&self, indices: &[usize]) -> Self {
//        unimplemented! {}
//    }
//    pub fn children(&self) -> Result<Self, String> {
//        unimplemented! {}
//    }
//    pub fn edges(&self) -> Result<Self, String> {
//        unimplemented! {}
//    }
//    pub fn internal_edges_of_children(&self) -> Result<Self, String> {
//        unimplemented! {}
//    }
//}
//
//macro_rules! dispatch {
//    (
//        $vis:vis fn $fn:ident$(<$genarg:ident: $genpath:path>)?(
//            &$self:ident $(, $arg:ident: $ty:ty)*
//        ) $(-> $ret:ty)?
//    ) => {
//        #[inline]
//        $vis fn $fn$(<$genarg: $genpath>)?(&$self $(, $arg: $ty)*) $(-> $ret)? {
//            $self.0.$fn($($arg),*)
//        }
//    };
//    ($vis:vis fn $fn:ident(&mut $self:ident $(, $arg:ident: $ty:ty)*) $(-> $ret:ty)?) => {
//        #[inline]
//        $vis fn $fn(&mut $self $(, $arg: $ty)*) $(-> $ret)? {
//            $self.0.$fn($($arg),*)
//        }
//    };
//}
//
//impl BoundedMap for Tesselation {
//    dispatch! {fn len_out(&self) -> usize}
//    dispatch! {fn len_in(&self) -> usize}
//    dispatch! {fn dim_out(&self) -> usize}
//    dispatch! {fn dim_in(&self) -> usize}
//    dispatch! {fn delta_dim(&self) -> usize}
//    dispatch! {fn apply_inplace_unchecked(&self, index: usize, coordinates: &mut [f64], stride: usize, offset: usize) -> usize}
//    dispatch! {fn apply_inplace(&self, index: usize, coordinates: &mut [f64], stride: usize, offset: usize) -> Option<usize>}
//    dispatch! {fn apply_index_unchecked(&self, index: usize) -> usize}
//    dispatch! {fn apply_index(&self, index: usize) -> Option<usize>}
//    dispatch! {fn apply_indices_inplace_unchecked(&self, indices: &mut [usize])}
//    dispatch! {fn apply_indices(&self, indices: &[usize]) -> Option<Vec<usize>>}
//    dispatch! {fn unapply_indices_unchecked<T: UnapplyIndicesData>(&self, indices: &[T]) -> Vec<T>}
//    dispatch! {fn unapply_indices<T: UnapplyIndicesData>(&self, indices: &[T]) -> Option<Vec<T>>}
//    dispatch! {fn is_identity(&self) -> bool}
//}
