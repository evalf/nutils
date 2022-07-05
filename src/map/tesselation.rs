use super::ops::UniformConcat;
use super::primitive::{
    AllPrimitiveDecompositions, Primitive, PrimitiveDecompositionIter, WithBounds,
};
use super::relative::RelativeTo;
use super::{AddOffset, Error, Map, UnapplyIndicesData};
use crate::simplex::Simplex;
use crate::util::{ReplaceNthIter, SkipNthIter};
use std::iter;
use std::ops::Mul;

#[derive(Debug, Clone, PartialEq)]
pub struct UniformTesselation {
    shapes: Vec<Simplex>,
    map: WithBounds<Vec<Primitive>>,
}

impl UniformTesselation {
    pub fn identity(shapes: Vec<Simplex>, len: usize) -> Self {
        let dim = shapes.iter().map(|simplex| simplex.dim()).sum();
        let map = WithBounds::new_unchecked(vec![], dim, len);
        Self { shapes, map }
    }
    fn extend<Primitives>(&self, primitives: Primitives, shapes: Vec<Simplex>, len: usize) -> Self
    where
        Primitives: IntoIterator<Item = Primitive>,
    {
        let map = self
            .map
            .unbounded()
            .iter()
            .cloned()
            .chain(primitives)
            .collect();
        let dim = shapes.iter().map(|shape| shape.dim()).sum();
        let map = WithBounds::new_unchecked(map, dim, len);
        Self { shapes, map }
    }
    fn offsets(&self) -> impl Iterator<Item = usize> + '_ {
        self.shapes.iter().scan(0, |offset, shape| {
            let item = *offset;
            *offset += shape.dim();
            Some(item)
        })
    }
    pub fn take(&self, indices: &[usize]) -> Self {
        assert!(indices.windows(2).all(|pair| pair[0] < pair[1]));
        if let Some(last) = indices.last() {
            assert!(*last < self.len_in());
        }
        if indices.is_empty() {
            // TODO: Disallow! Skip this map in the concatenation instead.
            Self {
                shapes: self.shapes.clone(),
                map: WithBounds::new_unchecked(Vec::new(), self.dim_in(), 0),
            }
        } else {
            let primitive = Primitive::new_take(indices, self.len_in());
            let result = self.extend([primitive], self.shapes.clone(), indices.len());
            result
        }
    }
    pub fn children(&self) -> Self {
        let primitives = self
            .shapes
            .iter()
            .zip(self.offsets())
            .map(|(shape, offset)| Primitive::new_children(*shape).with_offset(offset));
        let nchildren: usize = self.shapes.iter().map(|shape| shape.nchildren()).product();
        self.extend(primitives, self.shapes.clone(), self.len_in() * nchildren)
    }
    pub fn edges(&self, ishape: usize) -> Option<Self> {
        self.shapes.get(ishape).map(|shape| {
            let offset = self
                .shapes
                .iter()
                .take(ishape)
                .map(|shape| shape.dim())
                .sum();
            let primitive = Primitive::new_edges(*shape).with_offset(offset);
            let shapes: Vec<_> = if let Some(edge_shape) = shape.edge_simplex() {
                self.shapes
                    .iter()
                    .cloned()
                    .replace_nth(ishape, edge_shape)
                    .collect()
            } else {
                self.shapes.iter().cloned().skip_nth(ishape).collect()
            };
            self.extend([primitive], shapes, self.len_in() * shape.nedges())
        })
    }
    pub fn edges_iter(&self) -> Option<impl Iterator<Item = Self> + '_> {
        if self.shapes.is_empty() {
            None
        } else {
            Some((0..self.shapes.len()).map(|ishape| self.edges(ishape).unwrap()))
        }
    }
    pub fn centroids(&self) -> Self {
        let primitives = self
            .shapes
            .iter()
            .map(|shape| Primitive::new_uniform_points(shape.centroid(), shape.dim()));
        self.extend(primitives, Vec::new(), self.len_in())
    }
    pub fn vertices(&self) -> Self {
        let primitives = self
            .shapes
            .iter()
            .map(|shape| Primitive::new_uniform_points(shape.vertices(), shape.dim()));
        self.extend(primitives, Vec::new(), self.len_in())
    }
}

//impl Deref for UniformTesselation {
//    type Target = WithBounds<Vec<Elementary>>;
//
//    fn deref(&self) -> Self::Target {
//        self.map
//    }
//}

macro_rules! dispatch {
    (
        $vis:vis fn $fn:ident$(<$genarg:ident: $genpath:path>)?(
            &$self:ident $(, $arg:ident: $ty:ty)*
        ) $(-> $ret:ty)?
    ) => {
        #[inline]
        $vis fn $fn$(<$genarg: $genpath>)?(&$self $(, $arg: $ty)*) $(-> $ret)? {
            $self.map.$fn($($arg),*)
        }
    };
    ($vis:vis fn $fn:ident(&mut $self:ident $(, $arg:ident: $ty:ty)*) $(-> $ret:ty)?) => {
        #[inline]
        $vis fn $fn(&mut $self $(, $arg: $ty)*) $(-> $ret)? {
            $self.map.$fn($($arg),*)
        }
    };
}

impl Map for UniformTesselation {
    dispatch! {fn len_out(&self) -> usize}
    dispatch! {fn len_in(&self) -> usize}
    dispatch! {fn dim_out(&self) -> usize}
    dispatch! {fn dim_in(&self) -> usize}
    dispatch! {fn delta_dim(&self) -> usize}
    dispatch! {fn apply_inplace_unchecked(&self, index: usize, coordinates: &mut [f64], stride: usize, offset: usize) -> usize}
    dispatch! {fn apply_inplace(&self, index: usize, coordinates: &mut [f64], stride: usize, offset: usize) -> Result<usize, Error>}
    dispatch! {fn apply_index_unchecked(&self, index: usize) -> usize}
    dispatch! {fn apply_index(&self, index: usize) -> Option<usize>}
    dispatch! {fn apply_indices_inplace_unchecked(&self, indices: &mut [usize])}
    dispatch! {fn apply_indices(&self, indices: &[usize]) -> Option<Vec<usize>>}
    dispatch! {fn unapply_indices_unchecked<T: UnapplyIndicesData>(&self, indices: &[T]) -> Vec<T>}
    dispatch! {fn unapply_indices<T: UnapplyIndicesData>(&self, indices: &[T]) -> Option<Vec<T>>}
    dispatch! {fn is_identity(&self) -> bool}
    dispatch! {fn is_index_map(&self) -> bool}
}

impl AllPrimitiveDecompositions for UniformTesselation {
    fn all_primitive_decompositions<'a>(&'a self) -> PrimitiveDecompositionIter<'a, Self> {
        Box::new(self.map.all_primitive_decompositions().map(|(prim, map)| {
            (
                prim,
                Self {
                    shapes: self.shapes.clone(),
                    map: map,
                },
            )
        }))
    }
}

impl RelativeTo<Self> for UniformTesselation {
    type Output = WithBounds<Vec<Primitive>>;

    fn relative_to(&self, target: &Self) -> Option<Self::Output> {
        self.map.relative_to(&target.map)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Tesselation(UniformConcat<UniformTesselation>);

impl Tesselation {
    pub fn new(items: Vec<UniformTesselation>) -> Result<Self, Error> {
        let items = items.into_iter().filter(|map| map.len_in() > 0).collect();
        Ok(Self(UniformConcat::new(items)?))
    }
    pub fn concat_iter(items: impl Iterator<Item = Tesselation>) -> Result<Self, Error> {
        let mut maps = Vec::new();
        for item in items {
            if item.len() > 0 {
                maps.extend(item.0.iter().cloned());
            }
        }
        Self::new(maps)
    }
    pub fn identity(shapes: Vec<Simplex>, len: usize) -> Self {
        Self(UniformConcat::new_unchecked(vec![
            UniformTesselation::identity(shapes, len),
        ]))
    }
    pub fn len(&self) -> usize {
        self.0.len_in()
    }
    pub fn dim(&self) -> usize {
        self.0.dim_in()
    }
    pub fn product(&self, other: &Self) -> Self {
        self * other
    }
    pub fn concat(&self, other: &Self) -> Result<Self, Error> {
        let maps = self.0.iter().chain(other.0.iter()).cloned().collect();
        Ok(Self(UniformConcat::new(maps)?))
    }
    pub fn take(&self, indices: &[usize]) -> Self {
        assert!(indices.windows(2).all(|pair| pair[0] < pair[1]));
        if let Some(last) = indices.last() {
            assert!(*last < self.len());
        }
        // TODO: Make sure that `UniformConcat` accepts an empty vector.
        let maps = self.0.iter().scan((0, 0), |(start, offset), map| {
            let stop = *start + indices[*start..].partition_point(|i| *i < *offset + map.len_in());
            let map_indices: Vec<_> = indices[*start..stop].iter().map(|i| *i - *offset).collect();
            *start = stop;
            *offset += map.len_in();
            Some(map.take(&map_indices))
        });
        let result = Self(UniformConcat::new_unchecked(maps.collect()));
        assert_eq!(result.len(), indices.len());
        result
    }
    pub fn children(&self) -> Self {
        let maps = self.0.iter().map(|item| item.children()).collect();
        Self(UniformConcat::new_unchecked(maps))
    }
    pub fn edges(&self) -> Result<Self, Error> {
        if self.dim_in() == 0 {
            return Err(Error::DimensionZeroHasNoEdges);
        }
        let items: Vec<_> = self
            .0
            .iter()
            .flat_map(|item| item.edges_iter().unwrap())
            .collect();
        Ok(Self(UniformConcat::new(items)?))
    }
    pub fn centroids(&self) -> Self {
        Self(UniformConcat::new_unchecked(
            self.0.iter().map(|item| item.centroids()).collect(),
        ))
    }
    pub fn vertices(&self) -> Self {
        Self(UniformConcat::new_unchecked(
            self.0.iter().map(|item| item.vertices()).collect(),
        ))
    }
    pub fn apply_inplace(
        &self,
        index: usize,
        coords: &mut [f64],
        stride: usize,
    ) -> Result<usize, Error> {
        self.0.apply_inplace(index, coords, stride, 0)
    }
    pub fn apply_index(&self, index: usize) -> Option<usize> {
        self.0.apply_index(index)
    }
    pub fn unapply_indices<T: UnapplyIndicesData>(&self, indices: &[T]) -> Option<Vec<T>> {
        self.0.unapply_indices(indices)
    }
}

//impl Deref for Tesselation {
//    type Target = OptionReorder<UniformConcat<UniformTesselation>>, Vec<usize>>;
//
//    fn deref(&self) -> Self::Target {
//        self.0
//    }
//}

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

impl Map for Tesselation {
    dispatch! {fn len_out(&self) -> usize}
    dispatch! {fn len_in(&self) -> usize}
    dispatch! {fn dim_out(&self) -> usize}
    dispatch! {fn dim_in(&self) -> usize}
    dispatch! {fn delta_dim(&self) -> usize}
    dispatch! {fn apply_inplace_unchecked(&self, index: usize, coordinates: &mut [f64], stride: usize, offset: usize) -> usize}
    dispatch! {fn apply_inplace(&self, index: usize, coordinates: &mut [f64], stride: usize, offset: usize) -> Result<usize, Error>}
    dispatch! {fn apply_index_unchecked(&self, index: usize) -> usize}
    dispatch! {fn apply_index(&self, index: usize) -> Option<usize>}
    dispatch! {fn apply_indices_inplace_unchecked(&self, indices: &mut [usize])}
    dispatch! {fn apply_indices(&self, indices: &[usize]) -> Option<Vec<usize>>}
    dispatch! {fn unapply_indices_unchecked<T: UnapplyIndicesData>(&self, indices: &[T]) -> Vec<T>}
    dispatch! {fn unapply_indices<T: UnapplyIndicesData>(&self, indices: &[T]) -> Option<Vec<T>>}
    dispatch! {fn is_identity(&self) -> bool}
    dispatch! {fn is_index_map(&self) -> bool}
}

impl RelativeTo<Self> for Tesselation {
    type Output = <UniformConcat<UniformTesselation> as RelativeTo<
        UniformConcat<UniformTesselation>,
    >>::Output;

    fn relative_to(&self, target: &Self) -> Option<Self::Output> {
        self.0.relative_to(&target.0)
    }
}

impl Mul for &UniformTesselation {
    type Output = UniformTesselation;

    fn mul(self, rhs: Self) -> UniformTesselation {
        let offset = self.map.dim_in();
        let map = iter::once(Primitive::new_transpose(
            rhs.map.len_out(),
            self.map.len_out(),
        ))
        .chain(self.map.unbounded().iter().cloned())
        .chain(iter::once(Primitive::new_transpose(
            self.map.len_in(),
            rhs.map.len_out(),
        )))
        .chain(rhs.map.unbounded().iter().map(|item| {
            let mut item = item.clone();
            item.add_offset(offset);
            item
        }))
        .collect();
        let map = WithBounds::new_unchecked(
            map,
            self.map.dim_in() + rhs.map.dim_in(),
            self.map.len_in() * rhs.map.len_in(),
        );
        let shapes = self
            .shapes
            .iter()
            .chain(rhs.shapes.iter())
            .cloned()
            .collect();
        UniformTesselation { shapes, map }
    }
}

impl Mul for &Tesselation {
    type Output = Tesselation;

    fn mul(self, rhs: Self) -> Tesselation {
        let products = self
            .0
            .iter()
            .flat_map(|lhs| rhs.0.iter().map(move |rhs| lhs * rhs))
            .collect();
        Tesselation(UniformConcat::new_unchecked(products))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simplex::Simplex::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn children() {
        let tess = Tesselation::identity(vec![Line], 1).children();
        assert_eq!(tess.len(), 2);
        let tess = tess.children();
        assert_eq!(tess.len(), 4);
    }

    #[test]
    fn product() {
        let lhs = Tesselation::identity(vec![Line], 1).children();
        let rhs = Tesselation::identity(vec![Line], 1).edges().unwrap();
        let tess = &lhs * &rhs;
        let centroids = tess.centroids();
        let stride = 2;
        let mut work: Vec<_> = iter::repeat(-1.0).take(stride).collect();
        println!("tess: {tess:?}");
        assert_eq!(centroids.apply_inplace(0, &mut work, stride), Some(0));
        assert_abs_diff_eq!(work[..], [0.25, 1.0]);
        assert_eq!(centroids.apply_inplace(1, &mut work, stride), Some(0));
        assert_abs_diff_eq!(work[..], [0.25, 0.0]);
        assert_eq!(centroids.apply_inplace(2, &mut work, stride), Some(0));
        assert_abs_diff_eq!(work[..], [0.75, 1.0]);
        assert_eq!(centroids.apply_inplace(3, &mut work, stride), Some(0));
        assert_abs_diff_eq!(work[..], [0.75, 0.0]);
    }

    #[test]
    fn take() {
        let lhs = Tesselation::identity(vec![Line], 1).children();
        let rhs = Tesselation::identity(vec![Line], 1).edges().unwrap();
        let levels: Vec<_> = iter::successors(Some(&lhs * &rhs), |level| Some(level.children()))
            .take(3)
            .collect();
        let hierarch = levels[1].take(&[0, 1, 2]);
    }
}
