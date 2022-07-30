use super::ops::UniformConcat;
use super::primitive::{
    AllPrimitiveDecompositions, Primitive, PrimitiveDecompositionIter, UnboundedMap, WithBounds,
};
use super::relative::RelativeTo;
use super::{AddOffset, Error, Map, UnapplyIndicesData};
use crate::simplex::Simplex;
use crate::util::{ReplaceNthIter, SkipNthIter};
use std::iter;
use std::ops::Mul;
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq, Hash)]
pub struct UniformCoordSystem(WithBounds<Vec<Primitive>>);

impl UniformCoordSystem {
    pub fn new(dim: usize, len: usize) -> Self {
        Self(WithBounds::new_unchecked(Vec::new(), dim, len))
    }
    pub fn clone_and_push(&self, primitive: Primitive) -> Result<Self, Error> {
        if self.0.dim_in() < primitive.dim_out() {
            return Err(Error::DimensionMismatch);
        }
        if self.0.len_in() % primitive.mod_out() != 0 {
            return Err(Error::LengthMismatch);
        }
        let dim_in = self.0.dim_in() - primitive.delta_dim();
        let len_in = self.0.len_in() / primitive.mod_out() * primitive.mod_in();
        let map = self
            .0
            .unbounded()
            .iter()
            .cloned()
            .chain(iter::once(primitive))
            .collect();
        Ok(Self(WithBounds::new_unchecked(map, dim_in, len_in)))
    }
    fn mul(&self, rhs: &Self) -> Self {
        let offset = self.0.dim_in();
        let map = iter::once(Primitive::new_transpose(rhs.0.len_out(), self.0.len_out()))
            .chain(self.0.unbounded().iter().cloned())
            .chain(iter::once(Primitive::new_transpose(
                self.0.len_in(),
                rhs.0.len_out(),
            )))
            .chain(rhs.0.unbounded().iter().map(|item| {
                let mut item = item.clone();
                item.add_offset(offset);
                item
            }))
            .collect();
        Self(WithBounds::new_unchecked(
            map,
            self.0.dim_in() + rhs.0.dim_in(),
            self.0.len_in() * rhs.0.len_in(),
        ))
    }
}

//impl Deref for UniformCoordSystem {
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
            $self.0.$fn($($arg),*)
        }
    };
    ($vis:vis fn $fn:ident(&mut $self:ident $(, $arg:ident: $ty:ty)*) $(-> $ret:ty)?) => {
        #[inline]
        $vis fn $fn(&mut $self $(, $arg: $ty)*) $(-> $ret)? {
            $self.map.$fn($($arg),*)
        }
    };
}

impl Map for UniformCoordSystem {
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
    dispatch! {fn update_basis(&self, index: usize, basis: &mut [f64], dim_out: usize, dim_in: &mut usize, offset: usize) -> usize}
    dispatch! {fn basis_is_constant(&self) -> bool}
}

impl AllPrimitiveDecompositions for UniformCoordSystem {
    fn all_primitive_decompositions<'a>(&'a self) -> PrimitiveDecompositionIter<'a, Self> {
        Box::new(
            self.0
                .all_primitive_decompositions()
                .map(|(prim, map)| (prim, Self(map))),
        )
    }
}

impl RelativeTo<Self> for UniformCoordSystem {
    type Output = <WithBounds<Vec<Primitive>> as RelativeTo<WithBounds<Vec<Primitive>>>>::Output;

    fn relative_to(&self, target: &Self) -> Option<Self::Output> {
        self.0.relative_to(&target.0)
    }
}

#[derive(Debug, Clone, PartialEq, Hash)]
pub struct CoordSystem(UniformConcat<UniformCoordSystem>);

impl CoordSystem {
    pub fn new(dim: usize, len: usize) -> Self {
        let identity = UniformCoordSystem::new(dim, len);
        Self(UniformConcat::new_unchecked(vec![identity], dim, 0, len))
    }
    fn mul(&self, rhs: &Self) -> Self {
        let products = self
            .0
            .iter()
            .flat_map(|lhs| rhs.0.iter().map(move |rhs| lhs * rhs))
            .collect();
        CoordSystem(UniformConcat::new_unchecked(
            products,
            self.dim_in() + rhs.dim_in(),
            self.delta_dim() + rhs.delta_dim(),
            self.len_out() * rhs.len_out(),
        ))
    }
    pub fn len(&self) -> usize {
        self.len_in()
    }
    pub fn dim(&self) -> usize {
        self.dim_in()
    }
    pub fn concat(&self, other: &Self) -> Result<Self, Error> {
        let maps = self.0.iter().chain(other.0.iter()).cloned().collect();
        Ok(Self(UniformConcat::new(
            maps,
            self.dim_in(),
            self.delta_dim(),
            self.len_out(),
        )?))
    }
    pub fn slice(&self, mut start: usize, mut len: usize) -> Result<Self, Error> {
        if start + len > self.0.len_in() {
            return Err(Error::IndexOutOfRange);
        }
        let mut maps = Vec::new();
        for map in self.0.iter() {
            if len == 0 {
                break;
            } else if start >= map.len_in() {
                start -= map.len_in();
                continue;
            } else if start == 0 && len >= map.len_in() {
                maps.push(map.clone());
                len -= map.len_in();
            } else {
                let primitive = Primitive::new_slice(start, len, map.len_in());
                maps.push(map.clone_and_push(primitive).unwrap());
                if start + len <= map.len_in() {
                    break;
                }
                len -= map.len_in() - start;
                start = 0;
            }
        }
        Ok(Self(UniformConcat::new_unchecked(
            maps,
            self.dim_in(),
            self.delta_dim(),
            self.len_out(),
        )))
    }
    pub fn take(&self, indices: &[usize]) -> Result<Self, Error> {
        if !indices.windows(2).all(|pair| pair[0] < pair[1]) {
            return Err(Error::IndicesNotStrictIncreasing);
        }
        if let Some(last) = indices.last() {
            if *last >= self.0.len_in() {
                return Err(Error::IndexOutOfRange);
            }
        }
        let mut maps = Vec::new();
        let mut offset = 0;
        let mut start = 0;
        for map in self.0.iter() {
            let stop = start + indices[start..].partition_point(|i| *i < offset + map.len_in());
            if stop > start {
                let map_indices: Vec<_> =
                    indices[start..stop].iter().map(|i| *i - offset).collect();
                start = stop;
                let primitive = Primitive::new_take(map_indices, map.len_in());
                maps.push(map.clone_and_push(primitive).unwrap());
            }
            offset += map.len_in();
        }
        assert_eq!(start, indices.len());
        Ok(Self(UniformConcat::new_unchecked(
            maps,
            self.dim_in(),
            self.delta_dim(),
            self.len_out(),
        )))
    }
    fn clone_and_push(&self, primitive: Primitive) -> Result<Self, Error> {
        if self.0.dim_in() < primitive.dim_out() {
            return Err(Error::DimensionMismatch);
        }
        if self.0.len_in() % primitive.mod_out() != 0 {
            return Err(Error::LengthMismatch);
        }
        let maps = self
            .0
            .iter()
            .map(|map| map.clone_and_push(primitive.clone()).unwrap());
        Ok(Self(UniformConcat::new_unchecked(
            maps.collect(),
            self.dim_in() - primitive.delta_dim(),
            self.delta_dim() + primitive.delta_dim(),
            self.len_out(),
        )))
    }
    pub fn children(&self, simplex: Simplex, offset: usize) -> Result<Self, Error> {
        self.clone_and_push(Primitive::new_children(simplex).with_offset(offset))
    }
    pub fn edges(&self, simplex: Simplex, offset: usize) -> Result<Self, Error> {
        self.clone_and_push(Primitive::new_edges(simplex).with_offset(offset))
    }
    pub fn uniform_points(
        &self,
        points: impl Into<Arc<[f64]>>,
        point_dim: usize,
        offset: usize,
    ) -> Result<Self, Error> {
        self.clone_and_push(Primitive::new_uniform_points(points, point_dim).with_offset(offset))
    }
}

//impl Deref for CoordSystem {
//    type Target = OptionReorder<UniformConcat<UniformCoordSystem>>, Vec<usize>>;
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

impl Map for CoordSystem {
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
    dispatch! {fn update_basis(&self, index: usize, basis: &mut [f64], dim_out: usize, dim_in: &mut usize, offset: usize) -> usize}
    dispatch! {fn basis_is_constant(&self) -> bool}
}

impl RelativeTo<Self> for CoordSystem {
    type Output =
        <UniformConcat<UniformCoordSystem> as RelativeTo<UniformConcat<UniformCoordSystem>>>::Output;

    fn relative_to(&self, target: &Self) -> Option<Self::Output> {
        self.0.relative_to(&target.0)
    }
}

impl Mul for &UniformCoordSystem {
    type Output = UniformCoordSystem;

    fn mul(self, rhs: Self) -> UniformCoordSystem {
        UniformCoordSystem::mul(self, rhs)
    }
}

impl Mul for UniformCoordSystem {
    type Output = UniformCoordSystem;

    fn mul(self, rhs: Self) -> UniformCoordSystem {
        UniformCoordSystem::mul(&self, &rhs)
    }
}

impl Mul for &CoordSystem {
    type Output = CoordSystem;

    fn mul(self, rhs: Self) -> CoordSystem {
        CoordSystem::mul(self, rhs)
    }
}

impl Mul for CoordSystem {
    type Output = CoordSystem;

    fn mul(self, rhs: Self) -> CoordSystem {
        CoordSystem::mul(&self, &rhs)
    }
}
