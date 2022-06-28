use crate::{Error, Map, UnapplyIndicesData};
use std::ops::Deref;

#[derive(Debug, Clone, PartialEq)]
pub struct BinaryComposition<Inner: Map, Outer: Map>(Inner, Outer);

impl<Inner: Map, Outer: Map> BinaryComposition<Inner, Outer> {
    pub fn new(inner: Inner, outer: Outer) -> Result<Self, Error> {
        if inner.dim_out() != outer.dim_in() {
            Err(Error::DimensionMismatch)
        } else if inner.len_out() != outer.len_in() {
            Err(Error::LengthMismatch)
        } else {
            Ok(Self(inner, outer))
        }
    }
}

impl<Inner: Map, Outer: Map> Map for BinaryComposition<Inner, Outer> {
    #[inline]
    fn dim_in(&self) -> usize {
        self.0.dim_in()
    }
    #[inline]
    fn delta_dim(&self) -> usize {
        self.0.delta_dim() + self.1.delta_dim()
    }
    #[inline]
    fn len_in(&self) -> usize {
        self.0.len_in()
    }
    #[inline]
    fn len_out(&self) -> usize {
        self.1.len_out()
    }
    #[inline]
    fn apply_inplace_unchecked(
        &self,
        index: usize,
        coords: &mut [f64],
        stride: usize,
        offset: usize,
    ) -> usize {
        let index = self.0.apply_inplace_unchecked(index, coords, stride, offset);
        self.1.apply_inplace_unchecked(index, coords, stride, offset)
    }
    #[inline]
    fn apply_index_unchecked(&self, index: usize) -> usize {
        self.1.apply_index_unchecked(self.0.apply_index_unchecked(index))
    }
    #[inline]
    fn apply_indices_inplace_unchecked(&self, indices: &mut [usize]) {
        self.0.apply_indices_inplace_unchecked(indices);
        self.1.apply_indices_inplace_unchecked(indices);
    }
    #[inline]
    fn unapply_indices_unchecked<T: UnapplyIndicesData>(&self, indices: &[T]) -> Vec<T> {
        self.0.unapply_indices_unchecked(&self.1.unapply_indices_unchecked(indices))
    }
    #[inline]
    fn is_identity(&self) -> bool {
        self.0.is_identity() && self.1.is_identity()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct UniformComposition<M, Array>(Array)
where
    M: Map,
    Array: Deref<Target = [M]>;

impl<M, Array> UniformComposition<M, Array>
where
    M: Map,
    Array: Deref<Target = [M]>,
{
    pub fn new(array: Array) -> Result<Self, Error> {
        let mut iter = array.iter();
        if let Some(map) = iter.next() {
            let mut dim_out = map.dim_out();
            let mut len_out = map.len_out();
            for map in iter {
                if map.dim_in() != dim_out {
                    return Err(Error::DimensionMismatch);
                } else if map.len_in() != len_out {
                    return Err(Error::LengthMismatch);
                }
                dim_out = map.dim_out();
                len_out = map.len_out();
            }
            Ok(Self(array))
        } else {
            Err(Error::Empty)
        }
    }
    pub fn new_unchecked(array: Array) -> Self {
        Self(array)
    }
    pub fn iter<'a>(&'a self) -> std::slice::Iter<'a, M> {
        self.0.iter()
    }
}

impl<M, Array> Map for UniformComposition<M, Array>
where
    M: Map,
    Array: Deref<Target = [M]>,
{
    #[inline]
    fn dim_in(&self) -> usize {
        self.0.first().unwrap().dim_in()
    }
    #[inline]
    fn dim_out(&self) -> usize {
        self.0.last().unwrap().dim_out()
    }
    #[inline]
    fn delta_dim(&self) -> usize {
        self.dim_out() - self.dim_in()
    }
    #[inline]
    fn len_in(&self) -> usize {
        self.0.first().unwrap().len_in()
    }
    #[inline]
    fn len_out(&self) -> usize {
        self.0.last().unwrap().len_out()
    }
    #[inline]
    fn apply_inplace_unchecked(
        &self,
        index: usize,
        coords: &mut [f64],
        stride: usize,
        offset: usize,
    ) -> usize {
        self.iter().fold(index, |index, map| {
            map.apply_inplace_unchecked(index, coords, stride, offset)
        })
    }
    #[inline]
    fn apply_index_unchecked(&self, index: usize) -> usize {
        self.iter().fold(index, |index, map| map.apply_index_unchecked(index))
    }
    #[inline]
    fn apply_indices_inplace_unchecked(&self, indices: &mut [usize]) {
        self.iter()
            .for_each(|map| map.apply_indices_inplace_unchecked(indices));
    }
    #[inline]
    fn unapply_indices_unchecked<T: UnapplyIndicesData>(&self, indices: &[T]) -> Vec<T> {
        let mut iter = self.iter().rev();
        let indices = iter.next().unwrap().unapply_indices_unchecked(indices);
        iter.fold(indices, |indices, map| map.unapply_indices_unchecked(&indices))
    }
    #[inline]
    fn is_identity(&self) -> bool {
        self.iter().all(|map| map.is_identity())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct BinaryConcat<M0: Map, M1: Map>(M0, M1);

impl<M0: Map, M1: Map> BinaryConcat<M0, M1> {
    pub fn new(map0: M0, map1: M1) -> Result<Self, Error> {
        if map0.dim_in() != map1.dim_in() || map0.dim_out() != map1.dim_out() {
            Err(Error::DimensionMismatch)
        } else if map0.len_out() != map1.len_out() {
            Err(Error::LengthMismatch)
        } else {
            Ok(Self(map0, map1))
        }
    }
    pub fn new_unchecked(map0: M0, map1: M1) -> Self {
        Self(map0, map1)
    }
}

impl<M0: Map, M1: Map> Map for BinaryConcat<M0, M1> {
    #[inline]
    fn dim_in(&self) -> usize {
        self.0.dim_in()
    }
    #[inline]
    fn dim_out(&self) -> usize {
        self.0.dim_out()
    }
    #[inline]
    fn delta_dim(&self) -> usize {
        self.0.delta_dim()
    }
    #[inline]
    fn len_in(&self) -> usize {
        self.0.len_in() + self.1.len_in()
    }
    #[inline]
    fn len_out(&self) -> usize {
        self.0.len_out()
    }
    #[inline]
    fn apply_inplace_unchecked(
        &self,
        index: usize,
        coords: &mut [f64],
        stride: usize,
        offset: usize,
    ) -> usize {
        if index < self.0.len_in() {
            self.0.apply_inplace_unchecked(index, coords, stride, offset)
        } else {
            self.1.apply_inplace_unchecked(index - self.0.len_in(), coords, stride, offset)
        }
    }
    #[inline]
    fn apply_index_unchecked(&self, index: usize) -> usize {
        if index < self.0.len_in() {
            self.0.apply_index_unchecked(index)
        } else {
            self.1.apply_index_unchecked(index - self.0.len_in())
        }
    }
    #[inline]
    fn unapply_indices_unchecked<T: UnapplyIndicesData>(&self, indices: &[T]) -> Vec<T> {
        let mut result = self.0.unapply_indices_unchecked(indices);
        result.extend(self.1.unapply_indices_unchecked(indices).into_iter().map(|i| i.set(i.get() + self.0.len_in())));
        result
    }
    #[inline]
    fn is_identity(&self) -> bool {
        false
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct UniformConcat<M, Array>(Array)
where
    M: Map,
    Array: Deref<Target = [M]>;

impl<M, Array> UniformConcat<M, Array>
where
    M: Map,
    Array: Deref<Target = [M]>,
{
    pub fn new(array: Array) -> Result<Self, Error> {
        let mut iter = array.iter();
        if let Some(map) = iter.next() {
            let dim_out = map.dim_out();
            let dim_in = map.dim_in();
            let len_out = map.len_out();
            for map in iter {
                if map.dim_in() != dim_in || map.dim_out() != dim_out {
                    return Err(Error::DimensionMismatch);
                } else if map.len_out() != len_out {
                    return Err(Error::LengthMismatch);
                }
            }
            Ok(Self(array))
        } else {
            Err(Error::Empty)
        }
    }
    pub fn new_unchecked(array: Array) -> Self {
        Self(array)
    }
    pub fn iter<'a>(&'a self) -> std::slice::Iter<'a, M> {
        self.0.iter()
    }
}

impl<M, Array> Map for UniformConcat<M, Array>
where
    M: Map,
    Array: Deref<Target = [M]>,
{
    #[inline]
    fn dim_in(&self) -> usize {
        self.0.first().unwrap().dim_in()
    }
    #[inline]
    fn dim_out(&self) -> usize {
        self.0.first().unwrap().dim_out()
    }
    #[inline]
    fn delta_dim(&self) -> usize {
        self.0.first().unwrap().delta_dim()
    }
    #[inline]
    fn len_in(&self) -> usize {
        self.iter().map(|map| map.len_in()).sum()
    }
    #[inline]
    fn len_out(&self) -> usize {
        self.0.first().unwrap().len_out()
    }
    #[inline]
    fn apply_inplace_unchecked(
        &self,
        mut index: usize,
        coords: &mut [f64],
        stride: usize,
        offset: usize,
    ) -> usize {
        for map in self.iter() {
            if index < map.len_in() {
                return map.apply_inplace_unchecked(index, coords, stride, offset);
            }
            index -= map.len_in();
        }
        unreachable!{}
    }
    #[inline]
    fn apply_index_unchecked(&self, mut index: usize) -> usize {
        for map in self.iter() {
            if index < map.len_in() {
                return map.apply_index_unchecked(index);
            }
            index -= map.len_in();
        }
        unreachable!{}
    }
    #[inline]
    fn unapply_indices_unchecked<T: UnapplyIndicesData>(&self, indices: &[T]) -> Vec<T> {
        let mut iter = self.iter();
        let map = iter.next().unwrap();
        let mut result = map.unapply_indices_unchecked(indices);
        let mut offset = map.len_in();
        for map in iter {
            result.extend(map.unapply_indices_unchecked(indices).into_iter().map(|i| i.set(i.get() + offset)));
            offset += map.len_in();
        }
        result
    }
    #[inline]
    fn is_identity(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use crate::elementary::{Elementary, Identity, WithBounds};
    use crate::simplex::Simplex::*;
    use std::iter;
    use crate::assert_map_apply;

    type ElemsMap = WithBounds<Vec<Elementary>>;

    #[test]
    fn uniform_composition1() {
        let map = UniformComposition(vec![WithBounds::new(Identity, 0, 1).unwrap()]);
        assert_eq!(map.len_in(), 1);
        assert_eq!(map.len_out(), 1);
        assert_eq!(map.dim_in(), 0);
        assert_eq!(map.delta_dim(), 0);
        assert_map_apply!(map, 0, 0);
    }

//    #[test]
//    fn uniform_composition2() {
//        let map = UniformComposition(vec![
//            Elementary::new_take(vec![1, 0], 3),
//            Elementary::new_transpose(4, 3),
//        ]);
//        assert_eq!(map.mod_in(), 8);
//        assert_eq!(map.mod_out(), 12);
//        assert_eq!(map.dim_in(), 0);
//        assert_eq!(map.delta_dim(), 0);
//        assert_map_apply!(map, 0, 4);
//        assert_map_apply!(map, 1, 0);
//        assert_map_apply!(map, 2, 5);
//        assert_map_apply!(map, 3, 1);
//        assert_map_apply!(map, 4, 6);
//        assert_map_apply!(map, 5, 2);
//        assert_map_apply!(map, 6, 7);
//        assert_map_apply!(map, 7, 3);
//        assert_map_apply!(map, 8, 16);
//    }
//
//    #[test]
//    fn uniform_composition3() {
//        let map = UniformComposition(vec![
//            Elementary::new_edges(Line),
//            Elementary::new_children(Line),
//            Elementary::new_children(Line),
//        ]);
//        assert_eq!(map.mod_in(), 8);
//        assert_eq!(map.mod_out(), 1);
//        assert_eq!(map.dim_in(), 0);
//        assert_eq!(map.delta_dim(), 1);
//        assert_map_apply!(map, 0, [[]], 0, [[0.25]]);
//        assert_map_apply!(map, 1, [[]], 0, [[0.00]]);
//        assert_map_apply!(map, 2, [[]], 0, [[0.50]]);
//        assert_map_apply!(map, 3, [[]], 0, [[0.25]]);
//        assert_map_apply!(map, 4, [[]], 0, [[0.75]]);
//        assert_map_apply!(map, 5, [[]], 0, [[0.50]]);
//        assert_map_apply!(map, 6, [[]], 0, [[1.00]]);
//        assert_map_apply!(map, 7, [[]], 0, [[0.75]]);
//        assert_map_apply!(map, 8, [[]], 1, [[0.25]]);
//        assert_eq!(map.unapply_indices(&[0]), vec![0, 1, 2, 3, 4, 5, 6, 7]);
//    }
}
