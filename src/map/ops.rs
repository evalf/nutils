use super::{Error, Map, UnapplyIndicesData};
use num::Integer as _;
use std::ops::Deref;

/// The composition of two maps.
#[derive(Debug, Clone, PartialEq)]
pub struct BinaryComposition<Outer: Map, Inner: Map>(Outer, Inner);

impl<Outer: Map, Inner: Map> BinaryComposition<Outer, Inner> {
    /// Returns the composition of two maps.
    ///
    /// The input dimension and length of the first map must equal the output
    /// dimension and length of the second map.
    ///
    /// Returns an [`Error`] if the dimensions and lengths don't match.
    pub fn new(outer: Outer, inner: Inner) -> Result<Self, Error> {
        if inner.dim_out() != outer.dim_in() {
            Err(Error::DimensionMismatch)
        } else if inner.len_out() != outer.len_in() {
            Err(Error::LengthMismatch)
        } else {
            Ok(Self(outer, inner))
        }
    }
    /// Returns the outer map of the composition.
    pub fn outer(&self) -> &Outer {
        &self.0
    }
    /// Returns the inner map of the composition.
    pub fn inner(&self) -> &Inner {
        &self.1
    }
}

impl<Outer: Map, Inner: Map> Map for BinaryComposition<Outer, Inner> {
    #[inline]
    fn dim_in(&self) -> usize {
        self.1.dim_in()
    }
    #[inline]
    fn delta_dim(&self) -> usize {
        self.0.delta_dim() + self.1.delta_dim()
    }
    #[inline]
    fn len_in(&self) -> usize {
        self.1.len_in()
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
        let index = self
            .1
            .apply_inplace_unchecked(index, coords, stride, offset);
        self.0
            .apply_inplace_unchecked(index, coords, stride, offset)
    }
    #[inline]
    fn apply_index_unchecked(&self, index: usize) -> usize {
        self.0
            .apply_index_unchecked(self.1.apply_index_unchecked(index))
    }
    #[inline]
    fn apply_indices_inplace_unchecked(&self, indices: &mut [usize]) {
        self.1.apply_indices_inplace_unchecked(indices);
        self.0.apply_indices_inplace_unchecked(indices);
    }
    #[inline]
    fn unapply_indices_unchecked<T: UnapplyIndicesData>(&self, indices: &[T]) -> Vec<T> {
        self.1
            .unapply_indices_unchecked(&self.0.unapply_indices_unchecked(indices))
    }
    #[inline]
    fn is_identity(&self) -> bool {
        self.0.is_identity() && self.1.is_identity()
    }
    #[inline]
    fn is_index_map(&self) -> bool {
        self.0.is_index_map() && self.1.is_index_map()
    }
}

/// The composition of an unempty sequence of maps.
#[derive(Debug, Clone, PartialEq)]
pub struct UniformComposition<M, Array = Vec<M>>(Array)
where
    M: Map,
    Array: Deref<Target = [M]>;

impl<M, Array> UniformComposition<M, Array>
where
    M: Map,
    Array: Deref<Target = [M]>,
{
    /// Returns the composition of an unempty sequence of maps.
    ///
    /// For every consecutive pair of maps in the sequence the input dimension
    /// and length of the first map must equal the output dimension and length
    /// of the second map.
    ///
    /// Returns an [`Error`] if the dimensions and lengths don't match or the
    /// sequence is empty.
    pub fn new(array: Array) -> Result<Self, Error> {
        let mut iter = array.iter().rev();
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
        self.0.last().unwrap().dim_in()
    }
    #[inline]
    fn dim_out(&self) -> usize {
        self.0.first().unwrap().dim_out()
    }
    #[inline]
    fn delta_dim(&self) -> usize {
        self.dim_out() - self.dim_in()
    }
    #[inline]
    fn len_in(&self) -> usize {
        self.0.last().unwrap().len_in()
    }
    #[inline]
    fn len_out(&self) -> usize {
        self.0.first().unwrap().len_out()
    }
    #[inline]
    fn apply_inplace_unchecked(
        &self,
        index: usize,
        coords: &mut [f64],
        stride: usize,
        offset: usize,
    ) -> usize {
        self.iter().rev().fold(index, |index, map| {
            map.apply_inplace_unchecked(index, coords, stride, offset)
        })
    }
    #[inline]
    fn apply_index_unchecked(&self, index: usize) -> usize {
        self.iter()
            .rev()
            .fold(index, |index, map| map.apply_index_unchecked(index))
    }
    #[inline]
    fn apply_indices_inplace_unchecked(&self, indices: &mut [usize]) {
        self.iter()
            .rev()
            .for_each(|map| map.apply_indices_inplace_unchecked(indices));
    }
    #[inline]
    fn unapply_indices_unchecked<T: UnapplyIndicesData>(&self, indices: &[T]) -> Vec<T> {
        let mut iter = self.iter();
        let indices = iter.next().unwrap().unapply_indices_unchecked(indices);
        iter.fold(indices, |indices, map| {
            map.unapply_indices_unchecked(&indices)
        })
    }
    #[inline]
    fn is_identity(&self) -> bool {
        self.iter().all(|map| map.is_identity())
    }
    #[inline]
    fn is_index_map(&self) -> bool {
        self.iter().all(|map| map.is_index_map())
    }
}

/// The concatenation of two maps.
#[derive(Debug, Clone, PartialEq)]
pub struct BinaryConcat<M0: Map, M1: Map>(M0, M1);

impl<M0: Map, M1: Map> BinaryConcat<M0, M1> {
    /// Returns the concatenation of two maps.
    ///
    /// The two maps must have the same input and output dimensions and the
    /// same output length. The maps must not overlap.
    ///
    /// Returns an [`Error`] if the dimensions and lengths don't match.
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
    /// Returns the first map of the concatenation.
    pub fn first(&self) -> &M0 {
        &self.0
    }
    /// Returns the second map of the concatenation.
    pub fn second(&self) -> &M1 {
        &self.1
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
            self.0
                .apply_inplace_unchecked(index, coords, stride, offset)
        } else {
            self.1
                .apply_inplace_unchecked(index - self.0.len_in(), coords, stride, offset)
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
        result.extend(
            self.1
                .unapply_indices_unchecked(indices)
                .into_iter()
                .map(|i| i.set(i.get() + self.0.len_in())),
        );
        result
    }
    #[inline]
    fn is_identity(&self) -> bool {
        false
    }
    #[inline]
    fn is_index_map(&self) -> bool {
        self.0.is_index_map() && self.1.is_index_map()
    }
}

/// The concatenation of an unempty sequence of maps.
#[derive(Debug, Clone, PartialEq)]
pub struct UniformConcat<M, Array = Vec<M>>(Array)
where
    M: Map,
    Array: Deref<Target = [M]>;

impl<M, Array> UniformConcat<M, Array>
where
    M: Map,
    Array: Deref<Target = [M]>,
{
    /// Returns the concatenation of an unempty sequence of maps.
    ///
    /// The two maps must have the same input and output dimensions and the
    /// same output length. The maps must not overlap.
    ///
    /// Returns an [`Error`] if the dimensions and lengths don't match.
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
        unreachable! {}
    }
    #[inline]
    fn apply_index_unchecked(&self, mut index: usize) -> usize {
        for map in self.iter() {
            if index < map.len_in() {
                return map.apply_index_unchecked(index);
            }
            index -= map.len_in();
        }
        unreachable! {}
    }
    #[inline]
    fn unapply_indices_unchecked<T: UnapplyIndicesData>(&self, indices: &[T]) -> Vec<T> {
        let mut iter = self.iter();
        let map = iter.next().unwrap();
        let mut result = map.unapply_indices_unchecked(indices);
        let mut offset = map.len_in();
        for map in iter {
            result.extend(
                map.unapply_indices_unchecked(indices)
                    .into_iter()
                    .map(|i| i.set(i.get() + offset)),
            );
            offset += map.len_in();
        }
        result
    }
    #[inline]
    fn is_identity(&self) -> bool {
        false
    }
    #[inline]
    fn is_index_map(&self) -> bool {
        self.iter().all(|item| item.is_index_map())
    }
}

/// The product of two maps.
#[derive(Debug, Clone, PartialEq)]
pub struct BinaryProduct<M0: Map, M1: Map>(M0, M1);

impl<M0: Map, M1: Map> BinaryProduct<M0, M1> {
    /// Returns the product of two maps.
    pub fn new(map0: M0, map1: M1) -> Self {
        Self(map0, map1)
    }
    /// Returns the first term of the product.
    pub fn first(&self) -> &M0 {
        &self.0
    }
    /// Returns the second term of the product.
    pub fn second(&self) -> &M1 {
        &self.1
    }
}

impl<M0: Map, M1: Map> Map for BinaryProduct<M0, M1> {
    #[inline]
    fn dim_in(&self) -> usize {
        self.0.dim_in() + self.1.dim_in()
    }
    #[inline]
    fn dim_out(&self) -> usize {
        self.0.dim_out() + self.0.dim_out()
    }
    #[inline]
    fn delta_dim(&self) -> usize {
        self.0.delta_dim() + self.1.delta_dim()
    }
    #[inline]
    fn len_in(&self) -> usize {
        self.0.len_in() * self.1.len_in()
    }
    #[inline]
    fn len_out(&self) -> usize {
        self.0.len_out() * self.1.len_out()
    }
    #[inline]
    fn apply_inplace_unchecked(
        &self,
        index: usize,
        coords: &mut [f64],
        stride: usize,
        offset: usize,
    ) -> usize {
        let (index0, index1) = index.div_rem(&self.1.len_in());
        let index0 = self
            .0
            .apply_inplace_unchecked(index0, coords, stride, offset);
        let index1 =
            self.1
                .apply_inplace_unchecked(index1, coords, stride, offset + self.0.dim_out());
        index0 * self.1.len_out() + index1
    }
    #[inline]
    fn apply_index_unchecked(&self, index: usize) -> usize {
        let (index0, index1) = index.div_rem(&self.1.len_in());
        let index0 = self.0.apply_index_unchecked(index0);
        let index1 = self.1.apply_index_unchecked(index1);
        index0 * self.1.len_out() + index1
    }
    #[inline]
    fn unapply_indices_unchecked<T: UnapplyIndicesData>(&self, indices: &[T]) -> Vec<T> {
        // TODO: collect unique indices per map, unapply and merge
        let idx: Vec<_> = indices
            .iter()
            .enumerate()
            .map(|(i, j)| {
                let (j, k) = j.get().div_rem(&self.1.len_out());
                UnapplyBinaryProduct(i, k, j)
            })
            .collect();
        let mut idx = self.0.unapply_indices_unchecked(&idx);
        idx.iter_mut()
            .for_each(|UnapplyBinaryProduct(_, ref mut j, ref mut k)| std::mem::swap(j, k));
        let idx = self.1.unapply_indices_unchecked(&idx);
        let idx = idx
            .into_iter()
            .map(|UnapplyBinaryProduct(i, j, k)| indices[i].set(j * self.1.len_out() + k))
            .collect();
        idx
    }
    #[inline]
    fn is_identity(&self) -> bool {
        self.0.is_identity() && self.1.is_identity()
    }
    #[inline]
    fn is_index_map(&self) -> bool {
        self.0.is_index_map() && self.1.is_index_map()
    }
}

#[derive(Debug, Clone)]
struct UnapplyBinaryProduct(usize, usize, usize);

impl UnapplyIndicesData for UnapplyBinaryProduct {
    fn get(&self) -> usize {
        self.2
    }
    fn set(&self, index: usize) -> Self {
        Self(self.0, self.1, index)
    }
}

/// The product of a sequence of maps.
#[derive(Debug, Clone, PartialEq)]
pub struct UniformProduct<M, Array = Vec<M>>(Array)
where
    M: Map,
    Array: Deref<Target = [M]>;

impl<M, Array> UniformProduct<M, Array>
where
    M: Map,
    Array: Deref<Target = [M]>,
{
    /// Returns the product of a sequence of maps.
    pub fn new(array: Array) -> Self {
        Self(array)
    }
    /// Returns an iterator of the terms of the product.
    pub fn iter<'a>(&'a self) -> std::slice::Iter<'a, M> {
        self.0.iter()
    }
    pub fn strides_out(&self) -> Vec<usize> {
        let mut strides = Vec::with_capacity(self.0.len());
        let mut stride = 1;
        for map in self.iter().rev() {
            strides.push(stride);
            stride *= map.len_out();
        }
        strides.reverse();
        strides
    }
    pub fn offsets_out(&self) -> Vec<usize> {
        let mut offsets = Vec::with_capacity(self.0.len());
        let mut offset = 0;
        for map in self.iter() {
            offsets.push(offset);
            offset += map.dim_out();
        }
        offsets
    }
}

//impl<M, Array> Map for UniformProduct<M, Array>
//where
//    M: Map,
//    Array: Deref<Target = [M]>,
//{
//    #[inline]
//    fn dim_in(&self) -> usize {
//        self.iter().map(|map| map.dim_in()).sum()
//    }
//    #[inline]
//    fn dim_out(&self) -> usize {
//        self.iter().map(|map| map.dim_out()).sum()
//    }
//    #[inline]
//    fn delta_dim(&self) -> usize {
//        self.iter().map(|map| map.delta_dim()).sum()
//    }
//    #[inline]
//    fn len_in(&self) -> usize {
//        self.iter().map(|map| map.len_in()).product()
//    }
//    #[inline]
//    fn len_out(&self) -> usize {
//        self.iter().map(|map| map.len_out()).product()
//    }
//    #[inline]
//    fn apply_inplace_unchecked(
//        &self,
//        index: usize,
//        coords: &mut [f64],
//        stride: usize,
//        mut offset: usize,
//    ) -> usize {
//        let mut iout = 0;
//        for (map, iin) in self.iter().zip(self.unravel_index(index)) {
//            iout = iout * map.len_out() + self.apply_inplace_unchecked(iin, coords, stride, offset);
//            offset += map.dim_out();
//        }
//        out_index
//    }
//    #[inline]
//    fn apply_index_unchecked(&self, mut index: usize) -> usize {
//        let mut out_index = 0;
//        for (map, iin) in self.iter().zip(self.unravel_index(index)) {
//            iout = iout * map.len_out() + self.apply_index_unchecked(iin);
//        }
//        iout
//    }
//    #[inline]
//    fn unapply_indices_unchecked<T: UnapplyIndicesData>(&self, indices: &[T]) -> Vec<T> {
//        let indices: Vec<_> = indices
//            .iter()
//            .map(|k| {
//                let (i, j) = k.get().div_rem(&self.0.len_in());
//                UnapplyBinaryProduct(i, j, k.clone())
//            })
//            .collect();
//
//
//        let mut iter = self.iter();
//        let map = iter.next().unwrap();
//        let mut result = map.unapply_indices_unchecked(indices);
//        let mut offset = map.len_in();
//        for map in iter {
//            result.extend(
//                map.unapply_indices_unchecked(indices)
//                    .into_iter()
//                    .map(|i| i.set(i.get() + offset)),
//            );
//            offset += map.len_in();
//        }
//        result
//    }
//    #[inline]
//    fn is_identity(&self) -> bool {
//        false
//    }
//}

macro_rules! dispatch {
    (
        $vis:vis fn $fn:ident$(<$genarg:ident: $genpath:path>)?(
            &$self:ident $(, $arg:ident: $ty:ty)* $(,)?
        ) $(-> $ret:ty)?
    ) => {
        #[inline]
        $vis fn $fn$(<$genarg: $genpath>)?(&$self $(, $arg: $ty)*) $(-> $ret)? {
            if $self.0.deref().len() == 1 {
                $self.0.deref()[0].$fn($($arg),*)
            } else {
                BinaryProduct(
                    UniformProduct(&$self.0.deref()[..1]),
                    UniformProduct(&$self.0.deref()[1..]),
                ).$fn($($arg),*)
            }
        }
    };
    ($vis:vis fn $fn:ident(&mut $self:ident $(, $arg:ident: $ty:ty)*) $(-> $ret:ty)?) => {
        #[inline]
        $vis fn $fn(&mut $self $(, $arg: $ty)*) $(-> $ret)? {
            if $self.0.deref().len() == 1 {
                $self.0.deref_mut()[0].$fn($($arg),*)
            } else {
                BinaryProduct(
                    UniformProduct(&$self.0.deref_mut()[..1]),
                    UniformProduct(&$self.0.deref_mut()[1..]),
                ).$fn($($arg),*)
            }
        }
    };
}

impl<M, Array> Map for UniformProduct<M, Array>
where
    M: Map,
    Array: Deref<Target = [M]>,
{
    dispatch! {fn len_out(&self) -> usize}
    dispatch! {fn len_in(&self) -> usize}
    dispatch! {fn dim_out(&self) -> usize}
    dispatch! {fn dim_in(&self) -> usize}
    dispatch! {fn delta_dim(&self) -> usize}
    dispatch! {fn apply_inplace_unchecked(
        &self,
        index: usize,
        coords: &mut [f64],
        stride: usize,
        offset: usize,
    ) -> usize}
    dispatch! {fn apply_inplace(
        &self,
        index: usize,
        coords: &mut [f64],
        stride: usize,
        offset: usize,
    ) -> Result<usize, Error>}
    dispatch! {fn apply_index_unchecked(&self, index: usize) -> usize}
    dispatch! {fn apply_index(&self, index: usize) -> Option<usize>}
    dispatch! {fn apply_indices_inplace_unchecked(&self, indices: &mut [usize])}
    dispatch! {fn apply_indices(&self, indices: &[usize]) -> Option<Vec<usize>>}
    dispatch! {fn unapply_indices_unchecked<T: UnapplyIndicesData>(&self, indices: &[T]) -> Vec<T>}
    dispatch! {fn unapply_indices<T: UnapplyIndicesData>(&self, indices: &[T]) -> Option<Vec<T>>}
    dispatch! {fn is_identity(&self) -> bool}
    dispatch! {fn is_index_map(&self) -> bool}
}

impl<M: Map> FromIterator<M> for UniformProduct<M, Vec<M>> {
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = M>,
    {
        UniformProduct::new(iter.into_iter().collect())
    }
}

// pub struct OptionReorder<M: Map, I>(map, Option<I>);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert_map_apply;
    use crate::prim_comp;
    use approx::assert_abs_diff_eq;
    use std::iter;

    #[test]
    fn uniform_composition1() {
        let map = UniformComposition::new(vec![prim_comp![Point * 1]]).unwrap();
        assert_eq!(map.len_in(), 1);
        assert_eq!(map.len_out(), 1);
        assert_eq!(map.dim_in(), 0);
        assert_eq!(map.delta_dim(), 0);
        assert_eq!(map.apply_index(0), Some(0));
        assert_eq!(map.apply_index(1), None);
    }

    #[test]
    fn uniform_composition2() {
        let map = UniformComposition::new(vec![
            prim_comp![Point*12 <- Transpose(4, 3)],
            prim_comp![Point*12 <- Take([1, 0], 3)],
        ])
        .unwrap();
        assert_eq!(map.len_in(), 8);
        assert_eq!(map.len_out(), 12);
        assert_eq!(map.dim_in(), 0);
        assert_eq!(map.delta_dim(), 0);
        assert_eq!(map.apply_index(0), Some(4));
        assert_eq!(map.apply_index(1), Some(0));
        assert_eq!(map.apply_index(2), Some(5));
        assert_eq!(map.apply_index(3), Some(1));
        assert_eq!(map.apply_index(4), Some(6));
        assert_eq!(map.apply_index(5), Some(2));
        assert_eq!(map.apply_index(6), Some(7));
        assert_eq!(map.apply_index(7), Some(3));
        assert_eq!(map.apply_index(8), None);
    }

    #[test]
    fn uniform_composition3() {
        let map = UniformComposition::new(vec![
            prim_comp![Line*1 <- Children],
            prim_comp![Line*2 <- Children],
            prim_comp![Line*4 <- Edges],
        ])
        .unwrap();
        assert_eq!(map.len_in(), 8);
        assert_eq!(map.len_out(), 1);
        assert_eq!(map.dim_in(), 0);
        assert_eq!(map.delta_dim(), 1);
        assert_map_apply!(map, 0, [[]], 0, [[0.25]]);
        assert_map_apply!(map, 1, [[]], 0, [[0.00]]);
        assert_map_apply!(map, 2, [[]], 0, [[0.50]]);
        assert_map_apply!(map, 3, [[]], 0, [[0.25]]);
        assert_map_apply!(map, 4, [[]], 0, [[0.75]]);
        assert_map_apply!(map, 5, [[]], 0, [[0.50]]);
        assert_map_apply!(map, 6, [[]], 0, [[1.00]]);
        assert_map_apply!(map, 7, [[]], 0, [[0.75]]);
        assert_eq!(map.apply_index(8), None);
        assert_eq!(
            map.unapply_indices(&[0]),
            Some(vec![0, 1, 2, 3, 4, 5, 6, 7])
        );
    }

    #[test]
    fn uniform_concat() {
        let map = UniformConcat::new(vec![
            prim_comp![Line*3 <- Take([0], 3)],
            prim_comp![Line*3 <- Take([1], 3) <- Children],
            prim_comp![Line*3 <- Take([2], 3) <- Children <- Children],
        ])
        .unwrap();
        assert_eq!(map.len_out(), 3);
        assert_eq!(map.len_in(), 7);
        assert_eq!(map.dim_in(), 1);
        assert_eq!(map.dim_out(), 1);
        assert_map_apply!(map, 0, [[0.0], [1.0]], 0, [[0.00], [1.00]]);
        assert_map_apply!(map, 1, [[0.0], [1.0]], 1, [[0.00], [0.50]]);
        assert_map_apply!(map, 2, [[0.0], [1.0]], 1, [[0.50], [1.00]]);
        assert_map_apply!(map, 3, [[0.0], [1.0]], 2, [[0.00], [0.25]]);
        assert_map_apply!(map, 4, [[0.0], [1.0]], 2, [[0.25], [0.50]]);
        assert_map_apply!(map, 5, [[0.0], [1.0]], 2, [[0.50], [0.75]]);
        assert_map_apply!(map, 6, [[0.0], [1.0]], 2, [[0.75], [1.00]]);
        assert_eq!(map.apply_index(7), None);
        assert_eq!(map.unapply_indices(&[0, 2]), Some(vec![0, 3, 4, 5, 6]));
    }

    #[test]
    fn uniform_product1() {
        let map = UniformProduct::new(vec![prim_comp![Line*2 <- Edges], prim_comp![Line * 3]]);
        assert_eq!(map.len_out(), 6);
        assert_eq!(map.len_in(), 12);
        assert_eq!(map.dim_in(), 1);
        assert_eq!(map.dim_out(), 2);
        assert_map_apply!(map, 0, [[0.2], [0.3]], 0, [[1.0, 0.2], [1.0, 0.3]]);
        assert_map_apply!(map, 1, [[0.2], [0.3]], 1, [[1.0, 0.2], [1.0, 0.3]]);
        assert_map_apply!(map, 2, [[0.2], [0.3]], 2, [[1.0, 0.2], [1.0, 0.3]]);
        assert_map_apply!(map, 3, [[0.2], [0.3]], 0, [[0.0, 0.2], [0.0, 0.3]]);
        assert_map_apply!(map, 4, [[0.2], [0.3]], 1, [[0.0, 0.2], [0.0, 0.3]]);
        assert_map_apply!(map, 5, [[0.2], [0.3]], 2, [[0.0, 0.2], [0.0, 0.3]]);
        assert_map_apply!(map, 6, [[0.2], [0.3]], 3, [[1.0, 0.2], [1.0, 0.3]]);
        assert_map_apply!(map, 7, [[0.2], [0.3]], 4, [[1.0, 0.2], [1.0, 0.3]]);
        assert_map_apply!(map, 8, [[0.2], [0.3]], 5, [[1.0, 0.2], [1.0, 0.3]]);
        assert_map_apply!(map, 9, [[0.2], [0.3]], 3, [[0.0, 0.2], [0.0, 0.3]]);
        assert_map_apply!(map, 10, [[0.2], [0.3]], 4, [[0.0, 0.2], [0.0, 0.3]]);
        assert_map_apply!(map, 11, [[0.2], [0.3]], 5, [[0.0, 0.2], [0.0, 0.3]]);
        assert_eq!(map.apply_index(12), None);
        assert_eq!(map.unapply_indices(&[0]), Some(vec![0, 3]));
        assert_eq!(map.unapply_indices(&[1]), Some(vec![1, 4]));
        assert_eq!(map.unapply_indices(&[2]), Some(vec![2, 5]));
        assert_eq!(map.unapply_indices(&[3]), Some(vec![6, 9]));
        assert_eq!(map.unapply_indices(&[4]), Some(vec![7, 10]));
        assert_eq!(map.unapply_indices(&[5]), Some(vec![8, 11]));
    }
}
