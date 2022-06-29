use crate::{Error, Map, UnapplyIndicesData};
use std::ops::Deref;
use num::Integer as _;

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
        let index = self
            .0
            .apply_inplace_unchecked(index, coords, stride, offset);
        self.1
            .apply_inplace_unchecked(index, coords, stride, offset)
    }
    #[inline]
    fn apply_index_unchecked(&self, index: usize) -> usize {
        self.1
            .apply_index_unchecked(self.0.apply_index_unchecked(index))
    }
    #[inline]
    fn apply_indices_inplace_unchecked(&self, indices: &mut [usize]) {
        self.0.apply_indices_inplace_unchecked(indices);
        self.1.apply_indices_inplace_unchecked(indices);
    }
    #[inline]
    fn unapply_indices_unchecked<T: UnapplyIndicesData>(&self, indices: &[T]) -> Vec<T> {
        self.0
            .unapply_indices_unchecked(&self.1.unapply_indices_unchecked(indices))
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
        self.iter()
            .fold(index, |index, map| map.apply_index_unchecked(index))
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
        iter.fold(indices, |indices, map| {
            map.unapply_indices_unchecked(&indices)
        })
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
}

#[derive(Debug, Clone, PartialEq)]
pub struct BinaryProduct<M0: Map, M1: Map>(M0, M1);

impl<M0: Map, M1: Map> BinaryProduct<M0, M1> {
    pub fn new(map0: M0, map1: M1) -> Self {
        Self(map0, map1)
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
        let (index0, index1) = index.div_rem(&self.0.len_in());
        let index0 = self
            .0
            .apply_inplace_unchecked(index0, coords, stride, offset);
        let index1 =
            self.1
                .apply_inplace_unchecked(index1, coords, stride, offset + self.0.dim_out());
        index0 * self.0.len_out() + index1
    }
    #[inline]
    fn apply_index_unchecked(&self, index: usize) -> usize {
        let (index0, index1) = index.div_rem(&self.0.len_in());
        let index0 = self.0.apply_index_unchecked(index0);
        let index1 = self.1.apply_index_unchecked(index1);
        index0 * self.0.len_out() + index1
    }
    #[inline]
    fn unapply_indices_unchecked<T: UnapplyIndicesData>(&self, indices: &[T]) -> Vec<T> {
        let indices: Vec<_> = indices
            .iter()
            .map(|k| {
                let (i, j) = k.get().div_rem(&self.0.len_in());
                UnapplyBinaryProduct(i, j, k.clone())
            })
            .collect();
        let mut indices = self.0.unapply_indices_unchecked(&indices);
        indices
            .iter_mut()
            .for_each(|UnapplyBinaryProduct(ref mut i, ref mut j, _)| std::mem::swap(i, j));
        let indices = self.1.unapply_indices_unchecked(&indices);
        indices
            .into_iter()
            .map(|UnapplyBinaryProduct(i, j, k)| k.set(j * self.0.len_out() + i))
            .collect()
    }
    #[inline]
    fn is_identity(&self) -> bool {
        self.0.is_identity() && self.1.is_identity()
    }
}

#[derive(Debug, Clone)]
struct UnapplyBinaryProduct<T: UnapplyIndicesData>(usize, usize, T);

impl<T: UnapplyIndicesData> UnapplyIndicesData for UnapplyBinaryProduct<T> {
    fn get(&self) -> usize {
        self.0
    }
    fn set(&self, index: usize) -> Self {
        Self(index, self.1, self.2.clone())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct UniformProduct<M, Array>(Array)
where
    M: Map,
    Array: Deref<Target = [M]>;

impl<M, Array> UniformProduct<M, Array>
where
    M: Map,
    Array: Deref<Target = [M]>,
{
    pub fn new(array: Array) -> Self {
        Self(array)
    }
    pub fn iter<'a>(&'a self) -> std::slice::Iter<'a, M> {
        self.0.iter()
    }
    fn unravel_index(&self, index: usize) -> Vec<usize> {
        let mut indices: Vec<_> = self.iter().rev().scan(index, |index, map| {
            let (i0, i1) = index.div_rem(&map.len_in());
            *index = i0;
            Some(i1)
        }).collect();
        indices.reverse();
        indices
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
            //} else if $self.0.deref().len() == 2 {
            //    BinaryProduct(&$self.0.deref()[0], $self.0.deref()[1]).$fn($($arg),*)
            } else {
                BinaryProduct(UniformProduct(&$self.0.deref()[..1]), UniformProduct(&$self.0.deref()[1..])).$fn($($arg),*)
            }
        }
    };
    ($vis:vis fn $fn:ident(&mut $self:ident $(, $arg:ident: $ty:ty)*) $(-> $ret:ty)?) => {
        #[inline]
        $vis fn $fn(&mut $self $(, $arg: $ty)*) $(-> $ret)? {
            if $self.0.deref().len() == 1 {
                $self.0.deref_mut()[0].$fn($($arg),*)
            //} else if $self.0.deref().len() == 2 {
            //    BinaryProduct(&$self.0.deref_mut()[0], &$self.0.deref_mut()[1]).$fn($($arg),*)
            } else {
                BinaryProduct(UniformProduct(&$self.0.deref_mut()[..1]), UniformProduct(&$self.0.deref_mut()[1..])).$fn($($arg),*)
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
    ) -> Option<usize>}
    dispatch! {fn apply_index_unchecked(&self, index: usize) -> usize}
    dispatch! {fn apply_index(&self, index: usize) -> Option<usize>}
    dispatch! {fn apply_indices_inplace_unchecked(&self, indices: &mut [usize])}
    dispatch! {fn apply_indices(&self, indices: &[usize]) -> Option<Vec<usize>>}
    dispatch! {fn unapply_indices_unchecked<T: UnapplyIndicesData>(&self, indices: &[T]) -> Vec<T>}
    dispatch! {fn unapply_indices<T: UnapplyIndicesData>(&self, indices: &[T]) -> Option<Vec<T>>}
    dispatch! {fn is_identity(&self) -> bool}
}

// pub struct OptionReorder<M: Map, I>(map, Option<I>);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert_map_apply;
    use crate::elementary::{Elementary, WithBounds};
    use crate::simplex::Simplex::*;
    use approx::assert_abs_diff_eq;
    use std::iter;

    macro_rules! elementaries {
        (Point*$len_out:literal $($tail:tt)*) => {{
            let mut comp: Vec<Elementary> = Vec::new();
            elementaries!{@push comp, Point; $($tail)*}
            comp.reverse();
            WithBounds::from_output(comp, 0, $len_out).unwrap()
        }};
        ($simplex:ident*$len_out:literal $($tail:tt)*) => {{
            let mut comp = Vec::new();
            elementaries!{@push comp, $simplex; $($tail)*}
            comp.reverse();
            WithBounds::from_output(comp, $simplex.dim(), $len_out).unwrap()
        }};
        (@push $comp:ident, $simplex:expr;) => {};
        (@push $comp:ident, $simplex:expr; <- Children $($tail:tt)*) => {{
            $comp.push(Elementary::new_children($simplex));
            elementaries!{@push $comp, $simplex; $($tail)*}
        }};
        (@push $comp:ident, $simplex:expr; <- Edges $($tail:tt)*) => {{
            $comp.push(Elementary::new_edges($simplex));
            elementaries!{@push $comp, $simplex.edge_simplex(); $($tail)*}
        }};
        (@push $comp:ident, $simplex:expr; <- Transpose($len1:expr, $len2:expr) $($tail:tt)*) => {{
            $comp.push(Elementary::new_transpose($len1, $len2));
            elementaries!{@push $comp, $simplex; $($tail)*}
        }};
        (@push $comp:ident, $simplex:expr; <- Take($indices:expr, $len:expr) $($tail:tt)*) => {{
            $comp.push(Elementary::new_take($indices.to_vec(), $len));
            elementaries!{@push $comp, $simplex; $($tail)*}
        }};
    }

    #[test]
    fn uniform_composition1() {
        let map = UniformComposition::new(vec![elementaries![Point * 1]]).unwrap();
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
            elementaries![Point*12 <- Take([1, 0], 3)],
            elementaries![Point*12 <- Transpose(4, 3)],
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
            elementaries![Line*4 <- Edges],
            elementaries![Line*2 <- Children],
            elementaries![Line*1 <- Children],
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
            elementaries![Line*3 <- Take([0], 3)],
            elementaries![Line*3 <- Take([1], 3) <- Children],
            elementaries![Line*3 <- Take([2], 3) <- Children <- Children],
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
        let map = UniformProduct::new(vec![
            elementaries![Line*1 <- Edges],
            elementaries![Line*1 <- Children],
        ]);
        assert_eq!(map.len_out(), 1);
        assert_eq!(map.len_in(), 4);
        assert_eq!(map.dim_in(), 1);
        assert_eq!(map.dim_out(), 2);
        assert_map_apply!(map, 0, [[0.0], [1.0]], 0, [[1.0, 0.0], [1.0, 0.5]]);
        assert_map_apply!(map, 1, [[0.0], [1.0]], 0, [[1.0, 0.5], [1.0, 1.0]]);
        assert_map_apply!(map, 2, [[0.0], [1.0]], 0, [[0.0, 0.0], [0.0, 0.5]]);
        assert_map_apply!(map, 3, [[0.0], [1.0]], 0, [[0.0, 0.5], [0.0, 1.0]]);
    }
}
