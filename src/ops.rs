use crate::{Map, UnapplyIndicesData};
use num::Integer as _;
use std::ops::Deref;

#[derive(Debug, Clone, PartialEq)]
pub struct BinaryComposition<Inner: Map, Outer: Map> {
    inner: Inner,
    outer: Outer,
}

impl<Inner: Map, Outer: Map> Map for BinaryComposition<Inner, Outer> {
    #[inline]
    fn dim_in(&self) -> usize {
        comp_dim_out_in(&self.outer, self.inner.dim_out(), self.inner.dim_in()).1
    }
    #[inline]
    fn delta_dim(&self) -> usize {
        self.inner.delta_dim() + self.outer.delta_dim()
    }
    #[inline]
    fn mod_in(&self) -> usize {
        comp_mod_out_in(&self.outer, self.inner.mod_out(), self.inner.mod_in()).0
    }
    #[inline]
    fn mod_out(&self) -> usize {
        comp_mod_out_in(&self.outer, self.inner.mod_out(), self.inner.mod_in()).1
    }
    #[inline]
    fn apply_inplace(
        &self,
        index: usize,
        coordinates: &mut [f64],
        stride: usize,
        offset: usize,
    ) -> usize {
        let index = self.inner.apply_inplace(index, coordinates, stride, offset);
        self.outer.apply_inplace(index, coordinates, stride, offset)
    }
    #[inline]
    fn apply_index(&self, index: usize) -> usize {
        self.outer.apply_index(self.inner.apply_index(index))
    }
    #[inline]
    fn apply_indices_inplace(&self, indices: &mut [usize]) {
        self.inner.apply_indices_inplace(indices);
        self.outer.apply_indices_inplace(indices);
    }
    #[inline]
    fn unapply_indices<T: UnapplyIndicesData>(&self, indices: &[T]) -> Vec<T> {
        self.inner
            .unapply_indices(&self.outer.unapply_indices(indices))
    }
    #[inline]
    fn is_identity(&self) -> bool {
        self.inner.is_identity() && self.outer.is_identity()
    }
}

pub struct UniformComposition<M, Array>(Array)
where
    M: Map,
    Array: Deref<Target = [M]>;

impl<M, Array> UniformComposition<M, Array>
where
    M: Map,
    Array: Deref<Target = [M]>,
{
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
        self.iter()
            .fold((0, 0), |(o, i), map| comp_dim_out_in(map, o, i))
            .1
    }
    #[inline]
    fn delta_dim(&self) -> usize {
        self.iter().map(|map| map.delta_dim()).sum()
    }
    #[inline]
    fn mod_in(&self) -> usize {
        self.iter()
            .fold((1, 1), |(o, i), map| comp_mod_out_in(map, o, i))
            .1
    }
    #[inline]
    fn mod_out(&self) -> usize {
        self.iter()
            .fold((1, 1), |(o, i), map| comp_mod_out_in(map, o, i))
            .0
    }
    #[inline]
    fn apply_inplace(
        &self,
        index: usize,
        coords: &mut [f64],
        stride: usize,
        offset: usize,
    ) -> usize {
        self.iter().fold(index, |index, map| {
            map.apply_inplace(index, coords, stride, offset)
        })
    }
    #[inline]
    fn apply_index(&self, index: usize) -> usize {
        self.iter().fold(index, |index, map| map.apply_index(index))
    }
    #[inline]
    fn apply_indices_inplace(&self, indices: &mut [usize]) {
        self.iter()
            .for_each(|map| map.apply_indices_inplace(indices));
    }
    #[inline]
    fn unapply_indices<T: UnapplyIndicesData>(&self, indices: &[T]) -> Vec<T> {
        let mut iter = self.iter().rev();
        if let Some(map) = iter.next() {
            iter.fold(map.unapply_indices(indices), |indices, map| {
                map.unapply_indices(&indices)
            })
        } else {
            Vec::new()
        }
    }
    #[inline]
    fn is_identity(&self) -> bool {
        self.iter().all(|map| map.is_identity())
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConcatError {
    ModulusNotAMultiple,
    DeltaDimsDiffer,
}

impl std::error::Error for ConcatError {}

impl std::fmt::Display for ConcatError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::ModulusNotAMultiple => write!(
                f,
                "The given output modulus is not a multiple of the output moduli of the given maps.",
            ),
            Self::DeltaDimsDiffer => write!(
                f, "Cannot concatenate maps with different delta dimensions."),
        }
    }
}

//fold_map! {BinaryConcat, UniformConcat;
//    type Item<M> = (M, usize);
//    dim_in
//        |(map, _)| map.dim_in()
//        |dim_in, (map, _)| std::cmp::max(dim_in, map.dim_in())
//        |dim_in| dim_in
//    delta_dim
//        |(map, _)| map.delta_dim()
//        |delta_dim, (map, _)| delta_dim + map.delta_dim()
//        |delta_dim| delta_dim
//    mod_out_in
//        |(map, len)| (map.apply_mod_in_to_out(len).unwrap(), len)
//        |(mod_out, mod_in), (_, len)| (mod_out, mod_in + len)
//        |state| state
//    apply
//        |apply, index, (map, len)| {
//            let (i, j) = index.div_rem(&self.mod_in());
//            if j < len {
//                (apply(map, j + i * len), i, j, len)
//            } else {
//                (index, i, j, len)
//            }
//        }
//        |apply, (index, i, j, k), (map, len)| {
//            if k <= j && j < k + len {
//                (apply(map, j + i * len), i, j, k + len)
//            } else {
//                (index, i, j, k + len)
//            }
//        }
//        |(index, i, j, k)| done
//    unapply
//        |indices, (map, len)| (
//            map.unapply_indices(indices)
//                .into_iter()
//                .map(|i| i.set(i.get() / len * self.mod_in() + i.get() % len))
//                .collect::<Vec<_>>(),
//            len
//        )
//        |(mut indices, k), (map, len)| (
//            indices.extend(
//                map.unapply_indices(indices)
//                    .into_iter()
//                    .map(|i| i.set(i.get() / len * self.mod_in() + i.get() % len + k))),
//            k + len
//        )
//        |(indices, k)| indices
//}

pub struct BinaryConcat<M1: Map, M2: Map> {
    map1: M1,
    map2: M2,
    mod_out: usize,
    mod_in1: usize,
    mod_in2: usize,
}

impl<M1: Map, M2: Map> BinaryConcat<M1, M2> {
    pub fn new(map1: M1, map2: M2, mod_out: usize) -> Result<Self, ConcatError> {
        let mod_in1 = map1
            .apply_mod_out_to_in(mod_out)
            .ok_or(ConcatError::ModulusNotAMultiple)?;
        let mod_in2 = map2
            .apply_mod_out_to_in(mod_out)
            .ok_or(ConcatError::ModulusNotAMultiple)?;
        if map1.delta_dim() != map2.delta_dim() {
            return Err(ConcatError::DeltaDimsDiffer);
        }
        Ok(Self {
            map1,
            map2,
            mod_out,
            mod_in1,
            mod_in2,
        })
    }
}

impl<M1: Map, M2: Map> Map for BinaryConcat<M1, M2> {
    #[inline]
    fn dim_in(&self) -> usize {
        std::cmp::max(self.map1.dim_in(), self.map2.dim_in())
    }
    #[inline]
    fn delta_dim(&self) -> usize {
        self.map1.delta_dim()
    }
    #[inline]
    fn mod_in(&self) -> usize {
        self.mod_in1 + self.mod_in2
    }
    #[inline]
    fn mod_out(&self) -> usize {
        self.mod_out
    }
    #[inline]
    fn apply_inplace(
        &self,
        index: usize,
        coords: &mut [f64],
        stride: usize,
        offset: usize,
    ) -> usize {
        let (i, j) = index.div_rem(&self.mod_in());
        if j < self.mod_in1 {
            self.map1
                .apply_inplace(j + i * self.mod_in1, coords, stride, offset)
        } else {
            self.map2
                .apply_inplace(j - self.mod_in1 + i * self.mod_in1, coords, stride, offset)
        }
    }
    #[inline]
    fn apply_index(&self, index: usize) -> usize {
        let (i, j) = index.div_rem(&self.mod_in());
        if let Some(j) = j.checked_sub(self.mod_in1) {
            self.map2.apply_index(j + i * self.mod_in2)
        } else {
            self.map1.apply_index(j + i * self.mod_in1)
        }
    }
    #[inline]
    fn unapply_indices<T: UnapplyIndicesData>(&self, indices: &[T]) -> Vec<T> {
        let mut result: Vec<_> = self
            .map1
            .unapply_indices(indices)
            .into_iter()
            .map(|i| i.set(i.get() / self.mod_in1 * self.mod_in() + i.get() % self.mod_in1))
            .collect();
        result.extend(self.map2.unapply_indices(indices).into_iter().map(|i| {
            i.set(i.get() / self.mod_in2 * self.mod_in() + self.mod_in1 + i.get() % self.mod_in2)
        }));
        result
    }
    #[inline]
    fn is_identity(&self) -> bool {
        self.map1.is_identity() && self.map2.is_identity()
    }
}

struct UniformConcat<M, Array>
where
    M: Map,
    Array: Deref<Target = [(M, usize)]>,
{
    maps: Array,
    mod_out: usize,
}

//impl<M: Map> UniformConcat<M, Array> {
//where
//    M: Map,
//    Array: Deref<Target = [(M, usize)]>,
//{
//    pub fn new(maps: Array M2, mod_out: usize) -> Result<Self, ConcatError> {
//        // TODO: assert unempty
//        let mod_in1 = map1
//            .apply_mod_out_to_in(mod_out)
//            .ok_or(ConcatError::ModulusNotAMultiple)?;
//        let mod_in2 = map2
//            .apply_mod_out_to_in(mod_out)
//            .ok_or(ConcatError::ModulusNotAMultiple)?;
//        if map1.delta_dim() != map2.delta_dim() {
//            return Err(ConcatError::DeltaDimsDiffer);
//        }
//        Ok(Self {
//            map1,
//            map2,
//            mod_out,
//            mod_in1,
//            mod_in2,
//        })
//    }
//}

impl<M, Array> Map for UniformConcat<M, Array>
where
    M: Map,
    Array: Deref<Target = [(M, usize)]>,
{
    #[inline]
    fn dim_in(&self) -> usize {
        self.maps.iter().map(|(map, _)| map.dim_in()).max().unwrap()
    }
    #[inline]
    fn delta_dim(&self) -> usize {
        self.maps.first().unwrap().0.delta_dim()
    }
    #[inline]
    fn mod_in(&self) -> usize {
        self.maps.iter().map(|(_, mod_in)| mod_in).sum()
    }
    #[inline]
    fn mod_out(&self) -> usize {
        self.mod_out
    }
    #[inline]
    fn apply_inplace(
        &self,
        index: usize,
        coords: &mut [f64],
        stride: usize,
        offset: usize,
    ) -> usize {
        let (i, mut j) = index.div_rem(&self.mod_in());
        for (map, mod_in) in self.maps.iter() {
            if j < *mod_in {
                return map.apply_inplace(j + i * mod_in, coords, stride, offset);
            }
            j -= mod_in;
        }
        unreachable! {}
    }
    #[inline]
    fn apply_index(&self, index: usize) -> usize {
        let (i, mut j) = index.div_rem(&self.mod_in());
        for (map, mod_in) in self.maps.iter() {
            if j < *mod_in {
                return map.apply_index(j + i * mod_in);
            }
            j -= mod_in;
        }
        unreachable! {}
    }
    #[inline]
    fn unapply_indices<T: UnapplyIndicesData>(&self, indices: &[T]) -> Vec<T> {
        let mut result = Vec::new();
        let mut offset = 0;
        for (map, mod_in) in self.maps.iter() {
            result.extend(
                map.unapply_indices(indices)
                    .iter()
                    .map(|i| i.set(i.get() / mod_in * self.mod_in() + offset + i.get() % mod_in)),
            );
            offset += mod_in;
        }
        result
    }
    #[inline]
    fn is_identity(&self) -> bool {
        self.maps.iter().all(|(map, _)| map.is_identity())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use crate::elementary::{Elementary, Identity};
    use crate::simplex::Simplex::*;
    use std::iter;
    use crate::assert_map_apply;

    #[test]
    fn uniform_composition0() {
        let map = UniformComposition(Vec::<Elementary>::new());
        assert_eq!(map.mod_in(), 1);
        assert_eq!(map.mod_out(), 1);
        assert_eq!(map.dim_in(), 0);
        assert_eq!(map.delta_dim(), 0);
        assert_map_apply!(map, 0, 0);
        assert_map_apply!(map, 1, 1);
    }

    #[test]
    fn uniform_composition1() {
        let map = UniformComposition(vec![Identity]);
        assert_eq!(map.mod_in(), 1);
        assert_eq!(map.mod_out(), 1);
        assert_eq!(map.dim_in(), 0);
        assert_eq!(map.delta_dim(), 0);
        assert_map_apply!(map, 0, 0);
        assert_map_apply!(map, 1, 1);
    }

    #[test]
    fn uniform_composition2() {
        let map = UniformComposition(vec![
            Elementary::new_take(vec![1, 0], 3),
            Elementary::new_transpose(4, 3),
        ]);
        assert_eq!(map.mod_in(), 8);
        assert_eq!(map.mod_out(), 12);
        assert_eq!(map.dim_in(), 0);
        assert_eq!(map.delta_dim(), 0);
        assert_map_apply!(map, 0, 4);
        assert_map_apply!(map, 1, 0);
        assert_map_apply!(map, 2, 5);
        assert_map_apply!(map, 3, 1);
        assert_map_apply!(map, 4, 6);
        assert_map_apply!(map, 5, 2);
        assert_map_apply!(map, 6, 7);
        assert_map_apply!(map, 7, 3);
        assert_map_apply!(map, 8, 16);
    }

    #[test]
    fn uniform_composition3() {
        let map = UniformComposition(vec![
            Elementary::new_edges(Line),
            Elementary::new_children(Line),
            Elementary::new_children(Line),
        ]);
        assert_eq!(map.mod_in(), 8);
        assert_eq!(map.mod_out(), 1);
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
        assert_map_apply!(map, 8, [[]], 1, [[0.25]]);
        assert_eq!(map.unapply_indices(&[0]), vec![0, 1, 2, 3, 4, 5, 6, 7]);
    }
}

//#[derive(Debug, Clone, Copy, PartialEq)]
//pub struct ComposeCodomainDomainMismatch;
//
//impl std::error::Error for ComposeCodomainDomainMismatch {}
//
//impl std::fmt::Display for ComposeCodomainDomainMismatch {
//    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
//        write!(
//            f,
//            "The codomain of the first map doesn't match the domain of the second map,"
//        )
//    }
//}
//
//#[derive(Debug, Clone, PartialEq)]
//pub struct Composition<Inner, Outer> {
//    inner: Inner,
//    outer: Outer,
//}
//
//impl<Inner: BoundedMap, Outer: BoundedMap> Composition<Inner, Outer> {
//    pub fn new(inner: Inner, outer: Outer) -> Result<Self, ComposeCodomainDomainMismatch> {
//        if inner.len_out() == outer.len_in() && inner.dim_out() == outer.dim_in() {
//            Ok(Self { inner, outer })
//        } else {
//            Err(ComposeCodomainDomainMismatch)
//        }
//    }
//}
//
//impl<Inner: BoundedMap, Outer: BoundedMap> BoundedMap for Composition<Inner, Outer> {
//    fn len_in(&self) -> usize {
//        self.inner.len_in()
//    }
//    fn len_out(&self) -> usize {
//        self.outer.len_out()
//    }
//    fn dim_in(&self) -> usize {
//        self.inner.dim_in()
//    }
//    fn delta_dim(&self) -> usize {
//        self.inner.delta_dim() + self.outer.delta_dim()
//    }
//    fn apply_inplace_unchecked(
//        &self,
//        index: usize,
//        coordinates: &mut [f64],
//        stride: usize,
//        offset: usize,
//    ) -> usize {
//        let index = self
//            .inner
//            .apply_inplace_unchecked(index, coordinates, stride, offset);
//        self.outer
//            .apply_inplace_unchecked(index, coordinates, stride, offset)
//    }
//    fn apply_index_unchecked(&self, index: usize) -> usize {
//        let index = self.inner.apply_index_unchecked(index);
//        self.outer.apply_index_unchecked(index)
//    }
//    fn apply_indices_inplace_unchecked(&self, indices: &mut [usize]) {
//        self.inner.apply_indices_inplace_unchecked(indices);
//        self.outer.apply_indices_inplace_unchecked(indices);
//    }
//    fn unapply_indices_unchecked<T: UnapplyIndicesData>(&self, indices: &[T]) -> Vec<T> {
//        let indices = self.outer.unapply_indices_unchecked(indices);
//        self.inner.unapply_indices_unchecked(&indices)
//    }
//    fn is_identity(&self) -> bool {
//        self.inner.is_identity() && self.outer.is_identity()
//    }
//}
//
//impl<Inner: UnboundedMap, Outer: UnboundedMap> UnboundedMap for Composition<Inner, Outer> {
//    fn mod_in(&self) -> usize {
//        update_mod_out_in(&self.outer, self.inner.mod_out(), self.inner.mod_in()).1
//    }
//    fn mod_out(&self) -> usize {
//        update_mod_out_in(&self.outer, self.inner.mod_out(), self.inner.mod_in()).0
//    }
//    fn dim_in(&self) -> usize {
//        update_dim_out_in(&self.outer, self.inner.dim_out(), self.inner.dim_in()).0
//    }
//    fn delta_dim(&self) -> usize {
//        self.inner.delta_dim() + self.outer.delta_dim()
//    }
//    fn apply_inplace(
//        &self,
//        index: usize,
//        coordinates: &mut [f64],
//        stride: usize,
//        offset: usize,
//    ) -> usize {
//        let index = self.inner.apply_inplace(index, coordinates, stride, offset);
//        self.outer.apply_inplace(index, coordinates, stride, offset)
//    }
//    fn apply_index(&self, index: usize) -> usize {
//        let index = self.inner.apply_index(index);
//        self.outer.apply_index(index)
//    }
//    fn apply_indices_inplace(&self, indices: &mut [usize]) {
//        self.inner.apply_indices_inplace(indices);
//        self.outer.apply_indices_inplace(indices);
//    }
//    fn unapply_indices<T: UnapplyIndicesData>(&self, indices: &[T]) -> Vec<T> {
//        let indices = self.outer.unapply_indices(indices);
//        self.inner.unapply_indices(&indices)
//    }
//    fn is_identity(&self) -> bool {
//        self.inner.is_identity() && self.outer.is_identity()
//    }
//}
//
//impl<Inner: AddOffset, Outer: AddOffset> AddOffset for Composition<Inner, Outer> {
//    fn add_offset(&mut self, offset: usize) {
//        self.inner.add_offset(offset);
//        self.outer.add_offset(offset);
//    }
//}
//
//pub trait Compose: BoundedMap + Sized {
//    fn compose<Rhs: BoundedMap>(
//        self,
//        rhs: Rhs,
//    ) -> Result<Composition<Self, Rhs>, ComposeCodomainDomainMismatch> {
//        Composition::new(self, rhs)
//    }
//}
//
//impl<M: BoundedMap> Compose for M {}
//
//#[derive(Debug, Clone, PartialEq)]
//pub struct Concat<Item: BoundedMap>(Vec<Item>);
//
//impl<Item: BoundedMap> Concat<Item> {
//    pub fn new(items: Vec<Item>) -> Self {
//        // TODO: Return `Result<Self, ...>`.
//        let first = items.first().unwrap();
//        let dim_in = first.dim_in();
//        let delta_dim = first.delta_dim();
//        let len_out = first.len_out();
//        for item in items.iter() {
//            assert_eq!(item.dim_in(), dim_in);
//            assert_eq!(item.delta_dim(), delta_dim);
//            assert_eq!(item.len_out(), len_out);
//        }
//        Self(items)
//    }
//    fn resolve_item_unchecked(&self, mut index: usize) -> (&Item, usize) {
//        for item in self.0.iter() {
//            if index < item.len_in() {
//                return (item, index);
//            }
//            index -= item.len_in();
//        }
//        panic!("index out of range");
//    }
//    pub fn iter(&self) -> impl Iterator<Item = &Item> {
//        self.0.iter()
//    }
//    pub fn into_vec(self) -> Vec<Item> {
//        self.0
//    }
//}
//
//impl<Item: BoundedMap> BoundedMap for Concat<Item> {
//    fn dim_in(&self) -> usize {
//        self.0.first().unwrap().dim_in()
//    }
//    fn delta_dim(&self) -> usize {
//        self.0.first().unwrap().delta_dim()
//    }
//    fn len_out(&self) -> usize {
//        self.0.first().unwrap().len_out()
//    }
//    fn len_in(&self) -> usize {
//        self.iter().map(|item| item.len_in()).sum()
//    }
//    fn apply_inplace_unchecked(
//        &self,
//        index: usize,
//        coordinates: &mut [f64],
//        stride: usize,
//        offset: usize,
//    ) -> usize {
//        let (item, index) = self.resolve_item_unchecked(index);
//        item.apply_inplace_unchecked(index, coordinates, stride, offset)
//    }
//    fn apply_index_unchecked(&self, index: usize) -> usize {
//        let (item, index) = self.resolve_item_unchecked(index);
//        item.apply_index_unchecked(index)
//    }
//    fn unapply_indices_unchecked<T: UnapplyIndicesData>(&self, indices: &[T]) -> Vec<T> {
//        let mut result = Vec::new();
//        let mut offset = 0;
//        for item in &self.0 {
//            result.extend(
//                item.unapply_indices_unchecked(indices)
//                    .into_iter()
//                    .map(|i| i.set(i.get() + offset)),
//            );
//            offset += item.len_in();
//        }
//        result
//    }
//    fn is_identity(&self) -> bool {
//        false
//    }
//}
//
//impl<M: BoundedMap + AddOffset> AddOffset for Concat<M> {
//    fn add_offset(&mut self, offset: usize) {
//        for item in &mut self.0 {
//            item.add_offset(offset);
//        }
//    }
//}
//
//impl<M> PushElementary for Concat<M>
//where
//    M: BoundedMap + PushElementary,
//{
//    fn push_elementary(&mut self, map: &Elementary) {
//        match map {
//            Elementary::Children(_) | Elementary::Edges(_) => {
//                for item in &mut self.0 {
//                    item.push_elementary(map);
//                }
//            }
//            Elementary::Take(take) => {
//                let mut offset = 0;
//                let indices = take.get_indices();
//                let mut indices = indices.iter().cloned().peekable();
//                for item in &mut self.0 {
//                    let len = item.len_in();
//                    let mut item_indices = Vec::new();
//                    while let Some(index) = indices.next_if(|&i| i < offset + len) {
//                        if index < offset {
//                            unimplemented! {"take of concatenation with unordered indices"};
//                        }
//                        item_indices.push(index - offset);
//                    }
//                    item.push_elementary(&Elementary::new_take(item_indices, len));
//                    offset += len;
//                }
//            }
//            _ => unimplemented! {},
//        }
//    }
//}
//
//struct Product<M0, M1>(M0, M1);
//
//impl<M0: BoundedMap, M1: BoundedMap> BoundedMap for Product<M0, M1> {
//    fn dim_in(&self) -> usize {
//        self.0.dim_in() + self.1.dim_in()
//    }
//    fn delta_dim(&self) -> usize {
//        self.0.delta_dim() + self.1.delta_dim()
//    }
//    fn len_out(&self) -> usize {
//        self.0.len_out() * self.1.len_out()
//    }
//    fn len_in(&self) -> usize {
//        self.0.len_in() * self.1.len_in()
//    }
//    fn apply_inplace_unchecked(
//        &self,
//        index: usize,
//        coords: &mut [f64],
//        stride: usize,
//        offset: usize,
//    ) -> usize {
//        let index = self
//            .1
//            .apply_inplace_unchecked(index, coords, stride, offset + self.0.dim_in());
//        let index = Transpose::new(self.0.len_in(), self.1.len_in()).apply_index(index);
//        let index = self
//            .0
//            .apply_inplace_unchecked(index, coords, stride, offset);
//        Transpose::new(self.1.len_in(), self.0.len_in()).apply_index(index)
//    }
//    fn apply_index_unchecked(&self, index: usize) -> usize {
//        let index = self.1.apply_index_unchecked(index);
//        let index = Transpose::new(self.0.len_in(), self.1.len_in()).apply_index(index);
//        let index = self.0.apply_index_unchecked(index);
//        Transpose::new(self.1.len_in(), self.0.len_in()).apply_index(index)
//    }
//    fn apply_indices_inplace_unchecked(&self, indices: &mut [usize]) {
//        self.1.apply_indices_inplace_unchecked(indices);
//        Transpose::new(self.0.len_in(), self.1.len_in()).apply_indices_inplace(indices);
//        self.0.apply_indices_inplace_unchecked(indices);
//        Transpose::new(self.1.len_in(), self.0.len_in()).apply_indices_inplace(indices);
//    }
//    fn unapply_indices_unchecked<T: UnapplyIndicesData>(&self, indices: &[T]) -> Vec<T> {
//        let indices =
//            Transpose::new(self.1.len_in(), self.0.len_in()).unapply_indices(indices);
//        let indices = self.0.unapply_indices_unchecked(&indices);
//        let indices =
//            Transpose::new(self.0.len_in(), self.1.len_in()).unapply_indices(&indices);
//        self.1.unapply_indices_unchecked(&indices)
//    }
//    fn is_identity(&self) -> bool {
//        self.0.is_identity() && self.1.is_identity()
//    }
//}
//
//#[inline]
//fn update_dim_out_in<M: UnboundedMap>(map: &M, dim_out: usize, dim_in: usize) -> (usize, usize) {
//    if let Some(n) = map.dim_in().checked_sub(dim_out) {
//        (map.dim_out(), dim_in + n)
//    } else {
//        (dim_out, dim_in)
//    }
//}
//
//#[inline]
//fn update_mod_out_in<M: UnboundedMap>(map: &M, mod_out: usize, mod_in: usize) -> (usize, usize) {
//    let n = mod_out.lcm(&map.mod_in());
//    (n / map.mod_in() * map.mod_out(), mod_in * n / mod_out)
//}
//
///// Composition.
//impl<Item, Array> UnboundedMap for Array
//where
//    Item: UnboundedMap,
//    Array: Deref<Target = [Item]> + DerefMut + std::fmt::Debug,
//{
//    fn dim_in(&self) -> usize {
//        self.deref()
//            .iter()
//            .rev()
//            .fold((0, 0), |(dim_out, dim_in), item| {
//                update_dim_out_in(item, dim_out, dim_in)
//            })
//            .1
//    }
//    fn delta_dim(&self) -> usize {
//        self.iter().map(|item| item.delta_dim()).sum()
//    }
//    fn mod_in(&self) -> usize {
//        self.deref()
//            .iter()
//            .rev()
//            .fold((1, 1), |(mod_out, mod_in), item| {
//                update_mod_out_in(item, mod_out, mod_in)
//            })
//            .1
//    }
//    fn mod_out(&self) -> usize {
//        self.deref()
//            .iter()
//            .rev()
//            .fold((1, 1), |(mod_out, mod_in), item| {
//                update_mod_out_in(item, mod_out, mod_in)
//            })
//            .0
//    }
//    fn apply_inplace(
//        &self,
//        index: usize,
//        coordinates: &mut [f64],
//        stride: usize,
//        offset: usize,
//    ) -> usize {
//        self.iter().rev().fold(index, |index, item| {
//            item.apply_inplace(index, coordinates, stride, offset)
//        })
//    }
//    fn apply_index(&self, index: usize) -> usize {
//        self.iter()
//            .rev()
//            .fold(index, |index, item| item.apply_index(index))
//    }
//    fn apply_indices_inplace(&self, indices: &mut [usize]) {
//        for item in self.iter().rev() {
//            item.apply_indices_inplace(indices);
//        }
//    }
//    fn unapply_indices<T: UnapplyIndicesData>(&self, indices: &[T]) -> Vec<T> {
//        self.iter().fold(indices.to_vec(), |indices, item| {
//            item.unapply_indices(&indices)
//        })
//    }
//}
//
//impl<Item, Array> AddOffset for Array
//where
//    Item: AddOffset,
//    Array: Deref<Target = [Item]> + DerefMut + std::fmt::Debug,
//{
//    fn add_offset(&mut self, offset: usize) {
//        for item in self.iter_mut() {
//            item.add_offset(offset);
//        }
//    }
//}
