use crate::elementary::{
    AllElementaryDecompositions, Elementary, SwapElementaryComposition, UnboundedMap, WithBounds,
};
use crate::ops::{BinaryComposition, BinaryProduct, UniformProduct};
use crate::util::ReplaceNthIter as _;
use crate::{AddOffset, Map};
use std::collections::BTreeMap;

impl<M> AllElementaryDecompositions for UniformProduct<M, Vec<M>>
where
    M: Map + AllElementaryDecompositions + Clone,
{
    fn all_elementary_decompositions<'a>(
        &'a self,
    ) -> Box<dyn Iterator<Item = ((Elementary, usize), Self)> + 'a> {
        Box::new(
            self.iter()
                .enumerate()
                .zip(self.offsets_out())
                .zip(self.strides_out())
                .flat_map(move |(((iprod, term), prod_offset), prod_stride)| {
                    term.all_elementary_decompositions().into_iter().map(
                        move |((mut elmtry, mut stride), inner)| {
                            elmtry.add_offset(prod_offset);
                            if elmtry.mod_out() == 1 {
                                stride = 1;
                            } else {
                                stride *= prod_stride;
                            }
                            let product = self.iter().cloned().replace_nth(iprod, inner).collect();
                            ((elmtry, stride), product)
                        },
                    )
                }),
        )
    }
}

impl<M0, M1> AllElementaryDecompositions for BinaryProduct<M0, M1>
where
    M0: Map + AllElementaryDecompositions + Clone,
    M1: Map + AllElementaryDecompositions + Clone,
{
    fn all_elementary_decompositions<'a>(
        &'a self,
    ) -> Box<dyn Iterator<Item = ((Elementary, usize), Self)> + 'a> {
        let first = self
            .first()
            .all_elementary_decompositions()
            .into_iter()
            .map(|((elmtry, mut stride), first)| {
                stride *= self.second().len_out();
                let product = BinaryProduct::new(first, self.second().clone());
                ((elmtry, stride), product)
            });
        let second = self
            .second()
            .all_elementary_decompositions()
            .into_iter()
            .map(|((mut elmtry, stride), second)| {
                elmtry.add_offset(self.first().dim_out());
                let product = BinaryProduct::new(self.first().clone(), second);
                ((elmtry, stride), product)
            });
        Box::new(first.chain(second))
    }
}

// TODO: UniformComposition?

impl<Outer, Inner> AllElementaryDecompositions for BinaryComposition<Outer, Inner>
where
    Outer: Map + AllElementaryDecompositions + SwapElementaryComposition<Output = Outer>,
    Inner: Map + AllElementaryDecompositions + Clone,
{
    fn all_elementary_decompositions<'a>(
        &'a self,
    ) -> Box<dyn Iterator<Item = ((Elementary, usize), Self)> + 'a> {
        let split_outer = self
            .outer()
            .all_elementary_decompositions()
            .into_iter()
            .map(|(elmtry, outer)| (elmtry, Self::new(outer, self.inner().clone()).unwrap()));
        let split_inner = self
            .inner()
            .all_elementary_decompositions()
            .into_iter()
            .filter_map(|((elmtry, stride), inner)| {
                self.outer()
                    .swap_elementary_composition(&elmtry, stride)
                    .map(|(elmtry, outer)| (elmtry, Self::new(outer, inner).unwrap()))
            });
        Box::new(split_outer.chain(split_inner))
    }
}

/// Decompose two maps into two remainders and a common map.
///
/// The decomposition is such that the composition the common part with the
/// remainder gives a map that is equivalent to the original.
pub fn decompose_common<M1, M2>(mut map1: M1, mut map2: M2) -> (WithBounds<Vec<Elementary>>, M1, M2)
where
    M1: Map + AllElementaryDecompositions,
    M2: Map + AllElementaryDecompositions,
{
    // TODO: check output dimensions? and return error if dimensions don't match?
    assert_eq!(map1.len_out(), map2.len_out());
    assert_eq!(map1.dim_out(), map2.dim_out());
    let mut common = Vec::new();
    while !map1.is_identity() && !map2.is_identity() {
        let mut outers2: BTreeMap<_, _> = map2.all_elementary_decompositions().collect();
        (map1, map2) = if let Some(((outer, stride), map1, map2)) = map1
            .all_elementary_decompositions()
            .filter_map(|(key, t1)| outers2.remove(&key).map(|t2| (key, t1, t2)))
            .next()
        {
            let outer_mod_out = outer.mod_out();
            common.push(outer);
            if stride != 1 {
                common.push(Elementary::new_transpose(stride, outer_mod_out));
            }
            (map1, map2)
        } else {
            break;
        }
    }
    common.reverse();
    let common = WithBounds::from_input(common, map1.dim_out(), map1.len_out()).unwrap();
    (common, map1, map2)
}

//#[derive(Debug, Clone, PartialEq)]
//pub enum Relative {
//    Identity(WithBounds<Identity>),
//    Single(WithBounds<Vec<Elementary>>),
//    Multiple(RelativeMultiple),
//    Concatenation(Concatenation<Relative>),
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
//            dispatch!(@match $self; $fn; $($arg),*)
//        }
//    };
//    ($vis:vis fn $fn:ident(&mut $self:ident $(, $arg:ident: $ty:ty)*) $(-> $ret:ty)?) => {
//        #[inline]
//        $vis fn $fn(&mut $self $(, $arg: $ty)*) $(-> $ret)? {
//            dispatch!(@match $self; $fn; $($arg),*)
//        }
//    };
//    (@match $self:ident; $fn:ident; $($arg:ident),*) => {
//        match $self {
//            Relative::Identity(var) => var.$fn($($arg),*),
//            Relative::Single(var) => var.$fn($($arg),*),
//            Relative::Multiple(var) => var.$fn($($arg),*),
//            Relative::Concatenation(var) => var.$fn($($arg),*),
//        }
//    }
//}
//
//impl BoundedMap for Relative {
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
//
//impl AddOffset for Relative {
//    dispatch! {fn add_offset(&mut self, offset: usize)}
//}
//
//pub trait RelativeTo<Target: BoundedMap> {
//    fn relative_to(&self, target: &Target) -> Option<Relative>;
//    fn unapply_indices_from<T: UnapplyIndicesData>(
//        &self,
//        target: &Target,
//        indices: &[T],
//    ) -> Option<Vec<T>> {
//        self.relative_to(target)
//            .and_then(|rel| rel.unapply_indices(indices))
//    }
//}
//
//impl RelativeTo<Self> for WithBounds<Vec<Elementary>> {
//    fn relative_to(&self, target: &Self) -> Option<Relative> {
//        let (_, rem, rel) = target
//            .get_unbounded()
//            .split_common_prefix_opt_lhs(self.get_unbounded());
//        rem.is_identity()
//            .then(|| Relative::Single(Self::new_unchecked(rel, target.dim_in(), target.len_in())))
//    }
//}
//
//impl<Item, Target> RelativeTo<Target> for Concatenation<Item>
//where
//    Item: BoundedMap + RelativeTo<Target>,
//    Target: BoundedMap,
//{
//    fn relative_to(&self, target: &Target) -> Option<Relative> {
//        self.iter()
//            .map(|item| item.relative_to(target))
//            .collect::<Option<_>>()
//            .map(|rel_items| Relative::Concatenation(Concatenation::new(rel_items)))
//    }
//}
//
//fn pop_common<T: std::cmp::PartialEq>(vecs: &mut [&mut Vec<T>]) -> Option<T> {
//    let item = vecs.first().and_then(|vec| vec.last());
//    if item.is_some() && vecs[1..].iter().all(|vec| vec.last() == item) {
//        for vec in vecs[1..].iter_mut() {
//            vec.pop();
//        }
//        vecs[0].pop()
//    } else {
//        None
//    }
//}
//
//#[derive(Debug, Clone)]
//struct IndexOutIn(usize, usize);
//
//impl UnapplyIndicesData for IndexOutIn {
//    #[inline]
//    fn last(&self) -> usize {
//        self.1
//    }
//    #[inline]
//    fn push(&self, index: usize) -> Self {
//        Self(self.0, index)
//    }
//}
//
//#[derive(Debug, Clone, PartialEq)]
//pub struct RelativeMultiple {
//    rels: Vec<Vec<Elementary>>,
//    index_map: Rc<Vec<(usize, usize)>>,
//    common: Vec<Elementary>,
//    len_out: usize,
//    len_in: usize,
//    dim_in: usize,
//    delta_dim: usize,
//}
//
//impl BoundedMap for RelativeMultiple {
//    fn dim_in(&self) -> usize {
//        self.dim_in
//    }
//    fn delta_dim(&self) -> usize {
//        self.delta_dim
//    }
//    fn len_in(&self) -> usize {
//        self.len_in
//    }
//    fn len_out(&self) -> usize {
//        self.len_out
//    }
//    fn apply_inplace_unchecked(
//        &self,
//        index: usize,
//        coordinates: &mut [f64],
//        stride: usize,
//        offset: usize,
//    ) -> usize {
//        let index = self
//            .common
//            .apply_inplace(index, coordinates, stride, offset);
//        let (iout, iin) = self.index_map[index];
//        let n = self.index_map.len();
//        self.rels[iin / n].apply_inplace(iin % n, coordinates, stride, offset);
//        iout
//    }
//    fn apply_index_unchecked(&self, index: usize) -> usize {
//        self.index_map[self.common.apply_index(index)].0
//    }
//    fn apply_indices_inplace_unchecked(&self, indices: &mut [usize]) {
//        self.common.apply_indices_inplace(indices);
//        for index in indices.iter_mut() {
//            *index = self.index_map[*index].0;
//        }
//    }
//    fn unapply_indices_unchecked<T: UnapplyIndicesData>(&self, indices: &[T]) -> Vec<T> {
//        // FIXME: VERY EXPENSIVE!!!
//        let mut in_indices: Vec<T> = Vec::new();
//        for index in indices {
//            in_indices.extend(
//                self.index_map
//                    .iter()
//                    .enumerate()
//                    .filter_map(|(iin, (iout, _))| {
//                        (*iout == index.last()).then(|| index.push(iin))
//                    }),
//            );
//        }
//        self.common.unapply_indices(&in_indices)
//    }
//    fn is_identity(&self) -> bool {
//        false
//    }
//}
//
//impl AddOffset for RelativeMultiple {
//    fn add_offset(&mut self, offset: usize) {
//        self.common.add_offset(offset);
//        for rel in self.rels.iter_mut() {
//            rel.add_offset(offset);
//        }
//        self.dim_in += offset;
//    }
//}
//
//impl RelativeTo<Concatenation<Self>> for WithBounds<Vec<Elementary>> {
//    fn relative_to(&self, targets: &Concatenation<Self>) -> Option<Relative> {
//        let mut rels_indices = Vec::new();
//        let mut offset = 0;
//        for target in targets.iter() {
//            let (_, rem, rel) = target
//                .get_unbounded()
//                .split_common_prefix_opt_lhs(self.get_unbounded());
//            if rem.is_identity() {
//                let slice = Elementary::new_slice(offset, target.len_in(), targets.len_in());
//                let rel: Vec<Elementary> = iter::once(slice).chain(rel).collect();
//                let rel = WithBounds::new_unchecked(rel, targets.dim_in(), targets.len_in());
//                return Some(Relative::Single(rel));
//            }
//            if rem.dim_out() == 0 {
//                let mut indices: Vec<usize> = (0..target.len_in()).collect();
//                rem.apply_indices_inplace(&mut indices);
//                rels_indices.push((rel, offset, indices))
//            }
//            offset += target.len_in();
//        }
//        // Split off common tail. TODO: Only shape increasing items, not take, slice (and transpose?).
//        let common_len_out = self.len_in();
//        let common = Vec::new();
//        //let mut common_len_out = self.len_in();
//        //let mut common = Vec::new();
//        //{
//        //    let mut rels: Vec<_> = rels_indices.iter_mut().map(|(rel, _, _)| rel).collect();
//        //    while let Some(item) = pop_common(&mut rels[..]) {
//        //        common_len_out = common_len_out / item.mod_in() * item.mod_out();
//        //        common.push(item);
//        //    }
//        //}
//        // Build index map.
//        let mut index_map: Vec<Option<(usize, usize)>> =
//            iter::repeat(None).take(common_len_out).collect();
//        let mut rels = Vec::new();
//        for (irel, (rel, offset, out_indices)) in rels_indices.into_iter().enumerate() {
//            let rel_indices: Vec<_> = (offset..offset + out_indices.len())
//                .zip(out_indices)
//                .map(|(i, j)| IndexOutIn(i, j))
//                .collect();
//            for IndexOutIn(iout, iin) in rel.unapply_indices(&rel_indices) {
//                assert!(
//                    index_map[iin].is_none(),
//                    "target contains duplicate entries"
//                );
//                index_map[iin] = Some((iout, iin + irel * common_len_out));
//            }
//            rels.push(rel);
//        }
//        index_map
//            .into_iter()
//            .collect::<Option<Vec<_>>>()
//            .map(|index_map| {
//                Relative::Multiple(RelativeMultiple {
//                    index_map: index_map.into(),
//                    rels,
//                    common,
//                    delta_dim: targets.dim_in() - self.dim_in(),
//                    dim_in: self.dim_in(),
//                    len_out: targets.len_in(),
//                    len_in: self.len_in(),
//                })
//            })
//    }
//}
//
#[cfg(test)]
mod tests {
    use super::*;
    use crate::elementaries;
    use crate::simplex::Simplex::*;
    use approx::assert_abs_diff_eq;
    use std::iter;

    macro_rules! assert_equiv_maps {
        ($a:expr, $b:expr $(, $simplex:ident)*) => {{
            let a = $a;
            let b = $b;
            println!("a: {a:?}");
            println!("b: {b:?}");
            // Build coords: the outer product of the vertices of the given simplices, zero-padded
            // to the dimension of the root.
            let coords = iter::once([]);
            let simplex_dim = 0;
            $(
                let coords = coords.flat_map(|coord| {
                    $simplex
                        .vertices()
                        .chunks($simplex.dim())
                        .map(move |vert| [&coord, vert].concat())
                    });
                let simplex_dim = simplex_dim + $simplex.dim();
            )*
            assert_eq!(simplex_dim, a.dim_in(), "the given simplices don't add up to the input dimension");
            let pad: Vec<f64> = iter::repeat(0.0).take(a.delta_dim()).collect();
            let coords: Vec<f64> = coords.flat_map(|coord| [&coord[..], &pad].concat()).collect();
            // Test if every input maps to the same output for both `a` and `b`.
            for i in 0..2 * a.len_in() {
                let mut crds_a = coords.clone();
                let mut crds_b = coords.clone();
                let ja = a.apply_inplace(i, &mut crds_a, a.dim_out(), 0);
                let jb = b.apply_inplace(i, &mut crds_b, a.dim_out(), 0);
                assert_eq!(ja, jb, "i={i}");
                assert_abs_diff_eq!(crds_a[..], crds_b[..]);
            }
        }};
    }

    #[test]
    fn decompose_common_vec() {
        let map1 = elementaries![Line*2 <- Children <- Children];
        let map2 = elementaries![Line*2 <- Children <- Take([0, 2], 4)];
        assert_eq!(
            decompose_common(map1, map2),
            (
                elementaries![Line*2 <- Children],
                elementaries![Line*4 <- Children],
                elementaries![Line*4 <- Take([0, 2], 4)],
            )
        );
    }

    #[test]
    fn decompose_common_product() {
        let map1 = elementaries![Line*2 <- Children <- Children];
        let map2 = elementaries![Line*2 <- Children <- Take([0, 2], 4)];
        let (common, rel1, rel2) = decompose_common(map1.clone(), map2.clone());
        assert!(!common.is_identity());
        assert_equiv_maps!(
            BinaryComposition::new(common.clone(), rel1.clone()).unwrap(),
            map1.clone(),
            Line
        );
        assert_equiv_maps!(
            BinaryComposition::new(common.clone(), rel2.clone()).unwrap(),
            map2.clone(),
            Line
        );
        //assert_eq!(rel1, BinaryProduct::new(elementaries![Line*4 <- Children], elementaries![Line*4 <- Take([0, 2], 4)]));
        //assert_eq!(rel2, BinaryProduct::new(elementaries![Line*4 <- Take([0, 2], 4)], elementaries![Line*4 <- Children]));
        //assert_eq!(common, elementaries![Line*2]);
        // WithBounds { map: [Offset(Children(Line), 1), Transpose(2, 1), Offset(Children(Line), 0)], dim_in: 2, delta_dim: 0, len_in: 16, len_out: 4 }
    }
    //
    //    #[test]
    //    fn rel_to() {
    //        let a1 = elementaries![Line*2 <- Children <- Take([0, 2], 4)];
    //        let a2 = elementaries![Line*2 <- Children <- Take([1, 3], 4) <- Children];
    //        let a = BinaryConcat::new(a1, a2).unwrap();
    //        let b = elementaries![Line*2 <- Children <- Children <- Children];
    //        assert_equiv_maps!(
    //            BinaryComposition::new(b.relative_to(&a).unwrap(), a.clone()).unwrap(),
    //            b,
    //            Line
    //        );
    //    }
}
