use super::ops::{
    BinaryComposition, BinaryProduct, UniformComposition, UniformConcat, UniformProduct,
};
use super::primitive::{
    AllPrimitiveDecompositions, Identity, Primitive, PrimitiveDecompositionIter, Slice,
    SwapPrimitiveComposition, Transpose, UnboundedMap, WithBounds,
};
use super::{AddOffset, Error, Map, UnapplyIndicesData};
use crate::util::ReplaceNthIter as _;
use std::collections::BTreeMap;
use std::iter;
use std::sync::Arc;

//impl<M> AllPrimitiveDecompositions for UniformProduct<M, Vec<M>>
//where
//    M: Map + AllPrimitiveDecompositions + Clone,
//{
//    fn all_primitive_decompositions<'a>(&'a self) -> PrimitiveDecompositionIter<'a, Self> {
//        Box::new(
//            self.iter()
//                .enumerate()
//                .zip(self.offsets_out())
//                .zip(self.strides_out())
//                .flat_map(move |(((iprod, term), prod_offset), prod_stride)| {
//                    term.all_primitive_decompositions().into_iter().map(
//                        move |((mut prim, mut stride), inner)| {
//                            prim.add_offset(prod_offset);
//                            if prim.mod_out() == 1 {
//                                stride = 1;
//                            } else {
//                                stride *= prod_stride;
//                            }
//                            let product = self.iter().cloned().replace_nth(iprod, inner).collect();
//                            ((prim, stride), product)
//                        },
//                    )
//                }),
//        )
//    }
//}

//impl<M0, M1> AllPrimitiveDecompositions for BinaryProduct<M0, M1>
//where
//    M0: Map + AllPrimitiveDecompositions + Clone,
//    M1: Map + AllPrimitiveDecompositions + Clone,
//{
//    fn all_primitive_decompositions<'a>(&'a self) -> PrimitiveDecompositionIter<'a, Self> {
//        let first = self.first().all_primitive_decompositions().into_iter().map(
//            |((prim, mut stride), first)| {
//                stride *= self.second().len_out();
//                let product = BinaryProduct::new(first, self.second().clone());
//                ((prim, stride), product)
//            },
//        );
//        let second = self
//            .second()
//            .all_primitive_decompositions()
//            .into_iter()
//            .map(|((mut prim, stride), second)| {
//                prim.add_offset(self.first().dim_out());
//                let product = BinaryProduct::new(self.first().clone(), second);
//                ((prim, stride), product)
//            });
//        Box::new(first.chain(second))
//    }
//}

// TODO: UniformComposition?

impl<Outer, Inner> AllPrimitiveDecompositions for BinaryComposition<Outer, Inner>
where
    Outer: Map + AllPrimitiveDecompositions + SwapPrimitiveComposition<Output = Outer>,
    Inner: Map + AllPrimitiveDecompositions + Clone,
{
    fn all_primitive_decompositions<'a>(
        &'a self,
    ) -> Box<dyn Iterator<Item = ((Primitive, usize), Self)> + 'a> {
        let split_outer = self
            .outer()
            .all_primitive_decompositions()
            .into_iter()
            .map(|(prim, outer)| (prim, Self::new(outer, self.inner().clone()).unwrap()));
        let split_inner = self
            .inner()
            .all_primitive_decompositions()
            .into_iter()
            .filter_map(|((prim, stride), inner)| {
                self.outer()
                    .swap_primitive_composition(&prim, stride)
                    .map(|(prim, outer)| (prim, Self::new(outer, inner).unwrap()))
            });
        Box::new(split_outer.chain(split_inner))
    }
}

/// Decompose two maps into a common map and two remainders.
///
/// The decomposition is such that the composition of the common part with the
/// remainder gives a map that is equivalent to the original.
///
/// # Examples
///
/// ```
/// use nutils_test::prim_comp;
/// let map1 = prim_comp![Line*1 <- Children <- Children <- Edges];
/// let map2 = prim_comp![Line*1 <- Children <- Edges];
/// let (common, rem1, rem2) = nutils_test::relative::decompose_common(map1, map2);
/// assert_eq!(common, prim_comp![Line*1 <- Children <- Children <- Edges]);
/// assert_eq!(rem1, prim_comp![Point*8]);
/// assert_eq!(rem2, prim_comp![Point*8 <- Take([2, 1], 4)]);
/// ```
///
/// ```
/// use nutils_test::prim_comp;
/// let map1 = prim_comp![Triangle*1 <- Children];
/// let map2 = prim_comp![Triangle*1 <- Edges <- Children];
/// let (common, rem1, rem2) = nutils_test::relative::decompose_common(map1, map2);
/// assert_eq!(common, prim_comp![Triangle*1 <- Children]);
/// assert_eq!(rem1, prim_comp![Triangle*4]);
/// assert_eq!(rem2, prim_comp![Triangle*4 <- Edges <- Take([3, 6, 1, 7, 2, 5], 12)]);
/// ```
pub fn decompose_common<M1, M2>(mut map1: M1, mut map2: M2) -> (WithBounds<Vec<Primitive>>, M1, M2)
where
    M1: Map + AllPrimitiveDecompositions,
    M2: Map + AllPrimitiveDecompositions,
{
    // TODO: check output dimensions? and return error if dimensions don't match?
    assert_eq!(map1.len_out(), map2.len_out());
    assert_eq!(map1.dim_out(), map2.dim_out());
    let mut common = Vec::new();
    while !map1.is_identity() && !map2.is_identity() {
        let mut outers2: BTreeMap<_, _> = map2.all_primitive_decompositions().collect();
        (map1, map2) = if let Some(((outer, stride), map1, map2)) = map1
            .all_primitive_decompositions()
            .filter_map(|(key, t1)| outers2.remove(&key).map(|t2| (key, t1, t2)))
            .next()
        {
            if stride != 1 && outer.mod_out() != 1 {
                common.push(Primitive::new_transpose(stride, outer.mod_out()));
            }
            common.push(outer);
            (map1, map2)
        } else {
            break;
        };
        assert_eq!(map1.len_out(), map2.len_out());
        assert_eq!(map1.dim_out(), map2.dim_out());
    }
    let common = WithBounds::from_input(common, map1.dim_out(), map1.len_out()).unwrap();
    (common, map1, map2)
}

fn partial_relative_to<Source, Target>(source: Source, target: Target) -> PartialRelative<Source>
where
    Source: Map + AllPrimitiveDecompositions,
    Target: Map + AllPrimitiveDecompositions,
{
    let (_, rem, rel) = decompose_common(target, source);
    if rem.is_identity() {
        PartialRelative::All(rel, None)
    } else if let Some(mut transposes) = rem.as_transposes() {
        transposes.reverse();
        transposes
            .iter_mut()
            .for_each(|transpose| transpose.reverse());
        let transposes = WithBounds::new_unchecked(transposes, rem.dim_in(), rem.len_in());
        PartialRelative::All(rel, Some(transposes))
    } else if rem.is_index_map() {
        let mut indices: Vec<usize> = (0..rem.len_in()).collect();
        rem.apply_indices_inplace_unchecked(&mut indices);
        PartialRelative::Some(rel, indices)
    } else {
        PartialRelative::CannotEstablishRelation
    }
}

enum PartialRelative<M> {
    All(M, Option<WithBounds<Vec<Transpose>>>),
    Some(M, Vec<usize>),
    CannotEstablishRelation,
}

/// An interface for determining the relation between two maps.
pub trait RelativeTo<Target: Map>: Map + Sized {
    type Output: Map;

    /// Return the relative map from `self` to the given target.
    fn relative_to(&self, target: &Target) -> Option<Self::Output>;
    /// Map indices for the target to `self`.
    fn unapply_indices_from<T>(&self, target: &Target, indices: &[T]) -> Option<Vec<T>>
    where
        T: UnapplyIndicesData,
    {
        self.relative_to(target)
            .and_then(|rel| rel.unapply_indices(indices))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Relative<M: Map> {
    Identity(WithBounds<Identity>),
    Slice(WithBounds<Slice>),
    Map(M),
    TransposedMap(BinaryComposition<WithBounds<Vec<Transpose>>, M>),
    Composition(UniformComposition<Self>),
    RelativeToConcat(RelativeToConcat<M>),
}

macro_rules! dispatch {
    (
        $vis:vis fn $fn:ident$(<$genarg:ident: $genpath:path>)?(
            &$self:ident $(, $arg:ident: $ty:ty)*
        ) $(-> $ret:ty)?
    ) => {
        #[inline]
        $vis fn $fn$(<$genarg: $genpath>)?(&$self $(, $arg: $ty)*) $(-> $ret)? {
            dispatch!(@match $self; $fn; $($arg),*)
        }
    };
    ($vis:vis fn $fn:ident(&mut $self:ident $(, $arg:ident: $ty:ty)*) $(-> $ret:ty)?) => {
        #[inline]
        $vis fn $fn(&mut $self $(, $arg: $ty)*) $(-> $ret)? {
            dispatch!(@match $self; $fn; $($arg),*)
        }
    };
    (@match $self:ident; $fn:ident; $($arg:ident),*) => {
        match $self {
            Self::Identity(var) => var.$fn($($arg),*),
            Self::Slice(var) => var.$fn($($arg),*),
            Self::Map(var) => var.$fn($($arg),*),
            Self::TransposedMap(var) => var.$fn($($arg),*),
            Self::Composition(var) => var.$fn($($arg),*),
            Self::RelativeToConcat(var) => var.$fn($($arg),*),
        }
    }
}

impl<M: Map> Map for Relative<M> {
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
}

//impl AddOffset for Relative {
//    dispatch! {fn add_offset(&mut self, offset: usize)}
//}

impl RelativeTo<Self> for WithBounds<Vec<Primitive>> {
    type Output = Relative<Self>;

    fn relative_to(&self, target: &Self) -> Option<Self::Output> {
        let (_, rem, rel) = decompose_common(target.clone(), self.clone());
        if rem.is_identity() {
            Some(Relative::Map(rel))
        } else if let Some(mut transposes) = rem.as_transposes() {
            transposes.reverse();
            transposes
                .iter_mut()
                .for_each(|transpose| transpose.reverse());
            let transposes = WithBounds::new_unchecked(transposes, rem.dim_in(), rem.len_in());
            Some(Relative::TransposedMap(BinaryComposition::new_unchecked(
                transposes, rel,
            )))
        } else {
            None
        }
    }
}

impl<Source, Target> RelativeTo<Target> for UniformConcat<Source>
where
    Source: Map + RelativeTo<Target>,
    Target: Map,
{
    type Output = UniformConcat<Source::Output>;

    fn relative_to(&self, target: &Target) -> Option<Self::Output> {
        self.iter()
            .map(|item| item.relative_to(target))
            .collect::<Option<_>>()
            .map(|rels| {
                UniformConcat::new_unchecked(
                    rels,
                    self.dim_in(),
                    target.dim_in() - self.dim_in(),
                    target.len_in(),
                )
            })
    }
}

fn pop_common<T: std::cmp::PartialEq>(vecs: &mut [&mut Vec<T>]) -> Option<T> {
    let item = vecs.first().and_then(|vec| vec.last());
    if item.is_some() && vecs[1..].iter().all(|vec| vec.last() == item) {
        for vec in vecs[1..].iter_mut() {
            vec.pop();
        }
        vecs[0].pop()
    } else {
        None
    }
}

#[derive(Debug, Clone)]
struct IndexOutIn(usize, usize);

impl UnapplyIndicesData for IndexOutIn {
    #[inline]
    fn get(&self) -> usize {
        self.1
    }
    #[inline]
    fn set(&self, index: usize) -> Self {
        Self(self.0, index)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RelativeToConcat<M: Map> {
    rels: Vec<M>,
    index_map: Arc<Vec<(usize, usize)>>,
    //common: Vec<Primitive>,
    len_out: usize,
    len_in: usize,
    dim_in: usize,
    delta_dim: usize,
}

impl<M: Map> Map for RelativeToConcat<M> {
    fn dim_in(&self) -> usize {
        self.dim_in
    }
    fn delta_dim(&self) -> usize {
        self.delta_dim
    }
    fn len_in(&self) -> usize {
        self.len_in
    }
    fn len_out(&self) -> usize {
        self.len_out
    }
    fn apply_inplace_unchecked(
        &self,
        index: usize,
        coordinates: &mut [f64],
        stride: usize,
        offset: usize,
    ) -> usize {
        //let index = self
        //    .common
        //    .apply_inplace(index, coordinates, stride, offset);
        let (iout, iin) = self.index_map[index];
        let n = self.index_map.len();
        self.rels[iin / n].apply_inplace_unchecked(iin % n, coordinates, stride, offset);
        iout
    }
    fn apply_index_unchecked(&self, index: usize) -> usize {
        //self.index_map[self.common.apply_index(index)].0
        self.index_map[index].0
    }
    fn apply_indices_inplace_unchecked(&self, indices: &mut [usize]) {
        //self.common.apply_indices_inplace(indices);
        for index in indices.iter_mut() {
            *index = self.index_map[*index].0;
        }
    }
    fn unapply_indices_unchecked<T: UnapplyIndicesData>(&self, indices: &[T]) -> Vec<T> {
        // FIXME: VERY EXPENSIVE!!!
        let mut in_indices: Vec<T> = Vec::new();
        for index in indices {
            in_indices.extend(
                self.index_map
                    .iter()
                    .enumerate()
                    .filter_map(|(iin, (iout, _))| (*iout == index.get()).then(|| index.set(iin))),
            );
        }
        //self.common.unapply_indices(&in_indices)
        in_indices
    }
    fn is_identity(&self) -> bool {
        false
    }
    fn is_index_map(&self) -> bool {
        false // TODO
    }
    fn update_basis(&self, index: usize, basis: &mut [f64], dim_out: usize, dim_in: &mut usize, offset: usize) -> usize {
        let (iout, iin) = self.index_map[index];
        let n = self.index_map.len();
        self.rels[iin / n].update_basis(iin % n, basis, dim_out, dim_in, offset);
        iout
    }
}

//impl AddOffset for RelativeToConcat {
//    fn add_offset(&mut self, offset: usize) {
//        self.common.add_offset(offset);
//        for rel in self.rels.iter_mut() {
//            rel.add_offset(offset);
//        }
//        self.dim_in += offset;
//    }
//}

impl<Source, Target> RelativeTo<UniformConcat<Target>> for Source
where
    Source: Map + AllPrimitiveDecompositions + Clone,
    Target: Map + AllPrimitiveDecompositions + Clone,
{
    type Output = Relative<Self>;

    fn relative_to(&self, targets: &UniformConcat<Target>) -> Option<Relative<Self>> {
        let mut rels_indices = Vec::new();
        let mut offset = 0;
        for target in targets.iter().cloned() {
            let new_offset = offset + target.len_in();
            match partial_relative_to(self.clone(), target) {
                PartialRelative::All(rel, transposes) => {
                    let slice = Slice::new(offset, rel.len_out(), targets.len_in());
                    let slice =
                        WithBounds::from_input(slice, rel.dim_out(), rel.len_out()).unwrap();
                    let slice = Relative::Slice(slice);
                    let rel = if let Some(transposes) = transposes {
                        Relative::TransposedMap(BinaryComposition::new_unchecked(transposes, rel))
                    } else {
                        Relative::Map(rel)
                    };
                    return Some(Relative::Composition(UniformComposition::new_unchecked(
                        vec![slice, rel],
                    )));
                }
                PartialRelative::Some(rel, indices) => {
                    rels_indices.push((rel, offset, indices));
                }
                PartialRelative::CannotEstablishRelation => {
                    return None;
                }
            }
            offset = new_offset;
        }
        // Split off common tail. TODO: Only shape increasing items, not take, slice (and transpose?).
        let common_len_out = self.len_in();
        //let common = Vec::new();
        //let mut common_len_out = self.len_in();
        //let mut common = Vec::new();
        //{
        //    let mut rels: Vec<_> = rels_indices.iter_mut().map(|(rel, _, _)| rel).collect();
        //    while let Some(item) = pop_common(&mut rels[..]) {
        //        common_len_out = common_len_out / item.mod_in() * item.mod_out();
        //        common.push(item);
        //    }
        //}
        // Build index map.
        let mut index_map: Vec<Option<(usize, usize)>> =
            iter::repeat(None).take(common_len_out).collect();
        let mut rels = Vec::new();
        for (irel, (rel, offset, out_indices)) in rels_indices.into_iter().enumerate() {
            let rel_indices: Vec<_> = (offset..offset + out_indices.len())
                .zip(out_indices)
                .map(|(i, j)| IndexOutIn(i, j))
                .collect();
            for IndexOutIn(iout, iin) in rel.unapply_indices_unchecked(&rel_indices) {
                assert!(
                    index_map[iin].is_none(),
                    "target contains duplicate entries"
                );
                index_map[iin] = Some((iout, iin + irel * common_len_out));
            }
            rels.push(rel);
        }
        index_map
            .into_iter()
            .collect::<Option<Vec<_>>>()
            .map(|index_map| {
                Relative::RelativeToConcat(RelativeToConcat {
                    index_map: index_map.into(),
                    rels,
                    //common,
                    delta_dim: targets.dim_in() - self.dim_in(),
                    dim_in: self.dim_in(),
                    len_out: targets.len_in(),
                    len_in: self.len_in(),
                })
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prim_comp;
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
        let map1 = prim_comp![Line*2 <- Children <- Children];
        let map2 = prim_comp![Line*2 <- Children <- Take([0, 2], 4)];
        assert_eq!(
            decompose_common(map1, map2),
            (
                prim_comp![Line*2 <- Children],
                prim_comp![Line*4 <- Children],
                prim_comp![Line*4 <- Take([0, 2], 4)],
            )
        );
    }

    #[test]
    fn decompose_common_product() {
        let map1 = prim_comp![Line*2 <- Children <- Children];
        let map2 = prim_comp![Line*2 <- Children <- Take([0, 2], 4)];
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
        //assert_eq!(rel1, BinaryProduct::new(prim_comp![Line*4 <- Children], prim_comp![Line*4 <- Take([0, 2], 4)]));
        //assert_eq!(rel2, BinaryProduct::new(prim_comp![Line*4 <- Take([0, 2], 4)], prim_comp![Line*4 <- Children]));
        //assert_eq!(common, prim_comp![Line*2]);
        // WithBounds { map: [Offset(Children(Line), 1), Transpose(2, 1), Offset(Children(Line), 0)], dim_in: 2, delta_dim: 0, len_in: 16, len_out: 4 }
    }

    //  #[test]
    //  fn rel_to_single() {
    //      let a1 = prim_comp![Line*2 <- Children <- Take([0, 2], 4)];
    //      let a2 = prim_comp![Line*2 <- Children <- Take([1, 3], 4) <- Children];
    //      let a = BinaryConcat::new(a1, a2).unwrap();
    //      let b = prim_comp![Line*2 <- Children <- Children <- Children];
    //      assert_equiv_maps!(
    //          BinaryComposition::new(b.relative_to(&a).unwrap(), a.clone()).unwrap(),
    //          b,
    //          Line
    //      );
    //  }
}
