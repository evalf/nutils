use crate::finite::{BoundedMap, WithBounds, Compose, Composition, Concatenation, ComposeCodomainDomainMismatch};
use crate::infinite::{Edges, Elementary, UnboundedMap, Transpose};
use crate::simplex::Simplex;
use crate::UnapplyIndicesData;
use std::collections::BTreeMap;
use std::iter;

trait RemoveCommonPrefix: Sized {
    fn remove_common_prefix(&self, other: &Self) -> (Self, Self, Self);
    fn remove_common_prefix_opt_lhs(&self, other: &Self) -> (Self, Self, Self);
}

fn split_heads(items: &[Elementary]) -> BTreeMap<Elementary, (Option<Transpose>, Vec<Elementary>)> {
    let mut heads = BTreeMap::new();
    for (i, item) in items.iter().enumerate() {
        if let Some((transpose, head, mut tail)) = item.shift_left(&items[..i]) {
            tail.extend(items[i + 1..].iter().cloned());
            heads.insert(head, (transpose, tail));
        }
        if let Elementary::Edges(Edges(Simplex::Line, offset)) = item {
            let children = Elementary::new_children(Simplex::Line).with_offset(*offset);
            if let Some((transpose, head, mut tail)) = children.shift_left(&items[..i]) {
                tail.push(item.clone());
                tail.push(Elementary::new_take(
                    Simplex::Line.swap_edges_children_map(),
                    Simplex::Line.nedges() * Simplex::Line.nchildren(),
                ));
                tail.extend(items[i + 1..].iter().cloned());
                heads.insert(head, (transpose, tail));
            }
        }
    }
    heads
}

impl RemoveCommonPrefix for Vec<Elementary> {
    /// Remove and return the common prefix of two chains, transforming either if necessary.
    fn remove_common_prefix(&self, other: &Self) -> (Self, Self, Self) {
        let mut common = Vec::new();
        let mut tail1 = self.clone();
        let mut tail2 = other.clone();
        while !tail1.is_empty() && !tail2.is_empty() {
            if tail1.last() == tail2.last() {
                common.push(tail1.pop().unwrap());
                tail2.pop();
                continue;
            }
            let heads1: BTreeMap<Elementary, Vec<Elementary>> = split_heads(&tail1)
                .into_iter()
                .filter_map(|(head, (trans, tail))| trans.is_none().then(|| (head, tail)))
                .collect();
            let mut heads2: BTreeMap<Elementary, Vec<Elementary>> = split_heads(&tail2)
                .into_iter()
                .filter_map(|(head, (trans, tail))| trans.is_none().then(|| (head, tail)))
                .collect();
            if let Some((head, new_tail1, new_tail2)) = heads1
                .into_iter()
                .filter_map(|(h, t1)| heads2.remove(&h).map(|t2| (h, t1, t2)))
                .min_by_key(|(_, t1, t2)| std::cmp::max(t1.len(), t2.len()))
            {
                common.push(head);
                tail1 = new_tail1;
                tail2 = new_tail2;
                continue;
            }
            break;
        }
        let common = if tail1.is_empty() && (!tail2.is_empty() || self.len() <= other.len()) {
            self.clone()
        } else if tail2.is_empty() {
            other.clone()
        } else {
            common
        };
        (common, tail1, tail2)
    }
    fn remove_common_prefix_opt_lhs(&self, other: &Self) -> (Self, Self, Self) {
        let (common, mut tail1, mut tail2) = self.remove_common_prefix(other);
        // Move transposes at the front of `tail1` to `tail2`.
        tail1.reverse();
        tail2.reverse();
        while let Some(mut item) = tail1.pop() {
            if let Some(transpose) = item.as_transpose_mut() {
                transpose.reverse();
                tail2.push(item);
            } else {
                tail1.push(item);
                break;
            }
        }
        tail1.reverse();
        tail2.reverse();
        (common, tail1.into(), tail2.into())
    }
}

//trait RelativeTo<Target: BoundedMap> {
//    fn relative_to(&self, target: &Target) -> Option<Relative>;
//}
//
//impl RelativeTo<Self> for WithBounds<Vec<Elementary>> {
//    fn relative_to(&self, target: &Self) -> Option<Relative> {
//        let (common, rem, rel) = target
//            .get_infinite()
//            .remove_common_prefix_opt_lhs(&self.get_infinite());
//        rem.is_identity()
//            .then(|| Self::new_unchecked(rel, target.dim_in(), target.len_in()).into())
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
//            .map(|rel_items| Concatenation::new(rel_items).into())
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
//impl RelativeTo<Concatenation<Self>> for WithBounds<Vec<Elementary>> {
//    fn relative_to(&self, targets: &Concatenation<Self>) -> Option<Relative> {
//        let mut rels_indices = Vec::new();
//        let mut offset = 0;
//        for target in targets.iter() {
//            let (common, rem, rel) = target.get_infinite().remove_common_prefix_opt_lhs(&self.get_infinite());
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
//        // Split off common tail.
//        let mut common_len_out = self.len_in();
//        let mut common_dim_out = self.dim_in();
//        let mut common = Vec::new();
//        {
//            let mut rels: Vec<_> = rels_indices.iter_mut().map(|(rel, _, _)| rel).collect();
//            while let Some(item) = pop_common(&mut rels[..]) {
//                common_len_out = common_len_out / item.mod_in() * item.mod_out();
//                common_dim_out += item.delta_dim();
//                common.push(item);
//            }
//        }
//        // Build index map.
//        let mut index_map = iter::repeat(Some((0, 0))).take(self.len_in()).collect();
//        let mut rels = Vec::new();
//        for (irel, (rel, offset, out_indices)) in rels_indices.into_iter().enumerate() {
//            let rel_indices: Vec<_> = (offset..offset+out_indices.len()).zip(out_indices).collect();
//            for IndexOutIn(iout, iin) in rel.unapply_indices(&rel_indices) {
//                assert!(index_map[iin].is_none(), "target contains duplicate entries");
//                index_map[iin] = Some((iout, iin + irel * self.len_in()));
//            }
//            rels.push(rel);
//        }
//        let index_map = if let Some(index_map) = index_map.into_iter().collect() {
//            index_map
//        } else {
//            return None;
//        }
//        Some(WithBounds {
//            map: Relative::Concatenation { rels, index_map },
//            delta_dim: targets.dim_in() - self.dim_in(),
//            dim_in: self.dim_in(),
//            len_out: targets.len_out(),
//            len_in: self.len_in(),
//        })
//    }
//}
//
//#[derive(Debug, Clone, PartialEq)]
//pub enum Relative {
//    Identity { dim: usize, len: usize },
//    Single(WithBounds<Vec<Elementary>>),
//    Multiple {
//        rels: Vec<Vec<Elementary>>,
//        index_map: Vec<(usize, usize)>,
//        len_out: usize,
//    }
//    Concatenation(Concatenation<Relative>),
//}
//
//impl BoundedMap for Relative {
//    fn len_out(&self) -> usize {
//        match self {
//            Self::Identity { len, .. } => len,
//            Self::Single(map) => map.len_out(),
//            Self::Multiple { len_out, .. } => len_out,
//            Self::cConcn
//    fn len_in(&self) -> usize;
//    fn dim_out(&self) -> usize {
//        self.dim_in() + self.delta_dim()
//    }
//    fn dim_in(&self) -> usize;
//    fn delta_dim(&self) -> usize;
//    fn add_offset(&mut self, offset: usize);
//    fn apply_inplace_unchecked(&self, index: usize, coordinates: &mut [f64]) -> usize;
//    fn apply_inplace(&self, index: usize, coordinates: &mut [f64]) -> Option<usize> {
//        if index < self.len_in() {
//            Some(self.apply_inplace_unchecked(index, coordinates))
//        } else {
//            None
//        }
//    }
//    fn apply_index_unchecked(&self, index: usize) -> usize;
//    fn apply_index(&self, index: usize) -> Option<usize> {
//        if index < self.len_in() {
//            Some(self.apply_index_unchecked(index))
//        } else {
//            None
//        }
//    }
//    fn apply_indices_inplace_unchecked(&self, indices: &mut [usize]) {
//        for index in indices.iter_mut() {
//            *index = self.apply_index_unchecked(*index);
//        }
//    }
//    fn apply_indices(&self, indices: &[usize]) -> Option<Vec<usize>> {
//        if indices.iter().all(|index| *index < self.len_in()) {
//            let mut indices = indices.to_vec();
//            self.apply_indices_inplace_unchecked(&mut indices);
//            Some(indices)
//        } else {
//            None
//        }
//    }
//    fn unapply_indices_unchecked<T: UnapplyIndicesData>(&self, indices: &[T]) -> Vec<T>;
//    fn unapply_indices<T: UnapplyIndicesData>(&self, indices: &[T]) -> Option<Vec<T>> {
//        if indices.iter().all(|index| index.last() < self.len_out()) {
//            Some(self.unapply_indices_unchecked(indices))
//        } else {
//            None
//        }
//    }
//    fn is_identity(&self) -> bool;
//    
//
//impl Relative {
//    pub fn compose(self, other: Self) -> Result<Self, ComposeCodomainDomainMismatch> {
//        Ok(Composition::new(self, other)?.into())
//    }
//    pub fn new_concatenation(items: Vec<Relative>) -> Self {
//        // TODO: unravel nested concatenations?
//        Self::Concatenation(Concatenation::new(items))
//    }
//}
//
//macro_rules! dispatch {
//    (
//        $vis:vis fn $fn:ident$(<$genarg:ident: $genpath:path>)?(
//            &$self:ident $(, $arg:ident: $ty:ty)*
//        ) $($ret:tt)*
//    ) => {
//        #[inline]
//        $vis fn $fn$(<$genarg: $genpath>)?(&$self $(, $arg: $ty)*) $($ret)* {
//            dispatch!(@match $self; $fn; $($arg),*)
//        }
//    };
//    ($vis:vis fn $fn:ident(&mut $self:ident $(, $arg:ident: $ty:ty)*) $($ret:tt)*) => {
//        #[inline]
//        $vis fn $fn(&mut $self $(, $arg: $ty)*) $($ret)* {
//            dispatch!(@match $self; $fn; $($arg),*)
//        }
//    };
//    (@match $self:ident; $fn:ident; $($arg:ident),*) => {
//        match $self {
//            Relative::Elementary(var) => var.$fn($($arg),*),
//            Relative::Composition(var) => var.$fn($($arg),*),
//            Relative::Concatenation(var) => var.$fn($($arg),*),
//        }
//    }
//}
//
//impl BoundedMap for Relative {
//    dispatch!{fn len_out(&self) -> usize}
//    dispatch!{fn len_in(&self) -> usize}
//    dispatch!{fn dim_out(&self) -> usize}
//    dispatch!{fn dim_in(&self) -> usize}
//    dispatch!{fn delta_dim(&self) -> usize}
//    dispatch!{fn add_offset(&mut self, offset: usize)}
//    dispatch!{fn apply_inplace_unchecked(&self, index: usize, coordinates: &mut [f64]) -> usize}
//    dispatch!{fn apply_inplace(&self, index: usize, coordinates: &mut [f64]) -> Option<usize>}
//    dispatch!{fn apply_index_unchecked(&self, index: usize) -> usize}
//    dispatch!{fn apply_index(&self, index: usize) -> Option<usize>}
//    dispatch!{fn apply_indices_inplace_unchecked(&self, indices: &mut [usize])}
//    dispatch!{fn apply_indices(&self, indices: &[usize]) -> Option<Vec<usize>>}
//    dispatch!{fn unapply_indices_unchecked<T: UnapplyIndicesData>(&self, indices: &[T]) -> Vec<T>}
//    dispatch!{fn unapply_indices<T: UnapplyIndicesData>(&self, indices: &[T]) -> Option<Vec<T>>}
//    dispatch!{fn is_identity(&self) -> bool}
//}
//
//impl From<WithBounds<Vec<Elementary>>> for Relative {
//    fn from(map: WithBounds<Vec<Elementary>>) -> Self {
//        Relative::Elementary(map)
//    }
//}
//
//impl From<Composition<Relative, Relative>> for Relative {
//    fn from(map: Composition<Relative, Relative>) -> Self {
//        Relative::Composition(Box::new(map))
//    }
//}
//
//impl From<Concatenation<Relative>> for Relative {
//    fn from(map: Concatenation<Relative>) -> Self {
//        Relative::Concatenation(map)
//    }
//}
//
//#[cfg(test)]
//mod tests {
//    use super::*;
//    use crate::infinite::*;
//    use crate::simplex::Simplex::*;
//    use approx::assert_abs_diff_eq;
//    use std::iter;
//
//    #[test]
//    fn remove_common_prefix() {
//        let c1 = Elementary::new_children(Line);
//        let e1 = Elementary::new_edges(Line);
//        let swap_ec1 = Elementary::new_take([2, 1], 4);
//        let a = vec![c1.clone(), c1.clone()];
//        let b = vec![e1.clone()];
//        assert_eq!(
//            a.remove_common_prefix(&b),
//            (
//                vec![c1.clone(), c1.clone()],
//                vec![],
//                vec![e1.clone(), swap_ec1.clone(), swap_ec1.clone()],
//            )
//        );
//    }
//
//    macro_rules! elem_comp {
//        (dim=$dim_out:literal, len=$len_out:literal) => {
//            WithBounds::<Vec<Elementary>>::new(Vec::new(), $dim_out, $len_out).unwrap()
//        };
//        (dim=$dim_out:literal, len=$len_out:literal <- $($item:expr),*) => {
//            WithBounds::<Vec<Elementary>>::new(vec![$(Elementary::from($item)),*], $dim_out, $len_out).unwrap()
//        };
//    }
//
//    macro_rules! assert_equiv_chains {
//        ($a:expr, $b:expr $(, $simplex:ident)*) => {{
//            let a = $a;
//            let b = $b;
//            println!("a: {a:?}");
//            println!("b: {b:?}");
//            // Build coords: the outer product of the vertices of the given simplices, zero-padded
//            // to the dimension of the root.
//            let coords = iter::once([]);
//            let simplex_dim = 0;
//            $(
//                let coords = coords.flat_map(|coord| {
//                    $simplex
//                        .vertices()
//                        .chunks($simplex.dim())
//                        .map(move |vert| [&coord, vert].concat())
//                    });
//                let simplex_dim = simplex_dim + $simplex.dim();
//            )*
//            assert_eq!(simplex_dim, a.dim_in(), "the given simplices don't add up to the input dimension");
//            let pad: Vec<f64> = iter::repeat(0.0).take(a.delta_dim()).collect();
//            let coords: Vec<f64> = coords.flat_map(|coord| [&coord[..], &pad].concat()).collect();
//            // Test if every input maps to the same output for both `a` and `b`.
//            for i in 0..2 * a.len_in() {
//                let mut crds_a = coords.clone();
//                let mut crds_b = coords.clone();
//                let ja = a.apply_inplace(i, &mut crds_a);
//                let jb = b.apply_inplace(i, &mut crds_b);
//                assert_eq!(ja, jb, "i={i}");
//                assert_abs_diff_eq!(crds_a[..], crds_b[..]);
//            }
//        }};
//    }
//
//    #[test]
//    fn rel_to() {
//        let a1 = elem_comp!(dim=1, len=2 <- Children::new(Line), Take::new([0, 2], 4));
//        let a2 = elem_comp!(dim=1, len=2 <- Children::new(Line), Take::new([1, 3], 4), Children::new(Line));
//        let a = Concatenation::new(vec![a1, a2]);
//        let b = elem_comp!(dim=1, len=2 <- Children::new(Line), Children::new(Line), Children::new(Line));
//        assert_equiv_chains!(
//            Composition::new(b.relative_to(&a).unwrap(), a.clone()).unwrap(),
//            b,
//            Line);
//        panic!{}
//    }
//}
