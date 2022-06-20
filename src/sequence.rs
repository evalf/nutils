use crate::chain::RemoveCommonPrefix;
use crate::elementary::Elementary;
use crate::{Mapping, UnsizedMapping};
use std::iter;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NewSizeError {
    DimensionTooSmall,
    LengthNotAMultipleOfRepetition,
}

impl std::error::Error for NewSizeError {}

impl std::fmt::Display for NewSizeError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::DimensionTooSmall => write!(f, "The dimension of the sized mapping is smaller than the minimum dimension of the unsized mapping."),
            Self::LengthNotAMultipleOfRepetition => write!(f, "The length of the sized mapping is not a multiple of the repetition length of the unsized mapping."),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Size<M: UnsizedMapping> {
    mapping: M,
    delta_dim: usize,
    dim_in: usize,
    len_out: usize,
    len_in: usize,
}

impl<M: UnsizedMapping> Size<M> {
    pub fn new(mapping: M, dim_out: usize, len_out: usize) -> Result<Self, NewSizeError> {
        if dim_out < mapping.dim_out() {
            Err(NewSizeError::DimensionTooSmall)
        } else if len_out % mapping.mod_out() != 0 {
            Err(NewSizeError::LengthNotAMultipleOfRepetition)
        } else {
            let delta_dim = mapping.delta_dim();
            let dim_in = dim_out - delta_dim;
            let len_in = len_out / mapping.mod_out() * mapping.mod_in();
            Ok(Self {
                mapping,
                delta_dim,
                dim_in,
                len_out,
                len_in,
            })
        }
    }
}

impl<M: UnsizedMapping> Mapping for Size<M> {
    fn len_in(&self) -> usize {
        self.len_in
    }
    fn len_out(&self) -> usize {
        self.len_out
    }
    fn dim_in(&self) -> usize {
        self.dim_in
    }
    fn delta_dim(&self) -> usize {
        self.delta_dim
    }
    fn add_offset(&mut self, offset: usize) {
        self.mapping.add_offset(offset);
        self.dim_in += offset;
    }
    fn apply_inplace_unchecked(&self, index: usize, coordinates: &mut [f64]) -> usize {
        self.mapping
            .apply_inplace(index, coordinates, self.dim_out())
    }
    fn apply_index_unchecked(&self, index: usize) -> usize {
        self.mapping.apply_index(index)
    }
    fn apply_indices_inplace_unchecked(&self, indices: &mut [usize]) {
        self.mapping.apply_indices_inplace(indices)
    }
    fn unapply_indices_unchecked(&self, indices: &[usize]) -> Vec<usize> {
        self.mapping.unapply_indices(indices)
    }
    fn is_identity(&self) -> bool {
        self.mapping.is_identity()
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ComposeCodomainDomainMismatch;

impl std::error::Error for ComposeCodomainDomainMismatch {}

impl std::fmt::Display for ComposeCodomainDomainMismatch {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "The codomain of the first maping doesn't match the domain of the second mapping,"
        )
    }
}

#[derive(Debug, Clone)]
pub struct Composition<M1: Mapping, M2: Mapping>(M1, M2);

impl<M1: Mapping, M2: Mapping> Composition<M1, M2> {
    pub fn new(mapping1: M1, mapping2: M2) -> Result<Self, ComposeCodomainDomainMismatch> {
        if mapping1.len_out() == mapping2.len_in() && mapping1.dim_out() == mapping2.dim_in() {
            Ok(Self(mapping1, mapping2))
        } else {
            Err(ComposeCodomainDomainMismatch)
        }
    }
}

impl<M1: Mapping, M2: Mapping> Mapping for Composition<M1, M2> {
    fn len_in(&self) -> usize {
        self.0.len_in()
    }
    fn len_out(&self) -> usize {
        self.1.len_out()
    }
    fn dim_in(&self) -> usize {
        self.0.dim_in()
    }
    fn delta_dim(&self) -> usize {
        self.0.delta_dim() + self.1.delta_dim()
    }
    fn add_offset(&mut self, offset: usize) {
        self.0.add_offset(offset);
        self.1.add_offset(offset);
    }
    fn apply_inplace_unchecked(&self, index: usize, coordinates: &mut [f64]) -> usize {
        let index = self.0.apply_inplace_unchecked(index, coordinates);
        self.1.apply_inplace_unchecked(index, coordinates)
    }
    fn apply_index_unchecked(&self, index: usize) -> usize {
        let index = self.0.apply_index_unchecked(index);
        self.1.apply_index_unchecked(index)
    }
    fn apply_indices_inplace_unchecked(&self, indices: &mut [usize]) {
        self.0.apply_indices_inplace_unchecked(indices);
        self.1.apply_indices_inplace_unchecked(indices);
    }
    fn unapply_indices_unchecked(&self, indices: &[usize]) -> Vec<usize> {
        let indices = self.1.unapply_indices_unchecked(indices);
        self.0.unapply_indices_unchecked(&indices)
    }
    fn is_identity(&self) -> bool {
        self.0.is_identity() && self.1.is_identity()
    }
}

trait Compose: Mapping + Sized {
    fn compose<Rhs: Mapping>(self, rhs: Rhs) -> Result<Composition<Self, Rhs>, ComposeCodomainDomainMismatch> {
        Composition::new(self, rhs)
    }
}

impl<M: Mapping> Compose for M {}

#[derive(Debug, Clone)]
pub struct Chain {
    dim_out: usize,
    dim_in: usize,
    len_out: usize,
    len_in: usize,
    chain: Vec<Elementary>,
}

impl Chain {
    pub fn new(dim_out: usize, len_out: usize, chain: Vec<Elementary>) -> Self {
        assert!(chain.dim_out() <= dim_out);
        assert_eq!(len_out % chain.mod_out(), 0);
        let len_in = len_out / chain.mod_out() * chain.mod_in();
        let dim_in = dim_out - chain.delta_dim();
        Self {
            dim_out,
            dim_in,
            len_out,
            len_in,
            chain,
        }
    }
    pub fn identity(dim: usize, len: usize) -> Self {
        Self {
            dim_out: dim,
            dim_in: dim,
            len_out: len,
            len_in: len,
            chain: Vec::new(),
        }
    }
    pub fn push(&mut self, item: Elementary) {
        assert!(item.dim_out() <= self.dim_in);
        assert_eq!(self.len_in % item.mod_out(), 0);
        self.dim_in -= item.delta_dim();
        self.len_in = self.len_in / item.mod_out() * item.mod_in();
        self.chain.push(item);
    }
    pub fn partial_relative_to(&self, target: &Self) -> Option<(Self, Option<Vec<usize>>)> {
        let (common, rem, rel) = target.chain.remove_common_prefix_opt_lhs(&self.chain);
        let rel = Chain::new(target.dim_in(), target.len_in(), rel.into());
        if rem.dim_out() != 0 {
            None
        } else if rem.is_identity() {
            Some((rel, None))
        } else {
            let mut rem_indices: Vec<usize> = (0..).take(target.len_in()).collect();
            rem.apply_indices_inplace(&mut rem_indices);
            let rel_indices = rel.unapply_indices_unchecked(&rem_indices);
            Some((rel, Some(rel_indices)))
        }
    }
}

impl Mapping for Chain {
    fn len_in(&self) -> usize {
        self.len_in
    }
    fn len_out(&self) -> usize {
        self.len_out
    }
    fn dim_in(&self) -> usize {
        self.dim_in
    }
    fn dim_out(&self) -> usize {
        self.dim_out
    }
    fn delta_dim(&self) -> usize {
        self.dim_out - self.dim_in
    }
    fn add_offset(&mut self, offset: usize) {
        self.chain.add_offset(offset);
        self.dim_out += offset;
        self.dim_in += offset;
    }
    fn apply_inplace_unchecked(&self, index: usize, coordinates: &mut [f64]) -> usize {
        self.chain.apply_inplace(index, coordinates, self.dim_out)
    }
    fn apply_index_unchecked(&self, index: usize) -> usize {
        self.chain.apply_index(index)
    }
    fn apply_indices_inplace_unchecked(&self, indices: &mut [usize]) {
        self.chain.apply_indices_inplace(indices)
    }
    fn unapply_indices_unchecked(&self, indices: &[usize]) -> Vec<usize> {
        self.chain.unapply_indices(indices)
    }
    fn is_identity(&self) -> bool {
        self.chain.is_identity()
    }
}

//impl Mul for Chain {
//    type Output = Self;
//
//    fn mul(self, mut other: Self) -> Self {
//        let dim_out = self.dim_out() + other.dim_out();
//        let len_out = self.len_out() * other.len_out();
//        let trans1 = Elementary::new_transpose(other.len_out(), self.len_out());
//        let trans2 = Elementary::new_transpose(self.len_in(), other.len_out());
//        other.add_offset(self.dim_in());
//        let chain: Vec<Elementary> = iter::once(trans1)
//            .chain(self.chain)
//            .chain(iter::once(trans2))
//            .chain(other.chain)
//            .collect();
//        Chain::new(dim_out, len_out, chain)
//    }
//}

#[derive(Debug, Clone)]
struct Concat<Item: Mapping> {
    dim_in: usize,
    delta_dim: usize,
    len_out: usize,
    len_in: usize,
    items: Vec<Item>,
}

impl<Item: Mapping> Concat<Item> {
    pub fn new(items: Vec<Item>) -> Self {
        // TODO: Return `Result<Self, ...>`.
        let first = items.first().unwrap();
        let dim_in = first.dim_in();
        let delta_dim = first.delta_dim();
        let len_out = first.len_out();
        let mut len_in = 0;
        for item in items.iter() {
            assert_eq!(item.dim_in(), dim_in);
            assert_eq!(item.delta_dim(), delta_dim);
            assert_eq!(item.len_out(), len_out);
            len_in += item.len_in();
        }
        Self {
            dim_in,
            delta_dim,
            len_out,
            len_in,
            items,
        }
    }
    //  pub fn relative_to(&self, target: &Self) -> Option<RelativeParts> {
    //      let mut parts = Vec::new();
    //      let mut map = iter::repeat((0, 0)).take(self.len_in()).collect();
    //      for item in self.items.iter() {
    //          let mut seen: Vec<bool> = iter::repeat(false).take(self.len_in()).collect();
    //          let mut offset = 0;
    //          for titem in target.items.iter() {
    //              if let Some(rel, indices) = item.relative_to(titem) {
    //                  unimplemented!{}
    //              }
    //              offset += titem.len_in();
    //          }
    //          if !seen.all(|c| c) {
    //              return None;
    //          }
    //      }
    //      RelativeParts {
    //          dim_in: self.dim_in(),
    //          delta_dim: target.dim_in() - self.dim_in(),
    //          len_out: target.len_in(),
    //          map,
    //          parts,
    //      }
    //  }
    fn resolve_item_unchecked(&self, mut index: usize) -> (&Item, usize) {
        for item in self.items.iter() {
            if index < item.len_in() {
                return (item, index);
            }
            index -= item.len_in();
        }
        panic!("index out of range");
    }
}

impl<Item: Mapping> Mapping for Concat<Item> {
    fn dim_in(&self) -> usize {
        self.dim_in
    }
    fn delta_dim(&self) -> usize {
        self.delta_dim
    }
    fn len_out(&self) -> usize {
        self.len_out
    }
    fn len_in(&self) -> usize {
        self.len_in
    }
    fn add_offset(&mut self, offset: usize) {
        for item in self.items.iter_mut() {
            item.add_offset(offset);
        }
        self.dim_in += offset;
    }
    fn apply_inplace_unchecked(&self, index: usize, coordinates: &mut [f64]) -> usize {
        let (item, index) = self.resolve_item_unchecked(index);
        item.apply_inplace_unchecked(index, coordinates)
    }
    fn apply_index_unchecked(&self, index: usize) -> usize {
        let (item, index) = self.resolve_item_unchecked(index);
        item.apply_index_unchecked(index)
    }
    fn unapply_indices_unchecked(&self, indices: &[usize]) -> Vec<usize> {
        unimplemented! {}
    }
    fn is_identity(&self) -> bool {
        false
    }
}

trait RelativeTo<Target: Mapping> {
    type Output: Mapping;

    fn relative_to(&self, target: &Target) -> Option<Self::Output>;
}

impl RelativeTo<Self> for Chain {
    type Output = Self;

    fn relative_to(&self, target: &Self) -> Option<Self> {
        let (common, rem, rel) = target.chain.remove_common_prefix_opt_lhs(&self.chain);
        rem.is_identity()
            .then(|| Chain::new(target.dim_in(), target.len_in(), rel.into()))
    }
}

type ElemComp = Size<Vec<Elementary>>;

impl RelativeTo<Self> for ElemComp {
    type Output = Self;

    fn relative_to(&self, target: &Self) -> Option<Self> {
        let (common, rem, rel) = target.mapping.remove_common_prefix_opt_lhs(&self.mapping);
        rem.is_identity()
            .then(|| Self::new(rel, target.dim_in(), target.len_in()).unwrap())
        // TODO: Self::new_unchecked
    }
}

impl<Item, Target> RelativeTo<Target> for Concat<Item>
where
    Item: Mapping + RelativeTo<Target>,
    Target: Mapping,
{
    type Output = Concat<Item::Output>;

    fn relative_to(&self, target: &Target) -> Option<Self::Output> {
        self.items
            .iter()
            .map(|item| item.relative_to(target))
            .collect::<Option<_>>()
            .map(|rel_items| Concat::new(rel_items))
    }
}

//impl RelativeTo<Concat<Chain>> for Chain {
//    type Output = RelativeToConcatChain;
//
//    fn relative_to(&self, targets: &Concat<Chain>) -> Option<Self::Output> {
//        let mut rels_indices = Vec::new();
//        let mut offset = 0;
//        for (itarget, target) in targets.items.iter().enumerate() {
//            let (common, rem, rel) = target.chain.remove_common_prefix_opt_lhs(&self.chain);
//            // if rem.is_identity() {
//            //    return rel + offset;
//            // }
//            if rem.dim_out() == 0 {
//                let mut rem_indices: Vec<usize> = (0..).take(target.len_in()).collect();
//                rem.apply_indices_inplace(&mut rem_indices);
//                // TODO: First collect rem_indices for all targets, then remove
//                // the common tail from all rels and then unapply the rem
//                // indices on the rels.
//                rels_indices.push((rel, itarget, rem_indices, offset))
//            }
//            offset += target.len_in();
//        }
//        // TODO: Split off common tail.
//        // TODO: unapply indices and build map
//        //      let rel_indices = rel.unapply_indices_unchecked(&rem_indices);
//        //      if !rel_indices.is_empty() {
//        //          // update map
//        //          //tails.push((rel, offset, rel_indices));
//        //      }
//        // TODO: return None if we didn't find everything
//        unimplemented! {}
//        // RelativeToConcatChain { ... }
//        // Pair<Reorder<Concat<Offset<Chain>>>, Chain>
//    }
//}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simplex::Simplex::*;
    use approx::assert_abs_diff_eq;

    macro_rules! chain {
        (dim=$dim_out:literal, len=$len_out:literal) => {
            Chain::identity($dim_out, $len_out)
        };
        (dim=$dim_out:literal, len=$len_out:literal <- $($item:expr),*) => {
            Chain::new($dim_out, $len_out, vec![$(Elementary::from($item)),*])
        };
    }

    macro_rules! assert_equiv_chains {
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
            assert_eq!(simplex_dim, a.dim_in(), "given simplices don't add up to the input dimension");
            let pad: Vec<f64> = iter::repeat(0.0).take(a.delta_dim()).collect();
            let coords: Vec<f64> = coords.flat_map(|coord| [&coord[..], &pad].concat()).collect();
            // Test if every input maps to the same output for both `a` and `b`.
            for i in 0..2 * a.len_in() {
                let mut crds_a = coords.clone();
                let mut crds_b = coords.clone();
                let ja = a.apply_inplace(i, &mut crds_a);
                let jb = b.apply_inplace(i, &mut crds_b);
                assert_eq!(ja, jb, "i={i}");
                assert_abs_diff_eq!(crds_a[..], crds_b[..]);
            }
        }};
    }

    macro_rules! assert_partial_relative_to {
        ($a:expr, $b:expr $(, $simplex:ident)*) => {{
            let a = $a;
            let b = $b;
            let (rel, indices) = b.partial_relative_to(&a).unwrap();

            let (rel_compressed, b_compressed) = if let Some(indices) = indices {
                let b_indices: Vec<_> = (0..).take(b.len_in()).filter_map(|i| indices.contains(&i).then(|| i)).collect();
                let mut tmp: Vec<_> = (0..rel.len_in()).collect();
                tmp.sort_by_key(|&i| &indices[i]);
                let mut rel_indices: Vec<_> = (0..rel.len_in()).collect();
                rel_indices.sort_by_key(|&i| &tmp[i]);
                let mut b = b.clone();
                b.push(Elementary::new_take(b_indices, b.len_in()));
                let mut rel = rel.clone();
                rel.push(Elementary::new_take(rel_indices, rel.len_in()));
                (rel, b)
            } else {
                (rel, b)
            };

            assert_equiv_chains!(rel_compressed.compose(a).unwrap(), &b_compressed $(, $simplex)*);
        }};
    }

    #[test]
    fn partial_relative_to() {
        use crate::elementary::*;
        assert_partial_relative_to!(
            chain!(dim=1, len=2 <- Children::new(Line), Take::new([0,3,1], 4)),
            chain!(dim=1, len=2 <- Children::new(Line), Children::new(Line)),
            Line
        );
        assert_partial_relative_to!(
            chain!(dim=0, len=4 <- Take::new([0,3,1], 4)),
            chain!(dim = 0, len = 4)
        );
        assert_partial_relative_to!(
            chain!(dim=1, len=2 <- Children::new(Line)),
            chain!(dim=1, len=2 <- Edges::new(Line))
        );
    }

    #[test]
    fn rel_to() {
        let a = Size::new(vec![Elementary::new_children(Line)], 1, 2).unwrap();
        let b = Size::new(vec![Elementary::new_children(Line), Elementary::new_children(Line)], 1, 2).unwrap();
        assert_eq!(b.relative_to(&a), Some(Size::new(vec![Elementary::new_children(Line)], 1, 4).unwrap()));
    }
}
