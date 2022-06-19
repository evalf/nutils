use crate::chain::UnsizedChain;
use crate::operator::Operator;
use crate::{Mapping, UnsizedMapping};
use std::iter;
use std::ops::Mul;

#[derive(Debug, Clone)]
pub struct Chain {
    dim_out: usize,
    dim_in: usize,
    len_out: usize,
    len_in: usize,
    chain: UnsizedChain,
}

impl Chain {
    pub fn new(dim_out: usize, len_out: usize, chain: UnsizedChain) -> Self {
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
            chain: UnsizedChain::identity(),
        }
    }
    pub fn iter(&self) -> impl Iterator<Item = &Operator> + DoubleEndedIterator {
        self.chain.iter()
    }
    pub fn push(&mut self, operator: Operator) {
        assert!(operator.dim_out() <= self.dim_in);
        assert_eq!(self.len_in % operator.mod_out(), 0);
        self.dim_in -= operator.delta_dim();
        self.len_in = self.len_in / operator.mod_out() * operator.mod_in();
        self.chain.push(operator);
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
    pub fn join(&self, other: &Self) -> Option<Self> {
        // TODO: return Result
        if self.dim_in() == other.dim_out() && self.len_in() == other.len_out() {
            let ops = self.iter().chain(other.iter()).cloned().collect();
            Some(Chain::new(self.dim_out, self.len_out, ops))
        } else {
            None
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

impl Mul for Chain {
    type Output = Self;

    fn mul(self, mut other: Self) -> Self {
        let dim_out = self.dim_out() + other.dim_out();
        let len_out = self.len_out() * other.len_out();
        let trans1 = Operator::new_transpose(other.len_out(), self.len_out());
        let trans2 = Operator::new_transpose(self.len_in(), other.len_out());
        other.add_offset(self.dim_in());
        let chain: UnsizedChain = iter::once(trans1)
            .chain(self.chain)
            .chain(iter::once(trans2))
            .chain(other.chain)
            .collect();
        Chain::new(dim_out, len_out, chain)
    }
}

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

impl RelativeTo<Chain> for Chain {
    type Output = Self;

    fn relative_to(&self, target: &Self) -> Option<Self> {
        let (common, rem, rel) = target.chain.remove_common_prefix_opt_lhs(&self.chain);
        rem.is_identity()
            .then(|| Chain::new(target.dim_in(), target.len_in(), rel.into()))
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

impl RelativeTo<Concat<Chain>> for Chain {
    type Output = RelativeToConcatChain;

    fn relative_to(&self, targets: &Concat<Chain>) -> Option<Self::Output> {
        let mut rels_indices = Vec::new();
        let mut offset = 0;
        for (itarget, target) in targets.items.iter().enumerate() {
            let (common, rem, rel) = target.chain.remove_common_prefix_opt_lhs(&self.chain);
            // if rem.is_identity() {
            //    return rel + offset;
            // }
            if rem.dim_out() == 0 {
                let mut rem_indices: Vec<usize> = (0..).take(target.len_in()).collect();
                rem.apply_indices_inplace(&mut rem_indices);
                // TODO: First collect rem_indices for all targets, then remove
                // the common tail from all rels and then unapply the rem
                // indices on the rels.
                rels_indices.push((rel, itarget, rem_indices, offset))
            }
            offset += target.len_in();
        }
        // TODO: Split off common tail.
        // TODO: unapply indices and build map
        //      let rel_indices = rel.unapply_indices_unchecked(&rem_indices);
        //      if !rel_indices.is_empty() {
        //          // update map
        //          //tails.push((rel, offset, rel_indices));
        //      }
        // TODO: return None if we didn't find everything
        unimplemented!{}
        // RelativeToConcatChain { ... }
        // Pair<Reorder<Concat<Offset<Chain>>>, Chain>
    }
}

#[derive(Debug, Clone)]
struct RelativeToConcatChain {
    dim_in: usize,
    delta_dim: usize,
    len_out: usize,
    map: Vec<(usize, usize)>,
    parts: Vec<(UnsizedChain, usize)>,
    // common_tail: UnsizedChain,
}

impl Mapping for RelativeToConcatChain {
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
        self.map.len()
    }
    fn add_offset(&mut self, offset: usize) {
        for (chain, _) in self.parts.iter_mut() {
            chain.add_offset(offset);
        }
        self.dim_in += offset;
    }
    fn apply_inplace_unchecked(&self, index: usize, coordinates: &mut [f64]) -> usize {
        let (ipart, index) = self.map[index];
        let (chain, offset) = &self.parts[ipart];
        offset + chain.apply_inplace(index, coordinates, self.dim_out())
    }
    fn apply_index_unchecked(&self, index: usize) -> usize {
        let (ipart, index) = self.map[index];
        let (chain, offset) = &self.parts[ipart];
        offset + chain.apply_index(index)
    }
    fn unapply_indices_unchecked(&self, indices: &[usize]) -> Vec<usize> {
        unimplemented! {}
    }
    fn is_identity(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operator::Operator;
    use crate::simplex::Simplex::*;
    use approx::assert_abs_diff_eq;

    macro_rules! chain {
        (dim=$dim_out:literal, len=$len_out:literal) => {
            Chain::identity($dim_out, $len_out)
        };
        (dim=$dim_out:literal, len=$len_out:literal <- $($op:expr),*) => {
            Chain::new($dim_out, $len_out, vec![$(Operator::from($op)),*].into())
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
                b.push(Operator::new_take(b_indices, b.len_in()));
                let mut rel = rel.clone();
                rel.push(Operator::new_take(rel_indices, rel.len_in()));
                (rel, b)
            } else {
                (rel, b)
            };

            assert_equiv_chains!(a.join(&rel_compressed).unwrap(), &b_compressed $(, $simplex)*);
        }};
    }

    #[test]
    fn partial_relative_to() {
        use crate::operator::*;
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
}
