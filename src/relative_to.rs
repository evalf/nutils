use crate::finite::Mapping;
use crate::infinite::{InfiniteMapping, Edges, Elementary, Transpose};
use crate::simplex::Simplex;
use std::collections::BTreeMap;
use crate::finite::{Bounded, Concatenation, Composition, Compose};

trait RemoveCommonPrefix: Sized {
    fn remove_common_prefix(&self, other: &Self) -> (Self, Self, Self);
    fn remove_common_prefix_opt_lhs(&self, other: &Self) -> (Self, Self, Self);
}

fn split_heads(
    items: &[Elementary],
) -> BTreeMap<Elementary, (Option<Transpose>, Vec<Elementary>)> {
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
        let (common, tail1, tail2) = self.remove_common_prefix(other);
        // Move transposes at the front of `tail1` to `tail2`.
        let mut tail1: Vec<_> = tail1.into();
        let mut tail2: Vec<_> = tail2.into();
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

trait RelativeTo<Target: Mapping> {
    type Output: Mapping;

    fn relative_to(&self, target: &Target) -> Option<Self::Output>;
}

type ElemComp = Bounded<Vec<Elementary>>;

impl RelativeTo<Self> for ElemComp {
    type Output = Self;

    fn relative_to(&self, target: &Self) -> Option<Self> {
        let (common, rem, rel) = target.get_unsized().remove_common_prefix_opt_lhs(&self.get_unsized());
        rem.is_identity()
            .then(|| Self::new(rel, target.dim_in(), target.len_in()).unwrap())
        // TODO: Self::new_unchecked
    }
}

impl<Item, Target> RelativeTo<Target> for Concatenation<Item>
where
    Item: Mapping + RelativeTo<Target>,
    Target: Mapping,
{
    type Output = Concatenation<Item::Output>;

    fn relative_to(&self, target: &Target) -> Option<Self::Output> {
        self.iter()
            .map(|item| item.relative_to(target))
            .collect::<Option<_>>()
            .map(|rel_items| Concatenation::new(rel_items))
    }
}

//impl RelativeTo<Concatenation<ElemComp>> for ElemComp {
//    type Output = Composition<ElemComp, Concatenation<ElemComp>>;
//
//    fn relative_to(&self, targets: &Concatenation<ElemComp>) -> Option<Self::Output> {
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
//        // RelativeToConcatenateChain { ... }
//        // Composition<ElemComp, Concatenate<ElemComp>>
//    }
//}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simplex::Simplex::*;
    use approx::assert_abs_diff_eq;
    use crate::infinite::*;
    use std::iter;

    #[test]
    fn remove_common_prefix() {
        let c1 = Elementary::new_children(Line);
        let e1 = Elementary::new_edges(Line);
        let swap_ec1 = Elementary::new_take([2, 1], 4);
        let a = vec![c1.clone(), c1.clone()];
        let b = vec![e1.clone()];
        assert_eq!(
            a.remove_common_prefix(&b),
            (
                vec![c1.clone(), c1.clone()],
                vec![],
                vec![e1.clone(), swap_ec1.clone(), swap_ec1.clone()],
            )
        );
    }

    macro_rules! elem_comp {
        (dim=$dim_out:literal, len=$len_out:literal) => {
            ElemComp::new(Vec::new(), $dim_out, $len_out).unwrap()
        };
        (dim=$dim_out:literal, len=$len_out:literal <- $($item:expr),*) => {
            ElemComp::new(vec![$(Elementary::from($item)),*], $dim_out, $len_out).unwrap()
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

//    macro_rules! assert_partial_relative_to {
//        ($a:expr, $b:expr $(, $simplex:ident)*) => {{
//            let a = $a;
//            let b = $b;
//            let (rel, indices) = b.partial_relative_to(&a).unwrap();
//
//            let (rel_compressed, b_compressed) = if let Some(indices) = indices {
//                let b_indices: Vec<_> = (0..).take(b.len_in()).filter_map(|i| indices.contains(&i).then(|| i)).collect();
//                let mut tmp: Vec<_> = (0..rel.len_in()).collect();
//                tmp.sort_by_key(|&i| &indices[i]);
//                let mut rel_indices: Vec<_> = (0..rel.len_in()).collect();
//                rel_indices.sort_by_key(|&i| &tmp[i]);
//                let mut b = b.clone();
//                b.push(Elementary::new_take(b_indices, b.len_in()));
//                let mut rel = rel.clone();
//                rel.push(Elementary::new_take(rel_indices, rel.len_in()));
//                (rel, b)
//            } else {
//                (rel, b)
//            };
//
//            assert_equiv_chains!(rel_compressed.compose(a).unwrap(), &b_compressed $(, $simplex)*);
//        }};
//    }
//
//    #[test]
//    fn partial_relative_to() {
//        assert_partial_relative_to!(
//            elem_comp!(dim=1, len=2 <- Children::new(Line), Take::new([0,3,1], 4)),
//            elem_comp!(dim=1, len=2 <- Children::new(Line), Children::new(Line)),
//            Line
//        );
//        assert_partial_relative_to!(
//            elem_comp!(dim=0, len=4 <- Take::new([0,3,1], 4)),
//            elem_comp!(dim = 0, len = 4)
//        );
//        assert_partial_relative_to!(
//            elem_comp!(dim=1, len=2 <- Children::new(Line)),
//            elem_comp!(dim=1, len=2 <- Edges::new(Line))
//        );
//    }

    #[test]
    fn rel_to() {
        let a = elem_comp!(dim=1, len=2 <- Children::new(Line));
        let b = elem_comp!(dim=1, len=2 <- Children::new(Line), Children::new(Line));
        assert_eq!(b.relative_to(&a), Some(elem_comp!(dim=1, len=4 <- Children::new(Line))));
    }
}
