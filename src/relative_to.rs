use crate::finite::BoundedMap;
use crate::finite::{WithBounds, Compose, Composition, Concatenation, ConcreteMap};
use crate::infinite::{Edges, Elementary, UnboundedMap, Transpose};
use crate::simplex::Simplex;
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

trait RelativeTo<Target: BoundedMap> {
    fn relative_to(&self, target: &Target) -> Option<ConcreteMap>;
}

impl RelativeTo<Self> for WithBounds<Vec<Elementary>> {
    fn relative_to(&self, target: &Self) -> Option<ConcreteMap> {
        let (common, rem, rel) = target
            .get_infinite()
            .remove_common_prefix_opt_lhs(&self.get_infinite());
        rem.is_identity()
            .then(|| Self::new_unchecked(rel, target.dim_in(), target.len_in()).into())
    }
}

impl<Item, Target> RelativeTo<Target> for Concatenation<Item>
where
    Item: BoundedMap + RelativeTo<Target>,
    Target: BoundedMap,
{
    fn relative_to(&self, target: &Target) -> Option<ConcreteMap> {
        self.iter()
            .map(|item| item.relative_to(target))
            .collect::<Option<_>>()
            .map(|rel_items| Concatenation::new(rel_items).into())
    }
}

// trait PartialMakeRelative<Target> {
//     fn partial_make_relative(&self, target: Target) -> Option<Vec<(Vec<Elementary>, Vec<usize>)>>;
// }

impl RelativeTo<Concatenation<Self>> for WithBounds<Vec<Elementary>> {
    fn relative_to(&self, targets: &Concatenation<Self>) -> Option<ConcreteMap> {
        let mut rels_indices = Vec::new();
        let mut offset = 0;
        for target in targets.iter() {
            let (common, rem, rel) = target.get_infinite().remove_common_prefix_opt_lhs(&self.get_infinite());
            let slice = Elementary::new_slice(offset, target.len_in(), targets.len_in());
            if rem.is_identity() {
                let rel: Vec<Elementary> = iter::once(slice).chain(rel).collect();
                let rel = WithBounds::new_unchecked(rel, targets.dim_in(), targets.len_in());
                return Some(ConcreteMap::Elementary(rel));
            }
            if rem.dim_out() == 0 {
                let mut indices: Vec<usize> = (0..target.len_in()).collect();
                rem.apply_indices_inplace(&mut indices);
                rels_indices.push((rel, slice, indices))
            }
            offset += target.len_in();
        }
        // TODO: Split off common tail.
        let mut concat_indices = Vec::new();
        let mut rels = Vec::new();
        for (irel, (rel, slice, out_indices)) in rels_indices.into_iter().enumerate() {
            let in_indices = rel.unapply_indices(&out_indices);
            let offset = irel * self.len_in();
            concat_indices.extend(in_indices.into_iter().map(|i| i + offset));
            let rel: Vec<Elementary> = iter::once(slice).chain(rel).collect();
            let rel = WithBounds::new_unchecked(rel, targets.dim_in(), targets.len_in());
            let rel = ConcreteMap::Elementary(rel);
            rels.push(rel);
        }
        if concat_indices.len() != self.len_in() {
            return None;
        }
        let take = Elementary::new_take(concat_indices, self.len_in() * rels.len());
        let take: ConcreteMap = WithBounds::new_unchecked(vec![take], self.dim_in(), self.len_in()).into();
        let concat = ConcreteMap::new_concatenation(rels);
        Some(take.compose(concat).unwrap().into())
    }
}

impl RelativeTo<ConcreteMap> for WithBounds<Vec<Elementary>> {
    fn relative_to(&self, target: &ConcreteMap) -> Option<ConcreteMap> {
        match target {
            ConcreteMap::Elementary(target) => self.relative_to(target),
            ConcreteMap::Concatenation(target) => {
                let c: Option<Vec<WithBounds<Vec<Elementary>>>> = target.iter().map(|item| match item {
                    ConcreteMap::Elementary(item) => Some(item.clone()),
                    _ => None,
                }).collect();
                c.and_then(|c| self.relative_to(&Concatenation::new(c)))
            }
            _ => None
        }
    }
}

impl RelativeTo<Self> for ConcreteMap {
    fn relative_to(&self, target: &Self) -> Option<ConcreteMap> {
        match self {
            ConcreteMap::Elementary(source) => source.relative_to(target),
            ConcreteMap::Concatenation(source) => source.relative_to(target),
            _ => None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::infinite::*;
    use crate::simplex::Simplex::*;
    use approx::assert_abs_diff_eq;
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
            WithBounds::<Vec<Elementary>>::new(Vec::new(), $dim_out, $len_out).unwrap()
        };
        (dim=$dim_out:literal, len=$len_out:literal <- $($item:expr),*) => {
            WithBounds::<Vec<Elementary>>::new(vec![$(Elementary::from($item)),*], $dim_out, $len_out).unwrap()
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
            assert_eq!(simplex_dim, a.dim_in(), "the given simplices don't add up to the input dimension");
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

    #[test]
    fn rel_to() {
        let a1: ConcreteMap = elem_comp!(dim=1, len=2 <- Children::new(Line), Take::new([0, 2], 4)).into();
        let a2: ConcreteMap = elem_comp!(dim=1, len=2 <- Children::new(Line), Take::new([1, 3], 4), Children::new(Line)).into();
        let a = ConcreteMap::new_concatenation(vec![a1, a2].into());
        let b: ConcreteMap = elem_comp!(dim=1, len=2 <- Children::new(Line), Children::new(Line)).into();
        assert_equiv_chains!(
            b.relative_to(&a).unwrap().compose(a.clone()).unwrap(),
            b,
            Line);
    }
}
