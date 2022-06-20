use crate::elementary::{Edges, Elementary, Transpose};
use crate::simplex::Simplex;
use crate::UnsizedMapping;
use num::Integer as _;
use std::collections::BTreeMap;
use std::ops::{Deref, DerefMut};

fn dim_out_in<M: UnsizedMapping>(items: &[M]) -> (usize, usize) {
    let mut dim_in = 0;
    let mut dim_out = 0;
    for item in items.iter().rev() {
        if let Some(n) = item.dim_in().checked_sub(dim_out) {
            dim_in += n;
            dim_out += n;
        }
        dim_out += item.delta_dim();
    }
    (dim_out, dim_in)
}

fn mod_out_in<M: UnsizedMapping>(items: &[M]) -> (usize, usize) {
    let mut mod_out = 1;
    let mut mod_in = 1;
    for item in items.iter().rev() {
        let n = mod_out.lcm(&item.mod_in());
        mod_in *= n / mod_out;
        mod_out = n / item.mod_in() * item.mod_out();
    }
    (mod_out, mod_in)
}

impl<M, T> UnsizedMapping for T
where
    M: UnsizedMapping,
    T: Deref<Target = [M]> + DerefMut,
{
    fn dim_in(&self) -> usize {
        dim_out_in(self.deref()).1
    }
    fn delta_dim(&self) -> usize {
        self.iter().map(|item| item.delta_dim()).sum()
    }
    fn add_offset(&mut self, offset: usize) {
        for item in self.iter_mut() {
            item.add_offset(offset);
        }
    }
    fn mod_in(&self) -> usize {
        mod_out_in(self.deref()).1
    }
    fn mod_out(&self) -> usize {
        mod_out_in(self.deref()).0
    }
    fn apply_inplace(&self, index: usize, coordinates: &mut [f64], stride: usize) -> usize {
        self.iter().rev().fold(index, |index, item| {
            item.apply_inplace(index, coordinates, stride)
        })
    }
    fn apply_index(&self, index: usize) -> usize {
        self.iter()
            .rev()
            .fold(index, |index, item| item.apply_index(index))
    }
    fn apply_indices_inplace(&self, indices: &mut [usize]) {
        for item in self.iter().rev() {
            item.apply_indices_inplace(indices);
        }
    }
    fn unapply_indices(&self, indices: &[usize]) -> Vec<usize> {
        self.iter().fold(indices.to_vec(), |indices, item| {
            item.unapply_indices(&indices)
        })
    }
}

pub trait RemoveCommonPrefix: Sized {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simplex::Simplex::*;

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
}
