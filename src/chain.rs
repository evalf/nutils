use crate::operator::{Operator, Transpose, Edges};
use crate::simplex::Simplex;
use std::collections::BTreeMap;
use crate::UnsizedMapping;
use num::Integer as _;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnsizedChain(Vec<Operator>);

impl UnsizedChain {
    #[inline]
    pub fn new(operators: Vec<Operator>) -> Self {
        Self(operators)
    }
    #[inline]
    pub fn identity() -> Self {
        UnsizedChain(Vec::new())
    }
    fn dim_out_in(&self) -> (usize, usize) {
        let mut dim_in = 0;
        let mut dim_out = 0;
        for op in self.0.iter().rev() {
            if let Some(n) = op.dim_in().checked_sub(dim_out) {
                dim_in += n;
                dim_out += n;
            }
            dim_out += op.delta_dim();
        }
        (dim_out, dim_in)
    }
    fn mod_out_in(&self) -> (usize, usize) {
        let mut mod_out = 1;
        let mut mod_in = 1;
        for op in self.0.iter().rev() {
            let n = mod_out.lcm(&op.mod_in());
            mod_in *= n / mod_out;
            mod_out = n / op.mod_in() * op.mod_out();
        }
        (mod_out, mod_in)
    }
    pub fn iter(&self) -> impl Iterator<Item = &Operator> + DoubleEndedIterator {
        self.0.iter()
    }
    pub fn push(&mut self, operator: impl Into<Operator>) {
        self.0.push(operator.into())
    }
    fn split_heads(&self) -> BTreeMap<Operator, (Option<Transpose>, Vec<Operator>)> {
        let mut heads = BTreeMap::new();
        for (i, op) in self.0.iter().enumerate() {
            if let Some((transpose, head, mut tail)) = op.shift_left(&self.0[..i]) {
                tail.extend(self.0[i + 1..].iter().cloned());
                heads.insert(head, (transpose, tail));
            }
            if let Operator::Edges(Edges(Simplex::Line, offset)) = op {
                let children = Operator::new_children(Simplex::Line).with_offset(*offset);
                if let Some((transpose, head, mut tail)) = children.shift_left(&self.0[..i]) {
                    tail.push(op.clone());
                    tail.push(Operator::new_take(
                        Simplex::Line.swap_edges_children_map(),
                        Simplex::Line.nedges() * Simplex::Line.nchildren(),
                    ));
                    tail.extend(self.0[i + 1..].iter().cloned());
                    heads.insert(head, (transpose, tail));
                }
            }
        }
        heads
    }
    /// Remove and return the common prefix of two chains, transforming either if necessary.
    pub fn remove_common_prefix(&self, other: &Self) -> (Self, Self, Self) {
        let mut common = Vec::new();
        let mut tail1 = self.clone();
        let mut tail2 = other.clone();
        while !tail1.0.is_empty() && !tail2.0.is_empty() {
            if tail1.0.last() == tail2.0.last() {
                common.push(tail1.0.pop().unwrap());
                tail2.0.pop();
                continue;
            }
            let heads1: BTreeMap<Operator, Vec<Operator>> = tail1
                .split_heads()
                .into_iter()
                .filter_map(|(head, (trans, tail))| {
                    if trans.is_none() {
                        Some((head, tail))
                    } else {
                        None
                    }
                })
                .collect();
            let mut heads2: BTreeMap<Operator, Vec<Operator>> = tail2
                .split_heads()
                .into_iter()
                .filter_map(|(head, (trans, tail))| {
                    if trans.is_none() {
                        Some((head, tail))
                    } else {
                        None
                    }
                })
                .collect();
            if let Some((head, new_tail1, new_tail2)) = heads1
                .into_iter()
                .filter_map(|(h, t1)| heads2.remove(&h).map(|t2| (h, t1, t2)))
                .min_by_key(|(_, t1, t2)| std::cmp::max(t1.len(), t2.len()))
            {
                common.push(head);
                tail1.0 = new_tail1;
                tail2.0 = new_tail2;
                continue;
            }
            break;
        }
        let common = if tail1.0.is_empty() && (!tail2.0.is_empty() || self.0.len() <= other.0.len()) {
            self.clone()
        } else if tail2.0.is_empty() {
            other.clone()
        } else {
            Self::new(common)
        };
        (common, tail1, tail2)
    }
    pub fn remove_common_prefix_opt_lhs(&self, other: &Self) -> (Self, Self, Self) {
        let (common, tail1, tail2) = self.remove_common_prefix(other);
        // Move transposes at the front of `tail1` to `tail2`.
        let mut tail1: Vec<_> = tail1.into();
        let mut tail2: Vec<_> = tail2.into();
        tail1.reverse();
        tail2.reverse();
        while let Some(mut op) = tail1.pop() {
            if let Some(transpose) = op.as_transpose_mut() {
                transpose.reverse();
                tail2.push(op);
            } else {
                tail1.push(op);
                break;
            }
        }
        tail1.reverse();
        tail2.reverse();
        (common, tail1.into(), tail2.into())
    }
}

impl UnsizedMapping for UnsizedChain {
    fn dim_in(&self) -> usize {
        self.dim_out_in().1
    }
    fn delta_dim(&self) -> usize {
        self.0.iter().map(|op| op.delta_dim()).sum()
    }
    fn add_offset(&mut self, offset: usize) {
        for op in self.0.iter_mut() {
            op.add_offset(offset);
        }
    }
    fn mod_in(&self) -> usize {
        self.mod_out_in().1
    }
    fn mod_out(&self) -> usize {
        self.mod_out_in().0
    }
    fn apply_inplace(&self, index: usize, coordinates: &mut [f64], stride: usize) -> usize {
        self.0
            .iter()
            .rev()
            .fold(index, |index, op| op.apply_inplace(index, coordinates, stride))
    }
    fn apply_index(&self, index: usize) -> usize {
        self.0.iter().rev().fold(index, |index, op| op.apply_index(index))
    }
    fn apply_indices_inplace(&self, indices: &mut [usize]) {
        for op in self.0.iter().rev() {
            op.apply_indices_inplace(indices);
        }
    }
    fn unapply_indices(&self, indices: &[usize]) -> Vec<usize> {
        self.0
            .iter()
            .fold(indices.to_vec(), |indices, op| op.unapply_indices(&indices))
    }
}

impl FromIterator<Operator> for UnsizedChain {
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = Operator>,
    {
        Self::new(iter.into_iter().collect())
    }
}

impl IntoIterator for UnsizedChain {
    type Item = Operator;
    type IntoIter = std::vec::IntoIter<Operator>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<const N: usize> From<[Operator; N]> for UnsizedChain {
    fn from(operators: [Operator; N]) -> Self {
        operators.into_iter().collect()
    }
}

impl From<Vec<Operator>> for UnsizedChain {
    fn from(operators: Vec<Operator>) -> Self {
        Self(operators)
    }
}

impl From<UnsizedChain> for Vec<Operator> {
    fn from(chain: UnsizedChain) -> Self {
        chain.0
    }
}

///// A chain of [`Operator`]s.
//#[derive(Debug, Clone, PartialEq)]
//pub struct UnsizedChain {
//    rev_operators: Vec<Operator>,
//}
//
//impl UnsizedChain {
//    #[inline]
//    pub fn new<Operators>(operators: Operators) -> Self
//    where
//        Operators: IntoIterator<Item = Operator>,
//        Operators::IntoIter: DoubleEndedIterator,
//    {
//        UnsizedChain {
//            rev_operators: operators.into_iter().rev().collect(),
//        }
//    }
//    #[inline]
//    pub fn empty() -> Self {
//        UnsizedChain {
//            rev_operators: Vec::new(),
//        }
//    }
//    pub fn push(&mut self, operator: impl Into<Operator>) {
//        self.rev_operators.insert(0, operator.into())
//    }
//    /// Returns a clone of this [`UnsizedChain`] with the given `operator` appended.
//    #[inline]
//    pub fn clone_and_push(&self, operator: Operator) -> Self {
//        Self::new(
//            self.rev_operators
//                .iter()
//                .rev()
//                .cloned()
//                .chain(iter::once(operator)),
//        )
//    }
//    #[inline]
//    pub fn iter(&self) -> impl Iterator<Item = &Operator> + DoubleEndedIterator {
//        self.rev_operators.iter().rev()
//    }
//    fn split_heads(&self) -> BTreeMap<Operator, Vec<Operator>> {
//        let mut heads = BTreeMap::new();
//        'a: for (i, head) in self.rev_operators.iter().enumerate() {
//            let mut rev_tail: Vec<_> = self.rev_operators.iter().take(i).cloned().collect();
//            let mut head = head.clone();
//            for op in self.rev_operators.iter().skip(i + 1) {
//                if let Some(ops) = swap(op, &mut head) {
//                    rev_tail.extend(ops.into_iter().rev());
//                } else {
//                    continue 'a;
//                }
//            }
//            heads.insert(head, rev_tail);
//        }
//        'b: for (i, op) in self.rev_operators.iter().enumerate() {
//            if let Operator::Edges(Edges {
//                simplex: Simplex::Line,
//                offset,
//            }) = op
//            {
//                let simplex = Simplex::Line;
//                let mut rev_tail: Vec<_> = self.rev_operators.iter().take(i).cloned().collect();
//                let mut head = Operator::new_children(simplex, *offset);
//                let indices = simplex.swap_edges_children_map();
//                let take = Operator::new_take(indices, simplex.nchildren() * simplex.nedges());
//                rev_tail.push(take);
//                rev_tail.push(op.clone());
//                for op in self.rev_operators.iter().skip(i + 1) {
//                    if let Some(ops) = swap(op, &mut head) {
//                        rev_tail.extend(ops.into_iter().rev());
//                    } else {
//                        continue 'b;
//                    }
//                }
//                heads.insert(head, rev_tail);
//            }
//        }
//        heads
//    }
//    /// Remove and return the common prefix of two chains, transforming either if necessary.
//    pub fn remove_common_prefix(&self, other: &Self) -> (Self, Self, Self) {
//        let mut common = Vec::new();
//        let mut rev_a = self.rev_operators.clone();
//        let mut rev_b = other.rev_operators.clone();
//        let mut i = 0;
//        while !rev_a.is_empty() && !rev_b.is_empty() {
//            i += 1;
//            if i > 10 {
//                break;
//            }
//            if rev_a.last() == rev_b.last() {
//                common.push(rev_a.pop().unwrap());
//                rev_b.pop();
//                continue;
//            }
//            let heads_a = UnsizedChain::new(rev_a.iter().rev().cloned()).split_heads();
//            let heads_b = UnsizedChain::new(rev_b.iter().rev().cloned()).split_heads();
//            if let Some((head, a, b)) = heads_a
//                .iter()
//                .filter_map(|(h, a)| heads_b.get(h).map(|b| (h, a, b)))
//                .min_by_key(|(_, a, b)| std::cmp::max(a.len(), b.len()))
//            {
//                common.push(head.clone());
//                rev_a = a.clone();
//                rev_b = b.clone();
//                continue;
//            }
//            break;
//        }
//        let common = if rev_a.is_empty()
//            && (!rev_b.is_empty() || self.rev_operators.len() <= other.rev_operators.len())
//        {
//            self.clone()
//        } else if rev_b.is_empty() {
//            other.clone()
//        } else {
//            Self::new(common)
//        };
//        (
//            common,
//            Self::new(rev_a.into_iter().rev()),
//            Self::new(rev_b.into_iter().rev()),
//        )
//    }
//}
//
//impl UnsizedSequence for UnsizedChain {
//    #[inline]
//    fn delta_dim(&self) -> usize {
//        self.rev_operators.iter().map(|op| op.delta_dim()).sum()
//    }
//    #[inline]
//    fn delta_len(&self) -> (usize, usize) {
//        self.rev_operators.iter().rfold((1, 1), |(n, d), op| {
//            let (opn, opd) = op.delta_len();
//            (n * opn, d * opd)
//        })
//    }
//    #[inline]
//    fn increment_offset(&mut self, amount: usize) {
//        for op in self.rev_operators.iter_mut() {
//            op.increment_offset(amount);
//        }
//    }
//    #[inline]
//    fn decrement_offset(&mut self, amount: usize) {
//        for op in self.rev_operators.iter_mut() {
//            op.decrement_offset(amount);
//        }
//    }
//    #[inline]
//    fn increment_stride(&mut self, amount: usize) {
//        for op in self.rev_operators.iter_mut() {
//            op.increment_stride(amount);
//        }
//    }
//    #[inline]
//    fn decrement_stride(&mut self, amount: usize) {
//        for op in self.rev_operators.iter_mut() {
//            op.decrement_stride(amount);
//        }
//    }
//    #[inline]
//    fn apply_index(&self, index: usize) -> usize {
//        self.rev_operators
//            .iter()
//            .fold(index, |index, op| op.apply_index(index))
//    }
//    #[inline]
//    fn unapply_indices(&self, indices: &[usize]) -> Vec<usize> {
//        let indices = indices.iter().cloned().collect();
//        self.rev_operators
//            .iter()
//            .rev()
//            .fold(indices, |indices, op| op.unapply_indices(&indices))
//    }
//    #[inline]
//    fn apply_one_inplace(&self, index: usize, coordinate: &mut [f64]) -> usize {
//        self.rev_operators
//            .iter()
//            .fold(index, |index, op| op.apply_one_inplace(index, coordinate))
//    }
//    #[inline]
//    fn apply_many_inplace(&self, index: usize, coordinates: &mut [f64], dim: usize) -> usize {
//        self.rev_operators.iter().fold(index, |index, op| {
//            op.apply_many_inplace(index, coordinates, dim)
//        })
//    }
//}
//
//impl IntoIterator for UnsizedChain {
//    type Item = Operator;
//    type IntoIter = std::vec::IntoIter<Self::Item>;
//
//    fn into_iter(self) -> Self::IntoIter {
//        self.rev_operators.into_iter()
//    }
//}
//
//impl FromIterator<Operator> for UnsizedChain {
//    fn from_iter<T>(iter: T) -> Self
//    where
//        T: IntoIterator<Item = Operator>,
//    {
//        let ops: Vec<_> = iter.into_iter().collect();
//        UnsizedChain::new(ops)
//    }
//}
//
//#[derive(Debug, Clone)]
//pub struct Topology {
//    transforms: UnsizedChain,
//    dim: usize,
//    root_len: usize,
//    len: usize,
//}
//
//impl Topology {
//    pub fn new(dim: usize, len: usize) -> Self {
//        Self {
//            transforms: UnsizedChain::new([]),
//            dim,
//            root_len: len,
//            len,
//        }
//    }
//    pub fn derive(&self, operator: Operator) -> Self {
//        let (n, d) = operator.delta_len();
//        Self {
//            root_len: self.root_len,
//            len: self.len * n / d,
//            dim: self.dim - operator.delta_dim(),
//            transforms: self.transforms.clone_and_push(operator),
//        }
//    }
//}
//
//impl Mul for &Topology {
//    type Output = Topology;
//
//    fn mul(self, other: &Topology) -> Topology {
//        Topology {
//            transforms: UnsizedChain::new(
//                iter::once(Operator::new_transpose(other.root_len, self.root_len))
//                    .chain(self.transforms.iter().cloned())
//                    .chain(iter::once(Operator::new_transpose(
//                        self.len,
//                        other.root_len,
//                    )))
//                    .chain(other.transforms.iter().map(|op| {
//                        let mut op = op.clone();
//                        op.increment_offset(self.dim);
//                        op
//                    })),
//            ),
//            dim: self.dim + other.dim,
//            root_len: self.root_len * other.root_len,
//            len: self.len * other.len,
//        }
//    }
//}
//
//
//
//
//    macro_rules! assert_eq_op_apply {
//        ($op:expr, $ii:expr, $ic:expr, $oi:expr, $oc:expr) => {{
//            let ic = $ic;
//            let oc = $oc;
//            let mut work = oc.clone();
//            for i in 0..ic.len() {
//                work[i] = ic[i];
//            }
//            for i in ic.len()..oc.len() {
//                work[i] = 0.0;
//            }
//            assert_eq!($op.apply_one_inplace($ii, &mut work), $oi);
//            assert_abs_diff_eq!(work[..], oc[..]);
//        }};
//    }
//
//    #[test]
//    fn apply_children_line() {
//        let op = Operator::new_children(Line, 0);
//        assert_eq_op_apply!(op, 0 * 2 + 0, [0.0], 0, [0.0]);
//        assert_eq_op_apply!(op, 1 * 2 + 0, [1.0], 1, [0.5]);
//        assert_eq_op_apply!(op, 2 * 2 + 1, [0.0], 2, [0.5]);
//        assert_eq_op_apply!(op, 3 * 2 + 1, [1.0], 3, [1.0]);
//        assert_eq_op_apply!(op, 0, [0.0, 2.0], 0, [0.0, 2.0]);
//        assert_eq_op_apply!(op, 1, [0.0, 3.0, 4.0], 0, [0.5, 3.0, 4.0]);
//        let op = Operator::new_children(Line, 1);
//        assert_eq_op_apply!(op, 1, [2.0, 0.0], 0, [2.0, 0.5]);
//        assert_eq_op_apply!(op, 1, [3.0, 0.0, 4.0], 0, [3.0, 0.5, 4.0]);
//    }
//
//    #[test]
//    fn apply_edges_line() {
//        let op = Operator::new_edges(Line, 0);
//        assert_eq_op_apply!(op, 0, [], 0, [1.0]);
//        assert_eq_op_apply!(op, 3, [], 1, [0.0]);
//        assert_eq_op_apply!(op, 4, [], 2, [1.0]);
//        assert_eq_op_apply!(op, 7, [], 3, [0.0]);
//        assert_eq_op_apply!(op, 0, [2.0], 0, [1.0, 2.0]);
//        assert_eq_op_apply!(op, 1, [3.0, 4.0], 0, [0.0, 3.0, 4.0]);
//        let op = Operator::new_edges(Line, 1);
//        assert_eq_op_apply!(op, 0, [2.0], 0, [2.0, 1.0]);
//        assert_eq_op_apply!(op, 0, [3.0, 4.0], 0, [3.0, 1.0, 4.0]);
//    }
//
//    // #[test]
//    // fn apply_edges_square() {
//    //     let op = Operator::Edges {
//    //         simplices: Box::new([Line, Line]),
//    //         offset: 0,
//    //     };
//    //     assert_eq!(op.apply(0 * 4 + 0, &[0.0]), (0, vec![1.0, 0.0]));
//    //     assert_eq!(op.apply(1 * 4 + 0, &[1.0]), (1, vec![1.0, 1.0]));
//    //     assert_eq!(op.apply(2 * 4 + 1, &[0.0]), (2, vec![0.0, 0.0]));
//    //     assert_eq!(op.apply(3 * 4 + 1, &[1.0]), (3, vec![0.0, 1.0]));
//    //     assert_eq!(op.apply(4 * 4 + 2, &[0.0]), (4, vec![0.0, 1.0]));
//    //     assert_eq!(op.apply(5 * 4 + 2, &[1.0]), (5, vec![1.0, 1.0]));
//    //     assert_eq!(op.apply(6 * 4 + 3, &[0.0]), (6, vec![0.0, 0.0]));
//    //     assert_eq!(op.apply(7 * 4 + 3, &[1.0]), (7, vec![1.0, 0.0]));
//    //     assert_eq!(op.apply(0, &[0.0, 2.0]), (0, vec![1.0, 0.0, 2.0]));
//    //     assert_eq!(op.apply(1, &[0.0, 3.0, 4.0]), (0, vec![0.0, 0.0, 3.0, 4.0]));
//    // }
//
//    #[test]
//    fn apply_transpose_index() {
//        let op = Operator::new_transpose(2, 3);
//        for i in 0..3 {
//            for j in 0..2 {
//                for k in 0..3 {
//                    assert_eq!(
//                        op.apply_one((i * 2 + j) * 3 + k, &[]),
//                        ((i * 3 + k) * 2 + j, vec![])
//                    );
//                }
//            }
//        }
//    }
//
//    #[test]
//    fn apply_take_all() {
//        let op = Operator::new_take([3, 2, 0, 4, 1], 5); // inverse: [2, 4, 1, 0, 3]
//        assert_eq_op_apply!(op, 0, [], 3, []);
//        assert_eq_op_apply!(op, 6, [1.0], 7, [1.0]);
//        assert_eq_op_apply!(op, 12, [2.0, 3.0], 10, [2.0, 3.0]);
//        assert_eq_op_apply!(op, 18, [], 19, []);
//        assert_eq_op_apply!(op, 24, [], 21, []);
//    }
//
//    #[test]
//    fn apply_take_some() {
//        let op = Operator::new_take([4, 0, 1], 5); // inverse: [1, 2, x, x, 0]
//        assert_eq_op_apply!(op, 0, [], 4, []);
//        assert_eq_op_apply!(op, 4, [1.0], 5, [1.0]);
//        assert_eq_op_apply!(op, 8, [2.0, 3.0], 11, [2.0, 3.0]);
//    }
//
//    #[test]
//    fn apply_uniform_points() {
//        let op = Operator::new_uniform_points(Box::new([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]), 2, 0);
//        assert_eq_op_apply!(op, 0, [], 0, [0.0, 1.0]);
//        assert_eq_op_apply!(op, 4, [6.0], 1, [2.0, 3.0, 6.0]);
//        assert_eq_op_apply!(op, 8, [7.0, 8.0], 2, [4.0, 5.0, 7.0, 8.0]);
//    }
//
//    #[test]
//    fn mul_topo() {
//        let xtopo = Topology::new(1, 2).derive(Operator::new_children(Line, 0));
//        let ytopo = Topology::new(1, 3).derive(Operator::new_children(Line, 0));
//        let xytopo = &xtopo * &ytopo;
//        assert_eq!(xtopo.len, 4);
//        assert_eq!(ytopo.len, 6);
//        assert_eq!(xytopo.len, 24);
//        assert_eq!(xytopo.root_len, 6);
//        for i in 0..4 {
//            for j in 0..6 {
//                let x = xtopo.transforms.apply_many(i, &[0.0, 0.0, 1.0, 1.0], 1).1;
//                let y = ytopo.transforms.apply_many(j, &[0.0, 1.0, 0.0, 1.0], 1).1;
//                let mut xy = Vec::with_capacity(8);
//                for k in 0..4 {
//                    xy.push(x[k]);
//                    xy.push(y[k]);
//                }
//                assert_eq!(
//                    xytopo.transforms.apply_many(
//                        i * 6 + j,
//                        &[0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
//                        2
//                    ),
//                    ((i / 2) * 3 + j / 2, xy),
//                );
//            }
//        }
//    }
//
//    macro_rules! assert_equiv_topo {
//        ($topo1:expr, $topo2:expr$(, $simplex:ident)*) => {
//            #[allow(unused_mut)]
//            let mut topo1 = $topo1.clone();
//            #[allow(unused_mut)]
//            let mut topo2 = $topo2.clone();
//            assert_eq!(topo1.dim, topo2.dim, "topos have different dim");
//            assert_eq!(topo1.len, topo2.len, "topos have different len");
//            assert_eq!(topo1.root_len, topo2.root_len, "topos have different root_len");
//            let from_dim = 0$(+$simplex.dim())*;
//            assert_eq!(topo1.dim, from_dim, "dimension of topo differs from dimension of given simplices");
//            let nelems = topo1.len;
//            $(
//                let points = Operator::new_uniform_points(
//                    $simplex.vertices().into(),
//                    $simplex.dim(),
//                    0,
//                );
//                topo1 = topo1.derive(points.clone());
//                topo2 = topo2.derive(points.clone());
//            )*
//            let npoints = topo1.len;
//            let mut coord1: Vec<_> = iter::repeat(0.0).take((topo1.dim + topo1.transforms.delta_dim()) as usize).collect();
//            let mut coord2 = coord1.clone();
//            for i in 0..topo1.len {
//                let ielem = i / (npoints / nelems);
//                assert_eq!(
//                    topo1.transforms.apply_one_inplace(i, &mut coord1),
//                    topo2.transforms.apply_one_inplace(i, &mut coord2),
//                    "topo1 and topo2 map element {ielem} to different root elements"
//                );
//                assert_abs_diff_eq!(coord1[..], coord2[..]);
//            }
//        };
//    }
//
//    #[test]
//    fn swap_edges_children_1d() {
//        let topo1 = Topology::new(1, 3).derive(Operator::new_edges(Line, 0));
//        let topo2 = Topology::new(1, 3)
//            .derive(Operator::new_children(Line, 0))
//            .derive(Operator::new_edges(Line, 0))
//            .derive(Operator::new_take([2, 1], 4));
//        assert_equiv_topo!(topo1, topo2);
//    }
//
//    #[test]
//    fn swap_take_children() {
//        let take = Operator::new_take([2, 3, 1], 5);
//        let children = Operator::new_children(Line, 0);
//        let swapped = vec![
//            children.clone(),
//            Operator::new_transpose(2, 5),
//            take.clone(),
//            Operator::new_transpose(3, 2),
//        ];
//        let base = Topology::new(1, 5);
//        assert_eq!(take.swap(&children), Some(swapped.clone()));
//        assert_equiv_topo!(
//            base.derive(take).derive(children),
//            swapped
//                .iter()
//                .cloned()
//                .fold(base.clone(), |t, o| t.derive(o)),
//            Line
//        );
//    }
//
//    #[test]
//    fn swap_take_edges() {
//        let take = Operator::new_take([2, 3, 1], 5);
//        let edges = Operator::new_edges(Line, 0);
//        let swapped = vec![
//            edges.clone(),
//            Operator::new_transpose(2, 5),
//            take.clone(),
//            Operator::new_transpose(3, 2),
//        ];
//        let base = Topology::new(1, 5);
//        assert_eq!(take.swap(&edges), Some(swapped.clone()));
//        assert_equiv_topo!(
//            base.derive(take).derive(edges),
//            swapped
//                .iter()
//                .cloned()
//                .fold(base.clone(), |t, o| t.derive(o))
//        );
//    }
//
//    macro_rules! fn_test_operator_swap {
//        ($name:ident, $len:expr $(, $simplex:ident)*; $op1:expr, $op2:expr,) => {
//            #[test]
//            fn $name() {
//                let op1: Operator = $op1;
//                let op2: Operator = $op2;
//                let swapped = op1.swap(&op2).expect("not swapped");
//                println!("op1: {op1:?}");
//                println!("op2: {op2:?}");
//                println!("swapped: {swapped:?}");
//                let root_dim = op1.delta_dim() + op2.delta_dim() $(+ $simplex.dim())*;
//                let base = Topology::new(root_dim, 1);
//                let topo1 = [op1, op2].iter().fold(base.clone(), |t, o| t.derive(o.clone()));
//                let topo2 = swapped.iter().fold(base, |t, o| t.derive(o.clone()));
//                let len = $len;
//                assert_eq!(topo1.len, len, "unswapped topo has unexpected length");
//                assert_eq!(topo2.len, len, "swapped topo has unexpected length");
//                assert_equiv_topo!(topo1, topo2 $(, $simplex)*);
//            }
//        }
//    }
//
//    fn_test_operator_swap! {
//        swap_edges_children_triangle1, 6, Line, Line;
//        Operator::new_edges(Triangle, 0),
//        Operator::new_children(Line, 0),
//    }
//
//    fn_test_operator_swap! {
//        swap_unoverlapping_children_lt_children, 8, Triangle, Line;
//        Operator::new_children(Triangle, 0),
//        Operator::new_children(Line, 2),
//    }
//
//    fn_test_operator_swap! {
//        swap_unoverlapping_children_gt_children, 8, Line, Triangle;
//        Operator::new_children(Line, 2),
//        Operator::new_children(Triangle, 0),
//    }
//
//    fn_test_operator_swap! {
//        swap_unoverlapping_edges_lt_children, 6, Line, Line;
//        Operator::new_edges(Triangle, 0),
//        Operator::new_children(Line, 1),
//    }
//
//    fn_test_operator_swap! {
//        swap_unoverlapping_edges_gt_children, 6, Line, Line;
//        Operator::new_edges(Triangle, 1),
//        Operator::new_children(Line, 0),
//    }
//
//    fn_test_operator_swap! {
//        swap_unoverlapping_children_lt_edges, 6, Line, Line;
//        Operator::new_children(Line, 0),
//        Operator::new_edges(Triangle, 1),
//    }
//
//    fn_test_operator_swap! {
//        swap_unoverlapping_children_gt_edges, 6, Line, Line;
//        Operator::new_children(Line, 2),
//        Operator::new_edges(Triangle, 0),
//    }
//
//    fn_test_operator_swap! {
//        swap_unoverlapping_edges_lt_edges, 6, Line;
//        Operator::new_edges(Line, 0),
//        Operator::new_edges(Triangle, 0),
//    }
//
//    fn_test_operator_swap! {
//        swap_unoverlapping_edges_gt_edges, 6, Line;
//        Operator::new_edges(Line, 2),
//        Operator::new_edges(Triangle, 0),
//    }
//
//    #[test]
//    fn split_heads() {
//        let chain = UnsizedChain::new([
//            Operator::new_edges(Triangle, 1),
//            Operator::new_children(Line, 0),
//            Operator::new_edges(Line, 2),
//            Operator::new_children(Line, 1),
//            Operator::new_children(Line, 0),
//        ]);
//        let desired = chain
//            .iter()
//            .cloned()
//            .fold(Topology::new(4, 1), |topo, op| topo.derive(op));
//        for (head, tail) in chain.split_heads().into_iter() {
//            let actual = iter::once(head)
//                .chain(tail.into_iter().rev())
//                .fold(Topology::new(4, 1), |topo, op| topo.derive(op));
//            assert_equiv_topo!(actual, desired, Line, Line);
//        }
//    }
//

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simplex::Simplex::*;

    #[test]
    fn remove_common_prefix() {
        let c1 = Operator::new_children(Line);
        let e1 = Operator::new_edges(Line);
        let swap_ec1 = Operator::new_take([2, 1], 4);
        let a = UnsizedChain::new(vec![c1.clone(), c1.clone()]);
        let b = UnsizedChain::new(vec![e1.clone()]);
        assert_eq!(
            a.remove_common_prefix(&b),
            (
                UnsizedChain::new(vec![c1.clone(), c1.clone()]),
                UnsizedChain::identity(),
                UnsizedChain::new(vec![e1.clone(), swap_ec1.clone(), swap_ec1.clone()]),
            )
        );
    }
}
