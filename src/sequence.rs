use crate::chain::{Operator, UnsizedChain, UnsizedSequence as _};
use crate::types::Dim;
use std::iter;
use std::ops::Mul;

pub trait Sequence {
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn root_len(&self) -> usize;
    fn dim(&self) -> Dim;
    fn root_dim(&self) -> Dim;
    fn apply_index(&self, index: usize) -> Option<usize>;
    fn apply_inplace(&self, index: usize, coordinates: &mut [f64]) -> Option<usize>;
    fn apply(&self, index: usize, coordinates: &[f64]) -> Option<(usize, Vec<f64>)>;
}

#[derive(Debug, Clone)]
pub struct Chain {
    root_dim: Dim,
    root_len: usize,
    chain: UnsizedChain,
}

impl Chain {
    pub fn new(root_dim: Dim, root_len: usize, chain: impl Into<UnsizedChain>) -> Self {
        Self {
            root_dim,
            root_len,
            chain: chain.into(),
        }
    }
    pub fn empty(dim: Dim, len: usize) -> Self {
        Self {
            root_dim: dim,
            root_len: len,
            chain: UnsizedChain::empty(),
        }
    }
    pub fn push(&mut self, operator: impl Into<Operator>) {
        self.chain.push(operator.into());
    }
}

impl Sequence for Chain {
    fn len(&self) -> usize {
        let (n, d) = self.chain.delta_len();
        self.root_len * n / d
    }
    fn root_len(&self) -> usize {
        self.root_len
    }
    fn dim(&self) -> Dim {
        self.root_dim - self.chain.delta_dim()
    }
    fn root_dim(&self) -> Dim {
        self.root_dim
    }
    fn apply_index(&self, index: usize) -> Option<usize> {
        if index < self.len() {
            let root_index = self.chain.apply_index(index);
            assert!(root_index < self.root_len);
            Some(root_index)
        } else {
            None
        }
    }
    fn apply_inplace(&self, index: usize, coordinates: &mut [f64]) -> Option<usize> {
        if index < self.len() {
            let root_index = self
                .chain
                .apply_many_inplace(index, coordinates, self.root_dim);
            assert!(root_index < self.root_len);
            Some(root_index)
        } else {
            None
        }
    }
    fn apply(&self, index: usize, coordinates: &[f64]) -> Option<(usize, Vec<f64>)> {
        if index < self.len() {
            let (root_index, root_coords) = self.chain.apply_many(index, coordinates, self.dim());
            assert!(root_index < self.root_len);
            Some((root_index, root_coords))
        } else {
            None
        }
    }
}

impl Mul for Chain {
    type Output = Self;

    fn mul(self, mut other: Self) -> Self {
        let root_dim = self.root_dim() + other.root_dim();
        let root_len = self.root_len() * other.root_len();
        let trans1 = Operator::new_transpose(other.root_len(), self.root_len());
        let trans2 = Operator::new_transpose(self.len(), other.root_len());
        other.chain.increment_offset(self.dim());
        let chain: UnsizedChain = iter::once(trans1)
            .chain(self.chain.into_iter())
            .chain(iter::once(trans2))
            .chain(other.chain.into_iter())
            .collect();
        Chain::new(root_dim, root_len, chain)
    }
}

fn get_root_indices(seq: impl Sequence) -> Vec<usize> {
    (0..seq.len()).into_iter().map(|i| seq.apply_index(i).unwrap()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chain::{Children, Edges};
    use crate::simplex::Simplex::*;

    #[test]
    fn test() {
        let mut chain = Chain::empty(1, 2);
        chain.push(Children::new(Line, 0));
        assert_eq!(get_root_indices(chain), vec![0, 0, 1, 1]);
        // Find the root indices of the other tail.
    }
}
