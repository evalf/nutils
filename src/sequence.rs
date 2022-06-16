use crate::operator::Operator;
use crate::chain::UnsizedChain;
use std::iter;
use std::ops::Mul;
use crate::{Sequence, UnsizedMapping};

#[derive(Debug, Clone)]
pub struct Chain {
    root_dim: usize,
    root_len: usize,
    chain: UnsizedChain,
}

impl Chain {
    pub fn new(root_dim: usize, root_len: usize, chain: impl Into<UnsizedChain>) -> Self {
        Self {
            root_dim,
            root_len,
            chain: chain.into(),
        }
    }
    pub fn empty(dim: usize, len: usize) -> Self {
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
        self.root_len * self.chain.mod_in() / self.chain.mod_out()
    }
    fn root_len(&self) -> usize {
        self.root_len
    }
    fn dim(&self) -> usize {
        self.root_dim - self.chain.delta_dim()
    }
    fn root_dim(&self) -> usize {
        self.root_dim
    }
    fn apply_inplace_unchecked(&self, index: usize, coordinates: &mut [f64]) -> usize {
        self.chain.apply_inplace(index, coordinates, self.root_dim)
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
}

impl Mul for Chain {
    type Output = Self;

    fn mul(self, mut other: Self) -> Self {
        let root_dim = self.root_dim() + other.root_dim();
        let root_len = self.root_len() * other.root_len();
        let trans1 = Operator::new_transpose(other.root_len(), self.root_len());
        let trans2 = Operator::new_transpose(self.len(), other.root_len());
        other.chain.add_offset(self.dim());
        let chain: UnsizedChain = iter::once(trans1)
            .chain(self.chain.into_iter())
            .chain(iter::once(trans2))
            .chain(other.chain.into_iter())
            .collect();
        Chain::new(root_dim, root_len, chain)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operator::Operator;
    use crate::simplex::Simplex::*;

    #[test]
    fn test() {
        let mut chain = Chain::empty(1, 2);
        chain.push(Operator::new_children(Line));
        let indices: Vec<usize> = (0..chain.len()).into_iter().collect();
        assert_eq!(chain.apply_indices(&indices), Some(vec![0, 0, 1, 1]));
        // Find the root indices of the other tail.
    }
}
