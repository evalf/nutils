pub mod chain;
mod finite_f64;
pub mod operator;
pub mod simplex;
pub mod sequence;

pub trait Mapping {
    fn len_out(&self) -> usize;
    fn len_in(&self) -> usize;
    fn dim_out(&self) -> usize {
        self.dim_in() + self.delta_dim()
    }
    fn dim_in(&self) -> usize;
    fn delta_dim(&self) -> usize;
    fn add_offset(&mut self, offset: usize);
    fn apply_inplace_unchecked(&self, index: usize, coordinates: &mut [f64]) -> usize;
    fn apply_inplace(&self, index: usize, coordinates: &mut[f64]) -> Option<usize> {
        if index < self.len_in() {
            Some(self.apply_inplace_unchecked(index, coordinates))
        } else {
            None
        }
    }
    fn apply_index_unchecked(&self, index: usize) -> usize;
    fn apply_index(&self, index: usize) -> Option<usize> {
        if index < self.len_in() {
            Some(self.apply_index_unchecked(index))
        } else {
            None
        }
    }
    fn apply_indices_inplace_unchecked(&self, indices: &mut [usize]) {
        for index in indices.iter_mut() {
            *index = self.apply_index_unchecked(*index);
        }
    }
    fn apply_indices(&self, indices: &[usize]) -> Option<Vec<usize>> {
        if indices.iter().all(|index| *index < self.len_in()) {
            let mut indices = indices.to_vec();
            self.apply_indices_inplace_unchecked(&mut indices);
            Some(indices)
        } else {
            None
        }
    }
    fn unapply_indices_unchecked(&self, indices: &[usize]) -> Vec<usize>;
    fn unapply_indices(&self, indices: &[usize]) -> Option<Vec<usize>> {
        if indices.iter().all(|index| *index < self.len_out()) {
            Some(self.unapply_indices_unchecked(indices))
        } else {
            None
        }
    }
    fn is_identity(&self) -> bool;
}

pub trait UnsizedMapping {
    // Minimum dimension of the input coordinate. If the dimension of the input
    // coordinate of [UnsizedMapping::apply()] is larger than the minimum, then
    // the mapping of the surplus is the identity mapping.
    fn dim_in(&self) -> usize;
    // Minimum dimension of the output coordinate.
    fn dim_out(&self) -> usize {
        self.dim_in() + self.delta_dim()
    }
    // Difference in dimension of the output and input coordinate.
    fn delta_dim(&self) -> usize;
    fn add_offset(&mut self, offset: usize);
    // Modulus of the input index. The mapping repeats itself at index `mod_in`
    // and the output index is incremented with `in_index / mod_in * mod_out`.
    fn mod_in(&self) -> usize;
    // Modulus if the output index.
    fn mod_out(&self) -> usize;
    fn apply_inplace(&self, index: usize, coordinates: &mut[f64], stride: usize) -> usize;
    fn apply_index(&self, index: usize) -> usize;
    fn apply_indices_inplace(&self, indices: &mut [usize]) {
        for index in indices.iter_mut() {
            *index = self.apply_index(*index);
        }
    }
    fn unapply_indices(&self, indices: &[usize]) -> Vec<usize>;
    fn is_identity(&self) -> bool {
        self.mod_in() == 1 && self.mod_out() == 1 && self.dim_out() == 0
    }
}
