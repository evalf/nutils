pub mod elementary;
pub mod finite_f64;
pub mod ops;
//pub mod relative;
pub mod simplex;
//pub mod tesselation;
//pub mod topology;

use num::Integer as _;

pub trait Map {
    // Minimum dimension of the input coordinate. If the dimension of the input
    // coordinate of [Map::apply_inplace()] is larger than the minimum, then
    // the map of the surplus is the identity map.
    fn dim_in(&self) -> usize;
    // Minimum dimension of the output coordinate.
    fn dim_out(&self) -> usize {
        self.dim_in() + self.delta_dim()
    }
    // Difference in dimension of the output and input coordinate.
    fn delta_dim(&self) -> usize;
    // Modulus of the input index. The map repeats itself at index `mod_in`
    // and the output index is incremented with `in_index / mod_in * mod_out`.
    fn mod_in(&self) -> usize;
    // Modulus if the output index.
    fn mod_out(&self) -> usize;
    fn apply_mod_out_to_in(&self, n: usize) -> Option<usize> {
        let (i, rem) = n.div_rem(&self.mod_out());
        (rem == 0).then(|| i * self.mod_in())
    }
    fn apply_mod_in_to_out(&self, n: usize) -> Option<usize> {
        let (i, rem) = n.div_rem(&self.mod_in());
        (rem == 0).then(|| i * self.mod_out())
    }
    fn apply_inplace(
        &self,
        index: usize,
        coordinates: &mut [f64],
        stride: usize,
        offset: usize,
    ) -> usize;
    fn apply_index(&self, index: usize) -> usize;
    fn apply_indices_inplace(&self, indices: &mut [usize]) {
        for index in indices.iter_mut() {
            *index = self.apply_index(*index);
        }
    }
    fn unapply_indices<T: UnapplyIndicesData>(&self, indices: &[T]) -> Vec<T>;
    fn is_identity(&self) -> bool {
        self.mod_in() == 1 && self.mod_out() == 1 && self.dim_out() == 0
    }
}

pub trait AddOffset {
    fn add_offset(&mut self, offset: usize);
}

pub trait UnapplyIndicesData: Clone + std::fmt::Debug {
    fn get(&self) -> usize;
    fn set(&self, index: usize) -> Self;
}

impl UnapplyIndicesData for usize {
    #[inline]
    fn get(&self) -> usize {
        *self
    }
    #[inline]
    fn set(&self, index: usize) -> Self {
        index
    }
}

#[macro_export]
macro_rules! assert_map_apply {
    ($item:expr, $inidx:expr, $incoords:expr, $outidx:expr, $outcoords:expr) => {{
        use std::borrow::Borrow;
        let item = $item.borrow();
        let incoords = $incoords;
        let outcoords = $outcoords;
        assert_eq!(incoords.len(), outcoords.len());
        let stride;
        let mut work: Vec<_>;
        if incoords.len() == 0 {
            stride = item.dim_out();
            work = Vec::with_capacity(0);
        } else {
            stride = outcoords[0].len();
            work = iter::repeat(-1.0).take(outcoords.len() * stride).collect();
            for (work, incoord) in iter::zip(work.chunks_mut(stride), incoords.iter()) {
                work[..incoord.len()].copy_from_slice(incoord);
            }
        }
        assert_eq!(item.apply_inplace($inidx, &mut work, stride, 0), $outidx);
        for (actual, desired) in iter::zip(work.chunks(stride), outcoords.iter()) {
            assert_abs_diff_eq!(actual[..], desired[..]);
        }
    }};
    ($item:expr, $inidx:expr, $outidx:expr) => {{
        use std::borrow::Borrow;
        let item = $item.borrow();
        let mut work = Vec::with_capacity(0);
        assert_eq!(
            item.apply_inplace($inidx, &mut work, item.dim_out(), 0),
            $outidx
        );
    }};
}
