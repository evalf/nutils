pub mod elementary;
pub mod finite_f64;
pub mod ops;
pub mod relative;
pub mod simplex;
mod util;
//pub mod tesselation;
//pub mod topology;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    Empty,
    DimensionMismatch,
    LengthMismatch,
}

impl std::error::Error for Error {}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::Empty => write!(f, "The input array is empty."),
            Self::DimensionMismatch => write!(f, "The dimensions of the maps differ."),
            Self::LengthMismatch => write!(f, "The lengths of the maps differ."),
        }
    }
}

/// An interface for an index and coordinate map.
pub trait Map {
    /// Returns the exclusive upper bound of the indices in the codomain.
    fn len_out(&self) -> usize;
    /// Returns the exclusive upper bound of the indices of the domain.
    fn len_in(&self) -> usize;
    /// Returns the dimension of the coordinates of the codimain.
    fn dim_out(&self) -> usize {
        self.dim_in() + self.delta_dim()
    }
    /// Returns the dimension of the coordinates of the dimain.
    fn dim_in(&self) -> usize;
    /// Returns the dimension difference of the coordinates in the codomain and the domain.
    fn delta_dim(&self) -> usize;
    /// Apply the given index and coordinate, the latter in-place, without
    /// checking whether the index is inside the domain and the coordinates
    /// have at least dimension [`Self::dim_out()`].
    fn apply_inplace_unchecked(
        &self,
        index: usize,
        coords: &mut [f64],
        stride: usize,
        offset: usize,
    ) -> usize;
    /// Applies the given index and coordinate, the latter in-place. The
    /// coordinates must have a dimension not smaller than
    /// [`Self::dim_out()`]. Returns `None` if the index is outside the domain.
    fn apply_inplace(
        &self,
        index: usize,
        coords: &mut [f64],
        stride: usize,
        offset: usize,
    ) -> Option<usize> {
        if index < self.len_in() && offset + self.dim_out() <= stride {
            Some(self.apply_inplace_unchecked(index, coords, stride, offset))
        } else {
            None
        }
    }
    /// Applies the given index without checking that the index is inside the
    /// domain.
    fn apply_index_unchecked(&self, index: usize) -> usize;
    /// Apply the given index. Returns `None` if the index is outside the
    /// domain,
    fn apply_index(&self, index: usize) -> Option<usize> {
        if index < self.len_in() {
            Some(self.apply_index_unchecked(index))
        } else {
            None
        }
    }
    /// Applies the given indices in-place without checking that the indices are
    /// inside the domain.
    fn apply_indices_inplace_unchecked(&self, indices: &mut [usize]) {
        for index in indices.iter_mut() {
            *index = self.apply_index_unchecked(*index);
        }
    }
    /// Applies the given indices in-place. Returns `None` if any of the
    /// indices is outside the domain.
    fn apply_indices(&self, indices: &[usize]) -> Option<Vec<usize>> {
        if indices.iter().all(|index| *index < self.len_in()) {
            let mut indices = indices.to_vec();
            self.apply_indices_inplace_unchecked(&mut indices);
            Some(indices)
        } else {
            None
        }
    }
    fn unapply_indices_unchecked<T: UnapplyIndicesData>(&self, indices: &[T]) -> Vec<T>;
    fn unapply_indices<T: UnapplyIndicesData>(&self, indices: &[T]) -> Option<Vec<T>> {
        if indices.iter().all(|index| index.get() < self.len_out()) {
            Some(self.unapply_indices_unchecked(indices))
        } else {
            None
        }
    }
    /// Returns true if this map is the identity map.
    fn is_identity(&self) -> bool;
    /// Returns true if this map returns coordinates unaltered.
    fn is_index_map(&self) -> bool;
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
        assert_eq!(incoords.len(), outcoords.len(), "incoords outcoords");
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
        assert_eq!(
            item.apply_inplace($inidx, &mut work, stride, 0),
            Some($outidx),
            "apply_inplace",
        );
        assert_eq!(item.apply_index($inidx), Some($outidx), "apply_index");
        for (actual, desired) in iter::zip(work.chunks(stride), outcoords.iter()) {
            assert_abs_diff_eq!(actual[..], desired[..]);
        }
    }};
    ($item:expr, $inidx:expr, $outidx:expr) => {{
        use std::borrow::Borrow;
        let item = $item.borrow();
        let mut work = Vec::with_capacity(0);
        assert_eq!(
            item.apply_inplace($inidx, &mut work, item.dim_out(), 0)
                .unwrap(),
            $outidx
        );
        assert_eq!(item.apply_index($inidx), Some($outidx));
    }};
}
