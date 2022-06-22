pub mod finite;
pub mod finite_f64;
pub mod infinite;
pub mod relative_to;
pub mod simplex;

pub trait UnapplyIndicesData: Clone {
    fn last(&self) -> usize;
    fn push(&self, index: usize) -> Self;
}

impl UnapplyIndicesData for usize {
    #[inline]
    fn last(&self) -> usize {
        *self
    }
    #[inline]
    fn push(&self, index: usize) -> Self {
        index
    }
}
