pub mod elementary;
pub mod finite_f64;
//pub mod ops;
//pub mod relative;
pub mod simplex;
//pub mod tesselation;
//pub mod topology;

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
