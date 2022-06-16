use std::cmp::Ordering;

#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(transparent)]
pub struct FiniteF64(pub f64);

impl Eq for FiniteF64 {}

impl PartialOrd for FiniteF64 {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl Ord for FiniteF64 {
    fn cmp(&self, other: &Self) -> Ordering {
        if let Some(ord) = self.0.partial_cmp(&other.0) {
            ord
        } else {
            panic!("not finite");
        }
    }
}
