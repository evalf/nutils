use crate::finite_f64::FiniteF64;
use crate::simplex::Simplex;
use num::Integer;
use std::rc::Rc;
use crate::UnsizedMapping;

#[inline]
const fn divmod(x: usize, y: usize) -> (usize, usize) {
    (x / y, x % y)
}

fn coordinates_iter_mut(
    flat: &mut [f64],
    stride: usize,
    offset: usize,
    dim_out: usize,
    dim_in: usize,
) -> impl Iterator<Item = &mut [f64]> {
    flat.chunks_mut(stride).map(move |coord| {
        let coord = &mut coord[offset..];
        let delta = dim_out - dim_in;
        if delta != 0 {
            coord.copy_within(..coord.len() - delta, delta);
        }
        &mut coord[..dim_out]
    })
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Transpose(usize, usize);

impl Transpose {
    #[inline]
    pub const fn new(len1: usize, len2: usize) -> Self {
        Self(len1, len2)
    }
    #[inline]
    pub fn reverse(&mut self) {
        std::mem::swap(&mut self.0, &mut self.1);
    }
}

impl UnsizedMapping for Transpose {
    fn dim_in(&self) -> usize {
        0
    }
    fn delta_dim(&self) -> usize {
        0
    }
    fn add_offset(&mut self, _offset: usize) {}
    fn mod_in(&self) -> usize {
        if self.0 != 1 && self.1 != 1 {
            self.0 * self.1
        } else {
            1
        }
    }
    fn mod_out(&self) -> usize {
        self.mod_in()
    }
    fn apply_inplace(&self, index: usize, _coordinates: &mut[f64], _stride: usize) -> usize {
        self.apply_index(index)
    }
    fn apply_index(&self, index: usize) -> usize {
        let (j, k) = divmod(index, self.1);
        let (i, j) = divmod(j, self.0);
        (i * self.1 + k) * self.0 + j
    }
    fn unapply_indices(&self, indices: &[usize]) -> Vec<usize> {
        indices
            .iter()
            .map(|k| {
                let (j, k) = divmod(*k, self.0);
                let (i, j) = divmod(j, self.1);
                (i * self.0 + k) * self.1 + j
            })
            .collect()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Take {
    indices: Rc<Box<[usize]>>,
    nindices: usize,
    len: usize,
}

impl Take {
    pub fn new(indices: impl Into<Box<[usize]>>, len: usize) -> Self {
        let indices = Rc::new(indices.into());
        let nindices = indices.len();
        Take {
            indices,
            nindices,
            len,
        }
    }
}

impl UnsizedMapping for Take {
    fn dim_in(&self) -> usize {
        0
    }
    fn delta_dim(&self) -> usize {
        0
    }
    fn add_offset(&mut self, _offset: usize) {}
    fn mod_in(&self) -> usize {
        self.nindices
    }
    fn mod_out(&self) -> usize {
        self.len
    }
    fn apply_inplace(&self, index: usize, _coordinates: &mut[f64], _stride: usize) -> usize {
        self.apply_index(index)
    }
    fn apply_index(&self, index: usize) -> usize {
        self.indices[index % self.nindices] + index / self.nindices * self.len
    }
    fn unapply_indices(&self, indices: &[usize]) -> Vec<usize> {
        indices
            .iter()
            .filter_map(|index| {
                let (j, iout) = divmod(*index, self.len);
                let offset = j * self.nindices;
                self.indices
                    .iter()
                    .position(|i| *i == iout)
                    .map(|iin| offset + iin)
            })
            .collect()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Children(Simplex, usize);

impl Children {
    pub fn new(simplex: Simplex) -> Self {
        Self(simplex, 0)
    }
}

impl UnsizedMapping for Children {
    fn dim_in(&self) -> usize {
        self.0.dim() + self.1
    }
    fn delta_dim(&self) -> usize {
        0
    }
    fn add_offset(&mut self, offset: usize) {
        self.1 += offset;
    }
    fn mod_in(&self) -> usize {
        self.0.nchildren()
    }
    fn mod_out(&self) -> usize {
        1
    }
    fn apply_inplace(&self, index: usize, coordinates: &mut[f64], stride: usize) -> usize {
        self.0.apply_child(index, coordinates, stride, self.1)
    }
    fn apply_index(&self, index: usize) -> usize {
        self.0.apply_child_index(index)
    }
    fn unapply_indices(&self, indices: &[usize]) -> Vec<usize> {
         indices
            .iter()
            .map(|i| i * self.mod_in())
            .flat_map(|i| (0..self.mod_in()).map(move |j| i + j))
            .collect()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Edges(pub Simplex, pub usize);

impl Edges {
    pub fn new(simplex: Simplex) -> Self {
        Self(simplex, 0)
    }
}

impl UnsizedMapping for Edges {
    fn dim_in(&self) -> usize {
        self.0.edge_dim() + self.1
    }
    fn delta_dim(&self) -> usize {
        1
    }
    fn add_offset(&mut self, offset: usize) {
        self.1 += offset;
    }
    fn mod_in(&self) -> usize {
        self.0.nedges()
    }
    fn mod_out(&self) -> usize {
        1
    }
    fn apply_inplace(&self, index: usize, coordinates: &mut[f64], stride: usize) -> usize {
        self.0.apply_edge(index, coordinates, stride, self.1)
    }
    fn apply_index(&self, index: usize) -> usize {
        self.0.apply_edge_index(index)
    }
    fn unapply_indices(&self, indices: &[usize]) -> Vec<usize> {
         indices
            .iter()
            .map(|i| i * self.mod_in())
            .flat_map(|i| (0..self.mod_in()).map(move |j| i + j))
            .collect()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct UniformPoints {
    points: Rc<Box<[FiniteF64]>>,
    npoints: usize,
    point_dim: usize,
    offset: usize,
}

impl UniformPoints {
    pub fn new(points: impl Into<Box<[f64]>>, point_dim: usize) -> Self {
        let points: Rc<Box<[FiniteF64]>> = Rc::new(unsafe { std::mem::transmute(points.into()) });
        assert_eq!(points.len() % point_dim, 0);
        let npoints = points.len() / point_dim;
        UniformPoints {
            points,
            npoints,
            point_dim,
            offset: 0,
        }
    }
}

impl UnsizedMapping for UniformPoints {
    fn dim_in(&self) -> usize {
        self.offset
    }
    fn delta_dim(&self) -> usize {
        self.point_dim
    }
    fn add_offset(&mut self, offset: usize) {
        self.offset += offset;
    }
    fn mod_in(&self) -> usize {
        self.npoints
    }
    fn mod_out(&self) -> usize {
        1
    }
    fn apply_inplace(&self, index: usize, coordinates: &mut[f64], stride: usize) -> usize {
        let points: &[f64] = unsafe { std::mem::transmute(&self.points[..]) };
        let point = &points[(index % self.npoints) * self.point_dim..][..self.point_dim];
        for coord in coordinates_iter_mut(coordinates, stride, self.offset, self.point_dim, 0) {
            coord.copy_from_slice(point);
        }
        index / self.npoints
    }
    fn apply_index(&self, index: usize) -> usize {
        index / self.npoints
    }
    fn unapply_indices(&self, indices: &[usize]) -> Vec<usize> {
         indices
            .iter()
            .map(|i| i * self.mod_in())
            .flat_map(|i| (0..self.mod_in()).map(move |j| i + j))
            .collect()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum Operator {
    Transpose(Transpose),
    Take(Take),
    Children(Children),
    Edges(Edges),
    UniformPoints(UniformPoints),
}

impl Operator {
    #[inline]
    pub fn new_transpose(len1: usize, len2: usize) -> Self {
        Self::Transpose(Transpose::new(len1, len2))
    }
    #[inline]
    pub fn new_take(indices: impl Into<Box<[usize]>>, len: usize) -> Self {
        Self::Take(Take::new(indices, len))
    }
    #[inline]
    pub fn new_children(simplex: Simplex) -> Self {
        Self::Children(Children::new(simplex))
    }
    #[inline]
    pub fn new_edges(simplex: Simplex) -> Self {
        Self::Edges(Edges::new(simplex))
    }
    #[inline]
    pub fn new_uniform_points(points: impl Into<Box<[f64]>>, point_dim: usize) -> Self {
        Self::UniformPoints(UniformPoints::new(points, point_dim))
    }
    #[inline]
    fn offset_mut(&mut self) -> Option<&mut usize> {
        match self {
            Self::Children(Children(_, ref mut offset)) => Some(offset),
            Self::Edges(Edges(_, ref mut offset)) => Some(offset),
            Self::UniformPoints(UniformPoints { ref mut offset, .. }) => Some(offset),
            _ => None,
        }
    }
    #[inline]
    fn set_offset(&mut self, new_offset: usize) {
        self.add_offset(new_offset);
    }
    #[inline]
    pub fn with_offset(mut self, new_offset: usize) -> Self {
        self.set_offset(new_offset);
        self
    }
    pub fn shift_left(&self, operators: &[Self]) -> Option<(Option<Transpose>, Self, Vec<Self>)> {
        if self.is_transpose() {
            return None;
        }
        let mut target = self.clone();
        let mut shifted_ops: Vec<Self> = Vec::new();
        let mut queue: Vec<Self> = Vec::new();
        let mut stride_out = 1;
        let mut stride_in = 1;
        for mut op in operators.iter().rev().cloned() {
            // Swap matching edges and children at the same offset.
            if let Self::Edges(Edges(esimplex, eoffset)) = &op {
                if let Self::Children(Children(ref mut csimplex, coffset)) = &mut target {
                    if eoffset == coffset && esimplex.edge_dim() == csimplex.dim() {
                        if stride_in != 1 && self.mod_in() != 1 {
                            shifted_ops.push(Self::new_transpose(stride_in, self.mod_in()));
                        }
                        shifted_ops.append(&mut queue);
                        if stride_out != 1 && self.mod_in() != 1 {
                            shifted_ops.push(Self::new_transpose(self.mod_in(), stride_out));
                        }
                        shifted_ops.push(Self::new_take(esimplex.swap_edges_children_map(), esimplex.nedges() * esimplex.nchildren()));
                        shifted_ops.push(Self::Edges(Edges(*esimplex, *eoffset)));
                        *csimplex = *esimplex;
                        stride_in = 1;
                        stride_out = 1;
                        continue;
                    }
                }
            }
            // Update strides.
            if self.mod_in() == 1 && self.mod_out() == 1 {
            } else if self.mod_out() == 1 {
                let n = stride_out.gcd(&op.mod_in());
                stride_out = stride_out / n * op.mod_out();
                stride_in *= op.mod_in() / n;
            } else if let Some(Transpose(ref mut m, ref mut n)) = op.as_transpose_mut() {
                if stride_out % (*m * *n) == 0 {
                } else if stride_out % *n == 0 && (*m * *n) % (stride_out * self.mod_out()) == 0 {
                    stride_out /= *n;
                    *m = *m / self.mod_out() * self.mod_in();
                } else if *n % stride_out == 0 && *n % (stride_out * self.mod_out()) == 0 {
                    stride_out *= *m;
                    *n = *n / self.mod_out() * self.mod_in();
                } else {
                    return None;
                }
            } else if stride_out % op.mod_in() == 0 {
                stride_out = stride_out / op.mod_in() * op.mod_out();
            } else {
                return None;
            }
            // Update offsets.
            let op_delta_dim = op.delta_dim();
            let target_delta_dim = target.delta_dim();
            let op_dim_in = op.dim_in();
            let target_dim_out = target.dim_out();
            if let (Some(op_offset), Some(target_offset)) = (op.offset_mut(), target.offset_mut()) {
                if op_dim_in <= *target_offset {
                    *target_offset += op_delta_dim;
                } else if target_dim_out <= *op_offset {
                    *op_offset -= target_delta_dim;
                } else {
                    return None;
                }
            }
            if !op.is_identity() {
                queue.push(op);
            }
        }
        if stride_in != 1 && self.mod_in() != 1 {
            shifted_ops.push(Self::new_transpose(stride_in, self.mod_in()));
        }
        shifted_ops.extend(queue);
        if stride_out != 1 && self.mod_in() != 1 {
            shifted_ops.push(Self::new_transpose(self.mod_in(), stride_out));
        }
        let leading_transpose = if self.mod_out() == 1 || stride_out == 1 {
            None
        } else {
            Some(Transpose::new(stride_out, self.mod_out()))
        };
        shifted_ops.reverse();
        Some((
            leading_transpose,
            target,
            shifted_ops,
        ))
    }
    #[inline]
    const fn is_transpose(&self) -> bool {
        matches!(self, Self::Transpose(_))
    }
    fn as_transpose_mut(&mut self) -> Option<&mut Transpose> {
        match self {
            Self::Transpose(ref mut transpose) => Some(transpose),
            _ => None,
        }
    }
}

macro_rules! dispatch {
    ($vis:vis fn $fn:ident(&$self:ident $(, $arg:ident: $ty:ty)*) $($ret:tt)*) => {
        #[inline]
        $vis fn $fn(&$self $(, $arg: $ty)*) $($ret)* {
            match $self {
                Operator::Transpose(var) => var.$fn($($arg),*),
                Operator::Take(var) => var.$fn($($arg),*),
                Operator::Children(var) => var.$fn($($arg),*),
                Operator::Edges(var) => var.$fn($($arg),*),
                Operator::UniformPoints(var) => var.$fn($($arg),*),
            }
        }
    };
    ($vis:vis fn $fn:ident(&mut $self:ident $(, $arg:ident: $ty:ty)*) $($ret:tt)*) => {
        #[inline]
        $vis fn $fn(&mut $self $(, $arg: $ty)*) $($ret)* {
            match $self {
                Operator::Transpose(var) => var.$fn($($arg),*),
                Operator::Take(var) => var.$fn($($arg),*),
                Operator::Children(var) => var.$fn($($arg),*),
                Operator::Edges(var) => var.$fn($($arg),*),
                Operator::UniformPoints(var) => var.$fn($($arg),*),
            }
        }
    };
}

impl UnsizedMapping for Operator {
    dispatch!{fn dim_in(&self) -> usize}
    dispatch!{fn delta_dim(&self) -> usize}
    dispatch!{fn add_offset(&mut self, offset: usize)}
    dispatch!{fn mod_in(&self) -> usize}
    dispatch!{fn mod_out(&self) -> usize}
    dispatch!{fn apply_inplace(&self, index: usize, coordinates: &mut[f64], stride: usize) -> usize}
    dispatch!{fn apply_index(&self, index: usize) -> usize}
    dispatch!{fn apply_indices_inplace(&self, indices: &mut [usize])}
    dispatch!{fn unapply_indices(&self, indices: &[usize]) -> Vec<usize>}
    dispatch!{fn is_identity(&self) -> bool}
}

impl From<Transpose> for Operator {
    fn from(transpose: Transpose) -> Self {
        Self::Transpose(transpose)
    }
}

impl From<Take> for Operator {
    fn from(take: Take) -> Self {
        Self::Take(take)
    }
}

impl From<Children> for Operator {
    fn from(children: Children) -> Self {
        Self::Children(children)
    }
}

impl From<Edges> for Operator {
    fn from(edges: Edges) -> Self {
        Self::Edges(edges)
    }
}

impl From<UniformPoints> for Operator {
    fn from(uniform_points: UniformPoints) -> Self {
        Self::UniformPoints(uniform_points)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use std::iter;
    use Simplex::*;

    macro_rules! assert_eq_apply {
        ($op:expr, $inidx:expr, $incoords:expr, $outidx:expr, $outcoords:expr) => {{
            use std::borrow::Borrow;
            let op = $op.borrow();
            let incoords = $incoords;
            let outcoords = $outcoords;
            assert_eq!(incoords.len(), outcoords.len());
            let stride;
            let mut work: Vec<_>;
            if incoords.len() == 0 {
                stride = op.dim_out();
                work = Vec::with_capacity(0);
            } else {
                stride = outcoords[0].len();
                work = iter::repeat(-1.0).take(outcoords.len() * stride).collect();
                for (work, incoord) in iter::zip(work.chunks_mut(stride), incoords.iter()) {
                    work[..incoord.len()].copy_from_slice(incoord);
                }
            }
            assert_eq!(op.apply_inplace($inidx, &mut work, stride), $outidx);
            for (actual, desired) in iter::zip(work.chunks(stride), outcoords.iter()) {
                assert_abs_diff_eq!(actual[..], desired[..]);
            }
        }};
        ($op:expr, $inidx:expr, $outidx:expr) => {{
            use std::borrow::Borrow;
            let op = $op.borrow();
            let mut work = Vec::with_capacity(0);
            assert_eq!(op.apply_inplace($inidx, &mut work, op.dim_out()), $outidx);
        }};
    }

    #[test]
    fn apply_transpose() {
        let op = Operator::new_transpose(3, 2);
        assert_eq_apply!(op, 0, 0);
        assert_eq_apply!(op, 1, 3);
        assert_eq_apply!(op, 2, 1);
        assert_eq_apply!(op, 3, 4);
        assert_eq_apply!(op, 4, 2);
        assert_eq_apply!(op, 5, 5);
        assert_eq_apply!(op, 6, 6);
        assert_eq_apply!(op, 7, 9);
    }

    #[test]
    fn apply_take() {
        let op = Operator::new_take([4, 1, 2], 5);
        assert_eq_apply!(op, 0, 4);
        assert_eq_apply!(op, 1, 1);
        assert_eq_apply!(op, 2, 2);
        assert_eq_apply!(op, 3, 9);
        assert_eq_apply!(op, 4, 6);
        assert_eq_apply!(op, 5, 7);
    }

    #[test]
    fn apply_children_line() {
        let op = Operator::new_children(Line);
        assert_eq_apply!(op, 0, [[0.0], [1.0]], 0, [[0.0], [0.5]]);
        assert_eq_apply!(op, 1, [[0.0], [1.0]], 0, [[0.5], [1.0]]);
        assert_eq_apply!(op, 2, [[0.0], [1.0]], 1, [[0.0], [0.5]]);

        let op = op.with_offset(1);
        assert_eq_apply!(op, 3, [[0.2, 0.0], [0.3, 1.0]], 1, [[0.2, 0.5], [0.3, 1.0]]);
    }

    #[test]
    fn apply_edges_line() {
        let op = Operator::new_edges(Line);
        assert_eq_apply!(op, 0, [[]], 0, [[1.0]]);
        assert_eq_apply!(op, 1, [[]], 0, [[0.0]]);
        assert_eq_apply!(op, 2, [[]], 1, [[1.0]]);

        let op = op.with_offset(1);
        assert_eq_apply!(op, 0, [[0.2]], 0, [[0.2, 1.0]]);
    }

    #[test]
    fn apply_uniform_points() {
        let op = Operator::new_uniform_points([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2);
        assert_eq_apply!(op, 0, [[]], 0, [[1.0, 2.0]]);
        assert_eq_apply!(op, 1, [[]], 0, [[3.0, 4.0]]);
        assert_eq_apply!(op, 2, [[]], 0, [[5.0, 6.0]]);
        assert_eq_apply!(op, 3, [[]], 1, [[1.0, 2.0]]);

        let op = op.with_offset(1);
        assert_eq_apply!(op, 0, [[7.0]], 0, [[7.0, 1.0, 2.0]]);
    }

    macro_rules! assert_unapply {
        ($op:expr) => {{
            let op = $op;
            let nin = 2 * op.mod_in();
            let nout = 2 * op.mod_out();
            assert!(nout > 0);
            let mut map: Vec<Vec<usize>> = (0..nout).map(|_| Vec::new()).collect();
            let mut work = Vec::with_capacity(0);
            for i in 0..nin {
                map[op.apply_inplace(i, &mut work, op.dim_out())].push(i);
            }
            for (j, desired) in map.into_iter().enumerate() {
                let mut actual = op.unapply_indices(&[j]);
                actual.sort();
                assert_eq!(actual, desired);
            }
        }};
    }

    #[test]
    fn unapply_indices_transpose() {
        assert_unapply!(Operator::new_transpose(3, 2));
    }

    #[test]
    fn unapply_indices_take() {
        assert_unapply!(Operator::new_take([4, 1], 5));
    }

    #[test]
    fn unapply_indices_children() {
        assert_unapply!(Operator::new_children(Triangle));
    }

    #[test]
    fn unapply_indices_edges() {
        assert_unapply!(Operator::new_edges(Triangle));
    }

    #[test]
    fn unapply_indices_uniform_points() {
        assert_unapply!(Operator::new_uniform_points(
            [0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
            2
        ));
    }

    macro_rules! assert_equiv_chains {
        ($a:expr, $b:expr $(, $simplex:ident)*) => {{
            let a: &[Operator] = &$a;
            let b: &[Operator] = &$b;
            println!("a: {a:?}");
            println!("b: {b:?}");
            let tip_dim = 0 $(+ $simplex.dim())*;
            // Determine the length of the sequence and the root dimension for sequence `a`.
            let mut tip_len = 1;
            let mut root_len = 1;
            let mut root_dim = tip_dim;
            for op in a.iter().rev() {
                let i = (1..)
                    .into_iter()
                    .find(|i| (root_len * i) % op.mod_in() == 0)
                    .unwrap();
                tip_len *= i;
                root_len *= i;
                root_len = root_len / op.mod_in() * op.mod_out();
                assert!(op.dim_in() <= root_dim);
                root_dim += op.delta_dim();
            }
            assert!(tip_len > 0);
            // Verify the length and the root dimension for sequence `b`.
            let mut root_len_b = tip_len;
            let mut root_dim_b = tip_dim;
            for op in b.iter().rev() {
                assert_eq!(root_len_b % op.mod_in(), 0);
                root_len_b = root_len_b / op.mod_in() * op.mod_out();
                assert!(op.dim_in() <= root_dim_b);
                root_dim_b += op.delta_dim();
            }
            assert_eq!(root_len_b, root_len);
            assert_eq!(root_dim_b, root_dim);
            // Build coords: the outer product of the vertices of the given simplices, zero-padded
            // to the dimension of the root.
            let coords = iter::once([]);
            $(
                let coords = coords.flat_map(|coord| {
                    $simplex
                        .vertices()
                        .chunks($simplex.dim())
                        .map(move |vert| [&coord, vert].concat())
                    });
            )*
            let pad: Vec<f64> = iter::repeat(0.0).take(root_dim - tip_dim).collect();
            let coords: Vec<f64> = coords.flat_map(|coord| [&coord[..], &pad].concat()).collect();
            // Test if every tip index and coordinate maps to the same root index and coordinate.
            for itip in 0..2 * tip_len {
                let mut crds_a = coords.clone();
                let mut crds_b = coords.clone();
                let iroot_a = a.iter().rev().fold(itip, |i, op| op.apply_inplace(i, &mut crds_a, root_dim));
                let iroot_b = b.iter().rev().fold(itip, |i, op| op.apply_inplace(i, &mut crds_b, root_dim));
                assert_eq!(iroot_a, iroot_b, "itip={itip}");
                assert_abs_diff_eq!(crds_a[..], crds_b[..]);
            }
        }};
    }

    macro_rules! assert_shift_left {
        ($($op:expr),*; $($simplex:ident),*) => {{
            let unshifted = [$(Operator::from($op),)*];
            let (ltrans, lop, lchain) = unshifted.last().unwrap().shift_left(&unshifted[..unshifted.len()-1]).unwrap();
            let mut shifted: Vec<Operator> = Vec::new();
            if let Some(ltrans) = ltrans {
                shifted.push(ltrans.into());
            }
            shifted.push(lop);
            shifted.extend(lchain.into_iter());
            assert_equiv_chains!(&shifted[..], &unshifted[..] $(, $simplex)*);
        }};
    }

    #[test]
    fn shift_left() {
        assert_shift_left!(
            Transpose::new(4, 3), Operator::new_take([0, 1], 3);
        );
        assert_shift_left!(
            Transpose::new(3, 5), Transpose::new(5, 4*3), Operator::new_take([0, 1], 3);
        );
        assert_shift_left!(
            Transpose::new(5, 4), Transpose::new(5*4, 3), Operator::new_take([0, 1], 3);
        );
        assert_shift_left!(
            Operator::new_children(Line).with_offset(1), Operator::new_children(Line);
            Line, Line
        );
        assert_shift_left!(
            Operator::new_edges(Line), Operator::new_children(Line);
            Line
        );
        assert_shift_left!(
            Operator::new_edges(Line).with_offset(1), Operator::new_children(Line);
            Line
        );
        assert_shift_left!(
            Operator::new_edges(Triangle), Operator::new_children(Line);
            Line
        );
    }
}
