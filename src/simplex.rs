/// Simplex.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Simplex {
    Line,
    Triangle,
}

impl Simplex {
    /// Returns the dimension of the simplex.
    #[inline]
    pub const fn dim(&self) -> usize {
        match self {
            Self::Line => 1,
            Self::Triangle => 2,
        }
    }
    /// Returns the dimension of an edge of the simplex.
    #[inline]
    pub const fn edge_dim(&self) -> usize {
        self.dim() - 1
    }
    /// Returns the number of edges of the simplex.
    #[inline]
    pub const fn nedges(&self) -> usize {
        match self {
            Self::Line => 2,
            Self::Triangle => 3,
        }
    }
    /// Returns the number of children of the simplex.
    #[inline]
    pub const fn nchildren(&self) -> usize {
        match self {
            Self::Line => 2,
            Self::Triangle => 4,
        }
    }

    const LINE_SWAP_EDGES_CHILDREN_MAP: [usize; 2] = [2, 1];
    const TRIANGLE_SWAP_EDGES_CHILDREN_MAP: [usize; 6] = [3, 6, 1, 7, 2, 5];

    /// Returns an array of indices of edges of children corresponding to children of edges.
    #[inline]
    pub const fn swap_edges_children_map(&self) -> &'static [usize] {
        match self {
            Self::Line => &Self::LINE_SWAP_EDGES_CHILDREN_MAP,
            Self::Triangle => &Self::TRIANGLE_SWAP_EDGES_CHILDREN_MAP,
        }
    }

    const LINE_CONNECTIVITY: [Option<usize>; 4] = [Some(3), None, None, Some(0)];
    const TRIANGLE_CONNECTIVITY: [Option<usize>; 12] = [
        Some(11),
        None,
        None,
        None,
        Some(10),
        None,
        None,
        None,
        Some(9),
        Some(8),
        Some(4),
        Some(0),
    ];

    /// Returns an array of indices of opposite edges of children, or `None` if
    /// an edge lies on the boundary of the simplex.
    #[inline]
    pub const fn connectivity(&self) -> &'static [Option<usize>] {
        match self {
            Self::Line => &Self::LINE_CONNECTIVITY,
            Self::Triangle => &Self::TRIANGLE_CONNECTIVITY,
        }
    }

    const LINE_VERTICES: [f64; 2] = [0.0, 1.0];
    const TRIANGLE_VERTICES: [f64; 6] = [0.0, 0.0, 0.0, 1.0, 1.0, 0.0];

    /// Returns an array of vertices.
    #[inline]
    pub const fn vertices(&self) -> &'static [f64] {
        match self {
            Self::Line => &Self::LINE_VERTICES,
            Self::Triangle => &Self::TRIANGLE_VERTICES,
        }
    }

    /// Transform the given child `coordinates` for child `index` to this parent
    /// simplex in-place. The returned index is the index of the parent in an
    /// infinite, uniform sequence.
    pub fn apply_child(
        &self,
        index: usize,
        coordinates: &mut [f64],
        stride: usize,
        offset: usize,
    ) -> usize {
        match self {
            Self::Line => {
                for coordinate in coordinates.chunks_mut(stride) {
                    let coordinate = &mut coordinate[offset..];
                    coordinate[0] = 0.5 * (coordinate[0] + (index % 2) as f64);
                }
                index / 2
            }
            Self::Triangle => {
                for coordinate in coordinates.chunks_mut(stride) {
                    let coordinate = &mut coordinate[offset..];
                    coordinate[0] *= 0.5;
                    coordinate[1] *= 0.5;
                    match index % 4 {
                        1 => {
                            coordinate[0] += 0.5;
                        }
                        2 => {
                            coordinate[1] += 0.5;
                        }
                        3 => {
                            coordinate[1] += coordinate[0];
                            coordinate[0] = 0.5 - coordinate[0];
                        }
                        _ => {}
                    }
                }
                index / 4
            }
        }
    }
    pub const fn apply_child_index(&self, index: usize) -> usize {
        match self {
            Self::Line => index / 2,
            Self::Triangle => index / 4,
        }
    }

    pub fn unapply_child_indices(
        &self,
        indices: impl Iterator<Item = usize>,
    ) -> impl Iterator<Item = usize> {
        let n = self.nchildren();
        indices.flat_map(move |i| (0..n).map(move |j| i * n + j))
    }

    /// Transform the given edge `coordinate` for edge `index` to this parent
    /// simplex in-place. The returned index is the index of the parent in an
    /// infinite, uniform sequence.
    pub fn apply_edge(
        &self,
        index: usize,
        coordinates: &mut [f64],
        stride: usize,
        offset: usize,
    ) -> usize {
        match self {
            Self::Line => {
                for coordinate in coordinates.chunks_mut(stride) {
                    let coordinate = &mut coordinate[offset..];
                    coordinate.copy_within(
                        self.edge_dim() as usize..coordinate.len() - 1,
                        self.dim() as usize,
                    );
                    coordinate[0] = (1 - index % 2) as f64;
                }
                index / 2
            }
            Self::Triangle => {
                for coordinate in coordinates.chunks_mut(stride) {
                    let coordinate = &mut coordinate[offset..];
                    coordinate.copy_within(
                        self.edge_dim() as usize..coordinate.len() - 1,
                        self.dim() as usize,
                    );
                    match index % 3 {
                        0 => {
                            coordinate[1] = coordinate[0];
                            coordinate[0] = 1.0 - coordinate[0];
                        }
                        1 => {
                            coordinate[1] = coordinate[0];
                            coordinate[0] = 0.0;
                        }
                        _ => {
                            coordinate[1] = 0.0;
                        }
                    }
                }
                index / 3
            }
        }
    }
    pub const fn apply_edge_index(&self, index: usize) -> usize {
        match self {
            Self::Line => index / 2,
            Self::Triangle => index / 3,
        }
    }

    pub fn unapply_edge_indices(
        &self,
        indices: impl Iterator<Item = usize>,
    ) -> impl Iterator<Item = usize> {
        let n = self.nedges();
        indices.flat_map(move |i| (0..n).map(move |j| i * n + j))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use std::iter;
    use Simplex::*;

    macro_rules! assert_child_index_coord {
        ($simplex:ident, $inidx:expr, $incoords:expr, $outidx:expr, $outcoords:expr) => {{
            let incoords = $incoords;
            let outcoords = $outcoords;
            let stride = outcoords[0].len();
            let mut work: Vec<_> = iter::repeat(-1.0).take(outcoords.len() * stride).collect();
            for (work, incoord) in iter::zip(work.chunks_mut(stride), incoords.iter()) {
                work[..incoord.len()].copy_from_slice(incoord);
            }
            assert_eq!($simplex.apply_child($inidx, &mut work, stride, 0), $outidx);
            for (actual, desired) in iter::zip(work.chunks(stride), outcoords.iter()) {
                assert_abs_diff_eq!(actual[..], desired[..]);
            }
        }};
    }

    macro_rules! assert_edge_index_coord {
        ($simplex:ident, $inidx:expr, $incoords:expr, $outidx:expr, $outcoords:expr) => {{
            let incoords = $incoords;
            let outcoords = $outcoords;
            let stride = outcoords[0].len();
            let mut work: Vec<_> = iter::repeat(-1.0).take(outcoords.len() * stride).collect();
            for (work, incoord) in iter::zip(work.chunks_mut(stride), incoords.iter()) {
                work[..incoord.len()].copy_from_slice(incoord);
            }
            assert_eq!($simplex.apply_edge($inidx, &mut work, stride, 0), $outidx);
            for (actual, desired) in iter::zip(work.chunks(stride), outcoords.iter()) {
                assert_abs_diff_eq!(actual[..], desired[..]);
            }
        }};
    }

    #[test]
    fn line_child_coords() {
        assert_child_index_coord!(Line, 2, [[0.0], [0.5], [1.0]], 1, [[0.0], [0.25], [0.5]]);
        assert_child_index_coord!(Line, 5, [[0.0], [0.5], [1.0]], 2, [[0.5], [0.75], [1.0]]);
    }

    #[test]
    fn line_edge_coords() {
        assert_edge_index_coord!(Line, 2, [[]], 1, [[1.0]]);
        assert_edge_index_coord!(Line, 5, [[]], 2, [[0.0]]);
    }

    #[test]
    fn triangle_child_coords() {
        assert_child_index_coord!(
            Triangle,
            4,
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            1,
            [[0.0, 0.0], [0.5, 0.0], [0.0, 0.5]]
        );
        assert_child_index_coord!(
            Triangle,
            9,
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            2,
            [[0.5, 0.0], [1.0, 0.0], [0.5, 0.5]]
        );
        assert_child_index_coord!(
            Triangle,
            14,
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            3,
            [[0.0, 0.5], [0.5, 0.5], [0.0, 1.0]]
        );
        assert_child_index_coord!(
            Triangle,
            19,
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            4,
            [[0.5, 0.0], [0.0, 0.5], [0.5, 0.5]]
        );
    }

    #[test]
    fn triangle_edge_coords() {
        assert_edge_index_coord!(Triangle, 3, [[0.0], [1.0]], 1, [[1.0, 0.0], [0.0, 1.0]]);
        assert_edge_index_coord!(Triangle, 7, [[0.0], [1.0]], 2, [[0.0, 0.0], [0.0, 1.0]]);
        assert_edge_index_coord!(Triangle, 11, [[0.0], [1.0]], 3, [[0.0, 0.0], [1.0, 0.0]]);
    }

    #[test]
    fn line_swap_edges_children_map() {
        for (i, j) in Line.swap_edges_children_map().iter().cloned().enumerate() {
            let mut x = [0.5];
            let mut y = [0.5];
            Line.apply_edge(i, &mut x, 1, 0);
            Line.apply_child(Line.apply_edge(j, &mut y, 1, 0), &mut y, 1, 0);
            assert_abs_diff_eq!(x[..], y[..]);
        }
    }

    #[test]
    fn triangle_swap_edges_children_map() {
        for (i, j) in Triangle
            .swap_edges_children_map()
            .iter()
            .cloned()
            .enumerate()
        {
            let mut x = [0.5, 0.5];
            let mut y = [0.5, 0.5];
            Triangle.apply_edge(Line.apply_child(i, &mut x, 2, 0), &mut x, 2, 0);
            Triangle.apply_child(Triangle.apply_edge(j, &mut y, 2, 0), &mut y, 2, 0);
            assert_abs_diff_eq!(x[..], y[..]);
        }
    }

    #[test]
    fn line_connectivity() {
        for (i, j) in Line.connectivity().iter().cloned().enumerate() {
            if let Some(j) = j {
                let mut x = [0.5];
                let mut y = [0.5];
                Line.apply_child(Line.apply_edge(i, &mut x, 1, 0), &mut x, 1, 0);
                Line.apply_child(Line.apply_edge(j, &mut y, 1, 0), &mut y, 1, 0);
                assert_abs_diff_eq!(x[..], y[..]);
            }
        }
    }

    #[test]
    fn triangle_connectivity() {
        for (i, j) in Triangle.connectivity().iter().cloned().enumerate() {
            if let Some(j) = j {
                let mut x = [0.5, 0.5];
                let mut y = [0.5, 0.5];
                Triangle.apply_child(Triangle.apply_edge(i, &mut x, 2, 0), &mut x, 2, 0);
                Triangle.apply_child(Triangle.apply_edge(j, &mut y, 2, 0), &mut y, 2, 0);
                assert_abs_diff_eq!(x[..], y[..]);
            }
        }
    }
}
