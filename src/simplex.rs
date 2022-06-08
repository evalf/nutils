use crate::types::Dim;

/// Simplex.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Simplex {
    Line,
    Triangle,
}

impl Simplex {
    /// Returns the dimension of the simplex.
    #[inline]
    pub const fn dim(&self) -> Dim {
        match self {
            Self::Line => 1,
            Self::Triangle => 2,
        }
    }
    /// Returns the dimension of an edge of the simplex.
    #[inline]
    pub const fn edge_dim(&self) -> Dim {
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

    /// Transform the given child `coordinate` for child `index` to this parent
    /// simplex in-place. The returned index is the index of the parent in an
    /// infinite, uniform sequence.
    pub fn apply_child_inplace(&self, index: usize, coordinate: &mut [f64]) -> usize {
        match self {
            Self::Line => {
                coordinate[0] = 0.5 * (coordinate[0] + (index % 2) as f64);
                index / 2
            }
            Self::Triangle => {
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

    pub fn unapply_child_indices(&self, indices: &[usize]) -> Vec<usize> {
        let mut child_indices = Vec::with_capacity(indices.len() * self.nchildren());
        for index in indices.iter() {
            for ichild in 0..self.nchildren() {
                child_indices.push(index * self.nchildren() + ichild);
            }
        }
        child_indices
    }

    /// Transform the given edge `coordinate` for edge `index` to this parent
    /// simplex in-place. The returned index is the index of the parent in an
    /// infinite, uniform sequence.
    pub fn apply_edge_inplace(&self, index: usize, coordinate: &mut [f64]) -> usize {
        coordinate.copy_within(
            self.edge_dim() as usize..coordinate.len() - 1,
            self.dim() as usize,
        );
        match self {
            Self::Line => {
                coordinate[0] = (1 - index % 2) as f64;
                index / 2
            }
            Self::Triangle => {
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

    pub fn unapply_edge_indices(&self, indices: &[usize]) -> Vec<usize> {
        let mut edge_indices = Vec::with_capacity(indices.len() * self.nedges());
        for index in indices.iter() {
            for iedge in 0..self.nedges() {
                edge_indices.push(index * self.nedges() + iedge);
            }
        }
        edge_indices
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use Simplex::*;

    macro_rules! assert_child_index_coord {
        ($simplex:ident, $inidx:expr, $incoords:expr, $outidx:expr, $outcoords:expr) => {{
            let incoords = $incoords;
            let outcoords = $outcoords;
            for i in 0..outcoords.len() {
                let mut work = outcoords[i].clone();
                for j in 0..incoords[i].len() {
                    work[j] = incoords[i][j];
                }
                for j in incoords[i].len()..outcoords[i].len() {
                    work[j] = -1.0;
                }
                assert_eq!($simplex.apply_child_inplace($inidx, &mut work), $outidx);
                assert_abs_diff_eq!(work[..], outcoords[i][..]);
            }
        }};
    }

    macro_rules! assert_edge_index_coord {
        ($simplex:ident, $inidx:expr, $incoords:expr, $outidx:expr, $outcoords:expr) => {{
            let incoords = $incoords;
            let outcoords = $outcoords;
            for i in 0..outcoords.len() {
                let mut work = outcoords[i].clone();
                for j in 0..incoords[i].len() {
                    work[j] = incoords[i][j];
                }
                for j in incoords[i].len()..outcoords[i].len() {
                    work[j] = -1.0;
                }
                assert_eq!($simplex.apply_edge_inplace($inidx, &mut work), $outidx);
                assert_abs_diff_eq!(work[..], outcoords[i][..]);
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
            Line.apply_edge_inplace(i, &mut x);
            Line.apply_child_inplace(Line.apply_edge_inplace(j, &mut y), &mut y);
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
            Triangle.apply_edge_inplace(Line.apply_child_inplace(i, &mut x), &mut x);
            Triangle.apply_child_inplace(Triangle.apply_edge_inplace(j, &mut y), &mut y);
            assert_abs_diff_eq!(x[..], y[..]);
        }
    }

    #[test]
    fn line_connectivity() {
        for (i, j) in Line.connectivity().iter().cloned().enumerate() {
            if let Some(j) = j {
                let mut x = [0.5];
                let mut y = [0.5];
                Line.apply_child_inplace(Line.apply_edge_inplace(i, &mut x), &mut x);
                Line.apply_child_inplace(Line.apply_edge_inplace(j, &mut y), &mut y);
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
                Triangle.apply_child_inplace(Triangle.apply_edge_inplace(i, &mut x), &mut x);
                Triangle.apply_child_inplace(Triangle.apply_edge_inplace(j, &mut y), &mut y);
                assert_abs_diff_eq!(x[..], y[..]);
            }
        }
    }
}
