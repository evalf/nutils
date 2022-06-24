use crate::elementary::{Elementary, PushElementary as _};
use crate::ops::{Concatenation, WithBounds};
use crate::relative::RelativeTo as _;
use crate::simplex::Simplex;
use crate::BoundedMap as _;
use std::iter;
use std::rc::Rc;

type Tesselation = Concatenation<WithBounds<Vec<Elementary>>>;

#[derive(Debug, Clone, PartialEq)]
pub struct Root {
    pub name: String,
    pub dim: usize,
}

impl Root {
    pub fn new(name: String, dim: usize) -> Self {
        Self { name, dim }
    }
}

pub trait TopologyCore: std::fmt::Debug + Clone + Into<Topology> {
    fn tesselation(&self) -> &Tesselation;
    fn dim(&self) -> usize {
        self.tesselation().dim_in()
    }
    fn ntiles(&self) -> usize {
        self.tesselation().len_in()
    }
    fn refined(&self) -> Topology;
    fn map_itiles_to_refined(&self, itiles: &[usize]) -> Vec<usize> {
        self.refined()
            .tesselation()
            .unapply_indices_from(self.tesselation(), itiles)
            .unwrap()
    }
    fn boundary(&self) -> Topology;
    fn take(&self, itiles: &[usize]) -> Topology {
        let mut itiles: Vec<usize> = itiles.to_vec();
        itiles.sort_by_key(|&index| index);
        Take::new(self.clone().into(), itiles)
    }
    fn refined_by(&self, itiles: &[usize]) -> Topology {
        Hierarchical::new(self.clone().into(), vec![(0..self.ntiles()).collect()])
            .refined_by(itiles)
    }
}

#[derive(Clone, PartialEq)]
pub enum Topology {
    Point(Rc<Point>),
    Line(Rc<Line>),
    Product(Rc<Product>),
    DisjointUnion(Rc<DisjointUnion>),
    Take(Rc<Take>),
    Hierarchical(Rc<Hierarchical>),
}

macro_rules! dispatch {
    (
        $vis:vis fn $fn:ident$(<$genarg:ident: $genpath:path>)?(
            &$self:ident $(, $arg:ident: $ty:ty)*
        ) $(-> $ret:ty)?
    ) => {
        #[inline]
        $vis fn $fn$(<$genarg: $genpath>)?(&$self $(, $arg: $ty)*) $(-> $ret)? {
            dispatch!(@match $self; $fn; $($arg),*)
        }
    };
    ($vis:vis fn $fn:ident(&mut $self:ident $(, $arg:ident: $ty:ty)*) $(-> $ret:ty)?) => {
        #[inline]
        $vis fn $fn(&mut $self $(, $arg: $ty)*) $(-> $ret)? {
            dispatch!(@match $self; $fn; $($arg),*)
        }
    };
    (@match $self:ident; $fn:ident; $($arg:ident),*) => {
        match $self {
            Self::Point(var) => var.$fn($($arg),*),
            Self::Line(var) => var.$fn($($arg),*),
            Self::Product(var) => var.$fn($($arg),*),
            Self::DisjointUnion(var) => var.$fn($($arg),*),
            Self::Take(var) => var.$fn($($arg),*),
            Self::Hierarchical(var) => var.$fn($($arg),*),
        }
    }
}

impl TopologyCore for Topology {
    dispatch! {fn tesselation(&self) -> &Tesselation}
    dispatch! {fn dim(&self) -> usize}
    dispatch! {fn ntiles(&self) -> usize}
    dispatch! {fn refined(&self) -> Topology}
    dispatch! {fn map_itiles_to_refined(&self, itiles: &[usize]) -> Vec<usize>}
    dispatch! {fn boundary(&self) -> Topology}
    dispatch! {fn take(&self, itiles: &[usize]) -> Topology}
    dispatch! {fn refined_by(&self, itiles: &[usize]) -> Topology}
}

impl std::fmt::Debug for Topology {
    dispatch! {fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result}
}

macro_rules! impl_from_topo {
    ($($from:tt),*) => {
        $(
            impl From<$from> for Topology {
                fn from(topo: $from) -> Topology {
                    Topology::$from(Rc::new(topo))
                }
            }
        )*
    };
}

impl_from_topo! {Point, Line, Product, DisjointUnion, Take, Hierarchical}

#[derive(Debug, Clone, PartialEq)]
pub struct Point {
    tesselation: Tesselation,
}

impl TopologyCore for Point {
    fn dim(&self) -> usize {
        0
    }
    fn tesselation(&self) -> &Tesselation {
        &self.tesselation
    }
    fn refined(&self) -> Topology {
        self.clone().into()
    }
    fn boundary(&self) -> Topology {
        panic!("the boundary of a Point does not exist");
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Line {
    tesselation: Tesselation,
}

impl Line {
    pub fn new(tesselation: Tesselation) -> Topology {
        assert_eq!(tesselation.dim_in(), 1);
        Self { tesselation }.into()
    }
    pub fn from_len(len: usize) -> Topology {
        Self::new(Concatenation::new(vec![
            WithBounds::new(vec![], 1, len).unwrap()
        ]))
    }
}

impl TopologyCore for Line {
    fn dim(&self) -> usize {
        1
    }
    fn tesselation(&self) -> &Tesselation {
        &self.tesselation
    }
    fn refined(&self) -> Topology {
        let mut tesselation = self.tesselation.clone();
        tesselation.push_elementary(&Elementary::new_children(Simplex::Line));
        Self { tesselation }.into()
    }
    fn boundary(&self) -> Topology {
        let mut tesselation = self.tesselation.clone();
        tesselation.push_elementary(&Elementary::new_edges(Simplex::Line));
        let n = tesselation.len_in();
        tesselation.push_elementary(&Elementary::new_take(vec![1, n - 2], n));
        Point { tesselation }.into()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DisjointUnion {
    topos: [Topology; 2],
    tesselation: Tesselation,
}

impl DisjointUnion {
    pub fn new(topo0: Topology, topo1: Topology) -> Topology {
        let tesselation = Concatenation::new(
            topo0
                .tesselation()
                .iter()
                .chain(topo1.tesselation().iter())
                .cloned()
                .collect(),
        );
        Self {
            topos: [topo0, topo1],
            tesselation,
        }
        .into()
    }
}

impl TopologyCore for DisjointUnion {
    fn tesselation(&self) -> &Tesselation {
        &self.tesselation
    }
    fn refined(&self) -> Topology {
        Self::new(self.topos[0].refined(), self.topos[1].refined())
    }
    fn boundary(&self) -> Topology {
        Self::new(self.topos[0].boundary(), self.topos[1].boundary())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Product {
    topos: [Topology; 2],
    tesselation: Tesselation,
}

impl Product {
    pub fn new(topo0: Topology, topo1: Topology) -> Topology {
        let map0 = topo0.tesselation();
        let map1 = topo1.tesselation();
        let dim_out = map0.dim_out() + map1.dim_out();
        let len_out = map0.len_out() * map1.len_out();
        let mut parts = Vec::new();
        for part0 in topo0.tesselation().iter() {
            for part1 in topo1.tesselation().iter() {
                let part0 = part0.clone();
                let mut part1 = part1.clone();
                let trans1 = Elementary::new_transpose(part1.len_out(), part0.len_out());
                let trans2 = Elementary::new_transpose(part0.len_in(), part1.len_out());
                part1.add_offset(part0.dim_in());
                let part: Vec<Elementary> = iter::once(trans1)
                    .chain(part0.into_unbounded())
                    .chain(iter::once(trans2))
                    .chain(part1.into_unbounded())
                    .collect();
                parts.push(WithBounds::new(part, dim_out, len_out).unwrap());
            }
        }
        let tesselation = Concatenation::new(parts);
        // TODO: assert no common roots
        Self {
            topos: [topo0, topo1],
            tesselation,
        }
        .into()
    }
}

impl TopologyCore for Product {
    fn tesselation(&self) -> &Tesselation {
        &self.tesselation
    }
    fn refined(&self) -> Topology {
        Self::new(self.topos[0].refined(), self.topos[1].refined())
    }
    fn boundary(&self) -> Topology {
        DisjointUnion::new(
            Product::new(self.topos[0].clone(), self.topos[1].boundary()),
            Product::new(self.topos[0].boundary(), self.topos[1].clone()),
        )
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Take {
    topo: Topology,
    itiles: Vec<usize>,
    tesselation: Tesselation,
}

impl Take {
    pub fn new(topo: Topology, itiles: Vec<usize>) -> Topology {
        // TODO: requires sorted itiles?
        let mut tesselation = topo.tesselation().clone();
        tesselation.push_elementary(&Elementary::new_take(itiles.clone(), tesselation.len_in()));
        Self {
            topo,
            itiles,
            tesselation,
        }
        .into()
    }
}

impl TopologyCore for Take {
    fn tesselation(&self) -> &Tesselation {
        &self.tesselation
    }
    fn refined(&self) -> Topology {
        let refined = self.topo.refined();
        let mut itiles = refined
            .tesselation()
            .unapply_indices_from(self.topo.tesselation(), &self.itiles)
            .unwrap();
        itiles.sort_by_key(|&index| index);
        Take::new(refined, itiles)
    }
    fn boundary(&self) -> Topology {
        unimplemented! {}
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Hierarchical {
    base: Topology,
    itiles: Vec<Vec<usize>>,
    tesselation: Tesselation,
}

fn refine_iter(base: Topology) -> impl Iterator<Item = Topology> {
    iter::successors(Some(base), |topo| Some(topo.refined()))
}

impl Hierarchical {
    pub fn new(base: Topology, itiles: Vec<Vec<usize>>) -> Topology {
        let tesselation = Concatenation::new(
            itiles
                .iter()
                .zip(refine_iter(base.clone()))
                .filter(|(itiles, _)| !itiles.is_empty())
                .flat_map(|(itiles, level)| {
                    let mut tesselation = level.tesselation().clone();
                    tesselation.push_elementary(&Elementary::new_take(
                        itiles.clone(),
                        tesselation.len_in(),
                    ));
                    tesselation.into_vec()
                })
                .collect(),
        );
        Self {
            base,
            itiles,
            tesselation,
        }
        .into()
    }
    fn levels(&self) -> impl Iterator<Item = Topology> {
        refine_iter(self.base.clone())
    }
    fn itiles_levels(&self) -> impl Iterator<Item = (&Vec<usize>, Topology)> {
        self.itiles.iter().zip(self.levels())
    }
}

impl TopologyCore for Hierarchical {
    fn tesselation(&self) -> &Tesselation {
        &self.tesselation
    }
    fn refined(&self) -> Topology {
        let itiles = self
            .itiles_levels()
            .map(|(itiles, level)| level.map_itiles_to_refined(itiles))
            .collect();
        Hierarchical::new(self.base.refined(), itiles)
    }
    fn refined_by(&self, itiles: &[usize]) -> Topology {
        let mut global_itiles = itiles.to_vec();
        global_itiles.sort_by_key(|&index| index);
        let mut global_itiles = global_itiles.into_iter().peekable();
        let mut offset = 0;
        let mut queue = Vec::new();
        let mut refined_itiles = Vec::new();
        for (level_itiles, level) in self.itiles_levels() {
            let mut refined_level_itiles: Vec<usize> = queue.drain(..).collect();
            for (i, itile) in level_itiles.iter().cloned().enumerate() {
                if global_itiles.next_if(|&j| i + offset == j).is_some() {
                    queue.push(itile);
                } else {
                    refined_level_itiles.push(itile);
                }
            }
            refined_level_itiles.sort_by_key(|&index| index);
            refined_itiles.push(refined_level_itiles);
            queue = level.map_itiles_to_refined(&queue);
            offset += level_itiles.len();
        }
        if !queue.is_empty() {
            refined_itiles.push(queue);
        }
        Hierarchical::new(self.base.clone(), refined_itiles)
    }
    fn boundary(&self) -> Topology {
        let base_boundary = self.base.boundary();
        let itiles = self.itiles_levels().zip(refine_iter(base_boundary.clone())).map(
            |((itiles, level), blevel)| {
                let mut itiles = blevel.tesselation().unapply_indices_from(level.tesselation(), itiles).unwrap();
                itiles.sort_by_key(|&index| index);
                itiles
            },
        ).collect();
        Hierarchical::new(base_boundary, itiles)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    macro_rules! assert_centroids {
        ($topo:expr, $geom:expr, $desired:expr $(, $simplex:ident)*) => {{
            let topo = $topo;
            let geom = $geom;
            let tesselation = topo.tesselation();
            let desired = $desired;
            let dim_out = tesselation.dim_out();
            let centroid_in = iter::once([]);
            let simplex_dim = 0;
            $(
                let centroid_in = centroid_in.map(|coord| [&coord as &[f64], &Simplex::$simplex.centroid()].concat());
                let simplex_dim = simplex_dim + Simplex::$simplex.dim();
            )*
            assert_eq!(simplex_dim, topo.dim(), "the given simplices don't add up to the dimension of the topology");
            let pad: Vec<f64> = iter::repeat(0.0).take(dim_out - topo.dim()).collect();
            let centroid_in: Vec<f64> = centroid_in.flat_map(|coord| [&coord[..], &pad].concat()).collect();

            assert_eq!(desired.len(), topo.ntiles(), "invalid len of desired centroids");

            for (i, desired) in desired.into_iter().enumerate() {
                println!("i = {i}");
                let mut actual = centroid_in.clone();
                let iroot = tesselation.apply_inplace(i, &mut actual, dim_out).unwrap();
                geom(iroot, &mut actual);
                assert_abs_diff_eq!(actual[..], desired[..]);
            }
        }};
    }

    #[test]
    fn test1() {
        let x0 = Line::from_len(2);
        let geom = |i: usize, c: &mut [f64]| c[0] += i as f64;
        assert_centroids!(&x0, geom, [[0.5], [1.5]], Line);
        let x1 = x0.refined_by(&[1]);
        assert_centroids!(&x1, geom, [[0.5], [1.25], [1.75]], Line);
        let x2 = x1.refined_by(&[1]);
        println!("{:?}", x2.tesselation());
        assert_centroids!(&x2, geom, [[0.5], [1.75], [1.125], [1.375]], Line);

        let x0b = x0.boundary();
        assert_centroids!(&x0b, geom, [[0.0], [2.0]]);
    }

    #[test]
    fn test2() {
        let x = Line::from_len(2);
        let y = Line::from_len(2);
        let geom = |i: usize, c: &mut [f64]| {
            c[0] += (i / 2) as f64;
            c[1] += (i % 2) as f64;
        };
        let xy = Product::new(x.clone(), y.clone());
        assert_centroids!(
            &xy,
            geom,
            [[0.5, 0.5], [0.5, 1.5], [1.5, 0.5], [1.5, 1.5]],
            Line,
            Line
        );
        assert_centroids!(
            &xy.boundary(),
            geom,
            [
                [0.5, 0.0],
                [0.5, 2.0],
                [1.5, 0.0],
                [1.5, 2.0],
                [0.0, 0.5],
                [0.0, 1.5],
                [2.0, 0.5],
                [2.0, 1.5]
            ],
            Line
        );
    }

    #[test]
    fn hierarchical() {
        let x = Line::from_len(2);
        let y = Line::from_len(2);
        let geom = |i: usize, c: &mut [f64]| {
            c[0] += (i / 2) as f64;
            c[1] += (i % 2) as f64;
        };
        let xy0 = Product::new(x, y);
        assert_centroids!(
            &xy0,
            geom,
            [[0.5, 0.5], [0.5, 1.5], [1.5, 0.5], [1.5, 1.5]],
            Line,
            Line
        );
        assert_centroids!(
            xy0.boundary(),
            geom,
            [
                [0.5, 0.0],
                [0.5, 2.0],
                [1.5, 0.0],
                [1.5, 2.0],
                [0.0, 0.5],
                [0.0, 1.5],
                [2.0, 0.5],
                [2.0, 1.5]
            ],
            Line
        );
        let xy1 = xy0.refined_by(&[2]);
        assert_centroids!(
            &xy1,
            geom,
            [
                [0.5, 0.5],
                [0.5, 1.5],
                [1.5, 1.5],
                [1.25, 0.25],
                [1.25, 0.75],
                [1.75, 0.25],
                [1.75, 0.75]
            ],
            Line,
            Line
        );
        assert_centroids!(
            xy1.boundary(),
            geom,
            [
                [0.5, 0.0],
                [0.5, 2.0],
                [1.5, 2.0],
                [0.0, 0.5],
                [0.0, 1.5],
                [2.0, 1.5],
                [1.25, 0.0],
                [1.75, 0.0],
                [2.0, 0.25],
                [2.0, 0.75],
            ],
            Line
        );
        let xy2 = xy1.refined_by(&[3]);
        assert_centroids!(
            &xy2,
            geom,
            [
                [0.5, 0.5],
                [0.5, 1.5],
                [1.5, 1.5],
                [1.25, 0.75],
                [1.75, 0.25],
                [1.75, 0.75],
                [1.125, 0.125],
                [1.125, 0.375],
                [1.375, 0.125],
                [1.375, 0.375]
            ],
            Line,
            Line
        );
    }
}
